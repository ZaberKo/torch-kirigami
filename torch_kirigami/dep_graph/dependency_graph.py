"""Public structural analysis queries and model snapshot lifetime."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from types import MappingProxyType

import torch
from torch import fx, nn

from ..bindings import attribute_owner
from ..errors import StaleGraphError
from ..settings import config
from .capture import fingerprint
from .capture import validate_attribute_changes as _validate_attribute_changes
from .construction import _Analyzer
from .contracts import AttributeBinding, Constraint, Diagnostic, Impact, ShapeExpr
from .eligibility import has_native_query_semantics
from .graph_state import DependencyGraphState, resolve_attribute, validate_reference
from .identity_axes import IdentityAxisIndex
from .operation import Operation, OperatorRule, OperatorSpec, TensorFacts
from .registry import OperatorRegistry
from .relations import Relation
from .selection import Selection, TensorRef
from .solver import DependencySolver


class DependencyGraph:
    """An FX-backed structural analysis snapshot.

    The graph analyzes structure only; it does not choose what to prune. Use
    `DependencyGraph.build` to capture a model. The constructor accepts only
    completed internal records, so no empty or partly initialized graph can be
    queried. Queries do not modify weights, and all selections use the original
    captured coordinates. Rebuild after changes to structure or capture
    assumptions.
    """

    def __init__(self, data: DependencyGraphState) -> None:
        if not data.complete or data.fx_graph is None:
            raise ValueError("DependencyGraph requires completed analysis data")
        self._data = data
        self._solver = DependencySolver(data)
        self._valid = True
        self._queries_checked: ContextVar[bool] = ContextVar(
            "kirigami_native_queries", default=False
        )
        self._check_fresh()

    @contextmanager
    def _native_queries(
        self, constraints: tuple[Constraint, ...], *, enabled: bool = True
    ) -> Iterator[None]:
        """Reuse validation only inside a callback-free, native query batch.

        Entry and exit always inspect the full snapshot, including exceptions.
        Reuse state belongs to the current execution context.
        Callers must disable reuse around user code and tensor subclasses.
        """
        self._check_fresh(force=True)
        native = (
            enabled
            and type(self) is DependencyGraph
            and not any(callable(value) for value in vars(self).values())
            and has_native_query_semantics(self._data, constraints)
        )
        native = native and all(
            type(tensor) in (nn.Parameter, torch.Tensor) for _, tensor in self.tensor_bindings()
        )
        token = self._queries_checked.set(native)
        try:
            yield
        finally:
            self._queries_checked.reset(token)
            self._check_fresh(force=True)

    @property
    def id(self) -> str:
        """Identity of this snapshot, used to check reference and impact ownership."""
        return self._data.id

    @property
    def context(self) -> Mapping[str, object]:
        """Read-only execution assumptions and rule identities recorded at build time."""
        return self._data.context

    @classmethod
    def build(
        cls,
        model: nn.Module,
        *,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        operators: OperatorRegistry | None = None,
    ) -> DependencyGraph:
        """Capture a module and build its structural dependency relationships.

        Args:
            model: Source module whose registered tensors and aliases will be tracked.
            args: Example positional arguments for `forward`.
            kwargs: Example keyword arguments for `forward`.
            operators: Local semantics registry, copied for this build. Defaults
                to a registry with the built-in rules.

        Returns:
            A snapshot containing tensor references, relations, and constraints.
            Unsupported captured operators remain visible in diagnostics.

        Raises:
            TypeError: `model` is not an `nn.Module`, or an operator rule does not
                return an `OperatorSpec`.
            ValueError: An operator spec refers to a tensor outside the snapshot.
            torch_kirigami.CaptureError: Arguments, tracing, or isolated metadata
                execution fail.
            torch_kirigami.StaleGraphError: A tracked structural property changes
                during construction.

        Notes:
            Examples provide metadata; they do not specialize dynamic Python branches.
            Inputs and registered buffers are isolated. Explicitly declared
            parameter value updates execute with a warning and are not rolled
            back. Other parameter writes and external side effects are unsupported.
            Tensor identity and type tests, and Python decisions based on runtime
            state (such as `x.grad is None`), are outside the capture contract.
            FX can silently omit them; successful construction does not prove
            their absence or correctness.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        data = _Analyzer(model, operators).build(tuple(args), dict(kwargs or {}))
        return cls(data)

    @property
    def fx_graph(self) -> fx.Graph:
        """Deep copy of the captured FX graph, for inspection."""
        assert self._data.fx_graph is not None
        return copy.deepcopy(self._data.fx_graph)

    @property
    def relations(self) -> tuple[Relation, ...]:
        """Structural relations recorded by operator rules."""
        return tuple(self._data.relations)

    @property
    def identity_axes(self) -> IdentityAxisIndex:
        """Return the snapshot's shared index of identity-coordinate relations."""
        self.validate()
        return self._solver.identity_axes

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Graph constraints checked after propagation reaches a fixed point."""
        return tuple(self._data.constraints)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Build-time diagnostics, including unsupported captured operations."""
        return tuple(self._data.diagnostics)

    @property
    def shape_expressions(self) -> Mapping[str, ShapeExpr]:
        """Read-only shape provenance keyed by FX node name."""
        return MappingProxyType({node.name: expr for node, expr in self._data.expressions.items()})

    def parameter(self, path: str) -> TensorRef:
        """Find a parameter using an original model path.

        Args:
            path: Registered parameter path, including any original alias.

        Returns:
            The `TensorRef` shared by every alias of that parameter object.

        Raises:
            KeyError: The path was not registered when the graph was built.
        """
        return self._data.parameters[path]

    def buffer(self, path: str) -> TensorRef:
        """Find a registered buffer using an original model path.

        Args:
            path: Registered buffer path, including any original alias.

        Returns:
            The `TensorRef` shared by every alias of that buffer object.

        Raises:
            KeyError: The path was not registered when the graph was built.
        """
        return self._data.buffers[path]

    def metadata(self, tensor: TensorRef) -> TensorFacts:
        """Return captured tensor metadata without retaining its activation.

        Args:
            tensor: A reference belonging to this graph.

        Returns:
            Shape, strides, dtype, and device recorded during capture.

        Raises:
            ValueError: The tensor reference belongs to another graph or has
                altered metadata.
        """
        validate_reference(self._data.refs, tensor)
        return self._data.tensor_facts[tensor.id]

    def values(self) -> tuple[TensorRef, ...]:
        """Return every tensor reference in this snapshot.

        Returns:
            Parameter, buffer, input, and captured intermediate value references.
        """
        return tuple(self._data.refs.values())

    @property
    def model(self) -> nn.Module:
        """Original module analyzed by this snapshot."""
        return self._data.model

    def validate_attribute_changes(self, updates: Iterable[tuple[str, object]]) -> None:
        """Check proposed module attribute edits against the captured computation.

        The edits are applied to an isolated configuration copy, which is traced
        again and compared with the original capture. This method does not
        modify the model or its weights, and it does not choose removals. Opaque
        leaf internals remain the responsibility of their declared operator rules.

        Args:
            updates: Pairs of dotted attribute paths and proposed values.

        Raises:
            torch_kirigami.CaptureError: The configuration cannot be isolated or
                traced, or retracing changes the captured structure or constants.
            torch_kirigami.StaleGraphError: The model no longer matches this
                snapshot, before or after the check.
        """
        # Retracing executes the caller's Python forward, even for a graph whose
        # captured operators are all native. Never reuse a check across that code.
        with self._native_queries((), enabled=False):
            _validate_attribute_changes(
                self._data.model, self._data.registry, self._data.capture_signature, tuple(updates)
            )

    def validate(self, model: nn.Module | None = None) -> None:
        """Check that this snapshot is fresh and, optionally, owns a model.

        Args:
            model: Module expected to be the analyzed model, or None to check
                only freshness.

        Raises:
            ValueError: `model` is not the module analyzed by this snapshot.
            torch_kirigami.StaleGraphError: Analysis settings, model structure,
                modes, or guarded constants changed, or the snapshot was invalidated.
        """
        if model is not None and model is not self._data.model:
            raise ValueError("Graph belongs to a different model")
        self._check_fresh()

    def invalidate(self) -> None:
        """Mark this snapshot stale after an external structural mutation.

        Later freshness checks raise `torch_kirigami.StaleGraphError`.
        """
        self._valid = False

    def interfaces(self) -> tuple[TensorRef, ...]:
        """Return unique external input and output tensor references."""
        return tuple(ref for ref in self._data.refs.values() if ref.id in self._data.interfaces)

    def tensor(self, ref: TensorRef) -> torch.Tensor:
        """Return the live registered tensor for a parameter or buffer reference.

        Args:
            ref: Parameter or buffer reference belonging to this graph.

        Returns:
            The tensor currently bound at the reference's first registered path.

        Raises:
            torch_kirigami.StaleGraphError: The model no longer matches this snapshot.
            ValueError: The reference belongs to another graph, has altered
                metadata, or is not a parameter or buffer.
        """
        self._check_fresh()
        self.metadata(ref)
        if ref.kind not in ("parameter", "buffer"):
            raise ValueError("Only registered tensors have persistent bindings")
        return resolve_attribute(self._data.model, ref.paths[0])

    def tensor_bindings(self) -> tuple[tuple[TensorRef, torch.Tensor], ...]:
        """Check freshness once and return all registered tensor bindings.

        The caller must revalidate after user callbacks or other possible model
        changes; the returned objects are live and do not confer a read lock.

        Returns:
            Pairs of parameter or buffer references and their live tensors.

        Raises:
            torch_kirigami.StaleGraphError: The model no longer matches this snapshot.
        """
        self._check_fresh()
        return tuple(
            (ref, resolve_attribute(self._data.model, ref.paths[0]))
            for ref in self._data.refs.values()
            if ref.kind in ("parameter", "buffer")
        )

    def bindings(self, ref: TensorRef) -> tuple[tuple[nn.Module, str], ...]:
        """Return the module attribute slots that hold a registered tensor.

        Args:
            ref: Parameter or buffer reference belonging to this graph.

        Returns:
            Unique `(owner module, attribute name)` pairs, one per distinct slot
            among the reference's registered paths.

        Raises:
            torch_kirigami.StaleGraphError: The model no longer matches this snapshot.
            ValueError: The reference belongs to another graph, has altered
                metadata, or is not a parameter or buffer.
        """
        self.tensor(ref)
        result = []
        seen = set()
        for path in ref.paths:
            owner, name = attribute_owner(self._data.model, path)
            if (id(owner), name) not in seen:
                result.append((owner, name))
                seen.add((id(owner), name))
        return tuple(result)

    def constants(self) -> tuple[tuple[TensorRef, str], ...]:
        """Return captured tensor constants that are not registered parameters or buffers.

        FX can lift ordinary tensor attributes or closed-over tensors into its
        GraphModule. Their FX paths do not prove writable original-model bindings.

        Returns:
            Pairs of tensor references and their FX attribute paths.
        """
        return self._data.constants

    def operator_spec(self, operation: Operation) -> OperatorSpec | None:
        """Return the structural specification recorded for a captured operation.

        Args:
            operation: An operation context from this graph, such as one
                returned by `operations`.

        Returns:
            The recorded specification, or None if the operation is unsupported.

        Raises:
            ValueError: The operation belongs to another graph or its captured
                call identity was altered.
        """
        operation = self._canonical_operation(operation)
        return self._data.specs.get(operation.node.name)

    def operator_rule(self, operation: Operation) -> OperatorRule | None:
        """Return the operator rule that matched a captured operation during capture.

        Args:
            operation: An operation context from this graph, such as one
                returned by `operations`.

        Returns:
            The matching rule from the graph's registry copy, or None if no rule
            matches.

        Raises:
            ValueError: The operation belongs to another graph or its captured
                call identity was altered.
        """
        operation = self._canonical_operation(operation)
        return self._data.registry.lookup(operation.node, operation.module)

    def _canonical_operation(self, operation: Operation) -> Operation:
        """Accept inspection copies only when their captured call identity matches."""
        if not isinstance(operation, Operation) or operation.graph_id != self.id:
            raise ValueError("Operation belongs to another graph")
        original = self._data.operations.get(operation.node.name)
        if original is None:
            raise ValueError("Operation does not belong to this graph")
        if (
            original.node.op != operation.node.op
            or original.node.target != operation.node.target
            or original.module is not operation.module
        ):
            raise ValueError("Operation identity was altered")
        return original

    def constant_guards(self) -> tuple[str, ...]:
        """Return paths of registered tensors whose values guard the analysis.

        Returns:
            The first registered path of each guarded tensor. Freshness checks
            raise `torch_kirigami.StaleGraphError` if a guarded value changes.
        """
        return tuple(ref.paths[0] for ref in self._data.constant_refs)

    def affected_operations(self, impact: Impact) -> frozenset[str]:
        """Return the captured operations that an impact can affect.

        An operation is affected if a selected tensor is one of its inputs,
        outputs, or module bindings, or is read by the shape expression of one
        of its input nodes. Every FX successor of an affected operation, including
        later readers of its declared in-place writes, is also affected, even if
        its own tensors have no selected regions.

        Args:
            impact: A result produced by this graph.

        Returns:
            FX node names of the affected operations.

        Raises:
            ValueError: The impact belongs to another graph.
            torch_kirigami.StaleGraphError: A tracked model property has changed.
        """
        self.validate_impact(impact)
        self._check_fresh()
        return self._solver.affected_operations(impact.selections)

    def operations(self, module_path: str | None = None) -> tuple[Operation, ...]:
        """Return inspection contexts with copied FX nodes and argument containers.

        Module and tensor bindings refer to the original model. No activations are
        retained. Editing the returned argument containers cannot change analysis.

        Args:
            module_path: Module path used to filter operations, or None for all
                operations. An operation matches if the path is one of the
                registered paths of its called module.

        Returns:
            Operation contexts in capture order.
        """
        nodes = {node.name: node for node in self.fx_graph.nodes}
        expressions = MappingProxyType(
            {nodes[n.name]: e for n, e in self._data.expressions.items()}
        )
        return tuple(
            replace(
                ctx,
                node=nodes[ctx.node.name],
                args=copy.deepcopy(ctx.args),
                kwargs=copy.deepcopy(ctx.kwargs),
                result=copy.deepcopy(ctx.result),
                bindings=dict(ctx.bindings),
                expressions=expressions,
            )
            for ctx in self._data.operations.values()
            if module_path is None or module_path in ctx.module_paths
        )

    def _check_fresh(self, *, force: bool = False) -> None:
        """Reject detectable structural changes without comparing weight values."""
        if config.analysis_key() != self._data.settings_key:
            raise StaleGraphError(
                "Analysis configuration changed; rebuild the graph before using it"
            )
        if not self._valid:
            raise StaleGraphError("Model structure/mode/configuration changed; rebuild the graph")
        if not force and self._queries_checked.get():
            return
        if fingerprint(self._data.model) != self._data.fingerprint:
            raise StaleGraphError("Model structure/mode/configuration changed; rebuild the graph")
        for ref in self._data.constant_refs:
            value = resolve_attribute(self._data.model, ref.paths[0])
            if (
                tuple(value.detach().cpu().reshape(-1).tolist())
                != self._data.literal_values[ref.id]
            ):
                raise StaleGraphError(f"Structural constant {ref.paths[0]!r} changed")

    def propagate(
        self, *, remove: Iterable[Selection], constraints: Iterable[Constraint] = ()
    ) -> Impact:
        """Compute the structural closure of a joint removal request.

        Propagation follows relations to a fixed point, then checks graph and
        caller constraints against the closure. Constraints only report
        diagnostics; they never add removals.

        Args:
            remove: Selections in this snapshot's original coordinate system.
            constraints: Additional consumer constraints, such as protected axes.

        Returns:
            An `Impact` containing determined selections, provenance, diagnostics,
            constraints, and rewrite requirements. A resolved result is neither an
            executable pruning plan nor a numerical equivalence guarantee.

        Raises:
            torch_kirigami.StaleGraphError: A tracked model property has changed.
            ValueError: A selection or constraint refers to another graph.
            torch_kirigami.AnalysisLimitError: Merging the requested selections
                exceeds the region limit. Limits reached while following relations
                or checking constraints are reported as diagnostics instead.
        """
        self._check_fresh()
        return self._solver.propagate(remove=remove, constraints=constraints)

    def validate_impact(self, impact: Impact) -> None:
        """Reject analysis results from another snapshot without executing the model.

        Args:
            impact: Result to check.

        Raises:
            ValueError: `impact` is not an `Impact` produced by this graph.
        """
        if not isinstance(impact, Impact) or impact.graph_id != self.id:
            raise ValueError("Impact belongs to another graph")

    def explain(self, impact: Impact) -> str:
        """Format an impact's status, selections, provenance, diagnostics, and requirements.

        Args:
            impact: A result produced by this graph.

        Returns:
            A multi-line, human-readable explanation. The model is not executed.

        Raises:
            ValueError: The impact belongs to another graph.
        """
        self.validate_impact(impact)
        lines = [f"Structural analysis: {impact.status}"]
        for selection in impact.selections.values():
            label = ", ".join(selection.tensor.paths) or selection.tensor.id.split(":value:")[-1]
            lines.append(
                f"- {label}: {selection.count} selected elements; regions={selection.regions}"
            )
        for step in impact.provenance:
            lines.append(f"  via {step.reason}: {step.source.tensor.id} -> {step.target.tensor.id}")
        lines.extend(f"- {d}" for d in impact.diagnostics)
        lines.extend(
            f"- requires {'attribute' if isinstance(r, AttributeBinding) else r.kind}: {r.target}: {r.detail}"
            for r in impact.requirements
        )
        return "\n".join(lines)
