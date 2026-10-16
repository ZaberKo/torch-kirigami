"""Public structural analysis queries and model snapshot lifetime."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType

import torch
from torch import fx, nn

from ..errors import StaleGraphError
from ..settings import config
from .capture import fingerprint
from .capture import validate_attribute_changes as _validate_attribute_changes
from .construction import _Analyzer
from .contracts import AttributeBinding, Constraint, Diagnostic, Impact, ShapeExpr
from .graph_state import DependencyGraphState, resolve_attribute, validate_reference
from .operation import Operation, OperatorRule, OperatorSpec, TensorFacts
from .registry import OperatorRegistry
from .relations import Relation
from .selection import Selection, TensorRef
from .solver import DependencySolver


class DependencyGraph:
    """An FX-backed structural analysis snapshot.

    Use `build()` to capture a model. Construction accepts only completed internal
    records, so no empty or partly initialized graph can be queried. Queries do
    not modify weights. Rebuild after changes to structure or capture assumptions.
    """

    def __init__(self, data: DependencyGraphState) -> None:
        if not data.complete or data.fx_graph is None:
            raise ValueError("DependencyGraph requires completed analysis data")
        self._data = data
        self._solver = DependencySolver(data)
        self._valid = True
        self._check_fresh()

    @property
    def id(self) -> str:
        """Return this snapshot's identity for reference and impact ownership."""
        return self._data.id

    @property
    def context(self) -> Mapping[str, object]:
        """Return captured execution assumptions and the local rule identities."""
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
            args: Example positional arguments for forward.
            kwargs: Example keyword arguments for forward.
            operators: Local semantics registry. Defaults to the built-in rules.

        Returns:
            A snapshot containing tensor references, relations, and constraints.
            Unsupported captured operators remain visible in diagnostics.

        Raises:
            TypeError: model is not an nn.Module.
            CaptureError: Arguments, tracing, or isolated metadata execution fail.
            StaleGraphError: A tracked structural property changes during construction.

        Notes:
            Examples provide metadata; they do not specialize dynamic Python branches.
            Inputs and registered buffers are isolated. Explicitly declared
            parameter value updates execute with a warning and are not rolled
            back. Other parameter writes and external side effects are unsupported.
            Tensor identity/type tests and runtime-state Python decisions (e.g.
            x.grad is None) are outside the capture contract. FX can silently omit
            them; successful construction does not prove their absence or correctness.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        data = _Analyzer(model, operators).build(tuple(args), dict(kwargs or {}))
        return cls(data)

    @property
    def fx_graph(self) -> fx.Graph:
        """Return an inspection copy of the captured FX graph."""
        assert self._data.fx_graph is not None
        return copy.deepcopy(self._data.fx_graph)

    @property
    def relations(self) -> tuple[Relation, ...]:
        """Return the structural relations recorded by operation rules."""
        return tuple(self._data.relations)

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return the constraints checked after propagation reaches a fixed point."""
        return tuple(self._data.constraints)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        """Return build-time diagnostics, including unsupported captured operations."""
        return tuple(self._data.diagnostics)

    @property
    def shape_expressions(self) -> Mapping[str, ShapeExpr]:
        """Return shape provenance indexed by FX node name."""
        return MappingProxyType({node.name: expr for node, expr in self._data.expressions.items()})

    def parameter(self, path: str) -> TensorRef:
        """Find a parameter using an original model path.

        Args:
            path: Registered parameter path, including any original alias.

        Returns:
            The shared TensorRef for that parameter object.

        Raises:
            KeyError: The path was not registered when the graph was built.
        """
        return self._data.parameters[path]

    def buffer(self, path: str) -> TensorRef:
        """Return the buffer reference for an original path, or raise KeyError."""
        return self._data.buffers[path]

    def metadata(self, tensor: TensorRef) -> TensorFacts:
        """Return captured tensor metadata without retaining its activation.

        Args:
            tensor: A reference belonging to this graph.

        Returns:
            Shape, strides, dtype, and device recorded during capture.

        Raises:
            ValueError: The tensor reference belongs to another graph.
        """
        validate_reference(self._data.refs, tensor)
        return self._data.tensor_facts[tensor.id]

    def values(self) -> tuple[TensorRef, ...]:
        """Return all parameter, buffer, and captured value references."""
        return tuple(self._data.refs.values())

    @property
    def model(self) -> nn.Module:
        """Return the original module owning this snapshot."""
        return self._data.model

    def validate_attribute_changes(self, updates: Iterable[tuple[str, object]]) -> None:
        """Check (path, value) edits against the captured Python computation.

        Args:
            updates: Iterable of attribute paths and proposed configuration values.

        Raises:
            CaptureError: Configuration cannot be isolated or re-tracing changes
                the captured structure or constants.
            StaleGraphError: The source no longer matches this snapshot.

        This checks edits without modifying weights or choosing removals. Opaque
        leaf internals remain the responsibility of their declared operator rules.
        """
        self.validate()
        _validate_attribute_changes(
            self._data.model, self._data.registry, self._data.capture_signature, tuple(updates)
        )
        self.validate()

    def validate(self, model: nn.Module | None = None) -> None:
        """Check snapshot freshness and optional model ownership."""
        if model is not None and model is not self._data.model:
            raise ValueError("Graph belongs to a different model")
        self._check_fresh()

    def invalidate(self) -> None:
        """Invalidate this snapshot after an external structural mutation."""
        self._valid = False

    def interfaces(self) -> tuple[TensorRef, ...]:
        """Return unique external input and output tensor references."""
        return tuple(ref for ref in self._data.refs.values() if ref.id in self._data.interfaces)

    def tensor(self, ref: TensorRef) -> torch.Tensor:
        """Return the original registered tensor for a parameter or buffer reference."""
        self._check_fresh()
        self.metadata(ref)
        if ref.kind not in ("parameter", "buffer"):
            raise ValueError("Only registered tensors have persistent bindings")
        return resolve_attribute(self._data.model, ref.paths[0])

    def tensor_bindings(self) -> tuple[tuple[TensorRef, torch.Tensor], ...]:
        """Validate once and return all registered (reference, tensor) bindings.

        The caller must revalidate after user callbacks or other possible model
        changes; the returned objects are live and do not confer a read lock.
        """
        self._check_fresh()
        return tuple(
            (ref, resolve_attribute(self._data.model, ref.paths[0]))
            for ref in self._data.refs.values()
            if ref.kind in ("parameter", "buffer")
        )

    def bindings(self, ref: TensorRef) -> tuple[tuple[nn.Module, str], ...]:
        """Return unique registered (owner module, attribute name) binding slots."""
        self.tensor(ref)
        result = []
        seen = set()
        for path in ref.paths:
            parent, _, name = path.rpartition(".")
            owner = self._data.model.get_submodule(parent) if parent else self._data.model
            if (id(owner), name) not in seen:
                result.append((owner, name))
                seen.add((id(owner), name))
        return tuple(result)

    def constants(self) -> tuple[tuple[TensorRef, str], ...]:
        """Return (reference, FX attribute path) pairs without original registered bindings.

        FX can lift ordinary tensor attributes or closed-over tensors into its
        GraphModule. Their FX paths do not prove writable original-model bindings.
        """
        return self._data.constants

    def operator_spec(self, operation: Operation) -> OperatorSpec | None:
        """Return shared structural descriptors for a captured operation."""
        operation = self._canonical_operation(operation)
        return self._data.specs.get(operation.node.name)

    def operator_rule(self, operation: Operation) -> OperatorRule | None:
        """Return the exact operator definition used during capture."""
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
        """Return registered integer tensors whose values are analysis preconditions."""
        return tuple(ref.paths[0] for ref in self._data.constant_refs)

    def affected_operations(self, impact: Impact) -> frozenset[str]:
        """Include size-expression consumers even without removed tensor regions."""
        self.validate_impact(impact)

        return self._solver.affected_operations(impact.selections)

    def operations(self, module_path: str | None = None) -> tuple[Operation, ...]:
        """Return inspection contexts with copied FX nodes and argument containers.

        Module and tensor bindings refer to the original model. No activations are
        retained. Editing the returned argument containers cannot change analysis.
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

    def _check_fresh(self) -> None:
        """Reject detectable structural changes without comparing weight values."""
        if config.analysis_key() != self._data.settings_key:
            raise StaleGraphError(
                "Analysis configuration changed; rebuild the graph before using it"
            )
        if not self._valid or fingerprint(self._data.model) != self._data.fingerprint:
            raise StaleGraphError("Model structure/mode/configuration changed; rebuild the graph")
        for ref in self._data.constant_refs:
            value = resolve_attribute(self._data.model, ref.paths[0])
            if (
                tuple(value.detach().cpu().reshape(-1).tolist())
                != self._data.literal_values[ref.id]
            ):
                raise StaleGraphError("An integer tensor used as a structural constant changed")

    def propagate(
        self, *, remove: Iterable[Selection], constraints: Iterable[Constraint] = ()
    ) -> Impact:
        """Compute the structural closure of joint removal requests.

        Args:
            remove: Selections in this snapshot's original coordinate system.
            constraints: Additional consumer constraints, such as protected axes.

        Returns:
            An Impact containing determined selections, provenance, constraints,
            and rewrite requirements. A resolved result is neither an executable
            pruning plan nor a numerical equivalence guarantee.

        Raises:
            StaleGraphError: A tracked model property has changed.
            ValueError: A selection or constraint refers to another graph.
            AnalysisLimitError: A supplied selection exceeds the region limit.
                Limits reached while following relations are reported as diagnostics.
        """
        self._check_fresh()
        return self._solver.propagate(remove=remove, constraints=constraints)

    def validate_impact(self, impact: Impact) -> None:
        """Reject analysis results from another snapshot without executing the model."""
        if not isinstance(impact, Impact) or impact.graph_id != self.id:
            raise ValueError("Impact belongs to another graph")

    def explain(self, impact: Impact) -> str:
        """Format the selected regions, propagation reasons, and remaining requirements.

        Args:
            impact: A result produced by this graph.

        Returns:
            A human-readable explanation without executing the model.

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
