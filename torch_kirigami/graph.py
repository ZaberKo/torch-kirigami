"""FX-backed dependency analysis and monotone structural impact queries."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from types import MappingProxyType
from uuid import uuid4

import torch
from torch import fx, nn

from .capture import TensorFacts, capture, fingerprint, storage_key, tree_map
from .contracts import (
    Barrier,
    Diagnostic,
    Impact,
    Layout,
    NonEmpty,
    Provenance,
    Requirement,
)
from .errors import AnalysisLimitError, StaleGraphError
from .operators.native import UnsupportedOperation
from .operators.shapes import ArgumentDependencies, dependencies
from .registry import OperationContext, OperatorRegistry, OperatorSpec, tensors
from .relations import AxisRelation
from .selection import Selection, TensorRef


@dataclass(frozen=True)
class CallRef:
    """An operation invocation with flattened tensor ports.

    Attributes:
        name: FX node name, unique within this graph.
        module_paths: Original aliases of the called module, or an empty tuple.
        inputs: Tensor arguments in container traversal order.
        outputs: Tensor results in container traversal order.
    """

    name: str
    module_paths: tuple[str, ...]
    inputs: tuple[TensorRef, ...]
    outputs: tuple[TensorRef, ...]

    def input(self, index=0):
        """Return the tensor at the given flattened input index."""
        return self.inputs[index]

    def output(self, index=0):
        """Return the tensor at the given flattened output index."""
        return self.outputs[index]


def _attribute(module, path):
    """Resolve a dotted FX attribute path on a module."""
    for component in path.split("."):
        module = getattr(module, component)
    return module


class DependencyGraph:
    """An FX-backed structural analysis snapshot.

    Build snapshots with build(). Queries describe dependencies without modifying
    weights. Rebuild after structural changes or changes to capture assumptions.
    """

    @classmethod
    def build(
        cls, model: nn.Module, *, args=(), kwargs=None, operators: OperatorRegistry | None = None
    ):
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
            Inputs and registered buffers are isolated, but forward must not mutate
            parameters or external state.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        self = cls()
        self.id = uuid4().hex
        self._model = model
        self._registry = (operators if operators is not None else OperatorRegistry.default()).copy()
        self._fingerprint = fingerprint(model)
        self.context = MappingProxyType(
            {
                "torch_version": torch.__version__,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "module_modes": tuple(
                    (path, module.training)
                    for path, module in model.named_modules(remove_duplicate=False)
                ),
                "rule_snapshot": tuple(
                    (str(target), id(rule))
                    for table in (
                        self._registry.modules,
                        self._registry.functions,
                        self._registry.methods,
                    )
                    for target, rule in table.items()
                ),
            }
        )
        self._refs = {}
        self._tensor_facts = {}
        self._parameters, self._buffers = {}, {}
        self._by_object = {}
        self._module_paths = defaultdict(list)
        storage_groups = defaultdict(list)
        # FX may canonicalize shared module paths, so preserve aliases first.
        for path, module in model.named_modules(remove_duplicate=False):
            self._module_paths[id(module)].append(path)
        for kind, entries, paths in (
            ("parameter", model.named_parameters(remove_duplicate=False), self._parameters),
            ("buffer", model.named_buffers(remove_duplicate=False), self._buffers),
        ):
            grouped = {}
            for path, tensor in entries:
                grouped.setdefault(id(tensor), (tensor, []))[1].append(path)
            for identity, (tensor, aliases) in grouped.items():
                ref = TensorRef(
                    f"{self.id}:{kind}:{aliases[0]}", tuple(tensor.shape), kind, tuple(aliases)
                )
                self._tensor_facts[ref.id] = TensorFacts(
                    tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device
                )
                self._by_object[identity] = ref
                self._refs[ref.id] = ref
                for path in aliases:
                    paths[path] = ref
                key = storage_key(tensor)
                if key is not None:
                    storage_groups[key].append(ref)
        captured = capture(model, tuple(args), dict(kwargs or {}), self._registry)
        for identity, original in captured.buffer_aliases.items():
            self._by_object[identity] = self._by_object[id(original)]
        gm = captured.module
        self._fx_graph = gm.graph
        self._values = {}
        self._facts = captured.facts
        self._expressions = {}
        self._literal_values = {}
        self._relations, self._constraints, self._requirements = [], [], []
        self._diagnostics = []
        self._calls = []
        self._operations = []
        self._specs = {}
        self._valid = True
        self._interfaces = set()
        self._source_modules = {}
        self._unused = set(self._refs)
        for refs in storage_groups.values():
            unique = tuple(dict.fromkeys(refs))
            if len(unique) > 1:
                self._constraints.append(
                    Barrier(
                        unique,
                        "Distinct tensors share storage; overlapping/view compaction is unsupported",
                        code="storage_alias",
                    )
                )
        for node in gm.graph.nodes:
            if node.op == "output":
                self._values[node] = self._resolve(node.args[0])
                self._interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                value = _attribute(gm, str(node.target))
                if id(value) in self._by_object:
                    self._values[node] = self._by_object[id(value)]
                    if (
                        not value.is_floating_point()
                        and not value.is_complex()
                        and value.numel() <= 4096
                    ):
                        self._literal_values[self._values[node].id] = tuple(
                            value.detach().cpu().reshape(-1).tolist()
                        )
                    continue
            self._values[node] = self._make_values(node, captured.facts[node])
            if node.op == "placeholder":
                self._interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                value = _attribute(gm, str(node.target))
                if (
                    isinstance(value, torch.Tensor)
                    and not value.is_floating_point()
                    and not value.is_complex()
                    and value.numel() <= 4096
                ):
                    self._literal_values[self._values[node].id] = tuple(
                        value.detach().cpu().reshape(-1).tolist()
                    )
                continue
            args_, kwargs_ = self._resolve(node.args), self._resolve(node.kwargs)
            module = gm.get_submodule(str(node.target)) if node.op == "call_module" else None
            paths = tuple(self._module_paths.get(id(module), ())) if module is not None else ()
            path = paths[0] if paths else None
            bindings = {}
            if module is not None:
                for name, value in (
                    *module.named_parameters(remove_duplicate=False),
                    *module.named_buffers(remove_duplicate=False),
                ):
                    ref = self._by_object.get(id(value))
                    if ref is not None:
                        bindings[name] = ref
            ctx = OperationContext(
                node,
                args_,
                kwargs_,
                self._values[node],
                module,
                path,
                bindings,
                self._expressions,
                MappingProxyType(self._tensor_facts),
                MappingProxyType(self._literal_values),
            )
            self._calls.append(CallRef(node.name, paths, ctx.inputs, ctx.outputs))
            self._operations.append(ctx)
            rule = self._registry.lookup(node, module)
            try:
                if rule is None:
                    raise UnsupportedOperation(f"No semantics registered for {node.target}")
                result = rule.analyze(ctx)
                if not isinstance(result, OperatorSpec):
                    raise TypeError("OperatorRule.analyze must return OperatorSpec")
                self._specs[node.name] = result
                if result.expression is not None:
                    self._expressions[node] = result.expression
            except UnsupportedOperation as error:
                refs = tuple(
                    dict.fromkeys(
                        (*ctx.inputs, *ctx.outputs, *bindings.values(), *dependencies(ctx))
                    )
                )
                # Scalar/shape producers can hide structural dependencies across reductions.
                if not ctx.outputs or "provenance" in str(error):
                    refs = tuple(dict.fromkeys((*refs, *self._ancestor_refs(node))))
                barrier = Barrier(refs, str(error), node.name)
                self._constraints.append(barrier)
                self._diagnostics.append(
                    Diagnostic(
                        "unsupported", str(error), node=node.name, tensors=tuple(r.id for r in refs)
                    )
                )
                self._unused.difference_update(r.id for r in refs)
                continue
            self._relations.extend(result.relations)
            self._constraints.extend(result.constraints)
            self._requirements.extend(result.requirements)
            self._requirements.extend(
                Requirement(
                    "partitioned_compaction",
                    node.name,
                    (descriptor.tensor,),
                    "Retain declared physical partitions and concatenate in original order",
                )
                for descriptor in result.layouts
            )
            if result.expression is None and dependencies(ctx):
                self._constraints.append(
                    ArgumentDependencies(
                        ctx,
                        any(
                            r.kind
                            in (
                                "shape_arguments",
                                "partition_arguments",
                                "operator_argument",
                                "slice_arguments",
                                "call_arguments",
                            )
                            for r in result.requirements
                        ),
                    )
                )
            for relation in result.relations:
                self._unused.difference_update(r.id for r in relation.refs)
            for constraint in result.constraints:
                self._unused.difference_update(r.id for r in constraint.refs)
        ports = defaultdict(list)
        for relation in self._relations:
            if isinstance(relation, AxisRelation):
                for port in (relation.left, relation.right):
                    ports[port.tensor.id].append(port)
        for ref in self._refs.values():
            scoped = ports[ref.id] if any(p.scope is not None for p in ports[ref.id]) else ()
            self._constraints.append(Layout(ref, tuple(scoped)))
            self._constraints.extend(NonEmpty(ref.axis(d)) for d in range(len(ref.shape)))
            if ref.id in self._unused:
                self._constraints.append(
                    Barrier(
                        (ref,), "No structural usage rule for this tensor", code="unbound_tensor"
                    )
                )
        self._adjacency = defaultdict(list)
        for relation in self._relations:
            for ref in dict.fromkeys(relation.refs):
                self._adjacency[ref.id].append(relation)
        self._relations = tuple(self._relations)
        layouts = tuple(layout for spec in self._specs.values() for layout in spec.layouts)
        self._constraints = tuple(
            replace(constraint, layouts=layouts)
            if isinstance(constraint, ArgumentDependencies)
            else constraint
            for constraint in self._constraints
        )
        self._requirements = tuple(self._requirements)
        self._calls = tuple(self._calls)
        self._diagnostics = tuple(self._diagnostics)
        self._check_fresh()
        return self

    def _make_values(self, node, facts):
        """Replace tensor facts with references while preserving the result tree."""
        counter = [0]

        def make(value):
            if isinstance(value, TensorFacts):
                ref = TensorRef(
                    f"{self.id}:value:{node.name}:{counter[0]}",
                    value.shape,
                    "input" if node.op == "placeholder" else "value",
                )
                counter[0] += 1
                self._refs[ref.id] = ref
                self._tensor_facts[ref.id] = value
                return ref
            return value

        return tree_map(make, facts)

    def _resolve(self, value):
        """Replace FX node arguments with their recorded values or references."""
        return tree_map(
            lambda item: self._values[item] if isinstance(item, fx.Node) else item, value
        )

    def _ancestor_refs(self, node):
        """Collect upstream tensor references for conservative scalar barriers."""
        pending, visited, result = list(node.all_input_nodes), set(), []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            result.extend(tensors(self._values.get(current)))
            pending.extend(current.all_input_nodes)
        return result

    @property
    def fx_graph(self):
        """Return an inspection copy of the captured FX graph."""
        import copy

        return copy.deepcopy(self._fx_graph)

    @property
    def relations(self):
        """Return the structural relations recorded by operation rules."""
        return self._relations

    @property
    def constraints(self):
        """Return the constraints checked after propagation reaches a fixed point."""
        return self._constraints

    @property
    def diagnostics(self):
        """Return build-time diagnostics, including unsupported captured operations."""
        return self._diagnostics

    @property
    def shape_expressions(self):
        """Return shape provenance indexed by FX node name."""
        return MappingProxyType({node.name: expr for node, expr in self._expressions.items()})

    def parameter(self, path):
        """Find a parameter using an original model path.

        Args:
            path: Registered parameter path, including any original alias.

        Returns:
            The shared TensorRef for that parameter object.

        Raises:
            KeyError: The path was not registered when the graph was built.
        """
        return self._parameters[path]

    def buffer(self, path):
        """Return the buffer reference for an original path, or raise KeyError."""
        return self._buffers[path]

    def calls(self, module_path=None):
        """Return operation calls, optionally filtered by an original module alias.

        Args:
            module_path: Module path to match, or None for all operation calls.
                An empty string matches calls to a root module captured as a leaf.

        Returns:
            All matching calls, including repeated uses of a shared module.
            Aliases do not identify which Python attribute spelling each call used.
        """
        return (
            self._calls
            if module_path is None
            else tuple(call for call in self._calls if module_path in call.module_paths)
        )

    def metadata(self, tensor):
        """Return captured tensor metadata without retaining its activation.

        Args:
            tensor: A reference belonging to this graph.

        Returns:
            Shape, strides, dtype, and device recorded during capture.

        Raises:
            ValueError: The tensor reference belongs to another graph.
        """
        if tensor.id not in self._refs or self._refs[tensor.id] != tensor:
            raise ValueError("Tensor belongs to another graph")
        return self._tensor_facts[tensor.id]

    def values(self):
        """Return all parameter, buffer, and captured value references."""
        return tuple(self._refs.values())

    @property
    def model(self):
        """Return the original module owning this snapshot."""
        return self._model

    def validate(self, model=None):
        """Check snapshot freshness and optional model ownership."""
        if model is not None and model is not self._model:
            raise ValueError("Graph belongs to a different model")
        self._check_fresh()

    def invalidate(self):
        """Invalidate this snapshot after an external structural mutation."""
        self._valid = False

    def interfaces(self):
        """Return unique external input and output tensor references."""
        return tuple(ref for ref in self._refs.values() if ref.id in self._interfaces)

    def tensor(self, ref):
        """Return the original registered tensor for a parameter or buffer reference."""
        self._check_fresh()
        self.metadata(ref)
        if ref.kind not in ("parameter", "buffer"):
            raise ValueError("Only registered tensors have persistent bindings")
        return _attribute(self._model, ref.paths[0])

    def bindings(self, ref):
        """Return every original (owner module, attribute name) binding of a tensor."""
        self.tensor(ref)
        result = []
        seen = set()
        for path in ref.paths:
            parent, _, name = path.rpartition(".")
            owner = self._model.get_submodule(parent) if parent else self._model
            if (id(owner), name) not in seen:
                result.append((owner, name))
                seen.add((id(owner), name))
        return tuple(result)

    def constants(self):
        """Return (reference, FX attribute path) pairs without original registered bindings.

        FX can lift ordinary tensor attributes or closed-over tensors into its
        GraphModule. Their FX paths do not prove writable original-model bindings.
        """
        return tuple(
            (ref, str(node.target))
            for node in self._fx_graph.nodes
            if node.op == "get_attr"
            for ref in tensors(self._values[node])
            if ref.kind not in ("parameter", "buffer")
        )

    def operator_spec(self, operation):
        """Return shared structural descriptors for a captured operation."""
        return self._specs.get(operation.node.name, OperatorSpec())

    def operator_rule(self, operation):
        """Return the exact operator definition used during capture."""
        return self._registry.lookup(operation.node, operation.module)

    def constant_guards(self):
        """Return registered integer tensors whose values are analysis preconditions."""
        return tuple(
            dict.fromkeys(
                ref.paths[0] for spec in self._specs.values() for ref in spec.constants if ref.paths
            )
        )

    def affected_operations(self, impact):
        """Include size-expression consumers even without removed tensor regions."""

        def expression_refs(expr):
            if isinstance(expr.value, TensorRef):
                yield expr.value
            elif isinstance(expr.value, tuple):
                yield from tensors(expr.value)
            for arg in expr.args:
                yield from expression_refs(arg)

        affected = set()
        for op in self._operations:
            if any(impact.selection(r) for r in (*op.inputs, *op.outputs, *op.bindings.values())):
                affected.add(op.node.name)
            for node in op.node.all_input_nodes:
                expr = self._expressions.get(node)
                if expr and any(impact.selection(r) for r in expression_refs(expr)):
                    affected.add(op.node.name)
        return frozenset(affected)

    def operations(self):
        """Return inspection contexts with copied FX nodes and argument containers.

        Module and tensor bindings refer to the original model. No activations are
        retained. Editing the returned argument containers cannot change analysis.
        """
        import copy
        from dataclasses import replace

        nodes = {node.name: node for node in self.fx_graph.nodes}
        return tuple(
            replace(
                ctx,
                node=nodes[ctx.node.name],
                args=copy.deepcopy(ctx.args),
                kwargs=copy.deepcopy(ctx.kwargs),
                output=copy.deepcopy(ctx.output),
                bindings=dict(ctx.bindings),
                expressions={nodes[n.name]: e for n, e in self._expressions.items()},
            )
            for ctx in self._operations
        )

    def _check_fresh(self):
        """Reject detectable structural changes without comparing weight values."""
        if not self._valid or fingerprint(self._model) != self._fingerprint:
            raise StaleGraphError("Model structure/mode/configuration changed; rebuild the graph")
        for spec in self._specs.values():
            for ref in spec.constants:
                if ref.paths:
                    value = _attribute(self._model, ref.paths[0])
                    if (
                        tuple(value.detach().cpu().reshape(-1).tolist())
                        != self._literal_values[ref.id]
                    ):
                        raise StaleGraphError(
                            "An integer tensor used as a structural constant changed"
                        )

    def propagate(self, *, remove, constraints=()):
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
            AnalysisLimitError: A supplied selection exceeds the region budget.
                Limits reached while following relations are reported as diagnostics.
        """
        self._check_fresh()
        requested = tuple(remove)
        current, queue, queued = {}, deque(), set()
        provenance, diagnostics = [], []

        def add(selection):
            if (
                selection.tensor.id not in self._refs
                or self._refs[selection.tensor.id] != selection.tensor
            ):
                raise ValueError("Selection belongs to another graph")
            old = current.get(selection.tensor.id, Selection(selection.tensor))
            delta = selection.subtract(old)
            if delta:
                current[selection.tensor.id] = old.union(delta)
                if selection.tensor.id not in queued:
                    queue.append(selection.tensor.id)
                    queued.add(selection.tensor.id)
            return delta

        for selection in requested:
            add(selection)
        disabled = set()
        while queue:
            identity = queue.popleft()
            queued.remove(identity)
            # Relations need the accumulated selection: separate arrivals may
            # jointly complete a block or a broadcast fiber on a shared path.
            source = current[identity]
            for relation in self._adjacency[identity]:
                if id(relation) in disabled:
                    continue
                try:
                    for target in relation.propagate(source):
                        delta = add(target)
                        if delta:
                            provenance.append(Provenance(source, delta, relation.reason))
                except AnalysisLimitError as error:
                    disabled.add(id(relation))
                    diagnostics.append(
                        Diagnostic(
                            "analysis_limit",
                            str(error),
                            tensors=tuple(r.id for r in relation.refs),
                            complete=False,
                        )
                    )
        all_constraints = (*self._constraints, *tuple(constraints))
        for constraint in all_constraints:
            if any(ref.id not in self._refs for ref in constraint.refs):
                raise ValueError("Constraint belongs to another graph")
            try:
                diagnostic = constraint.check(current)
            except AnalysisLimitError as error:
                diagnostic = Diagnostic("analysis_limit", str(error), complete=False)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        requirements = tuple(
            req for req in self._requirements if any(ref.id in current for ref in req.tensors)
        )
        diagnostics = tuple(dict.fromkeys(diagnostics))
        status = (
            "conflict"
            if any(d.severity == "conflict" for d in diagnostics)
            else ("unresolved" if diagnostics else "resolved")
        )
        return Impact(
            self.id,
            requested,
            MappingProxyType(dict(sorted(current.items()))),
            diagnostics,
            requirements,
            tuple(provenance),
            tuple(self._refs[k] for k in sorted(self._interfaces) if k in current),
            status,
            all_constraints,
        )

    def explain(self, impact):
        """Format the selected regions, propagation reasons, and remaining requirements.

        Args:
            impact: A result produced by this graph.

        Returns:
            A human-readable explanation without executing the model.

        Raises:
            ValueError: The impact belongs to another graph.
        """
        if impact.graph_id != self.id:
            raise ValueError("Impact belongs to another graph")
        lines = [f"Structural analysis: {impact.status}"]
        for selection in impact.selections.values():
            label = ", ".join(selection.tensor.paths) or selection.tensor.id.split(":value:")[-1]
            lines.append(
                f"- {label}: {selection.count} selected elements; regions={selection.regions}"
            )
        for step in impact.provenance:
            lines.append(f"  via {step.reason}: {step.source.tensor.id} -> {step.target.tensor.id}")
        lines.extend(f"- {d.code}: {d.message}" for d in impact.diagnostics)
        lines.extend(f"- requires {r.kind}: {r.target}: {r.detail}" for r in impact.requirements)
        return "\n".join(lines)
