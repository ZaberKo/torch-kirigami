"""Build structural records from FX capture, then release construction state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import MappingProxyType
from uuid import uuid4

import torch
from torch import fx, nn

from ..bindings import storage_key
from ..errors import UnsupportedOperation
from .capture import CaptureResult, capture, fingerprint, tree_map
from .contracts import (
    Barrier,
    Constraint,
    Diagnostic,
    LayoutConstraint,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from .data import GraphData, resolve_attribute, validate_reference
from .operation import CallRef, OperationContext, OperatorSpec, TensorFacts, tensors
from .operators.shapes import CallArgumentConstraint, dependencies
from .registry import OperatorRegistry
from .relations import AxisRelation, Relation
from .selection import TensorRef


class GraphBuilder:
    """Own temporary mutable state for one build, including pre-capture aliases.

    Args:
        model: Original module whose bindings must survive FX path canonicalization.
        operators: Optional local registry; its tables are copied for this build.
    """

    def __init__(self, model: nn.Module, operators: OperatorRegistry | None) -> None:
        self.id = uuid4().hex
        self.model = model
        self.registry = (operators if operators is not None else OperatorRegistry.default()).copy()
        self.fingerprint = fingerprint(model)
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
                        self.registry.modules,
                        self.registry.functions,
                        self.registry.methods,
                    )
                    for target, rule in table.items()
                ),
            }
        )
        self.refs: dict[str, TensorRef] = {}
        self.tensor_facts: dict[str, TensorFacts] = {}
        self.parameters: dict[str, TensorRef] = {}
        self.buffers: dict[str, TensorRef] = {}
        self.expressions: dict[fx.Node, ShapeExpr] = {}
        self.literal_values: dict[str, tuple[int, ...]] = {}
        self.relations: list[Relation] = []
        self.constraints: list[Constraint] = []
        self.requirements: list[Requirement] = []
        self.diagnostics: list[Diagnostic] = []
        self.calls: list[CallRef] = []
        self.operations: dict[str, OperationContext] = {}
        self.specs: dict[str, OperatorSpec] = {}
        self.interfaces: set[str] = set()
        self._values: dict[fx.Node, object] = {}
        self._unused: set[str] = set()
        self._by_object: dict[int, TensorRef] = {}
        self._module_paths: defaultdict[int, list[str]] = defaultdict(list)

    def build(self, args: tuple[object, ...], kwargs: dict[str, object]) -> GraphData:
        """Capture and analyze once, returning completed records for graph queries."""
        storage_constraints = self._collect_bindings()
        captured = capture(self.model, args, kwargs, self.registry)
        for identity, original in captured.buffer_aliases.items():
            self._by_object[identity] = self._by_object[id(original)]
        self._record_static_metadata(captured.module)
        self.constraints.extend(storage_constraints)
        self._analyze_nodes(captured)
        constraints = self._complete_constraints()
        constants = tuple(
            (ref, str(node.target))
            for node in captured.module.graph.nodes
            if node.op == "get_attr"
            for ref in tensors(self._values[node])
            if ref.kind not in ("parameter", "buffer")
        )
        # Queries retain structural references, never the temporary GraphModule
        # and its isolated get_attr buffers or the builder's per-node value table.
        captured.module.graph.owning_module = None
        return GraphData(
            id=self.id,
            model=self.model,
            registry=self.registry,
            fingerprint=self.fingerprint,
            context=self.context,
            fx_graph=captured.module.graph,
            capture_signature=captured.signature,
            refs=MappingProxyType(self.refs),
            tensor_facts=MappingProxyType(self.tensor_facts),
            parameters=MappingProxyType(self.parameters),
            buffers=MappingProxyType(self.buffers),
            expressions=MappingProxyType(self.expressions),
            literal_values=MappingProxyType(self.literal_values),
            relations=tuple(self.relations),
            constraints=constraints,
            requirements=tuple(self.requirements),
            diagnostics=tuple(self.diagnostics),
            calls=tuple(self.calls),
            operations=MappingProxyType(self.operations),
            specs=MappingProxyType(self.specs),
            constant_refs=tuple(
                dict.fromkeys(ref for spec in self.specs.values() for ref in spec.constants)
            ),
            interfaces=frozenset(self.interfaces),
            constants=constants,
        )

    def _collect_bindings(self) -> tuple[Constraint, ...]:
        """Collect original aliases and storage conflicts before FX changes paths."""
        storage_groups: defaultdict[tuple[str, int], list[TensorRef]] = defaultdict(list)
        for path, module in self.model.named_modules(remove_duplicate=False):
            self._module_paths[id(module)].append(path)
        for kind, entries, paths in (
            ("parameter", self.model.named_parameters(remove_duplicate=False), self.parameters),
            ("buffer", self.model.named_buffers(remove_duplicate=False), self.buffers),
        ):
            grouped: dict[int, tuple[torch.Tensor, list[str]]] = {}
            for path, tensor in entries:
                grouped.setdefault(id(tensor), (tensor, []))[1].append(path)
            for identity, (tensor, aliases) in grouped.items():
                ref = TensorRef(
                    f"{self.id}:{kind}:{aliases[0]}", tuple(tensor.shape), kind, tuple(aliases)
                )
                self.tensor_facts[ref.id] = TensorFacts(
                    tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device
                )
                self._by_object[identity] = ref
                self.refs[ref.id] = ref
                for path in aliases:
                    paths[path] = ref
                key = storage_key(tensor)
                if key is not None:
                    storage_groups[key].append(ref)
        self._unused.update(self.refs)
        constraints = []
        for refs in storage_groups.values():
            unique = tuple(dict.fromkeys(refs))
            if len(unique) > 1:
                constraints.append(
                    Barrier(
                        unique,
                        "Distinct tensors share storage; overlapping/view compaction is unsupported",
                        code="storage_alias",
                    )
                )
        return tuple(constraints)

    def _record_static_metadata(self, module: fx.GraphModule) -> None:
        """Keep guards for shape and layout reads that FX evaluated as constants."""
        for path, kind, argument, observed in module.meta.get("kirigami_static_metadata", ()):
            ref = self.parameters.get(path, self.buffers.get(path))
            label = f"metadata {path}.{kind}"
            if kind in ("shape", "dimension", "numel"):
                expression = ShapeExpr(kind, (ref, argument) if kind == "dimension" else ref)
                hint = (
                    "If only one dimension is needed, read tensor.size(dim) instead of "
                    "the whole shape, then rebuild the dependency graph."
                    if kind == "shape"
                    else ""
                )
                self.constraints.append(
                    CallArgumentConstraint(label, ((expression, observed),), hint=hint)
                )
            elif kind in ("stride", "contiguous"):
                # Logical dependencies do not decide how parameters are allocated.
                # The execution layer checks this read against the final recipe.
                self.requirements.append(
                    Requirement(
                        "metadata_layout",
                        path,
                        (ref,),
                        "Preserve metadata read by unrecorded Python code",
                        (("kind", kind), ("argument", argument), ("observed", observed)),
                    )
                )

    def _analyze_nodes(self, captured: CaptureResult) -> None:
        """Create structural values and analyze calls in FX topological order."""
        module = captured.module
        for node in module.graph.nodes:
            if node.op == "output":
                self._values[node] = self._resolve(node.args[0])
                self.interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                value = resolve_attribute(module, str(node.target))
                if id(value) in self._by_object:
                    self._values[node] = self._by_object[id(value)]
                    if id(value) not in captured.mutable_buffers:
                        self._record_literal(self._values[node], value)
                    continue
            self._values[node] = self._make_values(node, captured.facts[node])
            if node.op == "placeholder":
                self.interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                self._record_literal(
                    self._values[node], resolve_attribute(module, str(node.target))
                )
                continue
            called_module = (
                module.get_submodule(str(node.target)) if node.op == "call_module" else None
            )
            paths = (
                tuple(self._module_paths.get(id(called_module), ()))
                if called_module is not None
                else ()
            )
            bindings = {}
            if called_module is not None:
                for name, value in (
                    *called_module.named_parameters(remove_duplicate=False),
                    *called_module.named_buffers(remove_duplicate=False),
                ):
                    ref = self._by_object.get(id(value))
                    if ref is not None:
                        bindings[name] = ref
            ctx = OperationContext(
                node,
                self._resolve(node.args),
                self._resolve(node.kwargs),
                self._values[node],
                called_module,
                paths[0] if paths else None,
                bindings,
                self.expressions,
                MappingProxyType(self.tensor_facts),
                MappingProxyType(self.literal_values),
                self.id,
            )
            self._analyze_operation(ctx, paths)

    def _record_literal(self, ref: TensorRef, value: object) -> None:
        """Retain small integer constants used by shape and indexing rules."""
        if (
            isinstance(value, torch.Tensor)
            and not value.is_floating_point()
            and not value.is_complex()
            and value.numel() <= 4096
        ):
            self.literal_values[ref.id] = tuple(value.detach().cpu().reshape(-1).tolist())

    def _complete_constraints(self) -> tuple[Constraint, ...]:
        """Combine call rules with cross-call layout and tensor usage constraints."""
        ports = defaultdict(list)
        for relation in self.relations:
            if isinstance(relation, AxisRelation):
                for port in (relation.left, relation.right):
                    ports[port.tensor.id].append(port)
        for ref in self.refs.values():
            scoped = ports[ref.id] if any(p.scope is not None for p in ports[ref.id]) else ()
            self.constraints.append(LayoutConstraint(ref, tuple(scoped)))
            if ref.id not in self._unused:
                self.constraints.extend(NonEmpty(ref.axis(d)) for d in range(len(ref.shape)))
            else:
                self.constraints.append(
                    Barrier(
                        (ref,), "No structural usage rule for this tensor", code="unbound_tensor"
                    )
                )
        layouts = tuple(layout for spec in self.specs.values() for layout in spec.layouts)
        return tuple(
            replace(
                constraint,
                layouts=tuple(layout for layout in layouts if layout.tensor in constraint.refs),
            )
            if isinstance(constraint, CallArgumentConstraint)
            else constraint
            for constraint in self.constraints
        )

    def _analyze_operation(self, ctx: OperationContext, paths: tuple[str, ...]) -> None:
        """Validate one call's semantics and collect its structural facts.

        Unsupported calls become barriers on their data dependencies. Other calls
        remain analyzable, and their declared shape expressions feed later nodes.
        """
        node = ctx.node
        self.calls.append(CallRef(node.name, paths, ctx.inputs, ctx.outputs))
        self.operations[node.name] = ctx
        rule = self.registry.lookup(node, ctx.module)
        try:
            if rule is None:
                raise UnsupportedOperation(f"No semantics registered for {node.target}")
            result = rule.analyze(ctx)
            if not isinstance(result, OperatorSpec):
                raise TypeError("OperatorRule.analyze must return OperatorSpec")
            self._validate_spec(result)
            if any(not ref.paths for ref in result.constants):
                raise UnsupportedOperation(
                    "Structural constants require registered parameter/buffer value guards; "
                    "register integer index tensors as buffers"
                )
            self.specs[node.name] = result
            if result.expression is not None:
                self.expressions[node] = result.expression
        except UnsupportedOperation as error:
            refs = tuple(
                dict.fromkeys(
                    (*ctx.inputs, *ctx.outputs, *ctx.bindings.values(), *dependencies(ctx))
                )
            )
            # Unknown operations can use values to choose sizes/indices. A
            # reduction preserves no deleted coordinates but still changes
            # those values; the barrier therefore covers every data ancestor.
            refs = tuple(dict.fromkeys((*refs, *self._ancestor_refs(node))))
            barrier = Barrier(refs, str(error), node.name)
            self.constraints.append(barrier)
            self.diagnostics.append(
                Diagnostic(
                    "unsupported", str(error), node=node.name, tensors=tuple(r.id for r in refs)
                )
            )
            self._unused.difference_update(r.id for r in refs)
            return
        self.relations.extend(result.relations)
        self.constraints.extend(result.constraints)
        self.requirements.extend(result.requirements)
        self.requirements.extend(
            Requirement(
                "partitioned_compaction",
                node.name,
                (descriptor.tensor,),
                "Retain declared physical partitions and concatenate in original order",
            )
            for descriptor in result.layouts
        )
        if result.expression is None and dependencies(ctx):
            self.constraints.append(
                CallArgumentConstraint.from_operation(
                    ctx,
                    checked_arguments=tuple(a for r in result.requirements for a in r.arguments),
                )
            )
        for relation in result.relations:
            self._unused.difference_update(r.id for r in relation.refs)
        for constraint in result.constraints:
            self._unused.difference_update(r.id for r in constraint.refs)

    def _make_values(self, node: fx.Node, facts: object) -> object:
        """Replace tensor facts with references while preserving the result tree."""
        counter = [0]

        def make(value: object) -> object:
            """Replace captured tensor leaves with graph-local references."""
            if isinstance(value, TensorFacts):
                ref = TensorRef(
                    f"{self.id}:value:{node.name}:{counter[0]}",
                    value.shape,
                    "input" if node.op == "placeholder" else "value",
                )
                counter[0] += 1
                self.refs[ref.id] = ref
                self.tensor_facts[ref.id] = value
                return ref
            return value

        return tree_map(make, facts)

    def _resolve(self, value: object) -> object:
        """Replace FX node arguments with their recorded values or references."""
        return tree_map(
            lambda item: self._values[item] if isinstance(item, fx.Node) else item, value
        )

    def _ancestor_refs(self, node: fx.Node) -> tuple[TensorRef, ...]:
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

    def _validate_spec(self, spec: OperatorSpec) -> None:
        """Reject extension descriptors referring outside the captured snapshot."""
        refs = [ref for item in (*spec.relations, *spec.constraints) for ref in item.refs]
        refs.extend(ref for item in spec.requirements for ref in item.refs)
        refs.extend(item.axis.tensor for item in spec.candidates)
        refs.extend(
            item.alignment_axis.tensor
            for item in spec.candidates
            if item.alignment_axis is not None
        )
        refs.extend(item.binding for item in spec.candidates if item.binding is not None)
        refs.extend(item.tensor for item in spec.layouts)
        refs.extend(spec.constants)
        if spec.expression is not None:
            refs.extend(spec.expression.refs)
        for ref in refs:
            validate_reference(self.refs, ref)
