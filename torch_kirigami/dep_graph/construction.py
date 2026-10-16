"""Build structural records from FX capture, then release construction state."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from types import MappingProxyType

import torch
from torch import fx, nn

from ..bindings import storage_key
from ..errors import UnsupportedOperation
from ..settings import config
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
from .graph_state import DependencyGraphState, resolve_attribute, validate_reference
from .operation import Operation, OperatorSpec, TensorFacts, tensors
from .operators.shapes import CallArgumentConstraint, dependencies
from .registry import OperatorRegistry
from .relations import AxisRelation
from .selection import TensorRef


class _Analyzer:
    """Own temporary mutable state for one build, including pre-capture aliases.

    Args:
        model: Original module whose bindings must survive FX path canonicalization.
        operators: Optional local registry; its tables are copied for this build.
    """

    def __init__(self, model: nn.Module, operators: OperatorRegistry | None) -> None:
        registry = (operators if operators is not None else OperatorRegistry.default()).copy()
        self.data = DependencyGraphState(
            model=model,
            registry=registry,
            fingerprint=fingerprint(model),
            context=MappingProxyType(
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
                        for table in (registry.modules, registry.functions, registry.methods)
                        for target, rule in table.items()
                    ),
                }
            ),
        )
        self._started = False
        self._values: dict[fx.Node, object] = {}
        self._unused: set[str] = set()
        self._by_object: dict[int, TensorRef] = {}
        self._module_paths: defaultdict[int, list[str]] = defaultdict(list)

    def build(self, args: tuple[object, ...], kwargs: dict[str, object]) -> DependencyGraphState:
        """Populate one analysis; publish only on success and release scratch state."""
        if self._started:
            raise RuntimeError("An analyzer can only build once")
        self._started = True
        try:
            storage_constraints = self._collect_bindings()
            captured = capture(self.data.model, args, kwargs, self.data.registry)
            for identity, original in captured.buffer_aliases.items():
                self._by_object[identity] = self._by_object[id(original)]
            self._record_static_metadata(captured.module)
            self.data.constraints.extend(storage_constraints)
            self._analyze_nodes(captured)
            self.data.constraints = self._complete_constraints()
            self.data.constants = tuple(
                (ref, str(node.target))
                for node in captured.module.graph.nodes
                if node.op == "get_attr"
                for ref in tensors(self._values[node])
                if ref.kind not in ("parameter", "buffer")
            )
            self.data.constant_refs = tuple(
                dict.fromkeys(ref for spec in self.data.specs.values() for ref in spec.constants)
            )
            # Retain the FX graph, without its temporary GraphModule/buffer bindings.
            captured.module.graph.owning_module = None
            self.data.fx_graph = captured.module.graph
            self.data.capture_signature = captured.signature
            self.data.complete = True
            return self.data
        finally:
            self._values.clear()
            self._unused.clear()
            self._by_object.clear()
            self._module_paths.clear()

    def _collect_bindings(self) -> tuple[Constraint, ...]:
        """Collect original aliases and storage conflicts before FX changes paths."""
        storage_groups: defaultdict[tuple[str, int], list[TensorRef]] = defaultdict(list)
        for path, module in self.data.model.named_modules(remove_duplicate=False):
            self._module_paths[id(module)].append(path)
        for kind, entries, paths in (
            (
                "parameter",
                self.data.model.named_parameters(remove_duplicate=False),
                self.data.parameters,
            ),
            ("buffer", self.data.model.named_buffers(remove_duplicate=False), self.data.buffers),
        ):
            grouped: dict[int, tuple[torch.Tensor, list[str]]] = {}
            for path, tensor in entries:
                grouped.setdefault(id(tensor), (tensor, []))[1].append(path)
            for identity, (tensor, aliases) in grouped.items():
                ref = TensorRef(
                    f"{self.data.id}:{kind}:{aliases[0]}", tuple(tensor.shape), kind, tuple(aliases)
                )
                self.data.tensor_facts[ref.id] = TensorFacts(
                    tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device
                )
                self._by_object[identity] = ref
                self.data.refs[ref.id] = ref
                for path in aliases:
                    paths[path] = ref
                key = storage_key(tensor)
                if key is not None:
                    storage_groups[key].append(ref)
        self._unused.update(self.data.refs)
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
            ref = self.data.parameters.get(path, self.data.buffers.get(path))
            label = f"metadata {path}.{kind}"
            if kind in ("shape", "dimension", "numel"):
                expression = ShapeExpr(kind, (ref, argument) if kind == "dimension" else ref)
                hint = (
                    "If only one dimension is needed, read tensor.size(dim) instead of "
                    "the whole shape, then rebuild the dependency graph."
                    if kind == "shape"
                    else ""
                )
                self.data.constraints.append(
                    CallArgumentConstraint(label, ((expression, observed),), hint=hint)
                )
            elif kind in ("stride", "contiguous"):
                # Logical dependencies do not decide how parameters are allocated.
                # The execution layer checks this read against the final recipe.
                self.data.requirements.append(
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
                self.data.interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                value = resolve_attribute(module, str(node.target))
                if id(value) in self._by_object:
                    self._values[node] = self._by_object[id(value)]
                    if storage_key(value) not in captured.mutable_storages:
                        self._record_literal(self._values[node], value)
                    continue
            self._values[node] = self._make_values(node, captured.facts[node])
            if node.op == "placeholder":
                self.data.interfaces.update(ref.id for ref in tensors(self._values[node]))
                continue
            if node.op == "get_attr":
                value = resolve_attribute(module, str(node.target))
                if (
                    not isinstance(value, torch.Tensor)
                    or storage_key(value) not in captured.mutable_storages
                ):
                    self._record_literal(self._values[node], value)
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
            ctx = Operation(
                node,
                self._resolve(node.args),
                self._resolve(node.kwargs),
                self._values[node],
                called_module,
                paths,
                bindings,
                MappingProxyType(self.data.expressions),
                MappingProxyType(self.data.tensor_facts),
                MappingProxyType(self.data.literal_values),
                self.data.id,
                captured.effects[node],
            )
            self._analyze_operation(ctx)

    def _record_literal(self, ref: TensorRef, value: object) -> None:
        """Retain small integer constants used by shape and indexing rules."""
        if (
            isinstance(value, torch.Tensor)
            and not value.is_floating_point()
            and not value.is_complex()
            and value.numel() <= config.max_literal_elements
        ):
            self.data.literal_values[ref.id] = tuple(value.detach().cpu().reshape(-1).tolist())

    def _complete_constraints(self) -> list[Constraint]:
        """Combine call rules with cross-call layout and tensor usage constraints."""
        ports = defaultdict(list)
        for relation in self.data.relations:
            if isinstance(relation, AxisRelation):
                for port in (relation.left, relation.right):
                    ports[port.tensor.id].append(port)
        for ref in self.data.refs.values():
            scoped = ports[ref.id] if any(p.scope is not None for p in ports[ref.id]) else ()
            self.data.constraints.append(LayoutConstraint(ref, tuple(scoped)))
            if ref.id not in self._unused:
                self.data.constraints.extend(NonEmpty(ref.axis(d)) for d in range(len(ref.shape)))
            else:
                self.data.constraints.append(
                    Barrier(
                        (ref,), "No structural usage rule for this tensor", code="unbound_tensor"
                    )
                )
        layouts = tuple(layout for spec in self.data.specs.values() for layout in spec.layouts)
        return [
            replace(
                constraint,
                layouts=tuple(layout for layout in layouts if layout.tensor in constraint.refs),
            )
            if isinstance(constraint, CallArgumentConstraint)
            else constraint
            for constraint in self.data.constraints
        ]

    def _analyze_operation(self, ctx: Operation) -> None:
        """Validate one call's semantics and collect its structural facts.

        Unsupported calls become barriers on their data dependencies. Other calls
        remain analyzable, and their declared shape expressions feed later nodes.
        """
        node = ctx.node
        self.data.operations[node.name] = ctx
        rule = self.data.registry.lookup(node, ctx.module)
        try:
            if rule is None:
                raise UnsupportedOperation(f"No semantics registered for {node.target}")
            result = rule.build_spec(ctx)
            if not isinstance(result, OperatorSpec):
                raise TypeError("OperatorRule.build_spec must return OperatorSpec")
            self._validate_spec(result)
            if any(not ref.paths for ref in result.constants):
                raise UnsupportedOperation(
                    "Structural constants require registered parameter/buffer value guards; "
                    "register integer index tensors as buffers"
                )
            self.data.specs[node.name] = result
            if result.expression is not None:
                self.data.expressions[node] = result.expression
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
            self.data.constraints.append(barrier)
            self.data.diagnostics.append(
                Diagnostic(
                    "unsupported", str(error), node=node.name, tensors=tuple(r.id for r in refs)
                )
            )
            self._unused.difference_update(r.id for r in refs)
            return
        self.data.relations.extend(result.relations)
        self.data.constraints.extend(result.constraints)
        self.data.requirements.extend(result.requirements)
        self.data.requirements.extend(
            Requirement(
                "partitioned_compaction",
                node.name,
                (descriptor.tensor,),
                "Retain declared physical partitions and concatenate in original order",
            )
            for descriptor in result.layouts
        )
        if result.expression is None and dependencies(ctx):
            self.data.constraints.append(
                CallArgumentConstraint.from_operation(
                    ctx,
                    checked_arguments=tuple(
                        a
                        for r in result.requirements
                        if isinstance(r, Requirement)
                        for a in r.arguments
                    ),
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
                    f"{self.data.id}:value:{node.name}:{counter[0]}",
                    value.shape,
                    "input" if node.op == "placeholder" else "value",
                )
                counter[0] += 1
                self.data.refs[ref.id] = ref
                self.data.tensor_facts[ref.id] = value
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
            validate_reference(self.data.refs, ref)
