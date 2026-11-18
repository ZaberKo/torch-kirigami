"""Explicit activation gates, their operator rule, and gate-based importance scores."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import fx, nn

from ..dep_graph.contracts import AttributeBinding, Impact, ShapeExpr
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.operation import Operation, OperatorRule, OperatorSpec, TensorEffects
from ..dep_graph.registry import OperatorRegistry
from ..dep_graph.relations import AxisRelation
from ..dep_graph.selection import TensorRef, gather_region
from ..pruning.pruner import Pruner
from ..pruning.types import Candidate, MetricContext


class ChannelGate(nn.Module):
    """Multiply one activation axis by a learnable scale per position.

    The output is `x * weight * mask`, broadcast along `axis`. Both `weight`
    and `mask` start at one. `set_mask()` installs a fixed binary mask; it
    does not prune the model physically. Physical pruning keeps the remaining
    gate weights, so they need not equal one after compaction.

    Call `register_gate_operators` on the operator registry before building a
    dependency graph for a model that contains gates. The gate rule adds no
    candidate axis of its own, so gates follow the candidates of neighboring
    layers.

    Args:
        size: Positive width of the gated axis.
        axis: Activation dimension to scale. Negative values count from the end.
        trainable: Whether `weight` requires gradients.

    Attributes:
        size: Number of activation positions scaled by the gate.
        axis: Activation dimension scaled by the gate.
        weight: Scale parameter for each position, trainable or frozen.
        mask: Binary buffer that multiplies each position. It is never trained.

    Raises:
        ValueError: `size` is not a positive integer or `axis` is not an integer.
        TypeError: `trainable` is not a boolean.
    """

    def __init__(self, size: int, axis: int, trainable: bool = True) -> None:
        super().__init__()
        if type(size) is not int or size <= 0 or type(axis) is not int:
            raise ValueError("Gate requires a positive size and integer axis")
        if type(trainable) is not bool:
            raise TypeError("trainable must be boolean")
        self.size, self.axis = size, axis
        self.weight = nn.Parameter(torch.ones(size), requires_grad=trainable)
        self.register_buffer("mask", torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the gated axis of an activation.

        Args:
            x: Activation whose dimension `axis` has width `size`.

        Returns:
            A new tensor with the same shape as `x`.

        Raises:
            ValueError: `x` has no dimension `axis`, or that dimension does
                not have width `size`.
        """
        if not -x.ndim <= self.axis < x.ndim or x.shape[self.axis] != self.size:
            raise ValueError("Gate activation axis has an incompatible width")
        shape = [1] * x.ndim
        shape[self.axis] = self.size
        return x * (self.weight * self.mask).reshape(shape)

    @torch.no_grad()
    def set_mask(self, mask: torch.Tensor) -> None:
        """Replace the binary mask in place.

        The copy runs under `torch.no_grad()`. Call it outside a live forward and
        backward pass, because an autograd graph may have saved the old mask.

        Args:
            mask: Tensor-like value with one entry per gate position, each zero
                or one.

        Raises:
            ValueError: The mask has the wrong shape or a value other than zero
                or one.
        """
        value = torch.as_tensor(mask, device=self.mask.device)
        if value.shape != self.mask.shape or not ((value == 0) | (value == 1)).all():
            raise ValueError("Expected one binary mask value per gate position")
        self.mask.copy_(value)


def _gate_rule(operation: Operation) -> OperatorSpec:
    """Relate gate input, output, weight, and mask axes and bind the `size` attribute."""
    x, y = operation.inputs[0], operation.outputs[0]
    weight, mask = operation.bindings.get("weight"), operation.bindings.get("mask")
    relations = [AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape))]
    relations.extend(
        (
            AxisRelation.equal(x.axis(operation.module.axis), weight.axis(0)),
            AxisRelation.equal(weight.axis(0), mask.axis(0)),
        )
    )
    return OperatorSpec(
        relations=tuple(relations),
        requirements=(
            AttributeBinding(
                f"{operation.module_path}.size".lstrip("."),
                ShapeExpr("dimension", (weight.axis(0).tensor, weight.axis(0).dim)),
                "Update the explicit gate width",
            ),
        ),
    )


def _gate_effects(node: fx.Node, module: nn.Module | None) -> TensorEffects:
    """Declare that gate multiplication always produces a fresh activation."""
    # Multiplication always allocates, even when every scale equals one. This
    # lets an immediate in-place activation prove its input is not an alias.
    return TensorEffects(writes_inputs=(), outputs_are_fresh=True)


class ChannelGateRule(OperatorRule):
    """Operator rule for `ChannelGate` calls.

    The rule relates the gate's input, output, weight, and mask coordinates,
    keeps the module's `size` attribute equal to the compact width, and declares
    that the output never aliases the input. It declares no candidate axis.
    """

    build_spec = staticmethod(_gate_rule)
    tensor_effects = staticmethod(_gate_effects)


def register_gate_operators(operators: OperatorRegistry) -> OperatorRegistry:
    """Register `ChannelGate` as a leaf module with its own operator rule.

    Call this before building a dependency graph for a model that contains
    gates. The rule adds no candidate axis, so a gate is pruned only through
    candidates declared by other operators.

    Args:
        operators: Registry to extend.

    Returns:
        The same registry, allowing registrations to be chained.

    Raises:
        ValueError: `operators` already has a rule for `ChannelGate`.
    """
    return operators.register(ChannelGate, ChannelGateRule())


@dataclass(frozen=True)
class GateBinding:
    """Associate one `ChannelGate` in a dependency graph with the candidates that prune it.

    The association is computed by dependency propagation, not by module names.

    Attributes:
        graph: Dependency graph built with a registry passed through
            `register_gate_operators`.
        path: Module path of the `ChannelGate`. An empty path means the root
            module is the gate.

    Raises:
        AttributeError: The model has no module at `path`.
        ValueError: The module at `path` is not exactly a `ChannelGate`.
        KeyError: The graph does not track the gate's weight.
        torch_kirigami.StaleGraphError: The graph no longer matches its model.
    """

    graph: DependencyGraph
    path: str

    def __post_init__(self) -> None:
        self.graph.validate()
        if type(self.graph.model.get_submodule(self.path)) is not ChannelGate:
            raise ValueError("GateBinding requires a ChannelGate module")
        self.graph.parameter(f"{self.path}.weight".lstrip("."))

    def candidates(
        self, pruner: Pruner, candidates: tuple[Candidate, ...]
    ) -> tuple[Candidate, ...]:
        """Return the candidates whose removal would remove at least one gate weight.

        Args:
            pruner: Pruner that uses this binding's graph.
            candidates: Candidates to inspect.

        Returns:
            The matching candidates in input order.

        Raises:
            ValueError: The pruner uses a different graph, or the dependency
                impact of a candidate is incomplete.
        """
        if pruner.graph is not self.graph:
            raise ValueError("Gate and candidate space belong to different graphs")
        ref = self.graph.parameter(f"{self.path}.weight".lstrip("."))
        result = []
        for candidate in candidates:
            impact = pruner.impact((candidate,))
            if not impact.complete:
                raise ValueError("Incomplete gate-candidate influence")
            if impact.selection(ref):
                result.append(candidate)
        return tuple(result)


class GateMagnitude:
    """Score candidates by the effective gate scales that their removal would add.

    A candidate's score is `sum(abs(weight * mask))` over the gate positions
    that its removal selects and that the already accepted removals do not.
    Positions already selected therefore contribute zero. Bindings that share
    both weight and mask count once. Distinct masks paired with a shared weight
    contribute separately, regardless of binding order. Scoring reads the
    current gate values supplied by the caller and does not train or
    recalibrate them. Effective scales are read once per `score()` call and
    are never cached between calls.

    Args:
        bindings: Nonempty `GateBinding` objects for the planning graph.
            Duplicate bindings are ignored.

    Attributes:
        bindings: Distinct bindings in first-occurrence order.

    Raises:
        ValueError: `bindings` is empty or contains a non-`GateBinding` object.
    """

    def __init__(self, bindings: tuple[GateBinding, ...]) -> None:
        self.bindings = tuple(dict.fromkeys(bindings))
        if not self.bindings or any(not isinstance(b, GateBinding) for b in self.bindings):
            raise ValueError("GateMagnitude requires nonempty GateBindings")

    @torch.no_grad()
    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score each candidate as an addition to the already accepted removals.

        Candidates are scored independently of each other.

        Args:
            context: Planner access to the graph and dependency impacts.
            candidates: Candidates to score.
            accepted_impact: Complete dependency impact of the removals that
                are already accepted.

        Returns:
            One real magnitude per candidate, in input order. Effective scales
            retain complex components and use at least float32 precision.

        Raises:
            ValueError: A binding uses a different graph, or a candidate does
                not affect any bound gate. Ungated candidates are rejected
                rather than scored as zero; restrict the candidates first.
            torch_kirigami.pruning.PlanningError: The accepted impact or a
                candidate impact is incomplete.
        """
        context.require_complete(accepted_impact)
        if not candidates:
            return []
        prepared: list[tuple[TensorRef, torch.Tensor]] = []
        seen: set[tuple[TensorRef, TensorRef]] = set()
        for binding in self.bindings:
            if binding.graph is not context.graph:
                raise ValueError("Gate score uses a different dependency graph")
            ref = binding.graph.parameter(f"{binding.path}.weight".lstrip("."))
            mask = binding.graph.buffer(f"{binding.path}.mask".lstrip("."))
            identity = (ref, mask)
            if identity in seen:
                continue
            seen.add(identity)
            gate = binding.graph.model.get_submodule(binding.path)
            dtype = torch.promote_types(torch.result_type(gate.weight, gate.mask), torch.float32)
            # Statistics are shared only within this call. Training or mask changes
            # between score calls must immediately affect the next batch.
            prepared.append((ref, gate.weight.to(dtype) * gate.mask.to(dtype)))
        scores = [0.0] * len(candidates)
        terms_by_device: dict[torch.device, list[torch.Tensor]] = {}
        accumulation_order: list[tuple[int, torch.device, int]] = []
        for candidate_index, candidate in enumerate(candidates):
            impact = context.impact(accepted_impact.requested + candidate.remove)
            context.require_complete(impact)
            found = False
            for ref, values in prepared:
                selection = impact.selection(ref).subtract(accepted_impact.selection(ref))
                if not selection:
                    continue
                found = True
                for region in selection.regions:
                    device_terms = terms_by_device.setdefault(values.device, [])
                    accumulation_order.append((candidate_index, values.device, len(device_terms)))
                    device_terms.append(gather_region(values, region).abs().sum())
            if not found:
                # A previously selected gated candidate has zero marginal cost.
                # An unrelated candidate must still be rejected, even if prior
                # selections happen to include one of this metric's gates.
                candidate_impact = context.impact(candidate.remove)
                context.require_complete(candidate_impact)
                if not any(candidate_impact.selection(ref) for ref, _mask in seen):
                    raise ValueError("Candidate has no bound gate; restrict candidates explicitly")
        # One transfer per device, preserving the original Python addition order.
        transferred = {
            device: torch.stack(values).cpu().tolist() for device, values in terms_by_device.items()
        }
        for candidate_index, device, term_index in accumulation_order:
            scores[candidate_index] += transferred[device][term_index]
        return scores
