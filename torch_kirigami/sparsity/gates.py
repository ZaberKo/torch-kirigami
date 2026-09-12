"""Explicit activation gates and their structural/importance adapters."""

from dataclasses import dataclass

import torch
from torch import nn

from ..contracts import Requirement
from ..graph import DependencyGraph
from ..operation import CallEffects, OperatorRule, OperatorSpec
from ..regions import gather_region
from ..relations import AxisRelation


class ChannelGate(nn.Module):
    """Scale an explicitly chosen activation axis with one parameter per position.

    Args:
        size: Positive width of the gated axis.
        axis: Activation dimension, including negative dimensions.
        trainable: Whether gate weights receive gradients.

    Weight and mask start at one. set_mask() installs a fixed binary mask, not
    physical pruning. Retained weights need not equal one after compaction.
    Register gate operators explicitly before building a dependency graph.
    """

    def __init__(self, size, axis, trainable=True):
        super().__init__()
        if type(size) is not int or size <= 0 or type(axis) is not int:
            raise ValueError("Gate requires a positive size and integer axis")
        if type(trainable) is not bool:
            raise TypeError("trainable must be boolean")
        self.size, self.axis = size, axis
        self.weight = nn.Parameter(torch.ones(size), requires_grad=trainable)
        self.register_buffer("mask", torch.ones(size))

    def forward(self, x):
        """Apply the scale without changing rank or choosing a pruning policy."""
        if not -x.ndim <= self.axis < x.ndim or x.shape[self.axis] != self.size:
            raise ValueError("Gate activation axis has an incompatible width")
        shape = [1] * x.ndim
        shape[self.axis] = self.size
        return x * (self.weight * self.mask).reshape(shape)

    @torch.no_grad()
    def set_mask(self, mask):
        """Validate and replace the binary mask outside a live backward graph."""
        value = torch.as_tensor(mask, device=self.mask.device)
        if value.shape != self.mask.shape or not ((value == 0) | (value == 1)).all():
            raise ValueError("Expected one binary mask value per gate position")
        self.mask.copy_(value)


def _gate_rule(ctx):
    x, y = ctx.inputs[0], ctx.outputs[0]
    weight, mask = ctx.binding("weight"), ctx.binding("mask")
    relations = [AxisRelation.equal(x.axis(d), y.axis(d)) for d in range(len(x.shape))]
    relations.extend(
        (
            AxisRelation.equal(x.axis(ctx.module.axis), weight.axis(0)),
            AxisRelation.equal(weight.axis(0), mask.axis(0)),
        )
    )
    return OperatorSpec(
        relations=tuple(relations),
        requirements=(
            Requirement(
                "attribute",
                f"{ctx.module_path}.size".lstrip("."),
                (weight,),
                "Update the explicit gate width",
                (("axis", weight.axis(0)),),
            ),
        ),
    )


def _gate_effects(node, module):
    # Multiplication always allocates, even when every scale equals one. This
    # lets an immediate in-place activation prove its input is not an alias.
    return CallEffects(fresh_output=True)


def register_gate_operators(operators):
    """Register ChannelGate as an explicit leaf without adding a budget domain."""
    return operators.register(ChannelGate, OperatorRule(analyze=_gate_rule, effects=_gate_effects))


@dataclass(frozen=True)
class GateBinding:
    """Bind a named gate to candidates through actual dependency propagation.

    Args:
        graph: Graph built with register_gate_operators.
        path: Module path of a ChannelGate, or an empty string for the root.
    """

    graph: DependencyGraph
    path: str

    def __post_init__(self):
        self.graph.validate()
        if type(self.graph.model.get_submodule(self.path)) is not ChannelGate:
            raise ValueError("GateBinding requires a ChannelGate module")
        self.graph.parameter(f"{self.path}.weight".lstrip("."))

    def candidates(self, space):
        """Return candidates whose closure removes at least one gate parameter."""
        if space.graph is not self.graph:
            raise ValueError("Gate and candidate space belong to different graphs")
        ref = self.graph.parameter(f"{self.path}.weight".lstrip("."))
        result = []
        for candidate in space.candidates:
            impact = space.impact((candidate,))
            if not impact.complete:
                raise ValueError("Incomplete gate-candidate influence")
            if impact.selection(ref):
                result.append(candidate)
        return tuple(result)


class GateMagnitude:
    """Score the union of affected effective gate scales using sum(abs(weight*mask)).

    Aliases sharing both weight and mask count once. Distinct masks paired with
    a shared weight contribute separately, independently of binding order.

    Args:
        bindings: Explicit GateBindings from the planning graph. Ungated
            candidates are rejected rather than silently receiving score zero.
    """

    def __init__(self, bindings):
        self.bindings = tuple(dict.fromkeys(bindings))
        if not self.bindings or any(not isinstance(b, GateBinding) for b in self.bindings):
            raise ValueError("GateMagnitude requires nonempty GateBindings")

    @torch.no_grad()
    def __call__(self, context, candidate_batch):
        """Return finite scores without modifying any model or training state."""
        scores = []
        for candidate in candidate_batch:
            impact = context.impact(candidate.remove)
            context.require_complete(impact)
            score, found = 0.0, False
            seen = set()
            for binding in self.bindings:
                if binding.graph is not context.graph:
                    raise ValueError("Gate score uses a different dependency graph")
                ref = binding.graph.parameter(f"{binding.path}.weight".lstrip("."))
                mask = binding.graph.buffer(f"{binding.path}.mask".lstrip("."))
                identity = (ref, mask)
                if identity in seen:
                    continue
                seen.add(identity)
                selection = impact.selection(ref)
                if not selection:
                    continue
                found = True
                gate = binding.graph.model.get_submodule(binding.path)
                dtype = torch.float64 if gate.weight.dtype == torch.float64 else torch.float32
                values = gate.weight.to(dtype) * gate.mask.to(dtype)
                score += sum(gather_region(values, r).abs().sum().item() for r in selection.regions)
            if not found:
                raise ValueError("Candidate has no bound gate; restrict candidates explicitly")
            scores.append(score)
        return scores
