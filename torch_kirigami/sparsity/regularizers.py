"""Scalar sparse penalties with a single autograd-based differentiation path."""

import math

import torch

from ..pruning.groups import ParameterGroup
from ..selection import Selection, full_region
from .values import group_values, stable_norm


def coefficients_for(count, coefficients):
    """Validate fixed nonnegative weights without differentiating through them."""
    values = (1.0,) * count if coefficients is None else tuple(coefficients)
    if len(values) != count or any(
        isinstance(v, (bool, torch.Tensor))
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or v < 0
        for v in values
    ):
        raise ValueError("Expected one finite nonnegative scalar coefficient per group")
    return tuple(float(v) for v in values)


class GroupLasso:
    """Return sum(a_g * ||W_g||_2), with no implicit size normalization.

    Args:
        groups: Live ParameterGroups from one graph/device. Equivalent groups
            count once; distinct overlapping groups contribute additively.
        coefficients: Fixed nonnegative weights aligned to supplied groups.
            Equivalent groups must have equal coefficients.

    Rebuild after structural changes. Calls do not mutate training state or
    retain graphs. Multiply the returned loss by an external overall strength.
    Frozen parameters remain part of the mathematical objective.
    """

    def __init__(self, groups, *, coefficients=None):
        groups = tuple(groups)
        weights = coefficients_for(len(groups), coefficients)
        unique, unique_weights = [], []
        for group, weight in zip(groups, weights, strict=True):
            if not isinstance(group, ParameterGroup):
                raise TypeError("Expected ParameterGroup")
            matches = [i for i, old in enumerate(unique) if group.equivalent(old)]
            if matches:
                if unique_weights[matches[0]] != weight:
                    raise ValueError("Equivalent groups have conflicting coefficients")
            else:
                unique.append(group)
                unique_weights.append(weight)
        self.groups, self.coefficients = tuple(unique), tuple(unique_weights)
        with torch.no_grad():
            group_values(self.groups)

    def __call__(self):
        """Return a differentiable scalar using current parameter values."""
        values = group_values(self.groups)
        result = torch.stack(
            [self.penalty(v) * a for v, a in zip(values, self.coefficients, strict=True)]
        ).sum()
        if not torch.isfinite(result):
            raise ValueError("Sparse penalty is nonfinite")
        return result

    def penalty(self, values):
        """Return the unweighted penalty for one flattened group."""
        return stable_norm(values)


class GroupSquaredL2(GroupLasso):
    """Return half of sum(a_g * ||W_g||_2**2); see GroupLasso bindings."""

    def penalty(self, values):
        """Return half the squared norm, whose gradient is the group itself."""
        return values.square().sum() * 0.5


class ScaleL1(GroupLasso):
    """Penalize explicitly named scale parameters, deduplicating aliases.

    Args:
        graph: Fresh graph tracking parameter identity and shape.
        parameters: Paths or TensorRefs of one-dimensional BN/gate parameters.

    No normalization/gate selection policy is inferred from the model.
    """

    def __init__(self, graph, parameters):
        refs = tuple(graph.parameter(p) if isinstance(p, str) else p for p in parameters)
        if any(len(ref.shape) != 1 for ref in refs):
            raise ValueError("ScaleL1 requires one-dimensional scale parameters")
        super().__init__(
            (
                ParameterGroup(
                    graph, tuple(Selection(ref, (full_region(ref.shape),)) for ref in refs)
                ),
            )
        )

    def penalty(self, values):
        """Return L1 with autograd's zero subgradient at zero."""
        return values.abs().sum()
