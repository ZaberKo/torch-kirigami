"""Scalar sparse penalties with a single autograd-based differentiation path."""

from __future__ import annotations

import math

import torch

from ..graph import DependencyGraph
from ..pruning.groups import ParameterGroup, group_equivalence_classes
from ..selection import Selection, TensorRef, full_region
from .values import group_penalty, group_values, reduction_plan, squared_norm, stable_norm


def coefficients_for(count: int, coefficients: tuple[int | float, ...] | None) -> tuple[float, ...]:
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
    Frozen parameters remain part of the mathematical objective. Built-in L2
    penalties support first-order autograd only; double backward is unsupported.
    """

    def __init__(
        self,
        groups: tuple[ParameterGroup, ...],
        *,
        coefficients: tuple[int | float, ...] | None = None,
    ) -> None:
        groups = tuple(groups)
        weights = coefficients_for(len(groups), coefficients)
        unique, unique_weights = [], []
        for indices in group_equivalence_classes(groups):
            weight = weights[indices[0]]
            if any(weights[i] != weight for i in indices):
                raise ValueError("Equivalent groups have conflicting coefficients")
            unique.append(groups[indices[0]])
            unique_weights.append(weight)
        self.groups, self.coefficients = tuple(unique), tuple(unique_weights)
        self._reduction_plan = reduction_plan(self.groups)
        with torch.no_grad():
            group_values(self.groups)

    def __call__(self) -> torch.Tensor:
        """Return a differentiable scalar using current parameter values."""
        native = {GroupLasso: "l2", GroupSquaredL2: "squared_l2", ScaleL1: "l1"}
        # Exact types keep the optimized path from bypassing a subclass penalty().
        if type(self) in native:
            kind = native[type(self)]
            if all(a == 1 for a in self.coefficients):
                result = group_penalty(self.groups, kind, self._reduction_plan)
            else:
                # Apply weights before the nonlinear reduction. Computing an
                # unweighted square first can lose a representable weighted
                # result. FP64 also keeps finite Python coefficients from being
                # rounded to zero/infinity when parameters are FP32 or lower.
                values = group_values(self.groups)
                if kind == "squared_l2":
                    result = squared_norm(*values, coefficients=self.coefficients)
                else:
                    terms = [
                        stable_norm(value.double(), coefficient=coefficient)
                        if kind == "l2"
                        else (value.double() * coefficient).abs().sum()
                        for value, coefficient in zip(values, self.coefficients, strict=True)
                    ]
                    result = torch.stack(terms).sum().to(values[0].dtype)
        else:
            # Custom penalties still receive the original flattened union.
            values = group_values(self.groups)
            result = torch.stack(
                [self.penalty(v) * a for v, a in zip(values, self.coefficients, strict=True)]
            ).sum()
        if not torch.isfinite(result):
            raise ValueError("Sparse penalty is nonfinite")
        return result

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return the unweighted penalty for one flattened group."""
        return stable_norm(values)


class GroupSquaredL2(GroupLasso):
    """Return half of sum(a_g * ||W_g||_2**2); see GroupLasso bindings."""

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return half the squared norm, whose gradient is the group itself."""
        return squared_norm(values)


class ScaleL1(GroupLasso):
    """Penalize explicitly named scale parameters, deduplicating aliases.

    Args:
        graph: Fresh graph tracking parameter identity and shape.
        parameters: Paths or TensorRefs of one-dimensional BN/gate parameters.

    No normalization/gate selection policy is inferred from the model.
    """

    def __init__(self, graph: DependencyGraph, parameters: tuple[str | TensorRef, ...]) -> None:
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

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return L1 with autograd's zero subgradient at zero."""
        return values.abs().sum()
