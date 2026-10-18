"""Scalar sparse penalties with a single autograd-based differentiation path."""

from __future__ import annotations

import math

import torch

from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import Selection, TensorRef, full_region
from ..pruning.groups import ParameterGroup
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
    """Compute a weighted sum of parameter-group L2 norms.

    Calling the object returns `sum(a_g * ||W_g||_2)`. Here `W_g` contains
    the current values selected by one `ParameterGroup`, and `a_g` is its
    coefficient. Repeated coordinates count once within each group. Repeated
    groups and overlaps between groups contribute to each group's loss term.

    To count equivalent groups once, call
    `torch_kirigami.pruning.unique_groups(groups)` before construction, then
    assign coefficients to the returned groups. No deduplication, averaging,
    or group-size normalization occurs in this regularizer.

    Rebuild groups after structural changes. Calls read current parameters
    without changing them or their gradients. Frozen parameters still contribute
    to the loss. The object does not retain autograd graphs between calls.
    Built-in L2 penalties support first-order autograd, not double backward.
    Multiply the returned scalar by the overall loss strength in the training loop.

    Args:
        groups: Nonempty tuple of `torch_kirigami.pruning.ParameterGroup` objects
            from one graph. Selected parameters must be dense, real floating-point
            tensors on one device, with finite selected values.
        coefficients: One fixed, finite, nonnegative Python number per group,
            in input order. `None` assigns one to every group.

    Attributes:
        groups: Supplied groups in input order, including repeated groups.
        coefficients: Validated scalar weights in the same order.

    Examples:
        Assume `group` selects parameter values `[3, 4]`. Its L2 norm is 5.
        Passing it twice adds two loss terms. Explicit deduplication keeps one::

            from torch_kirigami.pruning import unique_groups
            from torch_kirigami.sparsity import GroupLasso

            repeated = GroupLasso((group, group))
            repeated()  # 10 = 5 + 5

            groups = unique_groups((group, group))
            once = GroupLasso(groups)
            once()  # 5

            weighted = GroupLasso(groups, coefficients=(2.0,))
            weighted()  # 10 = 2 * 5

        The deduplicated tuple has one group, so it needs only one coefficient.
        `unique_groups` does not sum coefficients from the original tuple.
    """

    def __init__(
        self,
        groups: tuple[ParameterGroup, ...],
        *,
        coefficients: tuple[int | float, ...] | None = None,
    ) -> None:
        self.groups = tuple(groups)
        self.coefficients = coefficients_for(len(self.groups), coefficients)
        self._reduction_plan = reduction_plan(self.groups)
        with torch.no_grad():
            group_values(self.groups)

    def __call__(self) -> torch.Tensor:
        """Evaluate the loss on current parameters without running the model.

        Returns:
            A scalar on the selected parameters' device, connected to autograd
            where parameters require gradients. The dtype is float64 if any
            selected parameter is float64, otherwise float32.

        Raises:
            ValueError: Selected values or the resulting loss are nonfinite,
                or the groups violate the device or dtype requirements.
        """
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
    """Compute half the weighted sum of parameter-group squared L2 norms.

    Calling the object returns `0.5 * sum(a_g * ||W_g||_2**2)`. Here `W_g`
    contains one group's current selected parameter values, and `a_g` is its
    coefficient. Each supplied group contributes one term, including repeated
    groups and overlaps between groups. Coordinates within a group count once.
    There is no group-size normalization. Use
    `torch_kirigami.pruning.unique_groups` before construction for explicit
    deduplication, then assign coefficients to the returned groups.

    The live bindings, dtype, and first-order autograd contract are inherited
    from `GroupLasso`. Multiply the returned scalar by the overall loss strength.

    Args:
        groups: Nonempty tuple of `torch_kirigami.pruning.ParameterGroup` objects
            from one graph. Selected parameters must be dense, real floating-point
            tensors on one device, with finite selected values.
        coefficients: One fixed, finite, nonnegative Python number per group,
            in input order. `None` assigns one to every group.
    """

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return half the squared norm, whose gradient is the group itself."""
        return squared_norm(values)


class ScaleL1(GroupLasso):
    """Compute an L1 loss on explicitly named one-dimensional parameters.

    Typical inputs are BatchNorm scale vectors, such as `bn.weight`. Calling
    the object returns the sum of absolute values across the supplied parameters.
    Add this scalar to the task loss to encourage small scale values during
    training. The regularizer neither selects channels nor removes parameters.

    Each distinct parameter contributes once, even when paths repeat or alias
    the same object. This class does not propagate dependencies to other weights.
    It reuses `GroupLasso` binding and evaluation code but computes L1, not L2.
    The caller identifies scale parameters. Any one-dimensional parameter is
    accepted, regardless of its name or role in the model.

    Args:
        graph: Fresh graph tracking parameter identity and shape.
        parameters: Nonempty tuple of parameter paths or `TensorRef` objects
            from this graph. Parameters must be one-dimensional, dense, real
            floating-point tensors on one device, with finite values.
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
