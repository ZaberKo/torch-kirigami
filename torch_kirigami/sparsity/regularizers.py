"""Scalar sparse regularizers over live parameter groups, differentiated by autograd."""

from __future__ import annotations

import math

import torch

from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import Selection, TensorRef, full_region
from ..pruning.groups import ParameterGroup
from .values import (
    group_bindings,
    group_penalty,
    group_values,
    reduction_plan,
    squared_norm,
    stable_norm,
)


def coefficients_for(count: int, coefficients: tuple[int | float, ...] | None) -> tuple[float, ...]:
    """Validate per-group coefficients and convert them to Python floats.

    Args:
        count: Number of groups.
        coefficients: One finite, nonnegative Python number per group, or
            `None` for a coefficient of one per group. Tensors are rejected so
            that coefficients never join the autograd graph.

    Returns:
        The coefficients as floats, in input order.

    Raises:
        ValueError: The count does not match, or a coefficient is a boolean,
            a tensor, nonfinite, or negative.
    """
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

    The regularizer only returns a scalar loss. The caller multiplies it by the
    overall loss strength, adds it to the task loss, and owns the optimizer,
    loss scaling, and training loop.

    Args:
        groups: Nonempty tuple of `torch_kirigami.pruning.ParameterGroup` objects
            from one graph. Selected parameters must be dense, real floating-point
            tensors on one device, with finite selected values.
        coefficients: One fixed, finite, nonnegative Python number per group,
            in input order. `None` assigns one to every group.

    Attributes:
        groups: Supplied groups in input order, including repeated groups.
        coefficients: Validated coefficients as floats, in the same order.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: `groups` is empty or spans several graphs or devices, a
            selected parameter is not a dense real floating-point tensor, a
            selected value is nonfinite, or the coefficients are invalid.

    Examples:
        Assume `group` selects parameter values `[3, 4]`. Its L2 norm is 5.
        Passing it twice adds two loss terms. Explicit deduplication keeps one:

        ```python
        from torch_kirigami.pruning import unique_groups
        from torch_kirigami.sparsity import GroupLasso

        GroupLasso((group, group))()  # 10 = 5 + 5

        groups = unique_groups((group, group))
        GroupLasso(groups)()  # 5
        GroupLasso(groups, coefficients=(2.0,))()  # 10 = 2 * 5
        ```

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
        with torch.no_grad():
            bindings, dtype = group_bindings(self.groups)
            group_values(tuple(g.selections for g in self.groups), bindings, dtype)
            self._reduction_plan = (
                reduction_plan(self.groups, bindings)
                if type(self) in _BUILTIN_PENALTIES and all(a == 1 for a in self.coefficients)
                else None
            )

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
        # Exact types keep the optimized path from bypassing a subclass penalty().
        if self._reduction_plan is not None and all(a == 1 for a in self.coefficients):
            result = group_penalty(
                self.groups, _BUILTIN_PENALTIES[type(self)], self._reduction_plan
            )
        else:
            kind = _BUILTIN_PENALTIES.get(type(self))
            bindings, dtype = group_bindings(self.groups)
            values = group_values(tuple(g.selections for g in self.groups), bindings, dtype)
            # Built-in weighted reductions apply coefficients before a square
            # can overflow. FP64 preserves finite Python coefficients.
            if kind is None:
                result = torch.stack(
                    [self.penalty(v) * a for v, a in zip(values, self.coefficients, strict=True)]
                ).sum()
            elif kind == "squared_l2":
                result = squared_norm(*values, coefficients=self.coefficients)
            else:
                terms = [
                    stable_norm(value.double(), coefficient=coefficient)
                    if kind == "l2"
                    else (value.double() * coefficient).abs().sum()
                    for value, coefficient in zip(values, self.coefficients, strict=True)
                ]
                result = torch.stack(terms).sum().to(values[0].dtype)
        if not torch.isfinite(result):
            raise ValueError("Sparse penalty is nonfinite")
        return result

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return the unweighted penalty for one group.

        Subclasses override this method to define a custom penalty. The exact
        built-in classes use an equivalent batched implementation instead.

        Args:
            values: Current selected values of one group, flattened to one dimension.

        Returns:
            A scalar tensor. This implementation returns the L2 norm.
        """
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
    from `GroupLasso`. Like `GroupLasso`, it only returns a scalar loss; the
    caller multiplies it by the overall loss strength and owns training.

    Args:
        groups: Nonempty tuple of `torch_kirigami.pruning.ParameterGroup` objects
            from one graph. Selected parameters must be dense, real floating-point
            tensors on one device, with finite selected values.
        coefficients: One fixed, finite, nonnegative Python number per group,
            in input order. `None` assigns one to every group.

    Raises:
        TypeError: An item of `groups` is not a `ParameterGroup`.
        ValueError: The groups or coefficients violate the requirements above.
    """

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return half the squared L2 norm of one group.

        Args:
            values: Current selected values of one group, flattened to one dimension.

        Returns:
            A scalar tensor whose gradient with respect to `values` is `values`.
        """
        return squared_norm(values)


class ScaleL1(GroupLasso):
    """Compute an L1 loss on explicitly named one-dimensional parameters.

    Typical inputs are BatchNorm scale vectors, such as `bn.weight`. Calling
    the object returns the sum of absolute values across the supplied parameters.
    The caller multiplies this scalar by a loss strength and adds it to the task
    loss to encourage small scale values during training. The regularizer
    neither selects channels nor removes parameters.

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

    Raises:
        KeyError: A path is not a registered parameter of `graph`.
        ValueError: `parameters` is empty, or a parameter violates the
            requirements above.
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
        """Return the L1 norm of one group.

        Args:
            values: Current selected values, flattened to one dimension.

        Returns:
            A scalar tensor. Autograd uses a zero subgradient at zero.
        """
        return values.abs().sum()


_BUILTIN_PENALTIES = {GroupLasso: "l2", GroupSquaredL2: "squared_l2", ScaleL1: "l1"}
