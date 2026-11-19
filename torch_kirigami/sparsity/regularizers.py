"""Scalar sparse regularizers over live parameter groups, differentiated by autograd."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import TensorRef
from ..pruning.groups import ParameterGroup
from .values import (
    ReductionPlan,
    batched_slices,
    gather_group,
    group_bindings,
    group_values,
    half_squared,
    needs_direct_norm,
    parameter_dtype,
    reduction_plan,
    segment_norms,
    slice_norm,
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


@torch.no_grad()
def _prepare_groups(groups: tuple[ParameterGroup, ...], *, batch: bool) -> ReductionPlan | None:
    """Check the initial bindings and values and optionally prepare batched indices."""
    bindings, dtype = group_bindings(groups)
    group_values(tuple(group.selections for group in groups), bindings, dtype)
    return reduction_plan(groups, bindings) if batch else None


def _live_values(groups: tuple[ParameterGroup, ...]) -> tuple[torch.Tensor, ...]:
    """Read validated values for a weighted or custom reduction."""
    bindings, dtype = group_bindings(groups)
    return group_values(tuple(group.selections for group in groups), bindings, dtype)


def _custom_penalty(
    penalty: Callable[[torch.Tensor], torch.Tensor],
    values: tuple[torch.Tensor, ...],
    coefficients: tuple[float, ...],
) -> torch.Tensor:
    """Apply a subclass penalty to each group before weighting and summing it."""
    return torch.stack([penalty(v) * a for v, a in zip(values, coefficients, strict=True)]).sum()


def _finite(values: torch.Tensor, message: str = "Sparse penalty is nonfinite") -> torch.Tensor:
    """Validate the result or intermediate reductions together on their device."""
    if not torch.isfinite(values).all():
        raise ValueError(message)
    return values


def _batched_lasso(groups: tuple[ParameterGroup, ...], plan: ReductionPlan) -> torch.Tensor:
    """Combine per-slice L2 norms into one norm per group and sum the results."""
    bindings, dtype = group_bindings(groups, refs=plan.refs)
    contributions = [
        slice_norm(tensor, dims).index_select(0, indices)
        for tensor, dims, indices in batched_slices(plan, bindings, dtype)
    ]
    contributions.extend(
        stable_norm(gather_group(groups[i].selections, bindings, dtype)).reshape(1)
        for i in plan.fallback
    )
    norms = _finite(
        torch.cat(contributions), "Sparse operations require finite selected values and penalties"
    )
    if needs_direct_norm(norms, plan.owners, len(groups)):
        # Rounded subnormal row norms can lose their original relative magnitudes.
        return torch.stack(
            [stable_norm(gather_group(group.selections, bindings, dtype)) for group in groups]
        ).sum()
    return segment_norms(norms, plan.owners, len(groups)).sum()


def _batched_squared_l2(groups: tuple[ParameterGroup, ...], plan: ReductionPlan) -> torch.Tensor:
    """Reduce half squares of selected slices before summing across groups."""
    bindings, dtype = group_bindings(groups, refs=plan.refs)
    tiny = torch.finfo(dtype).tiny ** 0.5
    contributions, tiny_squares = [], []

    def track_underflow(values: torch.Tensor) -> None:
        """Record nonzero magnitudes whose individual squares may round away."""
        magnitude = values.detach().abs()
        tiny_squares.append(((magnitude != 0) & (magnitude < tiny)).any())

    for tensor, dims, indices in batched_slices(plan, bindings, dtype):
        track_underflow(tensor)
        values = half_squared(tensor)
        totals = values.sum(dim=dims) if dims else values
        contributions.append(totals.index_select(0, indices))
    for i in plan.fallback:
        values = gather_group(groups[i].selections, bindings, dtype)
        track_underflow(values)
        contributions.append(squared_norm(values).reshape(1))
    totals = _finite(
        torch.cat(contributions), "Sparse operations require finite selected values and penalties"
    )
    if torch.stack(tiny_squares).any():
        # The total can be representable even when individual squares are zero.
        return squared_norm(
            torch.cat([gather_group(g.selections, bindings, dtype) for g in groups])
        )
    return totals.sum()


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
        self._reduction_plan = _prepare_groups(
            self.groups, batch=type(self) is GroupLasso and all(a == 1 for a in self.coefficients)
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
            result = _batched_lasso(self.groups, self._reduction_plan)
        else:
            values = _live_values(self.groups)
            if type(self) is GroupLasso:
                # FP64 preserves the full range of finite Python coefficients.
                result = (
                    torch.stack(
                        [
                            stable_norm(v.double(), coefficient=a)
                            for v, a in zip(values, self.coefficients, strict=True)
                        ]
                    )
                    .sum()
                    .to(values[0].dtype)
                )
            else:
                result = _custom_penalty(self.penalty, values, self.coefficients)
        return _finite(result)

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


class GroupSquaredL2:
    """Compute half the weighted sum of parameter-group squared L2 norms.

    Calling the object returns `0.5 * sum(a_g * ||W_g||_2**2)`. Here `W_g`
    contains one group's current selected parameter values, and `a_g` is its
    coefficient. Each supplied group contributes one term, including repeated
    groups and overlaps between groups. Coordinates within a group count once.
    There is no group-size normalization. Use
    `torch_kirigami.pruning.unique_groups` before construction for explicit
    deduplication, then assign coefficients to the returned groups.

    Calls read current parameter values and support first-order autograd. The
    caller multiplies the result by the overall loss strength and owns training.

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
        ValueError: The groups or coefficients violate the requirements above.
    """

    def __init__(
        self,
        groups: tuple[ParameterGroup, ...],
        *,
        coefficients: tuple[int | float, ...] | None = None,
    ) -> None:
        self.groups = tuple(groups)
        self.coefficients = coefficients_for(len(self.groups), coefficients)
        self._reduction_plan = _prepare_groups(
            self.groups,
            batch=type(self) is GroupSquaredL2 and all(a == 1 for a in self.coefficients),
        )

    def __call__(self) -> torch.Tensor:
        """Evaluate the weighted loss on live parameters without running the model.

        Returns:
            A scalar in float64 if any selected parameter is float64, or float32 otherwise.

        Raises:
            ValueError: Selected values or the resulting loss are nonfinite.
        """
        if self._reduction_plan is not None and all(a == 1 for a in self.coefficients):
            result = _batched_squared_l2(self.groups, self._reduction_plan)
        else:
            values = _live_values(self.groups)
            if type(self) is GroupSquaredL2:
                result = squared_norm(*values, coefficients=self.coefficients)
            else:
                result = _custom_penalty(self.penalty, values, self.coefficients)
        return _finite(result)

    def penalty(self, values: torch.Tensor) -> torch.Tensor:
        """Return half the squared L2 norm of one group.

        Args:
            values: Current selected values of one group, flattened to one dimension.

        Returns:
            A scalar tensor whose gradient with respect to `values` is `values`.
        """
        return squared_norm(values)


class ScaleL1:
    """Compute an L1 loss on explicitly named one-dimensional parameters.

    Typical inputs are BatchNorm scale vectors, such as `bn.weight`. Calling
    the object returns the sum of absolute values across the supplied parameters.
    The caller multiplies this scalar by a loss strength and adds it to the task
    loss to encourage small scale values during training. The regularizer
    neither selects channels nor removes parameters.

    Each distinct parameter contributes once, even when paths repeat or alias
    the same object. This class does not propagate dependencies to other weights.
    The caller identifies scale parameters. Any one-dimensional parameter is
    accepted, regardless of its name or role in the model.

    Args:
        graph: Fresh graph tracking parameter identity and shape.
        parameters: Nonempty tuple of parameter paths or `TensorRef` objects
            from this graph. Parameters must be one-dimensional, dense, real
            floating-point tensors on one device, with finite values.

    Attributes:
        graph: Graph that supplies the live parameter bindings.
        parameters: Distinct parameter references, sorted by their paths.

    Raises:
        KeyError: A path is not a registered parameter of `graph`.
        ValueError: `parameters` is empty, or a parameter violates the
            requirements above.
    """

    def __init__(self, graph: DependencyGraph, parameters: tuple[str | TensorRef, ...]) -> None:
        self.graph = graph
        refs = {graph.parameter(p) if isinstance(p, str) else p for p in parameters}
        self.parameters = tuple(sorted(refs, key=lambda ref: ref.paths))
        if not self.parameters or any(
            ref.kind != "parameter" or len(ref.shape) != 1 for ref in self.parameters
        ):
            raise ValueError("ScaleL1 requires one-dimensional scale parameters")
        with torch.no_grad():
            _finite(self._values(), "Sparse operations require finite selected values")

    def _values(self) -> torch.Tensor:
        """Read a single vector of live scales in the common accumulation dtype."""
        bindings = dict(self.graph.tensor_bindings())
        try:
            values = [bindings[ref] for ref in self.parameters]
        except KeyError as error:
            raise ValueError("ScaleL1 requires parameter references from its graph") from error
        dtype = parameter_dtype(values)
        return torch.cat([value.to(dtype).abs() for value in values])

    def __call__(self) -> torch.Tensor:
        """Sum the absolute values of the live scales.

        Returns:
            A scalar loss with a zero subgradient at zero.

        Raises:
            ValueError: A scale value or the resulting loss is nonfinite.
        """
        return _finite(self._values().sum())
