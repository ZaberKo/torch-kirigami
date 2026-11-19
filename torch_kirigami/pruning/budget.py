"""Resolve and measure final-size targets without search or model mutation."""

from dataclasses import dataclass
from math import prod
from typing import Literal

from ..dep_graph.contracts import Impact
from ..dep_graph.selection import AxisRef
from ..operator_recipes import TensorRecipe
from .types import (
    ChannelCount,
    ChannelRatio,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    SelectionReport,
    StrategyResult,
    _decimal_ratio,
)


def channel_targets(budget: ChannelRatio, widths: tuple[int, ...]) -> tuple[int, ...]:
    """Convert a channel ratio into final-width limits with exact decimal rounding."""
    ratio = _decimal_ratio(budget.ratio)
    counted_widths = widths if budget.scope == "local" else (sum(widths),)
    return tuple(
        width * (ratio.denominator - ratio.numerator) // ratio.denominator
        for width in counted_widths
    )


def remaining_parameters(before: int, recipes: tuple[TensorRecipe, ...]) -> int:
    """Count unique parameter elements after verified joint tensor recipes."""
    return before - sum(
        prod(r.tensor.shape) - prod(r.shape) for r in recipes if r.tensor.kind == "parameter"
    )


@dataclass(frozen=True)
class _BudgetTarget:
    """Resolved resource, original sizes and upper bounds on final sizes.

    This record owns measurement and reporting, never dependency queries or
    recipe compilation. The caller validates Impact ownership and any recipes.

    Attributes:
        resource: Whether limits apply to parameters or channels.
        axes: Logical channel axes, empty for a parameter budget.
        widths: Original parameter count or original channel widths.
        limits: Upper bounds on final resource sizes.
        scope: Local per-axis limits or one global sum limit.
    """

    resource: Literal["parameters", "channels"]
    axes: tuple[AxisRef, ...]
    widths: tuple[int, ...]
    limits: tuple[int, ...]
    scope: Literal["local", "global"]

    def measure(self, impact: Impact, recipes: tuple[TensorRecipe, ...] = ()) -> tuple[int, ...]:
        """Measure the final parameter count or each logical channel width."""
        if self.resource == "parameters":
            return (remaining_parameters(self.widths[0], recipes),)
        return self.channel_widths(impact)

    def channel_widths(self, impact: Impact) -> tuple[int, ...]:
        """Measure final channel widths from the complete positions removed by the impact."""
        return tuple(
            width - len(impact.selection(a.tensor).fully_selected_indices(a.dim))
            for a, width in zip(self.axes, self.widths, strict=True)
        )

    def remaining(self, sizes: tuple[int, ...]) -> tuple[int, ...]:
        """Express measured sizes in the same order and scope as the limits."""
        return sizes if self.scope == "local" else (sum(sizes),)

    def met(self, remaining: tuple[int, ...]) -> bool:
        """Use the same final-size comparison for every resource and scope."""
        return self.deficit(remaining) == 0

    def deficit(self, remaining: tuple[int, ...]) -> int:
        """Count missing reduction without letting one local surplus cancel another."""
        return sum(max(0, n - cap) for n, cap in zip(remaining, self.limits, strict=True))

    def require(
        self, remaining: tuple[int, ...], trials: int, result: StrategyResult | None
    ) -> None:
        """Reject a missed final target; bounded search does not prove infeasibility."""
        if self.met(remaining):
            return
        detail = (
            "; ".join(f"{key}: {reason}" for key, reason in result.exclusions[-3:])
            if result
            else ""
        )
        limited = (
            "; strategy trial limit reached"
            if result and result.stop_reason == "trial_limit"
            else ""
        )
        target = (
            f"Parameter target not reached: {remaining[0]} remain, max_params={self.limits[0]}"
            if self.resource == "parameters"
            else f"Channel target not reached: remaining={remaining}, limits={self.limits}"
        )
        raise PlanningError(
            f"{target}; {trials} joint trials{limited}. "
            "No plan was produced or applied. This is not a proof of infeasibility."
            + (f" Last exclusions: {detail}" if detail else "")
        )

    def report(
        self,
        sizes: tuple[int, ...],
        trials: int,
        result: StrategyResult,
        exclusions: tuple[tuple[str, str], ...],
    ) -> ParameterReport | SelectionReport:
        """Build a static report from measured results and bounded-search diagnostics."""
        limited = result.stop_reason == "trial_limit"
        exclusions = (*exclusions, *result.exclusions)
        if self.resource == "parameters":
            return ParameterReport(
                self.widths[0], sizes[0], self.limits[0], trials, limited, exclusions
            )
        removed = tuple(old - new for old, new in zip(self.widths, sizes, strict=True))
        return SelectionReport(
            self.axes, self.widths, removed, self.limits, self.scope, trials, limited, exclusions
        )


def resolve_budget(
    budget: ChannelCount | ChannelRatio | ParameterBudget,
    axes: tuple[AxisRef, ...],
    parameter_count: int,
    *,
    excluded_axes: tuple[AxisRef, ...] = (),
) -> _BudgetTarget:
    """Freeze one final-size target; the candidate universe never changes its baseline."""
    if isinstance(budget, ParameterBudget):
        return _BudgetTarget("parameters", (), (parameter_count,), (budget.max_params,), "global")
    if not isinstance(budget, (ChannelRatio, ChannelCount)):
        raise TypeError("Expected a parameter or channel budget")
    if isinstance(budget, ChannelCount):
        active = set(axes)
        if tuple(axis for axis in budget.channel_axes if axis in active) != axes or not set(
            budget.channel_axes
        ) <= active.union(excluded_axes):
            raise ValueError(
                "ChannelCount axes must match the active space axes in order and may include excluded axes"
            )
        axes = budget.channel_axes
    widths = tuple(a.tensor.shape[a.dim] for a in axes)
    if isinstance(budget, ChannelRatio):
        limits = channel_targets(budget, widths)
    else:
        limits = budget.max_channels
        if isinstance(limits, int):
            limits = (limits,)
    return _BudgetTarget("channels", axes, widths, limits, budget.scope)
