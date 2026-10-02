"""Resolve and measure final-size targets without search or model mutation."""

from dataclasses import dataclass
from math import prod
from typing import Literal

from ..contracts import Impact
from ..selection import AxisRef
from .types import (
    ChannelCount,
    ChannelRatio,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    SelectionReport,
    StrategyResult,
    TensorRecipe,
    _decimal_ratio,
)


def channel_targets(
    budget: ChannelRatio | ChannelCount, widths: tuple[int, ...]
) -> tuple[int, ...]:
    """Resolve final-width limits with exact decimal rounding."""
    if isinstance(budget, ChannelRatio):
        ratio = _decimal_ratio(budget.ratio)
        counted_widths = widths if budget.scope == "local" else (sum(widths),)
        return tuple(
            width * (ratio.denominator - ratio.numerator) // ratio.denominator
            for width in counted_widths
        )
    if isinstance(budget, ChannelCount):
        if tuple(a.tensor.shape[a.dim] for a in budget.channel_axes) != tuple(widths):
            raise ValueError("Integer budget channel axes do not match planning widths")
        return (budget.max_channels,) if budget.scope == "global" else budget.max_channels
    raise TypeError("Expected ChannelRatio or ChannelCount")


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
    """

    resource: Literal["parameters", "channels"]
    axes: tuple[AxisRef, ...]
    widths: tuple[int, ...]
    limits: tuple[int, ...]
    scope: Literal["local", "global"]

    def remaining(self, impact: Impact, recipes: tuple[TensorRecipe, ...] = ()) -> tuple[int, ...]:
        """Measure final sizes in limit order, using compiled recipes for parameters."""
        if self.resource == "parameters":
            return (remaining_parameters(self.widths[0], recipes),)
        widths = self.channel_widths(impact)
        return widths if self.scope == "local" else (sum(widths),)

    def channel_widths(self, impact: Impact) -> tuple[int, ...]:
        """Measure surviving positions without counting aliases or overlaps twice."""
        return tuple(
            width - len(impact.selection(a.tensor).fully_selected_indices(a.dim))
            for a, width in zip(self.axes, self.widths, strict=True)
        )

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
        remaining: tuple[int, ...],
        impact: Impact,
        trials: int,
        result: StrategyResult,
        exclusions: tuple[tuple[str, str], ...],
    ) -> ParameterReport | SelectionReport:
        """Build a static report from measured results and bounded-search diagnostics."""
        limited = result.stop_reason == "trial_limit"
        exclusions = (*exclusions, *result.exclusions)
        if self.resource == "parameters":
            return ParameterReport(
                self.widths[0], remaining[0], self.limits[0], trials, limited, exclusions
            )
        widths = self.channel_widths(impact)
        removed = tuple(old - new for old, new in zip(self.widths, widths, strict=True))
        return SelectionReport(
            self.axes, self.widths, removed, self.limits, self.scope, trials, limited, exclusions
        )


def resolve_budget(
    budget: ChannelCount | ChannelRatio | ParameterBudget,
    axes: tuple[AxisRef, ...],
    parameter_count: int,
) -> _BudgetTarget:
    """Freeze one final-size target; the candidate universe never changes its baseline."""
    if isinstance(budget, ParameterBudget):
        return _BudgetTarget("parameters", (), (parameter_count,), (budget.max_params,), "global")
    if not isinstance(budget, (ChannelRatio, ChannelCount)):
        raise TypeError("Expected a parameter or channel budget")
    if isinstance(budget, ChannelCount) and budget.channel_axes != axes:
        raise ValueError("ChannelCount axes must match the candidate space in order")
    widths = tuple(a.tensor.shape[a.dim] for a in axes)
    return _BudgetTarget("channels", axes, widths, channel_targets(budget, widths), budget.scope)
