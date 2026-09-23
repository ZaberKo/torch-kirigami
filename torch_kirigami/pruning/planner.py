"""Read-only planning services shared by built-in and model-specific strategies."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import cast

import torch

from ..contracts import Constraint, Impact
from ..graph import DependencyGraph
from ..operation import OperationContext
from ..regions import Region
from ..selection import AxisRef, Selection, TensorRef
from .rewrite import compile_recipes
from .types import (
    AttributeRecipe,
    Candidate,
    ChannelCount,
    ChannelRatio,
    Metric,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    SelectionReport,
    StrategyResult,
    TensorRecipe,
    channel_targets,
)

_IMPACT_CACHE_SIZE = 32


class PlanningContext:
    """Read-only model access, candidates, joint analysis, scoring, and budget checks.

    Metrics own their statistics and are never cached. Impact queries use a bounded
    32-entry cache; four recent verified recipe sets and 32 configuration checks
    may also be reused. Caches retain no tensor data; version changes invalidate
    compilation caches, and callback boundaries still validate state.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        operations: tuple[OperationContext, ...],
        candidates: tuple[Candidate, ...],
        budget: ChannelCount | ChannelRatio | ParameterBudget,
        channel_axes: tuple[AxisRef, ...],
        constraints: tuple[Constraint, ...],
    ) -> None:
        self._graph, self._operations = graph, tuple(operations)
        self._candidates = tuple(candidates)
        self._budget, self._channel_axes = budget, tuple(dict.fromkeys(channel_axes))
        if isinstance(budget, ChannelCount) and budget.channel_axes != self.channel_axes:
            raise ValueError("ChannelCount axes must match the candidate space in order")
        self._constraints = tuple(constraints)
        self._widths = tuple(a.tensor.shape[a.dim] for a in self.channel_axes)
        self._targets = (
            () if isinstance(budget, ParameterBudget) else channel_targets(budget, self.widths)
        )
        self._parameter_count = sum(
            math.prod(ref.shape) for ref, _ in graph.tensor_bindings() if ref.kind == "parameter"
        )
        self._trials = 0
        self._cache: OrderedDict[tuple[tuple[TensorRef, tuple[Region, ...]], ...], Impact] = (
            OrderedDict()
        )
        self._compiled: OrderedDict[
            int,
            tuple[
                Impact,
                tuple[tuple[TensorRecipe, ...], tuple[AttributeRecipe, ...], tuple[str, ...]],
            ],
        ] = OrderedDict()
        self._attribute_checks: OrderedDict[tuple[tuple[str, object], ...], str | None] = (
            OrderedDict()
        )
        self._compile_versions: tuple[tuple[int, int | None], ...] | None = None

    @property
    def graph(self) -> DependencyGraph:
        """Return the source dependency snapshot."""
        return self._graph

    @property
    def operations(self) -> tuple[OperationContext, ...]:
        """Return captured calls used for recipe compilation."""
        return self._operations

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """Return the fixed candidate universe."""
        return self._candidates

    @property
    def budget(self) -> ChannelCount | ChannelRatio | ParameterBudget:
        """Return the caller's immutable budget."""
        return self._budget

    @property
    def channel_axes(self) -> tuple[AxisRef, ...]:
        """Return unique original logical channel axes."""
        return self._channel_axes

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return fixed analysis premises; callbacks must not mutate constraints."""
        return self._constraints

    @property
    def widths(self) -> tuple[int, ...]:
        """Return logical axis widths, used only by channel budgets."""
        return self._widths

    @property
    def targets(self) -> tuple[int, ...]:
        """Return channel removal caps; empty for a parameter budget."""
        return self._targets

    @property
    def trials(self) -> int:
        """Return joint attempts made through `attempt`, including cache hits."""
        return self._trials

    def attempt(self, remove: tuple[Selection, ...]) -> Impact:
        """Count and analyze a proposed joint request without mutating the model.

        Strategies own their attempt limit. Ordinary `impact` calls used for
        scoring and inspection do not count as selection trials.
        """
        self._trials += 1
        return self.impact(remove)

    def impact(self, remove: tuple[Selection, ...]) -> Impact:
        """Analyze joint original-coordinate seeds without executing the model."""
        remove = tuple(remove)
        for selection in remove:
            self.graph.metadata(selection.tensor)
        key = tuple((s.tensor, s.regions) for s in remove)
        if key not in self._cache:
            self._cache[key] = self.graph.propagate(remove=remove, constraints=self.constraints)
            if len(self._cache) > _IMPACT_CACHE_SIZE:
                self._cache.popitem(last=False)
        else:
            self.graph.validate()  # propagate performs the entry check on cache misses.
            self._cache.move_to_end(key)
        return self._cache[key]

    def require_complete(self, impact: Impact) -> None:
        """Reject incomplete influence ranges while allowing repairable constraints."""
        self.graph.validate_impact(impact)
        incomplete = [d for d in impact.diagnostics if not d.complete]
        if incomplete:
            raise PlanningError("Incomplete scoring influence: " + "; ".join(map(str, incomplete)))

    def score(
        self,
        metric: Metric,
        candidate_batch: tuple[Candidate, ...],
        *,
        selected: Impact | None = None,
    ) -> tuple[float, ...]:
        """Score additions to a committed closure, or to the empty request.

        Temporary multi-selection candidates are allowed. Influence is checked
        jointly: an isolated candidate can miss effects triggered by combining
        it with the accepted requests. Feasibility is checked by the strategy.
        """
        batch = tuple(candidate_batch)
        selected = self.impact(()) if selected is None else selected
        self.require_complete(selected)
        for candidate in batch:
            self.require_complete(self.impact((*selected.requested, *candidate.remove)))
        return self._score_complete(metric, batch, selected=selected)

    def _score_complete(
        self, metric: Metric, batch: tuple[Candidate, ...], *, selected: Impact
    ) -> tuple[float, ...]:
        """Score a batch whose influence was already checked by this context.

        Both greedy strategies check completeness while collecting axis summaries.
        Repeating that scan here would evict the bounded impact cache for large
        custom batches. Keep the public score entry fully checked for temporary
        candidates, and retain state and result validation at callback boundaries.
        """
        batch = tuple(batch)
        self.require_complete(selected)
        self.graph.validate()
        if not callable(getattr(metric, "score", None)):
            raise TypeError("Metric must implement score(context, candidates, *, selected)")
        values = metric.score(self, batch, selected=selected)
        self.graph.validate()
        if isinstance(values, torch.Tensor):
            if values.ndim != 1 or values.is_complex():
                raise PlanningError("Metric must return a real one-dimensional score batch")
            values = values.detach().cpu().tolist()
        try:
            values = tuple(float(v) for v in values)
        except (ValueError, TypeError) as error:
            raise PlanningError(
                "Metric must return an aligned one-dimensional score batch"
            ) from error
        if len(values) != len(batch) or not all(math.isfinite(v) for v in values):
            raise PlanningError("Metric returned nonfinite scores or an incorrect batch length")
        return values

    def counts(self, impact: Impact) -> tuple[int, ...]:
        """Measure actual full-axis removals, counting each logical axis once."""
        self.graph.validate_impact(impact)
        return tuple(
            len(impact.selection(a.tensor).fully_selected_indices(a.dim)) for a in self.channel_axes
        )

    def admissible(self, impact: Impact) -> bool:
        """Check intermediate removal caps, allowing progress toward a resource target.

        Structural/execution validity is checked separately by compile(). A
        parameter target cannot reject intermediate requests merely because they
        still leave too many parameters.
        """
        if isinstance(self.budget, ParameterBudget):
            self.graph.validate_impact(impact)
            return True
        return _within_targets(self, self.counts(impact))

    def parameter_count(self, impact: Impact) -> int:
        """Count final unique Parameter elements using verified joint recipes."""
        recipes, _, _ = self.compile(impact)
        return self._parameter_count - sum(
            math.prod(r.tensor.shape) - math.prod(r.shape)
            for r in recipes
            if r.tensor.kind == "parameter"
        )

    def within_budget(self, impact: Impact) -> bool:
        """Check the final budget, including an absolute parameter target if given."""
        if isinstance(self.budget, ParameterBudget):
            return self.parameter_count(impact) <= self.budget.max_params
        return self.admissible(impact)

    def require_budget(self, impact: Impact, *, result: StrategyResult | None = None) -> None:
        """Reject an unmet final target with actual counts and bounded-search diagnostics."""
        if self.within_budget(impact):
            return
        if isinstance(self.budget, ParameterBudget):
            detail = (
                "; ".join(f"{k}: {reason}" for k, reason in result.exclusions[-3:])
                if result
                else ""
            )
            raise PlanningError(
                f"Parameter target not reached: {self.parameter_count(impact)} remain, "
                f"max_params={self.budget.max_params}; {self.trials} joint trials"
                + (
                    "; strategy trial limit reached"
                    if result and result.stop_reason == "trial_limit"
                    else ""
                )
                + ". No plan was produced or applied. This is not a proof of infeasibility."
                + (f" Last exclusions: {detail}" if detail else "")
            )
        raise PlanningError(_budget_reason(self, self.counts(impact)))

    def compile(
        self, impact: Impact
    ) -> tuple[tuple[TensorRecipe, ...], tuple[AttributeRecipe, ...], tuple[str, ...]]:
        """Return (tensor recipes, attribute recipes, notes), without new weights."""
        self.graph.validate()
        self.graph.validate_impact(impact)
        try:
            versions = tuple((id(t), t._version) for _, t in self.graph.tensor_bindings())
        except RuntimeError:
            versions = None  # Inference tensors cannot safely key a mutation-aware cache.
        if versions is None or versions != self._compile_versions:
            self._compiled.clear()
            self._attribute_checks.clear()
            self._compile_versions = versions
        # Cache a specific analysis result, not merely seeds/status: extra
        # constraints can produce a different closure or execution requirements.
        key = id(impact)
        if key not in self._compiled:
            self._compiled[key] = (
                impact,
                compile_recipes(
                    self.graph,
                    self.operations,
                    impact,
                    attribute_checks=self._attribute_checks if versions is not None else None,
                ),
            )
            if len(self._compiled) > 4:
                self._compiled.popitem(last=False)
        else:
            self._compiled.move_to_end(key)
        return self._compiled[key][1]

    def report(
        self,
        impact: Impact,
        result: StrategyResult,
        *,
        exclusions: tuple[tuple[str, str], ...] = (),
    ) -> ParameterReport | SelectionReport:
        """Freeze the measured budget and strategy diagnostics."""
        if isinstance(self.budget, ParameterBudget):
            return ParameterReport(
                self._parameter_count,
                self.parameter_count(impact),
                self.budget.max_params,
                self.trials,
                result.stop_reason == "trial_limit",
                (*exclusions, *result.exclusions),
            )
        return SelectionReport(
            self.channel_axes,
            self.widths,
            self.counts(impact),
            self.targets,
            self.budget.scope,
            self.trials,
            result.stop_reason == "trial_limit",
            (*exclusions, *result.exclusions),
        )


def _within_targets(context: PlanningContext, counts: tuple[int, ...]) -> bool:
    """Apply the same caps to exact counts and proven lower bounds."""
    if isinstance(context.budget, ParameterBudget):
        return True
    return (
        all(a <= b for a, b in zip(counts, context.targets, strict=True))
        if context.budget.scope == "local"
        else sum(counts) <= context.targets[0]
    )


def _budget_reason(
    context: PlanningContext, counts: tuple[int, ...], *, lower_bound: bool = False
) -> str:
    """Describe an exceeded cap, distinguishing exact counts from lower bounds."""
    qualifier = "at least " if lower_bound else ""
    budget = cast(ChannelCount | ChannelRatio, context.budget)
    if budget.scope == "global":
        return (
            f"Joint request exceeds the global channel budget: "
            f"{qualifier}{sum(counts)} removals > {context.targets[0]} allowed"
        )
    details = (
        f"{axis.tensor.paths[0] if axis.tensor.paths else axis.tensor.id} "
        f"axis {axis.dim}: {qualifier}{count} removals > {cap} allowed"
        for axis, count, cap in zip(context.channel_axes, counts, context.targets, strict=True)
        if count > cap
    )
    return "Joint request exceeds the local channel budget: " + "; ".join(details)
