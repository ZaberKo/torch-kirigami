"""Read-only planning services shared by built-in and model-specific strategies."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ..contracts import Constraint, Impact
from ..graph import DependencyGraph
from ..operation import OperationContext
from ..regions import Region
from ..selection import AxisRef, Selection, TensorRef
from .budget import remaining_parameters, resolve_budget
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
)

_IMPACT_CACHE_SIZE = 32


class PlanningContext:
    """Read-only model access, candidates, joint analysis, scoring, and budget checks.

    Metrics own their statistics and are never cached. Impact queries use a bounded
    32-entry cache; four recent verified recipe sets and 32 configuration checks
    may also be reused. Pure native metadata outputs retain at most one fact tree
    per captured call. Caches retain no tensor data; version changes invalidate
    compilation caches, and callback boundaries still validate state.

    Args:
        graph: Fresh dependency snapshot supplying captured calls and bindings.
        candidates: Explicit candidate universe; no discovery is performed.
        budget: Upper bounds on final parameter count or channel widths.
        channel_axes: Logical axes used to count channel removals.
        constraints: Fixed structural premises for this planning round.
    """

    def __init__(
        self,
        graph: DependencyGraph,
        candidates: tuple[Candidate, ...],
        budget: ChannelCount | ChannelRatio | ParameterBudget,
        channel_axes: tuple[AxisRef, ...],
        constraints: tuple[Constraint, ...],
    ) -> None:
        graph.validate()
        self._graph, self._operations = graph, graph.operations()
        self._candidates = tuple(candidates)
        self._budget, self._channel_axes = budget, tuple(dict.fromkeys(channel_axes))
        for axis in self.channel_axes:
            graph.metadata(axis.tensor)
        self._constraints = tuple(constraints)
        self._widths = tuple(a.tensor.shape[a.dim] for a in self.channel_axes)
        self._parameter_count = sum(
            math.prod(ref.shape) for ref, _ in graph.tensor_bindings() if ref.kind == "parameter"
        )
        self._target = resolve_budget(budget, self.channel_axes, self._parameter_count)
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
        # One metadata-only result per captured call, never model tensor storage.
        self._meta_outputs: dict[str, tuple[object, object]] = {}
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
        """Return upper bounds on final sizes, aligned with remaining()."""
        return self._target.limits

    @property
    def budget_axes(self) -> tuple[AxisRef, ...]:
        """Return axes needing channel summaries; parameters use verified recipes."""
        return self._target.axes

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
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact | None = None,
    ) -> tuple[float, ...]:
        """Score each candidate's addition to the accepted removal requests.

        Temporary multi-selection candidates are allowed. Influence is checked
        jointly: an isolated candidate can miss effects triggered by combining
        it with the accepted requests. Feasibility is checked by the strategy.

        Args:
            metric: Batch-independent scorer with caller-owned statistics.
            candidates: Requests to score separately, in the returned score order.
            accepted_impact: Complete dependency analysis of previously accepted
                removal requests. None uses the empty request. This is a planning
                baseline; the original model has not been physically pruned.

        Returns:
            One finite real score per candidate, in the same order.
        """
        candidates = tuple(candidates)
        accepted_impact = self.impact(()) if accepted_impact is None else accepted_impact
        self.require_complete(accepted_impact)
        for candidate in candidates:
            self.require_complete(self.impact((*accepted_impact.requested, *candidate.remove)))
        return self._score_complete(metric, candidates, accepted_impact=accepted_impact)

    def _score_complete(
        self, metric: Metric, candidates: tuple[Candidate, ...], *, accepted_impact: Impact
    ) -> tuple[float, ...]:
        """Score a batch whose influence was already checked by this context.

        Both greedy strategies check completeness while collecting axis summaries.
        Repeating that scan here would evict the bounded impact cache for large
        custom batches. Keep the public score entry fully checked for temporary
        candidates, and retain state and result validation at callback boundaries.
        """
        candidates = tuple(candidates)
        self.require_complete(accepted_impact)
        self.graph.validate()
        if not callable(getattr(metric, "score", None)):
            raise TypeError("Metric must implement score(context, candidates, *, accepted_impact)")
        values = metric.score(self, candidates, accepted_impact=accepted_impact)
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
        if len(values) != len(candidates) or not all(math.isfinite(v) for v in values):
            raise PlanningError("Metric returned nonfinite scores or an incorrect batch length")
        return values

    def remaining(self, impact: Impact) -> tuple[int, ...]:
        """Measure final resource sizes in targets order, without compact weights."""
        self.graph.validate_impact(impact)
        recipes = self.compile(impact)[0] if self._target.resource == "parameters" else ()
        return self._target.remaining(impact, recipes)

    def parameter_count(self, impact: Impact) -> int:
        """Count final unique Parameter elements using verified joint recipes."""
        return remaining_parameters(self._parameter_count, self.compile(impact)[0])

    def within_budget(self, impact: Impact) -> bool:
        """Check all final-size upper bounds; execution validity is checked separately."""
        return self._target.met(self.remaining(impact))

    def deficit(self, impact: Impact) -> int:
        """Measure missing reduction; local excesses cannot cancel each other.

        A complete addition must lower this value before a greedy strategy
        accepts it. Coupled deletion may also shrink an already satisfied axis.
        """
        return self._target.deficit(self.remaining(impact))

    def require_budget(self, impact: Impact, *, result: StrategyResult | None = None) -> None:
        """Reject a missed final target with measured sizes and search diagnostics."""
        self._target.require(self.remaining(impact), self.trials, result)

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
            self._meta_outputs.clear()
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
                    meta_outputs=self._meta_outputs if versions is not None else None,
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
        self.graph.validate_impact(impact)
        return self._target.report(self.remaining(impact), impact, self.trials, result, exclusions)
