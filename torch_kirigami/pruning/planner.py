"""Read-only planning services shared by built-in and model-specific strategies."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import replace

import torch

from ..dep_graph.contracts import Constraint, Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.operation import Operation
from ..dep_graph.selection import AxisRef, Region, Selection, TensorRef
from ..operator_recipes import AttributeRecipe, TensorRecipe
from ..settings import config
from .budget import remaining_parameters, resolve_budget
from .candidate_checks import CandidateFilter, propagate_request
from .rewrite import compile_recipes
from .types import (
    Candidate,
    ChannelCount,
    ChannelRatio,
    Metric,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    SelectionReport,
    StrategyResult,
)

_ImpactKey = tuple[tuple[TensorRef, tuple[Region, ...]], ...]


class PlanningContext:
    """Read-only planning services passed to `Strategy.select`.

    The context provides the candidates, joint dependency analysis, scoring,
    recipe compilation, and budget measurement for one planning round. It never
    modifies the model; accepted removals exist only as `Impact` analyses.
    Metrics own their statistics, which are never cached here. Global settings
    bound the caches of impacts, verified recipes, and attribute checks, and
    native metadata outputs keep at most one fact tree per captured call. No
    cache keeps tensor data. Tensor version changes invalidate the compilation
    caches, and the graph is revalidated after user callbacks.

    `impact` reports analysis limits in the returned impact. Recipe
    compilation and budget measurement raise `AnalysisLimitError` when a
    selection they derive exceeds `max_parts`; the built-in strategies record
    it as the failure of the request being evaluated and continue.

    Args:
        graph: Fresh dependency graph that supplies captured calls and bindings.
        candidates: Candidates available to the strategy; no discovery is
            performed.
        budget: Cap on the final parameter count or on final channel widths.
        channel_axes: Logical axes used to count channel removals.
        constraints: Structural constraints fixed for this planning round.

    Raises:
        TypeError: `budget` is not a supported budget type.
        ValueError: A `ChannelCount` budget does not match `channel_axes`, or an
            axis belongs to another graph.
        torch_kirigami.StaleGraphError: The model changed after capture.
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
        self._candidate_filter = CandidateFilter(
            graph, self._constraints, query=self.impact, candidate_count=len(self._candidates)
        )
        self._widths = tuple(a.tensor.shape[a.dim] for a in self.channel_axes)
        self._parameter_count = sum(
            math.prod(ref.shape) for ref, _ in graph.tensor_bindings() if ref.kind == "parameter"
        )
        self._target = resolve_budget(budget, self.channel_axes, self._parameter_count)
        self._trials = 0
        # Single slices on equivalent axes use the same solver initialization.
        self._cache: OrderedDict[_ImpactKey, Impact] = OrderedDict()
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
        self._cache_settings: tuple[int, ...] | None = None

    def _sync_cache_settings(self) -> None:
        """Discard retained work when global cache capacities change."""
        settings = (config.impact_cache_size, config.recipe_cache_size, config.attribute_cache_size)
        if settings != self._cache_settings:
            self._cache.clear()
            self._compiled.clear()
            self._attribute_checks.clear()
            self._meta_outputs.clear()
            self._cache_settings = settings

    @property
    def graph(self) -> DependencyGraph:
        """Source dependency graph."""
        return self._graph

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Captured calls used for recipe compilation."""
        return self._operations

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """Candidates available to the strategy, fixed for this round."""
        return self._candidates

    @property
    def budget(self) -> ChannelCount | ChannelRatio | ParameterBudget:
        """Budget supplied by the caller."""
        return self._budget

    @property
    def channel_axes(self) -> tuple[AxisRef, ...]:
        """Unique logical channel axes in original coordinates."""
        return self._channel_axes

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Constraints fixed for this round; callbacks must not mutate them."""
        return self._constraints

    @property
    def widths(self) -> tuple[int, ...]:
        """Original widths of `channel_axes`, used only by channel budgets."""
        return self._widths

    @property
    def targets(self) -> tuple[int, ...]:
        """Caps on final resource sizes, aligned with `remaining()`."""
        return self._target.limits

    @property
    def budget_axes(self) -> tuple[AxisRef, ...]:
        """Logical axes counted by a channel budget; empty for a `ParameterBudget`."""
        return self._target.axes

    @property
    def trials(self) -> int:
        """Number of joint attempts made through `attempt()`, including cache hits."""
        return self._trials

    def attempt(self, remove: tuple[Selection, ...]) -> Impact:
        """Analyze a proposed joint request and count it as one trial.

        Strategies enforce their own trial limits using `trials`. Plain
        `impact()` calls for scoring and inspection do not count as trials.

        Args:
            remove: Selections in original coordinates, analyzed together.

        Returns:
            Their combined dependency impact. The model is not modified.

        Raises:
            ValueError: A selection belongs to another graph.
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
        self._trials += 1
        return self.impact(remove)

    def impact(self, remove: tuple[Selection, ...]) -> Impact:
        """Analyze selections jointly in original coordinates without running the model.

        The result is that of `DependencyGraph.propagate` for the selections in
        this order, except that a seed merge beyond `max_parts` returns an
        incomplete impact instead of raising. Strategies then skip the request
        like any other incomplete one.

        The bounded cache can share a complete result for the same ordered
        prefix and an equivalent last slice on disjoint seed tensors. The
        impact explains that result from the caller's original seeds.

        Args:
            remove: Hypothetical selections to propagate together.

        Returns:
            Their combined impact, including constraints and diagnostics.

        Raises:
            ValueError: A selection belongs to another graph.
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
        self._sync_cache_settings()
        remove = tuple(remove)
        for selection in remove:
            self.graph.metadata(selection.tensor)
        if not config.impact_cache_size or not has_native_semantics(remove):
            return propagate_request(self.graph, remove, self.constraints)
        key = self._candidate_filter.request_key(remove)
        impact = self._cache.get(key)
        if impact is not None and (impact.complete or impact.requested == remove):
            self.graph.validate()  # propagate performs the entry check on cache misses.
            self._cache.move_to_end(key)
            if remove == impact.requested:
                return impact
            # Single-slice aliases start from the same canonical seed. Rebuild
            # the explanation from the caller's seed, just as the solver does.
            slices = self._candidate_filter.index.equivalent_slices(remove[-1], prefix=remove[:-1])
            return replace(
                impact,
                requested=remove,
                provenance=(
                    *self._candidate_filter.index.slice_provenance(remove[-1], slices),
                    *impact.provenance[len(slices) - 1 :],
                ),
            )
        impact = propagate_request(self.graph, remove, self.constraints)
        self._remember(key, impact)
        return impact

    def _remember(self, key: _ImpactKey, impact: Impact) -> None:
        """Insert into the bounded least-recently-used impact cache."""
        self._cache[key] = impact
        self._cache.move_to_end(key)
        if len(self._cache) > config.impact_cache_size:
            self._cache.popitem(last=False)

    def require_complete(self, impact: Impact) -> None:
        """Reject an analysis with incomplete influence.

        Constraint violations, including repairable ones, are not rejected.

        Args:
            impact: Analysis from this context's graph.

        Raises:
            torch_kirigami.pruning.PlanningError: The influence is incomplete.
            ValueError: `impact` belongs to another graph.
        """
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
        """Score each candidate as an addition to the accepted removal requests.

        The scored candidates need not be members of this context's
        `candidates`; temporary candidates with several selections are allowed. Influence is checked for each candidate
        jointly with the accepted requests, because the combination can trigger
        effects that the candidate alone does not. Scoring does not check
        whether a removal can be executed; that is the strategy's job.

        Args:
            metric: Batch-independent scorer with caller-owned statistics.
            candidates: Requests to score separately.
            accepted_impact: Complete dependency analysis of previously accepted
                removal requests. `None` uses the empty request. The model has
                not been pruned; this analysis is only a planning baseline.

        Returns:
            One finite real score per candidate, in input order.

        Raises:
            TypeError: `metric` has no `score` method.
            torch_kirigami.pruning.PlanningError: An influence is incomplete, or
                the metric returns a misaligned, non-real, or nonfinite batch.
            ValueError: `accepted_impact` or a selection belongs to another graph.
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
        """Measure final resource sizes without creating compact weights.

        Args:
            impact: Joint analysis of the proposed removals.

        Returns:
            The final parameter count for a `ParameterBudget`, the final width of
            each axis for a local channel budget, or the summed final width for a
            global channel budget, aligned with `targets`.

        Raises:
            torch_kirigami.pruning.PlanningError: A `ParameterBudget` needs
                recipes and `impact` has no verified execution recipe.
            ValueError: `impact` belongs to another graph.
        """
        self.graph.validate_impact(impact)
        recipes = self.compile(impact)[0] if self._target.resource == "parameters" else ()
        return self._target.remaining(impact, recipes)

    def parameter_count(self, impact: Impact) -> int:
        """Count the final unique Parameter elements using verified joint recipes.

        Args:
            impact: Joint analysis of the proposed removals.

        Returns:
            The number of parameter elements left after the removal.

        Raises:
            torch_kirigami.pruning.PlanningError: `impact` has no verified
                execution recipe.
        """
        return remaining_parameters(self._parameter_count, self.compile(impact)[0])

    def within_budget(self, impact: Impact) -> bool:
        """Check whether a removal meets every final-size cap.

        Args:
            impact: Joint analysis of the proposed removals.

        Returns:
            Whether every cap is met. For channel budgets, this does not prove
            that the removal can be executed.

        Raises:
            torch_kirigami.pruning.PlanningError: A `ParameterBudget` needs
                recipes and `impact` has no verified execution recipe.
        """
        return self._target.met(self.remaining(impact))

    def deficit(self, impact: Impact) -> int:
        """Measure how far a removal is from meeting the budget.

        The deficit is the sum of the amounts by which final sizes exceed their
        caps, so one axis below its cap cannot offset another axis above its cap.
        The built-in greedy strategies accept an addition only when it lowers
        this value. Coupled removals can also shrink an axis that already meets
        its cap.

        Args:
            impact: Joint analysis of the proposed removals.

        Returns:
            Zero when every cap is met, otherwise the total excess.

        Raises:
            torch_kirigami.pruning.PlanningError: A `ParameterBudget` needs
                recipes and `impact` has no verified execution recipe.
        """
        return self._target.deficit(self.remaining(impact))

    def require_budget(self, impact: Impact, *, result: StrategyResult | None = None) -> None:
        """Raise unless a removal meets the budget.

        The error message reports the measured sizes and search diagnostics.

        Args:
            impact: Joint analysis of the proposed removals.
            result: Optional strategy diagnostics to include in the error.

        Raises:
            torch_kirigami.pruning.PlanningError: A cap is not met, or a
                `ParameterBudget` needs recipes and `impact` has none.
        """
        self._target.require(self.remaining(impact), self.trials, result)

    def compile(
        self, impact: Impact
    ) -> tuple[tuple[TensorRecipe, ...], tuple[AttributeRecipe, ...], tuple[str, ...]]:
        """Compile and verify execution recipes for a joint removal.

        No weights are allocated. Results are cached per `Impact` object until
        a tracked tensor changes.

        Args:
            impact: Resolved joint analysis from this context's graph.

        Returns:
            A tuple of tensor recipes, attribute recipes, and notes.

        Raises:
            torch_kirigami.pruning.PlanningError: The analysis is not resolved or
                the original forward cannot be proven to run on the compact
                structure.
            ValueError: `impact` belongs to another graph.
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
        self._sync_cache_settings()
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
            try:
                recipes = compile_recipes(
                    self.graph,
                    self.operations,
                    impact,
                    attribute_checks=(
                        self._attribute_checks
                        if versions is not None and config.attribute_cache_size
                        else None
                    ),
                    meta_outputs=self._meta_outputs if versions is not None else None,
                )
            finally:
                # Failed validation can also insert a diagnostic into this cache.
                # Its owner enforces retention on both successful and failed calls.
                while len(self._attribute_checks) > config.attribute_cache_size:
                    self._attribute_checks.popitem(last=False)
            self._compiled[key] = (impact, recipes)
            if len(self._compiled) > config.recipe_cache_size:
                self._compiled.popitem(last=False)
        else:
            self._compiled.move_to_end(key)
            recipes = self._compiled[key][1]
        return recipes

    def report(
        self,
        impact: Impact,
        result: StrategyResult,
        *,
        exclusions: tuple[tuple[str, str], ...] = (),
    ) -> ParameterReport | SelectionReport:
        """Build a static report of the measured budget and strategy diagnostics.

        Args:
            impact: Joint analysis of the selected removals.
            result: Strategy result whose stop reason and exclusions are recorded.
            exclusions: Earlier exclusions, such as candidate filtering results,
                listed before the strategy's own exclusions.

        Returns:
            A `ParameterReport` for a `ParameterBudget`, otherwise a
            `SelectionReport`.
        """
        self.graph.validate_impact(impact)
        return self._target.report(self.remaining(impact), impact, self.trials, result, exclusions)
