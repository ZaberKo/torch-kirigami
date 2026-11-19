"""Static and adaptive greedy selection with shared structural verification."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ..dep_graph.contracts import Balanced, Constraint, Divisible, Impact
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.selection import AxisRef, IndexSet
from ..errors import AnalysisLimitError
from ..settings import config
from .planner import PlanningContext
from .score_reuse import IndependentRanking, _has_native_score
from .types import (
    Candidate,
    ChannelCount,
    ChannelRatio,
    Metric,
    ParameterBudget,
    PlanningError,
    StrategyResult,
)

_EMPTY_INDICES = IndexSet()


def _failure(error: PlanningError | AnalysisLimitError) -> str:
    """Describe why one candidate request was not accepted."""
    if isinstance(error, AnalysisLimitError):
        return f"Analysis limit reached: {error}"
    return str(error)


def _can_reuse_score_checks(metric: Metric, context: PlanningContext) -> bool:
    """Require native scoring, budgets and candidates before batching validation."""
    if type(context) is not PlanningContext or not _has_native_score(metric):
        return False
    if not has_native_semantics(metric.p):
        return False
    if type(context.budget) not in (ChannelCount, ChannelRatio, ParameterBudget):
        return False
    if not has_native_semantics(
        (
            context.targets,
            context.widths,
            context.budget_axes,
            getattr(context.budget, "scope", "global"),
        )
    ):
        return False
    if any(callable(value) for value in vars(context).values()):
        return False
    return all(
        type(candidate) is Candidate and has_native_semantics((candidate.axis, candidate.remove))
        for candidate in context.candidates
    )


def _axis_removals(impact: Impact, axes: tuple[AxisRef, ...]) -> dict[AxisRef, IndexSet]:
    """Retain only nonempty axis contributions, without retaining candidate impacts."""
    removals = {}
    for axis in axes:
        indices = impact.selection(axis.tensor).fully_selected_indices(axis.dim)
        if indices:
            removals[axis] = indices
    return removals


def _merge_axis_removals(
    current: Mapping[AxisRef, IndexSet], addition: Mapping[AxisRef, IndexSet]
) -> dict[AxisRef, IndexSet]:
    """Union sparse summaries without materializing absent axes."""
    combined = dict(current)
    for axis, indices in addition.items():
        combined[axis] = current.get(axis, _EMPTY_INDICES).union(indices)
    return combined


def _ranked_axis_candidates(
    ranked: Sequence[Candidate], removals: Mapping[str, Mapping[AxisRef, IndexSet]]
) -> dict[AxisRef, list[Candidate]]:
    """Index known axis contributors, preserving the global score/key order."""
    by_axis: dict[AxisRef, list[Candidate]] = {}
    for candidate in ranked:
        for axis in removals[candidate.key]:
            by_axis.setdefault(axis, []).append(candidate)
    return by_axis


def _repair_partitions(repair: Balanced, before: IndexSet) -> tuple[IndexSet, ...]:
    """Return partitions that must lose more positions to equalize retained counts."""
    if not isinstance(repair, Balanced):
        return ()
    counts = [len(p.subtract(before)) for p in repair.partitions]
    return tuple(p for p, n in zip(repair.partitions, counts, strict=True) if n > min(counts))


def _completion_order(
    ranked: Sequence[Candidate],
    removals: Mapping[str, Mapping[AxisRef, IndexSet]],
    axis: AxisRef,
    before: IndexSet,
    partitions: tuple[IndexSet, ...],
    *,
    contributors: Sequence[Candidate],
    excluded: set[str] | frozenset[str] = frozenset(),
    known_only: bool = False,
) -> Iterator[Candidate]:
    """Yield known contributions first, retaining joint-only effects as fallback.

    Completion often accepts the first helpful candidate. Inspect the rest only
    if the caller continues; ranking within both preference classes stays intact.
    """
    helpful = set()
    for candidate in contributors:
        if candidate.key in excluded:
            continue
        delta = removals[candidate.key][axis].subtract(before)
        helps = bool(delta) and (not partitions or any(delta.intersect(p) for p in partitions))
        if helps:
            helpful.add(candidate.key)
            yield candidate
    if not known_only:
        # Include every non-helpful candidate, not just those absent from the
        # index: covered or partition-irrelevant contributions may help jointly.
        yield from (
            candidate
            for candidate in ranked
            if candidate.key not in helpful and candidate.key not in excluded
        )


def _balance_deficit(
    constraints: tuple[Balanced, ...],
    current: Mapping[AxisRef, IndexSet],
    addition: Mapping[AxisRef, IndexSet],
) -> float:
    """Estimate further removals needed for balance, solely as a selection heuristic.

    Overlapping constraints can count the same work twice. This value orders
    completion choices; it never proves feasibility or changes the channel budget.
    """
    deficit = 0
    for constraint in constraints:
        removed = current.get(constraint.axis, _EMPTY_INDICES).union(
            addition.get(constraint.axis, _EMPTY_INDICES)
        )
        remaining = [len(p.subtract(removed)) for p in constraint.partitions]
        if constraint.nonempty and min(remaining) == 0:
            return math.inf
        deficit += sum(remaining) - len(remaining) * min(remaining)
    return deficit


def _propose_batch(
    initial: list[Candidate],
    accepted_removals: Mapping[AxisRef, IndexSet],
    ranked: Sequence[Candidate],
    removals: Mapping[str, Mapping[AxisRef, IndexSet]],
    ranked_by_axis: Mapping[AxisRef, Sequence[Candidate]],
    constraints: tuple[Constraint, ...],
) -> list[Candidate]:
    """Construct a count-feasible batch before querying joint dependencies.

    Axis summaries are lower bounds, not full selections. Reuse the constraints'
    own checks on projected axis selections; joint analysis must still establish
    the real counts, tensor layout, and all other structural requirements.
    Every addition changes a counted position, so construction terminates within
    the supplied candidate universe. Failure here is not evidence of infeasibility.
    """
    trial = list(initial)
    used = {c.key for c in trial}
    seed = removals[trial[-1].key]
    predicted = _merge_axis_removals(accepted_removals, seed)
    balances = tuple(c for c in constraints if isinstance(c, Balanced))
    while True:
        repair = None
        for constraint in constraints:
            if not isinstance(constraint, (Balanced, Divisible)):
                continue
            axis = constraint.axis
            diagnostic = constraint.check(
                {axis.tensor.id: axis.select(predicted.get(axis, _EMPTY_INDICES))}
            )
            if diagnostic is not None:
                if diagnostic.severity == "conflict":
                    return initial
                repair = constraint
                break
        if repair is None:
            return trial
        before = predicted.get(repair.axis, _EMPTY_INDICES)
        best, best_deficit = None, math.inf
        for extra in _completion_order(
            ranked,
            removals,
            repair.axis,
            before,
            _repair_partitions(repair, before) if isinstance(repair, Balanced) else (),
            contributors=ranked_by_axis.get(repair.axis, ()),
            excluded=used,
            known_only=True,
        ):
            addition = removals[extra.key]
            deficit = _balance_deficit(balances, predicted, addition)
            if deficit < best_deficit:
                best, best_deficit = extra, deficit
            if deficit == 0:
                break  # No balancing work remains; score order breaks equal costs.
        if best is None:
            # Joint-only effects can satisfy a condition that these summaries
            # cannot predict. Let the normal joint analysis inspect the seed.
            return initial
        predicted = _merge_axis_removals(predicted, removals[best.key])
        trial.append(best)
        used.add(best.key)


def _complete_trial(
    context: PlanningContext,
    trial: list[Candidate],
    ranked: Sequence[Candidate],
    removals: Mapping[str, Mapping[AxisRef, IndexSet]],
    ranked_by_axis: Mapping[AxisRef, Sequence[Candidate]],
    attempt: Callable[[list[Candidate]], Impact | None],
) -> tuple[list[Candidate], Impact]:
    """Validate a batch and complete constraints revealed by actual joint effects."""
    impact = attempt(trial)
    if impact is None:
        raise PlanningError("Strategy trial limit reached before this candidate could be tested")
    while True:
        if impact.status == "resolved":
            context.compile(impact)
            return trial, impact
        if impact.status == "conflict":
            raise PlanningError("; ".join(map(str, impact.diagnostics)))
        repair = next(
            (
                c
                for c in impact.constraints
                if isinstance(c, (Balanced, Divisible)) and c.check(impact.selections) is not None
            ),
            None,
        )
        if repair is None:
            raise PlanningError(
                "No supported greedy completion for this joint request: "
                + "; ".join(map(str, impact.diagnostics))
            )
        axis = repair.axis
        before = impact.selection(axis.tensor).fully_selected_indices(axis.dim)
        partitions = _repair_partitions(repair, before) if isinstance(repair, Balanced) else ()
        trial_keys = {c.key for c in trial}
        added, last_blocker, limited = False, "", False
        for extra in _completion_order(
            ranked,
            removals,
            axis,
            before,
            partitions,
            contributors=ranked_by_axis.get(axis, ()),
            excluded=trial_keys,
        ):
            new = attempt([*trial, extra])
            if new is None:
                limited = True
                break
            delta = new.selection(axis.tensor).fully_selected_indices(axis.dim).subtract(before)
            if not delta or (partitions and not any(delta.intersect(p) for p in partitions)):
                continue
            if new.status == "conflict":
                last_blocker = "; ".join(map(str, new.diagnostics))
                continue
            trial, impact, added = [*trial, extra], new, True
            break
        if not added:
            stop = (
                "Strategy trial limit reached during completion"
                if limited
                else "Greedy completion found no acceptable addition from the provided candidates"
            )
            raise PlanningError(
                stop
                + ": "
                + "; ".join(map(str, impact.diagnostics))
                + (f". Last attempted addition: {last_blocker}" if last_blocker else "")
            )


def _validate_options(metric: Metric, max_trials: int) -> None:
    """Validate the shared explicit scoring and work-limit contract."""
    if type(max_trials) is not int or max_trials < 0:
        raise ValueError("max_trials must be a nonnegative integer")
    if not callable(getattr(metric, "score", None)):
        raise TypeError("A greedy strategy requires a metric with score()")


@dataclass(frozen=True)
class Greedy:
    """Greedy selection in one static score order.

    The metric scores every candidate once, relative to the empty request, and
    the scores are never updated. Candidates are tried in ascending
    `(score, key)` order. Each seed is verified jointly with the accepted
    selection; when grouping constraints such as `Balanced` or `Divisible`
    require more removals, other candidates are added to complete it. A complete
    selection is accepted only when it compiles and lowers the unmet budget.
    Selection stops when the budget is met, the candidates are exhausted, or
    `max_trials` is reached. The model is never modified during selection.

    Attributes:
        metric: Batch-independent scorer.
        max_trials: Maximum number of joint verification attempts, including
            cache hits. Scoring queries do not count. Seeds already covered by
            accepted removals and candidates with a proven irreversible conflict
            use no attempt.

    Raises:
        ValueError: `max_trials` is not a nonnegative integer.
        TypeError: `metric` has no `score` method.
    """

    metric: Metric
    max_trials: int = 10_000

    def __post_init__(self) -> None:
        _validate_options(self.metric, self.max_trials)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Select candidates that meet the budget, without modifying the model.

        Args:
            context: Planning services and the filtered candidates.

        Returns:
            The accepted candidate keys and search diagnostics.

        Raises:
            torch_kirigami.pruning.PlanningError: The accepted selection does not
                meet the budget, scoring fails, or no valid request exists.
        """
        native = _can_reuse_score_checks(self.metric, context)
        with context.graph._native_queries(context.constraints, enabled=native):
            return _select(context, self.metric, self.max_trials, dynamic=False, native=native)


@dataclass(frozen=True)
class DynamicGreedy:
    """Greedy selection that rescores the remaining candidates after each acceptance.

    Selection works like `Greedy`, but after each accepted complete selection
    the metric rescores the remaining candidates relative to the new accepted
    impact. A complete selection can contain several candidates required by
    grouping constraints; their completion uses the scores from before that
    selection. Rescoring happens only in the dependency analysis, in original
    coordinates: the model is never physically pruned, retrained, or
    recalibrated during planning. Rescoring can dominate the runtime.

    Attributes:
        metric: Batch-independent scorer that supports scoring relative to
            accepted removals. Its statistics remain caller-owned.
        max_trials: Maximum number of joint verification attempts, including
            cache hits. Scoring queries do not count.

    Raises:
        ValueError: `max_trials` is not a nonnegative integer.
        TypeError: `metric` has no `score` method.
    """

    metric: Metric
    max_trials: int = 10_000

    def __post_init__(self) -> None:
        _validate_options(self.metric, self.max_trials)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Select candidates that meet the budget, rescoring after each acceptance.

        Args:
            context: Planning services and the filtered candidates.

        Returns:
            The accepted candidate keys and search diagnostics.

        Raises:
            torch_kirigami.pruning.PlanningError: The accepted selection does not
                meet the budget, scoring fails, or no valid request exists.
        """
        native = _can_reuse_score_checks(self.metric, context)
        with context.graph._native_queries(context.constraints, enabled=native):
            return _select(context, self.metric, self.max_trials, dynamic=True, native=native)


def _rank_candidates(
    context: PlanningContext,
    metric: Metric,
    candidates: Sequence[Candidate],
    accepted_impact: Impact,
    axes: tuple[AxisRef, ...],
    exclusions: dict[str, str],
    independent: IndependentRanking | None = None,
    analysis_order: Mapping[str, tuple] | None = None,
) -> tuple[list[Candidate], dict[str, Mapping[AxisRef, IndexSet]]]:
    """Check complete joint influence and score within the bounded impact cache.

    The returned summaries exclude already selected axis positions and remain
    conservative lower bounds for combinations. They do not replace verification.
    Unproved candidates follow the ordinary metric batching contract. Proven
    native statistics join the same global score order, without separate pruning.
    `analysis_order` maps candidate keys to a query order that keeps equivalent
    native requests near each other within the bounded impact cache.
    """
    eligible, scores, removals = [], {}, {}
    proved = independent.score(accepted_impact) if independent is not None else None
    proved_scores = proved.scores if proved is not None else {}
    proved_covered = proved.covered if proved is not None else frozenset()
    accepted_removals = _axis_removals(accepted_impact, axes)
    remaining = [
        candidate
        for candidate in candidates
        if candidate.key in proved_scores
        or (
            candidate.key not in proved_covered
            and not all(s.issubset(accepted_impact.selection(s.tensor)) for s in candidate.remove)
        )
    ]
    if analysis_order is not None:
        # Every candidate is still scored. Final ordering remains (score, key).
        remaining.sort(key=lambda candidate: analysis_order[candidate.key])
    batch_size = config.score_batch_size
    for start in range(0, len(remaining), batch_size):
        batch = []
        for candidate in remaining[start : start + batch_size]:
            rejected = context._candidate_filter.rejections(candidate, accepted_impact)
            if rejected:
                exclusions[candidate.key] = "; ".join(map(str, rejected))
                continue
            if candidate.key in proved_scores:
                eligible.append(candidate)
                scores[candidate.key] = proved_scores[candidate.key]
                removals[candidate.key] = proved.removals[candidate.key]
                exclusions.pop(candidate.key, None)
                continue
            try:
                joint = context.impact((*accepted_impact.requested, *candidate.remove))
                context.require_complete(joint)
                removals[candidate.key] = {
                    axis: delta
                    for axis, indices in _axis_removals(joint, axes).items()
                    if (delta := indices.subtract(accepted_removals.get(axis, _EMPTY_INDICES)))
                }
            except (PlanningError, AnalysisLimitError) as error:
                exclusions[candidate.key] = _failure(error)
            else:
                exclusions.pop(candidate.key, None)
                batch.append(candidate)
        if not batch:
            continue
        try:
            values = context._score_complete(metric, tuple(batch), accepted_impact=accepted_impact)
        except (PlanningError, AnalysisLimitError) as batch_error:
            # Scores do not depend on batching, so scoring each candidate alone
            # excludes only the candidates whose scoring fails. A batch failure
            # that no single candidate reproduces breaks that contract.
            scored = []
            for candidate in batch:
                try:
                    (value,) = context._score_complete(
                        metric, (candidate,), accepted_impact=accepted_impact
                    )
                except (PlanningError, AnalysisLimitError) as error:
                    exclusions[candidate.key] = _failure(error)
                    del removals[candidate.key]
                else:
                    scored.append((candidate, value))
            if len(scored) == len(batch):
                raise batch_error
        else:
            scored = list(zip(batch, values, strict=True))
        scores.update((candidate.key, value) for candidate, value in scored)
        eligible.extend(candidate for candidate, _value in scored)
    ranked = sorted(eligible, key=lambda candidate: (scores[candidate.key], candidate.key))
    return ranked, removals


def _select(
    context: PlanningContext, metric: Metric, max_trials: int, *, dynamic: bool, native: bool
) -> StrategyResult:
    """Share budget, completion, and failure handling between both greedy loops.

    `native` is the result of `_can_reuse_score_checks` for this metric and context.
    """
    accepted_candidates: list[Candidate] = []
    accepted_impact = context.impact(())
    exclusions: dict[str, str] = {}
    failures: dict[str, tuple[int, str]] = {}
    revision, limited, empty_error = 0, False, ""
    start_trials = context.trials

    def finish(
        stop_reason: Literal["target_reached", "exhausted", "trial_limit"],
    ) -> StrategyResult:
        """Freeze local diagnostics; measured resource counts stay with the context."""
        for key, (attempted_revision, reason) in failures.items():
            if attempted_revision != revision:
                reason = (
                    "Earlier attempt: "
                    + reason
                    + ". Not retried after the accepted selection changed"
                    + ("; strategy trial limit reached" if limited else "")
                )
            # A newer completeness failure found during dynamic rescoring is
            # more useful than an earlier rejected structural proposal.
            exclusions.setdefault(key, reason)
        covered = {
            candidate.key
            for candidate in context.candidates
            if all(s.issubset(accepted_impact.selection(s.tensor)) for s in candidate.remove)
        }
        result = StrategyResult(
            keys=tuple(c.key for c in accepted_candidates),
            stop_reason=stop_reason,
            exclusions=tuple(
                (key, reason) for key, reason in exclusions.items() if key not in covered
            ),
        )
        context.require_budget(accepted_impact, result=result)
        return result

    try:
        context.compile(accepted_impact)
        valid = True
    except PlanningError as error:
        valid, empty_error = False, str(error)
    accepted_deficit = context.deficit(accepted_impact) if valid else None
    if accepted_deficit == 0:
        return finish("target_reached")
    if max_trials == 0:
        if not valid:
            raise PlanningError(
                "The empty request is invalid and the strategy limit is zero: " + empty_error
            )
        exclusions.update((c.key, "Strategy limit is zero") for c in context.candidates)
        return finish("trial_limit" if context.candidates else "exhausted")
    repair_constraints = tuple(
        c for c in accepted_impact.constraints if isinstance(c, (Balanced, Divisible))
    )
    # Extension constraints can inspect the entire closure. Only the exact built-in
    # count constraints can be projected onto sparse independent axis summaries.
    count_constraints = tuple(c for c in repair_constraints if type(c) in (Balanced, Divisible))
    counted_axes = context.budget_axes
    axes = tuple(dict.fromkeys((*counted_axes, *(c.axis for c in repair_constraints))))
    accepted_removals = _axis_removals(accepted_impact, axes)
    independent = IndependentRanking.build(context, metric, axes)
    analysis_order = None
    if native and context._candidate_filter.reuse_equivalent:
        index = context._candidate_filter.index
        analysis_order = {
            candidate.key: tuple(
                (s.tensor.id, tuple(tuple(axis.intervals for axis in r.axes) for r in s.regions))
                for s in index.canonical_seeds(candidate.remove)
            )
            for candidate in context.candidates
        }

    def rank() -> tuple[list[Candidate], dict[str, Mapping[AxisRef, IndexSet]]]:
        """Use proved independent statistics; retain ordinary joint scoring otherwise."""
        return _rank_candidates(
            context,
            metric,
            context.candidates,
            accepted_impact,
            axes,
            exclusions,
            independent,
            analysis_order,
        )

    ranked, removals = rank()
    ranked_by_axis = _ranked_axis_candidates(ranked, removals)

    def attempt(candidates: list[Candidate]) -> Impact | None:
        """Count every real joint attempt, including failing analysis and cache hits."""
        nonlocal limited
        if context.trials - start_trials >= max_trials:
            limited = True
            return None
        return context.attempt(tuple(s for candidate in candidates for s in candidate.remove))

    while True:
        progress = False
        for seed in ranked:
            if all(s.issubset(accepted_impact.selection(s.tensor)) for s in seed.remove):
                continue
            rejected = context._candidate_filter.rejections(seed, accepted_impact)
            if rejected:
                exclusions[seed.key] = "; ".join(map(str, rejected))
                continue
            initial = [*accepted_candidates, seed]
            try:
                batch = _propose_batch(
                    initial,
                    accepted_removals,
                    ranked,
                    removals,
                    ranked_by_axis,
                    count_constraints,
                )
            except AnalysisLimitError:
                batch = initial  # Joint analysis checks the bare seed.
            # Recheck the bare seed if the heuristic's preferred completion fails;
            # another partner can remain legal despite identical predicted counts.
            trials = (batch, initial) if len(batch) > len(initial) else (initial,)
            for trial in trials:
                attempts_before = context.trials
                try:
                    trial, impact = _complete_trial(
                        context, trial, ranked, removals, ranked_by_axis, attempt
                    )
                    deficit = context.deficit(impact)
                    trial_removals = _axis_removals(impact, axes)
                except (PlanningError, AnalysisLimitError) as error:
                    if context.trials > attempts_before or seed.key not in failures:
                        failures[seed.key] = (revision, _failure(error))
                    if limited:
                        break
                else:
                    if accepted_deficit is not None and deficit >= accepted_deficit:
                        failures[seed.key] = (revision, "No reduction of unmet budget targets")
                        continue
                    accepted_candidates, accepted_impact = trial, impact
                    valid = progress = True
                    accepted_removals = trial_removals
                    accepted_deficit = deficit
                    revision += 1
                    if deficit == 0:
                        return finish("target_reached")
                    break
            if limited or (dynamic and progress):
                break
        if not progress or limited:
            break
        if dynamic:
            if context.trials - start_trials >= max_trials:
                limited = True
                break
            ranked, removals = rank()
            ranked_by_axis = _ranked_axis_candidates(ranked, removals)
    if not valid:
        detail = f". Empty request: {empty_error}"
        if failures:
            key = next(reversed(failures))
            detail += f". Candidate attempt ({key}): {failures[key][1]}"
        if limited:
            detail += ". Strategy trial limit reached"
        raise PlanningError(
            "No valid request found within the budget and strategy limit; "
            "even the empty request is invalid" + detail
        )
    if limited:
        for candidate in ranked:
            if candidate.key not in failures:
                exclusions[candidate.key] = (
                    "Strategy trial limit reached before this candidate could be tested"
                )
    return finish("trial_limit" if limited else "exhausted")
