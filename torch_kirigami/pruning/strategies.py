"""Static and adaptive greedy selection with shared structural verification."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ..contracts import Balanced, Constraint, Divisible, Impact
from ..selection import AxisRef, IndexSet
from .planner import PlanningContext, _budget_reason, _within_targets
from .types import (
    Candidate,
    Metric,
    ParameterBudget,
    PlanningError,
    StrategyResult,
)

_SCORE_BATCH_SIZE = 32
_EMPTY_INDICES = IndexSet()


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


def _combined_counts(
    context: PlanningContext,
    current: Mapping[AxisRef, IndexSet],
    addition: Mapping[AxisRef, IndexSet],
) -> tuple[int, ...]:
    """Bound channel removals below; parameter budgets impose no channel caps."""
    if isinstance(context.budget, ParameterBudget):
        return ()
    return tuple(
        len(current.get(a, _EMPTY_INDICES).union(addition.get(a, _EMPTY_INDICES)))
        for a in context.channel_axes
    )


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
    known_only: bool = False,
) -> Iterator[Candidate]:
    """Yield known contributions first, retaining joint-only effects as fallback.

    Completion often accepts the first helpful candidate. Inspect the rest only
    if the caller continues; ranking within both preference classes stays intact.
    """
    helpful = set()
    for candidate in contributors:
        delta = removals[candidate.key][axis].subtract(before)
        helps = bool(delta) and (not partitions or any(delta.intersect(p) for p in partitions))
        if helps:
            helpful.add(candidate.key)
            yield candidate
    if not known_only:
        # Include every non-helpful candidate, not just those absent from the
        # index: covered or partition-irrelevant contributions may help jointly.
        yield from (candidate for candidate in ranked if candidate.key not in helpful)


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
    context: PlanningContext,
    initial: list[Candidate],
    committed_removals: Mapping[AxisRef, IndexSet],
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
    predicted = _merge_axis_removals(committed_removals, seed)
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
            known_only=True,
        ):
            if extra.key in used:
                continue
            addition = removals[extra.key]
            if not _within_targets(context, _combined_counts(context, predicted, addition)):
                continue
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
    axes: tuple[AxisRef, ...],
    attempt: Callable[[list[Candidate]], Impact | None],
) -> tuple[list[Candidate], Impact]:
    """Validate a batch and complete constraints revealed by actual joint effects."""
    impact = attempt(trial)
    if impact is None:
        raise PlanningError("Strategy trial limit reached before this candidate could be tested")
    while True:
        if not context.admissible(impact):
            raise PlanningError(_budget_reason(context, context.counts(impact)))
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
        trial_removals = _axis_removals(impact, axes)
        added, last_blocker, limited = False, "", False
        for extra in _completion_order(
            ranked,
            removals,
            axis,
            before,
            partitions,
            contributors=ranked_by_axis.get(axis, ()),
        ):
            if extra.key in trial_keys:
                continue
            counts = _combined_counts(context, trial_removals, removals[extra.key])
            if not _within_targets(context, counts):
                last_blocker = _budget_reason(context, counts, lower_bound=True)
                continue
            new = attempt([*trial, extra])
            if new is None:
                limited = True
                break
            delta = new.selection(axis.tensor).fully_selected_indices(axis.dim).subtract(before)
            if not delta or (partitions and not any(delta.intersect(p) for p in partitions)):
                continue
            if not context.admissible(new):
                last_blocker = _budget_reason(context, context.counts(new))
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
    """Select in one static score order, verifying complete structural changes.

    Args:
        metric: Batch-independent scoring object. Scores are computed once,
            against the initial empty deletion request.
        max_trials: Maximum joint attempts, including cache hits. Scoring queries
            are separate. Proven budget violations and covered seeds require no
            attempt; rejected candidates may be retried after a commitment.
    """

    metric: Metric
    max_trials: int = 10_000

    def __post_init__(self) -> None:
        _validate_options(self.metric, self.max_trials)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Return a verified choice without changing the model or context premises."""
        return _select(context, self.metric, self.max_trials, dynamic=False)


@dataclass(frozen=True)
class DynamicGreedy:
    """Rescore remaining candidates after every accepted structural change.

    A commitment can contain several candidates required by grouping constraints.
    Their completion uses scores from before that entire commitment. The metric
    receives the accepted impact in original coordinates; the model is neither
    physically compacted nor recalibrated. Rescoring may dominate runtime.

    Args:
        metric: Batch-independent scoring object that defines importance relative
            to already selected removals. Its statistics remain caller-owned.
        max_trials: Maximum joint verification attempts, excluding score queries.
    """

    metric: Metric
    max_trials: int = 10_000

    def __post_init__(self) -> None:
        _validate_options(self.metric, self.max_trials)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Return a verified choice, reranking after every complete commitment."""
        return _select(context, self.metric, self.max_trials, dynamic=True)


def _rank_candidates(
    context: PlanningContext,
    metric: Metric,
    candidates: Sequence[Candidate],
    selected: Impact,
    axes: tuple[AxisRef, ...],
    exclusions: dict[str, str],
) -> tuple[list[Candidate], dict[str, dict[AxisRef, IndexSet]]]:
    """Check complete joint influence and score within the bounded impact cache.

    The returned summaries exclude already selected axis positions and remain
    conservative lower bounds for combinations. They do not replace verification.
    Every metric follows the same batching contract; no concrete type is special.
    """
    eligible, scores, removals = [], [], {}
    selected_removals = _axis_removals(selected, axes)
    remaining = [
        candidate
        for candidate in candidates
        if any(s.subtract(selected.selection(s.tensor)) for s in candidate.remove)
    ]
    for start in range(0, len(remaining), _SCORE_BATCH_SIZE):
        batch = []
        for candidate in remaining[start : start + _SCORE_BATCH_SIZE]:
            try:
                joint = context.impact((*selected.requested, *candidate.remove))
                context.require_complete(joint)
            except PlanningError as error:
                exclusions[candidate.key] = str(error)
            else:
                exclusions.pop(candidate.key, None)
                removals[candidate.key] = {
                    axis: delta
                    for axis, indices in _axis_removals(joint, axes).items()
                    if (delta := indices.subtract(selected_removals.get(axis, _EMPTY_INDICES)))
                }
                batch.append(candidate)
        if batch:
            scores.extend(context._score_complete(metric, tuple(batch), selected=selected))
            eligible.extend(batch)
    ranked = [
        candidate
        for _, candidate in sorted(
            zip(scores, eligible, strict=True), key=lambda item: (item[0], item[1].key)
        )
    ]
    return ranked, removals


def _select(
    context: PlanningContext, metric: Metric, max_trials: int, *, dynamic: bool
) -> StrategyResult:
    """Share budget, completion, and failure handling between both greedy loops."""
    committed: list[Candidate] = []
    committed_impact = context.impact(())
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
            if all(not s.subtract(committed_impact.selection(s.tensor)) for s in candidate.remove)
        }
        result = StrategyResult(
            keys=tuple(c.key for c in committed),
            stop_reason=stop_reason,
            exclusions=tuple(
                (key, reason) for key, reason in exclusions.items() if key not in covered
            ),
        )
        context.require_budget(committed_impact, result=result)
        return result

    try:
        context.compile(committed_impact)
        valid = context.admissible(committed_impact)
    except PlanningError as error:
        valid, empty_error = False, str(error)
    parameter_target = isinstance(context.budget, ParameterBudget)
    if valid and parameter_target and context.within_budget(committed_impact):
        return finish("target_reached")
    if max_trials == 0:
        if not valid:
            raise PlanningError(
                "The empty request is invalid and the strategy limit is zero: " + empty_error
            )
        exclusions.update((c.key, "Strategy limit is zero") for c in context.candidates)
        return finish("trial_limit" if context.candidates else "exhausted")
    # Custom candidates may remove unbudgeted positions, so a zero channel cap
    # proves no progress possible only if every seed touches a counted axis.
    if (
        valid
        and not parameter_target
        and not any(context.targets)
        and all(
            any(
                s.tensor == a.tensor and s.fully_selected_indices(a.dim)
                for s in c.remove
                for a in context.channel_axes
            )
            for c in context.candidates
        )
    ):
        exclusions.update((c.key, "Zero channel budget") for c in context.candidates)
        return finish("target_reached")
    repair_constraints = tuple(
        c for c in committed_impact.constraints if isinstance(c, (Balanced, Divisible))
    )
    # Extension constraints can inspect the entire closure. Only the exact built-in
    # count constraints can be projected onto sparse independent axis summaries.
    count_constraints = tuple(c for c in repair_constraints if type(c) in (Balanced, Divisible))
    counted_axes = () if parameter_target else context.channel_axes
    axes = tuple(dict.fromkeys((*counted_axes, *(c.axis for c in repair_constraints))))
    committed_removals = _axis_removals(committed_impact, axes)
    ranked, removals = _rank_candidates(
        context, metric, context.candidates, committed_impact, axes, exclusions
    )
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
            if all(not s.subtract(committed_impact.selection(s.tensor)) for s in seed.remove):
                continue
            counts = _combined_counts(context, committed_removals, removals[seed.key])
            if not _within_targets(context, counts):
                failures[seed.key] = (revision, _budget_reason(context, counts, lower_bound=True))
                continue
            initial = [*committed, seed]
            batch = _propose_batch(
                context,
                initial,
                committed_removals,
                ranked,
                removals,
                ranked_by_axis,
                count_constraints,
            )
            # Recheck the bare seed if the heuristic's preferred completion fails;
            # another partner can remain legal despite identical predicted counts.
            trials = (batch, initial) if len(batch) > len(initial) else (initial,)
            for trial in trials:
                attempts_before = context.trials
                try:
                    trial, impact = _complete_trial(
                        context, trial, ranked, removals, ranked_by_axis, axes, attempt
                    )
                except PlanningError as error:
                    if context.trials > attempts_before or seed.key not in failures:
                        failures[seed.key] = (revision, str(error))
                    if limited:
                        break
                else:
                    committed, committed_impact, valid, progress = trial, impact, True, True
                    committed_removals = _axis_removals(impact, axes)
                    revision += 1
                    if parameter_target and context.within_budget(impact):
                        return finish("target_reached")
                    break
            if limited or (dynamic and progress):
                break
        if not progress or limited:
            break
        if dynamic:
            ranked, removals = _rank_candidates(
                context, metric, context.candidates, committed_impact, axes, exclusions
            )
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
