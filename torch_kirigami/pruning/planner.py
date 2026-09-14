"""A shared planning context and a bounded, deterministic greedy policy."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ..contracts import Balanced, Divisible
from .metrics import Magnitude, WeightTaylor
from .rewrite import compile_recipes
from .types import ChannelCount, PlanningError, SelectionReport, channel_targets

_IMPACT_CACHE_SIZE = 32


class PlanningContext:
    """Read-only model access, candidates, joint analysis, scoring, and budget checks.

    Metrics own their statistics and are never cached. Impact queries use a bounded
    32-entry cache; four recent verified recipe sets and 32 configuration checks
    may also be reused. Caches retain no tensor data; version changes invalidate
    compilation caches, and callback boundaries still validate state.
    """

    def __init__(self, graph, operations, candidates, budget, channel_axes, constraints):
        self._graph, self._operations = graph, tuple(operations)
        self._candidates = tuple(candidates)
        self._budget, self._channel_axes = budget, tuple(dict.fromkeys(channel_axes))
        if isinstance(budget, ChannelCount) and budget.channel_axes != self.channel_axes:
            raise ValueError("ChannelCount axes must match the candidate space in order")
        self._constraints = tuple(constraints)
        self._widths = tuple(a.tensor.shape[a.dim] for a in self.channel_axes)
        self._targets = channel_targets(budget, self.widths)
        self.trials, self.limit_reached = 0, False
        self.exclusions = []
        self._cache = OrderedDict()
        self._compiled = OrderedDict()
        self._attribute_checks = OrderedDict()
        self._compile_versions = None

    @property
    def graph(self):
        """Return the source dependency snapshot."""
        return self._graph

    @property
    def operations(self):
        """Return captured calls used for recipe compilation."""
        return self._operations

    @property
    def candidates(self):
        """Return the fixed candidate universe."""
        return self._candidates

    @property
    def budget(self):
        """Return the caller's immutable budget."""
        return self._budget

    @property
    def channel_axes(self):
        """Return unique original logical channel axes."""
        return self._channel_axes

    @property
    def constraints(self):
        """Return fixed analysis premises; callbacks must not mutate constraints."""
        return self._constraints

    @property
    def widths(self):
        """Return the frozen budget denominator."""
        return self._widths

    @property
    def targets(self):
        """Return fixed local caps or the single global cap."""
        return self._targets

    def impact(self, remove):
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

    def require_complete(self, impact):
        """Reject incomplete influence ranges while allowing repairable constraints."""
        incomplete = [d for d in impact.diagnostics if not d.complete]
        if incomplete:
            raise PlanningError("Incomplete scoring influence: " + "; ".join(map(str, incomplete)))

    def score(self, metric, candidate_batch):
        """Call the metric on a batch, including temporary combined candidates."""
        batch = tuple(candidate_batch)
        for candidate in batch:
            self.require_complete(self.impact(candidate.remove))
        values = metric(self, batch)
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

    def counts(self, impact):
        """Measure actual full-axis removals, counting each logical axis once."""
        self.graph.validate_impact(impact)
        return tuple(
            len(impact.selection(a.tensor).fully_selected_indices(a.dim)) for a in self.channel_axes
        )

    def within_budget(self, impact):
        """Check the frozen budget against the whole dependency closure."""
        counts = self.counts(impact)
        return (
            all(a <= b for a, b in zip(counts, self.targets, strict=True))
            if self.budget.scope == "local"
            else sum(counts) <= self.targets[0]
        )

    def compile(self, impact):
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

    def report(self, impact):
        """Freeze the measured budget and strategy diagnostics."""
        return SelectionReport(
            self.channel_axes,
            self.widths,
            self.counts(impact),
            self.targets,
            self.budget.scope,
            self.trials,
            self.limit_reached,
            tuple(self.exclusions),
        )


def _budget_reason(context, impact):
    """Describe the exceeded cap using actual joint removals, not seed counts."""
    counts = context.counts(impact)
    if context.budget.scope == "global":
        return (
            f"Joint request exceeds the global channel budget: "
            f"{sum(counts)} removals > {context.targets[0]} allowed"
        )
    details = (
        f"{axis.tensor.paths[0] if axis.tensor.paths else axis.tensor.id} "
        f"axis {axis.dim}: {count} removals > {cap} allowed"
        for axis, count, cap in zip(context.channel_axes, counts, context.targets, strict=True)
        if count > cap
    )
    return "Joint request exceeds the local channel budget: " + "; ".join(details)


class Greedy:
    """Static score order with bounded balance/divisibility completion and no backtracking.

    Args:
        metric: Batch scoring callable owned by this strategy.
        max_trials: Maximum joint attempts, including cache hits. Scoring queries
            are separate. Rejected candidates may be retried after a commitment.
    """

    def __init__(self, metric, *, max_trials=10_000):
        if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 0:
            raise ValueError("max_trials must be a nonnegative integer")
        if not callable(metric):
            raise TypeError("Greedy requires a callable metric")
        self.metric = metric
        self.max_trials = max_trials

    def __call__(self, context):
        """Return only a fully verified set, with an explicit underfill report."""
        committed = []
        committed_impact = context.impact(())
        empty_error = ""
        # Keep only the latest attempt per candidate. A revision identifies the
        # accepted selection against which that attempt was tested; a later
        # commitment can make an old failure obsolete without changing its seed.
        failures, revision = {}, 0
        try:
            context.compile(committed_impact)
            valid = context.within_budget(committed_impact)
        except PlanningError as error:
            valid = False
            empty_error = str(error)
        if self.max_trials == 0:
            context.limit_reached = bool(context.candidates)
            if not valid:
                raise PlanningError(
                    "The empty request is invalid and the strategy limit is zero: " + empty_error
                )
            context.exclusions.extend((c.key, "Strategy limit is zero") for c in context.candidates)
            return ()
        # A zero budget proves no choice is possible only when each candidate
        # directly removes a budgeted position. Custom unbudgeted seeds may still
        # be legal, so do not infer this solely from ratio or candidate count.
        if (
            valid
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
            context.exclusions.extend((c.key, "Zero channel budget") for c in context.candidates)
            return ()
        eligible, scores = [], []
        # Only our exact built-in metrics promise batch-independent scores.
        # Keep their eligibility/score queries inside the impact cache window;
        # arbitrary custom callables still receive one complete eligible batch.
        batch_size = (
            _IMPACT_CACHE_SIZE
            if type(self.metric) in (Magnitude, WeightTaylor)
            else max(1, len(context.candidates))
        )
        for start in range(0, len(context.candidates), batch_size):
            batch = []
            for candidate in context.candidates[start : start + batch_size]:
                try:
                    context.require_complete(context.impact(candidate.remove))
                except PlanningError as error:
                    context.exclusions.append((candidate.key, str(error)))
                else:
                    batch.append(candidate)
            if batch:
                scores.extend(context.score(self.metric, batch))
                eligible.extend(batch)
        ranked = [
            c
            for _, c in sorted(
                zip(scores, eligible, strict=True), key=lambda item: (item[0], item[1].key)
            )
        ]

        def attempt(keys):
            if context.trials >= self.max_trials:
                context.limit_reached = True
                return None
            context.trials += 1
            return context.impact(s for c in keys for s in c.remove)

        while True:
            progress = False
            for seed in ranked:
                if seed in committed:
                    continue
                trial = [*committed, seed]
                impact = attempt(trial)
                if impact is None:
                    break
                if not any(
                    impact.selection(s.tensor).subtract(committed_impact.selection(s.tensor))
                    for s in impact.selections.values()
                ):
                    continue
                while True:
                    if not context.within_budget(impact):
                        failures[seed.key] = (revision, _budget_reason(context, impact))
                        break
                    if impact.status == "resolved":
                        try:
                            context.compile(impact)
                        except PlanningError as error:
                            failures[seed.key] = (revision, str(error))
                        else:
                            committed, committed_impact, valid, progress = trial, impact, True, True
                            revision += 1
                        break
                    if impact.status == "conflict":
                        failures[seed.key] = (revision, "; ".join(map(str, impact.diagnostics)))
                        break
                    repair = next(
                        (
                            c
                            for c in impact.constraints
                            if isinstance(c, (Balanced, Divisible))
                            and c.check(impact.selections) is not None
                        ),
                        None,
                    )
                    if repair is None:
                        failures[seed.key] = (
                            revision,
                            "No supported greedy completion for this joint request: "
                            + "; ".join(map(str, impact.diagnostics)),
                        )
                        break
                    axis = repair.axis
                    before = impact.selection(axis.tensor).fully_selected_indices(axis.dim)
                    partitions = ()
                    if isinstance(repair, Balanced):
                        counts = [len(p.subtract(before)) for p in repair.partitions]
                        partitions = tuple(
                            p
                            for p, n in zip(repair.partitions, counts, strict=True)
                            if n > min(counts)
                        )
                    added, last_blocker = False, ""
                    for extra in ranked:
                        if extra in trial:
                            continue
                        new = attempt([*trial, extra])
                        if new is None:
                            break
                        delta = (
                            new.selection(axis.tensor)
                            .fully_selected_indices(axis.dim)
                            .subtract(before)
                        )
                        if not delta or (
                            partitions and not any(delta.intersect(p) for p in partitions)
                        ):
                            continue
                        if not context.within_budget(new):
                            last_blocker = _budget_reason(context, new)
                            continue
                        if new.status == "conflict":
                            last_blocker = "; ".join(map(str, new.diagnostics))
                            continue
                        trial, impact, added = [*trial, extra], new, True
                        break
                    if not added:
                        stop = (
                            "Strategy trial limit reached during completion"
                            if context.limit_reached
                            else "Greedy completion found no acceptable addition from the provided candidates"
                        )
                        failures[seed.key] = (
                            revision,
                            stop
                            + ": "
                            + "; ".join(map(str, impact.diagnostics))
                            + (
                                f". Last attempted addition: {last_blocker}" if last_blocker else ""
                            ),
                        )
                        break
                if context.limit_reached:
                    break
            if not progress or context.limit_reached:
                break
        if not valid:
            detail = f". Empty request: {empty_error}"
            if failures:
                key = next(reversed(failures))
                detail += f". Candidate attempt ({key}): {failures[key][1]}"
            if context.limit_reached:
                detail += ". Strategy trial limit reached"
            raise PlanningError(
                "No valid request found within the budget and strategy limit; "
                "even the empty request is invalid" + detail
            )
        chosen = {c.key for c in committed}
        covered = {
            c.key
            for c in context.candidates
            if all(not s.subtract(committed_impact.selection(s.tensor)) for s in c.remove)
        }
        for candidate in ranked:
            if candidate.key in chosen or candidate.key in covered:
                continue
            previous = failures.get(candidate.key)
            if previous is None:
                reason = "Strategy trial limit reached before this candidate could be tested"
            else:
                attempted_revision, reason = previous
                if attempted_revision != revision:
                    reason = (
                        "Earlier attempt: "
                        + reason
                        + ". Not retried after the accepted selection changed"
                        + ("; strategy trial limit reached" if context.limit_reached else "")
                    )
            context.exclusions.append((candidate.key, reason))
        context.exclusions = [
            (k, v) for k, v in context.exclusions if k not in chosen and k not in covered
        ]
        return tuple(c.key for c in committed)
