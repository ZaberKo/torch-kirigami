"""A shared planning context and a bounded, deterministic greedy policy."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ..contracts import Balanced, Divisible
from .metrics import Magnitude, WeightTaylor
from .rewrite import compile_recipes
from .types import BudgetReport, Candidate, PlanningError

_IMPACT_CACHE_SIZE = 32


class PlanningContext:
    """Read-only model access, candidates, joint analysis, scoring, and budget checks.

    Metrics own their statistics and are never cached. Impact queries use a bounded
    32-entry cache; four recent verified recipe sets may also be reused. No tensor
    data is copied into either cache, and callback boundaries still validate state.
    """

    def __init__(self, graph, operations, candidates, budget, axes, metric, constraints):
        self._graph, self._operations = graph, tuple(operations)
        self._candidates = tuple(candidates)
        self._budget, self._axes, self._metric = budget, tuple(dict.fromkeys(axes)), metric
        self._constraints = tuple(constraints)
        self._widths = tuple(a.tensor.shape[a.dim] for a in self.axes)
        self._targets = (
            tuple(math.floor(budget.ratio * w) for w in self.widths)
            if budget.scope == "local"
            else (math.floor(budget.ratio * sum(self.widths)),)
        )
        self.trials, self.limit_reached = 0, False
        self.exclusions = []
        self._cache = OrderedDict()
        self._compiled = OrderedDict()

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
    def axes(self):
        """Return unique original budget axes."""
        return self._axes

    @property
    def metric(self):
        """Return the supplied scoring callable, if any."""
        return self._metric

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
            raise PlanningError(
                "Incomplete scoring influence: "
                + "; ".join(f"{d.code}: {d.message}" for d in incomplete)
            )

    def score(self, candidate_batch):
        """Call the metric on a batch, including temporary combined candidates."""
        batch = tuple(candidate_batch)
        if self.metric is None:
            raise PlanningError("This strategy requested scores without a metric")
        for candidate in batch:
            self.require_complete(self.impact(candidate.remove))
        values = self.metric(self, batch)
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
            len(impact.selection(a.tensor).fully_selected_indices(a.dim)) for a in self.axes
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
        # Cache a specific analysis result, not merely seeds/status: extra
        # constraints can produce a different closure or execution requirements.
        key = id(impact)
        if key not in self._compiled:
            self._compiled[key] = (impact, compile_recipes(self.graph, self.operations, impact))
            if len(self._compiled) > 4:
                self._compiled.popitem(last=False)
        else:
            self._compiled.move_to_end(key)
        return self._compiled[key][1]

    def report(self, impact):
        """Freeze the measured budget and strategy diagnostics."""
        return BudgetReport(
            self.axes,
            self.widths,
            self.counts(impact),
            self.targets,
            self.budget.scope,
            self.trials,
            self.limit_reached,
            tuple(self.exclusions),
        )


def discover(graph, operations):
    """Generate declared logical-axis candidates, deduplicating shared domains."""
    result, domains, keys = [], {}, {}
    for op in operations:
        for domain in graph.operator_spec(op).candidates:
            axis, block = domain.axis, domain.block_size
            if domain.key in keys and keys[domain.key] != (axis, block):
                raise PlanningError(f"Conflicting candidate domain key: {domain.key}")
            keys[domain.key] = (axis, block)
            if axis in domains and domains[axis].block_size != block:
                raise PlanningError("Conflicting block sizes for one candidate axis")
            domains.setdefault(axis, domain)
    for axis, domain in domains.items():
        block, width = domain.block_size, axis.tensor.shape[axis.dim]
        if width % block:
            raise PlanningError("Candidate block must divide the logical width")
        for start in range(0, width, block):
            result.append(
                Candidate(
                    f"{domain.key}:{start:012d}",
                    (axis.select(range(start, start + block)),),
                    axis,
                )
            )
    return tuple(result), tuple(domains)


class Greedy:
    """Static score order with bounded balance/divisibility completion and no backtracking.

    Args:
        max_trials: Maximum joint attempts, including cache hits. Scoring queries
            are separate. Rejected candidates may be retried after a commitment.
    """

    def __init__(self, max_trials=10_000):
        if not isinstance(max_trials, int) or isinstance(max_trials, bool) or max_trials < 0:
            raise ValueError("max_trials must be a nonnegative integer")
        self.max_trials = max_trials

    def __call__(self, context):
        """Return only a fully verified set, with an explicit underfill report."""
        committed = []
        committed_impact = context.impact(())
        try:
            context.compile(committed_impact)
            valid = context.within_budget(committed_impact)
        except PlanningError:
            valid = False
        if self.max_trials == 0:
            context.limit_reached = bool(context.candidates)
            if not valid:
                raise PlanningError("The empty request is invalid and the strategy limit is zero")
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
                    for a in context.axes
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
            if type(context.metric) in (Magnitude, WeightTaylor)
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
                scores.extend(context.score(batch))
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
                while context.within_budget(impact):
                    if impact.status == "resolved":
                        try:
                            context.compile(impact)
                        except PlanningError as error:
                            context.exclusions.append((seed.key, str(error)))
                        else:
                            committed, committed_impact, valid, progress = trial, impact, True, True
                        break
                    if impact.status == "conflict":
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
                    added = False
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
                        if not context.within_budget(new) or new.status == "conflict":
                            continue
                        trial, impact, added = [*trial, extra], new, True
                        break
                    if not added:
                        break
                if context.limit_reached:
                    break
            if not progress or context.limit_reached:
                break
        if not valid:
            raise PlanningError(
                "No valid request found within the budget and strategy limit; even the empty request is invalid"
            )
        chosen = {c.key for c in committed}
        rejected = {k for k, _ in context.exclusions}
        context.exclusions.extend(
            (c.key, "Budget, constraints, or bounded completion prevented selection")
            for c in ranked
            if c.key not in chosen and c.key not in rejected
        )
        context.exclusions = [(k, v) for k, v in context.exclusions if k not in chosen]
        return tuple(c.key for c in committed)
