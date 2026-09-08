"""A shared planning context and a bounded, deterministic greedy policy."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch

from ..contracts import Balanced, Divisible
from .rewrite import compile_recipes
from .types import BudgetReport, Candidate, PlanningError


class PlanningContext:
    """Read-only model access, candidates, joint analysis, scoring, and budget checks.

    Metrics own their statistics and are never cached. Impact queries use a bounded
    32-entry cache; candidates and tensor data are not copied into that cache.
    """

    def __init__(self, graph, operations, candidates, budget, axes, metric, constraints):
        self.graph, self.operations = graph, operations
        self.candidates = tuple(candidates)
        self.budget, self.axes, self.metric = budget, tuple(axes), metric
        self.constraints = tuple(constraints)
        self.widths = tuple(a.tensor.shape[a.dim] for a in self.axes)
        self.targets = (
            tuple(math.floor(budget.ratio * w) for w in self.widths)
            if budget.scope == "local"
            else (math.floor(budget.ratio * sum(self.widths)),)
        )
        self.trials, self.limit_reached = 0, False
        self.exclusions = []
        self._cache = OrderedDict()

    def impact(self, remove):
        """Analyze joint original-coordinate seeds without executing the model."""
        self.graph.validate()
        remove = tuple(remove)
        key = tuple((s.tensor.id, s.regions) for s in remove)
        if key not in self._cache:
            self._cache[key] = self.graph.propagate(remove=remove, constraints=self.constraints)
            if len(self._cache) > 32:
                self._cache.popitem(last=False)
        else:
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
        return tuple(len(impact.selection(a.tensor).project(a.dim)) for a in self.axes)

    def within_budget(self, impact):
        """Check the frozen budget against the whole dependency closure."""
        counts = self.counts(impact)
        return (
            all(a <= b for a, b in zip(counts, self.targets, strict=True))
            if self.budget.scope == "local"
            else sum(counts) <= self.targets[0]
        )

    def compile(self, impact):
        """Check execution support and produce recipes without allocating weights."""
        return compile_recipes(self.graph, self.operations, impact)

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
    result, axes, seen = [], [], set()
    for op in operations:
        for domain in graph.operator_spec(op).candidates:
            if domain.key in seen:
                continue
            seen.add(domain.key)
            axis, block = domain.axis, domain.block
            axes.append(axis)
            width = axis.tensor.shape[axis.dim]
            if block < 1 or width % block:
                raise PlanningError("Candidate block must divide the logical width")
            for start in range(0, width, block):
                result.append(
                    Candidate(
                        f"{domain.key}:{start:012d}",
                        (axis.select(range(start, start + block)),),
                        axis,
                    )
                )
    return tuple(result), tuple(axes)


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
        eligible = []
        for candidate in context.candidates:
            try:
                context.require_complete(context.impact(candidate.remove))
            except PlanningError as error:
                context.exclusions.append((candidate.key, str(error)))
            else:
                eligible.append(candidate)
        scores = context.score(eligible) if eligible else ()
        ranked = [
            c
            for _, c in sorted(
                zip(scores, eligible, strict=True), key=lambda item: (item[0], item[1].key)
            )
        ]
        committed = []
        committed_impact = context.impact(())
        try:
            context.compile(committed_impact)
            valid = context.within_budget(committed_impact)
        except PlanningError:
            valid = False

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
                    before = impact.selection(axis.tensor).project(axis.dim)
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
                        delta = new.selection(axis.tensor).project(axis.dim).subtract(before)
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
