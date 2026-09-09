"""One public plan/apply flow for automatic and manual structural pruning."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import torch
from torch import nn

from ..contracts import Fixed
from ..selection import TensorRef
from .metrics import gather_region
from .plan import PruningPlan, PruningResult
from .planner import Greedy, PlanningContext, discover
from .recipes import coordinate_mapping
from .rewrite import compile_recipes
from .state import (
    attribute,
    check_structure,
    commit,
    managed_record,
    snapshot,
    transformed,
    validate_plan,
)
from .types import (
    AnalysisSummary,
    BudgetReport,
    ExecutionError,
    PlanningError,
)


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:  # Inference tensors do not expose a version counter.
        return None


def _snapshot(graph):
    return tuple(
        (ref, tensor, _version(tensor), tensor.requires_grad)
        for ref, tensor in graph.tensor_bindings()
    )


class Pruner:
    """Plan and physically compact the original Module using a dependency snapshot.

    Args:
        model: The original module, identical to graph.model.
        graph: A fresh DependencyGraph for plan/prune. May be omitted when
            applying a saved plan. Rebuild explicitly after each pruning round.
    """

    def __init__(self, model, *, graph=None):
        if graph is not None:
            graph.validate(model)
        self.model, self.graph = model, graph
        self.operations = graph.operations() if graph is not None else ()

    def prune(self, **kwargs) -> tuple[nn.Module, PruningResult]:
        """Plan and apply one round using the keyword arguments of :meth:`plan`."""
        return self.apply(self.plan(**kwargs))

    def plan(
        self,
        *,
        remove=None,
        metric=None,
        budget=None,
        candidates=None,
        strategy=None,
        preserve_io=True,
        constraints=(),
    ) -> PruningPlan:
        """Generate a verified immutable plan without materializing new weights.

        Manual remove is mutually exclusive with automatic selection options.
        All external tensor axes are protected unless preserve_io is False.
        Caller-supplied candidates require explicit ChannelRatio.axes. Strategies
        may omit a metric if they never request scores.

        Args:
            remove: Manual original-coordinate selections, or None for automatic selection.
            metric: Batch importance callable; required when the strategy requests scores.
            budget: ChannelRatio bound for automatic selection.
            candidates: Optional candidate iterable; requires explicit budget axes.
            strategy: Candidate selection callable; defaults to Greedy.
            preserve_io: Protect all external input/output axes by default.
            constraints: Additional structural constraints applied to the joint request.

        Returns:
            A portable static plan, containing no live model or newly allocated weights.

        Raises:
            ValueError: Manual and automatic options conflict or arguments are invalid.
            PlanningError: Analysis or physical execution cannot be proved valid.
            ExecutionError: A callback changes tracked tensor state during planning.
        """
        if self.graph is None:
            raise PlanningError("Planning requires a DependencyGraph")
        self.graph.validate(self.model)
        before = snapshot(self.model, guarded=self.graph.constant_guards())
        defaults = (
            tuple(
                Fixed(ref.axis(d)) for ref in self.graph.interfaces() for d in range(len(ref.shape))
            )
            if preserve_io
            else ()
        )
        constraints = (*defaults, *tuple(constraints))
        versions = _snapshot(self.graph)
        protected_domains = ()
        if remove is not None:
            if any(x is not None for x in (metric, budget, candidates, strategy)):
                raise ValueError(
                    "Manual remove and automatic selection options are mutually exclusive"
                )
            impact = self.graph.propagate(remove=tuple(remove), constraints=constraints)
            recipes, attributes, notes = compile_recipes(self.graph, self.operations, impact)
            keys, report = (), BudgetReport()
        else:
            if budget is None:
                raise ValueError("Automatic planning requires a ChannelRatio budget")
            if candidates is None:
                candidates, axes = discover(self.graph, self.operations)
                if budget.axes is None and defaults:
                    # Prove protection against the default interface constraints only.
                    # Later execution exclusions never change the frozen denominator.
                    protected = set()
                    for axis in axes:
                        domain = [c for c in candidates if c.axis == axis]
                        if domain and all(
                            any(
                                d.code == "fixed_axis"
                                for d in self.graph.propagate(
                                    remove=c.remove, constraints=defaults
                                ).diagnostics
                            )
                            for c in domain
                        ):
                            protected.add(axis)
                    protected_domains = tuple(
                        (
                            f"domain:{a.tensor.paths[0] if a.tensor.paths else a.tensor.id.split(':', 1)[-1]}:{a.dim}",
                            "All positions are protected by external interfaces",
                        )
                        for a in axes
                        if a in protected
                    )
                    candidates = tuple(c for c in candidates if c.axis not in protected)
                    axes = tuple(a for a in axes if a not in protected)
                if budget.axes is not None:
                    axes = budget.axes
            else:
                if budget.axes is None:
                    raise ValueError("Custom candidates require explicit ChannelRatio.axes")
                candidates, axes = tuple(candidates), budget.axes
            registered = {c.key: c for c in candidates}
            if len(registered) != len(candidates):
                raise ValueError("Duplicate candidate keys")
            for axis in axes:
                self.graph.metadata(axis.tensor)
            for c in candidates:
                for selection in c.remove:
                    self.graph.metadata(selection.tensor)
            context = PlanningContext(
                self.graph,
                self.operations,
                candidates,
                budget,
                axes,
                metric,
                constraints,
            )
            context.exclusions.extend(protected_domains)
            keys = tuple(dict.fromkeys((Greedy() if strategy is None else strategy)(context)))
            if any(key not in registered for key in keys):
                raise PlanningError("Strategy returned an unregistered candidate key")
            impact = self.graph.propagate(
                remove=(s for key in keys for s in registered[key].remove), constraints=constraints
            )
            # Reestablish the caller's premises after the strategy callback.
            final_context = PlanningContext(
                self.graph, self.operations, candidates, budget, axes, metric, constraints
            )
            if not final_context.within_budget(impact):
                raise PlanningError("Strategy exceeded the joint channel budget")
            final_context.trials = context.trials
            final_context.limit_reached = context.limit_reached
            final_context.exclusions.extend(context.exclusions)
            recipes, attributes, notes = final_context.compile(impact)
            report = final_context.report(impact)
        self.graph.validate(self.model)
        self._check_versions(versions)

        def freeze(value):
            if isinstance(value, TensorRef):
                return replace(value, id=value.id.split(":", 1)[-1])
            if isinstance(value, tuple):
                return tuple(freeze(v) for v in value)
            if is_dataclass(value):
                return type(value)(
                    **{f.name: freeze(getattr(value, f.name)) for f in fields(value)}
                )
            return value

        summary = AnalysisSummary(
            impact.status,
            impact.requested,
            tuple(impact.selections.values()),
            tuple(dict.fromkeys(p.reason for p in impact.provenance)),
        )
        plan = PruningPlan(
            freeze(summary),
            freeze(recipes),
            attributes,
            tuple(keys),
            freeze(report),
            notes,
            before,
            transformed(before, recipes, attributes),
        )
        validate_plan(plan)
        return plan

    def _check_versions(self, versions):
        bindings = dict(self.graph.tensor_bindings())
        for ref, old, version, requires_grad in versions:
            current = bindings[ref]
            if (
                current is not old
                or _version(current) != version
                or current.requires_grad != requires_grad
            ):
                raise ExecutionError("Tensor bindings or tracked values changed since planning")

    def apply(self, plan: PruningPlan) -> tuple[nn.Module, PruningResult]:
        """Validate and execute static recipes without scoring, tracing, or forward.

        All allocations precede binding changes. A portable decision is reusable
        on compatible original structures, independently of the originating Pruner.
        """
        if not isinstance(plan, PruningPlan):
            raise ExecutionError("Expected a PruningPlan")
        try:
            validate_plan(plan)
        except (TypeError, ValueError, PlanningError) as error:
            raise ExecutionError(f"Invalid pruning plan: {error}") from error
        check_structure(self.model, plan.before)
        states = {state.paths[0]: state for state in plan.before.tensors}
        replacements = []
        with torch.inference_mode(False), torch.no_grad():
            for recipe in plan.recipes:
                owner, name = attribute(self.model, recipe.tensor.paths[0])
                old = getattr(owner, name)
                parts = [gather_region(old.detach(), r) for r in recipe.segments]
                data = parts[0] if len(parts) == 1 else torch.cat(parts, dim=recipe.concat_dim)
                fmt = (
                    torch.contiguous_format
                    if recipe.memory_format == "contiguous"
                    else getattr(torch, recipe.memory_format)
                )
                data = data.clone(memory_format=fmt)
                new = (
                    nn.Parameter(data, requires_grad=old.requires_grad)
                    if recipe.tensor.kind == "parameter"
                    else data.requires_grad_(old.requires_grad)
                )
                replacements.append((states[recipe.tensor.paths[0]], old, new))
        check_structure(self.model, plan.before)
        record = managed_record(self.model, plan.after, plan.attributes)
        commit(self.model, replacements, plan.attributes, record)
        if self.graph is not None and (plan.recipes or plan.attributes):
            self.graph.invalidate()
        parameter_map = {old: new for state, old, new in replacements if state.kind == "parameter"}
        mappings = {r.tensor: coordinate_mapping(r) for r in plan.recipes}
        return self.model, PruningResult(
            plan,
            plan.after,
            parameter_map,
            mappings,
            (*plan.notes, "Recreate the optimizer; rebuild the graph before another round."),
        )
