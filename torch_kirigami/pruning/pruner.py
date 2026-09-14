"""One public plan/apply flow for automatic and manual structural pruning."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

import torch
from torch import nn

from ..bindings import has_tensor_hooks, storage_key
from ..regions import gather_region
from ..selection import TensorRef
from .candidates import CandidateSpace, discover_candidates, interface_constraints, parameter_groups
from .granularity import Granularity, alignment_constraints
from .plan import PruningPlan, PruningResult
from .planner import PlanningContext
from .recipes import compact_stride, coordinate_mapping
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
    ExecutionError,
    PlanningError,
    SelectionReport,
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


_DEFAULT_GRANULARITY = Granularity()


class Pruner:
    """Plan and physically compact the original Module using a dependency snapshot.

    Args:
        model: The original module, identical to graph.model.
        graph: A fresh DependencyGraph for plan/prune. May be omitted when
            applying a saved plan. Rebuild explicitly after each pruning round.
        preserve_io: Protect every external input/output axis with Fixed constraints.
        constraints: Extra constraints shared by queries and both planning methods.
        granularity: Immutable module alignment settings, resolved to Divisible
            constraints using the operator rules' declared logical channel axes.
    """

    def __init__(
        self,
        model,
        *,
        graph=None,
        preserve_io=True,
        constraints=(),
        granularity=_DEFAULT_GRANULARITY,
    ):
        if type(preserve_io) is not bool:
            raise TypeError("preserve_io must be boolean")
        if not isinstance(granularity, Granularity):
            raise TypeError("Expected a Granularity configuration")
        self.model, self.graph = model, graph
        self.operations = graph.operations() if graph is not None else ()
        self._interface_constraints = ()
        alignment, self._configuration_notes = (), ()
        if graph is not None:
            graph.validate(model)
            self._interface_constraints = interface_constraints(graph, preserve_io)
            alignment, self._configuration_notes = alignment_constraints(graph, granularity)
        elif constraints or granularity != Granularity() or not preserve_io:
            raise ValueError("Planning configuration requires a DependencyGraph")
        self._constraints = (*self._interface_constraints, *tuple(constraints), *alignment)

    @property
    def constraints(self):
        """Return the common interface, caller, and alignment requirements."""
        return self._constraints

    def _validate_graph(self):
        if self.graph is None:
            raise PlanningError("Planning requires a DependencyGraph")
        self.graph.validate(self.model)

    def discover_candidates(self, *, targets=None):
        """Discover declared candidates explicitly, optionally by module path patterns.

        Targets filter candidate entry points, not the dependency closure. Only
        domains proved wholly protected by external interfaces are excluded.
        """
        self._validate_graph()
        return discover_candidates(self.graph, self._interface_constraints, targets)

    def impact(self, candidates):
        """Analyze a joint candidate batch under this pruner's fixed constraints."""
        self._validate_graph()
        return self.graph.propagate(
            remove=tuple(s for c in candidates for s in c.remove),
            constraints=self.constraints,
        )

    def parameter_groups(self, candidates, *, parameter_filter=None):
        """Extract complete parameter groups for an explicit candidate iterable.

        The optional filter receives (TensorRef, Parameter). Repairable structural
        constraints do not prevent group extraction; incomplete influence does.
        """
        self._validate_graph()
        return parameter_groups(self.graph, candidates, self.constraints, parameter_filter)

    def prune(self, space, *, budget, strategy) -> tuple[nn.Module, PruningResult]:
        """Plan and apply one automatic round with an explicit candidate space."""
        return self.apply(self.plan(space, budget=budget, strategy=strategy))

    def plan_remove(self, remove) -> PruningPlan:
        """Plan exact manual selections, without supplementing or dropping seeds."""
        self._validate_graph()
        before = snapshot(self.model, guarded=self.graph.constant_guards())
        versions = _snapshot(self.graph)
        impact = self.graph.propagate(remove=tuple(remove), constraints=self.constraints)
        recipes, attributes, notes = compile_recipes(self.graph, self.operations, impact)
        return self._finish(
            impact, recipes, attributes, (), SelectionReport(), notes, before, versions
        )

    def plan(self, space, *, budget, strategy) -> PruningPlan:
        """Plan an explicit candidate space without allocating compact weights.

        Args:
            space: Explicit CandidateSpace, constructed manually or discovered.
            budget: ChannelRatio or ChannelCount upper bound on joint removals.
            strategy: Callable returning registered keys; owns any scoring metric.

        Returns:
            A portable static plan. No discovery, metric, or strategy is implicit.
        """
        self._validate_graph()
        if not isinstance(space, CandidateSpace):
            raise TypeError("Expected a CandidateSpace")
        for axis in (*space.channel_axes, *space.protected_channel_axes):
            self.graph.metadata(axis.tensor)
        for candidate in space.candidates:
            for selection in candidate.remove:
                self.graph.metadata(selection.tensor)
            if candidate.axis is not None:
                self.graph.metadata(candidate.axis.tensor)
        before = snapshot(self.model, guarded=self.graph.constant_guards())
        versions = _snapshot(self.graph)
        candidates, axes = space.candidates, space.channel_axes
        registered = {c.key: c for c in candidates}
        context = PlanningContext(
            self.graph, self.operations, candidates, budget, axes, self.constraints
        )
        context.exclusions.extend(space.exclusions)
        keys = tuple(dict.fromkeys(strategy(context)))
        if any(key not in registered for key in keys):
            raise PlanningError("Strategy returned an unregistered candidate key")
        impact = self.graph.propagate(
            remove=(s for key in keys for s in registered[key].remove),
            constraints=self.constraints,
        )
        # Reestablish premises after arbitrary strategy callbacks.
        final_context = PlanningContext(
            self.graph, self.operations, candidates, budget, axes, self.constraints
        )
        if not final_context.within_budget(impact):
            raise PlanningError("Strategy exceeded the joint channel budget")
        final_context.trials = context.trials
        final_context.limit_reached = context.limit_reached
        final_context.exclusions.extend(context.exclusions)
        recipes, attributes, notes = final_context.compile(impact)
        return self._finish(
            impact, recipes, attributes, keys, final_context.report(impact), notes, before, versions
        )

    def _finish(self, impact, recipes, attributes, keys, report, notes, before, versions):
        self.graph.validate(self.model)
        self._check_versions(versions)
        notes = (*self._configuration_notes, *notes)

        def freeze(value):
            if isinstance(value, TensorRef):
                return value.portable()
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
            self.graph.values(),
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
                if has_tensor_hooks(old):
                    raise ExecutionError(
                        "Remove Tensor gradient hooks before replacing their tensors"
                    )
                parts = [gather_region(old.detach(), r) for r in recipe.segments]
                data = parts[0] if len(parts) == 1 else torch.cat(parts, dim=recipe.concat_dim)
                stride = compact_stride(recipe.shape, recipe.memory_format)
                # A singleton dimension can satisfy multiple memory formats with
                # different strides. Allocate the exact contract, not just a format.
                if storage_key(data) == storage_key(old) or data.stride() != stride:
                    packed = torch.empty_strided(
                        recipe.shape, stride, dtype=data.dtype, device=data.device
                    )
                    packed.copy_(data)
                    data = packed
                del parts
                new = (
                    nn.Parameter(data, requires_grad=old.requires_grad)
                    if recipe.tensor.kind == "parameter"
                    else data.requires_grad_(old.requires_grad)
                )
                replacements.append((states[recipe.tensor.paths[0]], old, new))
        check_structure(self.model, plan.before)
        record = managed_record(self.model, plan.after, (a.path for a in plan.attributes))
        commit(self.model, replacements, plan.attributes, record, expected=plan.after)
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
