"""One public plan/apply flow for automatic and manual structural pruning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import fields, is_dataclass
from typing import Any, cast

import torch
from torch import nn

from ..bindings import has_tensor_hooks, storage_key
from ..dep_graph.contracts import Constraint, Fixed, Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.eligibility import has_native_semantics
from ..dep_graph.selection import Selection, TensorRef, gather_region
from ..operator_recipes import AttributeRecipe, TensorRecipe
from ..settings import config
from .candidates import (
    CandidateSpace,
    discover_candidates,
    filter_candidates,
    interface_constraints,
    parameter_groups,
)
from .granularity import Granularity, alignment_constraints
from .groups import ParameterGroup
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
    Candidate,
    ChannelCount,
    ChannelRatio,
    ExecutionError,
    ModelStructure,
    ParameterBudget,
    ParameterReport,
    PlanningError,
    SelectionReport,
    Strategy,
    StrategyResult,
)

_TensorVersions = tuple[tuple[TensorRef, torch.Tensor, int | None, bool], ...]


def _version(tensor: torch.Tensor) -> int | None:
    """Read the mutation counter when the tensor supports version tracking."""
    try:
        return tensor._version
    except RuntimeError:  # Inference tensors do not expose a version counter.
        return None


def _snapshot(graph: DependencyGraph) -> _TensorVersions:
    """Retain live bindings and mutation counters for planning validation."""
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
        model: nn.Module,
        *,
        graph: DependencyGraph | None = None,
        preserve_io: bool = True,
        constraints: Iterable[Constraint] = (),
        granularity: Granularity = _DEFAULT_GRANULARITY,
    ) -> None:
        if type(preserve_io) is not bool:
            raise TypeError("preserve_io must be boolean")
        if not isinstance(granularity, Granularity):
            raise TypeError("Expected a Granularity configuration")
        constraints = tuple(constraints)
        self.model, self.graph = model, graph
        interfaces: tuple[Fixed, ...] = ()
        alignment: tuple[Constraint, ...] = ()
        self._configuration_notes: tuple[str, ...] = ()
        if graph is not None:
            graph.validate(model)
            interfaces = interface_constraints(graph, preserve_io)
            alignment, self._configuration_notes = alignment_constraints(graph, granularity)
        elif constraints or granularity != Granularity() or not preserve_io:
            raise ValueError("Planning configuration requires a DependencyGraph")
        self._constraints = (*interfaces, *constraints, *alignment)
        self._filtered_space: (
            tuple[DependencyGraph, tuple[Constraint, ...], CandidateSpace] | None
        ) = None

    def _remember_filtered(self, graph: DependencyGraph, space: CandidateSpace) -> CandidateSpace:
        """Retain only the latest immutable native filtering result for this pruner."""
        native = (
            config.candidate_cache_size > 0
            and type(self) is Pruner
            and type(graph) is DependencyGraph
            and type(space) is CandidateSpace
            and not any(callable(value) for value in vars(graph).values())
            and all(
                has_native_semantics(value)
                for value in (*graph.relations, *graph.constraints, *self.constraints)
            )
            and all(
                type(c) is Candidate and has_native_semantics((c.axis, c.remove))
                for c in space.candidates
            )
            and has_native_semantics((*space.channel_axes, *space.excluded_channel_axes))
        )
        self._filtered_space = (graph, self.constraints, space) if native else None
        return space

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Return the common interface, caller, and alignment requirements."""
        return self._constraints

    def _validate_graph(self) -> DependencyGraph:
        """Return the graph after checking its presence, ownership and freshness."""
        if self.graph is None:
            raise PlanningError("Planning requires a DependencyGraph")
        self.graph.validate(self.model)
        return self.graph

    def discover_candidates(self, *, targets: Iterable[str] | None = None) -> CandidateSpace:
        """Discover declared candidates explicitly, optionally by module path patterns.

        Targets filter candidate entry points, not the dependency closure. Only
        requests proved irreversibly conflicting are excluded. Entire excluded
        domains are removed before freezing the discovered channel baseline.
        Patterns use `fnmatchcase`: `*` can match across path separators (`.`).
        """
        graph = self._validate_graph()
        return self._remember_filtered(graph, discover_candidates(graph, self.constraints, targets))

    def filter_candidates(self, space: CandidateSpace) -> CandidateSpace:
        """Remove proven irreversible requests from an explicit candidate space.

        Keep candidate order and all declared budget axes. Record rejected keys
        and reasons in `exclusions`. Other conflicts, incomplete influence and
        unresolved conditions remain for the caller's algorithm to inspect.
        This does not compile recipes or prove that retained candidates can execute.
        Use this query before sparse training with custom candidates. Automatic
        planning also applies it before invoking the strategy.

        Args:
            space: Explicit candidates and their frozen channel accounting axes.

        Returns:
            A space containing retained candidates and rejection diagnostics.
            The latest native filtered space can be returned unchanged after
            freshness validation. The input space and model remain unchanged.

        Raises:
            TypeError: The argument is not a CandidateSpace.
            ValueError: A supplied axis, selection or constraint belongs to another graph.
        """
        graph = self._validate_graph()
        previous = self._filtered_space
        if (
            config.candidate_cache_size > 0
            and previous is not None
            and previous[0] is graph
            and previous[2] is space
            and has_native_semantics(self.constraints)
            and previous[1] == self.constraints
            and not any(callable(value) for value in vars(graph).values())
        ):
            return space
        return self._remember_filtered(graph, filter_candidates(graph, self.constraints, space))

    def impact(self, candidates: Iterable[Candidate]) -> Impact:
        """Analyze a joint candidate batch under this pruner's fixed constraints."""
        graph = self._validate_graph()
        return graph.propagate(
            remove=tuple(s for c in candidates for s in c.remove),
            constraints=self.constraints,
        )

    def parameter_groups(
        self,
        candidates: Iterable[Candidate],
        *,
        parameter_filter: Callable[[TensorRef, nn.Parameter], bool] | None = None,
    ) -> tuple[ParameterGroup, ...]:
        """Extract one parameter group per candidate, in input order.

        Analyze candidates separately. Equal groups remain separate entries with
        their candidate keys. To count equivalent groups once in sparse training,
        pass the result to `torch_kirigami.pruning.unique_groups`.
        This query does not select candidates, produce a plan, or change weights.
        Use `Pruner.impact(candidates)` for joint dependency analysis instead.

        Args:
            candidates: Hypothetical removal requests to analyze individually.
            parameter_filter: Optional callable `(TensorRef, nn.Parameter) -> bool`.
                Keep a parameter's affected regions when it returns `True`.
                `None` keeps all affected parameters. Buffers are always excluded.
                This filter affects these groups, not metric scores or which
                parameters can be physically pruned.

        Returns:
            One `ParameterGroup` per candidate, with the candidate's key, in input
            order. An empty iterable returns an empty tuple. Complete analysis
            is required, but constraint satisfaction and execution support are
            not checked. A group alone is not proof that a request can be applied.

        Raises:
            PlanningError: A candidate has incomplete influence or no parameter
                regions after extraction and filtering. No candidate is skipped.
        """
        graph = self._validate_graph()
        return parameter_groups(graph, candidates, self.constraints, parameter_filter)

    def prune(
        self,
        space: CandidateSpace,
        *,
        budget: ChannelCount | ChannelRatio | ParameterBudget,
        strategy: Strategy[PlanningContext],
    ) -> tuple[nn.Module, PruningResult]:
        """Plan and apply one automatic round with an explicit candidate space.

        Args:
            space: Candidate universe for this round.
            budget: Final parameter or channel limits.
            strategy: Selection strategy for the supplied candidates.

        Returns:
            The original model and its pruning result.
        """
        return self.apply(self.plan(space, budget=budget, strategy=strategy))

    def plan_remove(self, remove: Iterable[Selection]) -> PruningPlan:
        """Plan exact manual selections, without supplementing or dropping seeds.

        Args:
            remove: Original-coordinate selections to apply as one request.

        Returns:
            A portable plan for the requested selections.

        Raises:
            PlanningError: Influence is incomplete or has no verified recipe.
        """
        graph = self._validate_graph()
        before = snapshot(self.model, guarded=graph.constant_guards())
        versions = _snapshot(graph)
        impact = graph.propagate(remove=tuple(remove), constraints=self.constraints)
        recipes, attributes, notes = compile_recipes(graph, graph.operations(), impact)
        return self._finish(
            impact, recipes, attributes, (), SelectionReport(), notes, before, versions
        )

    def plan(
        self,
        space: CandidateSpace,
        *,
        budget: ChannelCount | ChannelRatio | ParameterBudget,
        strategy: Strategy[PlanningContext],
    ) -> PruningPlan:
        """Plan an explicit candidate space without allocating compact weights.

        The strategy receives candidates after irreversible-conflict filtering.
        The space's declared budget axes and normalization domains do not shrink.
        Final joint analysis and recipe compilation still validate its choices.

        Args:
            space: Explicit CandidateSpace, constructed manually or discovered.
            budget: ParameterBudget for a final whole-model cap, or ChannelRatio /
                ChannelCount for upper bounds on final channel widths.
            strategy: Implements `select(context)` returning a StrategyResult;
                owns its scoring metric and any selection-specific settings.

        Returns:
            A portable static plan. No discovery, metric, or strategy is implicit.

        Raises:
            PlanningError: The strategy result misses its target or cannot be executed.
            TypeError: The candidate space or strategy result has the wrong type.
        """
        graph = self._validate_graph()
        if not isinstance(space, CandidateSpace):
            raise TypeError("Expected a CandidateSpace")
        for axis in (*space.channel_axes, *space.excluded_channel_axes):
            graph.metadata(axis.tensor)
        for candidate in space.candidates:
            for selection in candidate.remove:
                graph.metadata(selection.tensor)
            if candidate.axis is not None:
                graph.metadata(candidate.axis.tensor)
        before = snapshot(self.model, guarded=graph.constant_guards())
        versions = _snapshot(graph)
        space = self.filter_candidates(space)
        candidates, axes = space.candidates, space.channel_axes
        registered = {c.key: c for c in candidates}
        context = PlanningContext(graph, candidates, budget, axes, self.constraints)
        if not callable(getattr(strategy, "select", None)):
            raise TypeError("Strategy must implement select(context) -> StrategyResult")
        result = strategy.select(context)
        if not isinstance(result, StrategyResult):
            raise TypeError("Strategy.select must return StrategyResult")
        keys = result.keys
        if any(key not in registered for key in keys) or any(
            key not in registered for key, _reason in result.exclusions
        ):
            raise PlanningError("Strategy returned an unregistered candidate key")
        impact = graph.propagate(
            remove=(s for key in keys for s in registered[key].remove),
            constraints=self.constraints,
        )
        # Reestablish premises after arbitrary strategy callbacks.
        final_context = PlanningContext(graph, candidates, budget, axes, self.constraints)
        recipes, attributes, notes = final_context.compile(impact)
        # Measurements are independently recomputed; diagnostics cannot make an
        # infeasible selection executable or declare a resource target met.
        final_context.require_budget(impact, result=result)
        return self._finish(
            impact,
            recipes,
            attributes,
            keys,
            context.report(impact, result, exclusions=space.exclusions),
            notes,
            before,
            versions,
        )

    def _finish(
        self,
        impact: Impact,
        recipes: tuple[TensorRecipe, ...],
        attributes: tuple[AttributeRecipe, ...],
        keys: tuple[str, ...],
        report: SelectionReport | ParameterReport,
        notes: tuple[str, ...],
        before: ModelStructure,
        versions: _TensorVersions,
    ) -> PruningPlan:
        """Freeze validated analysis and recipes into a portable plan."""
        graph = self._validate_graph()
        self._check_versions(graph, versions)
        notes = (*self._configuration_notes, *notes)

        def freeze(value: Any) -> Any:
            """Rebuild heterogeneous dataclass trees with portable tensor references."""
            if isinstance(value, TensorRef):
                return value.portable()
            if isinstance(value, tuple):
                return tuple(freeze(v) for v in value)
            if is_dataclass(value):
                return cast(Callable[..., Any], type(value))(
                    **{f.name: freeze(getattr(value, f.name)) for f in fields(value)}
                )
            return value

        summary = AnalysisSummary(
            impact.status,
            impact.requested,
            tuple(impact.selections.values()),
            tuple(dict.fromkeys(p.reason for p in impact.provenance)),
            graph.values(),
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

    def _check_versions(self, graph: DependencyGraph, versions: _TensorVersions) -> None:
        """Reject changes to live tensor bindings, gradient flags or tracked values."""
        bindings = dict(graph.tensor_bindings())
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

        Args:
            plan: Validated portable plan for a compatible original model.

        Returns:
            The original model and execution details.

        Raises:
            ExecutionError: The plan is invalid, the model changed, or commit failed.
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
