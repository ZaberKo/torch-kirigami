"""The `Pruner` plan/apply entry point for automatic and manual structural pruning."""

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
from ..errors import AnalysisLimitError
from ..operator_recipes import AttributeRecipe, TensorRecipe
from ..settings import config
from .candidate_checks import propagate_request
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
    """Plan and apply structural pruning on the original module.

    Queries and planning methods read a fixed `DependencyGraph` snapshot and never
    change the model. Planning returns a static, checked `PruningPlan`. Only
    `apply()` mutates the model: it keeps the identity of every module, replaces
    each affected Parameter or buffer with a new tensor object, and does not
    migrate optimizer state. Rebuild the graph before another pruning round.

    Methods that need the graph raise `PlanningError` when no graph was supplied
    and `torch_kirigami.StaleGraphError` when the model structure changed after
    capture.

    Args:
        model: The original module. When `graph` is supplied, it must be
            `graph.model`.
        graph: A fresh `DependencyGraph` of `model`. Queries and planning require
            it. It can be omitted when the pruner only applies a saved plan.
        preserve_io: Whether to protect every axis of the model's external inputs
            and outputs with `Fixed` constraints. Default: `True`.
        constraints: Extra constraints shared by all queries and planning methods.
        granularity: Module alignment settings. They resolve to `Divisible`
            constraints on the logical channel axes that operator rules declare.

    Raises:
        TypeError: `preserve_io` is not a `bool`, or `granularity` is not a
            `Granularity`.
        ValueError: `graph` belongs to another model, or `constraints`, a
            non-default `granularity`, or `preserve_io=False` is supplied
            without a graph.
        torch_kirigami.pruning.PlanningError: A `granularity.by_path` entry does
            not exist or declares no logical channel axes.
        torch_kirigami.StaleGraphError: `graph` no longer matches the model.
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
        """Interface, caller-supplied, and alignment constraints used by every query and plan."""
        return self._constraints

    def _validate_graph(self) -> DependencyGraph:
        """Return the graph after checking its presence, ownership and freshness."""
        if self.graph is None:
            raise PlanningError("Planning requires a DependencyGraph")
        self.graph.validate(self.model)
        return self.graph

    def discover_candidates(self, *, targets: Iterable[str] | None = None) -> CandidateSpace:
        """Build a candidate space from the channel axes that operator rules declare.

        `targets` selects the modules whose declared axes become candidates; it
        does not limit the dependency closure of those candidates. Discovery
        excludes only candidates with a proven irreversible conflict. A domain
        whose candidates are all excluded moves to `excluded_channel_axes` and
        does not count toward channel budgets.

        Args:
            targets: Optional patterns matched against every registered path of a
                module with `fnmatchcase`, so `*` also matches across `.`
                separators. `None` considers every module that declares axes.

        Returns:
            The candidates, the logical channel axes used by channel budgets, the
            fully excluded axes, and the rejection reasons.

        Raises:
            TypeError: `targets` is a bare string or contains non-string patterns.
            torch_kirigami.pruning.PlanningError: No graph is available, a
                pattern matches no module that declares channel axes, or
                operator declarations conflict.
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
        graph = self._validate_graph()
        return self._remember_filtered(graph, discover_candidates(graph, self.constraints, targets))

    def filter_candidates(self, space: CandidateSpace) -> CandidateSpace:
        """Remove candidates with a proven irreversible conflict from a candidate space.

        Only diagnostics marked irreversible exclude a candidate. Repairable
        conflicts, incomplete influence, and unresolved conditions remain for the
        strategy to handle. Candidate order and all channel axes are kept, so the
        channel-budget baseline does not shrink. Rejected keys and reasons are
        recorded in `exclusions`. Filtering does not compile recipes or prove
        that the retained candidates can be pruned. `plan()` applies this filter
        before it calls the strategy; call it directly to prepare custom
        candidates, for example before sparse training.

        Args:
            space: Candidates and the logical channel axes used by channel budgets.

        Returns:
            A space with the retained candidates and the rejection reasons. When
            caching is enabled, the pruner can reuse its latest filtered native
            space after checking freshness. Filtering leaves the input space
            and model unchanged.

        Raises:
            TypeError: `space` is not a `CandidateSpace`.
            ValueError: An axis, selection, or constraint belongs to another graph.
            torch_kirigami.pruning.PlanningError: No graph is available.
            torch_kirigami.StaleGraphError: The model changed after capture.
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
        """Analyze the joint removal of several candidates under this pruner's constraints.

        Args:
            candidates: Hypothetical removals to analyze together as one request.

        Returns:
            The combined dependency impact. The model is not modified, and a
            resolved impact does not prove that the removal can be executed.

        Raises:
            ValueError: A selection belongs to another graph.
            torch_kirigami.pruning.PlanningError: No graph is available.
            torch_kirigami.AnalysisLimitError: Merging the requested selections
                exceeds the region limit.
            torch_kirigami.StaleGraphError: The model changed after capture.
        """
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

        Each candidate is analyzed separately. Equal groups remain separate
        entries with their own candidate keys; pass the result to
        `torch_kirigami.pruning.unique_groups` to count equivalent groups once,
        for example in sparse training. This query does not select candidates,
        produce a plan, or change weights. Use `Pruner.impact()` for joint
        dependency analysis.

        Args:
            candidates: Hypothetical removal requests to analyze individually.
            parameter_filter: Optional callable `(TensorRef, nn.Parameter) -> bool`.
                A parameter's affected regions are kept when it returns `True`.
                `None` keeps all affected parameters. Buffers are always excluded.
                The filter changes only these groups, not metric scores or which
                parameters can be physically pruned.

        Returns:
            One `ParameterGroup` per candidate, keyed by the candidate key, in
            input order. An empty iterable returns an empty tuple. Extraction
            requires complete dependency influence. A complete impact can
            violate a constraint and still yield a group. Extraction does not
            compile execution recipes.

        Raises:
            torch_kirigami.pruning.PlanningError: No graph is available, or a
                candidate has incomplete influence or no parameter regions after
                filtering. Candidates are never skipped.
            ValueError: A selection belongs to another graph.
            torch_kirigami.StaleGraphError: The model changed after capture.
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
        """Plan and apply one automatic pruning round.

        This is equivalent to `self.apply(self.plan(space, budget=budget,
        strategy=strategy))`. See `plan()` and `apply()` for details.

        Args:
            space: Candidates and logical channel axes for this round.
            budget: Final parameter cap or final channel-width caps.
            strategy: Selection strategy for the supplied candidates.

        Returns:
            The same model object, now pruned, and its `PruningResult`.

        Raises:
            TypeError: `space`, `strategy`, or the strategy result has the wrong
                type.
            ValueError: A `ChannelCount` budget does not use the space's channel
                axes in order.
            torch_kirigami.pruning.PlanningError: No graph is available, or no
                plan meets the budget and passes verification.
            torch_kirigami.pruning.ExecutionError: Tracked tensors changed during
                planning, or applying the plan failed.
        """
        return self.apply(self.plan(space, budget=budget, strategy=strategy))

    def plan_remove(self, remove: Iterable[Selection]) -> PruningPlan:
        """Plan exactly the requested manual removal.

        The selections are analyzed as one joint request together with their
        dependency closure. Seeds are never dropped and no candidates are added
        to repair constraints: if the exact request is invalid, planning fails.

        Args:
            remove: Selections in original coordinates, removed as one request.

        Returns:
            A static, checked plan for the requested removal. The model is not
            modified.

        Raises:
            torch_kirigami.pruning.PlanningError: No graph is available, or the
                analysis is unresolved, conflicts with a constraint, or has no
                verified execution recipe.
            torch_kirigami.pruning.ExecutionError: Tracked tensor bindings,
                values, or gradient flags changed during planning.
            ValueError: A selection belongs to another graph.
            torch_kirigami.AnalysisLimitError: Merging the requested selections
                exceeds the region limit.
            torch_kirigami.StaleGraphError: The model changed after capture.
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
        """Select candidates automatically and plan their joint removal.

        Nothing is implicit: the caller supplies the candidate space, the budget,
        and the strategy, which owns its metric. The strategy receives the
        candidates that remain after `filter_candidates()`; the channel axes of
        the space are kept, so the channel-budget baseline does not shrink. The
        planner then independently repeats the joint analysis, compiles the
        execution recipes, and measures the budget. The result either meets the
        budget or planning raises `PlanningError`; partial plans are never
        returned. No compact weights are allocated and the model is not modified.

        Args:
            space: Candidates and logical channel axes, built by hand or by
                `discover_candidates()`.
            budget: `ParameterBudget` for a cap on the final whole-model parameter
                count, or `ChannelRatio` / `ChannelCount` for caps on final logical
                channel widths.
            strategy: Object whose `select(context)` returns a `StrategyResult`.

        Returns:
            A static, checked plan that meets the budget.

        Raises:
            TypeError: `space` is not a `CandidateSpace`, `strategy` has no
                `select` method, or it returns something other than a
                `StrategyResult`.
            ValueError: A `ChannelCount` budget does not use the space's channel
                axes in order, or a reference belongs to another graph.
            torch_kirigami.pruning.PlanningError: No graph is available, the
                strategy returns an unknown key, or the selection misses the
                budget or fails verification.
            torch_kirigami.pruning.ExecutionError: Tracked tensor bindings,
                values, or gradient flags changed during planning.
            torch_kirigami.StaleGraphError: The model changed after capture.
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
        impact = propagate_request(
            graph, tuple(s for key in keys for s in registered[key].remove), self.constraints
        )
        # Reestablish premises after arbitrary strategy callbacks.
        final_context = PlanningContext(graph, candidates, budget, axes, self.constraints)
        try:
            recipes, attributes, notes = final_context.compile(impact)
            # Measurements are independently recomputed; diagnostics cannot make an
            # infeasible selection executable or declare a resource target met.
            final_context.require_budget(impact, result=result)
        except AnalysisLimitError as error:
            raise PlanningError(f"Verification reached an analysis limit: {error}") from error
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
        """Physically prune the model by executing a plan's static recipes.

        This is the only method that mutates the model. It does not score,
        trace, or run forward. Every compact tensor is gathered from the model's
        current weights and allocated before any binding changes; the commit is
        then checked and rolled back on failure. Module objects keep their
        identity, while each affected Parameter or buffer is replaced with a new
        tensor object. Optimizer state is not migrated: recreate the optimizer,
        and rebuild the graph before another round. A plan stores structural
        decisions, not weights, so any pruner whose model matches the plan's
        recorded original structure can apply it; the weights come from that
        receiving model.

        Args:
            plan: Plan for a model whose current structure matches `plan.before`.

        Returns:
            The same model object, now pruned, and a `PruningResult` with the
            parameter replacement map and coordinate mappings.

        Raises:
            torch_kirigami.pruning.ExecutionError: `plan` is not a valid
                `PruningPlan`, the model structure differs from the plan's
                preconditions, a replaced tensor has gradient hooks, or the
                commit failed and was rolled back.
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
