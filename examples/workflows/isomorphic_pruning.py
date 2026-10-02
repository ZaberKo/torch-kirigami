"""One-shot Isomorphic Pruning with calibrated Taylor scores and fixed family quotas.

Static scores are ranked only inside typed dependency families. At a common
channel ratio each family independently requests its lowest floor(ratio * size)
actions. The ratio is supplied directly, not searched against parameter counts.
Alignment rounds retained widths down, extending each root's
selection with its next lowest scores. All discovered domains are inspected.
"""

import argparse
import json
import math
import time
import weakref
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_kirigami import (
    AxisBarrier,
    AxisRef,
    Balanced,
    Barrier,
    BlockBalance,
    DependencyGraph,
    Divisible,
    Fixed,
    Impact,
    IndexSet,
    LayoutConstraint,
    NonEmpty,
    OperationContext,
    Region,
    Selection,
    TensorRef,
)
from torch_kirigami.measurement import count_parameters
from torch_kirigami.operators.shapes import CallArgumentConstraint
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    Granularity,
    IdentityAxisIndex,
    Metric,
    MetricContext,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    StrategyResult,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.regions import gather_region


def collect_task_gradients(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    batches: int,
) -> dict[str, int]:
    """Accumulate batch-mean cross-entropy gradients without weight updates.

    Match the paper's 100-batch calibration by default. Evaluation mode protects
    BatchNorm statistics and disables dropout. A short dataset is rejected rather
    than silently claiming the requested calibration was performed.
    """
    if type(batches) is not int or batches <= 0:
        raise ValueError("Calibration batches must be positive")
    modes = [(module, module.training) for module in model.modules()]
    model.zero_grad(set_to_none=True)
    completed = samples = 0
    try:
        model.eval()
        with (
            torch.enable_grad(),
            tqdm(
                total=batches, desc="Taylor calibration", unit="batch", dynamic_ncols=True
            ) as progress,
        ):
            for images, labels in loader:
                loss = F.cross_entropy(
                    model(images.to(device, non_blocking=True)),
                    labels.to(device, non_blocking=True),
                )
                if not torch.isfinite(loss):
                    raise ValueError("Taylor calibration produced a nonfinite loss")
                loss.backward()
                completed += 1
                samples += labels.numel()
                progress.update()
                if completed == batches:
                    break
        if completed != batches:
            raise ValueError(
                f"Taylor calibration requires {batches} batches, but only {completed} are available"
            )
    except BaseException:
        model.zero_grad(set_to_none=True)
        raise
    finally:
        for module, training in modes:
            module.training = training
    return {"batches": completed, "samples": samples}


class PaperTaylor:
    """Sum per-parameter L2 norms of weight-gradient products (paper Eq. 3-4).

    Include graph-bound weights, omit biases and buffers, and count each shared
    parameter region once. No domain normalization or model execution is used.
    This differs from the official repository's absolute-product sum reduction.
    """

    def __init__(self) -> None:
        """Bound cached scalar statistics by entry count and total axis positions."""
        self._graph: weakref.ReferenceType[DependencyGraph] | None = None
        self._weights: set[TensorRef] = set()
        self._norms: OrderedDict[
            tuple[object, ...], tuple[weakref.ReferenceType[torch.Tensor], tuple[float, ...]]
        ] = OrderedDict()
        self._norm_positions = 0

    def _axis_norms(
        self,
        ref: TensorRef,
        weight: torch.Tensor,
        dim: int,
        batch: dict[tuple[object, ...], tuple[float, ...]] | None = None,
    ) -> tuple[float, ...]:
        """Batch independent slices; refresh after weight or gradient changes."""
        gradient = weight.grad
        try:
            key = (ref, dim, id(weight), weight._version, id(gradient), gradient._version)
        except RuntimeError:
            key = None  # Inference tensors cannot support mutation-aware reuse.
        if key is not None and batch is not None and key in batch:
            return batch[key]
        if key is not None and key in self._norms:
            gradient_ref, norms = self._norms[key]
            # A freed gradient's Python id can be reused by its replacement.
            # Weak identity checks avoid stale hits without retaining GPU data.
            if gradient_ref() is gradient:
                self._norms.move_to_end(key)
                if batch is not None:
                    batch[key] = norms
                return norms
            self._norms.pop(key)
            self._norm_positions -= len(norms)
        values = weight.detach().movedim(dim, 0).double()
        gradients = gradient.detach().movedim(dim, 0).double()
        products = (values * gradients).reshape(weight.shape[dim], -1)
        norms = tuple(torch.linalg.vector_norm(products, dim=1).cpu().tolist())
        # A nonfinite value on an unselected position must not invalidate an
        # otherwise finite candidate. Validate only the selected score below.
        if key is not None:
            if batch is not None:
                batch[key] = norms
            self._norms[key] = (weakref.ref(gradient), norms)
            self._norm_positions += len(norms)
            # Residual closures can involve far more than 32 weight axes. Bound
            # scalar storage (~8 MiB), rather than repeatedly scanning all weights.
            while len(self._norms) > 256 or self._norm_positions > 262_144:
                _old, removed = self._norms.popitem(last=False)
                self._norm_positions -= len(removed[1])
        return norms

    def _bindings(self, context: MetricContext) -> dict[TensorRef, torch.Tensor]:
        """Resolve recognized weights once per graph, validating live bindings."""
        if self._graph is None or self._graph() is not context.graph:
            self._weights = {
                ref
                for operation in context.graph.operations()
                for name, ref in operation.bindings.items()
                if (
                    name == "weight"
                    or name.endswith(".weight")
                    or name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight")
                )
                and ref.kind == "parameter"
            }
            self._graph = weakref.ref(context.graph)
            self._norms.clear()
            self._norm_positions = 0
        return dict(context.graph.tensor_bindings())

    def _scores_for_axes(
        self, context: MetricContext, axes: tuple[AxisRef, ...]
    ) -> tuple[float, ...]:
        """Score proved single-position closures, preserving parameter-union semantics."""
        bindings = self._bindings(context)
        terms = []
        with torch.no_grad():
            for axis in axes:
                if axis.tensor not in self._weights:
                    continue
                weight = bindings[axis.tensor]
                if weight.is_complex() or weight.grad is None or weight.grad.is_sparse:
                    raise PlanningError("PaperTaylor requires real weights and dense gradients")
                terms.append(self._axis_norms(axis.tensor, weight, axis.dim))
        if not terms:
            raise PlanningError("PaperTaylor found no affected graph-bound weights")
        return tuple(math.fsum(values) for values in zip(*terms, strict=True))

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score additional weight regions using already accumulated gradients."""
        bindings = self._bindings(context)
        scores = []
        batch_norms: dict[tuple[object, ...], tuple[float, ...]] = {}
        with torch.no_grad():
            for candidate in candidates:
                impact = context.impact((*accepted_impact.requested, *candidate.remove))
                context.require_complete(impact)
                terms = []
                for selection in impact.selections.values():
                    if selection.tensor not in self._weights:
                        continue
                    previous = accepted_impact.selections.get(selection.tensor.id)
                    if previous is not None:
                        selection = selection.subtract(previous)
                    if not selection.regions:
                        continue
                    weight = bindings[selection.tensor]
                    if weight.is_complex() or weight.grad is None or weight.grad.is_sparse:
                        raise PlanningError("PaperTaylor requires real weights and dense gradients")
                    if len(selection.regions) == 1:
                        region = selection.regions[0]
                        changed = [
                            dim
                            for dim, indices in enumerate(region.axes)
                            if len(indices) != weight.shape[dim]
                        ]
                        if len(changed) == 1:
                            dim = changed[0]
                            norms = self._axis_norms(selection.tensor, weight, dim, batch_norms)
                            terms.append(math.hypot(*(norms[index] for index in region.axes[dim])))
                            continue
                    norm = torch.zeros((), dtype=torch.float64, device=weight.device)
                    for region in selection.regions:
                        values = gather_region(weight.detach(), region).double()
                        gradient = gather_region(weight.grad.detach(), region).double()
                        norm = torch.hypot(norm, torch.linalg.vector_norm(values * gradient))
                    terms.append(norm.item())
                if not terms:
                    raise PlanningError("PaperTaylor found no affected graph-bound weights")
                score = math.fsum(terms)
                if not math.isfinite(score):
                    raise PlanningError("PaperTaylor produced a nonfinite score")
                scores.append(score)
        return scores


_PAPER_SCORE = PaperTaylor.score


def attention_candidates(graph: DependencyGraph, space: CandidateSpace) -> CandidateSpace:
    """Use a head-dimension unit across all heads of native self-attention.

    Native MHA fixes the head count and requires equal retained head widths.
    Following the reference's channel-group unit, one action removes the same
    local dimension in every head. Other graph-declared candidates are unchanged.
    No independent per-channel scores are added to estimate a block's score.
    """
    index = IdentityAxisIndex(graph.relations)
    groups: dict[AxisRef, tuple[IndexSet, ...]] = {}
    for operation in graph.operations():
        if type(operation.module) is not torch.nn.MultiheadAttention:
            continue
        for constraint in graph.operator_spec(operation).constraints:
            if type(constraint) is not Balanced:
                continue
            partitions = constraint.partitions
            width = constraint.axis.tensor.shape[constraint.axis.dim]
            size = len(partitions[0])
            if partitions != tuple(
                IndexSet.span(start, start + size) for start in range(0, width, size)
            ):
                raise PlanningError("Native MHA requires contiguous equal-width head partitions")
            for axis in index.equivalent_axes(constraint.axis):
                if axis in groups and groups[axis] != partitions:
                    raise PlanningError("Shared attention width has incompatible head partitions")
                groups[axis] = partitions
    candidates = []
    for candidate in space.candidates:
        axis = candidate.axis
        partitions = groups.get(axis)
        if partitions is None:
            candidates.append(candidate)
            continue
        positions = candidate.remove[0].fully_selected_indices(axis.dim)
        if (
            len(candidate.remove) != 1
            or len(positions) != 1
            or candidate.remove != (axis.select(positions),)
        ):
            raise PlanningError(
                "Native attention adaptation requires singleton discovered candidates"
            )
        position = next(iter(positions))
        if position >= len(partitions[0]):
            continue  # This position is included by its first-head representative.
        indices = [partition.intervals[0][0] + position for partition in partitions]
        candidates.append(Candidate(candidate.key, (axis.select(indices),), axis))
    return CandidateSpace(
        tuple(candidates), space.channel_axes, space.protected_channel_axes, space.exclusions
    )


def dependency_signature(
    operations: tuple[OperationContext, ...], impact: Impact
) -> tuple[object, ...]:
    """Compare ordered typed DAGs, preserving directions, adjacency and sharing.

    Names, widths and selected indices are excluded. Different branch capture
    orders can conservatively split equivalent graphs; this follows ordered
    dependency comparison rather than implementing general graph isomorphism.
    """
    selected = {selection.tensor for selection in impact.selections.values()}
    affected = tuple(
        operation
        for operation in operations
        if any(
            ref in selected
            for ref in (*operation.inputs, *operation.outputs, *operation.bindings.values())
        )
    )
    return _signature(affected, impact)


def _signature(affected: tuple[OperationContext, ...], impact: Impact) -> tuple[object, ...]:
    """Describe only affected calls; memoize repeated dimension inspections."""
    selected = {selection.tensor: selection for selection in impact.selections.values()}
    dimensions_by_tensor: dict[TensorRef, tuple[int, ...]] = {}

    def dimensions(ref: TensorRef) -> tuple[int, ...]:
        if ref in dimensions_by_tensor:
            return dimensions_by_tensor[ref]
        selection = selected.get(ref)
        if selection is None:
            return ()
        dims = tuple(dim for dim in range(len(ref.shape)) if selection.fully_selected_indices(dim))
        if not dims:
            raise PlanningError("The family signature cannot describe a partial tensor region")
        dimensions_by_tensor[ref] = dims
        return dims

    producers = {
        ref: (index, port)
        for index, operation in enumerate(affected)
        for port, ref in enumerate(operation.outputs)
    }
    aliases: dict[TensorRef, int] = {}
    signature = []
    for operation in affected:
        if operation.module is not None:
            kind = type(operation.module)
            label = (kind.__module__, kind.__qualname__)
        elif callable(operation.node.target):
            target = operation.node.target
            label = (target.__module__, target.__qualname__)
        else:
            label = (operation.node.op, str(operation.node.target))
        bindings = tuple(
            (name, dimensions(ref), aliases.setdefault(ref, len(aliases)))
            for name, ref in sorted(operation.bindings.items())
            if ref in selected
        )
        signature.append(
            (
                label,
                tuple(
                    (port, dimensions(ref), producers.get(ref))
                    for port, ref in enumerate(operation.inputs)
                    if ref in selected
                ),
                tuple((port, dimensions(ref)) for port, ref in enumerate(operation.outputs)),
                bindings,
            )
        )
    return tuple(signature)


@dataclass
class RemovalAction:
    """One complete deletion closure, with equivalent entry keys counted once."""

    candidate: Candidate
    aliases: list[str]
    family: int
    axes: dict[AxisRef, IndexSet]
    score: float = 0.0
    rank: int = 0


@dataclass(frozen=True)
class _IdentityFamily:
    """One proved component's family and scalar scores; no retained impacts."""

    axes: tuple[AxisRef, ...]
    family: int
    scores: tuple[float, ...]


def rounded_request(
    requested: set[int],
    actions: list[RemovalAction],
    factors: dict[AxisRef, int],
    contributors: dict[AxisRef, tuple[int, ...]],
) -> set[int]:
    """Round retained widths down, as in the reference's `round_to` procedure.

    An affected root extends its deletion set with its next lowest-ranked action.
    Coupled axes are revisited after additions. No raw scores are compared across
    families; ambiguous or empty-axis proposals are rejected, not repaired by a
    different pruning policy. The compiler still checks the complete proposal.
    """
    selected = requested.copy()
    counts: dict[AxisRef, Counter[int]] = defaultdict(Counter)
    for index in sorted(selected):
        for axis, indices in actions[index].axes.items():
            counts[axis].update(indices)
    pending = deque(sorted(counts, key=lambda axis: (axis.tensor.id, axis.dim)))
    while pending:
        axis = pending.popleft()
        remaining = axis.tensor.shape[axis.dim] - len(counts[axis])
        factor = factors.get(axis, 1)
        if remaining > 0 and remaining % factor == 0:
            continue
        if remaining < factor:
            raise PlanningError("Retained-width rounding would delete an entire structural axis")
        available = contributors[axis]
        if len({actions[index].family for index in available}) != 1:
            raise PlanningError(
                "Alignment on this axis spans distinct structural families; use granularity 1"
            )
        addition = next(
            (
                index
                for index in available
                if index not in selected
                and any(position not in counts[axis] for position in actions[index].axes[axis])
            ),
            None,
        )
        if addition is None:
            raise PlanningError("Available candidates cannot realize retained-width rounding")
        selected.add(addition)
        for linked, indices in actions[addition].axes.items():
            counts[linked].update(indices)
            pending.append(linked)
    return selected


@dataclass
class Isomorphic:
    """Apply the supplied ratio independently to each structural family.

    Args:
        metric: Static importance compared only inside the same family.
        ratio: Fraction of each family's actions to remove; no resource search.

    Complete equivalent closures share one action and one score. The smallest
    candidate key is their deterministic representative. Incomplete influences
    are excluded explicitly. Execution failures never shrink family populations
    or substitute another algorithm: invalid allocations fail without mutation.
    """

    metric: Metric = field(default_factory=PaperTaylor)
    ratio: float | Fraction = 0.05
    families: tuple[dict[str, Any], ...] = field(default=(), init=False)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Rank once and validate one complete fixed-ratio allocation."""
        ratio = Fraction(str(self.ratio))
        if not 0 <= ratio < 1:
            raise ValueError("Family pruning ratio must be in [0, 1)")
        self.families = ()
        # A zero ratio never silently adds deletions to repair an invalid model.
        if ratio == 0:
            context.compile(context.impact(()))
            return StrategyResult((), "target_reached")
        axes = set(context.channel_axes)
        factors: dict[AxisRef, int] = {}
        for constraint in (*context.graph.constraints, *context.constraints):
            if isinstance(constraint, (NonEmpty, Divisible)):
                axes.add(constraint.axis)
            if isinstance(constraint, Divisible):
                factors[constraint.axis] = math.lcm(
                    factors.get(constraint.axis, 1), constraint.factor
                )
        by_tensor: dict[TensorRef, list[AxisRef]] = defaultdict(list)
        for axis in sorted(axes, key=lambda axis: (axis.tensor.id, axis.dim)):
            by_tensor[axis.tensor].append(axis)
        actions: list[RemovalAction] = []
        closures: dict[tuple[object, ...], int] = {}
        signatures: dict[tuple[object, ...], int] = {}
        exclusions = []
        operation_indices: dict[TensorRef, set[int]] = defaultdict(set)
        for index, operation in enumerate(context.operations):
            for ref in (*operation.inputs, *operation.outputs, *operation.bindings.values()):
                operation_indices[ref].add(index)
        accepted = context.impact(())
        pending_scores: list[RemovalAction] = []
        axis_index = IdentityAxisIndex(context.graph.relations)
        templates: dict[AxisRef, _IdentityFamily | None] = {}
        equivalent_actions: dict[tuple[tuple[AxisRef, ...], IndexSet], int] = {}
        all_constraints = (*context.graph.constraints, *context.constraints)
        native_constraints = (
            NonEmpty,
            Fixed,
            Divisible,
            Balanced,
            BlockBalance,
            Barrier,
            AxisBarrier,
            LayoutConstraint,
            CallArgumentConstraint,
        )
        reuse_axes = (
            accepted.complete
            and type(self.metric) is PaperTaylor
            and "score" not in vars(self.metric)
            and PaperTaylor.score is _PAPER_SCORE
            and all(type(constraint) in native_constraints for constraint in all_constraints)
        )

        def axis_seed(candidate: Candidate) -> tuple[AxisRef, IndexSet] | None:
            """Recognize a full-axis seed, never infer one from a shape."""
            axis = candidate.axis
            if (
                not reuse_axes
                or type(candidate) is not Candidate
                or type(axis) is not AxisRef
                or type(axis.tensor) is not TensorRef
                or len(candidate.remove) != 1
                or type(candidate.remove[0]) is not Selection
                or any(
                    type(region) is not Region
                    or any(type(indices) is not IndexSet for indices in region.axes)
                    for region in candidate.remove[0].regions
                )
            ):
                return None
            indices = candidate.remove[0].fully_selected_indices(axis.dim)
            if not indices or candidate.remove != (axis.select(indices),):
                return None
            return axis, indices

        def reusable_axes(axis: AxisRef, impact: Impact) -> tuple[AxisRef, ...] | None:
            """Require identity relations, position-invariant checks and an exact witness."""
            members = axis_index.component(axis)
            if members is None:
                return None
            refs = {member.tensor for member in members}
            for constraint in all_constraints:
                if not refs.intersection(constraint.refs):
                    continue
                if type(constraint) not in (NonEmpty, Fixed, Divisible, LayoutConstraint):
                    return None
                if type(constraint) is LayoutConstraint and constraint.ports:
                    return None
            indices = impact.selection(axis.tensor).fully_selected_indices(axis.dim)
            expected = {member.tensor.id: member.select(indices) for member in members}
            if expected != dict(impact.selections):
                return None
            return members

        def score_pending() -> None:
            """Score while recent impacts remain in the planner's bounded cache."""
            if not pending_scores:
                return
            scores = context.score(
                self.metric,
                tuple(action.candidate for action in pending_scores),
                accepted_impact=accepted,
            )
            for action, score in zip(pending_scores, scores, strict=True):
                action.score = score
            pending_scores.clear()

        for number, candidate in enumerate(
            tqdm(
                sorted(context.candidates, key=lambda item: item.key),
                desc="Isomorphic analysis and scoring",
                unit="candidate",
                dynamic_ncols=True,
                # Costs differ sharply between first proofs and cached positions;
                # a linear remaining-time estimate is misleading here.
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]",
            )
        ):
            # Bound queries between scoring batches even when candidates are
            # excluded or alias existing actions. Do not retain full impacts.
            if number % 16 == 0:
                score_pending()
            origin = axis_seed(candidate)
            equivalent = (
                (axis_index.equivalent_axes(origin[0]), origin[1]) if origin is not None else None
            )
            if equivalent in equivalent_actions:
                actions[equivalent_actions[equivalent]].aliases.append(candidate.key)
                continue
            template = (
                templates.get(origin[0]) if origin is not None and len(origin[1]) == 1 else None
            )
            if template is not None:
                axis, indices = origin
                closure = tuple(
                    sorted(
                        (member.tensor.id, member.select(indices).regions)
                        for member in template.axes
                    )
                )
                if closure in closures:
                    actions[closures[closure]].aliases.append(candidate.key)
                    equivalent_actions[equivalent] = closures[closure]
                    continue
                score = template.scores[next(iter(indices))]
                if not math.isfinite(score):
                    raise PlanningError("PaperTaylor produced a nonfinite score")
                closures[closure] = len(actions)
                actions.append(
                    RemovalAction(
                        candidate,
                        [candidate.key],
                        template.family,
                        dict.fromkeys(
                            (member for member in template.axes if member in axes), indices
                        ),
                        score,
                    )
                )
                equivalent_actions[equivalent] = len(actions) - 1
                continue
            impact = context.impact(candidate.remove)
            if not impact.complete:
                exclusions.append(
                    (candidate.key, "; ".join(str(d) for d in impact.diagnostics if not d.complete))
                )
                continue
            fixed = [d for d in impact.diagnostics if d.code in ("fixed_axis", "empty_axis")]
            if fixed:
                exclusions.append((candidate.key, "; ".join(map(str, fixed))))
                continue
            closure = tuple(
                sorted(
                    (ref_id, selection.regions) for ref_id, selection in impact.selections.items()
                )
            )
            if closure in closures:
                actions[closures[closure]].aliases.append(candidate.key)
                if equivalent is not None:
                    equivalent_actions[equivalent] = closures[closure]
                continue
            try:
                indices = set().union(
                    *(operation_indices[s.tensor] for s in impact.selections.values())
                )
                signature = _signature(
                    tuple(context.operations[index] for index in sorted(indices)), impact
                )
            except PlanningError as error:
                exclusions.append((candidate.key, str(error)))
                continue
            contributions = {
                axis: indices
                for selection in impact.selections.values()
                for axis in by_tensor[selection.tensor]
                if (indices := selection.fully_selected_indices(axis.dim))
            }
            closures[closure] = len(actions)
            actions.append(
                RemovalAction(
                    candidate,
                    [candidate.key],
                    signatures.setdefault(signature, len(signatures)),
                    contributions,
                )
            )
            if equivalent is not None:
                equivalent_actions[equivalent] = len(actions) - 1
            members = (
                reusable_axes(origin[0], impact)
                if origin is not None and len(origin[1]) == 1
                else None
            )
            if members is not None:
                template = _IdentityFamily(
                    members, actions[-1].family, self.metric._scores_for_axes(context, members)
                )
                score = template.scores[next(iter(origin[1]))]
                if not math.isfinite(score):
                    raise PlanningError("PaperTaylor produced a nonfinite score")
                actions[-1].score = score
                for member in members:
                    templates[member] = template
            else:
                if origin is not None:
                    templates[origin[0]] = None
                pending_scores.append(actions[-1])
        # Store topology once per family, not once per original channel.
        del closures, signatures
        score_pending()
        families: dict[int, list[int]] = defaultdict(list)
        for index, action in enumerate(actions):
            families[action.family].append(index)
        requested: set[int] = set()
        for members in families.values():
            members.sort(key=lambda index: (actions[index].score, actions[index].candidate.key))
            for rank, index in enumerate(members, start=1):
                actions[index].rank = rank
            quota = ratio.numerator * len(members) // ratio.denominator
            requested.update(members[:quota])
        by_axis: dict[AxisRef, list[int]] = defaultdict(list)
        for index, action in enumerate(actions):
            for axis in action.axes:
                by_axis[axis].append(index)
        contributors = {
            axis: tuple(
                sorted(
                    members, key=lambda index: (actions[index].rank, actions[index].candidate.key)
                )
            )
            for axis, members in by_axis.items()
        }
        selected = rounded_request(requested, actions, factors, contributors)
        keys = tuple(actions[index].candidate.key for index in sorted(selected))
        context.compile(
            context.attempt(
                tuple(
                    seed for index in sorted(selected) for seed in actions[index].candidate.remove
                )
            )
        )
        self.families = tuple(
            {
                "actions": len(members),
                "quota": ratio.numerator * len(members) // ratio.denominator,
                "selected_actions": sum(index in selected for index in members),
                "entry_axes": sorted(
                    {
                        actions[index].candidate.axis.tensor.paths[0]
                        for index in members
                        if actions[index].candidate.axis
                        and actions[index].candidate.axis.tensor.paths
                    }
                ),
            }
            for members in families.values()
        )
        return StrategyResult(keys, "target_reached", tuple(exclusions))


def parse_args() -> argparse.Namespace:
    """Parse one-shot independent family pruning and optional fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--family_pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of deletion actions selected independently in each structural family",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=8,
        help="Retained width alignment; 1 preserves exact family quotas",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=100,
        help="Training batches used to accumulate Taylor gradients (paper: 100)",
    )
    parser.add_argument("--calibration_batch_size", type=int, default=64)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("runs/isomorphic_pruning"))
    parser.add_argument(
        "--compile_latency", action="store_true", help="Compile latency measurement only"
    )
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "granularity",
        "calibration_batches",
        "calibration_batch_size",
        "train_batch_size",
        "val_batch_size",
        "latency_repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "finetune_epochs",
        "train_samples",
        "val_samples",
        "train_workers",
        "val_workers",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if (
        not 0 <= options.family_pruning_ratio < 1
        or not math.isfinite(options.lr)
        or options.lr <= 0
    ):
        parser.error("Require 0 <= family_pruning_ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install CUDA-enabled PyTorch or pass --device cpu")
    return options


def main() -> None:
    """Inspect all discovered domains, independently prune families, train and save."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=options.family_pruning_ratio > 0 or options.finetune_epochs > 0,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = (
        DataLoader(
            train,
            batch_size=options.train_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=options.train_workers,
            persistent_workers=options.train_workers > 0,
            multiprocessing_context="spawn" if options.train_workers else None,
            pin_memory=options.device == "cuda",
        )
        if train is not None
        else None
    )
    val_loader = DataLoader(
        validation,
        batch_size=options.val_batch_size,
        num_workers=options.val_workers,
        persistent_workers=options.val_workers > 0,
        multiprocessing_context="spawn" if options.val_workers else None,
        pin_memory=options.device == "cuda",
    )
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    # plan() requires a resource bound. The original count is only a safety
    # ceiling; fixed family quotas determine pruning, with no parameter search.
    budget = ParameterBudget(count_parameters(model))
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(weights=str(weights), dataset=dataset_info, max_params=budget.max_params)
    records: list[dict[str, Any]] = []
    options.output.mkdir(parents=True, exist_ok=True)

    def record(stage: str, **extra: object) -> None:
        """Persist eager accuracy, separate latency, and allocation diagnostics."""
        accuracy = evaluate(model, val_loader, options.device, description=f"{stage} evaluation")
        row = {
            "stage": stage,
            **accuracy,
            "top1_delta_pp": accuracy["top1"]
            - (records[0]["top1"] if records else accuracy["top1"]),
            **measure_model(model, example, options),
            **extra,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        (options.output / "metrics.json").write_text(
            json.dumps({"config": config, "stages": records}, indent=2) + "\n"
        )

    record("pretrained")
    graph = DependencyGraph.build(model, args=(example,))
    space = Pruner(model, graph=graph).discover_candidates()
    space = attention_candidates(graph, space)
    active_axes = set(space.channel_axes)
    alignment = {
        operation.module_path: options.granularity
        for operation in graph.operations()
        if operation.module_path is not None
        and any(domain.axis in active_axes for domain in graph.operator_spec(operation).candidates)
    }
    pruner = Pruner(model, graph=graph, granularity=Granularity(by_path=alignment))
    config["discovery_exclusions"] = space.exclusions
    config["analysis_diagnostics"] = [str(diagnostic) for diagnostic in graph.diagnostics]
    config["discovered_axes"] = [
        axis.tensor.paths or (axis.tensor.id,) for axis in space.channel_axes
    ]
    print(f"Inspecting all {len(space.candidates)} discovered candidates", flush=True)
    calibration = {"batches": 0, "samples": 0}
    if options.family_pruning_ratio > 0:
        if train is None:
            raise ValueError("Taylor calibration requires the training split")
        calibration_loader = DataLoader(
            train,
            batch_size=options.calibration_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(options.seed),
            num_workers=options.train_workers,
            pin_memory=options.device == "cuda",
            multiprocessing_context="spawn" if options.train_workers else None,
        )
        calibration = collect_task_gradients(
            model, calibration_loader, options.device, options.calibration_batches
        )
    strategy = Isomorphic(ratio=options.family_pruning_ratio)
    started = time.perf_counter()
    plan = pruner.plan(space, budget=budget, strategy=strategy)
    print(
        f"Plan completed in {time.perf_counter() - started:.1f}s; {len(strategy.families)} families; common ratio {strategy.ratio}",
        flush=True,
    )
    model, _ = pruner.apply(plan)
    record(
        "pruned",
        families=strategy.families,
        family_ratio=str(strategy.ratio),
        calibration=calibration,
        exclusions=plan.selection_report.exclusions,
        max_params=budget.max_params,
        before_params=plan.selection_report.before_params,
        after_params=plan.selection_report.after_params,
        target_met=plan.selection_report.target_met,
        planning_trials=plan.selection_report.trials,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        if train_loader is None:
            raise ValueError("Fine-tuning requires training data")
        model.train()
        total, count = 0.0, 0
        with tqdm(
            train_loader,
            desc=f"Fine-tuning {epoch + 1}/{options.finetune_epochs}",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for images, labels in progress:
                images, labels = (
                    images.to(options.device, non_blocking=True),
                    labels.to(options.device, non_blocking=True),
                )
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.detach().item() * labels.numel()
                count += labels.numel()
                progress.set_postfix(images=count, loss=f"{total / count:.4f}", refresh=False)
        if not count:
            raise ValueError("Cannot fine-tune on an empty training split")
    if options.finetune_epochs:
        record("finetuned")
    model.eval()
    save_checkpoint(model, options.output / "model.pt")
    restored = load_checkpoint(
        make_model(options.model, pretrained=False),
        options.output / "model.pt",
        map_location=options.device,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "config": config,
            "families": strategy.families,
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
