"""Independent Isomorphic Pruning family quotas with a whole-model parameter target.

Static scores are ranked only inside typed dependency families. At a common
channel ratio each family independently requests its lowest floor(ratio * size)
actions. A bounded ascending search over exact ratio breakpoints finds an
executable quota allocation meeting the parameter target; it is not global
Greedy ranking. Alignment rounds retained widths down, extending each root's
selection with its next lowest scores. All discovered domains are inspected.
"""

import argparse
import json
import math
import time
from collections import Counter, defaultdict, deque
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
    AxisRef,
    DependencyGraph,
    Divisible,
    Impact,
    IndexSet,
    NonEmpty,
    OperationContext,
    TensorRef,
)
from torch_kirigami.pruning import (
    Candidate,
    Granularity,
    GroupMagnitude,
    Metric,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    StrategyResult,
    load_checkpoint,
    save_checkpoint,
)


def dependency_signature(
    operations: tuple[OperationContext, ...], impact: Impact
) -> tuple[object, ...]:
    """Compare ordered typed DAGs, preserving directions, adjacency and sharing.

    Names, widths and selected indices are excluded. Different branch capture
    orders can conservatively split equivalent graphs; this follows ordered
    dependency comparison rather than implementing general graph isomorphism.
    """
    selected = {selection.tensor: selection for selection in impact.selections.values()}

    def dimensions(ref: TensorRef) -> tuple[int, ...]:
        selection = selected.get(ref)
        if selection is None:
            return ()
        dims = tuple(dim for dim in range(len(ref.shape)) if selection.fully_selected_indices(dim))
        if not dims:
            raise PlanningError("The family signature cannot describe a partial tensor region")
        return dims

    affected = tuple(
        operation
        for operation in operations
        if any(
            ref in selected
            for ref in (*operation.inputs, *operation.outputs, *operation.bindings.values())
        )
    )
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
    """Apply independent family quotas; search their common ratio for a parameter cap.

    Args:
        metric: Static importance compared only inside the same family.
        max_trials: Bound on ratio breakpoints examined, including those whose
            alignment produces an unchanged request. Joint attempts are reported
            separately by the planner.

    Complete equivalent closures share one action and one score. The smallest
    candidate key is their deterministic representative. Incomplete influences
    are excluded explicitly. Execution failures never shrink family populations
    or substitute another algorithm: later complete quota allocations are tried.
    """

    metric: Metric = field(default_factory=GroupMagnitude)
    max_trials: int = 10_000
    families: tuple[dict[str, Any], ...] = field(default=(), init=False)
    ratio: Fraction = field(default=Fraction(0), init=False)
    ratio_trials: int = field(default=0, init=False)
    rejections: tuple[str, ...] = field(default=(), init=False)

    def select(self, context: PlanningContext) -> StrategyResult:
        """Search ascending exact quota changes without assuming feasibility is monotone."""
        if not isinstance(context.budget, ParameterBudget):
            raise PlanningError("Isomorphic's ratio search requires ParameterBudget")
        if type(self.max_trials) is not int or self.max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")
        self.families, self.ratio, self.ratio_trials, self.rejections = (), Fraction(0), 0, ()
        # A valid no-op needs no model-wide scoring. An initially violated
        # constraint can still become legal after deletion, so continue on that
        # failure instead of treating an empty request as universally feasible.
        try:
            if context.within_budget(context.impact(())):
                return StrategyResult((), "target_reached")
        except PlanningError:
            pass
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
        for candidate in sorted(context.candidates, key=lambda item: item.key):
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
                continue
            try:
                signature = dependency_signature(context.operations, impact)
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
        # Store topology once per family, not once per original channel.
        del closures, signatures
        for start in range(0, len(actions), 32):
            batch = actions[start : start + 32]
            scores = context.score(self.metric, tuple(action.candidate for action in batch))
            for action, score in zip(batch, scores, strict=True):
                action.score = score
        families: dict[int, list[int]] = defaultdict(list)
        for index, action in enumerate(actions):
            families[action.family].append(index)
        events: dict[Fraction, list[int]] = defaultdict(list)
        for members in families.values():
            members.sort(key=lambda index: (actions[index].score, actions[index].candidate.key))
            for rank, index in enumerate(members, start=1):
                actions[index].rank = rank
                events[Fraction(rank, len(members))].append(index)
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
        requested: set[int] = set()
        previous: frozenset[int] | None = None
        last_count: int | None = None
        rejected: deque[str] = deque(maxlen=8)
        stop_reason = "all independent quota allocations exhausted"
        # Quotas floor(r*N) change only at these exact rational breakpoints.
        for ratio in (Fraction(0), *sorted(events)):
            if self.ratio_trials >= self.max_trials:
                stop_reason = "ratio trial limit reached"
                break
            self.ratio_trials += 1
            requested.update(events.get(ratio, ()))
            try:
                selected = rounded_request(requested, actions, factors, contributors)
                if frozenset(selected) == previous:
                    continue
                previous = frozenset(selected)
                keys = tuple(actions[index].candidate.key for index in sorted(selected))
                impact = context.attempt(
                    tuple(
                        seed
                        for index in sorted(selected)
                        for seed in actions[index].candidate.remove
                    )
                )
                last_count = context.parameter_count(impact)
                if last_count > context.budget.max_params:
                    continue
            except PlanningError as error:
                rejected.append(f"ratio={ratio}: {error}")
                continue
            self.ratio = ratio
            self.rejections = tuple(rejected)
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
        self.rejections = tuple(rejected)
        details = "; ".join(self.rejections[-2:]) or "No executable quota reached the target"
        if exclusions:
            details += "; excluded example: " + exclusions[-1][1]
        raise PlanningError(
            f"Isomorphic parameter target not reached: {stop_reason}; "
            f"last executable count={last_count}, max_params={context.budget.max_params}. "
            f"{details}. No model was changed; this does not prove the target infeasible."
        )


def parse_args() -> argparse.Namespace:
    """Parse one-shot independent family pruning and optional fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.05, help="Whole-model parameter reduction"
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=8,
        help="Retained width alignment; 1 preserves exact family quotas",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=10_000,
        help="Maximum exact family-ratio breakpoints to examine",
    )
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
        "max_trials",
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
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
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
        need_train=options.finetune_epochs > 0,
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
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
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
    strategy = Isomorphic(max_trials=options.max_trials)
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
        ratio_trials=strategy.ratio_trials,
        rejected_allocations=strategy.rejections,
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
