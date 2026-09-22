"""Regularize pretrained-model groups until channel selection stabilizes."""

import argparse
import json
import math
from collections.abc import Callable, Generator, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    Granularity,
    Greedy,
    ParameterBudget,
    PlanningContext,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import GroupSquaredL2, SelectionWindow


def parse_args() -> argparse.Namespace:
    """Configure selection checks, regularization and ImageNet execution."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--data_dir", type=Path, help="Local ImageNet snapshot; default: HF cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--train_samples", type=int, default=0, help="0 selects the full training split (default)"
    )
    parser.add_argument(
        "--val_samples", type=int, default=0, help="0 selects the full validation split (default)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Sparse training and fine-tuning batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=256,
        help="Accuracy evaluation batch size",
    )
    parser.add_argument(
        "--train_workers",
        type=int,
        default=8,
        help="Training loader processes; 0 runs in the main process",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=8,
        help="Validation loader processes; 0 runs in the main process",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of whole-model parameters to remove (default: 0.05); not a channel ratio",
    )
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument(
        "--max_selection_checks",
        type=int,
        default=11,
        help="Maximum selection checks, including the initial check (default: 11); 1 skips sparse training",
    )
    parser.add_argument(
        "--selection_interval_steps",
        type=int,
        default=100,
        help="Optimizer steps between selection checks (default: 100)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2,
        help="Consecutive selection comparisons required; first stability test needs window + 1 checks (default: 2)",
    )
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument(
        "--sparse_loss_weight",
        type=float,
        default=1e-4,
        help="Final sparse-loss multiplier if training reaches its step limit (default: 1e-4)",
    )
    parser.add_argument(
        "--sparsity_schedule",
        choices=("linear", "cosine"),
        default="cosine",
        help="Increase the sparse-loss weight each optimizer step (default: cosine)",
    )
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/stability_pruning"))
    parser.add_argument(
        "--compile_latency",
        action="store_true",
        help="Use torch.compile only for latency measurement; training and accuracy evaluation stay eager",
    )
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "train_batch_size",
        "val_batch_size",
        "granularity",
        "max_selection_checks",
        "selection_interval_steps",
        "latency_repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "train_samples",
        "val_samples",
        "val_workers",
        "train_workers",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if options.window < 2:
        parser.error("--window must be at least 2")
    if not 0 <= options.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if not math.isfinite(options.sparse_loss_weight) or options.sparse_loss_weight < 0:
        parser.error("--sparse_loss_weight must be finite and nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def make_plan(pruner: Pruner, space: CandidateSpace, budget: ParameterBudget) -> PruningPlan:
    """Score producer channels and check stability on feasible aligned selections."""
    scores = {}
    for axis in space.channel_axes:
        weight = pruner.model.get_parameter(axis.tensor.paths[0])
        values = weight.detach().float().flatten(1).square().sum(1)
        values = values.cpu().tolist()  # One device transfer per producer axis.
        for candidate in space.candidates:
            if candidate.axis == axis:
                indices = candidate.remove[0].fully_selected_indices(0)
                scores[candidate.key] = sum(values[i] for i in indices)
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite pruning score")

    def score(context: PlanningContext, batch: Sequence[Candidate]) -> list[float]:
        """Look up the producer scores for this candidate batch."""
        return [scores[c.key] for c in batch]

    return pruner.plan(space, budget=budget, strategy=Greedy(score))


def train_steps(
    model: nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    steps: int,
    *,
    regularizer: Callable[[], torch.Tensor] | None = None,
    sparse_loss_weight: float = 0.0,
    sparsity_schedule: str = "cosine",
    completed_steps: int = 0,
    total_steps: int = 0,
    description: str = "Training",
) -> None:
    """Train a fixed number of batches without restarting the input iterator.

    A positive `total_steps` enables the sparse-weight ramp over the full
    search, using `completed_steps` as its offset. Fine-tuning omits the
    regularizer and ramp. Logs report the last weight and mean weighted loss.
    """
    if steps <= 0:
        raise ValueError("Training steps must be positive")
    if sparsity_schedule not in ("linear", "cosine"):
        raise ValueError("Unknown sparsity schedule")
    if (
        completed_steps < 0
        or total_steps < 0
        or (total_steps and completed_steps + steps > total_steps)
    ):
        raise ValueError("Training interval exceeds the sparse-weight schedule")
    model.train()
    total, sparse_total, weighted_sparse_total, count = 0.0, 0.0, 0.0, 0
    with tqdm(
        range(steps), desc=f"{description} ({device})", unit="batch", dynamic_ncols=True
    ) as progress:
        for step in progress:
            try:
                images, labels = next(batches)
            except StopIteration as error:
                raise ValueError("Training data ended before the requested steps") from error
            current_weight = sparse_loss_weight
            if total_steps:
                fraction = (completed_steps + step + 1) / total_steps
                if sparsity_schedule == "cosine":
                    fraction = 0.5 * (1 - math.cos(math.pi * fraction))
                current_weight *= fraction
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            task_loss = F.cross_entropy(model(images), labels)
            sparse_loss = regularizer() if regularizer is not None else task_loss.new_zeros(())
            (task_loss + current_weight * sparse_loss).backward()
            optimizer.step()
            total += task_loss.detach().item() * labels.numel()
            sparse_total += sparse_loss.detach().item() * labels.numel()
            weighted_sparse_total += current_weight * sparse_loss.detach().item() * labels.numel()
            count += labels.numel()
            progress.set_postfix(
                images=count,
                loss=f"{total / count:.4f}",
                sparse=f"{sparse_total / count:.4f}",
                sparse_loss_weight=f"{current_weight:.3g}",
                refresh=False,
            )
    if not count:
        raise ValueError("Cannot train on an empty loader")
    print(
        json.dumps(
            {
                "task_loss": total / count,
                "sparse_loss": sparse_total / count,
                "weighted_sparse_loss": weighted_sparse_total / count,
                "sparse_loss_weight": current_weight,
                "training_steps": completed_steps + steps,
            }
        ),
        flush=True,
    )


def main() -> None:
    """Regularize channel groups until selection stabilizes, then prune them."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    layers = (
        tuple(
            f"layer{stage}.{block}"
            for stage in range(1, 5)
            for block in range(len(getattr(model, f"layer{stage}")))
        )
        if options.model.startswith("resnet")
        else tuple(f"encoder.layers.{name}" for name, _ in model.encoder.layers.named_children())
    )
    print(f"Model: {options.model}; considering all {len(layers)} supported blocks", flush=True)
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=options.max_selection_checks > 1 or options.finetune_epochs > 0,
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
    config.update(
        weights=str(weights), dataset=dataset_info, layers=layers, max_params=budget.max_params
    )
    options.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def record(stage: str, **extra: object) -> None:
        """Evaluate the current model and persist this stage's measurements."""
        accuracy = evaluate(model, val_loader, options.device, description=f"{stage} evaluation")
        baseline = records[0]["top1"] if records else accuracy["top1"]
        row = {
            "stage": stage,
            **accuracy,
            "top1_delta_pp": accuracy["top1"] - baseline,
            **measure_model(model, example, options),
            **extra,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        (options.output / "metrics.json").write_text(
            json.dumps({"config": config, "stages": records}, indent=2) + "\n"
        )

    record("pretrained")
    suffix = "conv1.weight" if options.model.startswith("resnet") else "mlp.0.weight"
    paths = tuple(f"{layer}.{suffix}" for layer in layers)
    model.train(options.max_selection_checks > 1 or options.finetune_epochs > 0)
    targets = tuple(path.removesuffix(".weight") for path in paths)
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    window = SelectionWindow(options.window)
    stable, training_steps = False, 0
    total_steps = (options.max_selection_checks - 1) * options.selection_interval_steps

    def training_batches() -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Continue across checks; reopen the training stream only at its end."""
        if train_loader is None or not len(train_loader):
            raise ValueError("Cannot train on an empty training split")
        while True:
            yielded = False
            for batch in train_loader:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("Cannot train on an empty training split")

    batches = training_batches()

    for check in range(options.max_selection_checks):
        plan = make_plan(pruner, space, budget)
        selected = tuple(c for c in space.candidates if c.key in plan.selected)
        retained = {}
        for axis in space.channel_axes:
            removed = {
                index
                for candidate in selected
                if candidate.axis == axis
                for index in candidate.remove[0].fully_selected_indices(0)
            }
            retained[axis.tensor.paths[0]] = sorted(set(range(axis.tensor.shape[0])) - removed)
        similarity = window.update(retained)
        stable = similarity is not None and similarity >= options.threshold
        # A check measures the current weights. Do not train after the final
        # check: that would apply a selection never assessed for stability.
        if stable or check + 1 == options.max_selection_checks:
            break
        groups = pruner.parameter_groups(selected)
        regularizer = GroupSquaredL2(groups) if groups else None
        train_steps(
            model,
            batches,
            optimizer,
            options.device,
            options.selection_interval_steps,
            regularizer=regularizer,
            sparse_loss_weight=options.sparse_loss_weight,
            sparsity_schedule=options.sparsity_schedule,
            completed_steps=training_steps,
            total_steps=total_steps,
            description=f"Stability training before check {check + 2}",
        )
        training_steps += options.selection_interval_steps

    batches.close()
    selection_checks = check + 1
    record(
        "search_completed",
        stable=stable,
        selection_checks=selection_checks,
        training_steps=training_steps,
        similarity=similarity,
    )
    # Evaluation preserves weights and modes; regenerate from the same checked
    # coordinates immediately before apply to make execution preconditions clear.
    selection_report = plan.selection_report
    plan = pruner.plan_remove(
        [selection for candidate in selected for selection in candidate.remove]
    )
    model, _ = pruner.apply(plan)
    window.reset()  # Original channel coordinates no longer describe the compact model.
    record(
        "pruned",
        max_params=selection_report.max_params,
        before_params=selection_report.before_params,
        after_params=selection_report.after_params,
        target_met=selection_report.target_met,
        planning_trials=selection_report.trials,
        planning_limit_reached=selection_report.limit_reached,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        train_steps(
            model,
            iter(train_loader),
            optimizer,
            options.device,
            len(train_loader),
            description=f"Fine-tuning {epoch + 1}/{options.finetune_epochs}",
        )
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
            "algorithm": {
                "stable": stable,
                "selection_checks": selection_checks,
                "training_steps": training_steps,
                "window": window.state_dict(),
            },
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
