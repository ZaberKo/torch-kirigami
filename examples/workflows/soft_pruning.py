"""Train with group zeroing or norm decay, then physically remove those groups."""

import argparse
import json
import math
from collections.abc import Sequence
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
    ParameterGroup,
    PlanningContext,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import GroupLasso, set_group_norms_, zero_groups_


def parse_args() -> argparse.Namespace:
    """Parse soft-projection settings and ImageNet execution options."""
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
        help="Projection training and fine-tuning batch size",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=256,
        help="Accuracy evaluation and latency measurement batch size",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=0,
        help="Validation loader workers; training streams in the main process",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of whole-model parameters to remove (default: 0.05); not a channel ratio",
    )
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument("--operation", choices=("zero", "decay"), default="decay")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--projection_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/soft_pruning"))
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
        "cycles",
        "projection_epochs",
        "latency_repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "train_samples",
        "val_samples",
        "val_workers",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def make_plan(pruner: Pruner, space: CandidateSpace, budget: ParameterBudget) -> PruningPlan:
    """Score producer channels and enforce alignment before soft projection."""
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    *,
    groups: Sequence[ParameterGroup] = (),
    operation: str | None = None,
    initial_norm: float = 0.0,
    projection_epoch: int = 0,
    projection_epochs: int = 1,
    description: str = "Training",
) -> None:
    """Take SGD steps, optionally projecting the selected union after each step.

    Momentum is deliberately retained. An unprojected epoch between cycles
    lets previously zeroed regions regrow before the next magnitude selection.
    """
    model.train()
    total, count = 0.0, 0
    with tqdm(
        loader, desc=f"{description} ({device})", unit="batch", dynamic_ncols=True
    ) as progress:
        for batch, (images, labels) in enumerate(progress, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            if groups and operation == "zero":
                zero_groups_(groups)
            elif groups and operation == "decay":
                fraction = (projection_epoch * len(loader) + batch) / (
                    projection_epochs * len(loader)
                )
                set_group_norms_(groups, (initial_norm * (1 - fraction),))
            total += loss.detach().item() * labels.numel()
            count += labels.numel()
            progress.set_postfix(images=count, loss=f"{total / count:.4f}", refresh=False)
    if not count:
        raise ValueError("Cannot train on an empty loader")
    print(json.dumps({"task_loss": total / count, "projection": operation}), flush=True)


def main() -> None:
    """Train with soft group projection, then physically prune the selected groups."""
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
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(train, batch_size=options.train_batch_size, generator=generator)
    val_loader = DataLoader(
        validation, batch_size=options.val_batch_size, num_workers=options.val_workers
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
    model.train()
    targets = tuple(path.removesuffix(".weight") for path in paths)
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    train_epoch(model, train_loader, optimizer, options.device)
    record("warmup_trained")

    for cycle in range(options.cycles):
        selection_plan = make_plan(pruner, space, budget)
        selected = tuple(c for c in space.candidates if c.key in selection_plan.selected)
        # Project one union, not each candidate separately: their dependency
        # regions may overlap and must not be scaled multiple times.
        impact = pruner.impact(selected)
        groups = (ParameterGroup(graph, impact.parameters),) if impact.parameters else ()
        initial_norm = GroupLasso(groups)().detach().item() if groups else 0.0
        for epoch in range(options.projection_epochs):
            train_epoch(
                model,
                train_loader,
                optimizer,
                options.device,
                groups=groups,
                operation=options.operation,
                initial_norm=initial_norm,
                projection_epoch=epoch,
                projection_epochs=options.projection_epochs,
                description=f"Cycle {cycle + 1} projection {epoch + 1}/{options.projection_epochs}",
            )
        record(f"cycle_{cycle + 1}_projected")
        if cycle + 1 < options.cycles:
            train_epoch(
                model,
                train_loader,
                optimizer,
                options.device,
                description=f"Cycle {cycle + 1} recovery",
            )

    # Preserve the projected coordinates and validate them against current state.
    selection_report = selection_plan.selection_report
    plan = pruner.plan_remove(
        [selection for candidate in selected for selection in candidate.remove]
    )
    model, _ = pruner.apply(plan)
    record(
        "pruned",
        max_params=selection_report.max_params,
        before_params=selection_report.before_params,
        after_params=selection_report.after_params,
        target_met=selection_report.target_met,
        planning_trials=selection_report.trials,
        planning_limit_reached=selection_report.limit_reached,
    )
    # Physical compaction replaces Parameter objects; momentum cannot be reused.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        train_epoch(
            model,
            train_loader,
            optimizer,
            options.device,
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
            "algorithm": {"operation": options.operation, "cycles": options.cycles},
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
