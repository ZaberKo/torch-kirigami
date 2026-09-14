"""Regularize pretrained-model groups until channel selection stabilizes."""

import argparse
import json
import math
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ChannelRatio,
    Granularity,
    Greedy,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import GroupSquaredL2, SelectionWindow


def parse_args():
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
        help="Accuracy evaluation and latency measurement batch size",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=0,
        help="Validation loader workers; training streams in the main process",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument(
        "--search_steps",
        type=int,
        default=3,
        help="Maximum selection checks; train one epoch between checks",
    )
    parser.add_argument(
        "--window", type=int, default=2, help="Selection history length, at least 2"
    )
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--strength", type=float, default=1e-4)
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
        "search_steps",
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
    if options.window < 2:
        parser.error("--window must be at least 2")
    if not 0 <= options.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if not math.isfinite(options.strength) or options.strength < 0:
        parser.error("--strength must be finite and nonnegative")
    if not 0 < options.ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 < ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def make_plan(pruner, space, ratio):
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

    def score(context, batch):
        return [scores[c.key] for c in batch]

    return pruner.plan(space, budget=ChannelRatio(ratio), strategy=Greedy(score))


def train_epoch(
    model, loader, optimizer, device, *, regularizer=None, strength=0.0, description="Training"
):
    """Add the selected-group penalty to task loss before backward and SGD."""
    model.train()
    total, sparse_total, count = 0.0, 0.0, 0
    with tqdm(
        loader, desc=f"{description} ({device})", unit="batch", dynamic_ncols=True
    ) as progress:
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            task_loss = F.cross_entropy(model(images), labels)
            sparse_loss = regularizer() if regularizer is not None else task_loss.new_zeros(())
            (task_loss + strength * sparse_loss).backward()
            optimizer.step()
            total += task_loss.detach().item() * labels.numel()
            sparse_total += sparse_loss.detach().item() * labels.numel()
            count += labels.numel()
            progress.set_postfix(
                images=count,
                loss=f"{total / count:.4f}",
                sparse=f"{sparse_total / count:.4f}",
                refresh=False,
            )
    if not count:
        raise ValueError("Cannot train on an empty loader")
    print(
        json.dumps(
            {
                "task_loss": total / count,
                "sparse_loss": sparse_total / count,
                "strength": strength,
            }
        ),
        flush=True,
    )


def main():
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
        need_train=options.search_steps > 1 or options.finetune_epochs > 0,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = (
        DataLoader(train, batch_size=options.train_batch_size, generator=generator)
        if train is not None
        else None
    )
    val_loader = DataLoader(
        validation, batch_size=options.val_batch_size, num_workers=options.val_workers
    )
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(weights=str(weights), dataset=dataset_info, layers=layers)
    options.output.mkdir(parents=True, exist_ok=True)
    records = []

    def record(stage, **extra):
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
    model.train(options.search_steps > 1 or options.finetune_epochs > 0)
    targets = tuple(path.removesuffix(".weight") for path in paths)
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    original_width = sum(axis.tensor.shape[axis.dim] for axis in space.channel_axes)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    window = SelectionWindow(options.window)
    stable, training_epochs = False, 0

    for check in range(options.search_steps):
        plan = make_plan(pruner, space, options.ratio)
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
        if stable or check + 1 == options.search_steps:
            break
        groups = pruner.parameter_groups(selected)
        regularizer = GroupSquaredL2(groups) if groups else None
        train_epoch(
            model,
            train_loader,
            optimizer,
            options.device,
            regularizer=regularizer,
            strength=options.strength * (check + 1) / (options.search_steps - 1),
            description=f"Stability training {training_epochs + 1}",
        )
        training_epochs += 1

    selection_checks = check + 1
    record(
        "search_completed",
        stable=stable,
        selection_checks=selection_checks,
        training_epochs=training_epochs,
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
    current_width = sum(model.get_parameter(path).shape[0] for path in paths)
    record(
        "pruned",
        target_ratio=options.ratio,
        actual_ratio=1 - current_width / original_width,
        target=selection_report.targets,
        removed=selection_report.removed,
        shortfall=selection_report.shortfall,
        planning_trials=selection_report.trials,
        planning_limit_reached=selection_report.limit_reached,
    )
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
            "algorithm": {
                "stable": stable,
                "selection_checks": selection_checks,
                "training_epochs": training_epochs,
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
