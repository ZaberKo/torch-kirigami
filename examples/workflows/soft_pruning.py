"""Train with group zeroing or norm decay, then physically remove those groups."""

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

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ChannelRatio,
    Granularity,
    Greedy,
    ParameterGroup,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import GroupLasso, set_group_norms_, zero_groups_


def parse_args():
    """Parse soft-projection settings and ImageNet execution options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--layers", help="Comma-separated blocks, or all; default: first block")
    parser.add_argument("--data-dir", type=Path, help="Local ImageNet snapshot; default: HF cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train-samples", type=int, default=512, help="0 selects the full split")
    parser.add_argument("--val-samples", type=int, default=0, help="0 selects the full split")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument("--operation", choices=("zero", "decay"), default="decay")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--projection-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/soft_pruning"))
    parser.add_argument("--compile", action="store_true", help="Measure compiled inference")
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "batch_size",
        "threads",
        "granularity",
        "cycles",
        "projection_epochs",
        "benchmark_batch_size",
        "repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_samples", "val_samples", "workers", "finetune_epochs", "warmup"):
        if getattr(options, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 < options.ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 < ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def make_plan(pruner, space, ratio):
    """Score producer channels and enforce alignment before soft projection."""
    scores = {}
    for axis in space.channel_axes:
        weight = pruner.model.get_parameter(axis.tensor.paths[0])
        values = weight.detach().float().flatten(1).square().sum(1)
        for candidate in space.candidates:
            if candidate.axis == axis:
                indices = candidate.remove[0].fully_selected_indices(0)
                scores[candidate.key] = values[list(indices)].sum().item()
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite pruning score")

    def score(context, batch):
        return [scores[c.key] for c in batch]

    return pruner.plan(space, budget=ChannelRatio(ratio), strategy=Greedy(score))


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    groups=(),
    operation=None,
    initial_norm=0.0,
    projection_epoch=0,
    projection_epochs=1,
):
    """Take SGD steps, optionally projecting the selected union after each step.

    Momentum is deliberately retained. An unprojected epoch between cycles
    lets previously zeroed regions regrow before the next magnitude selection.
    """
    model.train()
    total, count = 0.0, 0
    for batch, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        if groups and operation == "zero":
            zero_groups_(groups)
        elif groups and operation == "decay":
            progress = (projection_epoch * len(loader) + batch) / (projection_epochs * len(loader))
            set_group_norms_(groups, (initial_norm * (1 - progress),))
        total += loss.detach().item() * labels.numel()
        count += labels.numel()
    if not count:
        raise ValueError("Cannot train on an empty loader")
    print(json.dumps({"task_loss": total / count, "projection": operation}), flush=True)


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
    torch.set_num_threads(options.threads)
    blocks = (
        tuple(f"layer{stage}.{block}" for stage in range(1, 5) for block in range(2))
        if options.model == "resnet18"
        else tuple(f"encoder.layers.encoder_layer_{index}" for index in range(12))
    )
    if options.layers == "all":
        layers = blocks
    elif options.layers:
        layers = tuple(options.layers.split(","))
    else:
        layers = blocks[:1]
    if len(set(layers)) != len(layers) or not set(layers).issubset(blocks):
        raise ValueError("--layers must name distinct supported blocks or all")
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    model = make_model(options.model).to(options.device).eval()
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(train, batch_size=options.batch_size, generator=generator)
    val_loader = DataLoader(validation, batch_size=options.batch_size, num_workers=options.workers)
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(weights=str(weights), dataset=dataset_info, layers=layers)
    options.output.mkdir(parents=True, exist_ok=True)
    records = []

    def record(stage, **extra):
        accuracy = evaluate(model, val_loader, options.device, progress_every=100)
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
    suffix = "conv1.weight" if options.model == "resnet18" else "mlp.0.weight"
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
    original_width = sum(axis.tensor.shape[axis.dim] for axis in space.channel_axes)
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    train_epoch(model, train_loader, optimizer, options.device)
    record("warmup_trained")

    for cycle in range(options.cycles):
        selection_plan = make_plan(pruner, space, options.ratio)
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
            )
        record(f"cycle_{cycle + 1}_projected")
        if cycle + 1 < options.cycles:
            train_epoch(model, train_loader, optimizer, options.device)

    # Preserve the projected coordinates and validate them against current state.
    selection_report = selection_plan.selection_report
    plan = pruner.plan_remove(
        [selection for candidate in selected for selection in candidate.remove]
    )
    model, _ = pruner.apply(plan)
    current_width = sum(model.get_parameter(path).shape[0] for path in paths)
    record(
        "pruned",
        target_ratio=options.ratio,
        actual_ratio=1 - current_width / original_width,
        target=selection_report.targets,
        removed=selection_report.removed,
        shortfall=selection_report.shortfall,
    )
    # Physical compaction replaces Parameter objects; momentum cannot be reused.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for _ in range(options.finetune_epochs):
        train_epoch(model, train_loader, optimizer, options.device)
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
