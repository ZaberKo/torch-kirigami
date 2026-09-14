"""Pretrained model -> magnitude/Taylor pruning -> optional ImageNet fine-tuning."""

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
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


def parse_args():
    """Parse options for one pruning pass and optional fine-tuning."""
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
    parser.add_argument("--metric", choices=("magnitude", "taylor"), default="magnitude")
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/prune_finetune"))
    parser.add_argument("--compile", action="store_true", help="Measure compiled inference")
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in ("batch_size", "threads", "granularity", "benchmark_batch_size", "repetitions"):
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


def collect_task_gradients(model, loader, device):
    """Collect task-only gradients without updating weights, BN statistics or modes."""
    if loader is None:
        raise ValueError("Taylor requires a separate ImageNet training batch")
    model.zero_grad(set_to_none=True)
    images, labels = next(iter(loader))
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        F.cross_entropy(model(images.to(device)), labels.to(device)).backward()
    finally:
        for module, training in modes:
            module.training = training


def make_plan(pruner, space, ratio, metric):
    """Score producer channels; Greedy enforces joint constraints and alignment."""
    scores = {}
    for axis in space.channel_axes:
        weight = pruner.model.get_parameter(axis.tensor.paths[0])
        if metric == "taylor":
            if weight.grad is None:
                raise ValueError("Collect task-only gradients before Taylor selection")
            values = (
                (weight.detach().float() * weight.grad.detach().float()).abs().flatten(1).sum(1)
            )
        else:
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
        need_train=options.metric == "taylor" or options.finetune_epochs > 0,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    model = make_model(options.model).to(options.device).eval()
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = (
        DataLoader(train, batch_size=options.batch_size, generator=generator)
        if train is not None
        else None
    )
    val_loader = DataLoader(validation, batch_size=options.batch_size, num_workers=options.workers)
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    records = []
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(weights=str(weights), dataset=dataset_info, layers=layers)
    options.output.mkdir(parents=True, exist_ok=True)

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
    targets = tuple(path.removesuffix(".weight") for path in paths)
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    original_width = sum(axis.tensor.shape[axis.dim] for axis in space.channel_axes)
    if options.metric == "taylor":
        collect_task_gradients(model, train_loader, options.device)
    plan = make_plan(pruner, space, options.ratio, options.metric)
    model, _ = pruner.apply(plan)
    current_width = sum(model.get_parameter(path).shape[0] for path in paths)
    record(
        "pruned",
        target_ratio=options.ratio,
        actual_ratio=1 - current_width / original_width,
        target=plan.selection_report.targets,
        removed=plan.selection_report.removed,
        shortfall=plan.selection_report.shortfall,
    )

    # Parameters were replaced by apply; create the optimizer only afterwards.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        model.train()
        total, count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(options.device), labels.to(options.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * labels.numel()
            count += labels.numel()
        if not count:
            raise ValueError("Cannot fine-tune on an empty training split")
        print(json.dumps({"epoch": epoch + 1, "task_loss": total / count}), flush=True)
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
            "algorithm": {"metric": options.metric},
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
