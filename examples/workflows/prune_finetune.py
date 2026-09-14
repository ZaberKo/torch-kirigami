"""Pretrained model -> magnitude/Taylor pruning -> optional ImageNet fine-tuning."""

import argparse
import json
import math
import time
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


def parse_args():
    """Parse options for one pruning pass and optional fine-tuning."""
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
        help="Training and Taylor calibration batch size",
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
    parser.add_argument("--metric", choices=("magnitude", "taylor"), default="magnitude")
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/prune_finetune"))
    parser.add_argument(
        "--compile_latency",
        action="store_true",
        help="Use torch.compile only for latency measurement; training and accuracy evaluation stay eager",
    )
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in ("train_batch_size", "val_batch_size", "granularity", "latency_repetitions"):
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
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with tqdm(
            total=1, desc=f"Taylor calibration ({device})", unit="batch", dynamic_ncols=True
        ) as progress:
            images, labels = next(iter(loader))
            loss = F.cross_entropy(model(images.to(device)), labels.to(device))
            loss.backward()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=False)
            progress.update()
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
        # Transfer all channel scores together; per-candidate .item() synchronizes CUDA.
        values = values.cpu().tolist()
        for candidate in space.candidates:
            if candidate.axis == axis:
                indices = candidate.remove[0].fully_selected_indices(0)
                scores[candidate.key] = sum(values[i] for i in indices)
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite pruning score")

    def score(context, batch):
        return [scores[c.key] for c in batch]

    return pruner.plan(space, budget=ChannelRatio(ratio), strategy=Greedy(score))


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
    print(
        f"Model: {options.model}; considering all {len(layers)} supported blocks; "
        f"parameters on {next(model.parameters()).device}",
        flush=True,
    )
    print("Loading local ImageNet data (CPU decoding and preprocessing)", flush=True)
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=options.metric == "taylor" or options.finetune_epochs > 0,
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
    records = []
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(weights=str(weights), dataset=dataset_info, layers=layers)
    options.output.mkdir(parents=True, exist_ok=True)

    def record(stage, **extra):
        print(f"[{stage}] Evaluating {len(validation)} images on {options.device}", flush=True)
        started = time.perf_counter()
        accuracy = evaluate(model, val_loader, options.device, description=f"{stage} evaluation")
        print(f"[{stage}] Evaluation completed in {time.perf_counter() - started:.1f}s", flush=True)
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
    targets = tuple(path.removesuffix(".weight") for path in paths)
    print(f"Capturing dependency graph; sample forward on {options.device}", flush=True)
    started = time.perf_counter()
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    print(
        f"Graph and {len(space.candidates)} candidates ready in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    original_width = sum(axis.tensor.shape[axis.dim] for axis in space.channel_axes)
    if options.metric == "taylor":
        collect_task_gradients(model, train_loader, options.device)
    print("Planning pruning: dependency propagation and constraint search run on CPU", flush=True)
    started = time.perf_counter()
    plan = make_plan(pruner, space, options.ratio, options.metric)
    print(
        f"Plan completed in {time.perf_counter() - started:.1f}s; "
        f"{plan.selection_report.trials} joint trials; applying on {options.device}",
        flush=True,
    )
    model, _ = pruner.apply(plan)
    current_width = sum(model.get_parameter(path).shape[0] for path in paths)
    record(
        "pruned",
        target_ratio=options.ratio,
        actual_ratio=1 - current_width / original_width,
        target=plan.selection_report.targets,
        removed=plan.selection_report.removed,
        shortfall=plan.selection_report.shortfall,
        planning_trials=plan.selection_report.trials,
        planning_limit_reached=plan.selection_report.limit_reached,
    )

    # Parameters were replaced by apply; create the optimizer only afterwards.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        print(
            f"Fine-tuning epoch {epoch + 1}: {len(train)} images on {options.device}; "
            "training data decoded in the main process",
            flush=True,
        )
        model.train()
        total, count = 0.0, 0
        with tqdm(
            train_loader,
            desc=f"Fine-tuning {epoch + 1}/{options.finetune_epochs} ({options.device})",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for images, labels in progress:
                images, labels = images.to(options.device), labels.to(options.device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.detach().item() * labels.numel()
                count += labels.numel()
                progress.set_postfix(images=count, loss=f"{total / count:.4f}", refresh=False)
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
