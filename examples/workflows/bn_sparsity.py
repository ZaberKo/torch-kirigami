"""Train pretrained ResNet BN scales with L1, prune channels, then fine-tune."""

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
    Granularity,
    Greedy,
    ParameterBudget,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.sparsity import ScaleL1


def parse_args():
    """Parse BN sparsity options for the supported ResNet models."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--model",
        choices=tuple(name for name in MODELS if name.startswith("resnet")),
        default="resnet18",
    )
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
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of whole-model parameters to remove (default: 0.05); not a channel ratio",
    )
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument("--sparse_epochs", type=int, default=1)
    parser.add_argument("--strength", type=float, default=1e-4)
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/bn_sparsity"))
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
        "sparse_epochs",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if not math.isfinite(options.strength) or options.strength < 0:
        parser.error("--strength must be finite and nonnegative")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    layers = tuple(
        f"layer{stage}.{block}"
        for stage in range(1, 5)
        for block in range(len(getattr(model, f"layer{stage}")))
    )
    print(f"Model: {options.model}; considering all {len(layers)} supported blocks", flush=True)
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=options.sparse_epochs > 0 or options.finetune_epochs > 0,
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
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(
        weights=str(weights), dataset=dataset_info, layers=layers, max_params=budget.max_params
    )
    records = []
    options.output.mkdir(parents=True, exist_ok=True)

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
    model.train(options.sparse_epochs > 0 or options.finetune_epochs > 0)
    graph = DependencyGraph.build(model, args=(example,))
    scale_paths = tuple(f"{layer}.bn1.weight" for layer in layers)
    regularizer = ScaleL1(graph, scale_paths) if options.sparse_epochs else None
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.sparse_epochs):
        task_total, sparse_total, count = 0.0, 0.0, 0
        model.train()
        with tqdm(
            train_loader,
            desc=f"BN sparse training {epoch + 1}/{options.sparse_epochs} ({options.device})",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for images, labels in progress:
                images, labels = images.to(options.device), labels.to(options.device)
                optimizer.zero_grad(set_to_none=True)
                task_loss = F.cross_entropy(model(images), labels)
                sparse_loss = regularizer()
                loss = task_loss + options.strength * sparse_loss
                loss.backward()
                optimizer.step()
                task_total += task_loss.detach().item() * labels.numel()
                sparse_total += sparse_loss.detach().item() * labels.numel()
                count += labels.numel()
                progress.set_postfix(
                    images=count,
                    loss=f"{task_total / count:.4f}",
                    sparse=f"{sparse_total / count:.4f}",
                    refresh=False,
                )
        if not count:
            raise ValueError("Cannot train on an empty ImageNet split")
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "task_loss": task_total / count,
                    "sparse_loss": sparse_total / count,
                    "strength": options.strength,
                }
            ),
            flush=True,
        )
    if options.sparse_epochs:
        record("sparse_trained")

    # Conv1 output removals propagate to its BN scales and the following conv2 inputs.
    axes = tuple(graph.parameter(f"{layer}.conv1.weight").axis(0) for layer in layers)
    targets = tuple(f"{layer}.conv1" for layer in layers)
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    scores = {}
    for axis, path in zip(axes, scale_paths, strict=True):
        values = model.get_parameter(path).detach().float().abs()
        values = values.cpu().tolist()  # One device transfer per producer axis.
        for candidate in space.candidates:
            if candidate.axis == axis:
                indices = candidate.remove[0].fully_selected_indices(0)
                scores[candidate.key] = sum(values[i] for i in indices)
    if not all(math.isfinite(score) for score in scores.values()):
        raise ValueError("Nonfinite BN scale score")

    def score(context, batch):
        return [scores[c.key] for c in batch]

    plan = pruner.plan(
        space,
        budget=budget,
        strategy=Greedy(score),
    )
    model, _ = pruner.apply(plan)
    record(
        "pruned",
        max_params=plan.selection_report.max_params,
        before_params=plan.selection_report.before_params,
        after_params=plan.selection_report.after_params,
        target_met=plan.selection_report.target_met,
        planning_trials=plan.selection_report.trials,
        planning_limit_reached=plan.selection_report.limit_reached,
    )

    # apply replaces Parameter objects; recreate the optimizer before fine-tuning.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        model.train()
        task_total, count = 0.0, 0
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
                task_total += loss.detach().item() * labels.numel()
                count += labels.numel()
                progress.set_postfix(images=count, loss=f"{task_total / count:.4f}", refresh=False)
        if not count:
            raise ValueError("Cannot fine-tune on an empty ImageNet split")
        print(
            json.dumps({"finetune_epoch": epoch + 1, "task_loss": task_total / count}), flush=True
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
            "algorithm": {"strength": options.strength},
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
