"""Prune complete attention heads and FFN hidden channels in torchvision ViT.

Attention is explicitly converted before graph capture. Its external embedding
width stays fixed while Q/K/V projection rows and output-projection columns
shrink together. This is a magnitude baseline, not a paper reproduction.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, HeadPrunableViT, make_head_prunable_model
from model_metrics import measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph, TensorRef
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


def candidate_space(pruner: Pruner) -> CandidateSpace:
    """Combine FFN channels with the SDPA query's explicit logical head axis.

    Existing SDPA and reshape relationships propagate one head to matching Q/K/V
    projection rows, biases and output-projection columns. Using a logical head
    axis also makes GroupMagnitude normalize over heads, rather than scalar
    projection rows. FFN alignment is configured separately on the Pruner.
    """
    targets = [
        f"encoder.layers.{name}.mlp.0" for name, _ in pruner.model.encoder.layers.named_children()
    ]
    ffn = pruner.discover_candidates(targets=targets)
    candidates, axes = list(ffn.candidates), list(ffn.channel_axes)
    for operation in pruner.graph.operations():
        if operation.node.target is not F.scaled_dot_product_attention:
            continue
        query = operation.argument("query", 0)
        if not isinstance(query, TensorRef) or len(query.shape) != 4:
            raise ValueError("Expected an explicit [batch, heads, tokens, features] query")
        axis = query.axis(1)
        for head in range(query.shape[1]):
            candidates.append(
                Candidate(f"{operation.node.name}:head:{head:04d}", (axis.select([head]),), axis)
            )
        axes.append(axis)
    return CandidateSpace(tuple(candidates), tuple(axes), exclusions=ffn.exclusions)


def parse_args() -> argparse.Namespace:
    """Parse the independent ViT head/FFN pruning and fine-tuning workflow."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=("vit_b_16", "vit_b_32"), default="vit_b_16")
    parser.add_argument("--data_dir", type=Path, help="Local ImageNet snapshot; default: HF cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--train_samples", type=int, default=0, help="0 uses the full training split"
    )
    parser.add_argument(
        "--val_samples", type=int, default=0, help="0 uses the full validation split"
    )
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of whole-model parameters to remove; not a head/channel ratio",
    )
    parser.add_argument(
        "--granularity",
        type=int,
        default=8,
        help="Retained FFN width alignment; attention always removes complete heads",
    )
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/vit_head_pruning"))
    parser.add_argument(
        "--compile_latency",
        action="store_true",
        help="Use torch.compile only for latency; accuracy and fine-tuning stay eager",
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
        "train_workers",
        "val_workers",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install CUDA PyTorch or pass --device cpu")
    return options


def structure(model: HeadPrunableViT) -> dict[str, dict[str, int]]:
    """Report head counts and FFN widths without implying both must be pruned."""
    return {
        name: {
            "heads": block.self_attention.num_heads,
            "head_dim": block.self_attention.head_dim,
            "ffn_width": block.mlp[0].out_features,
        }
        for name, block in model.encoder.layers.named_children()
    }


def main() -> None:
    """Evaluate, jointly prune attention/FFN, fine-tune, and verify restoration."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_head_prunable_model(options.model).to(options.device).eval()
    weights = MODELS[options.model][1]
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    print(f"Model: {options.model}; complete heads + FFN; device={options.device}", flush=True)
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
    config = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(options).items()
    }
    config.update(
        weights=str(weights),
        dataset=dataset_info,
        max_params=budget.max_params,
        pruning_scope="attention_heads_and_ffn",
        metric="group_magnitude",
    )
    records: list[dict[str, Any]] = []
    options.output.mkdir(parents=True, exist_ok=True)

    def record(stage: str, **extra: object) -> None:
        """Persist accuracy, resource measurements and actual per-block widths."""
        accuracy = evaluate(model, val_loader, options.device, description=f"{stage} evaluation")
        baseline = records[0]["top1"] if records else accuracy["top1"]
        row = {
            "stage": stage,
            **accuracy,
            "top1_delta_pp": accuracy["top1"] - baseline,
            **measure_model(model, example, options),
            "structure": structure(model),
            **extra,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        (options.output / "metrics.json").write_text(
            json.dumps({"config": config, "stages": records}, indent=2) + "\n"
        )

    record("pretrained")
    print("Capturing FX graph and constructing whole-head/FFN candidates", flush=True)
    graph = DependencyGraph.build(model, args=(example,))
    ffn_targets = [
        f"encoder.layers.{name}.mlp.0" for name, _ in model.encoder.layers.named_children()
    ]
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(ffn_targets, options.granularity)),
    )
    space = candidate_space(pruner)
    config["candidate_axes"] = [
        {"tensor": axis.tensor.paths[0] if axis.tensor.paths else axis.tensor.id, "dim": axis.dim}
        for axis in space.channel_axes
    ]
    config["discovery_exclusions"] = space.exclusions
    print(
        f"Planning {len(space.candidates)} candidates on CPU; weight scoring on {options.device}",
        flush=True,
    )
    started = time.perf_counter()
    plan = pruner.plan(space, budget=budget, strategy=Greedy(GroupMagnitude(p=2)))
    print(f"Plan completed in {time.perf_counter() - started:.1f}s", flush=True)
    model, _ = pruner.apply(plan)
    report = plan.selection_report
    record(
        "pruned",
        max_params=report.max_params,
        before_params=report.before_params,
        after_params=report.after_params,
        target_met=report.target_met,
        planning_trials=report.trials,
        planning_limit_reached=report.limit_reached,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        model.train()
        total, count = 0.0, 0
        with tqdm(
            train_loader,
            desc=f"Fine-tuning {epoch + 1}/{options.finetune_epochs} ({options.device})",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for images, labels in progress:
                images = images.to(options.device, non_blocking=True)
                labels = labels.to(options.device, non_blocking=True)
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
        make_head_prunable_model(options.model, pretrained=False),
        options.output / "model.pt",
        map_location=options.device,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "config": config,
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
