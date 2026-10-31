"""Prune with the fixed Pruner-Zero formula and optionally recover with task loss.

Reference: Pruner-Zero, ICML 2024, https://arxiv.org/abs/2406.02924.
Use the published best formula W² * minmax(abs(G)), not evolutionary search.
Sample-mean cross-entropy gradients and summing dependency-coupled matrix/kernel
scores adapt the LLM weight metric to structural vision pruning. Normalization
uses each complete weight tensor. Constant gradient magnitudes score zero.
ResNet block internals and ViT MLP hidden axes are eligible; interfaces stay fixed.
No teacher or knowledge distillation is used. This is not a benchmark reproduction.
"""

import argparse
import json
import math
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import EncoderBlock

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    EvolvedMetric,
    Granularity,
    Greedy,
    ParameterBudget,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


def collect_task_gradients(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    batches: int,
) -> dict[str, int]:
    """Leave dataset-mean CE gradients, restoring modes and clearing failed work.

    Clear old gradients. Evaluate with dropout disabled and BN buffers unchanged.
    Accumulate summed losses, divide by the actual sample count, and never step
    an optimizer. A positive limit requires exactly that many available batches;
    zero consumes all batches. A failure leaves no partial calibration gradients.
    """
    if type(batches) is not int or batches < 0:
        raise ValueError("Calibration batches must be a nonnegative integer")
    modes = [(module, module.training) for module in model.modules()]
    model.zero_grad(set_to_none=True)
    count = completed = 0
    try:
        model.eval()
        with torch.enable_grad():
            for images, labels in islice(loader, batches) if batches else loader:
                labels = labels.to(device)
                loss = F.cross_entropy(model(images.to(device)), labels, reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("Calibration loss must be finite")
                loss.backward()
                count += labels.numel()
                completed += 1
        if not count or (batches and completed != batches):
            raise ValueError("Not enough calibration batches")
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
    except BaseException:
        model.zero_grad(set_to_none=True)
        raise
    finally:
        for module, training in modes:
            module.training = training
    return {"samples": count, "batches": completed}


def parse_args() -> argparse.Namespace:
    """Configure calibration, whole-model budget, recovery and measurements."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="vit_b_16")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.15,
        help="Whole-model parameter reduction, not weight sparsity",
    )
    parser.add_argument("--granularity", type=int, default=8)
    parser.add_argument(
        "--calibration_batches", type=int, default=8, help="0 uses all training batches"
    )
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("runs/pruner_zero"))
    parser.add_argument("--compile_latency", action="store_true")
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
        "calibration_batches",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1:
        parser.error("Require 0 <= pruning_ratio < 1")
    if not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require positive finite lr")
    if not math.isfinite(options.weight_decay) or options.weight_decay < 0:
        parser.error("Require finite nonnegative weight_decay")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or install CUDA PyTorch")
    return options


def main() -> None:
    """Evaluate, calibrate, plan, compact, recover with CE and verify restoration."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    targets = []
    for path, block in model.named_modules():
        if type(block) is BasicBlock:
            targets.append(f"{path}.conv1")
        elif type(block) is Bottleneck:
            targets.extend((f"{path}.conv1", f"{path}.conv2"))
        elif type(block) is EncoderBlock:
            targets.append(f"{path}.mlp.0")
    if not targets:
        raise ValueError("No supported block-internal axes")
    weights = MODELS[options.model][1]
    train, validation, info = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    if train is None:
        raise ValueError("Pruner-Zero requires training data for gradients")
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train,
        batch_size=options.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=options.train_workers,
        multiprocessing_context="spawn" if options.train_workers else None,
    )
    val_loader = DataLoader(
        validation,
        batch_size=options.val_batch_size,
        num_workers=options.val_workers,
        multiprocessing_context="spawn" if options.val_workers else None,
    )
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    config.update(
        dataset=info,
        weights=str(weights),
        targets=targets,
        formula="W^2 * minmax(abs(G))",
        gradient_reduction="sample_mean_ce",
    )
    options.output.mkdir(parents=True, exist_ok=True)
    records = []

    def record(stage, **extra):
        accuracy = evaluate(model, val_loader, options.device, description=stage)
        row = {"stage": stage, **accuracy, **measure_model(model, example, options), **extra}
        row["top1_delta_pp"] = accuracy["top1"] - (
            records[0]["top1"] if records else accuracy["top1"]
        )
        records.append(row)
        (options.output / "metrics.json").write_text(
            json.dumps({"config": config, "stages": records}, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)

    record("pretrained")
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    config["calibration"] = collect_task_gradients(
        model, train_loader, options.device, options.calibration_batches
    )
    plan = pruner.plan(space, budget=budget, strategy=Greedy(EvolvedMetric()))
    model, _ = pruner.apply(plan)
    model.zero_grad(set_to_none=True)
    record(
        "pruned",
        target_met=plan.selection_report.target_met,
        before_params=plan.selection_report.before_params,
        after_params=plan.selection_report.after_params,
    )
    save_checkpoint(model, options.output / "pruned.pt")
    # Structural application replaces Parameters. Build the optimizer afterwards.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.lr, weight_decay=options.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, options.finetune_epochs)
    )
    for epoch in range(options.finetune_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images.to(options.device)), labels.to(options.device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"CE recovery epoch {epoch + 1}/{options.finetune_epochs}", flush=True)
    model.eval()
    if options.finetune_epochs:
        record("finetuned")
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
            "scheduler": scheduler.state_dict(),
            "config": config,
            "algorithm": "pruner_zero_fixed_metric",
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
        },
        options.output / "training.pt",
    )


if __name__ == "__main__":
    main()
