"""DVBP + OB²C: spectral MLP ranking, structural pruning and local reconstruction.

Paper: https://arxiv.org/abs/2608.17657.
Use torchvision ViT/ConvNeXt dense MLP pairs discovered by the VBP workflow.
Calibrate centered covariance online, use image counts for MP thresholds, blend
raw/denoised variance and reconstruct retained consumer weights and biases.
Default repair uses denoised covariance as in the authors' code; `raw` uses the
paper's centered-covariance objective. Damping and a ridge floor stabilize solves.
`pruning_ratio` controls the whole-model parameter budget; `score_ratio` is the
separate hybrid-score mixing coefficient. Neither implies a per-layer ratio.
Optional recovery uses only cross-entropy. No distillation or paper benchmark
reproduction is claimed. Full covariance storage remains quadratic in MLP width.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MLP_MODELS, make_mlp_model
from model_metrics import measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from variance_pruning import discover_mlps, vbp_operators

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    DenoisedVariance,
    Granularity,
    Greedy,
    ParameterBudget,
    Pruner,
    apply_ob2c_updates,
    collect_covariances,
    denoise_covariance,
    load_checkpoint,
    prepare_ob2c_updates,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Configure covariance calibration, scoring, reconstruction and CE recovery."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MLP_MODELS), default="vit_b_16")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.15, help="Whole-model parameter reduction"
    )
    parser.add_argument(
        "--score_ratio",
        type=float,
        default=0.5,
        help="Hybrid variance mixing coefficient, independent of budget",
    )
    parser.add_argument("--granularity", type=int, default=8)
    parser.add_argument(
        "--calibration_batches", type=int, default=32, help="0 uses all training batches"
    )
    parser.add_argument("--statistics_dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--statistics_device", choices=("cpu", "model"), default="cpu")
    parser.add_argument("--covariance_chunk_size", type=int, default=2048)
    parser.add_argument("--damping", type=float, default=0.1)
    parser.add_argument("--ridge_floor", type=float, default=1e-8)
    parser.add_argument("--repair_covariance", choices=("raw", "denoised"), default="denoised")
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("runs/dvbp"))
    parser.add_argument("--compile_latency", action="store_true")
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "train_batch_size",
        "val_batch_size",
        "granularity",
        "latency_repetitions",
        "covariance_chunk_size",
    ):
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
    for name in ("pruning_ratio", "score_ratio"):
        if not 0 <= getattr(options, name) < 1:
            parser.error(f"--{name} must lie in [0,1)")
    for name in ("weight_decay", "damping", "ridge_floor"):
        if not math.isfinite(getattr(options, name)) or getattr(options, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    if not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require positive finite lr")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or install CUDA PyTorch")
    return options


def main() -> None:
    """Calibrate, compensate a structural plan, recover with CE and save values."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_mlp_model(options.model).to(options.device).eval()
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    graph = DependencyGraph.build(model, args=(example,), operators=vbp_operators())
    pairs, exclusions = discover_mlps(graph)
    if not pairs:
        raise ValueError(f"No supported independent dense MLPs: {exclusions}")
    weights = MLP_MODELS[options.model][1]
    train, validation, info = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    if train is None:
        raise ValueError("DVBP requires training calibration images")
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
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    config.update(dataset=info, weights=str(weights), mlps=pairs, excluded_producers=exclusions)
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
    moments = collect_covariances(
        model,
        tuple(pairs.values()),
        train_loader,
        options.device,
        options.calibration_batches,
        dtype=getattr(torch, options.statistics_dtype),
        statistics_device="cpu" if options.statistics_device == "cpu" else options.device,
        chunk_size=options.covariance_chunk_size,
    )
    statistics = {}
    # Release each raw accumulator after decomposition; no activation history is kept.
    for path in tuple(moments):
        statistics[path] = denoise_covariance(
            moments.pop(path),
            options.score_ratio,
            damping=options.damping,
            ridge_floor=options.ridge_floor,
            repair_covariance=options.repair_covariance,
        )
    config["calibration"] = {path: stats.report() for path, stats in statistics.items()}
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(pairs, options.granularity)),
    )
    space = pruner.discover_candidates(targets=tuple(pairs))
    plan = pruner.plan(
        space, budget=budget, strategy=Greedy(DenoisedVariance(graph, pairs, statistics))
    )
    updates = prepare_ob2c_updates(graph, plan, pairs, statistics)
    model, _ = pruner.apply(plan)
    apply_ob2c_updates(model, updates)
    del statistics, updates
    record(
        "pruned",
        target_met=plan.selection_report.target_met,
        before_params=plan.selection_report.before_params,
        after_params=plan.selection_report.after_params,
    )
    save_checkpoint(model, options.output / "pruned.pt")
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
        make_mlp_model(options.model, pretrained=False),
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
            "algorithm": "dvbp_ob2c",
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
        },
        options.output / "training.pt",
    )


if __name__ == "__main__":
    main()
