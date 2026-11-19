"""Prune dense vision MLPs with denoised variance and OB2C compensation.

Research basis:
    Denoised Variance-Based Pruning with Optimal Brain Bias Compensation
    (ECCV 2026).

Collect centered input covariances before pruning. Rank hidden neurons by a
mixture of raw and spectrally denoised variance. Use the same global hidden
neuron ratio for the mixture and the pruning target. After selection, fit
retained consumer weights and biases using the damped denoised covariance.

This example uses torchvision models and one covariance per MLP, including
ConvNeXt. The authors' ConvNeXt code computes a joint covariance for each
stage. Width alignment can round the requested reduction upward. The default
run is training-free. Optional recovery uses task cross-entropy.
"""

import argparse
import json
import math
from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from itertools import islice
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MLP_MODELS, discover_mlps, make_mlp_model, mlp_operators
from model_metrics import measure_model, write_metrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    Granularity,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class ActivationCovariance:
    """Accumulate centered population covariance without retaining activation batches.

    All token or spatial positions contribute rows. The separate image count
    determines the Marchenko-Pastur aspect ratio.

    Attributes:
        rows: Number of activation rows processed.
        images: Number of images processed.
        mean: Running feature mean on the activation device.
        scatter: Running sum of centered outer products on the activation device.
    """

    rows: int = field(default=0, init=False)
    images: int = field(default=0, init=False)
    mean: torch.Tensor | None = field(default=None, init=False)
    scatter: torch.Tensor | None = field(default=None, init=False)

    @torch.no_grad()
    def update(self, activation: torch.Tensor) -> None:
        """Merge one channels-last activation batch, in bounded row chunks."""
        if activation.ndim < 2 or not activation.is_floating_point() or not activation.numel():
            raise ValueError("DVBP requires nonempty real channels-last calibration activations")
        width = activation.shape[-1]
        dtype = torch.float64 if activation.dtype == torch.float64 else torch.float32
        if self.mean is not None and (
            self.mean.shape != (width,)
            or self.mean.device != activation.device
            or self.mean.dtype != dtype
        ):
            raise ValueError(
                "DVBP calibration width, device and accumulation dtype must stay fixed"
            )
        for chunk in activation.reshape(-1, width).split(2048):
            values = chunk.to(dtype=dtype)
            batch_rows = values.shape[0]
            batch_mean = values.mean(0)
            centered = values - batch_mean
            if self.mean is None:
                self.mean = batch_mean
                self.scatter = centered.T @ centered
            else:
                assert self.scatter is not None
                total = self.rows + batch_rows
                delta = batch_mean - self.mean
                self.scatter.addmm_(centered.T, centered)
                self.scatter.addr_(delta, delta, alpha=self.rows * batch_rows / total)
                self.mean.add_(delta, alpha=batch_rows / total)
            self.rows += batch_rows
        self.images += activation.shape[0]


def _observe_input(
    accumulator: ActivationCovariance,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, torch.Tensor],
) -> None:
    """Observe the activation passed to the consumer Linear."""
    accumulator.update(args[0] if args else kwargs["input"])


def collect_covariances(
    model: nn.Module,
    consumers: Iterable[str],
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    max_batches: int,
) -> dict[str, ActivationCovariance]:
    """Observe each consumer's input in eval mode and restore hooks and modes on exit."""
    if max_batches < 0:
        raise ValueError("DVBP calibration batch limit must be nonnegative")
    stats = {path: ActivationCovariance() for path in consumers}
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    total = len(loader) if isinstance(loader, Sized) else None
    if max_batches and total is not None:
        total = min(total, max_batches)

    try:
        model.eval()
        for path, accumulator in stats.items():
            handles.append(
                model.get_submodule(path).register_forward_pre_hook(
                    partial(_observe_input, accumulator), with_kwargs=True
                )
            )
        batches = islice(loader, max_batches) if max_batches else iter(loader)
        with (
            torch.no_grad(),
            tqdm(
                batches,
                total=total,
                desc=f"DVBP calibration ({device})",
                unit="batch",
                dynamic_ncols=True,
            ) as progress,
        ):
            for images, _labels in progress:
                model(images.to(device, non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return stats


def mp_median(aspect: float) -> float:
    """Compute the unit-noise median of the continuous Marchenko-Pastur bulk."""
    if aspect > 1:
        # The nonzero spectrum of a wide matrix is the reciprocal-aspect
        # spectrum, scaled by the aspect ratio.
        return aspect * mp_median(1 / aspect)
    points = 16384
    step = math.pi / points
    theta = (torch.arange(points, dtype=torch.float64, device="cpu") + 0.5) * step
    root = math.sqrt(aspect)
    lower = (1 - root) ** 2
    # x = lower + 4 sqrt(aspect) sin²(theta / 2) removes endpoint
    # square-root singularities from the density times dx.
    density = theta.sin().square() / (lower + 4 * root * (theta / 2).sin().square())
    cumulative = density.cumsum(0)
    half = cumulative[-1] / 2
    index = int(torch.searchsorted(cumulative, half))
    fraction = 1 - float((cumulative[index] - half) / density[index])
    return lower + 4 * root * math.sin((index + fraction) * step / 2) ** 2


@dataclass(frozen=True)
class DenoisedCovariance:
    """Store calibration statistics and a factor of the denoised covariance.

    Attributes:
        mean: Original feature mean on CPU.
        scores: Hybrid neuron scores on CPU in original hidden order.
        signal: CPU factor such that `signal @ signal.T` is the denoised covariance.
        ridge: Diagonal damping used only for compensation.
        images: Image count used to estimate the noise threshold.
        rows: Activation row count used to estimate the covariance.
    """

    mean: torch.Tensor
    scores: torch.Tensor
    signal: torch.Tensor
    ridge: float
    images: int
    rows: int


@torch.no_grad()
def denoise(stats: ActivationCovariance, ratio: float, damping: float) -> DenoisedCovariance:
    """Compute hybrid scores and retain only the signal factor needed for OB2C."""
    if stats.rows < 2 or stats.mean is None or stats.scatter is None:
        raise ValueError("DVBP requires at least two calibration rows per MLP")
    covariance = stats.scatter / stats.rows
    if not torch.isfinite(covariance).all() or not torch.isfinite(stats.mean).all():
        raise ValueError("DVBP calibration produced nonfinite statistics")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    aspect = len(eigenvalues) / stats.images
    noise_variance = float(eigenvalues.median().clamp_min(0)) / mp_median(aspect)
    signal_mask = eigenvalues > noise_variance * (1 + math.sqrt(aspect)) ** 2
    # Match the reference fallback when the fit finds no outlier.
    signal_mask[-1] = True
    signal = eigenvectors[:, signal_mask] * eigenvalues[signal_mask].clamp_min(0).sqrt()
    signal_variance = signal.double().square().sum(1)
    scores = (1 - ratio) * covariance.diagonal().double() + ratio * signal_variance
    ridge = damping * float(signal_variance.mean())
    if not torch.isfinite(scores).all() or not math.isfinite(ridge):
        raise ValueError("DVBP denoising produced nonfinite statistics")
    return DenoisedCovariance(
        stats.mean.cpu(), scores.cpu(), signal.cpu(), ridge, stats.images, stats.rows
    )


def select_hidden_neurons(
    scores: dict[str, torch.Tensor], target: int, granularity: int
) -> dict[str, tuple[int, ...]]:
    """Globally remove low-scoring neurons while keeping aligned nonempty widths.

    With granularity one, rank individual neurons as in the paper. Reserve the
    highest-scoring aligned block in every layer. With a larger granularity,
    first remove each width's alignment remainder, then rank aligned removal
    blocks by their total score. The caller validates all selections jointly
    against the dependency graph.
    """
    ranked: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    blocks: list[tuple[float, str, int]] = []
    for path, score in scores.items():
        values = score.tolist()
        width = len(values)
        if width < granularity:
            raise ValueError(f"DVBP cannot retain {granularity} hidden neurons in {path!r}")
        order = sorted(range(width), key=lambda index: (values[index], index))
        ranked[path] = order
        counts[path] = width % granularity
        for start in range(counts[path], width - granularity, granularity):
            block_score = math.fsum(values[i] for i in order[start : start + granularity])
            blocks.append((block_score, path, start))
    removed = sum(counts.values())
    for _score, path, _start in sorted(blocks):
        if removed >= target:
            break
        counts[path] += granularity
        removed += granularity
    if removed < target:
        raise ValueError(f"DVBP can remove at most {removed} hidden neurons; requested {target}")
    return {path: tuple(order[: counts[path]]) for path, order in ranked.items()}


@torch.no_grad()
def compensate(
    weight: torch.Tensor,
    bias: torch.Tensor,
    stats: DenoisedCovariance,
    removed: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit the consumer using the retained covariance block and preserve its mean.

    Solve the normal equation for the added retained weights. This gives the
    same update as the reference block-inverse formula for a positive definite
    system. The zero-variance case only needs a bias correction.
    """
    keep = torch.ones(weight.shape[1], dtype=torch.bool, device="cpu")
    keep[list(removed)] = False
    w = weight.detach().to(device="cpu", dtype=torch.float64)
    mean = stats.mean.double()
    removed_weight = w[:, ~keep]
    corrected = w[:, keep]
    corrected_bias = (
        bias.detach().to(device="cpu", dtype=torch.float64) + removed_weight @ mean[~keep]
    )
    if stats.ridge > 0:
        retained_signal = stats.signal[keep].double()
        retained_covariance = retained_signal @ retained_signal.T
        retained_covariance.diagonal().add_(stats.ridge)
        cross_covariance = retained_signal @ stats.signal[~keep].double().T
        modification = torch.linalg.solve(
            retained_covariance, cross_covariance @ removed_weight.T
        ).T
        corrected = corrected + modification
        corrected_bias = corrected_bias - modification @ mean[keep]
    corrected = corrected.to(dtype=weight.dtype)
    corrected_bias = corrected_bias.to(dtype=bias.dtype)
    if not torch.isfinite(corrected).all() or not torch.isfinite(corrected_bias).all():
        raise ValueError("OB2C compensation overflows the model parameter dtype")
    return corrected, corrected_bias


@torch.no_grad()
def prepare_compensation(
    graph: DependencyGraph,
    plan: PruningPlan,
    pairs: dict[str, str],
    statistics: dict[str, DenoisedCovariance],
) -> dict[str, torch.Tensor]:
    """Prepare compact consumer values for a plan over the discovered MLP pairs."""
    graph.validate()
    removals = {
        path: plan.analysis.selection(graph.parameter(f"{path}.weight")).fully_selected_indices(0)
        for path in pairs
    }
    inputs = {consumer: removals[producer] for producer, consumer in pairs.items()}
    matched_changes = set()
    for path in dict.fromkeys((*pairs, *inputs)):
        actual = plan.analysis.selection(graph.parameter(f"{path}.weight"))
        # The plan stores portable references. Compare both selections on
        # that reference, even though the graph lookup uses the live reference.
        weight = actual.tensor
        expected = (
            weight.axis(0)
            .select(removals.get(path, ()))
            .union(weight.axis(1).select(inputs.get(path, ())))
        )
        if actual != expected:
            raise ValueError("OB2C plan must remove matching producer rows and consumer columns")
        matched_changes.add(weight)
        layer = graph.model.get_submodule(path)
        if layer.bias is not None:
            bias = plan.analysis.selection(graph.parameter(f"{path}.bias"))
            if bias != bias.tensor.axis(0).select(removals.get(path, ())):
                raise ValueError("OB2C plan must remove matching producer bias positions")
            matched_changes.add(bias.tensor)
    if any(
        selection.tensor.kind in ("parameter", "buffer") and selection.tensor not in matched_changes
        for selection in plan.analysis.selections
        if selection.regions
    ):
        raise ValueError("OB2C plan changes parameters or buffers outside the discovered MLP pairs")
    updates = {}
    for path, positions in inputs.items():
        if positions:
            layer = graph.model.get_submodule(path)
            assert type(layer) is nn.Linear and layer.bias is not None
            rows = removals.get(path)
            keep = [i for i in range(layer.out_features) if rows is None or i not in rows]
            weight, bias = compensate(
                layer.weight[keep], layer.bias[keep], statistics[path], tuple(positions)
            )
            updates[f"{path}.weight"] = weight
            updates[f"{path}.bias"] = bias
    return updates


def parse_args() -> argparse.Namespace:
    """Configure DVBP calibration, hidden-width reduction and optional recovery."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MLP_MODELS), default="vit_b_16")
    parser.add_argument("--data_dir", type=Path, help="Local ImageNet snapshot; default: HF cache")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--train_samples", type=int, default=0, help="0 exposes the full train split"
    )
    parser.add_argument("--val_samples", type=int, default=0, help="0 evaluates the full val split")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--hidden_pruning_ratio",
        type=float,
        default=0.2,
        help="Global fraction of eligible MLP hidden neurons to remove and denoised score weight",
    )
    parser.add_argument(
        "--granularity", type=int, default=8, help="Retained hidden-width alignment"
    )
    parser.add_argument(
        "--calibration_batches", type=int, default=32, help="Forward-only batches; 0 uses all"
    )
    parser.add_argument("--damping", type=float, default=0.1, help="OB2C relative diagonal damping")
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.5e-5, help="AdamW peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("runs/dvbp_pruning"))
    parser.add_argument(
        "--compile_latency", action="store_true", help="Compile latency measurement only"
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
        "calibration_batches",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.hidden_pruning_ratio < 1:
        parser.error("Require 0 <= hidden_pruning_ratio < 1")
    if any(not math.isfinite(value) or value <= 0 for value in (options.damping, options.lr)):
        parser.error("--damping and --lr must be positive and finite")
    if not math.isfinite(options.weight_decay) or options.weight_decay < 0:
        parser.error("--weight_decay must be finite and nonnegative")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install a CUDA build or pass --device cpu")
    return options


def main() -> None:
    """Calibrate, prune, compensate and verify a saved compact model."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_mlp_model(options.model).to(options.device).eval()
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    graph = DependencyGraph.build(model, args=(example,), operators=mlp_operators())
    pairs, exclusions = discover_mlps(graph)
    if not pairs:
        raise ValueError(f"No dense MLP pairs support DVBP: {exclusions}")
    weights = MLP_MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    if train is None:
        raise ValueError("DVBP requires training images for calibration")
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train,
        batch_size=options.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=options.train_workers,
        persistent_workers=options.train_workers > 0,
        multiprocessing_context="spawn" if options.train_workers else None,
        pin_memory=options.device == "cuda",
    )
    val_loader = DataLoader(
        validation,
        batch_size=options.val_batch_size,
        num_workers=options.val_workers,
        persistent_workers=options.val_workers > 0,
        multiprocessing_context="spawn" if options.val_workers else None,
        pin_memory=options.device == "cuda",
    )
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(
        weights=str(weights), dataset=dataset_info, mlps=pairs, excluded_producers=exclusions
    )
    options.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    baseline: float | None = None

    def record(stage: str, **extra: object) -> None:
        """Evaluate accuracy, complexity and latency for a workflow stage."""
        nonlocal baseline
        accuracy = evaluate(model, val_loader, options.device, description=f"{stage} evaluation")
        if baseline is None:
            baseline = accuracy["top1"]
        row = {
            "stage": stage,
            **accuracy,
            "top1_delta_pp": accuracy["top1"] - baseline,
            **measure_model(model, example, options),
            **extra,
        }
        records.append(row)
        print(json.dumps(row), flush=True)
        write_metrics(options.output, config, records)

    record("pretrained")
    ratio = options.hidden_pruning_ratio
    covariances = collect_covariances(
        model, pairs.values(), train_loader, options.device, options.calibration_batches
    )
    statistics = {
        path: denoise(covariances.pop(path), ratio, options.damping)
        for path in tqdm(pairs.values(), desc="DVBP spectral denoising", unit="MLP")
    }
    config["calibration"] = {
        path: {"images": stats.images, "rows": stats.rows, "signal_rank": stats.signal.shape[1]}
        for path, stats in statistics.items()
    }
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(pairs, options.granularity)),
    )
    scores = {path: statistics[consumer].scores for path, consumer in pairs.items()}
    hidden = sum(score.numel() for score in scores.values())
    target_removed = math.floor(Fraction(str(ratio)) * hidden)
    print(f"Planning removal of {target_removed} of {hidden} hidden neurons", flush=True)
    selected = select_hidden_neurons(scores, target_removed, options.granularity)
    plan = pruner.plan_remove(
        graph.parameter(f"{path}.weight").axis(0).select(positions)
        for path, positions in selected.items()
        if positions
    )
    removed_hidden = sum(
        len(plan.analysis.selection(graph.parameter(f"{path}.weight")).fully_selected_indices(0))
        for path in pairs
    )
    updates = prepare_compensation(graph, plan, pairs, statistics)
    model, _ = pruner.apply(plan)
    with torch.no_grad():
        for path, value in updates.items():
            model.get_parameter(path).copy_(value)
    del statistics, updates
    record(
        "pruned",
        original_hidden=hidden,
        target_removed_hidden=target_removed,
        removed_hidden=removed_hidden,
    )

    if options.finetune_epochs:
        pruned_checkpoint = options.output / "pruned.pt"
        save_checkpoint(model, pruned_checkpoint)
        print(f"Saved pre-fine-tuning checkpoint: {pruned_checkpoint}", flush=True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=options.lr, weight_decay=options.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=options.finetune_epochs * len(train_loader)
        )
        for epoch in range(options.finetune_epochs):
            model.train()
            total, count = 0.0, 0
            with tqdm(
                train_loader,
                desc=f"DVBP fine-tuning {epoch + 1}/{options.finetune_epochs} ({options.device})",
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
                    scheduler.step()
                    total += loss.detach().item() * labels.numel()
                    count += labels.numel()
                    progress.set_postfix(images=count, loss=f"{total / count:.4f}", refresh=False)
            if not count:
                raise ValueError("Cannot fine-tune on an empty training split")
            print(json.dumps({"epoch": epoch + 1, "task_loss": total / count}), flush=True)
        model.eval()
        record("finetuned")

    checkpoint = options.output / "model.pt"
    save_checkpoint(model, checkpoint)
    restored = load_checkpoint(
        make_mlp_model(options.model, pretrained=False), checkpoint, map_location=options.device
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example))
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
