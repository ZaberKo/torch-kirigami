"""Pretrained model -> scored structural pruning -> optional ImageNet fine-tuning."""

import argparse
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import torch
from imagenet_data import Images, evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import EncoderBlock
from tqdm.auto import tqdm

from torch_kirigami import AxisRef, DependencyGraph, Impact, Selection
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    DynamicGreedy,
    Granularity,
    Greedy,
    GroupMagnitude,
    MetricContext,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    WeightTaylor,
    load_checkpoint,
    save_checkpoint,
)


class GeometricMedian:
    """Static output-filter redundancy scores using original signed weights.

    Each filter receives the sum of Euclidean distances to every filter on the
    same output axis. Low scores identify filters near the geometric median.
    This uses the FPGM criterion (CVPR 2019), adapted to Conv/Linear candidates;
    global parameter budgeting and one-shot removal are not the paper's full
    layerwise soft-pruning training procedure. Unlike Torch-Pruning's variant,
    the distances use signed weights, without an absolute-value or power transform.

    Args:
        graph: Graph whose current weights supply the fixed score snapshot.
        axes: Explicit output-weight axes, normally `space.channel_axes`.
        block_size: Maximum rows and columns of each pairwise-distance tile.

    All scores are calculated once before planning. No weights or activations
    are retained, and dynamic conditional scoring is deliberately unsupported.
    Recreate this metric after changing weights or the model structure.
    """

    def __init__(
        self, graph: DependencyGraph, axes: Iterable[AxisRef], *, block_size: int = 256
    ) -> None:
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        graph.validate()
        bindings = dict(graph.tensor_bindings())
        self.graph_id = graph.id
        self.scores: dict[AxisRef, tuple[float, ...]] = {}
        with torch.no_grad():
            for axis in dict.fromkeys(axes):
                if (
                    axis.tensor.kind != "parameter"
                    or axis.dim != 0
                    or len(axis.tensor.shape) < 2
                    or axis.tensor not in bindings
                ):
                    raise ValueError("GeometricMedian requires registered output-weight axes")
                weight = bindings[axis.tensor]
                if not weight.is_floating_point() or not torch.isfinite(weight).all():
                    raise PlanningError("GeometricMedian requires finite real floating weights")
                dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
                rows = weight.detach().to(dtype=dtype).flatten(1)
                scores = torch.zeros(rows.shape[0], dtype=dtype, device=rows.device)
                for start in range(0, len(rows), block_size):
                    current = rows[start : start + block_size]
                    for other in range(0, len(rows), block_size):
                        distances = torch.cdist(
                            current,
                            rows[other : other + block_size],
                            p=2,
                            compute_mode="donot_use_mm_for_euclid_dist",
                        )
                        scores[start : start + len(current)] += distances.sum(1)
                if not torch.isfinite(scores).all():
                    raise PlanningError("GeometricMedian produced nonfinite distance scores")
                self.scores[axis] = tuple(scores.cpu().tolist())

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        selected: Impact,
    ) -> list[float]:
        """Sum unique output-row scores; reject partial slices and dynamic use."""
        if context.graph.id != self.graph_id:
            raise PlanningError("GeometricMedian scores belong to a different dependency graph")
        context.require_complete(selected)
        if selected.selections:
            raise PlanningError("GeometricMedian supports static selection only")
        values = []
        for candidate in candidates:
            axis = candidate.axis
            if axis not in self.scores:
                raise PlanningError("GeometricMedian candidate needs a scored output-weight axis")
            selection = Selection(axis.tensor)
            for seed in candidate.remove:
                if seed.tensor != axis.tensor:
                    raise PlanningError("GeometricMedian candidate must select one output axis")
                selection = selection.union(seed)
            indices = selection.fully_selected_indices(axis.dim)
            if selection != axis.select(indices):
                raise PlanningError("GeometricMedian requires complete output-filter selections")
            value = math.fsum(self.scores[axis][index] for index in indices)
            if not math.isfinite(value):
                raise PlanningError("GeometricMedian candidate score must be finite")
            values.append(value)
        return values


def parse_args() -> argparse.Namespace:
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
        help="Accuracy evaluation batch size",
    )
    parser.add_argument(
        "--train_workers",
        type=int,
        default=8,
        help="Training loader processes; 0 runs in the main process",
    )
    parser.add_argument(
        "--val_workers",
        type=int,
        default=8,
        help="Validation loader processes; 0 runs in the main process",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Fraction of whole-model parameters to remove (default: 0.05); not a channel ratio",
    )
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument(
        "--metric",
        choices=("magnitude", "taylor", "geometric_median"),
        default="magnitude",
        help="Group magnitude, affected-weight Taylor, or static signed-filter redundancy scores",
    )
    parser.add_argument(
        "--selection",
        choices=("static", "dynamic"),
        default="static",
        help="Static ranking or rescoring after accepted deletions; Taylor retains the original gradients",
    )
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
        "train_workers",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if options.metric == "geometric_median" and options.selection != "static":
        parser.error("--metric geometric_median requires --selection static")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def collect_task_gradients(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device | str,
) -> None:
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
            loss = F.cross_entropy(
                model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True)
            )
            loss.backward()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}", refresh=False)
            progress.update()
    finally:
        for module, training in modes:
            module.training = training


def make_plan(
    pruner: Pruner,
    space: CandidateSpace,
    budget: ParameterBudget,
    metric: str,
    selection: str = "static",
) -> PruningPlan:
    """Select a feasible request; dynamic Taylor reuses its calibrated gradients."""
    if selection not in ("static", "dynamic"):
        raise ValueError("selection must be static or dynamic")
    if metric == "geometric_median":
        if selection != "static":
            raise ValueError("geometric_median requires static selection")
        score = GeometricMedian(pruner.graph, space.channel_axes)
    elif metric == "taylor":
        score = WeightTaylor(mode="elementwise_abs")
    elif metric == "magnitude":
        score = GroupMagnitude(p=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    strategy = Greedy(score) if selection == "static" else DynamicGreedy(score)
    return pruner.plan(space, budget=budget, strategy=strategy)


def main() -> None:
    """Evaluate, prune and optionally fine-tune a pretrained ImageNet model."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    # Visit every block, but retain its external width and residual interface.
    targets = []
    for path, block in model.named_modules():
        if type(block) is BasicBlock:
            targets.append(f"{path}.conv1")
        elif type(block) is Bottleneck:
            targets.extend((f"{path}.conv1", f"{path}.conv2"))
        elif type(block) is EncoderBlock:
            targets.append(f"{path}.mlp.0")
    if not targets:
        raise PlanningError("No supported block-internal pruning positions were found")
    print(
        f"Model: {options.model}; {len(targets)} block-internal pruning axes; "
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
    records: list[dict[str, Any]] = []
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(
        weights=str(weights),
        dataset=dataset_info,
        max_params=budget.max_params,
        pruning_scope="block_internal",
        pruning_targets=targets,
    )
    options.output.mkdir(parents=True, exist_ok=True)

    def record(stage: str, **extra: object) -> None:
        """Evaluate the current model and persist this stage's measurements."""
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
    print(f"Capturing dependency graph; sample forward on {options.device}", flush=True)
    started = time.perf_counter()
    graph = DependencyGraph.build(model, args=(example,))
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(targets, options.granularity)),
    )
    space = pruner.discover_candidates(targets=targets)
    config["candidate_axes"] = [
        {"parameter": axis.tensor.paths[0], "dim": axis.dim} for axis in space.channel_axes
    ]
    config["discovery_exclusions"] = space.exclusions
    (options.output / "metrics.json").write_text(
        json.dumps({"config": config, "stages": records}, indent=2) + "\n"
    )
    print(
        f"Graph and {len(space.candidates)} candidates ready in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    if options.metric == "taylor":
        collect_task_gradients(model, train_loader, options.device)
    print("Planning pruning: dependency propagation and constraint search run on CPU", flush=True)
    started = time.perf_counter()
    plan = make_plan(pruner, space, budget, options.metric, selection=options.selection)
    print(
        f"Plan completed in {time.perf_counter() - started:.1f}s; "
        f"{plan.selection_report.trials} joint trials; applying on {options.device}",
        flush=True,
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

    # Parameters were replaced by apply; create the optimizer only afterwards.
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        print(
            f"Fine-tuning epoch {epoch + 1}: {len(cast(Images, train))} images on {options.device}; "
            f"data loader workers: {options.train_workers}",
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
            "algorithm": {"metric": options.metric, "selection": options.selection},
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
