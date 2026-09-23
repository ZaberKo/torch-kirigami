"""Variance-Based Pruning: calibrate MLP activations, compact, compensate and fine-tune.

Implements the pruning criterion and mean-shift compensation of VBP (ICCV 2025).
The parameter budget, width alignment and torchvision models are explicit
adaptations; this example does not reproduce the paper's accuracy experiments.
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MLP_MODELS, make_mlp_model
from model_metrics import measure_model
from torch import fx, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.ops import stochastic_depth
from tqdm.auto import tqdm

from torch_kirigami import (
    AxisRef,
    CaptureError,
    DependencyGraph,
    Impact,
    OperatorRegistry,
    OperatorRule,
)
from torch_kirigami.operation import OperationContext
from torch_kirigami.pruning import (
    Candidate,
    Granularity,
    Greedy,
    MetricContext,
    ParameterBudget,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def vbp_operators() -> OperatorRegistry:
    """Add torchvision stochastic depth with an explicit evaluation-only contract.

    Both `row` and `batch` modes return their input unchanged in evaluation.
    Reuse the built-in identity relations and preserve alias/layout uncertainty.
    The original module remains in the compact model and still samples its
    usual masks during subsequent fine-tuning; only graph capture requires eval.
    """
    registry = OperatorRegistry.default()

    def require_evaluation(node: fx.Node, module: nn.Module | None) -> None:
        """Reject training capture before its stochastic sample can run."""
        training = node.kwargs.get("training", node.args[3] if len(node.args) > 3 else True)
        if training is not False:
            raise CaptureError(
                "VBP stochastic_depth capture requires model.eval() and training=False"
            )

    return registry.register(
        stochastic_depth,
        OperatorRule(
            registry.modules[nn.Identity].analyze,
            preflight=require_evaluation,
            evaluate_on_meta=True,
        ),
    )


@dataclass
class ActivationMoments:
    """Streaming channel moments, pooling all batch, token and spatial positions."""

    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, activation: torch.Tensor) -> None:
        """Merge a batch using its old-mean difference, without retaining activations."""
        if activation.ndim < 2 or not activation.is_floating_point():
            raise ValueError("Expected real channels-last activations with at least two dimensions")
        values = activation.detach().reshape(-1, activation.shape[-1])
        if not values.numel() or not torch.isfinite(values).all():
            raise ValueError("Calibration activations must be nonempty and finite")
        # Do not expand the entire activation to float64. Promote low precision
        # before reduction; accumulate only the resulting channel vectors in fp64.
        if values.dtype in (torch.float16, torch.bfloat16):
            values = values.float()
        variance, batch_mean = torch.var_mean(values, dim=0, correction=0)
        batch_mean = batch_mean.double()
        batch_m2 = variance.double() * values.shape[0]
        if self.mean is None:
            self.count, self.mean, self.m2 = values.shape[0], batch_mean, batch_m2
            return
        if self.mean.shape != batch_mean.shape or self.mean.device != batch_mean.device:
            raise ValueError("Calibration channel width and device must stay fixed")
        count = self.count + values.shape[0]
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + delta.square() * (self.count * values.shape[0] / count)
        self.mean = self.mean + delta * (values.shape[0] / count)
        self.count = count

    def variance(self) -> torch.Tensor:
        """Return the sample variance used by VBP, requiring at least two observations."""
        if self.count < 2 or self.m2 is None:
            raise ValueError("VBP needs at least two activation observations per MLP")
        result = self.m2 / (self.count - 1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite calibration variance")
        return result


def _coordinatewise(operation: OperationContext) -> bool:
    """Recognize coordinatewise transforms without writes or active randomness."""
    module = operation.module
    if module is not None:
        if type(module) is nn.Dropout:
            return not module.training and not module.inplace
        return type(module) in (
            nn.GELU,
            nn.ReLU,
            nn.SiLU,
            nn.Tanh,
            nn.Sigmoid,
            nn.Identity,
        ) and not getattr(module, "inplace", False)
    if operation.node.op == "call_method":
        return operation.node.target in ("relu", "tanh", "sigmoid")
    if operation.node.target in (F.relu, F.silu):
        return not operation.argument("inplace", 1, False)
    if operation.node.target in (F.gelu, torch.relu, torch.tanh, torch.sigmoid):
        return True
    return (
        operation.node.target is F.dropout
        and not operation.argument("training", 2, True)
        and not operation.argument("inplace", 3, False)
    )


def discover_mlps(graph: DependencyGraph) -> tuple[dict[str, str], dict[str, str]]:
    """Find dense MLPs from captured dataflow, without model-name rules.

    The producer output must reach one biased Linear through a single-use chain
    of supported coordinatewise transforms. Its last axis remains the hidden
    coordinate at every step. Shared parameters, repeated calls and hidden
    branch points are excluded. Consecutive pairs may share a boundary Linear.

    Args:
        graph: An evaluation-mode graph captured before calibration hooks exist.

    Returns:
        Producer-to-consumer module paths and exclusion reasons by producer path.
        Unsupported paths are not inferred from similar shapes or module names.
    """
    graph.validate()
    operations = graph.operations()
    by_node = {operation.node: operation for operation in operations}
    calls = Counter(
        id(operation.module) for operation in operations if operation.module is not None
    )
    uses = Counter(
        ref for operation in operations for ref in {*operation.bindings.values(), *operation.inputs}
    )
    pairs: dict[str, str] = {}
    excluded: dict[str, str] = {}
    for producer in operations:
        if type(producer.module) is not nn.Linear:
            continue
        path = producer.module_path
        assert path is not None
        if calls[id(producer.module)] != 1 or any(
            uses[ref] != 1 for ref in producer.bindings.values()
        ):
            excluded[path] = "Producer is called repeatedly or shares a parameter with another use"
            continue
        current = producer
        while True:
            if len(current.outputs) != 1 or len(current.node.users) != 1:
                excluded[path] = "Hidden value branches or does not have exactly one consumer"
                break
            node = next(iter(current.node.users))
            following = by_node.get(node)
            if following is None or following.inputs != current.outputs:
                excluded[path] = "Hidden value is returned or combined with other inputs"
                break
            if type(following.module) is nn.Linear:
                consumer = following.module
                if (
                    consumer.bias is None
                    or consumer.in_features != producer.module.out_features
                    or calls[id(consumer)] != 1
                    or any(uses[ref] != 1 for ref in following.bindings.values())
                ):
                    excluded[path] = (
                        "Consumer must be biased, width-matched, and used without parameter sharing"
                    )
                    break
                assert following.module_path is not None
                pairs[path] = following.module_path
                break
            if (
                not _coordinatewise(following)
                or len(following.outputs) != 1
                or following.outputs[0].shape != producer.outputs[0].shape
                or following.bindings
            ):
                excluded[path] = f"Unsupported hidden transform: {node.op} {node.target}"
                break
            current = following
    for consumer in pairs.values():
        excluded.pop(consumer, None)
    return pairs, excluded


def _observe_input(
    moments: ActivationMoments,
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, torch.Tensor],
) -> None:
    """Observe the consumer's actual post-activation, post-dropout input."""
    moments.update(args[0] if args else kwargs["input"])


def collect_moments(
    model: nn.Module,
    pairs: dict[str, str],
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    max_batches: int,
) -> dict[str, ActivationMoments]:
    """Calibrate in evaluation mode, restoring modes and removing hooks on failure.

    Labels are ignored. `max_batches=0` consumes the full training loader. The
    caller owns the loader RNG; calibration intentionally advances its iterator.
    Existing gradients are neither read nor modified.
    """
    if type(max_batches) is not int or max_batches < 0:
        raise ValueError("max_batches must be a nonnegative integer")
    if not pairs or len(set(pairs.values())) != len(pairs):
        raise ValueError("Calibration requires distinct MLP consumers")
    moments = {path: ActivationMoments() for path in pairs.values()}
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    total = len(loader) if isinstance(loader, Sized) else None
    if max_batches and total is not None:
        total = min(total, max_batches)
    batches = islice(loader, max_batches) if max_batches else iter(loader)
    try:
        model.eval()
        for path, stats in moments.items():
            handles.append(
                model.get_submodule(path).register_forward_pre_hook(
                    partial(_observe_input, stats), with_kwargs=True
                )
            )
        with (
            torch.no_grad(),
            tqdm(
                batches,
                total=total,
                desc=f"VBP calibration ({device})",
                unit="batch",
                dynamic_ncols=True,
            ) as progress,
        ):
            for images, _labels in progress:
                model(images.to(device, non_blocking=True))
        for stats in moments.values():
            stats.variance()
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return moments


class ActivationVariance:
    """Rank hidden positions by raw activation variance across all declared MLPs.

    These are fixed calibration statistics, with no magnitude weighting or
    layer-wise normalization. Multi-position candidates sum their position scores.
    This is a ranking criterion, not an estimate of a joint output reconstruction
    error; correlated activations are not assumed independent.
    """

    def __init__(
        self, graph: DependencyGraph, pairs: dict[str, str], moments: dict[str, ActivationMoments]
    ) -> None:
        self.graph = graph
        self.scores: dict[AxisRef, tuple[float, ...]] = {}
        for first, second in pairs.items():
            axis = graph.parameter(f"{first}.weight").axis(0)
            values = moments[second].variance().cpu().tolist()
            if len(values) != axis.tensor.shape[axis.dim]:
                raise ValueError("Calibration width differs from the graph's hidden width")
            self.scores[axis] = tuple(values)

    def score(
        self, context: MetricContext, candidates: tuple[Candidate, ...], *, selected: Impact
    ) -> list[float]:
        """Return batch-independent scores for complete hidden-axis selections."""
        if context.graph is not self.graph:
            raise ValueError("Calibration and candidates belong to different graphs")
        context.require_complete(selected)
        result = []
        for candidate in candidates:
            axis = candidate.axis
            if axis not in self.scores or len(candidate.remove) != 1:
                raise ValueError("VBP candidates must select one declared MLP hidden axis")
            selection = candidate.remove[0]
            indices = selection.fully_selected_indices(axis.dim)
            if selection != axis.select(indices):
                raise ValueError("VBP candidates must select complete hidden-axis slices")
            indices = indices.subtract(selected.selection(axis.tensor).fully_selected_indices(0))
            result.append(math.fsum(self.scores[axis][index] for index in indices))
        return result


@torch.no_grad()
def prepare_bias_compensation(
    graph: DependencyGraph,
    plan: PruningPlan,
    pairs: dict[str, str],
    moments: dict[str, ActivationMoments],
) -> dict[str, torch.Tensor]:
    """Prepare complete corrected biases using old coordinates, without mutation.

    Compensation represents replacing each removed hidden activation with its
    calibration mean. It is exact for this local substitution, not an assertion
    of end-to-end equivalence to the original model. Apply the structural plan
    first, then copy these biases and save the corrected checkpoint. The static
    structural plan alone does not contain this algorithm's value correction.
    """
    graph.validate()
    if len(set(pairs.values())) != len(pairs):
        raise ValueError("VBP compensation requires distinct MLP consumers")
    removed_by_producer = {
        path: plan.analysis.selection(graph.parameter(f"{path}.weight")).fully_selected_indices(0)
        for path in pairs
    }
    input_producers = {second: first for first, second in pairs.items()}
    for path in dict.fromkeys((*pairs, *pairs.values())):
        actual = plan.analysis.selection(graph.parameter(f"{path}.weight"))
        weight = actual.tensor
        expected = weight.axis(0).select(removed_by_producer.get(path, ()))
        if path in input_producers:
            expected = expected.union(
                weight.axis(1).select(removed_by_producer[input_producers[path]])
            )
        if actual != expected:
            raise ValueError("VBP compensation requires only matched hidden-width removals")
        layer = graph.model.get_submodule(path)
        if layer.bias is not None:
            actual_bias = plan.analysis.selection(graph.parameter(f"{path}.bias"))
            if actual_bias != actual_bias.tensor.axis(0).select(removed_by_producer.get(path, ())):
                raise ValueError("VBP compensation requires only matched hidden-width removals")
    corrections = {}
    for first, second in pairs.items():
        removed = removed_by_producer[first]
        if not removed:
            continue
        layer = graph.model.get_submodule(second)
        mean = moments[second].mean
        if mean is None or mean.shape != (layer.in_features,):
            raise ValueError("Missing or stale activation means")
        indices = torch.tensor(tuple(removed), device=layer.weight.device)
        correction = layer.weight[:, indices].double() @ mean.to(layer.weight.device)[indices]
        updated = (layer.bias.double() + correction).to(layer.bias.dtype)
        # A consumer can also produce the next pruned hidden layer. Compute its
        # correction in original coordinates, then retain that layer's output rows.
        removed_outputs = removed_by_producer.get(second, ())
        if removed_outputs:
            retained = [
                index for index in range(layer.out_features) if index not in removed_outputs
            ]
            updated = updated[retained]
        if not torch.isfinite(updated).all():
            raise ValueError("VBP compensation produced a nonfinite bias")
        corrections[f"{second}.bias"] = updated
    return corrections


def parse_args() -> argparse.Namespace:
    """Configure VBP calibration, physical pruning and optional task-only fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MLP_MODELS), default="convnext_tiny")
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
        "--pruning_ratio", type=float, default=0.05, help="Whole-model parameter reduction"
    )
    parser.add_argument(
        "--granularity", type=int, default=8, help="Retained hidden-width alignment"
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=16,
        help="Forward-only training batches; 0 uses all",
    )
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.5e-5, help="AdamW peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("runs/variance_pruning"))
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
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if not math.isfinite(options.weight_decay) or options.weight_decay < 0:
        parser.error("--weight_decay must be finite and nonnegative")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def main() -> None:
    """Calibrate once, apply VBP with compensation, then save a verified compact model."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_mlp_model(options.model).to(options.device).eval()
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    graph = DependencyGraph.build(model, args=(example,), operators=vbp_operators())
    pairs, exclusions = discover_mlps(graph)
    print(json.dumps({"eligible_mlps": pairs, "excluded_producers": exclusions}), flush=True)
    if not pairs:
        raise ValueError("No independent dense MLPs support VBP; see excluded_producers")
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
        raise ValueError("VBP requires a separate training split for calibration")
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
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(
        weights=str(weights),
        dataset=dataset_info,
        mlps=pairs,
        excluded_producers=exclusions,
        max_params=budget.max_params,
    )
    options.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    def record(stage: str, **extra: object) -> None:
        """Record full validation and the same complexity/latency inputs at every stage."""
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
    moments = collect_moments(
        model, pairs, train_loader, options.device, options.calibration_batches
    )
    config["calibration_observations"] = {path: stats.count for path, stats in moments.items()}
    # Calibration removed its temporary hooks and restored modes. Its statistics
    # belong to the method, not the graph or the serialized structural plan.
    graph.validate()
    pruner = Pruner(
        model,
        graph=graph,
        granularity=Granularity(by_path=dict.fromkeys(pairs, options.granularity)),
    )
    space = pruner.discover_candidates(targets=tuple(pairs))
    print(
        f"Planning VBP across {len(pairs)} MLPs and {len(space.candidates)} candidates", flush=True
    )
    plan = pruner.plan(
        space, budget=budget, strategy=Greedy(ActivationVariance(graph, pairs, moments))
    )
    corrected_biases = prepare_bias_compensation(graph, plan, pairs, moments)
    model, _result = pruner.apply(plan)
    with torch.no_grad():
        for path, value in corrected_biases.items():
            model.get_parameter(path).copy_(value)
    del moments, corrected_biases
    record(
        "pruned",
        before_params=plan.selection_report.before_params,
        after_params=plan.selection_report.after_params,
        max_params=budget.max_params,
        target_met=plan.selection_report.target_met,
        planning_trials=plan.selection_report.trials,
        planning_limit_reached=plan.selection_report.limit_reached,
    )

    # This task-only adaptation uses the reference AdamW rate and a cosine
    # schedule. Teacher distillation and the paper's augmentation recipe are omitted.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.lr, weight_decay=options.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, options.finetune_epochs * len(train_loader))
    )
    for epoch in range(options.finetune_epochs):
        model.train()
        total, count = 0.0, 0
        with tqdm(
            train_loader,
            desc=f"VBP fine-tuning {epoch + 1} ({options.device})",
            unit="batch",
            dynamic_ncols=True,
        ) as progress:
            for images, labels in progress:
                images, labels = (
                    images.to(options.device, non_blocking=True),
                    labels.to(options.device, non_blocking=True),
                )
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
    if options.finetune_epochs:
        record("finetuned")
    model.eval()
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
            "algorithm": {"method": "variance_based", "selection": "static"},
            "config": config,
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Corrected compact checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
