"""Train and physically prune ResNet or ViT with CPD-style importance scores.

*Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression*
(ITSC 2024).

Each interval trains against labels and a frozen dense teacher. After each
backward pass, square the signed weight-gradient sum for each output channel,
divide by the current output activation count, and accumulate across batches.
Rank groups by the sum of their lowest-scoring channels using the published
code's group normalization and removal sizes. Apply one feasible group, then
rebuild the dependency graph for the next interval.

This ImageNet adaptation uses Kirigami's dependency graph and physical
compaction. It targets ResNet block interiors and ViT FFN hidden widths and
stops at a cap on surviving parameters. The public CPD code stops at an output
mask sparsity threshold and provides a semantic segmentation configuration.
This example uses classification logit KL and optional recovery with the same
objective. The final interval can remove more parameters than the cap requires.
"""

import argparse
import copy
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch_prisma import count_parameters
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import EncoderBlock
from tqdm.auto import tqdm

from torch_kirigami import AnalysisLimitError, DependencyGraph
from torch_kirigami.dep_graph.selection import gather_region
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningResult,
    StrategyResult,
    load_checkpoint,
    save_checkpoint,
)

Batch = tuple[torch.Tensor, torch.Tensor]


def pruning_groups(model: nn.Module) -> dict[str, tuple[tuple[nn.Module, ...], nn.Module]]:
    """Find scored output members and the consumer of each supported hidden width."""
    groups = {}
    for path, block in model.named_modules():
        prefix = f"{path}." if path else ""
        if type(block) in (BasicBlock, Bottleneck):
            positions = (1,) if type(block) is BasicBlock else (1, 2)
            for index in positions:
                producer = block.get_submodule(f"conv{index}")
                norm = block.get_submodule(f"bn{index}")
                consumer = block.get_submodule(f"conv{index + 1}")
                if (
                    type(producer) is nn.Conv2d
                    and producer.groups == 1
                    and type(consumer) is nn.Conv2d
                    and consumer.groups == 1
                    and type(norm) is nn.BatchNorm2d
                ):
                    groups[f"{prefix}conv{index}"] = ((producer, norm), consumer)
        elif type(block) is EncoderBlock:
            groups[f"{prefix}mlp.0"] = ((block.mlp[0],), block.mlp[3])
    if not groups:
        raise PlanningError("No supported CPD block interiors were found")
    return groups


def channel_scores(parameter: nn.Parameter, activations: int) -> torch.Tensor:
    """Compute the squared gate-gradient proxy for one backward pass."""
    gradient = parameter.grad
    if gradient is None or gradient.layout != torch.strided:
        raise PlanningError("CPD requires dense gradients for every scored parameter")
    dtype = torch.float64 if parameter.dtype == torch.float64 else torch.float32
    products = parameter.detach().to(dtype) * gradient.detach().to(dtype)
    return products.reshape(parameter.shape[0], -1).sum(1).square().div_(activations)


def train_batches(
    model: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.SGD,
    batches: Iterable[Batch],
    device: torch.device | str,
    *,
    temperature: float,
    distillation_weight: float,
    after_backward: Callable[[], None] | None = None,
) -> None:
    """Train batches, optionally collecting scores between backward and SGD."""
    for images, labels in batches:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        if distillation_weight:
            with torch.no_grad():
                teacher_probs = F.softmax(teacher(images) / temperature, dim=-1)
            loss = loss + distillation_weight * temperature**2 * F.kl_div(
                F.log_softmax(logits / temperature, dim=-1), teacher_probs, reduction="batchmean"
            )
        if not torch.isfinite(loss):
            raise ValueError("CPD training produced a nonfinite loss")
        loss.backward()
        if after_backward is not None:
            after_backward()
        optimizer.step()


def train_interval(
    model: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.SGD,
    batches: Iterable[Batch],
    options: argparse.Namespace,
) -> dict[str, list[float]]:
    """Accumulate fresh activation-normalized scores while training one interval."""
    groups = pruning_groups(model)
    parameters = {
        module: tuple(module.parameters(recurse=False))
        for members, _ in groups.values()
        for module in members
    }
    # Normalization parameters occur on both sides of the source group.
    cardinalities = {
        path: sum(len(parameters[module]) for module in (*members, *members[1:]))
        + len(tuple(consumer.parameters(recurse=False)))
        for path, (members, consumer) in groups.items()
    }
    totals: dict[str, torch.Tensor] = {}
    activations: dict[nn.Module, int] = {}

    def count_output(module: nn.Module, _args: tuple[object, ...], output: torch.Tensor) -> None:
        """Count all output elements for each call to a scored module."""
        activations[module] = activations.get(module, 0) + output.numel()

    def accumulate() -> None:
        """Accumulate one batch's scores in the current channel coordinates."""
        for path, (members, _) in groups.items():
            scores = [
                channel_scores(parameter, activations[module])
                for module in members
                for parameter in parameters[module]
            ]
            batch_score = torch.stack(scores).mean(0) / cardinalities[path]
            if path in totals:
                totals[path].add_(batch_score)
            else:
                totals[path] = batch_score
        activations.clear()

    hooks = [module.register_forward_hook(count_output) for module in parameters]
    try:
        train_batches(
            model,
            teacher,
            optimizer,
            batches,
            options.device,
            temperature=options.temperature,
            distillation_weight=options.distillation_weight,
            after_backward=accumulate,
        )
    finally:
        for hook in hooks:
            hook.remove()
    if not totals:
        raise ValueError("CPD requires at least one training batch per interval")
    result = {path: total.tolist() for path, total in totals.items()}
    if any(not math.isfinite(score) for scores in result.values() for score in scores):
        raise ValueError("CPD produced nonfinite channel scores")
    return result


@dataclass(frozen=True)
class LeastFirst:
    """Choose the cheapest executable group for the current training interval."""

    scores: Mapping[str, float]

    def select(self, context: PlanningContext) -> StrategyResult:
        """Try groups in score order through the public planner checks."""
        exclusions = []
        for candidate in sorted(context.candidates, key=lambda c: (self.scores[c.key], c.key)):
            try:
                impact = context.attempt(candidate.remove)
                context.compile(impact)
                context.require_budget(impact)
            except (PlanningError, AnalysisLimitError) as error:
                exclusions.append((candidate.key, str(error)))
                continue
            return StrategyResult((candidate.key,), "target_reached", tuple(exclusions))
        raise PlanningError(f"No executable CPD group remains. Rejections: {exclusions}")


def prune_interval(
    model: nn.Module, example: torch.Tensor, scores: Mapping[str, list[float]]
) -> PruningResult:
    """Rank one channel block per hidden axis and apply the best feasible block."""
    graph = DependencyGraph.build(model, args=(example,))
    candidates, ranking = [], {}
    for path, values in scores.items():
        module = model.get_submodule(path)
        axis = graph.parameter(f"{path}.weight").axis(0)
        width = axis.tensor.shape[0]
        if len(values) != width:
            raise ValueError(f"CPD scores for {path} must match the current width")
        if width <= 1:
            continue
        # The official FFN pruner uses 32. BasePruner rounds width/16 down to a
        # power of two and caps the result at four for dense convolutions.
        count = (
            32 if type(module) is nn.Linear else min(4, 1 << max(0, (width // 16).bit_length() - 1))
        )
        indices = sorted(range(width), key=lambda index: (values[index], index))[
            : min(count, width - 1)
        ]
        candidates.append(Candidate(path, (axis.select(indices),), axis))
        ranking[path] = math.fsum(values[index] for index in indices)
    _, result = Pruner(model, graph=graph).prune(
        CandidateSpace(tuple(candidates), ()),
        budget=ParameterBudget(count_parameters(model) - 1),
        strategy=LeastFirst(ranking),
    )
    return result


def rebuild_optimizer(
    model: nn.Module,
    optimizer: torch.optim.SGD,
    before: Mapping[str, nn.Parameter],
    result: PruningResult,
) -> torch.optim.SGD:
    """Carry SGD momentum through the same coordinate selections as the weights."""
    rebuilt = torch.optim.SGD(model.parameters(), **optimizer.defaults)
    recipes = {
        before[r.tensor.paths[0]]: r for r in result.plan.recipes if r.tensor.kind == "parameter"
    }
    for old, state in optimizer.state.items():
        momentum = state["momentum_buffer"]
        if old in recipes:
            recipe = recipes[old]
            parts = [gather_region(momentum, region) for region in recipe.segments]
            momentum = parts[0] if len(parts) == 1 else torch.cat(parts, dim=recipe.concat_dim)
        rebuilt.state[result.parameter_map.get(old, old)]["momentum_buffer"] = momentum
    return rebuilt


def parse_args() -> argparse.Namespace:
    """Parse the CPD training and final parameter target options."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--pruning_ratio", type=float, default=0.05)
    parser.add_argument("--pruning_interval_batches", type=int, default=50)
    parser.add_argument("--max_intervals", type=int, default=250)
    parser.add_argument("--recovery_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distillation_weight", type=float, default=1.0)
    for split in ("train", "val"):
        parser.add_argument(f"--{split}_samples", type=int, default=0)
        parser.add_argument(f"--{split}_batch_size", type=int, default=256)
        parser.add_argument(f"--{split}_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compile_latency", action="store_true")
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("runs/cpd_pruning"))
    args = parser.parse_args()
    if not 0 <= args.pruning_ratio < 1:
        parser.error("Require 0 <= pruning_ratio < 1")
    for name in (
        "pruning_interval_batches",
        "max_intervals",
        "train_batch_size",
        "val_batch_size",
        "latency_repetitions",
        "lr",
        "temperature",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name} must be finite and positive")
    for name in (
        "recovery_epochs",
        "train_samples",
        "val_samples",
        "train_workers",
        "val_workers",
        "latency_warmup",
        "distillation_weight",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative and finite")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Pass --device cpu to run on CPU.")
    return args


def main() -> None:
    """Train with CPD intervals, evaluate the compact model, and save a checkpoint."""
    args = parse_args()
    torch.manual_seed(args.seed)
    model = make_model(args.model).to(args.device).eval()
    example = torch.zeros(1, 3, 224, 224, device=args.device)
    target = ParameterBudget.from_ratio(model, args.pruning_ratio).max_params
    train, validation, data = load_images(
        MODELS[args.model][1],
        args.data_dir,
        need_train=args.pruning_ratio > 0 or args.recovery_epochs > 0,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=args.val_batch_size,
        num_workers=args.val_workers,
        persistent_workers=args.val_workers > 0,
        multiprocessing_context="spawn" if args.val_workers else None,
        pin_memory=args.device == "cuda",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    def record(stage: str) -> None:
        """Save accuracy and model measurements at a workflow boundary."""
        row = {
            "stage": stage,
            **evaluate(model, validation_loader, args.device, description=f"{stage} evaluation"),
            **measure_model(model, example, args),
        }
        records.append(row)
        print(json.dumps(row), flush=True)

    record("pretrained")
    if train is not None:
        if not len(train):
            raise ValueError("CPD requires a nonempty training split")
        teacher = copy.deepcopy(model).requires_grad_(False) if args.distillation_weight else model
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        train_loader = DataLoader(
            train,
            batch_size=args.train_batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            num_workers=args.train_workers,
            persistent_workers=args.train_workers > 0,
            multiprocessing_context="spawn" if args.train_workers else None,
            pin_memory=args.device == "cuda",
        )

        def training_stream() -> Iterator[Batch]:
            """Keep consuming the shuffled loader across pruning boundaries."""
            while True:
                yield from train_loader

        stream = training_stream()
        try:
            for interval in range(args.max_intervals):
                if count_parameters(model) <= target:
                    break
                batches = islice(stream, args.pruning_interval_batches)
                with tqdm(
                    batches,
                    total=args.pruning_interval_batches,
                    desc=f"CPD interval {interval + 1}",
                    unit="batch",
                ) as progress:
                    scores = train_interval(model, teacher, optimizer, progress, args)
                before = dict(model.named_parameters())
                result = prune_interval(model, example, scores)
                optimizer = rebuild_optimizer(model, optimizer, before, result)
                print(
                    f"Pruned {result.plan.selected[0]}: {count_parameters(model):,} parameters",
                    flush=True,
                )
                del before, result
        finally:
            stream.close()
        if count_parameters(model) > target:
            raise PlanningError(
                f"CPD interval limit reached with {count_parameters(model):,} parameters; cap is {target:,}"
            )
        if args.pruning_ratio > 0:
            record("pruned")
        for epoch in range(args.recovery_epochs):
            with tqdm(train_loader, desc=f"Recovery {epoch + 1}", unit="batch") as progress:
                train_batches(
                    model,
                    teacher,
                    optimizer,
                    progress,
                    args.device,
                    temperature=args.temperature,
                    distillation_weight=args.distillation_weight,
                )
        if args.recovery_epochs:
            record("recovered")
        optimizer.zero_grad(set_to_none=True)
        del optimizer, teacher
    model.eval()
    checkpoint = args.output / "model.pt"
    save_checkpoint(model, checkpoint)
    restored = load_checkpoint(
        make_model(args.model, pretrained=False), checkpoint, map_location=args.device
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example))
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(dataset=data, max_params=target)
    (args.output / "metrics.json").write_text(
        json.dumps({"config": config, "stages": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
