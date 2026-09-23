"""Prune a pretrained ImageNet model toward a final absolute parameter limit."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import EncoderBlock
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph
from torch_kirigami.measurement import count_parameters
from torch_kirigami.pruning import (
    CandidateSpace,
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    """Configure repeated pruning and the fine-tuning performed after each round."""
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
    parser.add_argument("--train_batch_size", type=int, default=256, help="Fine-tuning batch size")
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
        help="Final fraction of initial whole-model parameters to remove (default: 0.05)",
    )
    parser.add_argument("--granularity", type=int, default=8, help="Retained channel alignment")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--finetune_epochs", type=int, default=0, help="Training epochs after each round"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/iterative_pruning"))
    parser.add_argument(
        "--compile_latency",
        action="store_true",
        help="Use torch.compile only for latency measurement; training and accuracy evaluation stay eager",
    )
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "train_batch_size",
        "val_batch_size",
        "granularity",
        "rounds",
        "latency_repetitions",
    ):
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
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def main() -> None:
    """Alternate pruning and fine-tuning toward the final parameter limit."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    # Keep the same block-internal domains across all physical pruning rounds.
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
    print(f"Model: {options.model}; {len(targets)} block-internal pruning axes", flush=True)
    weights = MODELS[options.model][1]
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
    records: list[dict[str, Any]] = []

    def record(stage: str, **extra: object) -> None:
        """Evaluate the current model and persist this stage's measurements."""
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
    graph = DependencyGraph.build(model, args=(example,))
    alignment = Granularity(by_path=dict.fromkeys(targets, options.granularity))
    pruner = Pruner(model, graph=graph, granularity=alignment)
    space = pruner.discover_candidates(targets=targets)
    config["candidate_axes"] = [
        {"parameter": axis.tensor.paths[0], "dim": axis.dim} for axis in space.channel_axes
    ]
    config["discovery_exclusions"] = space.exclusions
    original_params = count_parameters(model)
    reduction = original_params - budget.max_params

    for index in range(1, options.rounds + 1):
        # Interpolate absolute caps from the fixed initial model. Alignment may
        # overshoot an intermediate target; the next round can then be a no-op.
        max_params = (
            budget.max_params
            if index == options.rounds
            else original_params - reduction * index // options.rounds
        )
        plan = make_plan(pruner, space, ParameterBudget(max_params))
        model, _ = pruner.apply(plan)
        graph = DependencyGraph.build(model, args=(example,))
        pruner = Pruner(model, graph=graph, granularity=alignment)
        space = pruner.discover_candidates(targets=targets)
        record(
            f"round_{index}_pruned",
            max_params=plan.selection_report.max_params,
            before_params=plan.selection_report.before_params,
            after_params=plan.selection_report.after_params,
            target_met=plan.selection_report.target_met,
            planning_trials=plan.selection_report.trials,
            planning_limit_reached=plan.selection_report.limit_reached,
        )

        # apply replaces Parameters. Recreate SGD instead of retaining stale state.
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
        for epoch in range(options.finetune_epochs):
            model.train()
            total, count = 0.0, 0
            with tqdm(
                train_loader,
                desc=f"Round {index} fine-tuning {epoch + 1}/{options.finetune_epochs} ({options.device})",
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
            print(
                json.dumps({"round": index, "epoch": epoch + 1, "task_loss": total / count}),
                flush=True,
            )
        # Restore the inference mode in which the current dependency graph was built.
        model.eval()
        if options.finetune_epochs:
            record(f"round_{index}_finetuned")

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
            "budget": {"initial_params": original_params, "max_params": budget.max_params},
            "config": config,
            "algorithm": {"completed_rounds": options.rounds},
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


def make_plan(pruner: Pruner, space: CandidateSpace, budget: ParameterBudget) -> PruningPlan:
    """Recompute normalized group scores once for the current pruning round."""
    return pruner.plan(space, budget=budget, strategy=Greedy(GroupMagnitude(p=2)))


if __name__ == "__main__":
    main()
