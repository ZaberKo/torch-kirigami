"""Shared ImageNet training, evaluation and physical pruning helpers."""

import argparse
import json
import math
from pathlib import Path

import torch
from imagenet_data import evaluate, load_images, train_epoch
from imagenet_models import MODELS, build_space, layers_for, make_model, positions
from model_metrics import add_arguments, measure_model
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torch_kirigami.pruning import ParameterGroup, Pruner, load_checkpoint, save_checkpoint
from torch_kirigami.sparsity import CumulativeChannelBudget


def arguments(description, *, configure=None, rounds=1):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument(
        "--layers", help="Comma-separated module paths, or all; default: first block"
    )
    parser.add_argument(
        "--data-dir", type=Path, help="Optional dataset directory; default: HF cache"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--train-samples", type=int, default=512, help="0 uses the full train split"
    )
    parser.add_argument(
        "--val-samples", type=int, default=0, help="0 uses all 50,000 validation images"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--group-size", type=int, default=8, help="Adjacent channels per candidate")
    parser.add_argument("--rounds", type=int, default=rounds)
    parser.add_argument("--sparse-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=0)
    parser.add_argument("--strength", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/imagenet"))
    add_arguments(parser)
    if configure:
        configure(parser)
    options = parser.parse_args()
    choices = layers_for(options.model)
    options.layers = (
        choices
        if options.layers == "all"
        else tuple(options.layers.split(","))
        if options.layers
        else choices[:1]
    )
    if len(set(options.layers)) != len(options.layers) or any(
        layer not in choices for layer in options.layers
    ):
        parser.error("--layers must name distinct model blocks or all")
    for name in (
        "batch_size",
        "threads",
        "rounds",
        "group_size",
        "benchmark_batch_size",
        "repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "train_samples",
        "val_samples",
        "workers",
        "sparse_epochs",
        "finetune_epochs",
        "warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if (
        not 0 < options.ratio < 1
        or not math.isfinite(options.strength)
        or options.strength < 0
        or not math.isfinite(options.lr)
        or options.lr <= 0
    ):
        parser.error("Require 0 < ratio < 1, finite nonnegative strength and positive finite lr")
    return options


class Experiment:
    """Common mechanics; each standalone script owns its algorithm and stages."""

    def __init__(self, options, *, training=False, gated=False, calibration=False):
        self.options, self.gated = options, gated
        torch.manual_seed(options.seed)
        torch.set_num_threads(options.threads)
        self.weights = MODELS[options.model][1]
        print(f"Loading pretrained {options.model} and local ImageNet...", flush=True)
        need_train = training or options.finetune_epochs > 0 or calibration
        train, validation, dataset_info = load_images(
            self.weights,
            options.data_dir,
            need_train=need_train,
            train_samples=options.train_samples,
            val_samples=options.val_samples,
            seed=options.seed,
        )
        self.model = make_model(options.model, options.layers, gated=gated).to(options.device)
        self.generator = torch.Generator().manual_seed(options.seed)
        self.train_loader = (
            DataLoader(
                train, batch_size=options.batch_size, generator=self.generator, num_workers=0
            )
            if train is not None
            else None
        )
        self.val_loader = DataLoader(
            validation, batch_size=options.batch_size, num_workers=options.workers
        )
        self.example = torch.zeros(1, 3, 224, 224, device=options.device)
        self.records = []
        self.config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        }
        self.config.update(weights=str(self.weights), dataset=dataset_info, gated=gated)
        options.output.mkdir(parents=True, exist_ok=True)
        self.record("pretrained")
        self.model.train(training or options.finetune_epochs > 0)
        self.rebuild()
        self.accounting = CumulativeChannelBudget(self.space)
        self.initial_width = sum(axis.tensor.shape[0] for axis in self.space.axes)
        self.optimizer = self.new_optimizer()

    def new_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.options.lr, momentum=0.9)

    def rebuild(self):
        self.space = build_space(
            self.model,
            self.example,
            self.options.model,
            self.options.layers,
            gated=self.gated,
            group_size=self.options.group_size,
        )

    def record(self, stage, **extra):
        print(f"Evaluating {stage}...", flush=True)
        accuracy = evaluate(self.model, self.val_loader, self.options.device, progress_every=100)
        metrics = measure_model(self.model, self.example, self.options)
        baseline = self.records[0]["top1"] if self.records else accuracy["top1"]
        row = {
            "stage": stage,
            **accuracy,
            "top1_delta_pp": accuracy["top1"] - baseline,
            **metrics,
            **extra,
        }
        self.records.append(row)
        print(json.dumps(row), flush=True)
        (self.options.output / "metrics.json").write_text(
            json.dumps({"config": self.config, "stages": self.records}, indent=2) + "\n"
        )

    def train(self, epochs=1, *, regularizer=None, strength=0.0, after_step=None):
        if epochs == 0:
            return
        if self.train_loader is None:
            raise ValueError("This workflow did not request ImageNet training data")
        for epoch in range(epochs):
            metrics = train_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.options.device,
                regularizer,
                strength,
                after_step=after_step,
            )
            print(json.dumps({"epoch": epoch + 1, "strength": strength, **metrics}), flush=True)

    def task_gradients(self):
        if self.train_loader is None:
            raise ValueError("Taylor requires a separate ImageNet training batch")
        self.model.zero_grad(set_to_none=True)
        images, labels = next(iter(self.train_loader))
        # Calibration scores the current inference model; it must not adapt BN
        # statistics before the immediate post-pruning accuracy measurement.
        modes = [(module, module.training) for module in self.model.modules()]
        try:
            self.model.eval()
            F.cross_entropy(
                self.model(images.to(self.options.device)), labels.to(self.options.device)
            ).backward()
        finally:
            for module, training in modes:
                module.training = training

    def scale_paths(self, kind):
        return tuple(
            gate + ".weight" if kind == "gate" else bn
            for _, gate, bn in positions(self.options.model, self.options.layers, gated=self.gated)
        )

    def plan(self, ratio=None, *, metric="magnitude"):
        """Rank known internal channel groups, then validate their joint physical plan.

        Magnitude/Taylor score producer weights (not the full dependency group).
        Vectorized per-layer scores avoid thousands of per-channel model copies.
        Joint dependency completeness and budget checks remain in Pruner.plan.
        """
        ratio = self.options.ratio if ratio is None else ratio
        budget = self.accounting.budget(self.space, ratio)
        scores = {}
        for axis, (producer, gate, bn) in zip(
            self.space.axes,
            positions(self.options.model, self.options.layers, gated=self.gated),
            strict=True,
        ):
            weight = self.model.get_parameter(producer)
            if metric == "taylor":
                if weight.grad is None:
                    raise ValueError("Collect task-only gradients before Taylor selection")
                values = (
                    (weight.detach().float() * weight.grad.detach().float()).abs().flatten(1).sum(1)
                )
            elif metric == "bn":
                values = self.model.get_parameter(bn).detach().abs()
            elif metric == "gate":
                module = self.model.get_submodule(gate)
                values = (module.weight.detach() * module.mask).abs()
            else:
                values = weight.detach().float().flatten(1).square().sum(1)
            for candidate in self.space.candidates:
                if candidate.axis == axis:
                    indices = candidate.remove[0].fully_selected_indices(0)
                    scores[candidate.key] = values[list(indices)].sum().item()
        if not all(math.isfinite(value) for value in scores.values()):
            raise ValueError("Nonfinite pruning score")

        def select(context):
            selected = []
            for axis, target in zip(context.axes, context.targets, strict=True):
                removed = 0
                candidates = sorted(
                    (c for c in context.candidates if c.axis == axis),
                    key=lambda c: (scores[c.key], c.key),
                )
                for candidate in candidates:
                    count = len(candidate.remove[0].fully_selected_indices(0))
                    if removed + count <= min(target, axis.tensor.shape[0] - 1):
                        selected.append(candidate.key)
                        removed += count
            return selected

        return Pruner(self.model, graph=self.space.graph).plan(
            budget=budget, candidates=self.space.candidates, strategy=select
        )

    def groups(self, plan=None):
        selected = set(plan.selected) if plan is not None else None
        return self.space.parameter_groups(
            c for c in self.space.candidates if selected is None or c.key in selected
        )

    def union_group(self, plan):
        impact = self.space.impact(c for c in self.space.candidates if c.key in plan.selected)
        return (ParameterGroup(self.space.graph, impact.parameters),) if impact.parameters else ()

    def prune(self, plan, *, stage="pruned", target_ratio=None):
        _, result = Pruner(self.model, graph=self.space.graph).apply(plan)
        self.rebuild()
        self.accounting.update(result, self.space)
        self.optimizer = self.new_optimizer()
        actual = 1 - sum(axis.tensor.shape[0] for axis in self.space.axes) / self.initial_width
        self.record(
            stage,
            target_ratio=self.options.ratio if target_ratio is None else target_ratio,
            actual_ratio=actual,
            target=plan.budget.targets,
            removed=plan.budget.removed,
            shortfall=plan.budget.shortfall,
        )

    def finetune(self, *, stage="finetuned"):
        if self.options.finetune_epochs:
            self.train(self.options.finetune_epochs)
            self.record(stage)

    def finish(self, **algorithm):
        self.model.eval()
        save_checkpoint(self.model, self.options.output / "model.pt")
        restored = load_checkpoint(
            make_model(self.options.model, self.options.layers, pretrained=False, gated=self.gated),
            self.options.output / "model.pt",
            map_location=self.options.device,
        ).eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(self.example), self.model(self.example))
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "budget": self.accounting.state_dict(),
                "config": self.config,
                "algorithm": algorithm,
                "rng": torch.get_rng_state(),
                "data_rng": self.generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            self.options.output / "training.pt",
        )
        print(f"Checkpoint verified; results: {self.options.output}", flush=True)
