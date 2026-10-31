"""CPD: calibrated structured pruning with logit distillation on ImageNet.

This classification adaptation follows Comb, Prune, Distill (CPD): it
accumulates source-style Fisher scores during training intervals, physically
removes the least-important dependency-aware group batch after each
interval, and recovers with an unpruned frozen teacher. CPD's
segmentation-specific distillation losses are intentionally not part of this
ImageNet workflow.
"""

import argparse
import copy
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.vision_transformer import EncoderBlock
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph
from torch_kirigami.dep_graph.selection import gather_region
from torch_kirigami.pruning import (
    CPDLeastFirst,
    ParameterBudget,
    PlanningError,
    Pruner,
    TensorRecipe,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.pruning.cpd import CPDMetric, fisher_score_tensor
from torch_kirigami.pruning.plan import PruningResult


def pruning_targets(model: nn.Module) -> list[str]:
    """Return CPD's CNN filter and ViT FFN producer widths for supported models."""
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
    return targets


def target_widths(model: nn.Module, targets: Iterable[str]) -> dict[str, int]:
    """Return the surviving producer width at every configured CPD target."""
    widths = {}
    for path in targets:
        weight = getattr(model.get_submodule(path), "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim < 1:
            raise PlanningError(f"CPD target has no rank-one-or-higher weight: {path}")
        widths[path] = weight.shape[0]
    return widths


def scoring_groups(
    model: nn.Module, targets: Iterable[str]
) -> dict[str, tuple[tuple[str, ...], int]]:
    """Describe source-style output members and output/input group cardinality.

    Only ordinary torchvision block interiors are supported. BN affine members
    occur in both source lists; their repeated membership counts in the divisor,
    but their Fisher values are averaged only once. Consumer parameters contribute
    to cardinality, not to the numerator (including the FFN consumer bias).
    """
    groups = {}
    for target in targets:
        parent, _, name = target.rpartition(".")
        block = model.get_submodule(parent)
        producer = model.get_submodule(target)
        members = tuple(f"{target}.{name}" for name, _ in producer.named_parameters(recurse=False))
        if type(block) in (BasicBlock, Bottleneck) and name in ("conv1", "conv2"):
            if type(block) is BasicBlock and name != "conv1":
                raise PlanningError(f"Unsupported CPD residual output: {target}")
            index = int(name[-1])
            norm_path = f"{parent}.bn{index}" if parent else f"bn{index}"
            norm = model.get_submodule(norm_path)
            consumer = block.get_submodule(f"conv{index + 1}")
            if producer.groups != 1 or consumer.groups != 1 or type(norm) is not nn.BatchNorm2d:
                raise PlanningError(f"Unsupported CPD grouped convolution/normalization: {target}")
            norm_members = tuple(f"{norm_path}.{key}" for key, _ in norm.named_parameters())
            members += norm_members
            following_count = len(norm_members) + len(tuple(consumer.parameters(recurse=False)))
        elif (
            parent.endswith(".mlp")
            and name == "0"
            and type(model.get_submodule(parent.removesuffix(".mlp"))) is EncoderBlock
        ):
            consumer = block.get_submodule("3")
            following_count = len(tuple(consumer.parameters(recurse=False)))
        else:
            raise PlanningError(f"Unsupported CPD scoring group: {target}")
        groups[target] = (members, len(members) + following_count)
    return groups


def train_and_collect_cpd_metric(
    model: nn.Module,
    teacher: nn.Module,
    graph: DependencyGraph,
    targets: Iterable[str],
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    batches: int,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    distillation_weight: float,
) -> tuple[CPDMetric, dict[str, float | int]]:
    """Train one CPD interval; return its Fisher scores and mean interval losses."""
    if type(batches) is not int or batches <= 0:
        raise ValueError("CPD calibration batches must be positive")
    target_axes = {path: graph.parameter(f"{path}.weight").axis(0) for path in targets}
    groups = scoring_groups(model, target_axes)
    parameters = dict(model.named_parameters())
    scores = {
        axis: torch.zeros(axis.tensor.shape[axis.dim], dtype=torch.float64, device=device)
        for axis in target_axes.values()
    }
    outputs: dict[str, torch.Tensor] = {}
    hooks = [
        model.get_submodule(path).register_forward_hook(
            lambda _module, _inputs, output, path=path: outputs.__setitem__(path, output)
        )
        for path in dict.fromkeys(
            member.rpartition(".")[0] for members, _ in groups.values() for member in members
        )
    ]
    modes = [(module, module.training) for module in model.modules()]
    batches_seen, samples_seen = 0, 0
    task_total, distillation_total = 0.0, 0.0
    try:
        model.train()
        teacher.eval()
        for batch_index, (images, labels) in enumerate(islice(loader, batches), start=1):
            batches_seen = batch_index
            outputs.clear()
            images, labels = (
                images.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            task = F.cross_entropy(logits, labels)
            distillation = _logit_distillation(logits, teacher_logits, temperature)
            samples_seen += labels.numel()
            task_total += float(task.detach()) * labels.numel()
            distillation_total += float(distillation.detach()) * labels.numel()
            (task + distillation_weight * distillation).backward()
            for path, axis in target_axes.items():
                members, cardinality = groups[path]
                for member in members:
                    weight = parameters[member]
                    output = outputs.get(member.rpartition(".")[0])
                    if weight.grad is None or weight.grad.is_sparse:
                        raise PlanningError(f"CPD requires dense gradients for {member}")
                    if weight.shape[0] != len(scores[axis]):
                        raise PlanningError(f"CPD could not observe aligned group member {member}")
                    module = model.get_submodule(member.rpartition(".")[0])
                    scores[axis] += fisher_score_tensor(
                        weight,
                        weight.grad,
                        dim=0,
                        activation_elements=cpd_activation_elements(
                            module, output, weight.shape[0]
                        ),
                    ) / (len(members) * cardinality)
            optimizer.step()
        if not batches_seen:
            raise ValueError("CPD calibration loader produced no batches")
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in modes:
            module.training = training
        model.zero_grad(set_to_none=True)
    return (
        CPDMetric({axis: tuple(values.tolist()) for axis, values in scores.items()}),
        {
            "interval_batches": batches_seen,
            "interval_samples": samples_seen,
            "interval_task_loss": task_total / samples_seen,
            "interval_kd_loss": distillation_total / samples_seen,
        },
    )


def cpd_activation_elements(module: nn.Module, output: object, channels: int) -> int:
    """Count surviving NLC FFN or NCHW CNN activations, validating channel layout.

    For a compact FFN this is B*N*C_live, not B*C_original*C_live. The source
    hook's global default acts_chan_dim=1 is unsuitable for NLC outputs; do not
    reproduce that layout assumption. Each member uses its own module output.
    """
    rank, axis = (3, 2) if isinstance(module, nn.Linear) else (4, 1)
    if (
        not isinstance(output, torch.Tensor)
        or output.ndim != rank
        or output.shape[axis] != channels
        or output.numel() == 0
    ):
        raise PlanningError("CPD requires aligned NLC FFN or NCHW CNN output activations")
    return output.numel()


def make_cpd_strategy(model: nn.Module, metric: CPDMetric, cnn_cap: int) -> CPDLeastFirst:
    """Assign fixed-32 policy to the calibrated FFN producers, never to CNN axes."""
    ffn_axes = tuple(
        axis
        for axis in metric.scores
        if type(model.get_submodule(axis.tensor.paths[0].removesuffix(".weight"))) is nn.Linear
    )
    return CPDLeastFirst(metric, cnn_cap, ffn_axes)


def configure_determinism(seed: int) -> None:
    """Pin seeding and CUDA kernels so a pruning trajectory can be reproduced.

    CUDNN autotuning and non-deterministic convolution backward passes perturb
    Fisher scores in the last bits. That is enough to flip the argmin between
    near-tied candidates and change the deletion sequence between otherwise
    identical runs. ``CUBLAS_WORKSPACE_CONFIG`` is set before any CUDA context
    exists because cuBLAS reads it once, at handle creation.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def parse_args() -> argparse.Namespace:
    """Parse CPD's structural, calibration, and recovery settings."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CUDA kernels so pruning decisions reproduce across runs",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.05,
        help="Whole-model physical parameter reduction, not mask/channel sparsity",
    )
    parser.add_argument(
        "--pruning_interval_batches",
        type=int,
        default=50,
        help="Training batches used for one Fisher interval before one structural deletion",
    )
    parser.add_argument(
        "--control_intervals",
        type=int,
        default=0,
        help="Run this many CPD online-training intervals without pruning (requires ratio 0)",
    )
    parser.add_argument(
        "--candidates_per_interval",
        type=int,
        default=4,
        help="CNN adaptive deletion cap only (default 4); ViT FFN independently removes 32",
    )
    parser.add_argument(
        "--evaluate_every",
        type=int,
        default=1,
        help="Run full validation and complexity/latency measurement every N pruning steps",
    )
    parser.add_argument("--recovery_epochs", type=int, default=1)
    parser.add_argument(
        "--diagnostic_every",
        type=int,
        default=0,
        help="Compare fixed validation subset before/after deletion every N intervals; 0 disables",
    )
    parser.add_argument("--diagnostic_samples", type=int, default=256)
    parser.add_argument("--distillation_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("runs/cpd_pruning"))
    parser.add_argument("--compile_latency", action="store_true")
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to a progress.pt file written by an interrupted CPD run",
    )
    options = parser.parse_args()
    for name in (
        "train_batch_size",
        "val_batch_size",
        "pruning_interval_batches",
        "candidates_per_interval",
        "evaluate_every",
        "latency_repetitions",
        "diagnostic_samples",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "train_samples",
        "val_samples",
        "train_workers",
        "val_workers",
        "recovery_epochs",
        "latency_warmup",
        "control_intervals",
        "diagnostic_every",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    valid_scalars = all(
        math.isfinite(value) and value > 0 for value in (options.lr, options.temperature)
    )
    if (
        not 0 <= options.pruning_ratio < 1
        or not valid_scalars
        or not math.isfinite(options.distillation_weight)
        or options.distillation_weight < 0
    ):
        parser.error(
            "Require 0 <= pruning_ratio < 1, positive finite lr/temperature, "
            "and nonnegative distillation_weight"
        )
    if options.control_intervals and options.pruning_ratio:
        parser.error("--control_intervals requires --pruning_ratio 0")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install CUDA PyTorch or pass --device cpu")
    return options


def train_recovery(
    model: nn.Module,
    teacher: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: str,
    temperature: float,
    weight: float,
    round_index: int,
) -> dict[str, float | int]:
    """Fine-tune one compact CPD student epoch against frozen teacher logits."""
    model.train()
    teacher.eval()
    total, count, steps = 0.0, 0, 0
    for images, labels in tqdm(
        loader, desc=f"CPD recovery {round_index} ({device})", unit="batch", dynamic_ncols=True
    ):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        student = model(images)
        with torch.no_grad():
            teacher_logits = teacher(images)
        task = F.cross_entropy(student, labels)
        distillation = _logit_distillation(student, teacher_logits, temperature)
        loss = task + weight * distillation
        loss.backward()
        optimizer.step()
        steps += 1
        total += loss.detach().item() * labels.numel()
        count += labels.numel()
    if not count:
        raise ValueError("Cannot recover from an empty training split")
    print(json.dumps({"round": round_index, "recovery_loss": total / count}), flush=True)
    return {
        "recovery_epoch_steps": steps,
        "recovery_epoch_samples": count,
        "recovery_loss": total / count,
    }


def _logit_distillation(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Return conventional temperature-scaled classification logit distillation."""
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )


def _cycle_batches(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]], generator: torch.Generator
) -> "CyclingBatches":
    """Create a resumable, endless training-batch stream."""
    return CyclingBatches(loader, generator)


class CyclingBatches(Iterator[tuple[torch.Tensor, torch.Tensor]]):
    """Cycle a DataLoader while retaining enough state to replay its current epoch.

    Its generator belongs only to the explicit RandomSampler; DataLoader worker
    seeding has a separate generator. Saving only the sampler's current state
    cannot resume an interrupted partial traversal, because the sampler has
    already consumed the epoch's random permutation. This stream therefore saves
    its state immediately before iterator construction and replays the emitted
    prefix on restore.
    """

    def __init__(
        self,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        generator: torch.Generator,
    ) -> None:
        self._loader = loader
        self._generator = generator
        self._iterator: Iterator[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._epoch_rng: torch.Tensor | None = None
        self._next_epoch_rng: torch.Tensor | None = None
        self._batches = 0

    def __iter__(self) -> "CyclingBatches":
        """Return this stateful stream."""
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch, starting a new shuffled traversal as needed."""
        while True:
            if self._iterator is None:
                self._epoch_rng = self._generator.get_state().clone()
                self._iterator = iter(self._loader)
                self._next_epoch_rng = self._generator.get_state().clone()
                self._batches = 0
            try:
                batch = next(self._iterator)
            except StopIteration:
                if not self._batches:
                    raise ValueError("Cannot prune from an empty training split") from None
                self._iterator = None
                self._epoch_rng = None
                self._next_epoch_rng = None
                continue
            self._batches += 1
            return batch

    def state_dict(self) -> dict[str, torch.Tensor | int | None]:
        """Return replay state at a batch boundary without serializing workers."""
        return {
            "epoch_rng": self._epoch_rng,
            "next_epoch_rng": self._next_epoch_rng,
            "batches": self._batches,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore an interrupted traversal and validate its deterministic replay."""
        expected = {"epoch_rng", "next_epoch_rng", "batches"}
        if set(state) != expected:
            raise ValueError("Invalid CPD batch-stream checkpoint")
        epoch_rng, next_epoch_rng, batches = (
            state["epoch_rng"],
            state["next_epoch_rng"],
            state["batches"],
        )
        if epoch_rng is None and next_epoch_rng is None and batches == 0:
            return
        if (
            not isinstance(epoch_rng, torch.Tensor)
            or not isinstance(next_epoch_rng, torch.Tensor)
            or type(batches) is not int
            or batches <= 0
        ):
            raise ValueError("Invalid active CPD batch-stream checkpoint")
        self._generator.set_state(epoch_rng)
        self._iterator = iter(self._loader)
        if not torch.equal(self._generator.get_state(), next_epoch_rng):
            raise RuntimeError("CPD DataLoader generator did not replay its saved epoch")
        try:
            for _ in range(batches):
                next(self._iterator)
        except StopIteration as error:
            raise ValueError("CPD batch-stream checkpoint exceeds its epoch") from error
        self._epoch_rng = epoch_rng.clone()
        self._next_epoch_rng = next_epoch_rng.clone()
        self._batches = batches


def _compact_state_tensor(tensor: torch.Tensor, recipe: TensorRecipe) -> torch.Tensor:
    """Apply a structural recipe to an SGD state tensor with matching old shape."""
    parts = [gather_region(tensor, region) for region in recipe.segments]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=recipe.concat_dim)


def rebuild_sgd_optimizer(
    optimizer: torch.optim.SGD,
    result: PruningResult,
    parameters: Mapping[str, nn.Parameter],
) -> torch.optim.SGD:
    """Recreate SGD after compaction while retaining compatible per-parameter state.

    Physical pruning replaces only tensors touched by recipes. SGD's momentum
    buffers have the same old shapes as their parameters, so the corresponding
    recipe can compact them exactly. Scalar or non-parameter-shaped entries are
    copied unchanged for forward compatibility with PyTorch's SGD state.
    """
    if type(optimizer) is not torch.optim.SGD:
        raise TypeError("CPD optimizer migration currently supports torch.optim.SGD only")
    replacement = result.parameter_map
    groups = []
    for group in optimizer.param_groups:
        config = {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        groups.append(
            {
                "params": [replacement.get(parameter, parameter) for parameter in group["params"]],
                **config,
            }
        )
    rebuilt = torch.optim.SGD(groups)
    recipes = {
        parameters[recipe.tensor.paths[0]]: recipe
        for recipe in result.plan.recipes
        if recipe.tensor.kind == "parameter"
    }
    for old_parameter, old_state in optimizer.state.items():
        new_parameter = replacement.get(old_parameter, old_parameter)
        recipe = recipes.get(old_parameter)
        new_state = {}
        for key, value in old_state.items():
            if isinstance(value, torch.Tensor):
                new_state[key] = (
                    _compact_state_tensor(value, recipe)
                    if recipe is not None and tuple(value.shape) == tuple(old_parameter.shape)
                    else value.detach().clone()
                )
            else:
                new_state[key] = copy.deepcopy(value)
        rebuilt.state[new_parameter] = new_state
    return rebuilt


def _atomic_torch_save(value: object, path: Path) -> None:
    """Write a Torch payload atomically so a prior checkpoint remains recoverable."""
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_progress(
    output: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.SGD,
    pruning_steps: int,
    current_params: int,
    records: list[dict[str, Any]],
    config: Mapping[str, object],
    batch_stream: CyclingBatches,
    worker_generator: torch.Generator,
    recovery_epochs_completed: int = 0,
) -> None:
    """Persist a complete interval or recovery epoch without overwriting the live slot."""
    manifest = output / "progress.pt"
    previous = load_progress(manifest)["model_checkpoint"] if manifest.exists() else None
    slot = 1 if previous == "progress-model-0.pt" else 0
    checkpoint = output / f"progress-model-{slot}.pt"
    temporary = checkpoint.with_suffix(f"{checkpoint.suffix}.tmp")
    save_checkpoint(model, temporary)
    temporary.replace(checkpoint)
    _atomic_torch_save(
        {
            "model_checkpoint": checkpoint.name,
            "pruning_steps": pruning_steps,
            "current_params": current_params,
            "optimizer": optimizer.state_dict(),
            "records": records,
            "config": dict(config),
            "rng": torch.get_rng_state(),
            "cuda_rng": tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
            "batch_stream": batch_stream.state_dict(),
            "worker_rng": worker_generator.get_state(),
            "recovery_epochs_completed": recovery_epochs_completed,
        },
        output / "progress.pt",
    )


def load_progress(path: Path) -> dict[str, object]:
    """Load and minimally validate a CPD progress manifest."""
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {
        "model_checkpoint",
        "pruning_steps",
        "current_params",
        "optimizer",
        "records",
        "config",
        "rng",
        "cuda_rng",
        "batch_stream",
        "worker_rng",
        "recovery_epochs_completed",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise ValueError("Invalid CPD progress checkpoint")
    if (
        not isinstance(state["model_checkpoint"], str)
        or type(state["pruning_steps"]) is not int
        or type(state["current_params"]) is not int
        or not isinstance(state["records"], list)
        or not isinstance(state["config"], dict)
        or not isinstance(state["rng"], torch.Tensor)
        or not isinstance(state["cuda_rng"], (list, tuple))
        or not isinstance(state["batch_stream"], dict)
        or not isinstance(state["worker_rng"], torch.Tensor)
        or type(state["recovery_epochs_completed"]) is not int
        or state["recovery_epochs_completed"] < 0
    ):
        raise ValueError("Invalid CPD progress checkpoint values")
    return state


def _check_resume_config(config: Mapping[str, object], saved: Mapping[str, object]) -> None:
    """Reject resuming a compact model with training-affecting CLI differences."""
    keys = (
        "model",
        "data_dir",
        "device",
        "train_samples",
        "val_samples",
        "train_batch_size",
        "val_batch_size",
        "train_workers",
        "val_workers",
        "seed",
        "deterministic",
        "pruning_ratio",
        "pruning_interval_batches",
        "control_intervals",
        "candidates_per_interval",
        "distillation_weight",
        "temperature",
        "lr",
        "deletion_policy",
        "activation_normalization",
    )
    mismatched = [key for key in keys if config.get(key) != saved.get(key)]
    if mismatched:
        raise ValueError("CPD resume settings differ: " + ", ".join(mismatched))


def main() -> None:
    """Run online CPD-style pruning, final recovery, and checkpoint validation."""
    options = parse_args()
    if options.resume is not None:
        options.output = options.resume.parent
    if options.deterministic:
        configure_determinism(options.seed)
    else:
        torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    targets = pruning_targets(model)
    weights = MODELS[options.model][1]
    train, validation, dataset = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    sampler_generator = torch.Generator().manual_seed(options.seed)
    worker_generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train,
        batch_size=options.train_batch_size,
        sampler=RandomSampler(train, generator=sampler_generator),
        generator=worker_generator,
        num_workers=options.train_workers,
        persistent_workers=options.train_workers > 0,
        multiprocessing_context="spawn" if options.train_workers else None,
        pin_memory=options.device == "cuda",
    )
    val_loader = DataLoader(
        validation,
        generator=torch.Generator().manual_seed(options.seed),
        batch_size=options.val_batch_size,
        num_workers=options.val_workers,
        persistent_workers=options.val_workers > 0,
        multiprocessing_context="spawn" if options.val_workers else None,
        pin_memory=options.device == "cuda",
    )
    diagnostic_loader = DataLoader(
        Subset(validation, range(min(options.diagnostic_samples, len(validation)))),
        batch_size=options.val_batch_size,
        generator=torch.Generator().manual_seed(options.seed),
        num_workers=0,
    )
    # Epoch-seeded recovery is independent of any partially consumed online
    # iterator. Workers are recreated each epoch so resume has identical seeds.
    recovery_generator = torch.Generator()
    recovery_loader = DataLoader(
        train,
        batch_size=options.train_batch_size,
        shuffle=True,
        generator=recovery_generator,
        num_workers=options.train_workers,
        multiprocessing_context="spawn" if options.train_workers else None,
        pin_memory=options.device == "cuda",
    )
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    final_budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    original_params = sum(parameter.numel() for parameter in model.parameters())
    records: list[dict[str, Any]] = []
    options.output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(options).items()
    }
    config.update(
        weights=str(weights),
        dataset=dataset,
        max_params=final_budget.max_params,
        pruning_scope="block_internal",
        pruning_targets=targets,
        scoring_groups=scoring_groups(model, targets),
        initial_params=original_params,
        stopping_metric="whole_model_parameter_count",
        deletion_policy={
            path: "ffn_fixed_32_keep_one"
            if type(model.get_submodule(path)) is nn.Linear
            else "cnn_adaptive_keep_one"
            for path in targets
        },
        activation_normalization="surviving_output_elements_NLC_or_NCHW",
        training_budget={
            "online_stop": "control_intervals" if options.control_intervals else "parameter_target",
            "batches_per_interval": options.pruning_interval_batches,
            "online_intervals": options.control_intervals if options.control_intervals else None,
            "recovery_epochs": options.recovery_epochs,
            "recovery_steps_per_epoch": len(recovery_loader),
            "recovery_samples_per_epoch": len(train),
        },
    )
    online_optimizer_steps, online_samples = 0, 0
    recovery_optimizer_steps, recovery_samples = 0, 0

    def record(stage: str, *, full_evaluation: bool, **extra: object) -> None:
        """Write a stage row, optionally paying for validation and latency measurement."""
        params = sum(parameter.numel() for parameter in model.parameters())
        row: dict[str, object] = {
            "stage": stage,
            "full_evaluation": full_evaluation,
            "#Params": params,
            "parameter_reduction_ratio": (original_params - params) / original_params,
            "parameter_target_reached": params <= final_budget.max_params,
            "target_overshoot_params": max(0, final_budget.max_params - params),
            "online_optimizer_steps": online_optimizer_steps,
            "online_samples": online_samples,
            "target_widths": target_widths(model, targets),
            "recovery_optimizer_steps": recovery_optimizer_steps,
            "recovery_samples": recovery_samples,
            "total_optimizer_steps": online_optimizer_steps + recovery_optimizer_steps,
            "total_samples": online_samples + recovery_samples,
            **extra,
        }
        if full_evaluation:
            accuracy = evaluate(
                model,
                val_loader,
                options.device,
                description=f"{stage} evaluation",
            )
            baseline = records[0]["top1"] if records else accuracy["top1"]
            row.update(
                {
                    **accuracy,
                    "top1_delta_pp": accuracy["top1"] - baseline,
                    **measure_model(model, example, options),
                }
            )
        else:
            row.update({"top1": None, "top5": None, "loss": None, "top1_delta_pp": None})
        records.append(row)
        (options.output / "metrics.json").write_text(
            json.dumps({"config": config, "stages": records}, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)

    train_batches = _cycle_batches(train_loader, sampler_generator)
    dense_control = options.control_intervals > 0
    recovery_epochs_completed = 0
    if options.resume is None:
        record(
            "pretrained",
            full_evaluation=True,
            online_optimizer_steps=0,
            online_samples=0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
        pruning_steps = 0
        current_params = original_params
    else:
        progress = load_progress(options.resume)
        saved_config = progress["config"]
        assert isinstance(saved_config, dict)
        _check_resume_config(config, saved_config)
        checkpoint = options.resume.parent / str(progress["model_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CPD progress model checkpoint is missing: {checkpoint}")
        model = load_checkpoint(model, checkpoint, map_location=options.device).eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
        optimizer.load_state_dict(progress["optimizer"])
        worker_generator.set_state(progress["worker_rng"])
        if (
            progress["current_params"] > final_budget.max_params
            or progress["pruning_steps"] < options.control_intervals
        ):
            train_batches.load_state_dict(progress["batch_stream"])
        # Restore model RNG after replay; loader reconstruction must not perturb dropout.
        torch.set_rng_state(progress["rng"])
        if options.device == "cuda":
            cuda_rng = progress["cuda_rng"]
            if not cuda_rng:
                raise ValueError("CPD progress checkpoint has no CUDA RNG state")
            torch.cuda.set_rng_state_all(cuda_rng)
        pruning_steps = progress["pruning_steps"]
        current_params = progress["current_params"]
        records = progress["records"]
        if current_params != sum(parameter.numel() for parameter in model.parameters()):
            raise ValueError("CPD progress parameter count does not match its compact model")
        if not records:
            raise ValueError("CPD progress checkpoint has no pretrained baseline")
        online_optimizer_steps = sum(int(row.get("interval_batches", 0)) for row in records)
        online_samples = sum(int(row.get("interval_samples", 0)) for row in records)
        recovery_epochs_completed = progress["recovery_epochs_completed"]
        if options.recovery_epochs < recovery_epochs_completed:
            raise ValueError("recovery_epochs is a total and cannot be below completed epochs")
        recovery_optimizer_steps = sum(int(row.get("recovery_epoch_steps", 0)) for row in records)
        recovery_samples = sum(int(row.get("recovery_epoch_samples", 0)) for row in records)
        print(f"Resumed CPD after pruning step {pruning_steps}", flush=True)
    while current_params > final_budget.max_params or pruning_steps < options.control_intervals:
        graph = DependencyGraph.build(model.eval(), args=(example,))
        metric, interval = train_and_collect_cpd_metric(
            model,
            teacher,
            graph,
            targets,
            train_batches,
            options.device,
            options.pruning_interval_batches,
            optimizer,
            options.temperature,
            options.distillation_weight,
        )
        pruning_steps += 1
        online_optimizer_steps += int(interval["interval_batches"])
        online_samples += int(interval["interval_samples"])
        if dense_control:
            record(
                f"control_step_{pruning_steps}",
                full_evaluation=(
                    pruning_steps % options.evaluate_every == 0
                    or pruning_steps == options.control_intervals
                ),
                control_only=True,
                final_max_params=final_budget.max_params,
                pruning_interval_batches=options.pruning_interval_batches,
                online_optimizer_steps=online_optimizer_steps,
                online_samples=online_samples,
                **interval,
            )
            save_progress(
                options.output,
                model=model,
                optimizer=optimizer,
                pruning_steps=pruning_steps,
                current_params=current_params,
                records=records,
                config=config,
                batch_stream=train_batches,
                worker_generator=worker_generator,
            )
            continue
        pruner = Pruner(model, graph=graph)
        space = pruner.discover_candidates(targets=targets)
        strategy = make_cpd_strategy(model, metric, options.candidates_per_interval)
        ranking = {
            axis.tensor.paths[0]: {
                "width": len(values),
                "k": strategy.channel_count(axis),
                "prefix_sum": math.fsum(sorted(values)[: strategy.channel_count(axis)])
                if strategy.channel_count(axis)
                else None,
            }
            for axis, values in metric.scores.items()
        }
        plan = pruner.plan(
            space,
            # CPD selects one group by importance; the final parameter target
            # controls the outer loop rather than this one-selection plan.
            budget=ParameterBudget(current_params),
            strategy=strategy,
        )
        selected_scores = {}
        for candidate in space.candidates:
            if candidate.key in plan.selected:
                axis = candidate.axis
                indices = {
                    index
                    for seed in candidate.remove
                    if seed.tensor == axis.tensor
                    for index in seed.fully_selected_indices(axis.dim)
                }
                selected_scores[candidate.key] = math.fsum(metric.scores[axis][i] for i in indices)
        diagnostic = {}
        if options.diagnostic_every and pruning_steps % options.diagnostic_every == 0:
            diagnostic["before"] = evaluate(
                model,
                diagnostic_loader,
                options.device,
                description="Before deletion",
            )
        parameters = dict(model.named_parameters())
        model, result = pruner.apply(plan)
        optimizer = rebuild_sgd_optimizer(optimizer, result, parameters)
        if diagnostic:
            diagnostic["after"] = evaluate(
                model,
                diagnostic_loader,
                options.device,
                description="After deletion",
            )
            diagnostic["top1_delta_pp"] = diagnostic["after"]["top1"] - diagnostic["before"]["top1"]
            diagnostic["loss_delta"] = diagnostic["after"]["loss"] - diagnostic["before"]["loss"]
        current_params = sum(parameter.numel() for parameter in model.parameters())
        if current_params >= plan.selection_report.before_params:
            raise PlanningError("CPD structural application did not reduce parameter count")
        record(
            f"pruning_step_{pruning_steps}",
            full_evaluation=(
                pruning_steps % options.evaluate_every == 0
                or current_params <= final_budget.max_params
            ),
            selected=list(plan.selected),
            selected_scores=selected_scores,
            selected_score_sum=math.fsum(selected_scores.values()),
            axis_ranking=ranking,
            deletion_diagnostic=diagnostic or None,
            removed_params=plan.selection_report.before_params - current_params,
            before_params=plan.selection_report.before_params,
            after_params=current_params,
            final_max_params=final_budget.max_params,
            pruning_interval_batches=options.pruning_interval_batches,
            online_optimizer_steps=online_optimizer_steps,
            online_samples=online_samples,
            **interval,
        )
        save_progress(
            options.output,
            model=model,
            optimizer=optimizer,
            pruning_steps=pruning_steps,
            current_params=current_params,
            records=records,
            config=config,
            batch_stream=train_batches,
            worker_generator=worker_generator,
        )
    if options.resume is None and pruning_steps == 0:
        save_progress(
            options.output,
            model=model,
            optimizer=optimizer,
            pruning_steps=pruning_steps,
            current_params=current_params,
            records=records,
            config=config,
            batch_stream=train_batches,
            worker_generator=worker_generator,
        )
    for epoch in range(recovery_epochs_completed, options.recovery_epochs):
        recovery_generator.manual_seed(options.seed + epoch + 1)
        recovery = train_recovery(
            model,
            teacher,
            recovery_loader,
            optimizer,
            options.device,
            options.temperature,
            options.distillation_weight,
            epoch + 1,
        )
        recovery_epochs_completed = epoch + 1
        recovery_optimizer_steps += int(recovery["recovery_epoch_steps"])
        recovery_samples += int(recovery["recovery_epoch_samples"])
        model.eval()
        record(
            "recovered"
            if recovery_epochs_completed == options.recovery_epochs
            else f"recovery_epoch_{recovery_epochs_completed}",
            full_evaluation=True,
            pruning_steps=pruning_steps,
            online_optimizer_steps=online_optimizer_steps,
            online_samples=online_samples,
            recovery_epochs_completed=recovery_epochs_completed,
            recovery_optimizer_steps=recovery_optimizer_steps,
            recovery_samples=recovery_samples,
            total_optimizer_steps=online_optimizer_steps + recovery_optimizer_steps,
            total_samples=online_samples + recovery_samples,
            **recovery,
        )
        save_progress(
            options.output,
            model=model,
            optimizer=optimizer,
            pruning_steps=pruning_steps,
            current_params=current_params,
            records=records,
            config=config,
            batch_stream=train_batches,
            worker_generator=worker_generator,
            recovery_epochs_completed=recovery_epochs_completed,
        )
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
            "optimizer": optimizer.state_dict() if optimizer else {},
            "config": config,
            "algorithm": {
                "name": "CPD",
                "pruning_steps": pruning_steps,
                "pruning_interval_batches": options.pruning_interval_batches,
                "control_intervals": options.control_intervals,
                "online_stop_reason": "control_intervals_completed"
                if dense_control
                else "parameter_target_reached",
                "online_optimizer_steps": online_optimizer_steps,
                "online_samples": online_samples,
                "recovery_epochs_completed": recovery_epochs_completed,
                "total_optimizer_steps": online_optimizer_steps + recovery_optimizer_steps,
                "total_samples": online_samples + recovery_samples,
            },
            "rng": torch.get_rng_state(),
            "sampler_rng": sampler_generator.get_state(),
            "worker_rng": worker_generator.get_state(),
            "batch_stream": train_batches.state_dict(),
        },
        options.output / "training.pt",
    )
    print(f"Checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
