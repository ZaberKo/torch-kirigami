"""OSSCAR: sequential teacher reconstruction, grouped deletion and local swaps.

Implements the ICML 2024 quadratic reconstruction objective and grouped search.
Whole-model parameter allocation, sampled calibration positions and torchvision
ViT FFNs are explicit adaptations, not reproductions of published accuracy.
"""

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sized
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from imagenet_data import evaluate, load_images
from imagenet_models import MODELS, make_model
from model_metrics import measure_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from torch_kirigami import DependencyGraph, OperationContext
from torch_kirigami.measurement import count_parameters
from torch_kirigami.pruning import (
    Granularity,
    ParameterBudget,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


def _preserves_channels(operation: OperationContext, convolution: bool) -> bool:
    """Recognize paths whose retained channels do not depend on removed channels.

    This numerical contract is stronger than structural dependency propagation:
    LayerNorm and channel softmax cannot be treated like an elementwise activation.
    Evaluation BatchNorm is pointwise with running statistics; without them,
    pruning must act on its channel axis rather than a statistics-reduction axis.
    """
    module = operation.module
    if type(module) in (nn.BatchNorm1d, nn.BatchNorm2d):
        return (
            not module.training
            and bool(operation.inputs)
            and (
                convolution
                or len(operation.inputs[0].shape) == 2
                or (module.running_mean is not None and module.running_var is not None)
            )
        )
    if type(module) in (nn.Dropout, nn.Dropout2d):
        return not module.training
    if type(module) in (
        nn.Identity,
        nn.ReLU,
        nn.ReLU6,
        nn.GELU,
        nn.SiLU,
        nn.LeakyReLU,
        nn.ELU,
        nn.Sigmoid,
        nn.Tanh,
        nn.Softplus,
        nn.Hardswish,
        nn.Hardtanh,
    ):
        return True
    if convolution and type(module) in (
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveMaxPool2d,
    ):
        return len(operation.outputs) == 1
    if operation.node.op == "call_function":
        if operation.node.target is F.dropout:
            return operation.argument("training", 2, True) is False
        return operation.node.target in (
            torch.relu,
            torch.sigmoid,
            torch.tanh,
            F.relu,
            F.relu6,
            F.gelu,
            F.silu,
            F.leaky_relu,
            F.elu,
            F.softplus,
            F.hardswish,
            F.hardtanh,
        )
    return operation.node.op == "call_method" and operation.node.target in (
        "relu",
        "relu_",
        "sigmoid",
        "tanh",
    )


def discover_reconstruction_pairs(
    graph: DependencyGraph,
) -> tuple[dict[str, str], dict[str, str]]:
    """Find proven producer/consumer chains without model names or path conventions.

    Accept exact Linear or ordinary Conv2d endpoints, connected by one unbranched
    channel-independent path. Shared calls, parameters or channel buffers, mixing/reindexing
    operators and unresolved dependencies are excluded with reasons. Adjacent
    pairs may overlap: sequential reconstruction re-calibrates after every change.
    """
    graph.validate()
    operations = graph.operations()
    by_node = {operation.node: operation for operation in operations}
    calls = Counter(
        id(operation.module) for operation in operations if operation.module is not None
    )
    binding_users: dict[str, set[str]] = defaultdict(set)
    for operation in operations:
        for tensor in (*operation.inputs, *operation.bindings.values()):
            # A shared BN mean/variance couples otherwise separate chains.
            # Its scalar batch counter does not carry a removable coordinate.
            if tensor.kind in ("parameter", "buffer") and tensor.shape:
                binding_users[tensor.id].add(operation.node.name)

    def independent(operation: OperationContext) -> bool:
        return (operation.module is None or calls[id(operation.module)] == 1) and all(
            len(binding_users[tensor.id]) == 1
            for tensor in operation.bindings.values()
            if tensor.kind in ("parameter", "buffer") and tensor.shape
        )

    pairs, exclusions = {}, {}
    for producer in operations:
        if type(producer.module) not in (nn.Linear, nn.Conv2d):
            continue
        path = producer.module_path
        reason = "No single supported consumer before a branch or model output"
        if not independent(producer):
            exclusions[path] = "Producer is called repeatedly or shares parameters"
            continue
        if type(producer.module) is nn.Conv2d and producer.module.groups != 1:
            exclusions[path] = "Grouped/depthwise convolution requires a different reconstruction"
            continue
        current = producer
        while len(current.node.users) == 1:
            following = next(iter(current.node.users))
            consumer = by_node.get(following)
            if consumer is None or len(consumer.inputs) != 1 or len(consumer.outputs) != 1:
                break
            if type(consumer.module) is type(producer.module):
                if not independent(consumer):
                    reason = "Consumer is called repeatedly or shares parameters"
                    break
                if type(consumer.module) is nn.Conv2d and (
                    consumer.module.groups != 1 or consumer.module.padding_mode != "zeros"
                ):
                    reason = "Consumer reconstruction requires ordinary zero-padded Conv2d"
                    break
                if producer.module.weight.shape[0] != consumer.module.weight.shape[1]:
                    reason = "Producer output and consumer input widths do not match"
                    break
                # A sample structural request verifies the declared path against
                # the graph's actual rules, including aliases and unsupported ops.
                source = graph.parameter(f"{path}.weight")
                target = graph.parameter(f"{consumer.module_path}.weight")
                impact = graph.propagate(remove=[source.axis(0).select([0])])
                if impact.status != "resolved" or impact.selection(target) != target.axis(1).select(
                    [0]
                ):
                    reason = "Dependency analysis cannot prove an isolated matching input-channel removal"
                    break
                pairs[path] = consumer.module_path
                break
            if not _preserves_channels(consumer, type(producer.module) is nn.Conv2d):
                reason = f"{consumer.node.name} is not a supported channel-independent operation"
                break
            # Parameter-free activation modules can be reused safely. Stateful
            # normalization requires a single call and independent parameters.
            if (
                consumer.bindings or type(consumer.module) in (nn.BatchNorm1d, nn.BatchNorm2d)
            ) and not independent(consumer):
                reason = (
                    "Intermediate normalization has shared calls, parameters or channel buffers"
                )
                break
            current = consumer
        if path not in pairs:
            exclusions[path] = reason
    return pairs, exclusions


def feature_rows(layer: nn.Module, inputs: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    """Sample the consumer design matrix, unfolding at most one image at a time.

    Conv rows follow batch, output-height, output-width order. Columns follow
    input-channel, kernel-height, kernel-width order, matching weight.flatten(1).
    Row IDs must be sorted so sampled images can be processed consecutively.
    """
    if type(layer) is nn.Linear:
        return inputs.reshape(-1, layer.in_features)[row_ids]
    if type(layer) is not nn.Conv2d or layer.groups != 1 or layer.padding_mode != "zeros":
        raise ValueError("Feature sampling supports Linear and ordinary zero-padded Conv2d")
    same_padding = layer.padding == "same"
    padding = (0, 0) if isinstance(layer.padding, str) else layer.padding
    totals = (
        tuple(
            dilation * (kernel - 1)
            for dilation, kernel in zip(layer.dilation, layer.kernel_size, strict=True)
        )
        if same_padding
        else tuple(2 * amount for amount in padding)
    )
    height, width = (
        (size + total - dilation * (kernel - 1) - 1) // stride + 1
        for size, total, dilation, kernel, stride in zip(
            inputs.shape[-2:],
            totals,
            layer.dilation,
            layer.kernel_size,
            layer.stride,
            strict=True,
        )
    )
    locations = height * width
    if row_ids.ndim != 1 or row_ids.numel() == 0 or bool((row_ids[1:] < row_ids[:-1]).any()):
        raise ValueError("Sampling requires nonempty sorted row IDs")
    image_ids = row_ids // locations
    rows = []
    for image in image_ids.unique_consecutive().tolist():
        image_input = inputs[image : image + 1]
        if same_padding:
            # Even kernels can require asymmetric `same` padding, which
            # unfold's symmetric padding argument cannot express.
            vertical, horizontal = totals
            image_input = F.pad(
                image_input,
                (
                    horizontal // 2,
                    horizontal - horizontal // 2,
                    vertical // 2,
                    vertical - vertical // 2,
                ),
            )
        columns = F.unfold(
            image_input,
            layer.kernel_size,
            dilation=layer.dilation,
            padding=padding,
            stride=layer.stride,
        )[0].T
        rows.append(columns[row_ids[image_ids == image] % locations])
    return torch.cat(rows)


@dataclass
class ReconstructionMoments:
    """Streaming design Gram and teacher cross-product for one consumer."""

    gram: torch.Tensor | None = None
    cross: torch.Tensor | None = None
    count: int = 0

    @torch.no_grad()
    def update(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate matched observations in float64 without retaining activations."""
        if (
            features.ndim != 2
            or targets.ndim != 2
            or len(features) != len(targets)
            or not features.numel()
            or not targets.numel()
            or not features.is_floating_point()
            or not targets.is_floating_point()
            or not torch.isfinite(features).all()
            or not torch.isfinite(targets).all()
        ):
            raise ValueError("Reconstruction needs finite, nonempty matched real matrices")
        x, y = features.detach().double(), targets.detach().double()
        gram, cross = x.T @ x, x.T @ y
        if self.gram is not None and (
            self.gram.shape != gram.shape
            or self.cross.shape != cross.shape
            or self.gram.device != gram.device
        ):
            raise ValueError("Reconstruction dimensions and device must remain fixed")
        if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
            raise ValueError("Nonfinite reconstruction moments")
        self.gram = gram if self.gram is None else self.gram + gram
        self.cross = cross if self.cross is None else self.cross + cross
        self.count += len(x)

    def system(self, original: torch.Tensor, damping: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return averaged moments with ridge regularization toward original weights.

        The tiny positive scale floor handles entirely dead calibration inputs.
        With damping=0 a singular system is rejected by the solver.
        """
        if not math.isfinite(damping) or damping < 0:
            raise ValueError("damping must be finite and nonnegative")
        if not self.count or self.gram is None or self.cross is None:
            raise ValueError("No reconstruction observations collected")
        if original.shape != self.cross.shape or not torch.isfinite(original).all():
            raise ValueError("Original coefficients must match finite reconstruction dimensions")
        h, g = self.gram / self.count, self.cross / self.count
        scale = h.diagonal().mean().clamp_min(torch.finfo(h.dtype).eps)
        ridge = damping * scale
        h.diagonal().add_(ridge)
        g = g + ridge * original.detach().to(g)
        return h, g


def collect_reconstruction(
    model: nn.Module,
    teacher: nn.Module,
    consumer_path: str,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    max_batches: int,
    max_rows: int,
) -> ReconstructionMoments:
    """Match current consumer inputs to original teacher preactivation outputs.

    Sample up to max_rows uniformly spaced positions per batch; use identical
    position IDs on both sides. Labels are ignored. Existing modes, hooks and
    gradients are preserved. max_batches=0 traverses the whole training loader.
    """
    if (
        type(max_batches) is not int
        or max_batches < 0
        or type(max_rows) is not int
        or max_rows <= 0
    ):
        raise ValueError("Require nonnegative max_batches and positive max_rows")
    if model is teacher:
        raise ValueError("Sequential reconstruction requires a separate fixed teacher")
    consumer = model.get_submodule(consumer_path)
    original = teacher.get_submodule(consumer_path)
    if type(consumer) not in (nn.Conv2d, nn.Linear) or type(original) is not type(consumer):
        raise ValueError("Teacher and student need matching Conv2d or Linear consumers")
    moments, sampled = ReconstructionMoments(), {}

    def observe_teacher(
        module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """Save sampled preactivation targets before downstream in-place operations."""
        if sampled:
            raise ValueError("Repeated consumer calls are outside this workflow's scope")
        locations = output.shape[-2] * output.shape[-1] if type(module) is nn.Conv2d else 1
        count = (
            output.shape[0] * locations
            if type(module) is nn.Conv2d
            else output.numel() // output.shape[-1]
        )
        if count == 0:
            raise ValueError("Calibration outputs must contain observations")
        ids = (
            torch.arange(min(max_rows, count), device=output.device) * count // min(max_rows, count)
        )
        if type(module) is nn.Conv2d:
            targets = output[
                ids // locations, :, (ids % locations) // output.shape[-1], ids % output.shape[-1]
            ]
        else:
            targets = output.reshape(-1, output.shape[-1])[ids]
        sampled["ids"] = ids
        sampled["targets"] = (
            targets.detach() - consumer.bias.detach()
            if consumer.bias is not None
            else targets.detach()
        )

    def observe_student(
        module: nn.Module, args: tuple[torch.Tensor, ...], kwargs: dict[str, Any]
    ) -> None:
        """Merge the matched current design rows into this consumer's moments."""
        if "targets" not in sampled:
            raise ValueError("Expected one teacher and one student consumer call per batch")
        inputs = args[0] if args else kwargs["input"]
        moments.update(feature_rows(module, inputs, sampled["ids"]), sampled.pop("targets"))

    modes = [(module, module.training) for root in (model, teacher) for module in root.modules()]
    handles = []
    total = len(loader) if isinstance(loader, Sized) else None
    if max_batches and total is not None:
        total = min(total, max_batches)
    try:
        model.eval()
        teacher.eval()
        handles.append(original.register_forward_hook(observe_teacher))
        handles.append(consumer.register_forward_pre_hook(observe_student, with_kwargs=True))
        with (
            torch.no_grad(),
            tqdm(
                islice(loader, max_batches) if max_batches else loader,
                total=total,
                desc=f"OSSCAR calibration {consumer_path} ({device})",
                unit="batch",
            ) as progress,
        ):
            for images, _labels in progress:
                sampled.clear()
                images = images.to(device, non_blocking=True)
                teacher(images)
                model(images)
                if "targets" in sampled or "ids" not in sampled:
                    raise ValueError("Calibration did not execute both declared consumers")
        if not moments.count:
            raise ValueError("No reconstruction observations collected")
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return moments


def _coordinates(groups: tuple[int, ...], size: int, device: torch.device) -> torch.Tensor:
    """Expand explicit channel membership to consecutive kernel/feature rows."""
    return (
        torch.tensor(groups, dtype=torch.long, device=device)[:, None] * size
        + torch.arange(size, device=device)
    ).flatten()


def solve_support(
    gram: torch.Tensor,
    cross: torch.Tensor,
    retained: tuple[int, ...],
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a fixed support and return its inverse for subsequent Schur updates."""
    rows = _coordinates(retained, group_size, gram.device)
    if not retained:
        return cross.new_empty((0, cross.shape[1])), gram.new_empty((0, 0))
    matrix = gram[rows[:, None], rows]
    try:
        factor = torch.linalg.cholesky(matrix)
        coefficients = torch.cholesky_solve(cross[rows], factor)
        inverse = torch.cholesky_inverse(factor)
    except torch.linalg.LinAlgError as error:
        raise ValueError(
            "Reconstruction system must be positive definite; increase damping"
        ) from error
    if not torch.isfinite(coefficients).all() or not torch.isfinite(inverse).all():
        raise ValueError("Nonfinite reconstruction solve")
    return coefficients, inverse


def deletion_costs(
    coefficients: torch.Tensor, inverse: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Compute exact single-group loss increases at the current optimum."""
    n = len(coefficients) // group_size
    blocks = inverse.reshape(n, group_size, n, group_size).diagonal(dim1=0, dim2=2).movedim(-1, 0)
    weights = coefficients.reshape(n, group_size, -1)
    return (weights * torch.linalg.solve(blocks, weights)).sum((1, 2)) / 2


def remove_groups(
    retained: tuple[int, ...],
    coefficients: torch.Tensor,
    inverse: torch.Tensor,
    positions: tuple[int, ...],
    group_size: int,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
    """Delete current group positions and update coefficients/inverse jointly."""
    keep = tuple(i for i in range(len(retained)) if i not in positions)
    r = _coordinates(keep, group_size, inverse.device)
    q = _coordinates(positions, group_size, inverse.device)
    block = inverse[q[:, None], q]
    coupling = inverse[r[:, None], q]
    weights = coefficients[r] - coupling @ torch.linalg.solve(block, coefficients[q])
    updated = inverse[r[:, None], r] - coupling @ torch.linalg.solve(block, inverse[q[:, None], r])
    return tuple(retained[i] for i in keep), weights, (updated + updated.T) / 2


def restoration_gains(
    gram: torch.Tensor,
    cross: torch.Tensor,
    retained: tuple[int, ...],
    coefficients: torch.Tensor,
    inverse: torch.Tensor,
    group_size: int,
) -> tuple[tuple[int, ...], torch.Tensor]:
    """Compute the objective decrease from restoring each absent group separately."""
    absent = tuple(i for i in range(len(gram) // group_size) if i not in retained)
    q = _coordinates(absent, group_size, gram.device).reshape(-1, group_size)
    r = _coordinates(retained, group_size, gram.device)
    coupling = gram[q.flatten()[:, None], r]
    blocks = coupling.reshape(len(absent), group_size, len(r))
    schur = gram[q[:, :, None], q[:, None, :]] - (coupling @ inverse).reshape_as(
        blocks
    ) @ blocks.transpose(1, 2)
    residual = (cross[q.flatten()] - coupling @ coefficients).reshape(
        len(absent), group_size, cross.shape[1]
    )
    gains = (residual * torch.linalg.solve(schur, residual)).sum((1, 2)) / 2
    return absent, gains


@dataclass(frozen=True)
class ReconstructionSolution:
    """Compact coefficients and the strictly improving fixed-cardinality history."""

    retained: tuple[int, ...]
    coefficients: torch.Tensor
    objectives: tuple[float, ...]
    swap_attempts: int


@torch.no_grad()
def solve_osscar(
    gram: torch.Tensor,
    cross: torch.Tensor,
    num_groups: int,
    keep: int,
    *,
    prune_batch: int = 8,
    swap_steps: int = 5,
) -> ReconstructionSolution:
    """Run grouped least-squares deletion followed by bounded remove/restore search.

    Delete groups with the smallest exact single-group increase, applying each
    batch jointly. Then remove one inexpensive retained group and restore the
    best absent group, including the just-removed group. Accept only an actual
    improvement after a fresh solve. This is not exhaustive swap search.
    """
    if (
        type(num_groups) is not int
        or num_groups <= 0
        or type(keep) is not int
        or not 0 < keep <= num_groups
        or type(prune_batch) is not int
        or prune_batch <= 0
        or type(swap_steps) is not int
        or swap_steps < 0
    ):
        raise ValueError("Invalid group cardinality or search limits")
    if (
        gram.ndim != 2
        or len(gram) == 0
        or gram.shape[0] != gram.shape[1]
        or len(gram) % num_groups
        or cross.ndim != 2
        or len(cross) != len(gram)
        or not cross.shape[1]
        or not gram.is_floating_point()
        or not cross.is_floating_point()
        or not torch.isfinite(gram).all()
        or not torch.isfinite(cross).all()
        or not torch.allclose(gram, gram.T, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("Expected finite symmetric Gram and matching cross-product matrices")
    h, g = gram.double(), cross.double()
    size = len(h) // num_groups
    retained = tuple(range(num_groups))
    coefficients, inverse = solve_support(h, g, retained, size)
    while len(retained) > keep:
        costs = deletion_costs(coefficients, inverse, size)
        if not torch.isfinite(costs).all():
            raise ValueError("Nonfinite deletion costs")
        positions = tuple(
            torch.argsort(costs, stable=True)[: min(prune_batch, len(retained) - keep)].tolist()
        )
        retained, coefficients, inverse = remove_groups(
            retained, coefficients, inverse, positions, size
        )
    # Re-factor after downdates before judging swap improvements or exporting weights.
    coefficients, inverse = solve_support(h, g, retained, size)

    def objective(groups: tuple[int, ...], weights: torch.Tensor) -> float:
        value = float(-(weights * g[_coordinates(groups, size, g.device)]).sum() / 2)
        if not math.isfinite(value):
            raise ValueError("Nonfinite reconstruction objective")
        return value

    history, attempts = [objective(retained, coefficients)], 0
    if keep < num_groups:
        for _ in range(swap_steps):
            attempts += 1
            position = int(torch.argmin(deletion_costs(coefficients, inverse, size)))
            reduced, weights, reduced_inverse = remove_groups(
                retained, coefficients, inverse, (position,), size
            )
            absent, gains = restoration_gains(h, g, reduced, weights, reduced_inverse, size)
            if not torch.isfinite(gains).all():
                raise ValueError("Nonfinite restoration gains")
            restored = absent[int(torch.argmax(gains))]
            proposal = tuple(sorted((*reduced, restored)))
            if proposal == retained:
                break
            weights, updated_inverse = solve_support(h, g, proposal, size)
            value = objective(proposal, weights)
            tolerance = 1e-10 * max(1.0, abs(history[-1]))
            if value >= history[-1] - tolerance:
                break
            retained, coefficients, inverse = proposal, weights, updated_inverse
            history.append(value)
    return ReconstructionSolution(retained, coefficients, tuple(history), attempts)


def allocate_widths(
    pruner: Pruner,
    pairs: dict[str, str],
    budget: ParameterBudget,
    granularity: int,
) -> dict[str, int]:
    """Find a common channel-removal fraction reaching the actual parameter cap.

    This is an explicit allocation policy, separate from OSSCAR's within-layer
    optimizer. Test aligned width breakpoints using real joint structural plans;
    no affine parameter-cost approximation or model mutation is involved.
    """
    if type(granularity) is not int or granularity <= 0 or not pairs:
        raise ValueError("Allocation requires pairs and positive granularity")
    graph = pruner.graph
    widths = {path: graph.parameter(f"{path}.weight").shape[0] for path in pairs}
    choices = {
        path: sorted({width - kept for kept in range(granularity, width + 1, granularity)})
        for path, width in widths.items()
    }
    if any(not values for values in choices.values()):
        raise PlanningError("Granularity leaves no positive retained width")
    fractions = sorted(
        {Fraction(n, widths[path]) for path, values in choices.items() for n in values}
    )
    initial = count_parameters(pruner.model)

    def evaluate_fraction(fraction: Fraction) -> tuple[dict[str, int], int]:
        removed = {
            path: max(
                (n for n in choices[path] if Fraction(n, width) <= fraction),
                default=choices[path][0],
            )
            for path, width in widths.items()
        }
        plan = pruner.plan_remove(
            [
                graph.parameter(f"{path}.weight").axis(0).select(range(n))
                for path, n in removed.items()
                if n
            ]
        )
        remaining = initial - sum(
            math.prod(recipe.tensor.shape) - math.prod(recipe.shape)
            for recipe in plan.recipes
            if recipe.tensor.kind == "parameter"
        )
        return {path: widths[path] - n for path, n in removed.items()}, remaining

    minimal, remaining = evaluate_fraction(fractions[0])
    if remaining <= budget.max_params:
        return minimal
    _, remaining = evaluate_fraction(fractions[-1])
    if remaining > budget.max_params:
        raise PlanningError("Parameter target is unreachable within the declared OSSCAR widths")
    low, high = 0, len(fractions) - 1
    while low < high:
        middle = (low + high) // 2
        _, remaining = evaluate_fraction(fractions[middle])
        if remaining <= budget.max_params:
            high = middle
        else:
            low = middle + 1
    return evaluate_fraction(fractions[low])[0]


def apply_reconstruction(
    pruner: Pruner,
    producer_path: str,
    consumer_path: str,
    solution: ReconstructionSolution,
) -> None:
    """Validate matched channel removal, apply the plan and copy reconstructed weights."""
    graph = pruner.graph
    producer = graph.parameter(f"{producer_path}.weight")
    consumer = graph.parameter(f"{consumer_path}.weight")
    if (
        not solution.retained
        or tuple(sorted(set(solution.retained))) != solution.retained
        or any(
            type(index) is not int or not 0 <= index < producer.shape[0]
            for index in solution.retained
        )
    ):
        raise ValueError("Retained channels must be unique original coordinates in sorted order")
    removed = sorted(set(range(producer.shape[0])) - set(solution.retained))
    plan = pruner.plan_remove([producer.axis(0).select(removed)])
    consumer_selection = plan.analysis.selection(consumer)
    if consumer_selection != consumer_selection.tensor.axis(1).select(removed):
        raise PlanningError("Reconstruction requires only matching consumer input-column removal")
    layer = pruner.model.get_submodule(consumer_path)
    shape = (layer.weight.shape[0], len(solution.retained), *layer.weight.shape[2:])
    weight = solution.coefficients.T.reshape(shape).to(layer.weight)
    if not torch.isfinite(weight).all():
        raise ValueError("Reconstructed weights overflow the model dtype")
    pruner.apply(plan)
    with torch.no_grad():
        pruner.model.get_submodule(consumer_path).weight.copy_(weight)


def parse_args() -> argparse.Namespace:
    """Configure sequential reconstruction, calibration and optional fine-tuning."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=tuple(MODELS), default="resnet18")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--train_samples", type=int, default=0)
    parser.add_argument("--val_samples", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=256)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument("--val_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pruning_ratio", type=float, default=0.05, help="Whole-model parameter reduction"
    )
    parser.add_argument("--granularity", type=int, default=8)
    parser.add_argument(
        "--calibration_batches", type=int, default=2, help="Batches per consumer; 0 uses all"
    )
    parser.add_argument(
        "--calibration_rows",
        type=int,
        default=4096,
        help="Maximum sampled positions per batch per consumer",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.01,
        help="Original-weight ridge relative to mean Gram diagonal",
    )
    parser.add_argument(
        "--prune_batch", type=int, default=8, help="Channels deleted per quadratic search update"
    )
    parser.add_argument(
        "--swap_steps",
        type=int,
        default=5,
        help="Maximum fixed-cardinality remove/restore attempts per consumer",
    )
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=Path("runs/osscar_pruning"))
    parser.add_argument(
        "--compile_latency", action="store_true", help="Compile latency measurement only"
    )
    parser.add_argument("--latency_warmup", type=int, default=5)
    parser.add_argument("--latency_repetitions", type=int, default=20)
    options = parser.parse_args()
    for name in (
        "train_batch_size",
        "val_batch_size",
        "granularity",
        "calibration_rows",
        "prune_batch",
        "latency_repetitions",
    ):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "train_samples",
        "val_samples",
        "train_workers",
        "val_workers",
        "calibration_batches",
        "swap_steps",
        "finetune_epochs",
        "latency_warmup",
    ):
        if getattr(options, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= options.pruning_ratio < 1 or not math.isfinite(options.lr) or options.lr <= 0:
        parser.error("Require 0 <= pruning_ratio < 1 and positive finite lr")
    if not math.isfinite(options.damping) or options.damping < 0:
        parser.error("--damping must be finite and nonnegative")
    if options.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu"
        )
    return options


def main() -> None:
    """Allocate widths, reconstruct sequentially, then verify a compact checkpoint."""
    options = parse_args()
    torch.manual_seed(options.seed)
    model = make_model(options.model).to(options.device).eval()
    teacher = copy.deepcopy(model).requires_grad_(False).eval()
    budget = ParameterBudget.from_ratio(model, options.pruning_ratio)
    before_params = count_parameters(model)
    example = torch.zeros(1, 3, 224, 224, device=options.device)
    initial_graph = DependencyGraph.build(model, args=(example,))
    pairs, exclusions = discover_reconstruction_pairs(initial_graph)
    if not pairs:
        raise PlanningError(f"No eligible reconstruction chains: {exclusions}")
    print(
        f"OSSCAR discovered {len(pairs)} reconstruction pairs; exclusions: {exclusions}", flush=True
    )
    initial_pruner = Pruner(
        model,
        graph=initial_graph,
        granularity=Granularity(by_path=dict.fromkeys(pairs, options.granularity)),
    )
    widths = allocate_widths(initial_pruner, pairs, budget, options.granularity)
    del initial_pruner, initial_graph
    weights = MODELS[options.model][1]
    train, validation, dataset_info = load_images(
        weights,
        options.data_dir,
        need_train=True,
        train_samples=options.train_samples,
        val_samples=options.val_samples,
        seed=options.seed,
    )
    if train is None:
        raise ValueError("OSSCAR requires a training split for calibration")
    generator = torch.Generator().manual_seed(options.seed)
    train_loader = DataLoader(
        train,
        batch_size=options.train_batch_size,
        # Keep sample order independent of worker-startup RNG consumption, so
        # reseeding below really replays the same images with persistent workers.
        sampler=RandomSampler(train, generator=generator),
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
        weights=str(weights),
        dataset=dataset_info,
        retained_widths=widths,
        reconstruction_pairs=pairs,
        excluded_producers=exclusions,
        max_params=budget.max_params,
    )
    options.output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    reconstruction: list[dict[str, Any]] = []

    def record(stage: str, **extra: object) -> None:
        """Evaluate and persist measured stages with the algorithm's actual settings."""
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
            json.dumps(
                {"config": config, "stages": records, "reconstruction": reconstruction}, indent=2
            )
            + "\n"
        )

    record("pretrained")
    for first, second in pairs.items():
        width = model.get_submodule(first).weight.shape[0]
        if widths[first] == width:
            continue  # A zero deletion request must not refit or alter weights.
        # Replay the same seeded image order, while current-model activations
        # include all earlier physical deletions and reconstructed weights.
        generator.manual_seed(options.seed)
        moments = collect_reconstruction(
            model,
            teacher,
            second,
            train_loader,
            options.device,
            options.calibration_batches,
            options.calibration_rows,
        )
        original = teacher.get_submodule(second).weight.detach().flatten(1).T
        h, g = moments.system(original, options.damping)
        print(
            f"OSSCAR solve {second}: {width} -> {widths[first]} channels; {moments.count} observations",
            flush=True,
        )
        solution = solve_osscar(
            h,
            g,
            width,
            widths[first],
            prune_batch=options.prune_batch,
            swap_steps=options.swap_steps,
        )
        graph = DependencyGraph.build(model, args=(example,))
        pruner = Pruner(
            model, graph=graph, granularity=Granularity(by_path={first: options.granularity})
        )
        apply_reconstruction(pruner, first, second, solution)
        reconstruction.append(
            {
                "consumer": second,
                "observations": moments.count,
                "retained": list(solution.retained),
                "objectives": solution.objectives,
                "swap_attempts": solution.swap_attempts,
            }
        )
        del graph, pruner, moments, h, g, solution
    del teacher
    after_params = count_parameters(model)
    if after_params > budget.max_params:
        raise PlanningError("Final reconstruction did not reach the prevalidated parameter target")
    record(
        "pruned",
        before_params=before_params,
        after_params=after_params,
        max_params=budget.max_params,
        target_met=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=options.lr, momentum=0.9)
    for epoch in range(options.finetune_epochs):
        model.train()
        total, count = 0.0, 0
        with tqdm(
            train_loader, desc=f"OSSCAR fine-tuning {epoch + 1} ({options.device})", unit="batch"
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
        make_model(options.model, pretrained=False),
        options.output / "model.pt",
        map_location=options.device,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(example), model(example))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "algorithm": {"method": "osscar", "reconstruction": reconstruction},
            "config": config,
            "rng": torch.get_rng_state(),
            "data_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        options.output / "training.pt",
    )
    print(f"Reconstructed compact checkpoint verified; results: {options.output}", flush=True)


if __name__ == "__main__":
    main()
