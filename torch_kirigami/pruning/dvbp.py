"""DVBP statistics, spectral denoising and OB²C value reconstruction.

These functions own algorithm statistics and parameter values, not dependency
analysis. Structural deletion still goes through the public Pruner plan/apply
API. The default reconstruction covariance follows the authors' implementation
(denoised covariance); raw covariance is an explicit paper-equation ablation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import islice

import torch
from torch import nn

from ..dep_graph.contracts import Impact
from ..dep_graph.dependency_graph import DependencyGraph
from ..dep_graph.selection import AxisRef
from .plan import PruningPlan
from .types import Candidate, MetricContext


@dataclass
class ActivationCovariance:
    """Merge population covariance over channels-last activations in bounded chunks.

    Args:
        dtype: Accumulation precision; float32 saves memory, float64 is more accurate.
        device: Statistics device; None follows the observed activation device.
        chunk_size: Maximum flattened activation rows in one temporary matrix.

    ``count`` counts token/spatial observations; ``images`` counts input images.
    MP noise estimation uses images, never the correlated token count.
    """

    dtype: torch.dtype = torch.float32
    device: torch.device | str | None = None
    chunk_size: int = 2048
    count: int = 0
    images: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Covariance dtype must be float32 or float64")
        if type(self.chunk_size) is not int or self.chunk_size <= 0:
            raise ValueError("Covariance chunk_size must be positive")

    @torch.no_grad()
    def update(self, activation: torch.Tensor) -> None:
        """Merge a full image batch, counting images separately from spatial rows."""
        if activation.ndim < 2 or not activation.is_floating_point():
            raise ValueError("Expected real channels-last activations with at least two dimensions")
        width = activation.shape[-1]
        if not activation.numel() or not width:
            raise ValueError("Calibration activations must be nonempty")
        target = activation.device if self.device is None else torch.device(self.device)
        if self.mean is not None and (self.mean.numel() != width or self.mean.device != target):
            raise ValueError("Calibration channel width and device must stay fixed")
        rows = activation.detach().reshape(-1, width)
        for start in range(0, len(rows), self.chunk_size):
            values = rows[start : start + self.chunk_size].to(device=target, dtype=self.dtype)
            if not torch.isfinite(values).all():
                raise ValueError("Calibration activations must be finite")
            batch_mean = values.mean(0)
            centered = values - batch_mean
            batch_count = len(values)
            if self.mean is None:
                self.mean = batch_mean
                self.m2 = centered.T @ centered
                self.count = batch_count
                continue
            assert self.m2 is not None
            count = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2.addmm_(centered.T, centered)
            self.m2.addr_(delta, delta, alpha=self.count * batch_count / count)
            self.mean.add_(delta, alpha=batch_count / count)
            self.count = count
        self.images += activation.shape[0]

    def covariance(self) -> torch.Tensor:
        """Return population covariance, matching the reference OnlineCov algorithm."""
        if self.count < 2 or self.m2 is None:
            raise ValueError("DVBP needs at least two activation observations")
        result = self.m2 / self.count
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite calibration covariance")
        return result


def collect_covariances(
    model: nn.Module,
    consumers: Sequence[str],
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | str,
    max_batches: int = 0,
    *,
    dtype: torch.dtype = torch.float32,
    statistics_device: torch.device | str | None = None,
    chunk_size: int = 2048,
) -> dict[str, ActivationCovariance]:
    """Observe consumer inputs without updates, restoring hooks and mixed modes.

    Labels are ignored. Zero max_batches traverses the full loader; a positive
    limit must be available in full. Existing gradients remain untouched.
    Each consumer must execute exactly once per input batch.
    """
    paths = tuple(consumers)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Calibration requires distinct nonempty consumers")
    if type(max_batches) is not int or max_batches < 0:
        raise ValueError("max_batches must be a nonnegative integer")
    stats = {path: ActivationCovariance(dtype, statistics_device, chunk_size) for path in paths}
    modes = [(module, module.training) for module in model.modules()]
    handles = []
    seen: dict[str, int] = {}

    def observe(path, module, args, kwargs):
        """Accumulate a consumer's actual input and invocation count."""
        seen[path] = seen.get(path, 0) + 1
        stats[path].update(args[0] if args else kwargs["input"])

    completed = 0
    try:
        model.eval()
        for path in paths:
            # Bind path at registration, so every hook observes its own consumer.
            def hook(module, args, kwargs, path=path):
                """Bind the consumer path for this observation hook."""
                observe(path, module, args, kwargs)

            handles.append(
                model.get_submodule(path).register_forward_pre_hook(hook, with_kwargs=True)
            )
        batches = islice(loader, max_batches) if max_batches else iter(loader)
        with torch.no_grad():
            for images, _labels in batches:
                seen.clear()
                model(images.to(device, non_blocking=True))
                if seen != dict.fromkeys(paths, 1):
                    raise ValueError("Each DVBP consumer must execute once per calibration batch")
                completed += 1
        if not completed or (max_batches and completed != max_batches):
            raise ValueError("Not enough calibration batches for the requested limit")
        for value in stats.values():
            value.covariance()
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return stats


@lru_cache(maxsize=128)
def mp_median(aspect: float) -> float:
    """Compute the normalized continuous MP median with deterministic quadrature.

    The sine-angle change of variables removes endpoint square-root singularities.
    For aspect > 1 this is the continuous-bulk median, excluding the zero atom,
    matching the reference's normalization. No SciPy dependency is required.
    """
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("MP aspect must be positive and finite")
    if aspect > 1:
        return aspect * mp_median(1 / aspect)
    points = 16384
    theta = (torch.arange(points, dtype=torch.float64, device="cpu") + 0.5) * (math.pi / points)
    root = math.sqrt(aspect)
    x = (1 - root) ** 2 + 4 * root * torch.sin(theta / 2).square()
    mass = 2 * torch.sin(theta).square() / (math.pi * x)
    cdf = mass.cumsum(0)
    target = cdf[-1] / 2
    index = int(torch.searchsorted(cdf, target).item())
    previous = cdf[index - 1] if index else cdf.new_zeros(())
    fraction = float((target - previous) / mass[index])
    angle = (index + fraction) * math.pi / points
    return (1 - root) ** 2 + 4 * root * math.sin(angle / 2) ** 2


@dataclass(frozen=True)
class DenoisedStatistics:
    """CPU snapshots of scores, mean and the damped reconstruction system."""

    mean: torch.Tensor
    raw_scores: torch.Tensor
    denoised_scores: torch.Tensor
    scores: torch.Tensor
    system: torch.Tensor
    observations: int
    images: int
    aspect: float
    noise_variance: float
    threshold: float
    signal_components: int
    ridge: float
    repair_covariance: str

    def report(self) -> dict[str, int | float | str]:
        """Return small JSON-compatible diagnostics without serializing matrices."""
        return {
            "observations": self.observations,
            "images": self.images,
            "hidden_width": self.mean.numel(),
            "mp_aspect": self.aspect,
            "noise_variance": self.noise_variance,
            "mp_threshold": self.threshold,
            "signal_components": self.signal_components,
            "ridge": self.ridge,
            "repair_covariance": self.repair_covariance,
        }


@torch.no_grad()
def denoise_covariance(
    stats: ActivationCovariance,
    pruning_ratio: float,
    *,
    damping: float = 0.1,
    ridge_floor: float = 1e-8,
    repair_covariance: str = "denoised",
    compute_device: torch.device | str = "cpu",
) -> DenoisedStatistics:
    """Build hybrid variance scores and an explicitly regularized repair system.

    Default reconstruction uses the denoised covariance as in the official code.
    ``raw`` instead uses the original centered covariance from the paper equations.
    Unlike the reference, nonfinite values fail rather than being silently replaced.
    An absolute ridge floor handles all-constant activations and is reported.
    """
    if not math.isfinite(pruning_ratio) or not 0 <= pruning_ratio < 1:
        raise ValueError("pruning_ratio must be finite and in [0, 1)")
    if any(not math.isfinite(value) or value < 0 for value in (damping, ridge_floor)):
        raise ValueError("Damping and ridge_floor must be finite and nonnegative")
    if repair_covariance not in ("raw", "denoised") or stats.images <= 0 or stats.mean is None:
        raise ValueError("Require raw/denoised covariance and calibrated image counts")
    covariance = stats.covariance().to(device=compute_device, dtype=torch.float64)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    tolerance = (
        torch.finfo(stats.dtype).eps
        * covariance.shape[0]
        * max(1.0, float(eigenvalues.abs().max()))
        * 32
    )
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("Covariance is not positive semidefinite; use higher statistics precision")
    eigenvalues = eigenvalues.clamp_min(0)
    aspect = covariance.shape[0] / stats.images
    noise = float(eigenvalues.median()) / mp_median(aspect)
    threshold = noise * (1 + math.sqrt(aspect)) ** 2
    selected = eigenvalues > threshold
    if not selected.any():
        selected[-1] = True  # Match the reference's keep-largest-component safeguard.
    denoised = (vectors[:, selected] * eigenvalues[selected]) @ vectors[:, selected].T
    raw_scores = covariance.diagonal().clamp_min(0)
    denoised_scores = denoised.diagonal().clamp_min(0)
    scores = (1 - pruning_ratio) * raw_scores + pruning_ratio * denoised_scores
    system = denoised if repair_covariance == "denoised" else covariance
    ridge = max(ridge_floor, damping * float(system.diagonal().mean()))
    system = system.clone()
    system.diagonal().add_(ridge)
    if not torch.isfinite(system).all() or not torch.isfinite(scores).all():
        raise ValueError("Nonfinite denoising result")
    return DenoisedStatistics(
        stats.mean.detach().double().cpu().clone(),
        raw_scores.cpu().clone(),
        denoised_scores.cpu().clone(),
        scores.cpu().clone(),
        system.cpu(),
        stats.count,
        stats.images,
        aspect,
        noise,
        threshold,
        int(selected.sum()),
        ridge,
        repair_covariance,
    )


@torch.no_grad()
def solve_ob2c(
    weight: torch.Tensor,
    bias: torch.Tensor,
    system: torch.Tensor,
    mean: torch.Tensor,
    removed: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return compact reconstructed weight and bias without mutating inputs.

    Solve the retained covariance block directly, avoiding explicit inverses.
    This is algebraically equivalent to the block inverse update under an SPD
    system: W_new[K] = W[K] + W[P] H[P,K] inv(H[K,K]). The bias preserves
    the original calibration mean. Singular systems fail explicitly.
    """
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("OB2C requires a real matrix weight")
    width = weight.shape[1]
    indices = tuple(removed)
    if len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < width for i in indices
    ):
        raise ValueError("Removed indices must be distinct valid integers")
    if bias.shape != (weight.shape[0],) or mean.shape != (width,) or system.shape != (width, width):
        raise ValueError("OB2C dimensions do not match")
    if any(
        not value.is_floating_point() or not torch.isfinite(value).all()
        for value in (weight, bias, system, mean)
    ):
        raise ValueError("OB2C inputs must be real and finite")
    if len(indices) == width:
        raise ValueError("OB2C cannot remove an entire hidden layer")
    if not indices:
        return weight.detach().clone(), bias.detach().clone()
    device = system.device
    w, b, mu = (
        value.detach().to(device=device, dtype=torch.float64) for value in (weight, bias, mean)
    )
    h = system.double()
    if not torch.allclose(h, h.T, rtol=1e-8, atol=1e-10):
        raise ValueError("OB2C system must be symmetric")
    removed_set = set(indices)
    retained = [i for i in range(width) if i not in removed_set]
    k = torch.tensor(retained, device=device, dtype=torch.long)
    p = torch.tensor(indices, device=device, dtype=torch.long)
    hkk = h[k][:, k]
    factor, info = torch.linalg.cholesky_ex(hkk)
    if int(info) != 0:
        raise ValueError("OB2C retained system is not positive definite; increase damping")
    transfer = torch.cholesky_solve(h[k][:, p], factor)
    delta = w[:, p] @ transfer.T
    updated = w[:, k] + delta
    updated_bias = b + w[:, p] @ mu[p] - delta @ mu[k]
    updated = updated.to(device=weight.device, dtype=weight.dtype)
    updated_bias = updated_bias.to(device=bias.device, dtype=bias.dtype)
    if not torch.isfinite(updated).all() or not torch.isfinite(updated_bias).all():
        raise ValueError("OB2C result overflows the model parameter dtype")
    return updated, updated_bias


class DenoisedVariance:
    """Static DVBP metric for declared independent dense hidden axes."""

    def __init__(
        self,
        graph: DependencyGraph,
        pairs: dict[str, str],
        statistics: dict[str, DenoisedStatistics],
    ) -> None:
        self.graph = graph
        self.scores: dict[AxisRef, tuple[float, ...]] = {}
        for first, second in pairs.items():
            axis = graph.parameter(f"{first}.weight").axis(0)
            values = statistics[second].scores
            if values.shape != (axis.tensor.shape[0],) or not torch.isfinite(values).all():
                raise ValueError("DVBP statistics do not match the hidden axis")
            self.scores[axis] = tuple(values.cpu().tolist())

    def score(
        self, context: MetricContext, candidates: tuple[Candidate, ...], *, accepted_impact: Impact
    ) -> list[float]:
        """Sum fixed scores over newly removed complete hidden-axis slices."""
        if context.graph is not self.graph:
            raise ValueError("DVBP statistics and candidates belong to different graphs")
        context.require_complete(accepted_impact)
        result = []
        for candidate in candidates:
            axis = candidate.axis
            if axis not in self.scores or len(candidate.remove) != 1:
                raise ValueError("DVBP candidates must select one declared hidden axis")
            selection = candidate.remove[0]
            indices = selection.fully_selected_indices(axis.dim)
            if selection != axis.select(indices):
                raise ValueError("DVBP candidates must select complete hidden-axis slices")
            indices = indices.subtract(
                accepted_impact.selection(axis.tensor).fully_selected_indices(axis.dim)
            )
            result.append(math.fsum(self.scores[axis][i] for i in indices))
        return result


@torch.no_grad()
def prepare_ob2c_updates(
    graph: DependencyGraph,
    plan: PruningPlan,
    pairs: dict[str, str],
    statistics: dict[str, DenoisedStatistics],
    *,
    solve_device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Prepare compact consumer values in original coordinates before applying a plan.

    Reject any deletion outside matched hidden widths, including classifier,
    normalization and external interfaces. Adjacent pairs use one fixed calibration
    snapshot; no end-to-end equivalence after earlier corrections is claimed.
    Returned CPU tensors must be copied after structural apply and then saved.
    """
    graph.validate()
    if not pairs or len(set(pairs.values())) != len(pairs):
        raise ValueError("OB2C requires distinct nonempty MLP consumers")
    if set(statistics) != set(pairs.values()):
        raise ValueError("OB2C statistics must cover exactly the declared consumers")
    removed = {
        path: plan.analysis.selection(graph.parameter(f"{path}.weight")).fully_selected_indices(0)
        for path in pairs
    }
    input_producers = {second: first for first, second in pairs.items()}
    allowed = set()
    for path in dict.fromkeys((*pairs, *pairs.values())):
        layer = graph.model.get_submodule(path)
        if type(layer) is not nn.Linear:
            raise ValueError("OB2C supports only dense Linear MLP endpoints")
        actual = plan.analysis.selection(graph.parameter(f"{path}.weight"))
        expected = actual.tensor.axis(0).select(removed.get(path, ()))
        if path in input_producers:
            expected = expected.union(actual.tensor.axis(1).select(removed[input_producers[path]]))
        if actual != expected:
            raise ValueError("OB2C requires only matched hidden-width removals")
        allowed.add(actual.tensor)
        if layer.bias is not None:
            bias = plan.analysis.selection(graph.parameter(f"{path}.bias"))
            if bias != bias.tensor.axis(0).select(removed.get(path, ())):
                raise ValueError("OB2C requires only matched hidden-width removals")
            allowed.add(bias.tensor)
    if any(
        selection.regions and selection.tensor not in allowed
        for selection in plan.analysis.selections
        if selection.tensor.kind in ("parameter", "buffer")
    ):
        raise ValueError("OB2C plan changes parameters outside declared MLPs")
    updates = {}
    for first, second in pairs.items():
        if not removed[first]:
            continue
        layer = graph.model.get_submodule(second)
        if layer.bias is None:
            raise ValueError("OB2C consumer requires a bias")
        stats = statistics[second]
        weight, bias = solve_ob2c(
            layer.weight,
            layer.bias,
            stats.system.to(solve_device),
            stats.mean,
            tuple(removed[first]),
        )
        output_removals = removed.get(second, ())
        if output_removals:
            retained = [i for i in range(layer.out_features) if i not in output_removals]
            weight, bias = weight[retained], bias[retained]
        updates[f"{second}.weight"] = weight.cpu()
        updates[f"{second}.bias"] = bias.cpu()
    return updates


@torch.no_grad()
def apply_ob2c_updates(model: nn.Module, updates: dict[str, torch.Tensor]) -> None:
    """Validate every compact value before copying prepared updates after apply."""
    prepared = []
    for path, value in updates.items():
        parameter = model.get_parameter(path)
        converted = value.to(parameter)
        if converted.shape != parameter.shape or not torch.isfinite(converted).all():
            raise ValueError(f"Invalid compact OB2C update: {path}")
        prepared.append((parameter, converted))
    for parameter, value in prepared:
        parameter.copy_(value)
