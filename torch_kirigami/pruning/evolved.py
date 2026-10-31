"""Fixed Pruner-Zero importance formula adapted to structural parameter regions."""

import math
import weakref
from collections import OrderedDict

import torch

from ..dep_graph.contracts import Impact
from ..dep_graph.selection import gather_region
from .metrics import parameter_regions
from .types import Candidate, MetricContext, PlanningError


class EvolvedMetric:
    """Sum `W² * minmax(abs(G))` over newly removed matrix/kernel weights.

    Reuse the published Pruner-Zero best expression, without evolution/search.
    The caller supplies dense, unscaled task gradients on `parameter.grad`.
    Normalize over each complete original weight tensor, never a candidate
    slice or a batch of candidates. Sum dependency-coupled weight regions once;
    biases and one-dimensional normalization scales do not contribute.
    Constant gradient magnitudes produce zero importance instead of division by
    zero. This explicit numerical convention differs from the unguarded original.

    Structural group summation and caller-provided vision task gradients adapt
    the paper's per-weight LLM pruning. No paper benchmark equivalence is claimed.
    Caches contain at most 256 axis vectors and 262144 scalar positions. Values
    are refreshed when a weight/gradient version or identity changes; unsupported
    inference-tensor versions bypass caching. Calibration must precede planning.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._positions = 0

    @staticmethod
    def _terms(weight: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        """Evaluate on a complete tensor before any axis reduction."""
        w, g = weight.detach().double(), gradient.detach().double().abs()
        if not torch.isfinite(w).all() or not torch.isfinite(g).all():
            raise PlanningError("EvolvedMetric requires finite weights and gradients")
        low, high = g.amin(), g.amax()
        scaled = (g - low) / (high - low) if high > low else torch.zeros_like(g)
        terms = w.square() * scaled
        if not torch.isfinite(terms).all():
            raise PlanningError("EvolvedMetric importance overflow")
        return terms

    def _axis_sums(self, weight: torch.Tensor, dim: int) -> tuple[float, ...]:
        """Cache only reduced scalars, with weak identity and version checks."""
        gradient = weight.grad
        try:
            key = (id(weight), weight._version, id(gradient), gradient._version, dim)
        except RuntimeError:
            key = None
        if key is not None and key in self._cache:
            wref, gref, values = self._cache[key]
            if wref() is weight and gref() is gradient:
                self._cache.move_to_end(key)
                return values
            self._positions -= len(values)
            del self._cache[key]
        terms = self._terms(weight, gradient).movedim(dim, 0)
        values = tuple(terms.reshape(weight.shape[dim], -1).sum(1).cpu().tolist())
        if key is not None:
            self._cache[key] = (weakref.ref(weight), weakref.ref(gradient), values)
            self._positions += len(values)
            while len(self._cache) > 256 or self._positions > 262144:
                _, (_, _, removed) = self._cache.popitem(last=False)
                self._positions -= len(removed)
        return values

    @torch.no_grad()
    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Score marginal dependency unions; retain full-tensor normalization.

        Args:
            context: Current graph, parameter bindings and dependency queries.
            candidates: Removal proposals, scored independently in input order.
            accepted_impact: Complete closure of already accepted removals.

        Returns:
            One finite importance per candidate; smaller is preferred.

        Raises:
            PlanningError: Weight gradients are missing, sparse, nonfinite or
                complex, or a candidate has no matrix/kernel weight effects.
        """
        context.graph.validate_impact(accepted_impact)
        context.require_complete(accepted_impact)
        bindings = dict(context.graph.tensor_bindings())
        weights = {
            ref
            for operation in context.graph.operations()
            for name, ref in operation.bindings.items()
            if ref.kind == "parameter"
            and len(ref.shape) >= 2
            and (name == "weight" or name.endswith(".weight") or name.endswith("_proj_weight"))
        }
        results = []
        for candidate in candidates:
            selections = parameter_regions(context, candidate, accepted_impact=accepted_impact)
            total = 0.0
            for selection in selections:
                if selection.tensor not in weights:
                    continue
                weight = bindings[selection.tensor]
                gradient = weight.grad
                if weight.is_complex() or gradient is None or gradient.layout != torch.strided:
                    raise PlanningError(
                        "EvolvedMetric requires real weights and dense current gradients"
                    )
                for region in selection.regions:
                    partial = [
                        d
                        for d, indices in enumerate(region.axes)
                        if len(indices) != weight.shape[d]
                    ]
                    if len(partial) == 1:
                        dim = partial[0]
                        values = self._axis_sums(weight, dim)
                        total += math.fsum(values[i] for i in region.axes[dim])
                    else:
                        # Reduce after full-tensor normalization, including for
                        # intersections left by conditional row/column removal.
                        total += float(gather_region(self._terms(weight, gradient), region).sum())
            if not any(s.tensor in weights for s in selections):
                original = context.impact(candidate.remove)
                context.require_complete(original)
                if not any(s.tensor in weights for s in original.parameters):
                    raise PlanningError(
                        "EvolvedMetric candidate has no matrix/kernel weight effects"
                    )
            if not math.isfinite(total):
                raise PlanningError("EvolvedMetric produced a nonfinite score")
            results.append(total)
        return results
