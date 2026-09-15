"""Pure scalar schedules and selection statistics, without training policy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast


def _step(step: int) -> None:
    """Validate a nonnegative integer schedule step."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")


def _finite(*values: int | float) -> None:
    """Validate finite scalar schedule values."""
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("Expected finite scalar schedule values")


@dataclass(frozen=True)
class Constant:
    """Return a fixed finite value for every nonnegative step."""

    value: float

    def __post_init__(self) -> None:
        _finite(self.value)

    def __call__(self, step: int) -> float:
        """Evaluate without changing progress."""
        _step(step)
        return float(self.value)


@dataclass(frozen=True)
class Polynomial:
    """Interpolate start + (end-start) * progress**power, clamped at endpoints.

    Steps before begin return start; steps at/after finish return end. Progress
    is supplied by the caller, so schedules need no mutable checkpoint state.
    """

    start: float
    end: float
    finish: int
    begin: int = 0
    power: float = 1.0

    def __post_init__(self) -> None:
        _finite(self.start, self.end, self.power)
        _step(self.begin)
        _step(self.finish)
        if self.finish <= self.begin or self.power <= 0:
            raise ValueError("Require finish > begin and power > 0")

    def __call__(self, step: int) -> float:
        """Evaluate at an explicit step."""
        _step(step)
        if step <= self.begin:
            return float(self.start)
        if step >= self.finish:
            return float(self.end)
        progress = ((step - self.begin) / (self.finish - self.begin)) ** self.power
        # Opposite signs can overflow end-start despite a finite convex result.
        # Same-sign subtraction is bounded and avoids summing rounded endpoints.
        if (self.start < 0) != (self.end < 0):
            return (1 - progress) * self.start + progress * self.end
        return self.start + (self.end - self.start) * progress


def Linear(start: float, end: float, finish: int, *, begin: int = 0) -> Polynomial:
    """Construct a linear schedule with explicit endpoint steps."""
    return Polynomial(start, end, finish, begin)


@dataclass(frozen=True)
class Piecewise:
    """Return the value at the latest milestone; first milestone must be zero."""

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        points: tuple[tuple[int, float], ...] = tuple(self.points)
        if not points or points[0][0] != 0:
            raise ValueError("Piecewise requires a zero starting milestone")
        previous = -1
        for step, value in points:
            _step(step)
            _finite(value)
            if step <= previous:
                raise ValueError("Milestones must be strictly increasing")
            previous = step
        object.__setattr__(self, "points", points)

    def __call__(self, step: int) -> float:
        """Evaluate without mutating the milestones."""
        _step(step)
        return next(float(v) for s, v in reversed(self.points) if s <= step)


def _sets(selection: Mapping[str, Iterable[int]]) -> dict[str, frozenset[int]]:
    """Normalize a selection mapping into immutable integer sets."""
    result = {}
    for key, values in selection.items():
        if not isinstance(key, str):
            raise ValueError("Selection domains must have string keys")
        indices = tuple(values)
        if any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Selection identities must be nonnegative integers")
        result[key] = frozenset(indices)
    if not result:
        raise ValueError("At least one selection domain is required")
    return result


def selection_similarity(
    left: Mapping[str, Iterable[int]], right: Mapping[str, Iterable[int]]
) -> float:
    """Average per-domain Jaccard of retained identities, with empty/empty = 1."""
    left, right = _sets(left), _sets(right)
    if left.keys() != right.keys():
        raise ValueError("Selection domains changed; reset the statistics")
    return sum(
        len(left[k] & right[k]) / len(left[k] | right[k]) if left[k] | right[k] else 1.0
        for k in left
    ) / len(left)


class SelectionWindow:
    """Track adjacent-selection similarity, leaving thresholds and triggers external.

    Args:
        size: Number of adjacent comparisons needed before returning an average.

    Call reset after physical pruning or a change of coordinate universe. State
    contains only domain strings, integer identities and finite similarities.
    """

    def __init__(self, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("Window size must be a positive integer")
        self.size = size
        self.reset()

    def reset(self) -> None:
        """Forget comparisons at an explicit structural-stage boundary."""
        self._previous: dict[str, frozenset[int]] | None = None
        self._values: list[float] = []

    def update(self, selection: Mapping[str, Iterable[int]]) -> float | None:
        """Observe retained identities; return None until the window is full."""
        current = _sets(selection)
        values = list(self._values)
        if self._previous is not None:
            values.append(selection_similarity(self._previous, current))
        self._previous, self._values = current, values[-self.size :]
        return sum(self._values) / self.size if len(self._values) == self.size else None

    def state_dict(self) -> dict[str, object]:
        """Return independent plain data for caller-owned training checkpoints."""
        return {
            "size": self.size,
            "previous": None
            if self._previous is None
            else {k: sorted(v) for k, v in self._previous.items()},
            "values": list(self._values),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Validate the complete state before replacing any current statistics."""
        if (
            set(state) != {"size", "previous", "values"}
            or type(state["size"]) is not int
            or state["size"] != self.size
        ):
            raise ValueError("Incompatible selection window state")
        previous = (
            None
            if state["previous"] is None
            else _sets(cast(Mapping[str, Iterable[int]], state["previous"]))
        )
        values = tuple(cast(Iterable[float], state["values"]))
        _finite(*values)
        if (
            len(values) > self.size
            or any(not 0 <= v <= 1 for v in values)
            or (previous is None and values)
        ):
            raise ValueError("Invalid selection window history")
        self._previous, self._values = previous, list(values)
