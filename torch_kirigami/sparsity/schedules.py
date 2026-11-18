"""Stateless scalar schedules and selection-stability statistics.

Schedules keep no progress state: the caller passes the current step on every
call and owns the training policy that uses the value.
"""

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
    """A schedule that returns the same value at every step.

    Attributes:
        value: Finite Python number returned at every step.

    Raises:
        ValueError: `value` is not a finite Python `int` or `float`.
    """

    value: float

    def __post_init__(self) -> None:
        _finite(self.value)

    def __call__(self, step: int) -> float:
        """Return the value at a step.

        Args:
            step: Nonnegative integer step supplied by the caller.

        Returns:
            `value` as a float.

        Raises:
            ValueError: `step` is not a nonnegative integer.
        """
        _step(step)
        return float(self.value)


@dataclass(frozen=True)
class Polynomial:
    """A schedule that moves from `start` to `end` along a power curve.

    Between `begin` and `finish`, the value is
    `start + (end - start) * progress**power`, where
    `progress = (step - begin) / (finish - begin)`. Steps at or before `begin`
    return `start`, and steps at or after `finish` return `end`. The caller
    supplies the step, so the schedule has no state to checkpoint.

    Attributes:
        start: Finite value at and before `begin`.
        end: Finite value at and after `finish`.
        finish: First step that returns `end`. It must be greater than `begin`.
        begin: Last step that returns `start`.
        power: Finite, positive exponent applied to the progress.

    Raises:
        ValueError: A value is not a finite Python number, a step is not a
            nonnegative integer, `finish <= begin`, or `power <= 0`.
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
        """Return the scheduled value at a step.

        Args:
            step: Nonnegative integer step supplied by the caller.

        Returns:
            The interpolated value as a float.

        Raises:
            ValueError: `step` is not a nonnegative integer.
        """
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
    """Construct a schedule that moves linearly from `start` to `end`.

    Args:
        start: Value at and before `begin`.
        end: Value at and after `finish`.
        finish: First step that returns `end`. It must be greater than `begin`.
        begin: Last step that returns `start`.

    Returns:
        A `Polynomial` schedule with power one.

    Raises:
        ValueError: The arguments violate the `Polynomial` requirements.
    """
    return Polynomial(start, end, finish, begin)


@dataclass(frozen=True)
class Piecewise:
    """A step schedule that holds each milestone value until the next milestone.

    Attributes:
        points: `(step, value)` pairs with strictly increasing nonnegative
            integer steps, starting at step zero, and finite values. Any
            iterable of two-item pairs is accepted; the pairs are copied into
            tuples, so later changes to the input do not affect the schedule.

    Raises:
        ValueError: The first step is not zero, the steps do not increase
            strictly, a pair does not have two items, or a step or value is
            invalid.
    """

    points: tuple[tuple[int, float], ...]

    def __post_init__(self) -> None:
        points: tuple[tuple[int, float], ...] = tuple((step, value) for step, value in self.points)
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
        """Return the value of the latest milestone at or before a step.

        Args:
            step: Nonnegative integer step supplied by the caller.

        Returns:
            The milestone value as a float.

        Raises:
            ValueError: `step` is not a nonnegative integer.
        """
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
    """Return the mean Jaccard similarity of two selections across their domains.

    For each domain, the similarity is the size of the intersection of the
    retained identities divided by the size of their union. Two empty sets
    have similarity one.

    Args:
        left: Retained nonnegative integer identities keyed by domain name.
        right: Next retained identities with the same domain names.

    Returns:
        The mean similarity, from zero to one.

    Raises:
        ValueError: A mapping is empty, has a non-string key or an invalid
            identity, or the mappings have different domain names.
    """
    left, right = _sets(left), _sets(right)
    if left.keys() != right.keys():
        raise ValueError("Selection domains changed; reset the statistics")
    return sum(
        len(left[k] & right[k]) / len(left[k] | right[k]) if left[k] | right[k] else 1.0
        for k in left
    ) / len(left)


class SelectionWindow:
    """Average the similarity of consecutive selections over a sliding window.

    Each `update()` compares the new selection with the previous one using
    `selection_similarity`. The caller decides what threshold or trigger to
    apply to the average. Call `reset()` after physical pruning or whenever the
    identities start to refer to a different set of coordinates. The state
    contains only domain names, integer identities, and finite similarities.

    Args:
        size: Positive number of consecutive comparisons to average.

    Attributes:
        size: Number of comparisons in the window.

    Raises:
        ValueError: `size` is not a positive integer.
    """

    def __init__(self, size: int) -> None:
        if type(size) is not int or size <= 0:
            raise ValueError("Window size must be a positive integer")
        self.size = size
        self.reset()

    def reset(self) -> None:
        """Forget the previous selection and every recorded comparison."""
        self._previous: dict[str, frozenset[int]] | None = None
        self._values: list[float] = []

    def update(self, selection: Mapping[str, Iterable[int]]) -> float | None:
        """Record a new selection and return the current window average.

        The state changes only if the selection is valid.

        Args:
            selection: Retained nonnegative integer identities keyed by domain name.

        Returns:
            The mean of the last `size` similarities, or `None` until `size`
            comparisons exist, which needs `size + 1` selections after a reset.

        Raises:
            ValueError: The selection is invalid or its domain names differ
                from those of the previous selection.
        """
        current = _sets(selection)
        values = list(self._values)
        if self._previous is not None:
            values.append(selection_similarity(self._previous, current))
        self._previous, self._values = current, values[-self.size :]
        return sum(self._values) / self.size if len(self._values) == self.size else None

    def state_dict(self) -> dict[str, object]:
        """Return the window state as new plain data for a caller-owned checkpoint.

        Returns:
            A dictionary with the window size, the previous selection as sorted
            lists (or `None`), and the recorded similarities.
        """
        return {
            "size": self.size,
            "previous": None
            if self._previous is None
            else {k: sorted(v) for k, v in self._previous.items()},
            "values": list(self._values),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore state returned by `state_dict()`.

        The complete state is validated before any current statistics change.

        Args:
            state: Mapping returned by `state_dict()` of a window with the same size.

        Raises:
            ValueError: The keys or size differ, the previous selection is
                invalid, or the similarity history is too long, outside
                `[0, 1]`, nonfinite, or present without a previous selection.
        """
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
