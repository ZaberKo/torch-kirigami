"""Static operator recipes shared by analysis extensions and pruning execution.

These records contain coordinates and literal configuration, never a graph,
model, callback, or allocated replacement weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .configuration import FrozenDict, FrozenList, FrozenScalar, freeze, thaw
from .dep_graph.selection import Region, Selection, TensorRef, concatenated_shape


@dataclass(frozen=True)
class TensorRecipe:
    """Gather Cartesian retained segments and concatenate in original group order.

    Segments are disjoint regions in the original tensor. Concatenation defines
    the old-to-new coordinate mapping without allocating an elementwise map.

    Attributes:
        tensor: Original tensor reference and its captured metadata.
        segments: Retained source regions in concatenation order.
        concat_dim: Dimension along which the regions are concatenated.
        memory_format: Layout used for the compact tensor.
        shape: Compact tensor shape after applying the segments.
    """

    tensor: TensorRef
    segments: tuple[Region, ...]
    concat_dim: int = 0
    memory_format: str = "contiguous"

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments or not isinstance(self.tensor, TensorRef):
            raise ValueError("Tensor recipe requires a tensor and nonempty segments")
        if type(self.concat_dim) is not int or not 0 <= self.concat_dim < max(
            1, len(self.tensor.shape)
        ):
            raise ValueError("Invalid recipe concatenation dimension")
        if self.memory_format not in ("contiguous", "channels_last", "channels_last_3d"):
            raise ValueError("Unsupported recipe memory format")
        rank = len(self.tensor.shape)
        if self.memory_format != "contiguous" and rank != (
            4 if self.memory_format == "channels_last" else 5
        ):
            raise ValueError("Memory format is incompatible with tensor rank")
        for region in segments:
            if not isinstance(region, Region) or region.empty:
                raise ValueError("Recipe segments must be nonempty regions")
            Selection(self.tensor, (region,))
        try:
            concatenated_shape(segments, self.concat_dim)
        except ValueError as error:
            raise ValueError("Recipe segments cannot concatenate") from error
        object.__setattr__(self, "segments", segments)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the resulting tensor shape without allocating tensor data."""
        return concatenated_shape(self.segments, self.concat_dim)


@dataclass(frozen=True)
class AttributeRecipe:
    """Assign a verified attribute; lists use detached FrozenList configuration.

    Attributes:
        path: Qualified path to the module attribute.
        old: Captured value required before the edit.
        new: Value to assign after validation.
    """

    path: str
    old: Any
    new: Any

    def __post_init__(self) -> None:
        def frozen(value: object) -> object:
            """Freeze supported configuration values for portable storage."""
            if isinstance(value, (dict, FrozenDict, list, FrozenList, FrozenScalar, torch.Size)):
                return freeze(thaw(value))
            if isinstance(value, tuple):
                return tuple(frozen(v) for v in value)
            if value is None or type(value) in (int, float, bool, str):
                return value
            raise ValueError("Attribute recipes require scalar/list/tuple configuration values")

        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Attribute recipes require a path")
        object.__setattr__(self, "old", frozen(self.old))
        object.__setattr__(self, "new", frozen(self.new))


@dataclass(frozen=True)
class OperatorRecipe:
    """Static per-call edits and declared compact-output facts.

    Custom rules are responsible for proving original-forward compatibility.
    Output strides can be supplied when proved, allowing downstream view checks.

    Attributes:
        tensors: Tensor recipes for this call.
        attributes: Module attribute recipes for this call.
        notes: Explanatory notes emitted with the compiled plan.
        output_strides: Optional proved output strides keyed by tensor reference.
    """

    tensors: tuple[TensorRecipe, ...] = ()
    attributes: tuple[AttributeRecipe, ...] = ()
    notes: tuple[str, ...] = ()
    output_strides: tuple[tuple[TensorRef, tuple[int, ...]], ...] = ()

    def __post_init__(self) -> None:
        for name, cls in (
            ("tensors", TensorRecipe),
            ("attributes", AttributeRecipe),
            ("notes", str),
        ):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, cls) for value in values):
                raise TypeError(f"Invalid rewrite {name}")
            object.__setattr__(self, name, values)
        strides = tuple((ref, tuple(stride)) for ref, stride in self.output_strides)
        if any(
            not isinstance(ref, TensorRef)
            or len(stride) != len(ref.shape)
            or any(type(n) is not int or n < 0 for n in stride)
            for ref, stride in strides
        ):
            raise ValueError("Output strides must match tensor ranks")
        if len({ref for ref, _ in strides}) != len(strides):
            raise ValueError("Duplicate output stride declarations")
        object.__setattr__(self, "output_strides", strides)
