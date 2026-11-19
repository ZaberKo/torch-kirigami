"""Static operator recipes shared by analysis extensions and pruning execution.

These records contain coordinates and literal configuration, never a graph,
model, callback, or allocated replacement weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from .configuration import freeze, thaw
from .dep_graph.selection import Region, Selection, TensorRef, concatenated_shape


@dataclass(frozen=True)
class TensorRecipe:
    """A static recipe that builds a compact tensor from retained original regions.

    Execution gathers each Cartesian segment from the original tensor and
    concatenates the segments in order along `concat_dim`. The segments should
    be disjoint. Their order defines the old-to-new coordinate mapping, so no
    elementwise index map is allocated.

    Attributes:
        tensor: Original tensor reference and its captured metadata.
        segments: Nonempty retained source regions in concatenation order.
        concat_dim: Dimension along which the segments are concatenated. A
            scalar tensor uses dimension zero.
        memory_format: Layout of the compact tensor: `"contiguous"`,
            `"channels_last"` for rank-4 tensors, or `"channels_last_3d"` for
            rank-5 tensors.

    Raises:
        ValueError: The tensor or segments are missing, a segment is empty or
            has the wrong rank, `concat_dim` or `memory_format` is invalid for
            the tensor rank, or the segments cannot be concatenated.
        IndexError: A segment lies outside the tensor shape.
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
        """Compact tensor shape produced by the segments, computed without tensor data."""
        return concatenated_shape(self.segments, self.concat_dim)


@dataclass(frozen=True)
class AttributeRecipe:
    """A static module-attribute assignment guarded by the attribute's captured value.

    Construction stores `old` and `new` as frozen configuration, preserving
    their exact types and values. Use `torch_kirigami.configuration.thaw` to read
    the values. Execution checks the original configuration before assignment.

    Attributes:
        path: Qualified path to the module attribute.
        old: Frozen value that the attribute must have before the edit.
        new: Frozen value to assign. Both inputs accept configuration supported
            by `torch_kirigami.configuration.freeze`.

    Raises:
        ValueError: `path` is empty or not a string, or a value is not
            supported configuration.
    """

    path: str
    old: object
    new: object

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Attribute recipes require a path")
        try:
            old, new = freeze(thaw(self.old)), freeze(thaw(self.new))
        except TypeError as error:
            raise ValueError("Attribute recipes require portable configuration values") from error
        object.__setattr__(self, "old", old)
        object.__setattr__(self, "new", new)


@dataclass(frozen=True)
class OperatorRecipe:
    """Static edits for one operator call and declared facts about its compact outputs.

    A custom rule returns this record from `build_recipe` and is responsible
    for proving that the edited call stays compatible with the original forward
    computation. A rule can declare output strides that it has proved, so that
    downstream view checks can use them.

    Attributes:
        tensors: Tensor recipes for this call.
        attributes: Module attribute recipes for this call.
        notes: Explanatory notes copied into the compiled plan.
        output_strides: Proved strides of compact outputs, as pairs of a tensor
            reference and one nonnegative stride per dimension. Each tensor
            appears at most once.

    Raises:
        TypeError: An entry of `tensors`, `attributes`, or `notes` has the wrong type.
        ValueError: A stride declaration has the wrong rank, a negative or
            non-integer stride, or a duplicate tensor reference.
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
