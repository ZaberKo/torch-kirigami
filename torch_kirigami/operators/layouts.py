"""Small structural descriptors consumed by analysis, candidates, and lowering."""

from dataclasses import dataclass

from ..selection import Region, TensorRef


@dataclass(frozen=True)
class PartitionedLayout:
    """Compact each original partition independently, then concatenate in order.

    Regions describe physical storage coordinates once. Every consumer uses
    these same boundaries, including grouped and depthwise convolution.
    """

    tensor: TensorRef
    partitions: tuple[Region, ...]
    concat_dim: int = 0

    def retained(self, selection):
        """Return Cartesian retained regions without allocating tensor data."""
        result = []
        for region in self.partitions:
            axes = tuple(
                indices.subtract(selection.project(dim, region))
                for dim, indices in enumerate(region.axes)
            )
            if all(axes):
                result.append(Region(axes))
        return tuple(result)


@dataclass(frozen=True)
class CallContract:
    """Declare reusable original-call argument and layout behavior.

    Shape arguments may change only when explicitly declared. Other scalar
    arguments retain their captured values, including reduction and slice axes.
    Native evaluation checks shapes; it cannot by itself prove backend strides.
    """

    shape_arguments: bool = False
    fresh_output: bool = False
    output_layout: str = "unknown"
