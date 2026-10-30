"""Fourier transforms preserve coordinates outside their transformed axes."""

from torch import fft

from ...errors import UnsupportedOperation
from ..contracts import AxisBarrier
from ..operation import Operation, OperatorSpec, OutputContract
from .native import equal, one


def fourier(operation: Operation) -> OperatorSpec:
    """Map independent axes and protect transform sizes and frequency coordinates."""
    x, y = one(operation.argument("input", 0)), one(operation.result)
    target = operation.node.target
    name = target.__name__.removeprefix("fft_")
    if target in (fft.fftshift, fft.ifftshift):
        dims = operation.argument("dim", 1)
    else:
        default = (-2, -1) if name.endswith("2") else None if name.endswith("n") else -1
        dims = operation.argument("dim", 2, default)
        if dims is None:
            sizes = operation.argument("s", 1)
            dims = (
                tuple(range(len(x.shape) - len(sizes), len(x.shape))) if sizes is not None else None
            )
    if dims is None:
        dims = tuple(range(len(x.shape)))
    elif type(dims) is int:
        dims = (dims,)
    if not isinstance(dims, (tuple, list)) or not all(type(d) is int for d in dims):
        raise UnsupportedOperation("Fourier transform dimensions must be static integers")
    axes = {d % len(x.shape) for d in dims} if x.shape else set()
    return OperatorSpec(
        relations=tuple(
            equal(x, d, y, d, operation.name) for d in range(len(x.shape)) if d not in axes
        ),
        constraints=tuple(
            AxisBarrier(
                ref.axis(d),
                "Fourier transform and frequency coordinates stay fixed",
                operation.name,
            )
            for ref in (x, y)
            for d in sorted(axes)
        ),
        contract=OutputContract(output_layout="backend_dependent"),
    )
