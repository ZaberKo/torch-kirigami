"""Spectral axes remain fixed while independent features physically compact."""

import copy
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import PlanningError, Pruner, load_checkpoint, save_checkpoint

TRANSFORMS = [
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "hfft",
    "ihfft",
    "fft2",
    "ifft2",
    "rfft2",
    "irfft2",
    "hfft2",
    "ihfft2",
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "hfftn",
    "ihfftn",
    "fftshift",
    "ifftshift",
]


@pytest.mark.parametrize("name", TRANSFORMS)
def test_fourier_nontransform_axes_compact(name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            y = self.pre(x)
            if name.endswith(("2", "n")):
                y = getattr(torch.fft, name)(y, dim=(1, 2))
            else:
                y = getattr(torch.fft, name)(y, dim=2)
            return self.head(y.abs())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3, 4, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    keep = [0, 2, 3, 5]
    # Feature selection commutes with transforms on disjoint axes.
    projected = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    dims = (1, 2) if name.endswith(("2", "n")) else 2
    expected = F.linear(
        getattr(torch.fft, name)(projected, dim=dims).abs(),
        dense.head.weight[:, keep],
        dense.head.bias,
    )
    actual = model(sample)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    if name == "fft":
        stream = io.BytesIO()
        save_checkpoint(model, stream)
        stream.seek(0)
        torch.testing.assert_close(load_checkpoint(Model(), stream)(sample), expected)


def test_fft_matches_explicit_dft_and_rejects_spectral_axis_changes(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            return self.head(torch.fft.fft(self.pre(x), dim=1).abs())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 4, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    call = next(op for op in graph.operations() if op.node.target is torch.fft.fft)
    pruner = Pruner(model, graph=graph)
    with pytest.raises(PlanningError, match="unsupported_axis"):
        pruner.plan_remove([call.outputs[0].axis(1).select([0])])
    torch.testing.assert_close(model.pre.weight, dense.pre.weight)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    y = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    positions = torch.arange(4, dtype=y.dtype)
    phase = -2 * torch.pi * positions[:, None] * positions[None, :] / 4
    matrix = torch.complex(phase.cos(), phase.sin())
    transformed = torch.einsum("kt,btc->bkc", matrix, y.to(matrix.dtype))
    expected = F.linear(transformed.abs(), dense.head.weight[:, keep], dense.head.bias)
    torch.testing.assert_close(model(sample), expected)


def test_fftn_sizes_choose_trailing_axes_and_default_shift_blocks_changes(execution_device):
    class Model(nn.Module):
        def __init__(self, shift=False):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)
            self.shift = shift

        def forward(self, x):
            y = self.pre(x).transpose(1, 2)
            y = torch.fft.fftshift(y) if self.shift else torch.fft.fftn(y, s=(8,))
            return self.head(y.abs().transpose(1, 2))

    sample = torch.randn(2, 4, 3)
    shifted = Model(shift=True)
    graph = DependencyGraph.build(shifted, args=(sample,))
    with pytest.raises(PlanningError, match="unsupported_axis"):
        Pruner(shifted, graph=graph).plan_remove(
            [graph.parameter("pre.weight").axis(0).select([1])]
        )
    assert shifted.pre.out_features == 6
    model = Model().eval()
    dense = copy.deepcopy(model)
    graph = DependencyGraph.build(model, args=(sample,))
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    keep = [0, 2, 3, 4, 5]
    projected = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    padded = F.pad(projected.transpose(1, 2), (0, 4)).transpose(1, 2)
    positions = torch.arange(8, dtype=sample.dtype)
    phase = -2 * torch.pi * positions[:, None] * positions[None, :] / 8
    matrix = torch.complex(phase.cos(), phase.sin())
    transformed = torch.einsum("kt,btc->bkc", matrix, padded.to(matrix.dtype))
    expected = F.linear(transformed.abs(), dense.head.weight[:, keep], dense.head.bias)
    torch.testing.assert_close(model(sample), expected, atol=1e-5, rtol=1e-4)
