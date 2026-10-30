"""Independent nominal arithmetic references and native formula regressions."""

import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import flop_registry

from torch_kirigami.measurement import ModelComplexity, calculate_model_complexity


def test_default_uncovered_ops_dictionaries_are_not_shared():
    first = ModelComplexity(macs=0, flops=0, params=0)
    second = ModelComplexity(macs=0, flops=0, params=0)
    assert first.uncovered_ops == {"macs": (), "flops": ()}
    first.uncovered_ops["flops"] = ("changed",)
    assert second.uncovered_ops == {"macs": (), "flops": ()}


def test_uncovered_ops_are_sorted_unique_and_independent(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            # Repeated calls and different uncovered operators must produce
            # unique names, while recording the two counting conventions separately.
            return F.gelu(x), F.gelu(x), torch.lgamma(x)

    first = calculate_model_complexity(Model(), torch.ones(2, 4), device=execution_device)
    second = calculate_model_complexity(Model(), torch.ones(2, 4), device=execution_device)
    assert first.uncovered_ops == {"macs": (), "flops": ("aten::gelu", "aten::lgamma")}
    assert json.loads(json.dumps(first.uncovered_ops)) == {
        "macs": [],
        "flops": ["aten::gelu", "aten::lgamma"],
    }
    first.uncovered_ops["macs"] = ("changed",)
    assert second.uncovered_ops["macs"] == ()
    assert not hasattr(second, "unsupported_ops")
    assert not hasattr(second, "unsupported_flop_ops")


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda x: x + x, 8),
        (lambda x: torch.add(x, x, alpha=2), 16),
        (lambda x: x * x, 8),
        (lambda x: x / 2, 8),
        (lambda x: x.sum(), 7),
        (lambda x: x.sum(-1, keepdim=True), 6),
        (lambda x: x.mean(-1), 8),
        (lambda x: x.softmax(-1), 30),
        (lambda x: x.log_softmax(-1), 32),
        (lambda x: x.exp(), 8),
    ],
)
def test_nominal_arithmetic_formulas(operation, expected, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return operation(x)

    result = calculate_model_complexity(Model(), torch.ones(2, 4), device=execution_device)
    assert result.flops == expected
    assert result.macs == 0
    assert result.uncovered_ops["flops"] == ()
    assert result.uncovered_ops["macs"] == ()


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("kind", ["layer", "group", "batch"])
def test_normalization_counts_explicit_arithmetic(kind, affine, execution_device):
    x = torch.randn(2, 4, 3)
    if kind == "layer":
        model = nn.LayerNorm((4, 3), elementwise_affine=affine)
        # Two groups of twelve: mean (12), centering (12), variance (24),
        # epsilon/rsqrt (2), normalization (12), optional affine (24).
        expected = 124 + (48 if affine else 0)
    elif kind == "group":
        model = nn.GroupNorm(2, 4, affine=affine)
        # Four groups of six with the same operations as the LayerNorm reference.
        expected = 128 + (48 if affine else 0)
    else:
        model = nn.BatchNorm1d(4, affine=affine)
        # Four variance adjustments and rsqrts, then 24 subtracts and multiplies.
        expected = 56 + (48 if affine else 0)
    result = calculate_model_complexity(model, x, device=execution_device)
    assert result.flops == expected
    assert result.macs == 0
    assert result.uncovered_ops["flops"] == ()


def test_flop_coverage_is_separate_from_mac_coverage(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return torch.special.gammaln(x)

    result = calculate_model_complexity(Model(), torch.ones(2, 4), device=execution_device)
    assert result.macs == result.flops == 0
    assert result.uncovered_ops["macs"] == ()
    assert "aten::lgamma" in result.uncovered_ops["flops"]


@pytest.mark.parametrize("count", [-1, 2.5, float("nan"), float("inf"), True])
def test_invalid_non_mac_formula_fails_before_aggregation(count, monkeypatch, execution_device):
    calls = []

    def formula(*args, **kwargs):
        calls.append(1)
        return count if len(calls) == 1 else 100

    monkeypatch.setitem(flop_registry, torch.ops.aten.relu, formula)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("state", torch.zeros(1))

        def forward(self, x):
            self.state.add_(1)
            return x.relu().relu()

    model = Model().train()
    buffer = model.state
    fastpath = torch.backends.mha.get_fastpath_enabled()
    with pytest.raises(ValueError, match="nonnegative integer FLOP count"):
        calculate_model_complexity(model, torch.ones(2, 4), device=execution_device)
    assert calls == [1]
    assert model.training and model.state is buffer
    assert buffer.item() == 0
    assert torch.backends.mha.get_fastpath_enabled() == fastpath


def test_valid_external_flops_remain_python_integers(monkeypatch, execution_device):
    huge = 10**30 + 1
    monkeypatch.setitem(flop_registry, torch.ops.aten.relu, lambda *args, **kwargs: huge)
    result = calculate_model_complexity(nn.ReLU(), torch.ones(2, 4), device=execution_device)
    assert type(result.flops) is int and result.flops == huge
    assert result.macs == 0
    assert not result.uncovered_ops["flops"]


@pytest.mark.parametrize("benchmark", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_raw_convolution_signatures_and_bias(benchmark, bias, execution_device):
    groups = 2 if execution_device == "cuda" else 1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(6, 4 // groups, 3, 3))
            self.bias = nn.Parameter(torch.randn(6)) if bias else None

        def forward(self, x):
            if execution_device == "cpu":
                return torch.ops.aten._slow_conv2d_forward.default(
                    x, self.weight, [3, 3], self.bias, [2, 2], [1, 1]
                )
            y = torch.ops.aten.cudnn_convolution.default(
                x, self.weight, [1, 1], [2, 2], [1, 1], groups, benchmark, False, False
            )
            return y if self.bias is None else y + self.bias[None, :, None, None]

    model = Model()
    x = torch.randn(2, 4, 7, 9)
    torch.testing.assert_close(
        model(x),
        F.conv2d(x, model.weight, model.bias, stride=2, padding=1, groups=groups),
        rtol=1e-4,
        atol=1e-4,
    )
    result = calculate_model_complexity(model, x, device=execution_device)
    macs = 2 * 6 * 4 * 5 * (4 // groups) * 9
    assert result.macs == macs
    assert result.flops == 2 * macs + (2 * 6 * 4 * 5 if bias else 0)
    assert result.uncovered_ops["flops"] == result.uncovered_ops["macs"] == ()


def test_integer_work_does_not_become_floating_point_work(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return (x @ x) + 1 if execution_device == "cpu" else x + 1

    # CUDA int64 GEMM is not generally available; integer pointwise work is.
    macs = 0 if execution_device == "cuda" else 8
    result = calculate_model_complexity(
        Model(), torch.ones(2, 2, dtype=torch.int64), device=execution_device
    )
    assert result.flops == 0
    assert result.macs == macs


def test_training_statistics_remain_explicitly_uncovered(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return F.batch_norm(x, None, None, training=True)

    result = calculate_model_complexity(Model(), torch.randn(2, 4, 3), device=execution_device)
    assert any("batch_norm" in op for op in result.uncovered_ops["flops"])


def test_complex_abs_does_not_claim_a_real_scalar_formula(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return x.abs()

    result = calculate_model_complexity(
        Model(), torch.ones(2, 4, dtype=torch.complex64), device=execution_device
    )
    assert result.flops == 0
    assert "aten::abs" in result.uncovered_ops["flops"]


@pytest.mark.parametrize("form", ["dot", "mv", "mm", "linear"])
def test_complex_products_preserve_logical_macs_without_real_flop_conversion(
    form, execution_device
):
    class Model(nn.Module):
        def forward(self, a, b):
            if form == "linear":
                return F.linear(a, b, torch.ones(3, device=a.device, dtype=a.dtype))
            return getattr(torch, form)(a, b)

    if form == "dot":
        a, b, macs = torch.ones(4, dtype=torch.complex64), torch.ones(4, dtype=torch.complex64), 4
    elif form == "mv":
        a, b, macs = (
            torch.ones(3, 4, dtype=torch.complex64),
            torch.ones(4, dtype=torch.complex64),
            12,
        )
    else:
        a = torch.ones(2, 4, dtype=torch.complex64)
        b = torch.ones((3, 4) if form == "linear" else (4, 3), dtype=torch.complex64)
        macs = 24
    result = calculate_model_complexity(Model(), (a, b), device=execution_device)
    assert result.macs == macs
    assert result.flops == 0
    assert result.uncovered_ops["flops"]
    assert result.uncovered_ops["macs"] == ()


def test_uncovered_complex_product_keeps_other_known_real_flops(execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return x @ x, x.real.square().sum()

    result = calculate_model_complexity(
        Model(), torch.ones(2, 2, dtype=torch.complex64), device=execution_device
    )
    assert result.macs == 8
    assert result.flops == 7  # Four real squares and three reduction additions.
    assert "aten::mm" in result.uncovered_ops["flops"]
