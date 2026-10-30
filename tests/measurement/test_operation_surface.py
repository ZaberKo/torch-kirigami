"""Equivalent spellings, fused kernels, and per-call coverage regressions."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.flop_counter import flop_registry

from torch_kirigami.measurement import calculate_model_complexity


@torch.library.custom_op("kirigami_measurement::square", mutates_args=())
def custom_square(x: torch.Tensor) -> torch.Tensor:
    return x * x


@pytest.mark.parametrize(
    "name",
    [
        "abs",
        "absolute",
        "neg",
        "negative",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "reciprocal",
        "square",
    ],
)
@pytest.mark.parametrize("form", ["torch", "out", "method", "inplace"])
def test_unary_spellings_share_counts(name, form, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            if form == "torch":
                return getattr(torch, name)(x)
            if form == "out":
                return getattr(torch, name)(x, out=torch.empty_like(x))
            return getattr(x, name + ("_" if form == "inplace" else ""))()

    sample = torch.ones(2, 3)
    result = calculate_model_complexity(Model(), sample, device=execution_device)
    assert result.flops == 6
    assert result.macs == 0
    assert result.unsupported_flop_ops == result.unsupported_ops == ()
    torch.testing.assert_close(sample, torch.ones(2, 3))


@pytest.mark.parametrize(
    "name", ["sub", "subtract", "mul", "multiply", "div", "divide", "true_divide"]
)
@pytest.mark.parametrize("form", ["torch", "out", "method", "inplace"])
def test_binary_spellings_share_counts(name, form, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            if form == "torch":
                return getattr(torch, name)(x, 2)
            if form == "out":
                return getattr(torch, name)(x, 2, out=torch.empty_like(x))
            return getattr(x, name + ("_" if form == "inplace" else ""))(2)

    result = calculate_model_complexity(Model(), torch.ones(2, 3), device=execution_device)
    assert result.flops == 6
    assert result.macs == 0
    assert result.unsupported_flop_ops == result.unsupported_ops == ()


@pytest.mark.parametrize("name", ["addcmul", "addcdiv"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("value", [1, 3])
def test_scaled_elementwise_product(name, inplace, value, execution_device):
    class Model(nn.Module):
        def forward(self, x, y):
            return getattr(x, name + ("_" if inplace else ""))(x, y, value=value)

    result = calculate_model_complexity(
        Model(), (torch.ones(2, 3), torch.ones(3)), device=execution_device
    )
    assert result.flops == 6 * (2 if value == 1 else 3)
    assert result.macs == 0
    assert result.unsupported_flop_ops == result.unsupported_ops == ()


@pytest.mark.parametrize("name", ["addmm", "baddbmm", "addbmm"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("alpha,beta", [(1, 1), (2, 3), (1, 0)])
def test_biased_matrix_spellings(name, inplace, alpha, beta, execution_device):
    class Model(nn.Module):
        def forward(self, bias, a, b):
            return getattr(bias, name + ("_" if inplace else ""))(a, b, alpha=alpha, beta=beta)

    batch_shape = () if name == "addmm" else (5,)
    bias_shape = (5, 2, 4) if name == "baddbmm" else (2, 4)
    a, b = torch.ones(*batch_shape, 2, 3), torch.ones(*batch_shape, 3, 4)
    bias = torch.ones(bias_shape)
    result = calculate_model_complexity(Model(), (bias, a, b), device=execution_device)
    macs = 2 * 3 * 4 * (1 if name == "addmm" else 5)
    extras = {(1, 1): 1, (2, 3): 3, (1, 0): 0}[alpha, beta]
    assert result.macs == macs
    assert result.flops == 2 * macs + bias.numel() * extras
    assert result.unsupported_flop_ops == result.unsupported_ops == ()


@pytest.mark.parametrize("include_visible", [False, True])
def test_hidden_arithmetic_is_not_covered_by_another_call(include_visible, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            # deg2rad executes a multiplication below Python dispatch. This
            # separate visible multiplication must not mark that call covered.
            hidden = torch.deg2rad(x)
            return (hidden, x * x) if include_visible else hidden

    result = calculate_model_complexity(Model(), torch.ones(2, 3), device=execution_device)
    assert result.flops == (6 if include_visible else 0)
    assert "aten::deg2rad" in result.unsupported_flop_ops
    assert "aten::mul" in result.unsupported_flop_ops


@pytest.mark.parametrize("safe", [False, True])
def test_softmax_and_safe_softmax_are_both_counted(safe, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            plain = x.softmax(-1)
            return (plain, torch.ops.aten._safe_softmax(x, -1)) if safe else plain

    # Four subtractions, four exponentials, three additions, four divisions per row.
    result = calculate_model_complexity(Model(), torch.ones(2, 4), device=execution_device)
    assert result.flops == 30 * (2 if safe else 1)
    assert result.unsupported_flop_ops == result.unsupported_ops == ()


@pytest.mark.parametrize("key_length", [3, 7])
def test_raw_efficient_attention_uses_sequence_before_heads(key_length, execution_device):
    if execution_device != "cuda":
        pytest.skip("The efficient attention kernel requires CUDA")

    class Model(nn.Module):
        def forward(self, q, k, v):
            return torch.ops.aten._efficient_attention_forward(
                q, k, v, None, None, None, None, None, 0.0, 0
            )[0]

    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, key_length, 4, 8)
    v = torch.randn(2, key_length, 4, 16)
    reference = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    torch.testing.assert_close(Model()(q, k, v), reference, atol=1e-5, rtol=1e-5)
    result = calculate_model_complexity(Model(), (q, k, v), device=execution_device)
    assert result.macs == 2 * 4 * 3 * key_length * (8 + 16)
    assert result.flops == 2 * result.macs
    assert result.unsupported_ops == ()
    assert "aten::_efficient_attention_forward" in result.unsupported_flop_ops


def test_raw_cpu_attention_counts_both_matrix_products():
    class Model(nn.Module):
        def forward(self, q, k, v):
            return torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(q, k, v)[0]

    q, k, v = torch.randn(2, 4, 3, 8), torch.randn(2, 4, 7, 8), torch.randn(2, 4, 7, 8)
    torch.testing.assert_close(Model()(q, k, v), F.scaled_dot_product_attention(q, k, v))
    result = calculate_model_complexity(Model(), (q, k, v))
    assert result.macs == 2 * 4 * 3 * 7 * (8 + 8)
    assert result.unsupported_ops == ()
    assert "aten::_scaled_dot_product_flash_attention_for_cpu" in result.unsupported_flop_ops


def test_custom_namespace_formula_covers_its_own_call(monkeypatch, execution_device):
    class Model(nn.Module):
        def forward(self, x):
            return custom_square(x)

    monkeypatch.setitem(
        flop_registry, torch.ops.kirigami_measurement.square, lambda x, **kwargs: x.numel()
    )
    result = calculate_model_complexity(Model(), torch.ones(2, 3), device=execution_device)
    assert result.flops == 6
    assert result.unsupported_flop_ops == ()
    # A user FLOP formula does not define how to convert arbitrary work to MACs.
    assert result.unsupported_ops == ("kirigami_measurement::square",)


def test_native_backward_formula_retains_its_partial_coverage(execution_device):
    class Model(nn.Module):
        def forward(self, x, w):
            return torch.ops.aten.convolution_backward(
                x, x, w, [1], [1], [0], [1], False, [0], 1, [True, False, True]
            )

    result = calculate_model_complexity(
        Model(), (torch.ones(1, 1, 3), torch.ones(1, 1, 1)), device=execution_device
    )
    # The native formula counts input gradients but omits the bias reduction.
    assert result.flops == 6
    assert "aten::convolution_backward" in result.unsupported_flop_ops


def test_static_split_indices_do_not_require_a_host_tensor(execution_device):
    class Model(nn.Module):
        def forward(self, x, indices):
            assert x.device.type == execution_device
            return torch.tensor_split(x, indices, dim=-1)

    result = calculate_model_complexity(
        Model(), (torch.ones(2, 6), [2, 4]), device=execution_device
    )
    assert result.flops == result.macs == 0
