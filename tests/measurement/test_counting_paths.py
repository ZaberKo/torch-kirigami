"""Inference-mode and equivalent-operation checks independent of counter formulas."""

import pytest
import torch
from torch import nn
from torch.utils.flop_counter import flop_registry

from torch_kirigami.measurement import calculate_model_complexity


@pytest.mark.parametrize("mode", [torch.enable_grad, torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("fastpath", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_counting_uses_no_grad_and_restores_modes(mode, fastpath, fail, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.inference = nn.Linear(4, 3)
            self.other = nn.Linear(4, 10)

        def forward(self, x):
            assert not torch.is_inference_mode_enabled()
            assert not torch.backends.mha.get_fastpath_enabled()
            output = self.other(x) if torch.is_grad_enabled() else self.inference(x)
            assert not output.requires_grad
            if fail:
                raise RuntimeError("injected counting failure")
            return output

    original = torch.backends.mha.get_fastpath_enabled()
    model = Model().train()
    sample = torch.ones(2, 4)
    backend_flags = (
        torch.backends.cuda.flash_sdp_enabled,
        torch.backends.cuda.math_sdp_enabled,
        torch.backends.cuda.mem_efficient_sdp_enabled,
        torch.backends.cuda.cudnn_sdp_enabled,
    )
    saved_flags = tuple(read() for read in backend_flags)
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        with mode():
            before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
            if fail:
                with pytest.raises(RuntimeError, match="injected counting failure"):
                    calculate_model_complexity(model, sample)
            else:
                result = calculate_model_complexity(model, sample)
                assert result.macs == 2 * 4 * 3
                assert result.uncovered_ops["macs"] == ()
            assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == before
        assert torch.backends.mha.get_fastpath_enabled() == fastpath
        assert tuple(read() for read in backend_flags) == saved_flags
        assert model.training and all(p.grad is None for p in model.parameters())
    finally:
        torch.backends.mha.set_fastpath_enabled(original)


@pytest.mark.parametrize("need_weights", [False, True])
def test_mha_counts_do_not_depend_on_parameter_freezing(need_weights, execution_device):
    model = nn.MultiheadAttention(8, 2, batch_first=True).eval()
    x = torch.randn(2, 5, 8)
    results = []
    for frozen in (False, True):
        model.requires_grad_(not frozen)
        result = calculate_model_complexity(
            model, (x, x, x), input_kwargs={"need_weights": need_weights}
        )
        # Three input projections, output projection, QK and attention-times-V.
        assert result.macs == 4 * 2 * 5 * 8 * 8 + 2 * 2 * 5 * 5 * 8
        assert result.flops > 2 * result.macs  # Bias, scaling and softmax arithmetic.
        assert result.uncovered_ops["macs"] == ()
        assert all(p.requires_grad is not frozen for p in model.parameters())
        results.append(result)
    assert results[0] == results[1]


def test_transformer_counts_do_not_depend_on_parameter_freezing(execution_device):
    model = nn.TransformerEncoderLayer(8, 2, dim_feedforward=12, batch_first=True).eval()
    x = torch.randn(2, 5, 8)
    before = calculate_model_complexity(model, x)
    model.requires_grad_(False)
    after = calculate_model_complexity(model, x)
    assert before == after
    assert after.macs == 3360 + 2 * 2 * 5 * 8 * 12
    assert after.uncovered_ops["macs"] == ()


@pytest.mark.parametrize("width", [1, 7])
@pytest.mark.parametrize(
    "form", ["dot", "vdot", "inner_matmul", "mv", "mv_matmul", "mm", "addmv", "addmv_"]
)
def test_vector_products_have_consistent_dense_counts(form, width, execution_device):
    class Model(nn.Module):
        def forward(self, a, b):
            if form in ("dot", "vdot"):
                return getattr(torch, form)(a, b)
            if form in ("inner_matmul", "mv_matmul"):
                return a @ b
            if form == "mm":
                return (a @ b[:, None])[:, 0]
            if form == "mv":
                return torch.mv(a, b)
            bias = torch.ones(3, device=a.device)
            if form == "addmv_":
                return bias.addmv_(a, b, alpha=2, beta=3)
            return torch.addmv(bias, a, b, alpha=2, beta=3)

    vector = form in ("dot", "vdot", "inner_matmul")
    a = torch.randn(width) if vector else torch.randn(3, width)
    b = torch.randn(width)
    registry = dict(flop_registry)
    result = calculate_model_complexity(Model(), (a, b), device=execution_device)
    assert result.macs == width * (1 if vector else 3)
    assert result.flops == 2 * result.macs + (9 if form in ("addmv", "addmv_") else 0)
    assert result.uncovered_ops["macs"] == ()
    assert flop_registry == registry  # Supplementary formulas are local to this counter.


@pytest.mark.parametrize("width,expected", [(1, 0), (7, 7)])
def test_einsum_counts_exposed_matrix_work(width, expected, execution_device):
    class Model(nn.Module):
        def forward(self, a, b):
            return torch.einsum("i,i->", a, b)

    result = calculate_model_complexity(
        Model(), (torch.ones(width), torch.ones(width)), device=execution_device
    )
    # PyTorch simplifies the singleton contraction into elementwise multiply.
    # This counter excludes that work. It does not reconstruct symbolic MACs.
    assert result.macs == expected
    assert result.uncovered_ops["macs"] == ()


def test_known_zero_mac_calls_do_not_report_missing_coverage(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3, bias=False)

        def forward(self, x):
            return torch.special.gammaln(self.linear(x).abs() + 1)

    result = calculate_model_complexity(Model(), torch.ones(4), device=execution_device)
    assert result.macs == 12
    assert result.uncovered_ops["macs"] == ()
