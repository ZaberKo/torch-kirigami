# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.14.0"]
# [tool.uv]
# index-url = "https://download.pytorch.org/whl/cpu"
# ///
"""Bounded capture experiments; not a dependency analyzer or pruning library.

Run: uv run experiments/capture_probe.py
The CPU index makes this probe independent of the application's CUDA setup.
"""

import copy
import json
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import prune


class FusedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 2
        self.attn_dim = 8
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        heads = F.scaled_dot_product_attention(q, k, v)
        return self.proj(heads.transpose(1, 2).reshape(batch, tokens, self.attn_dim))


class SharedWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=False)
        self.b = nn.Linear(8, 8, bias=False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.a(x) + self.b(x)


class Detached(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.requires_grad_(False)

    def forward(self, x):
        return self.linear(x).detach().sin()


class DataBranch(nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        return x - 1


class FixedLoop(nn.Module):
    def forward(self, x):
        for _ in range(3):
            x = x + 1
        return x


class ShapeBranch(nn.Module):
    def forward(self, x):
        if x.shape[0] > 2:
            return x + 1
        return x - 1


class PythonWhile(nn.Module):
    def forward(self, x, steps):
        i = torch.zeros((), dtype=torch.int64)
        while i < steps:
            x = x + 1
            i = i + 1
        return x


class StructuredCond(nn.Module):
    def forward(self, x):
        return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))


class StructuredWhile(nn.Module):
    def forward(self, x, steps):
        def condition(i, value):
            return i < steps

        def body(i, value):
            return i + 1, value + 1

        _, result = torch.while_loop(condition, body, (torch.zeros((), dtype=torch.int64), x))
        return result


def control_flow_checks():
    result = {}
    x = torch.ones(3, 8)
    fixed = torch.export.export(FixedLoop(), (x,), strict=True)
    torch.testing.assert_close(fixed.module()(x), x + 3)
    fixed_ops = operation_names(fixed)
    assert sum("aten.add" in op for op in fixed_ops) == 3
    result["fixed_loop"] = {"unrolled": True, "operations": fixed_ops}
    shape_ep = torch.export.export(
        ShapeBranch(),
        (x,),
        strict=True,
        dynamic_shapes=({0: torch.export.Dim("batch", min=3, max=8)},),
    )
    torch.testing.assert_close(shape_ep.module()(torch.ones(4, 8)), torch.ones(4, 8) + 1)
    result["shape_branch_restricted_domain"] = {
        "constraints": str(shape_ep.range_constraints),
        "operations": operation_names(shape_ep),
    }
    try:
        shape_ep.module()(torch.ones(2, 8))
    except (AssertionError, RuntimeError):
        result["shape_branch_rejects_other_domain"] = True
    else:
        raise AssertionError("Expected exported input domain to be enforced")
    for strict in (True, False):
        try:
            torch.export.export(PythonWhile(), (x, torch.tensor(3)), strict=strict)
        except Exception as exc:  # noqa: BLE001 - capture failure types are experiment results.
            result[f"python_while_strict_{strict}"] = type(exc).__name__
        else:
            raise AssertionError("Unexpected capture of an ordinary tensor-conditioned while")
    cond_ep = torch.export.export(StructuredCond(), (x,), strict=True)
    for sign in (1, -1):
        value = x * sign
        torch.testing.assert_close(cond_ep.module()(value), value + sign)
    assert any("cond" in op for op in operation_names(cond_ep))
    result["structured_cond"] = {
        "both_branches_match": True,
        "operations": operation_names(cond_ep),
        "graph_regions": [
            name
            for name, m in cond_ep.graph_module.named_modules()
            if isinstance(m, torch.fx.GraphModule)
        ],
    }
    loop_ep = torch.export.export(StructuredWhile(), (x, torch.tensor(3)), strict=True)
    for steps in (0, 2, 5):
        torch.testing.assert_close(loop_ep.module()(x, torch.tensor(steps)), x + steps)
    assert any("while_loop" in op for op in operation_names(loop_ep))
    result["structured_while"] = {
        "iteration_counts_checked": [0, 2, 5],
        "operations": operation_names(loop_ep),
        "graph_regions": [
            name
            for name, m in loop_ep.graph_module.named_modules()
            if isinstance(m, torch.fx.GraphModule)
        ],
    }
    return result


def operation_names(ep):
    return [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    report = {"torch": torch.__version__, "device": "cpu", "checks": {}}
    checks = report["checks"]
    x = torch.randn(2, 5, 8)
    model = FusedAttention().eval()
    batch = torch.export.Dim("batch", min=1, max=8)
    tokens = torch.export.Dim("tokens", min=1, max=16)
    ep = torch.export.export(model, (x,), dynamic_shapes=({0: batch, 1: tokens},), strict=True)
    torch.testing.assert_close(ep.module()(x), model(x))
    x2 = torch.randn(3, 7, 8)
    torch.testing.assert_close(ep.module()(x2), model(x2))
    ops = operation_names(ep)
    assert any("reshape" in op or "view" in op for op in ops)
    assert any("permute" in op for op in ops)
    checks["fused_attention"] = {
        "operations": ops,
        "shape_operations": [
            str(n) + ": " + str(n.args)
            for n in ep.graph.nodes
            if any(s in str(n.target) for s in ("reshape", "view", "permute", "unbind"))
        ],
        "parameter_targets": dict(ep.graph_signature.inputs_to_parameters),
        "range_constraints": str(ep.range_constraints),
        "two_input_shapes_match_eager": True,
        "module_stack_present": any("nn_module_stack" in n.meta for n in ep.graph.nodes),
        "empty_decomposition_operations": operation_names(ep.run_decompositions({})),
        "full_decomposition_operations": operation_names(ep.run_decompositions()),
    }
    compact = copy.deepcopy(model)
    kept_channels = torch.tensor([0, 1, 4, 5])  # Heads 0 and 2, each of width 2.
    packed_rows = torch.cat([kept_channels + offset for offset in (0, 8, 16)])
    compact.qkv.weight = nn.Parameter(compact.qkv.weight[packed_rows].clone())
    compact.qkv.bias = nn.Parameter(compact.qkv.bias[packed_rows].clone())
    compact.qkv.out_features = 12
    compact.proj.weight = nn.Parameter(compact.proj.weight[:, kept_channels].clone())
    compact.proj.in_features = 4
    try:
        compact(x)
    except RuntimeError:
        checks["weights_only_surgery_fails"] = True
    else:
        raise AssertionError("Expected stale reshape attributes to fail")
    compact.num_heads = 2
    compact.attn_dim = 4
    q, k, v = model.qkv(x).reshape(2, 5, 3, 4, 2).permute(2, 0, 3, 1, 4).unbind(0)
    heads = F.scaled_dot_product_attention(q, k, v)
    heads = heads * torch.tensor([1, 0, 1, 0]).reshape(1, 4, 1, 1)
    masked = model.proj(heads.transpose(1, 2).reshape(2, 5, 8))
    torch.testing.assert_close(compact(x), masked)
    compact_ep = torch.export.export(compact, (x,), strict=True)
    torch.testing.assert_close(compact_ep.module()(x), masked)
    checks["head_surgery_with_attribute_updates"] = {
        "masked_reference_matches": True,
        "recapture_matches": True,
        "max_abs_error": (compact(x) - masked).abs().max().item(),
    }
    shared = SharedWeight().eval()
    shared_ep = torch.export.export(shared, (x,), strict=True)
    torch.testing.assert_close(shared_ep.module()(x), shared(x))
    checks["shared_parameters"] = {
        "signature": dict(shared_ep.graph_signature.inputs_to_parameters),
        "source_aliases": [name for name, _ in shared.named_parameters(remove_duplicate=False)],
        "state_dict_alias_preserved": shared_ep.state_dict["a.weight"]
        is shared_ep.state_dict["b.weight"],
    }
    frozen = Detached().eval()
    with torch.no_grad():
        frozen_ep = torch.export.export(frozen, (x,), strict=True)
        torch.testing.assert_close(frozen_ep.module()(x), frozen(x))
    assert any("detach" in op for op in operation_names(frozen_ep))
    checks["frozen_and_detach"] = {
        "captures_under_no_grad": True,
        "operations": operation_names(frozen_ep),
    }
    for strict in (True, False):
        try:
            torch.export.export(DataBranch(), (x,), strict=strict)
        except Exception as exc:  # noqa: BLE001 - the diagnostic type is an experimental result.
            checks[f"data_branch_strict_{strict}"] = type(exc).__name__
        else:
            raise AssertionError("Unexpected export of unrestricted data-dependent branch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(DataBranch(), (torch.ones(2, 8),))
    negative = -torch.ones(2, 8)
    assert not torch.equal(traced(negative), DataBranch()(negative))
    checks["jit_trace_specializes_branch"] = True
    try:
        torch.fx.symbolic_trace(DataBranch())
    except torch.fx.proxy.TraceError:
        checks["symbolic_trace_rejects_data_branch"] = True
    layer = nn.Linear(8, 8)
    prune.ln_structured(layer, "weight", amount=0.5, n=2, dim=0)
    prune.remove(layer, "weight")
    assert layer.weight.shape == (8, 8) and layer.out_features == 8
    checks["prune_remove"] = {
        "weight_shape": list(layer.weight.shape),
        "out_features": layer.out_features,
    }
    checks["control_flow"] = control_flow_checks()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
