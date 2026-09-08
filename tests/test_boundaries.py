import pytest
import torch
from torch import nn

from torch_kirigami import (
    CaptureError,
    DependencyGraph,
    Divisible,
    IndexSet,
    Region,
    Selection,
    TensorRef,
)


@pytest.mark.parametrize("kind", ["contiguous", "function", "module"])
def test_all_known_parameter_alias_mutations_rejected(kind):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            if kind == "contiguous":
                self.weight.contiguous().zero_()
            elif kind == "function":
                torch.relu_(self.weight)
            else:
                self.relu(self.weight)
            return x

    model = Model()
    original = model.weight.detach().clone()
    with pytest.raises(CaptureError, match="write"):
        DependencyGraph.build(model, args=(torch.randn(2, 4),))
    torch.testing.assert_close(model.weight, original)


def test_parameter_clone_is_safe_to_mutate():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 4))

        def forward(self, x):
            temporary = self.weight.clone()
            temporary.relu_()
            return x @ temporary

    model = Model()
    original = model.weight.detach().clone()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    torch.testing.assert_close(model.weight, original)
    assert (
        graph.propagate(remove=[graph.parameter("weight").axis(1).select([1])]).status == "resolved"
    )


def test_divisibility_reports_choices_not_automatic_completion():
    graph = DependencyGraph.build(nn.Linear(4, 8), args=(torch.randn(2, 4),))
    axis = graph.parameter("weight").axis(0)
    impact = graph.propagate(remove=[axis.select([0])], constraints=[Divisible(axis, 4)])
    assert impact.status == "unresolved"
    assert set(impact.selection(axis.tensor).project(0)) == {0}
    assert (
        graph.propagate(remove=[axis.select([0, 1, 2, 3])], constraints=[Divisible(axis, 4)]).status
        == "resolved"
    )


def test_selection_equality_is_geometric():
    ref = TensorRef("r", (3, 3))
    a = ref.axis(0).select([0]).union(ref.axis(1).select([0]))
    b = Selection(
        ref,
        (
            Region((IndexSet.span(0, 3), IndexSet.of([0]))),
            Region((IndexSet.of([0]), IndexSet.span(1, 3))),
        ),
    )
    assert a == b
    assert not a.subtract(b) and not b.subtract(a)
    with pytest.raises(TypeError):
        hash(a)


def test_metadata_and_capture_context_are_queryable():
    model = nn.Linear(4, 4).double()
    with torch.no_grad():
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4, dtype=torch.float64),))
    facts = graph.metadata(graph.parameter("weight"))
    assert facts.dtype == torch.float64 and facts.stride == (4, 1)
    assert not graph.context["grad_enabled"]
    assert graph.context["torch_version"] == torch.__version__
    assert graph.context["rule_snapshot"]


def test_softmax_reports_reduced_domain():
    graph = DependencyGraph.build(nn.Softmax(dim=1), args=(torch.randn(2, 4),))
    call = graph.calls("")[0]
    impact = graph.propagate(remove=[call.input().axis(1).select([1])])
    assert impact.status == "resolved"
    assert any(r.kind == "reduction_domain" for r in impact.requirements)


def test_parameter_kernel_axis_is_explicitly_unsupported():
    graph = DependencyGraph.build(nn.Conv2d(4, 6, 3), args=(torch.randn(2, 4, 7, 7),))
    impact = graph.propagate(remove=[graph.parameter("weight").axis(2).select([1])])
    assert impact.status == "unresolved"
    assert any(d.code == "unsupported_axis" for d in impact.diagnostics)


def test_partial_parameter_region_is_not_claimed_compact():
    graph = DependencyGraph.build(nn.Linear(4, 4), args=(torch.randn(2, 4),))
    weight = graph.parameter("weight")
    impact = graph.propagate(remove=[weight.select([Region((IndexSet.of([1]), IndexSet.of([2])))])])
    assert impact.status == "unresolved"


def test_analysis_limit_remains_explicit():
    class Model(nn.Module):
        def forward(self, x):
            return x.flatten()

    graph = DependencyGraph.build(Model(), args=(torch.randn(5000, 2),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([0])])
    assert impact.status == "unresolved"
    assert any(d.code == "analysis_limit" for d in impact.diagnostics)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_cuda_rng_and_buffers_are_restored():
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout(0.5)).cuda().train()
    x = torch.randn(3, 4, device="cuda")
    before = torch.cuda.get_rng_state().clone()
    original = model[0].running_mean
    DependencyGraph.build(model, args=(x,))
    assert torch.equal(torch.cuda.get_rng_state(), before)
    assert model[0].running_mean is original
    assert model[0].num_batches_tracked.item() == 0


def test_shared_convolution_and_linear_require_both_layouts():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, image, vector):
            return F.conv1d(image, self.weight, groups=2), F.linear(vector, self.weight.squeeze(-1))

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 3)))
    input_ = next(v for v in graph.values() if v.kind == "input" and len(v.shape) == 3)
    result = graph.propagate(remove=[input_.axis(1).select([0, 4])])
    assert result.status == "unresolved"
    assert any(d.code == "unsupported_layout" for d in result.diagnostics)


def test_shared_weight_different_groupings_are_intersected():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, a, b):
            return F.conv1d(a, self.weight, groups=2), F.conv1d(b, self.weight, groups=4)

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 12, 5)))
    input_ = next(v for v in graph.values() if v.kind == "input" and v.shape[1] == 12)
    invalid = graph.propagate(remove=[input_.axis(1).select([0, 4, 6, 10])])
    assert invalid.status == "unresolved"
    valid = graph.propagate(remove=[input_.axis(1).select([0, 3, 7, 10])])
    assert valid.status == "resolved"


def test_broadcast_partial_regions_match_dense_reference():
    from torch_kirigami import BroadcastRelation

    small = TensorRef("small", (1, 3, 4))
    big = TensorRef("big", (2, 3, 4))
    chosen = Selection(small, (Region((IndexSet.of([0]), IndexSet.of([1]), IndexSet.of([2]))),))
    relation = BroadcastRelation(small, big)
    forward = relation.propagate(chosen)[0]
    assert forward.count == 2
    assert relation.propagate(forward)[0] == chosen
    # Removing only one broadcast copy cannot remove the shared source position.
    partial = Selection(big, (Region((IndexSet.of([0]), IndexSet.of([1]), IndexSet.of([2]))),))
    assert not relation.propagate(partial)[0]


def test_normalization_cannot_bypass_shared_weight_layout():
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 3, 1))

        def forward(self, image, normalized):
            return F.conv1d(image, self.weight, groups=2), F.layer_norm(
                normalized, (4, 3, 1), weight=self.weight
            )

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 6, 5), torch.randn(2, 4, 3, 1)))
    input_ = next(v for v in graph.values() if v.kind == "input" and len(v.shape) == 3)
    impact = graph.propagate(remove=[input_.axis(1).select([0, 4])])
    assert impact.status == "unresolved"
    assert any(
        d.code == "unsupported_layout" and d.node == "layer_norm" for d in impact.diagnostics
    )


def test_squeeze_requires_rank_preserving_rewrite_when_axis_becomes_one():
    class Model(nn.Module):
        def forward(self, x):
            return x.squeeze()

    graph = DependencyGraph.build(Model(), args=(torch.randn(2, 3),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([0, 1])])
    assert impact.status == "resolved"
    assert any(r.kind == "dimension_transform" for r in impact.requirements)


def test_unflatten_has_explicit_attribute_binding():
    graph = DependencyGraph.build(nn.Unflatten(1, (4, 2)), args=(torch.randn(2, 8),))
    input_ = next(v for v in graph.values() if v.kind == "input")
    impact = graph.propagate(remove=[input_.axis(1).select([2, 3])])
    assert impact.status == "resolved"
    req = next(r for r in impact.requirements if r.kind == "attribute")
    assert req.target == "unflattened_size"
    assert tuple(a.dim for a in dict(req.data)["axes"]) == (1, 2)


def test_reshape_keeps_large_token_prefix_symbolic():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 24)

        def forward(self, x):
            return self.proj(x).reshape(x.shape[0], x.shape[1], 3, 4, 2)

    graph = DependencyGraph.build(Model(), args=(torch.randn(1, 5000, 8),))
    call = next(c for c in graph.calls() if c.name == "reshape")
    impact = graph.propagate(remove=[call.output().axis(3).select([1])])
    assert impact.status == "resolved"
    assert set(impact.selection(graph.parameter("proj.weight")).project(0)) == {
        2,
        3,
        10,
        11,
        18,
        19,
    }


def test_physical_axis_constraints_do_not_guess_partitioned_sizes():
    from torch_kirigami import Fixed

    graph = DependencyGraph.build(nn.Conv1d(12, 4, 1, groups=2), args=(torch.randn(2, 12, 5),))
    request = graph.calls("")[0].input().axis(1).select([0, 1, 2, 9, 10, 11])
    axis = graph.parameter("weight").axis(1)
    assert graph.propagate(remove=[request]).status == "resolved"
    for constraint in (Fixed(axis), Divisible(axis, 2)):
        impact = graph.propagate(remove=[request], constraints=[constraint])
        assert impact.status == "unresolved"
        assert any(d.code == "partitioned_constraint" for d in impact.diagnostics)


def test_depthwise_partial_outputs_do_not_choose_input_removal():
    model = nn.Conv1d(4, 8, 1, groups=4)
    x = torch.randn(2, 4, 5)
    graph = DependencyGraph.build(model, args=(x,))
    root = graph.parameter("weight").axis(0)
    input_ = graph.calls("")[0].input()
    partial = graph.propagate(remove=[root.select([2])])
    assert partial.status == "unresolved"
    assert not partial.selection(input_)
    assert set(partial.selection(root.tensor).project(0)) == {2}
    balanced = graph.propagate(remove=[root.select([0, 2, 4, 6])])
    assert balanced.status == "resolved"
    assert not balanced.selection(input_)
    compact = nn.Conv1d(4, 4, 1, groups=4)
    keep = [1, 3, 5, 7]
    with torch.no_grad():
        compact.weight.copy_(model.weight[keep])
        compact.bias.copy_(model.bias[keep])
    torch.testing.assert_close(compact(x), model(x)[:, keep])
    whole = graph.propagate(remove=[root.select([2, 3])])
    assert whole.status == "resolved"
    assert set(whole.selection(input_).project(1)) == {1}


def test_successful_metadata_execution_releases_intermediate_activations():
    import gc
    import weakref

    references = []
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    handles = [
        module.register_forward_hook(
            lambda module, args, output: references.append(weakref.ref(output))
        )
        for module in model
    ]
    try:
        graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
        gc.collect()
        assert references and all(ref() is None for ref in references)
        assert (
            graph.propagate(remove=[graph.parameter("0.weight").axis(0).select([1])]).status
            == "resolved"
        )
    finally:
        for handle in handles:
            handle.remove()
