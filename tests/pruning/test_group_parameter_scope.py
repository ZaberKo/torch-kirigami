"""Group magnitude uses parameter effects without an operator whitelist."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from examples.fused_attention import FusedGQA, FusedGQARule
from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.dep_graph.contracts import Impact
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelRatio,
    DynamicGreedy,
    Greedy,
    GroupMagnitude,
    Magnitude,
    PlanningContext,
    PlanningError,
    Pruner,
    WeightTaylor,
)
from torch_kirigami.pruning.ranking import IndependentRanking
from torch_kirigami.pruning.types import MetricContext


class MixedAttention(nn.Module):
    """Keep an unrelated native layer beside an opaque attention module."""

    def __init__(self) -> None:
        super().__init__()
        self.fused = FusedGQA()
        self.other = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two independent branches."""
        return self.fused(x), self.other(x)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
def test_opaque_gqa_score_and_public_pruning(
    p: int, mixed: bool, strategy_type: type[Greedy], execution_device: str
) -> None:
    """Opaque parameter groups score correctly with or without native neighbors."""
    model = (MixedAttention() if mixed else FusedGQA()).double().eval()
    attention = model.fused if mixed else model
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(
                torch.arange(1, parameter.numel() + 1, dtype=torch.float64).reshape_as(parameter)
                / 100
            )
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    operators = OperatorRegistry.default().register(FusedGQA, FusedGQARule())
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("fused",) if mixed else None)
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.5), space.channel_axes, pruner.constraints
    )
    # Each KV group has four K/V rows, eight Q rows and eight output columns.
    # A singleton KV coordinate expands to its full group. Each group therefore
    # occurs four times in the normalization population of eight coordinates.
    energy = []
    for group in range(2):
        kv, query = slice(4 * group, 4 * group + 4), slice(8 * group, 8 * group + 8)
        values = torch.cat(
            (
                attention.k.weight[kv].flatten(),
                attention.v.weight[kv].flatten(),
                attention.q.weight[query].flatten(),
                attention.out.weight[:, query].flatten(),
            )
        )
        energy.append(values.abs().pow(p).sum())
    energies = torch.stack(energy)
    assert context.score(GroupMagnitude(p), space.candidates) == pytest.approx(
        (energies / energies.mean()).tolist(), rel=1e-12
    )
    assert energies[0] < energies[1]
    # Independent compact reference retains the second KV group.
    q = F.linear(x, attention.q.weight[8:]).reshape(2, 3, 2, 4).transpose(1, 2)
    k = F.linear(x, attention.k.weight[4:]).reshape(2, 3, 1, 4).transpose(1, 2)
    v = F.linear(x, attention.v.weight[4:]).reshape(2, 3, 1, 4).transpose(1, 2)
    attended = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    expected = F.linear(attended.transpose(1, 2).reshape(2, 3, 8), attention.out.weight[:, 8:])
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    plan = pruner.plan(space, budget=ChannelRatio(0.5), strategy=strategy_type(GroupMagnitude(p)))
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    compact, _ = pruner.apply(plan)
    assert compact is model
    torch.testing.assert_close(compact(x)[0] if mixed else compact(x), expected)
    assert (attention.q_heads, attention.kv_heads) == (2, 1)


class BufferBranch(nn.Module):
    """Expose a parameter-free branch and an unrelated parameterized branch."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(4))
        self.other = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, other_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep buffer-only effects independent of parameter effects."""
        return x * self.scale, self.other(other_x)


class UnitScore:
    """Use an explicit policy for changes that have no parameter magnitude."""

    def score(
        self,
        context: MetricContext,
        candidates: tuple[Candidate, ...],
        *,
        accepted_impact: Impact,
    ) -> list[float]:
        """Assign equal importance without claiming a parameter-based estimate."""
        return [1.0] * len(candidates)


@pytest.mark.parametrize("strategy_type", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("metric_type", [GroupMagnitude, Magnitude, WeightTaylor])
def test_parameter_free_scoring_fails_without_mutating_model(
    strategy_type: type[Greedy], metric_type: type, execution_device: str
) -> None:
    """An unrelated parameter or accepted request cannot provide scoring evidence."""
    model = BufferBranch()
    x = torch.ones(2, 4)
    graph = DependencyGraph.build(model, args=(x, x.clone()))
    axis = graph.buffer("scale").axis(0)
    candidate = Candidate("buffer_channel", (axis.select([0]),), axis)
    space = CandidateSpace((candidate,), (axis,))
    context = PlanningContext(graph, space.candidates, ChannelRatio(0.25), space.channel_axes, ())
    metric = metric_type()
    assert IndependentRanking.build(context, metric, space.channel_axes) is None
    for accepted in ((), (graph.parameter("other.weight").axis(0).select([0]),)):
        with pytest.raises(PlanningError, match="no affected parameters"):
            context.score(metric, space.candidates, accepted_impact=context.impact(accepted))
    original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    pruner = Pruner(model, graph=graph, preserve_io=False)
    with pytest.raises(PlanningError, match="no affected parameters"):
        pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(metric))
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
    graph.validate()
    # The metric rejected the scoring basis, not the graph or execution support.
    expected = model(x, x.clone())
    plan = pruner.plan(space, budget=ChannelRatio(0.25), strategy=strategy_type(UnitScore()))
    compact, _ = pruner.apply(plan)
    actual = compact(x[:, 1:], x.clone())
    torch.testing.assert_close(actual[0], expected[0][:, 1:])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("kind", ["embedding", "group_norm"])
def test_native_parameters_need_no_scoring_registration(
    p: int, kind: str, execution_device: str
) -> None:
    """Previously excluded operators use the same parameter-region formula."""
    if kind == "embedding":
        model = nn.Embedding(5, 4).double()
        x = torch.tensor([[0, 1, 2]])
        dimension = 1
    else:
        model = nn.GroupNorm(2, 4).double()
        x = torch.randn(2, 4, 3, dtype=torch.float64)
        dimension = 0
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, model.weight.numel() + 1).reshape_as(model.weight))
        if kind == "group_norm":
            model.bias.copy_(torch.arange(4, dtype=torch.float64))
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("weight").axis(dimension)
    candidates = tuple(Candidate(str(i), (axis.select([i]),), axis) for i in range(4))
    context = PlanningContext(graph, candidates, ChannelRatio(0.5), (axis,), ())
    energy = (
        model.weight.abs().pow(p).sum(0)
        if kind == "embedding"
        else model.weight.abs().pow(p) + model.bias.abs().pow(p)
    )
    assert context.score(GroupMagnitude(p), candidates) == pytest.approx(
        (energy / energy.mean()).tolist()
    )


@pytest.mark.parametrize("metric_type", [GroupMagnitude, Magnitude, WeightTaylor])
def test_metrics_have_no_parameter_filter(metric_type: type) -> None:
    """Scoring-scope customization belongs to a custom Metric."""
    with pytest.raises(TypeError, match="parameter_filter"):
        metric_type(parameter_filter=lambda ref, parameter: True)


@pytest.mark.parametrize("metric_type", [GroupMagnitude, Magnitude, WeightTaylor])
def test_zero_parameters_and_covered_requests_are_valid(
    metric_type: type, execution_device: str
) -> None:
    """Zero energy remains distinct from an absent parameter basis on both paths."""
    model = nn.Sequential(nn.Linear(2, 4), nn.Linear(4, 2)).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
            parameter.grad = torch.ones_like(parameter)
    graph = DependencyGraph.build(model, args=(torch.ones(2, 2, dtype=torch.float64),))
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates(targets=("0",))
    context = PlanningContext(
        graph, space.candidates, ChannelRatio(0.5), space.channel_axes, pruner.constraints
    )
    metric = metric_type()
    ranking = IndependentRanking.build(context, metric, space.channel_axes)
    accepted = context.impact(space.candidates[0].remove)
    assert context.score(metric, space.candidates, accepted_impact=accepted) == (0.0,) * 4
    if metric_type is GroupMagnitude:
        assert ranking is not None
        accelerated = ranking.score(accepted)
        assert accelerated is not None
        assert accelerated.covered == {space.candidates[0].key}
        assert tuple(accelerated.scores.values()) == (0.0,) * 3
    else:
        assert ranking is None
