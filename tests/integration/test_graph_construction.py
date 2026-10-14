"""Graph construction isolates analysis state and publishes complete snapshots."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    CaptureError,
    DependencyGraph,
    Fixed,
    OperationContext,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    StaleGraphError,
    TensorRef,
)
from torch_kirigami.dep_graph.construction import _Analyzer
from torch_kirigami.pruning import PlanningError, Pruner


class SharedProjection(nn.Module):
    """Call one hidden projection twice through different registered aliases."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 8)
        self.alias = self.hidden
        self.output = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output((self.hidden(x) + self.alias(x)).relu())


def test_graphs_keep_independent_rules_queries_and_invalidation(execution_device: str) -> None:
    """A rejected graph must not change another graph's rules, cache, or plan."""
    model = SharedProjection().to(execution_device).eval()
    x = torch.randn(2, 4, device=execution_device)
    registry = OperatorRegistry.default()
    native = registry.modules[nn.Linear]

    def protect_hidden(context: OperationContext) -> OperatorSpec:
        spec = native.analyze(context)
        if context.module is model.hidden:
            spec = replace(
                spec,
                constraints=(*spec.constraints, Fixed(context.binding("weight").axis(0))),
            )
        return spec

    registry.modules[nn.Linear] = OperatorRule(protect_hidden)
    protected = DependencyGraph.build(model, args=(x,), operators=registry)
    # A later table edit applies to future builds, including on the same model.
    registry.modules[nn.Linear] = native
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    protected_seed = protected.parameter("hidden.weight").axis(0).select([1, 5])
    seed = graph.parameter("hidden.weight").axis(0).select([1, 5])
    original_parameters = tuple(model.parameters())
    original_output = model(x).detach().clone()
    for _ in range(2):
        assert protected.propagate(remove=[protected_seed]).status == "conflict"
        assert graph.propagate(remove=[seed]).status == "resolved"
    with pytest.raises(PlanningError):
        Pruner(model, graph=protected).plan_remove([protected_seed])
    with pytest.raises(ValueError, match="another graph"):
        graph.propagate(remove=[protected_seed])
    protected.validate()
    graph.validate()
    assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), original_output, rtol=0, atol=0)
    assert graph.parameter("alias.weight") is graph.parameter("hidden.weight")
    assert len(graph.calls("alias")) == 2

    kept = [0, 2, 3, 4, 6, 7]
    expected = F.linear(
        (2 * F.linear(x, model.hidden.weight[kept], model.hidden.bias[kept])).relu(),
        model.output.weight[:, kept],
        model.output.bias,
    )
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([seed]))
    assert model.alias is model.hidden
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    for snapshot in (protected, graph):
        with pytest.raises(StaleGraphError):
            snapshot.propagate(remove=[])
    rebuilt = DependencyGraph.build(model, args=(x,))
    assert rebuilt.parameter("hidden.weight").shape == (6, 4)
    assert rebuilt.propagate(remove=[]).status == "resolved"


def test_failed_analysis_does_not_publish_or_contaminate_a_graph(execution_device: str) -> None:
    """Reject invalid extension records and allow a subsequent valid build and prune."""
    with pytest.raises(TypeError):
        DependencyGraph()
    model = SharedProjection().to(execution_device).eval()
    x = torch.randn(2, 4, device=execution_device)
    original_parameters = tuple(model.parameters())
    original_output = model(x).detach().clone()
    graph = DependencyGraph.build(model, args=(x,))
    seed = graph.parameter("hidden.weight").axis(0).select([0, 7])
    expected_impact = graph.propagate(remove=[seed])
    registry = OperatorRegistry.default()
    native = registry.modules[nn.Linear]

    def invalid_reference(context: OperationContext) -> OperatorSpec:
        spec = native.analyze(context)
        if context.module is model.output:
            foreign = TensorRef("foreign", (2, 8))
            return replace(spec, constraints=(*spec.constraints, Fixed(foreign.axis(0))))
        return spec

    registry.modules[nn.Linear] = OperatorRule(invalid_reference)
    with pytest.raises(ValueError, match="another graph"):
        DependencyGraph.build(model, args=(x,), operators=registry)
    assert graph.propagate(remove=[seed]) == expected_impact
    assert all(a is b for a, b in zip(original_parameters, model.parameters(), strict=True))
    torch.testing.assert_close(model(x), original_output, rtol=0, atol=0)

    rebuilt = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=rebuilt)
    plan = pruner.plan_remove([rebuilt.parameter("hidden.weight").axis(0).select([0, 7])])
    kept = list(range(1, 7))
    expected = F.linear(
        (2 * F.linear(x, model.hidden.weight[kept], model.hidden.bias[kept])).relu(),
        model.output.weight[:, kept],
        model.output.bias,
    )
    pruner.apply(plan)
    torch.testing.assert_close(model(x), expected)


def test_analyzer_owns_one_record_and_releases_scratch():
    model = SharedProjection()
    analyzer = _Analyzer(model, None)
    data = analyzer.data
    with pytest.raises(ValueError, match="complete"):
        DependencyGraph(data)
    assert analyzer.build((torch.randn(2, 4),), {}) is data
    assert data.complete
    assert not analyzer._values and not analyzer._by_object
    assert not analyzer._unused and not analyzer._module_paths
    assert not hasattr(analyzer, "refs") and not hasattr(analyzer, "relations")
    graph = DependencyGraph(data)
    for container in (graph.relations, graph.constraints, graph.diagnostics, graph.calls()):
        assert isinstance(container, tuple)
    with pytest.raises(TypeError):
        graph.context["grad_enabled"] = False
    for context in graph.operations():
        with pytest.raises(TypeError):
            context.expressions[context.node] = None
    before = graph.propagate(remove=[])
    with pytest.raises(RuntimeError, match="only build once"):
        analyzer.build((torch.randn(2, 4),), {})
    assert graph.propagate(remove=[]) == before


def test_failed_analyzer_is_unpublished_and_clears_scratch():
    class Dynamic(nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x

    analyzer = _Analyzer(Dynamic(), None)
    with pytest.raises(CaptureError):
        analyzer.build((torch.ones(2),), {})
    assert not analyzer.data.complete
    assert not analyzer._values and not analyzer._by_object
    assert not analyzer._unused and not analyzer._module_paths
    with pytest.raises(ValueError, match="complete"):
        DependencyGraph(analyzer.data)
    with pytest.raises(RuntimeError, match="only build once"):
        analyzer.build((torch.ones(2),), {})
