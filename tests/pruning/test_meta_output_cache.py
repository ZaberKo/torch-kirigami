"""Metadata reuse preserves joint planning, coordinate checks and state boundaries."""

import copy
from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import CallEffects, DependencyGraph, OperatorRegistry, OperatorRule
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    ChannelRatio,
    DynamicGreedy,
    Granularity,
    Greedy,
    GroupMagnitude,
    ParameterBudget,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
    StrategyResult,
    validation,
)


def count_meta_modules(monkeypatch: pytest.MonkeyPatch) -> dict[type[nn.Module], int]:
    """Count actual native module evaluations without changing operator semantics."""
    counts: dict[type[nn.Module], int] = {}
    original = validation._meta_module

    def counted(*args: object, **kwargs: object) -> nn.Module:
        module = original(*args, **kwargs)
        counts[type(module)] = counts.get(type(module), 0) + 1
        return module

    monkeypatch.setattr(validation, "_meta_module", counted)
    return counts


@pytest.mark.parametrize("strategy", [Greedy, DynamicGreedy])
@pytest.mark.parametrize("parameter_budget", [False, True])
def test_public_plan_restore_apply_matches_uncached_validation(
    strategy: type[Greedy], parameter_budget: bool, execution_device: str
) -> None:
    """Independent native validation yields the same persisted plan and compact output."""
    torch.manual_seed(17)
    model = (
        nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Linear(3, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        .to(execution_device)
        .eval()
    )
    x = torch.randn(2, 3, device=execution_device)
    plans, outputs, states = [], [], []
    for cached in (False, True):
        current = copy.deepcopy(model)
        registry = OperatorRegistry.default()
        for rule in (
            *registry.modules.values(),
            *registry.functions.values(),
            *registry.methods.values(),
        ):
            rule.cache_meta_output = cached
        graph = DependencyGraph.build(current, args=(x,), operators=registry)
        pruner = Pruner(current, graph=graph, granularity=Granularity(by_path={"0": 2, "4": 2}))
        space = pruner.discover_candidates(targets=("0", "4"))
        budget = (
            ParameterBudget.from_ratio(current, 0.2) if parameter_budget else ChannelRatio(0.25)
        )
        plan = pruner.plan(space, budget=budget, strategy=strategy(GroupMagnitude()))
        plans.append(plan.to_dict())
        compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
        outputs.append(compact(x))
        states.append(compact.state_dict())
        compact(x).sum().backward()
    assert plans[0] == plans[1]
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_same_shape_requests_still_check_slice_coordinates(
    monkeypatch: pytest.MonkeyPatch, execution_device: str
) -> None:
    """Reusing a linear meta output cannot approve a different original-coordinate slice."""

    class Sliced(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(3, 6)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)[:, :3]

    model = Sliced().to(execution_device).eval()
    x = torch.randn(2, 3, device=execution_device)
    expected = F.linear(x, model.fc.weight[:3], model.fc.bias[:3]).detach()
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("fc.weight").axis(0)
    safe = Candidate("safe", (axis.select([5]),), axis)
    unsafe = Candidate("unsafe", (axis.select([1]),), axis)
    counts = count_meta_modules(monkeypatch)

    class CheckedStrategy:
        def select(self, context: PlanningContext) -> StrategyResult:
            context.compile(context.impact(safe.remove))
            before = counts[nn.Linear]
            with pytest.raises(PlanningError, match="slice"):
                context.compile(context.impact(unsafe.remove))
            assert counts[nn.Linear] == before
            return StrategyResult((safe.key,))

    before = model.fc.weight
    pruner = Pruner(model, graph=graph, preserve_io=False)
    plan = pruner.plan(
        CandidateSpace((safe, unsafe), (axis,)),
        budget=ChannelCount((5,), (axis,)),
        strategy=CheckedStrategy(),
    )
    assert model.fc.weight is before and model.fc.out_features == 6
    compact, _ = pruner.apply(PruningPlan.from_dict(plan.to_dict()))
    torch.testing.assert_close(compact(x), expected)


@pytest.mark.parametrize("custom", [False, True])
def test_unchanged_native_call_reuses_metadata_but_extensions_default_to_reexecution(
    custom: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CustomLinear(nn.Linear):
        """An explicitly declared extension whose inherited forward has linear semantics."""

    registry = OperatorRegistry.default()
    if custom:
        registry.register(
            CustomLinear,
            OperatorRule(
                registry.modules[nn.Linear].analyze,
                evaluate_on_meta=True,
                effects=lambda node, module: CallEffects(fresh_output=True),
            ),
        )
    layer: Callable[..., nn.Module] = CustomLinear if custom else nn.Linear
    model = nn.Sequential(layer(3, 8), nn.ReLU(inplace=True), nn.Linear(8, 3)).eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3),), operators=registry)
    axis = graph.parameter("0.weight").axis(0)
    context = PlanningContext(graph, (), ChannelRatio(0.5), (axis,), ())
    counts = count_meta_modules(monkeypatch)
    context.compile(context.impact((axis.select([0]),)))
    previous = counts[type(model[0])]
    context.compile(context.impact((axis.select([1]),)))
    assert counts[type(model[0])] == previous + int(custom)
    assert counts[nn.ReLU] == 2  # In-place calls execute even with unchanged metadata.
    # Training updates clear metadata/recipe proofs without invalidating structure.
    with torch.no_grad():
        model[0].weight.add_(1)
    context.compile(context.impact((axis.select([1]),)))
    assert counts[type(model[0])] == previous + int(custom) + (1 if custom else 2)


def test_meta_argument_keys_preserve_layout_and_container_semantics() -> None:
    """Equal shapes with different strides, offsets or argument containers cannot collide."""
    contiguous = torch.empty((2, 3), device="meta")
    transposed = torch.empty((3, 2), device="meta").T
    offset = torch.empty((3, 3), device="meta")[1:]
    assert len({validation._meta_key(t) for t in (contiguous, transposed, offset)}) == 3
    assert validation._meta_key([2, 3]) != validation._meta_key((2, 3))
    output = validation._freeze_meta_output(contiguous)
    first = validation._restore_meta_output(output)
    second = validation._restore_meta_output(output)
    first.transpose_(0, 1)
    assert second.shape == (2, 3) and second.stride() == (3, 1)


def test_cached_conv_output_keeps_unknown_layout_and_view_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restored meta strides cannot upgrade a backend-dependent layout into a view proof."""

    class FlattenView(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 1)
            self.last = nn.Linear(54, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = self.conv(x)
            return self.last(output.view(output.size(0), -1))

    model = FlattenView().eval()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 3, 3, 3),))
    axis = graph.parameter("conv.weight").axis(0)
    context = PlanningContext(graph, (), ChannelRatio(0.5), (axis,), ())
    counts = count_meta_modules(monkeypatch)
    for position in (0, 1):
        with pytest.raises(PlanningError, match=r"stride.*reshape.*contiguous"):
            context.compile(context.impact((axis.select([position]),)))
    assert counts[nn.Conv2d] == 1


def test_tensor_subclasses_do_not_key_or_populate_native_metadata_cache() -> None:
    """Extra dispatch behavior cannot be inferred from ordinary shape and stride facts."""

    class MetadataSubclass(torch.Tensor):
        """An otherwise ordinary tensor with a distinct dispatch type."""

    value = torch.empty((2, 3), device="meta").as_subclass(MetadataSubclass)
    with pytest.raises(TypeError, match="metadata tensors"):
        validation._meta_key(value)
    with pytest.raises(TypeError, match="ordinary tensor metadata"):
        validation._freeze_meta_output(value)
