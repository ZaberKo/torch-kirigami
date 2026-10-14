"""Global settings cover construction, cached queries, and portable execution."""

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from torch_kirigami import (
    AnalysisLimitError,
    DependencyGraph,
    IndexSet,
    Requirement,
    StaleGraphError,
    config,
)
from torch_kirigami.configuration import freeze
from torch_kirigami.pruning import (
    ChannelRatio,
    ExecutionError,
    IdentityAxisIndex,
    PlanningContext,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.pruning.serialization import decode
from torch_kirigami.pruning.types import ModuleState


def test_global_settings_are_visible_to_other_threads():
    config.update(max_parts=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(lambda: config.max_parts).result() == 2
    with pytest.raises(AnalysisLimitError):
        IndexSet.of([0, 2, 4])
    config.reset()
    assert tuple(IndexSet.of([0, 2, 4])) == (0, 2, 4)


def test_nested_data_limits_cover_all_existing_consumers():
    nested = (((1,),),)
    encoded = {"tuple": [{"tuple": [{"tuple": [1]}]}]}
    config.update(max_container_depth=2, max_record_depth=2)
    with pytest.raises(TypeError, match="deeply nested"):
        freeze(nested)
    with pytest.raises(ValueError, match="deeply nested"):
        Requirement("test", "node", (), "test", (("data", nested),))
    with pytest.raises(ValueError, match="nesting limit"):
        decode(encoded)
    with pytest.raises(ValueError, match="nesting limit"):
        ModuleState(("",), "Example", (("data", nested),), ())
    config.update(max_container_depth=8, max_record_depth=8)
    assert freeze(nested)
    assert Requirement("test", "node", (), "test", (("data", nested),))
    assert decode(encoded) == nested
    assert ModuleState(("",), "Example", (("data", nested),), ())


def test_cached_analysis_rejects_changed_analysis_settings(execution_device):
    model = nn.Sequential(nn.Linear(3, 6), nn.Linear(6, 2)).to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    config.update(max_parts=8)
    graph = DependencyGraph.build(model, args=(x,))
    axis = graph.parameter("0.weight").axis(0)
    context = PlanningContext(graph, (), ChannelRatio(0.5), (axis,), ())
    request = (axis.select([1]),)
    impact = context.impact(request)
    recipes = context.compile(impact)
    config.update(max_parts=4)
    for operation in (
        graph.validate,
        lambda: graph.propagate(remove=request),
        lambda: context.impact(request),
        lambda: context.compile(impact),
    ):
        with pytest.raises(StaleGraphError, match="Analysis configuration changed"):
            operation()
    config.update(max_parts=8)
    assert context.impact(request) is impact
    assert context.compile(impact) == recipes
    index = IdentityAxisIndex(graph.relations)
    config.update(max_parts=4)
    assert index.component(axis) is None
    config.update(max_parts=8)
    assert index.component(axis) is not None
    config.update(max_parts=4)
    assert index.component(axis) is None


@pytest.mark.parametrize("kind", ["literal", "shuffle"])
def test_build_limits_control_support_without_mutation(kind, execution_device, tmp_path):

    class Indexed(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.last = nn.Linear(4, 2)
            self.register_buffer("indices", torch.tensor([0, 1, 0, 1]))

        def forward(self, x):
            return self.last(self.first(x).index_select(0, self.indices))

    if kind == "literal":
        model = Indexed().to(execution_device)
        x = torch.randn(2, 3, device=execution_device)
        parameter, positions, consumer_positions = ("first.weight", [1], [1])
        output_parameter = "last.weight"
        option = "max_literal_elements"
    else:
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ChannelShuffle(2), nn.Conv2d(4, 2, 1))
        model = model.to(execution_device)
        x = torch.randn(2, 3, 2, 2, device=execution_device)
        parameter, positions, consumer_positions = ("0.weight", [0, 2], [0, 1])
        output_parameter = "2.weight"
        option = "max_shuffle_channels"
    reference = copy.deepcopy(model)
    original_parameters = tuple(model.parameters())
    config.update(**{option: 3})
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph, preserve_io=False)
    with pytest.raises(PlanningError):
        pruner.plan_remove([graph.parameter(parameter).axis(0).select(positions)])
    assert all((a is b for a, b in zip(original_parameters, model.parameters(), strict=True)))
    assert pruner.plan_remove([graph.parameter(output_parameter).axis(0).select([0])])
    graph.validate()
    config.update(**{option: 4})
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter(parameter).axis(0).select(positions)])
    path = tmp_path / "plan.pt"
    torch.save(plan.to_dict(), path)
    restored = PruningPlan.from_dict(torch.load(path, weights_only=True))
    compact, _ = pruner.apply(restored)
    with torch.no_grad():
        dict(reference.named_parameters())[output_parameter][:, consumer_positions] = 0
    torch.testing.assert_close(compact(x), reference(x))
    compact(x).sum().backward()
    assert all(p.grad is not None for p in compact.parameters())
    checkpoint = tmp_path / "compact.pt"
    save_checkpoint(compact, checkpoint)
    rebuilt = copy.deepcopy(reference)
    load_checkpoint(rebuilt, checkpoint)
    torch.testing.assert_close(rebuilt(x), compact(x))


def test_restore_enforces_current_region_limit_without_mutation(tmp_path):
    model = nn.Sequential(nn.Linear(3, 8), nn.Linear(8, 2))
    x = torch.randn(2, 3)
    reference = copy.deepcopy(model)
    config.update(max_parts=8)
    graph = DependencyGraph.build(model, args=(x,))
    plan = Pruner(model, graph=graph).plan_remove(
        [graph.parameter("0.weight").axis(0).select([1, 3, 5])]
    )
    path = tmp_path / "plan.pt"
    torch.save(plan.to_dict(), path)
    parameters = tuple(model.parameters())
    config.update(max_parts=2)
    with pytest.raises(AnalysisLimitError):
        PruningPlan.from_dict(torch.load(path, weights_only=True))
    assert all((a is b for a, b in zip(parameters, model.parameters(), strict=True)))
    torch.testing.assert_close(model(x), reference(x))
    config.update(max_parts=8)
    restored = PruningPlan.from_dict(torch.load(path, weights_only=True))
    compact, _ = Pruner(model).apply(restored)
    with torch.no_grad():
        reference[1].weight[:, [1, 3, 5]] = 0
    torch.testing.assert_close(compact(x), reference(x))


def test_checkpoint_extra_state_uses_container_limit(tmp_path):

    class ExtraState(nn.Linear):
        def get_extra_state(self):
            return {"payload": (((1,),),)}

        def set_extra_state(self, state):
            assert state == self.get_extra_state()

    source = ExtraState(2, 3)
    path = tmp_path / "extra.pt"
    config.update(max_container_depth=2)
    with pytest.raises(ExecutionError, match="deeply nested"):
        save_checkpoint(source, path)
    assert not path.exists()
    config.update(max_container_depth=8)
    save_checkpoint(source, path)
    target = ExtraState(2, 3)
    original = tuple(target.parameters())
    config.update(max_container_depth=2)
    with pytest.raises(ExecutionError, match="deeply nested"):
        load_checkpoint(target, path)
    assert all((a is b for a, b in zip(original, target.parameters(), strict=True)))
    config.update(max_container_depth=8)
    load_checkpoint(target, path)
    x = torch.randn(1, 2)
    torch.testing.assert_close(target(x), source(x))
