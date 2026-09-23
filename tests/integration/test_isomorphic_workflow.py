"""Independent family quotas, coupling, rounding, and real model lifecycle checks."""

import copy
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import imagenet_models
import isomorphic_pruning as workflow
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import VisionTransformer

from torch_kirigami import AxisRef, DependencyGraph, Diagnostic
from torch_kirigami.pruning import (
    CandidateSpace,
    Granularity,
    ParameterBudget,
    PlanningError,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def signature(model, path, x, index=0):
    graph = DependencyGraph.build(model.eval(), args=(x,))
    impact = graph.propagate(remove=[graph.parameter(path).axis(0).select([index])])
    return workflow.dependency_signature(graph.operations(), impact)


def test_signature_ignores_names_widths_and_indices(execution_device):
    first = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    second = nn.Sequential(
        OrderedDict(renamed=nn.Linear(3, 7), activation=nn.ReLU(), downstream=nn.Linear(7, 2))
    )
    x = torch.randn(2, 3)
    assert signature(first, "0.weight", x) == signature(second, "renamed.weight", x, 5)
    second.activation = nn.GELU()
    assert signature(first, "0.weight", x) != signature(second, "renamed.weight", x, 5)


class Branched(nn.Module):
    def __init__(self, chain):
        super().__init__()
        self.producer = nn.Linear(3, 4)
        self.first, self.second = nn.ReLU(), nn.ReLU()
        self.consumer = nn.Linear(4, 2)
        self.chain = chain

    def forward(self, x):
        hidden = self.producer(x)
        first = self.first(hidden)
        second = self.second(first if self.chain else hidden)
        return self.consumer(first + second)


def test_signature_preserves_actual_edges(execution_device):
    x = torch.randn(2, 3)
    chain = signature(Branched(True), "producer.weight", x)
    fork = signature(Branched(False), "producer.weight", x)
    assert sorted(repr(node[0]) for node in chain) == sorted(repr(node[0]) for node in fork)
    assert chain != fork


class TwoFamilies(nn.Module):
    def __init__(self, widths=(4, 8)):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(3, widths[0]), nn.ReLU(), nn.Linear(widths[0], 2))
        self.b = nn.Sequential(nn.Linear(3, widths[1]), nn.GELU(), nn.Linear(widths[1], 2))

    def forward(self, x):
        return self.a(x) + self.b(x)


class ExplicitScores:
    def score(self, context, candidates, *, selected):
        values = []
        for candidate in candidates:
            index = next(iter(candidate.remove[0].fully_selected_indices(0)))
            scale = 100 if candidate.axis.tensor.paths[0].startswith("b.") else 1
            values.append(scale * (index + 1))
        return values


def build(model, x, granularity=1):
    graph = DependencyGraph.build(model.eval(), args=(x,))
    space = Pruner(model, graph=graph).discover_candidates()
    active = set(space.channel_axes)
    alignment = {
        operation.module_path: granularity
        for operation in graph.operations()
        if operation.module_path is not None
        and any(domain.axis in active for domain in graph.operator_spec(operation).candidates)
    }
    return graph, Pruner(model, graph=graph, granularity=Granularity(by_path=alignment)), space


def test_independent_family_quotas_are_not_global_early_stopping(execution_device, tmp_path):
    model = TwoFamilies().eval()
    original = copy.deepcopy(model)
    x = torch.randn(3, 3)
    graph, pruner, space = build(model, x)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    strategy = workflow.Isomorphic(ExplicitScores())
    plan = pruner.plan(space, budget=ParameterBudget(64), strategy=strategy)
    # 76 initial parameters; each action deletes six. A global algorithm can
    # stop after two actions at 64. Independent family quotas instead jump from
    # (0,1) at ratio 1/8 to (1,2) at 1/4 and must keep that full allocation.
    assert strategy.ratio == Fraction(1, 4)
    assert [
        (family["actions"], family["quota"], family["selected_actions"])
        for family in strategy.families
    ] == [(4, 1, 1), (8, 2, 2)]
    assert plan.selection_report.after_params == 58
    for path, count in (("a.0.weight", 1), ("b.0.weight", 2)):
        expected = tuple(range(count))
        assert (
            tuple(plan.analysis.selection(graph.parameter(path)).fully_selected_indices(0))
            == expected
        )
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    pruner.apply(plan)
    with torch.no_grad():
        original.a[2].weight[:, :1] = 0
        original.b[2].weight[:, :2] = 0
    torch.testing.assert_close(model(x), original(x))
    model(x).sum().backward()
    save_checkpoint(model, tmp_path / "model.pt")
    restored = load_checkpoint(
        TwoFamilies(), tmp_path / "model.pt", map_location=execution_device
    ).eval()
    torch.testing.assert_close(restored(x), model(x))


def test_reference_retained_width_rounding_adds_next_lowest_actions(execution_device):
    model = TwoFamilies((16, 16)).eval()
    graph, pruner, space = build(model, torch.randn(2, 3), granularity=8)
    strategy = workflow.Isomorphic(ExplicitScores())
    plan = pruner.plan(space, budget=ParameterBudget.from_ratio(model, 0.01), strategy=strategy)
    assert strategy.ratio == Fraction(1, 16)
    assert [family["quota"] for family in strategy.families] == [1, 1]
    # Reference: retain floor((16 - 1)/8)*8 = 8; independently retain the eight
    # highest-scoring channels in each affected layer.
    assert [family["selected_actions"] for family in strategy.families] == [8, 8]
    for path in ("a.0.weight", "b.0.weight"):
        assert tuple(
            plan.analysis.selection(graph.parameter(path)).fully_selected_indices(0)
        ) == tuple(range(8))
    pruner.apply(plan)
    assert model.a[0].out_features == model.b[0].out_features == 8


def test_ratio_breakpoints_do_not_lose_a_channel_to_float_roundoff(execution_device):
    model = TwoFamilies((50, 3)).eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    strategy = workflow.Isomorphic(ExplicitScores())
    # 322 parameters minus 30 actions * six parameters = 142. In binary
    # floating point, (29/50)*50 can be 28.999..., which must not produce 28.
    plan = pruner.plan(space, budget=ParameterBudget(142), strategy=strategy)
    assert strategy.ratio == Fraction(29, 50)
    assert [family["quota"] for family in strategy.families] == [29, 1]
    assert (
        len(plan.analysis.selection(graph.parameter("a.0.weight")).fully_selected_indices(0)) == 29
    )


def test_valid_noop_does_not_call_metric_but_invalid_empty_alignment_is_checked(execution_device):
    class NeverScore:
        def score(self, *args, **kwargs):
            raise AssertionError("No-op must not score the model")

    model = TwoFamilies().eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    plan = pruner.plan(
        space, budget=ParameterBudget(76), strategy=workflow.Isomorphic(NeverScore())
    )
    assert not plan.analysis.selections
    assert plan.selection_report.after_params == 76
    # Width six is initially incompatible with factor four; parameter count
    # already meeting the cap must not return the invalid empty request.
    other = TwoFamilies((6, 6)).eval()
    other_graph, other_pruner, other_space = build(other, torch.randn(2, 3), granularity=4)
    strategy = workflow.Isomorphic(ExplicitScores())
    fixed = other_pruner.plan(other_space, budget=ParameterBudget(1000), strategy=strategy)
    assert strategy.ratio == Fraction(1, 6)
    other_pruner.apply(fixed)
    assert other.a[0].out_features == other.b[0].out_features == 4
    graph.validate()
    assert other_graph is not graph


class CoupledRoots(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(3, 4), nn.Linear(3, 4)
        self.consumer = nn.Linear(4, 2)

    def forward(self, x):
        return self.consumer(self.a(x) + self.b(x))


def test_equivalent_residual_roots_count_one_action_and_preserve_quotas(execution_device):
    model = CoupledRoots().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3)
    graph, pruner, both = build(model, x)
    axis_a = graph.parameter("a.weight").axis(0)
    only_a = CandidateSpace(
        tuple(candidate for candidate in both.candidates if candidate.axis == axis_a), (axis_a,)
    )
    single, double = workflow.Isomorphic(), workflow.Isomorphic()
    first = pruner.plan(only_a, budget=ParameterBudget(32), strategy=single)
    second = pruner.plan(
        CandidateSpace(tuple(reversed(both.candidates)), both.channel_axes),
        budget=ParameterBudget(32),
        strategy=double,
    )
    assert single.ratio == double.ratio == Fraction(1, 4)
    assert (
        [family["actions"] for family in single.families]
        == [family["actions"] for family in double.families]
        == [4]
    )
    for path in ("a.weight", "b.weight", "consumer.weight"):
        assert first.analysis.selection(graph.parameter(path)) == second.analysis.selection(
            graph.parameter(path)
        )
    removed = tuple(
        second.analysis.selection(graph.parameter("a.weight")).fully_selected_indices(0)
    )
    pruner.apply(second)
    with torch.no_grad():
        original.consumer.weight[:, removed] = 0
    torch.testing.assert_close(model(x), original(x))


def test_trial_limit_preserves_graph_and_allows_valid_alternative(execution_device):
    model = TwoFamilies().eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(PlanningError, match="trial limit"):
        pruner.plan(
            space,
            budget=ParameterBudget(64),
            strategy=workflow.Isomorphic(ExplicitScores(), max_trials=1),
        )
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    graph.validate()
    pruner.apply(
        pruner.plan(
            space, budget=ParameterBudget(64), strategy=workflow.Isomorphic(ExplicitScores())
        )
    )
    assert model.a[0].out_features == 3 and model.b[0].out_features == 6


@dataclass(frozen=True)
class AvoidSingleRemoval:
    axis: AxisRef

    @property
    def refs(self):
        return (self.axis.tensor,)

    def check(self, selections):
        selection = selections.get(self.axis.tensor.id)
        if selection and len(selection.fully_selected_indices(self.axis.dim)) == 1:
            return Diagnostic(
                "temporary_quota",
                "Exactly one removal is disallowed",
                tensors=(self.axis.tensor.id,),
            )
        return None


def test_invalid_quota_does_not_shrink_family_or_stop_later_search(execution_device):
    model = TwoFamilies().eval()
    graph, _pruner, space = build(model, torch.randn(2, 3))
    constraint = AvoidSingleRemoval(graph.parameter("a.0.weight").axis(0))
    pruner = Pruner(model, graph=graph, constraints=(constraint,))
    strategy = workflow.Isomorphic(ExplicitScores())
    plan = pruner.plan(space, budget=ParameterBudget(64), strategy=strategy)
    # Ratios 1/4 and 3/8 are invalid, but 1/2 is legal. Individual candidates
    # must not be filtered merely because this repairable constraint rejects them.
    assert strategy.ratio == Fraction(1, 2)
    assert [(family["actions"], family["quota"]) for family in strategy.families] == [
        (4, 2),
        (8, 4),
    ]
    assert any("temporary_quota" in reason for reason in strategy.rejections)
    pruner.apply(plan)
    assert model.a[0].out_features == 2 and model.b[0].out_features == 4


def test_exhausted_alignment_reports_failure_without_mutating_model(execution_device):
    model = TwoFamilies((4, 4)).eval()
    graph, pruner, space = build(model, torch.randn(2, 3), granularity=4)
    with pytest.raises(PlanningError, match="entire structural axis"):
        pruner.plan(
            space, budget=ParameterBudget(40), strategy=workflow.Isomorphic(ExplicitScores())
        )
    graph.validate()
    assert model.a[0].out_features == model.b[0].out_features == 4


def miniature_model(name):
    if name.startswith("resnet"):
        return ResNet(BasicBlock, [1, 1, 1, 1])
    model = VisionTransformer(
        image_size=224,
        patch_size=32,
        num_layers=1,
        num_heads=2,
        hidden_dim=8,
        mlp_dim=16,
        num_classes=1000,
    )
    with torch.no_grad():
        model.heads.head.weight.normal_(std=0.02)
    return model


@pytest.mark.parametrize("name", ["resnet18", "vit_b_32"])
def test_entry_discovers_all_domains_evaluates_finetunes_and_restores(
    name, monkeypatch, tmp_path, execution_device
):
    weights = imagenet_models.MODELS[name][1]
    requested = []

    def builder(*, weights):
        requested.append(weights)
        return miniature_model(name)

    def data(*args, **kwargs):
        dataset = TensorDataset(
            torch.randn(2, 3, 224, 224, device="cpu"), torch.tensor([0, 1], device="cpu")
        )
        return dataset, dataset, {}

    monkeypatch.setitem(imagenet_models.MODELS, name, (builder, weights))
    monkeypatch.setattr(workflow, "load_images", data)
    monkeypatch.setattr(workflow, "measure_model", lambda *args: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "isomorphic_pruning.py",
            "--model",
            name,
            "--device",
            execution_device,
            "--pruning_ratio",
            "0.001",
            "--granularity",
            "2",
            "--finetune_epochs",
            "1",
            "--train_batch_size",
            "2",
            "--val_batch_size",
            "2",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--output",
            str(tmp_path),
        ],
    )
    with torch.device("cpu"):
        workflow.main()
    report = json.loads((tmp_path / "metrics.json").read_text())
    assert [item["stage"] for item in report["stages"]] == ["pretrained", "pruned", "finetuned"]
    assert report["stages"][1]["target_met"]
    if name.startswith("resnet"):
        # All discovered residual-width and internal-width groups participate;
        # no hard-coded conv1 target list can satisfy this assertion.
        assert len(report["stages"][1]["families"]) >= 2
        assert any(
            "conv2.weight" in path
            for paths in report["config"]["discovered_axes"]
            for path in paths
        )
    assert requested == [weights, None]


@pytest.mark.parametrize(
    "args",
    [
        ("--max_trials", "0"),
        ("--granularity", "0"),
        ("--lr", "nan"),
        ("--pruning_ratio", "1"),
        ("--train_workers", "-1"),
    ],
)
def test_cli_rejects_invalid_values(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isomorphic_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit):
        workflow.parse_args()
