"""Independent family quotas, coupling, rounding, and real model lifecycle checks."""

import copy
import json
import sys
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

pytest.importorskip("torchvision")
pytest.importorskip("datasets")

import imagenet_models
import isomorphic_pruning as workflow
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models.vision_transformer import VisionTransformer

from torch_kirigami import AxisRef, DependencyGraph, Diagnostic
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    Granularity,
    Magnitude,
    ParameterBudget,
    PlanningContext,
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
    def score(self, context, candidates, *, accepted_impact):
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
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=0.25)
    plan = pruner.plan(space, budget=ParameterBudget(64), strategy=strategy)
    # 76 initial parameters; each action deletes six. A global algorithm can
    # stop after two actions at 64. The supplied family ratio 1/4 requests
    # (1,2) and must keep that complete allocation, despite exceeding the cap.
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
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=Fraction(1, 16))
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


def test_fixed_ratio_does_not_lose_a_channel_to_float_roundoff(execution_device):
    model = TwoFamilies((50, 3)).eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=Fraction(29, 50))
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
        space, budget=ParameterBudget(76), strategy=workflow.Isomorphic(NeverScore(), ratio=0)
    )
    assert not plan.analysis.selections
    assert plan.selection_report.after_params == 76
    # Width six is initially incompatible with factor four; parameter count
    # already meeting the cap must not return the invalid empty request.
    other = TwoFamilies((6, 6)).eval()
    other_graph, other_pruner, other_space = build(other, torch.randn(2, 3), granularity=4)
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=Fraction(1, 6))
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
    single, double = (
        workflow.Isomorphic(Magnitude(), ratio=0.25),
        workflow.Isomorphic(Magnitude(), ratio=0.25),
    )
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


def test_fixed_quota_failure_preserves_graph_and_allows_valid_alternative(execution_device):
    model = TwoFamilies().eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    original = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(PlanningError, match="target not reached"):
        pruner.plan(
            space,
            budget=ParameterBudget(64),
            strategy=workflow.Isomorphic(ExplicitScores(), ratio=0.125),
        )
    assert all(torch.equal(original[key], value) for key, value in model.state_dict().items())
    graph.validate()
    pruner.apply(
        pruner.plan(
            space,
            budget=ParameterBudget(64),
            strategy=workflow.Isomorphic(ExplicitScores(), ratio=0.25),
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


def test_invalid_quota_is_not_silently_replaced_by_a_larger_ratio(execution_device):
    model = TwoFamilies().eval()
    graph, _pruner, space = build(model, torch.randn(2, 3))
    constraint = AvoidSingleRemoval(graph.parameter("a.0.weight").axis(0))
    pruner = Pruner(model, graph=graph, constraints=(constraint,))
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=0.25)
    with pytest.raises(PlanningError, match="temporary_quota"):
        pruner.plan(space, budget=ParameterBudget(64), strategy=strategy)
    graph.validate()
    assert model.a[0].out_features == 4 and model.b[0].out_features == 8
    strategy = workflow.Isomorphic(ExplicitScores(), ratio=0.5)
    plan = pruner.plan(space, budget=ParameterBudget(64), strategy=strategy)
    assert [(family["actions"], family["quota"]) for family in strategy.families] == [
        (4, 2),
        (8, 4),
    ]
    pruner.apply(plan)
    assert model.a[0].out_features == 2 and model.b[0].out_features == 4


def test_exhausted_alignment_reports_failure_without_mutating_model(execution_device):
    model = TwoFamilies((4, 4)).eval()
    graph, pruner, space = build(model, torch.randn(2, 3), granularity=4)
    with pytest.raises(PlanningError, match="entire structural axis"):
        pruner.plan(
            space,
            budget=ParameterBudget(40),
            strategy=workflow.Isomorphic(ExplicitScores(), ratio=0.25),
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
            "--family_pruning_ratio",
            "0.1",
            "--calibration_batches",
            "1",
            "--calibration_batch_size",
            "2",
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
        ("--calibration_batches", "0"),
        ("--calibration_batch_size", "0"),
        ("--granularity", "0"),
        ("--lr", "nan"),
        ("--family_pruning_ratio", "1"),
        ("--train_workers", "-1"),
    ],
)
def test_cli_rejects_invalid_values(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isomorphic_pruning.py", "--device", "cpu", *args])
    with pytest.raises(SystemExit):
        workflow.parse_args()


def test_calibration_matches_accumulated_task_gradients_and_preserves_state(execution_device):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Dropout(), nn.Linear(4, 2))
    model.train()
    model[1].eval()
    reference = copy.deepcopy(model).eval()
    modes = [module.training for module in model.modules()]
    before = {key: value.clone() for key, value in model.state_dict().items()}
    x, y = torch.randn(6, 3), torch.tensor([0, 1, 0, 1, 1, 0])
    for start in (0, 2):
        F.cross_entropy(reference(x[start : start + 2]), y[start : start + 2]).backward()
    loader = DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=2)
    with torch.no_grad():
        report = workflow.collect_task_gradients(model, loader, execution_device, 2)
    assert report == {"batches": 2, "samples": 4}
    assert [module.training for module in model.modules()] == modes
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="only 3 are available"):
        workflow.collect_task_gradients(model, loader, execution_device, 4)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert [module.training for module in model.modules()] == modes


def test_paper_taylor_formula_aliases_joint_regions_and_public_plan(execution_device, tmp_path):
    model = CoupledRoots().eval()
    model.b.weight = model.a.weight
    original = copy.deepcopy(model)
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 1, 0])
    loader = DataLoader(TensorDataset(x.cpu(), y.cpu()), batch_size=2)
    workflow.collect_task_gradients(model, loader, execution_device, 2)
    graph, pruner, space = build(model, x)
    context = PlanningContext(
        graph, space.candidates, ParameterBudget(1000), space.channel_axes, ()
    )
    axis = graph.parameter("a.weight").axis(0)
    candidates = (
        Candidate("single", (axis.select([0]),), axis),
        Candidate("joint", (axis.select([0, 2]),), axis),
    )
    expected = []
    for indices in ([0], [0, 2]):
        first = (model.a.weight.double() * model.a.weight.grad.double())[indices].norm()
        second = (model.consumer.weight.double() * model.consumer.weight.grad.double())[
            :, indices
        ].norm()
        expected.append((first + second).item())
    metric = workflow.PaperTaylor()
    assert context.score(metric, candidates) == pytest.approx(expected)
    accepted = context.impact((axis.select([0]),))
    conditional = context.score(metric, (candidates[1],), accepted_impact=accepted)
    expected_new = (
        (model.a.weight.double() * model.a.weight.grad.double())[2].norm()
        + (model.consumer.weight.double() * model.consumer.weight.grad.double())[:, 2].norm()
    ).item()
    assert conditional == pytest.approx([expected_new])
    strategy = workflow.Isomorphic(ratio=0.25)
    plan = pruner.plan(space, budget=ParameterBudget(1000), strategy=strategy)
    removed = tuple(plan.analysis.selection(graph.parameter("a.weight")).fully_selected_indices(0))
    pruner.apply(plan)
    with torch.no_grad():
        original.consumer.weight[:, removed] = 0
    torch.testing.assert_close(model(x), original(x))
    assert model.a.weight is model.b.weight
    model(x).sum().backward()
    save_checkpoint(model, tmp_path / "model.pt")
    restored_base = CoupledRoots()
    restored_base.b.weight = restored_base.a.weight
    restored = load_checkpoint(
        restored_base, tmp_path / "model.pt", map_location=execution_device
    ).eval()
    torch.testing.assert_close(restored(x), model(x))


def test_taylor_missing_or_nonfinite_gradients_fail_before_mutation(execution_device):
    model = TwoFamilies().eval()
    graph, pruner, space = build(model, torch.randn(2, 3))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(PlanningError, match="dense gradients"):
        pruner.plan(space, budget=ParameterBudget(1000), strategy=workflow.Isomorphic(ratio=0.25))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    model.a[0].weight.grad.fill_(float("nan"))
    with pytest.raises(PlanningError, match="nonfinite"):
        pruner.plan(space, budget=ParameterBudget(1000), strategy=workflow.Isomorphic(ratio=0.25))
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    graph.validate()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    pruner.apply(
        pruner.plan(space, budget=ParameterBudget(1000), strategy=workflow.Isomorphic(ratio=0.25))
    )


def test_identity_family_reuse_matches_generic_plan_and_reduces_queries(execution_device, tmp_path):
    class GenericTaylor(workflow.PaperTaylor):
        """Use the same formula through ordinary per-candidate analysis."""

    torch.manual_seed(19)
    first = TwoFamilies((32, 48)).double().eval()
    second = copy.deepcopy(first)
    x = torch.randn(2, 3, dtype=torch.float64)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        left.grad = torch.randn_like(left)
        right.grad = left.grad.clone()
    plans, counts = [], []
    for model, metric in ((first, workflow.PaperTaylor()), (second, GenericTaylor())):
        graph, pruner, space = build(model, x, granularity=4)
        propagate = graph.propagate
        calls = []

        def counted(*args, _calls=calls, _propagate=propagate, **kwargs):
            _calls.append(1)
            return _propagate(*args, **kwargs)

        graph.propagate = counted
        strategy = workflow.Isomorphic(metric, ratio=0.25)
        plan = pruner.plan(space, budget=ParameterBudget(1000), strategy=strategy)
        plans.append(plan)
        counts.append(len(calls))
        # Save/load plan data and check all affected bindings, not just ranks.
        path = tmp_path / f"{len(plans)}.json"
        path.write_text(json.dumps(plan.to_dict()))
        pruner.apply(type(plan).from_dict(json.loads(path.read_text())))
    assert plans[0].selected == plans[1].selected
    assert counts[0] < counts[1] / 4
    for path, values in first.state_dict().items():
        torch.testing.assert_close(values, second.state_dict()[path], rtol=0, atol=0)
    torch.testing.assert_close(first(x), second(x), rtol=0, atol=0)
    first(x).sum().backward()


def test_axis_statistics_refresh_after_gradient_and_weight_changes(execution_device):
    model = TwoFamilies().double().eval()
    x = torch.randn(2, 3, dtype=torch.float64)
    graph, _pruner, space = build(model, x)
    context = PlanningContext(
        graph, space.candidates, ParameterBudget(1000), space.channel_axes, ()
    )
    axis = graph.parameter("a.0.weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    metric = workflow.PaperTaylor()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    def reference():
        return (
            (model.a[0].weight[0] * model.a[0].weight.grad[0]).norm()
            + (model.a[2].weight[:, 0] * model.a[2].weight.grad[:, 0]).norm()
        ).item()

    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    model.a[0].weight.grad.mul_(3)
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    model.a[2].weight.grad = torch.full_like(model.a[2].weight, 2)
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    with torch.no_grad():
        model.a[0].weight.mul_(2)
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    # Unselected nonfinite positions must not invalidate this finite request.
    model.a[0].weight.grad[3].fill_(float("nan"))
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    assert len(metric._norms) <= 256
    assert metric._norm_positions <= 262_144
    # Simulate id reuse deterministically: a matching numeric key must still
    # refer to the actual gradient object, not a dead predecessor.
    for key, (_gradient_ref, norms) in tuple(metric._norms.items()):
        if key[0] != axis.tensor:
            continue
        predecessor = torch.empty_like(model.a[0].weight)
        dead = weakref.ref(predecessor)
        del predecessor
        metric._norms[key] = (dead, tuple(-1.0 for _ in norms))
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])
    model.a[0].weight.grad[0].fill_(float("nan"))
    with pytest.raises(PlanningError, match="nonfinite"):
        context.score(metric, (candidate,))
    model.a[0].weight.grad = torch.ones_like(model.a[0].weight)
    assert context.score(metric, (candidate,)) == pytest.approx([reference()])


class ResidualAttention(nn.Module):
    """Small attention residual with independently shrinkable FFN width."""

    def __init__(self, width=8, hidden=16, heads=2, second_heads=None):
        super().__init__()
        self.stem = nn.Linear(3, width)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.other = (
            nn.MultiheadAttention(width, second_heads, batch_first=True)
            if second_heads is not None
            else nn.Identity()
        )
        self.first = nn.Linear(width, hidden)
        self.second = nn.Linear(hidden, width)
        self.output = nn.Linear(width, 2)

    def forward(self, x):
        hidden = self.stem(x)
        normalized = self.norm(hidden)
        attention, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        hidden = hidden + attention
        if isinstance(self.other, nn.MultiheadAttention):
            attention, _ = self.other(hidden, hidden, hidden, need_weights=False)
            hidden = hidden + attention
        hidden = hidden + self.second(F.relu(self.first(hidden)))
        return self.output(hidden)


def test_attention_units_joint_taylor_and_independent_compact_reference(execution_device, tmp_path):
    torch.manual_seed(23)
    model = ResidualAttention().double().eval()
    original = copy.deepcopy(model)
    x = torch.randn(2, 3, 3, dtype=torch.float64)
    graph, pruner, raw = build(model, x, granularity=2)
    space = workflow.attention_candidates(graph, raw)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    axis = graph.parameter("stem.weight").axis(0)
    units = [candidate for candidate in space.candidates if candidate.axis == axis]
    assert len(units) == 4
    assert [tuple(candidate.remove[0].fully_selected_indices(0)) for candidate in units] == [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    context = PlanningContext(
        graph, space.candidates, ParameterBudget(10000), space.channel_axes, ()
    )
    # Independent union reference: row/column intersections count once, and
    # native MHA's nested output projection participates alongside packed QKV.
    indices, kept = [0, 4], [1, 2, 3, 5, 6, 7]
    expected = []
    for path, weight in model.named_parameters():
        if not path.endswith("weight"):
            continue
        products = weight.detach() * weight.grad
        if path in ("stem.weight", "norm.weight"):
            values = products[indices].flatten()
        elif path == "attention.in_proj_weight":
            rows = [block * 8 + index for block in range(3) for index in indices]
            rest = [block * 8 + index for block in range(3) for index in kept]
            values = torch.cat((products[rows].flatten(), products[rest][:, indices].flatten()))
        elif path in ("attention.out_proj.weight", "second.weight"):
            # The second FFN matrix has width 16 on its input, so only its
            # output rows are affected by this embedding-width request.
            values = products[indices].flatten()
            if path == "attention.out_proj.weight":
                values = torch.cat((values, products[kept][:, indices].flatten()))
        elif path in ("first.weight", "output.weight"):
            values = products[:, indices].flatten()
        else:
            raise AssertionError(path)
        expected.append(values.norm().item())
    assert context.score(workflow.PaperTaylor(), (units[0],)) == pytest.approx([sum(expected)])
    strategy = workflow.Isomorphic(ratio=0.25)
    plan = pruner.plan(space, budget=ParameterBudget(10000), strategy=strategy)

    class GenericTaylor(workflow.PaperTaylor):
        """Bypass native proof reuse while retaining the same metric formula."""

    generic = copy.deepcopy(model)
    for source, target in zip(model.parameters(), generic.parameters(), strict=True):
        target.grad = source.grad.clone()
    other_graph, other_pruner, other_raw = build(generic, x, granularity=2)
    other_space = workflow.attention_candidates(other_graph, other_raw)
    other_strategy = workflow.Isomorphic(GenericTaylor(), ratio=0.25)
    other_plan = other_pruner.plan(
        other_space, budget=ParameterBudget(10000), strategy=other_strategy
    )
    assert plan.selected == other_plan.selected
    assert strategy.families == other_strategy.families
    removed = set(plan.analysis.selection(axis.tensor).fully_selected_indices(0))
    removed_hidden = set(
        plan.analysis.selection(graph.parameter("first.weight")).fully_selected_indices(0)
    )
    keep = [index for index in range(8) if index not in removed]
    keep_hidden = [index for index in range(16) if index not in removed_hidden]
    assert len(keep) == 6 and len(keep_hidden) == 12
    assert len(removed & set(range(4))) == len(removed & set(range(4, 8))) == 1
    reference = ResidualAttention(width=6, hidden=12).double().eval()
    packed = [block * 8 + index for block in range(3) for index in keep]
    with torch.no_grad():
        reference.stem.weight.copy_(original.stem.weight[keep])
        reference.stem.bias.copy_(original.stem.bias[keep])
        reference.norm.weight.copy_(original.norm.weight[keep])
        reference.norm.bias.copy_(original.norm.bias[keep])
        reference.attention.in_proj_weight.copy_(original.attention.in_proj_weight[packed][:, keep])
        reference.attention.in_proj_bias.copy_(original.attention.in_proj_bias[packed])
        reference.attention.out_proj.weight.copy_(original.attention.out_proj.weight[keep][:, keep])
        reference.attention.out_proj.bias.copy_(original.attention.out_proj.bias[keep])
        reference.first.weight.copy_(original.first.weight[keep_hidden][:, keep])
        reference.first.bias.copy_(original.first.bias[keep_hidden])
        reference.second.weight.copy_(original.second.weight[keep][:, keep_hidden])
        reference.second.bias.copy_(original.second.bias[keep])
        reference.output.weight.copy_(original.output.weight[:, keep])
        reference.output.bias.copy_(original.output.bias)
    restored_plan = type(plan).from_dict(json.loads(json.dumps(plan.to_dict())))
    pruner.apply(restored_plan)
    other_pruner.apply(other_plan)
    torch.testing.assert_close(model(x), reference(x))
    torch.testing.assert_close(generic(x), reference(x))
    model(x).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    save_checkpoint(model, tmp_path / "attention.pt")
    restored = load_checkpoint(
        ResidualAttention().double(), tmp_path / "attention.pt", map_location=execution_device
    ).eval()
    torch.testing.assert_close(restored(x), reference(x))


def test_attention_units_incompatible_shared_heads_fail_without_mutation(execution_device):
    x = torch.randn(2, 3, 3)
    model = ResidualAttention(second_heads=4).eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    graph, _pruner, space = build(model, x)
    with pytest.raises(PlanningError, match="incompatible head partitions"):
        workflow.attention_candidates(graph, space)
    graph.validate()
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    supported = ResidualAttention(second_heads=2).eval()
    graph, pruner, space = build(supported, x)
    space = workflow.attention_candidates(graph, space)
    for parameter in supported.parameters():
        parameter.grad = torch.ones_like(parameter)
    pruner.apply(
        pruner.plan(space, budget=ParameterBudget(10000), strategy=workflow.Isomorphic(ratio=0.25))
    )
    assert supported.attention.embed_dim == supported.other.embed_dim == 6
    assert supported(x).shape == (2, 3, 2)
