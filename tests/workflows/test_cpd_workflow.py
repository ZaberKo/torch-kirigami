"""CPD workflow score and public planning coverage."""

import copy
import sys

import cpd_pruning as workflow
import pytest
import torch
from cpd_pruning import CyclingBatches, load_progress, rebuild_sgd_optimizer, save_progress
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torchvision.models.resnet import BasicBlock
from torchvision.models.vision_transformer import EncoderBlock

from tests.support.pruning import build
from torch_kirigami.pruning import (
    Candidate,
    CandidateSpace,
    ChannelCount,
    CPDLeastFirst,
    CPDMetric,
    ParameterBudget,
    PlanningError,
    Pruner,
    PruningPlan,
    load_checkpoint,
    save_checkpoint,
)
from torch_kirigami.pruning.cpd import fisher_scores


def test_cpd_fisher_squares_the_channel_sum() -> None:
    """Match CPD's signed channel reduction before the Fisher square."""
    weight = torch.tensor(((1.0, 2.0), (3.0, 4.0)))
    gradient = torch.tensor(((2.0, -1.0), (1.0, -1.0)))

    scores = fisher_scores(weight, gradient, dim=0, activation_elements=4)

    assert scores == pytest.approx((0.0, 0.25))


def test_cpd_metric_drives_structural_plan_and_checkpoint(tmp_path, execution_device):
    model = nn.Linear(2, 4, bias=False, dtype=torch.float64, device=execution_device)
    graph, pruner = build(model, torch.randn(3, 2, dtype=torch.float64, device=execution_device))
    axis = graph.parameter("weight").axis(0)
    first = Candidate("first", (axis.select([0]),), axis)
    pair = Candidate("alternative", (axis.select([1]),), axis)
    metric = CPDMetric({axis: (1.0, 4.0, 9.0, 16.0)})

    compact_pruner = Pruner(pruner.model, graph=pruner.graph, preserve_io=False)
    plan = compact_pruner.plan(
        CandidateSpace((first, pair), (axis,)),
        budget=ChannelCount((3,), (axis,)),
        strategy=CPDLeastFirst(metric),
    )
    assert plan.selected == ("first",)
    compact, _result = compact_pruner.apply(plan)
    assert compact.out_features == 3
    assert compact(torch.randn(2, 2, dtype=torch.float64, device=execution_device)).shape == (2, 3)

    replay = nn.Linear(2, 4, bias=False, dtype=torch.float64, device=execution_device)
    restored_plan = PruningPlan.from_dict(plan.to_dict())
    replay, _result = Pruner(replay).apply(restored_plan)
    path = tmp_path / "cpd.pt"
    save_checkpoint(compact, path)
    loaded = load_checkpoint(
        nn.Linear(2, 4, bias=False, dtype=torch.float64, device=execution_device),
        path,
        map_location=execution_device,
    )
    torch.testing.assert_close(loaded.weight, compact.weight)
    assert replay.out_features == 3


def test_cpd_replays_an_interrupted_training_epoch() -> None:
    dataset = TensorDataset(torch.arange(18), torch.arange(18))
    sampler_generator = torch.Generator().manual_seed(17)
    worker_generator = torch.Generator().manual_seed(18)
    stream = CyclingBatches(
        DataLoader(
            dataset,
            batch_size=3,
            sampler=RandomSampler(dataset, generator=sampler_generator),
            generator=worker_generator,
        ),
        sampler_generator,
    )
    for _ in range(4):
        next(stream)
    state = stream.state_dict()
    expected = [next(stream)[0].clone() for _ in range(5)]

    restored_generator = torch.Generator().manual_seed(999)
    restored = CyclingBatches(
        DataLoader(
            dataset,
            batch_size=3,
            sampler=RandomSampler(dataset, generator=restored_generator),
            generator=torch.Generator().manual_seed(1000),
        ),
        restored_generator,
    )
    restored.load_state_dict(state)

    actual = [next(restored)[0] for _ in range(5)]
    assert all(torch.equal(left, right) for left, right in zip(actual, expected, strict=True))


def test_cpd_migrates_sgd_momentum_through_a_structural_recipe(tmp_path, execution_device) -> None:
    model = nn.Linear(2, 4, bias=False, dtype=torch.float64, device=execution_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model(torch.ones(3, 2, dtype=torch.float64, device=execution_device)).sum().backward()
    optimizer.step()
    momentum = optimizer.state[model.weight]["momentum_buffer"].clone()

    graph, pruner = build(model, torch.randn(3, 2, dtype=torch.float64, device=execution_device))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("first", (axis.select([0]),), axis)
    plan = Pruner(model, graph=graph, preserve_io=False).plan(
        CandidateSpace((candidate,), (axis,)),
        budget=ChannelCount((3,), (axis,)),
        strategy=CPDLeastFirst(CPDMetric({axis: (1.0, 2.0, 3.0, 4.0)})),
    )
    parameters = dict(model.named_parameters())
    compact, result = pruner.apply(plan)

    rebuilt = rebuild_sgd_optimizer(optimizer, result, parameters)

    torch.testing.assert_close(rebuilt.state[compact.weight]["momentum_buffer"], momentum[1:])
    with torch.device("cpu"):
        generator = torch.Generator().manual_seed(5)
        stream = CyclingBatches(
            DataLoader(
                TensorDataset(torch.arange(6), torch.arange(6)), batch_size=2, generator=generator
            ),
            generator,
        )
        next(stream)
    save_progress(
        tmp_path,
        model=compact,
        optimizer=rebuilt,
        pruning_steps=1,
        current_params=compact.weight.numel(),
        records=[{"stage": "pretrained", "top1": 0.0}],
        config={"model": "linear"},
        batch_stream=stream,
        worker_generator=generator,
    )

    progress = load_progress(tmp_path / "progress.pt")
    restored = load_checkpoint(
        nn.Linear(2, 4, bias=False, dtype=torch.float64, device=execution_device),
        tmp_path / str(progress["model_checkpoint"]),
        map_location=execution_device,
    )

    assert progress["pruning_steps"] == 1
    torch.testing.assert_close(restored.weight, compact.weight)


def test_cpd_removes_a_group_batch_sized_like_the_source_hook(execution_device) -> None:
    """A wide axis loses CPD's whole group batch, not a single channel."""
    model = nn.Linear(2, 128, bias=False, dtype=torch.float64, device=execution_device)
    graph, pruner = build(model, torch.randn(3, 2, dtype=torch.float64, device=execution_device))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(
        Candidate(f"c{index}", (axis.select([index]),), axis) for index in range(128)
    )
    metric = CPDMetric({axis: tuple(float(index) for index in range(128))})

    compact_pruner = Pruner(pruner.model, graph=pruner.graph, preserve_io=False)
    plan = compact_pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelCount((124,), (axis,)),
        strategy=CPDLeastFirst(metric),
    )

    assert plan.selected == ("c0", "c1", "c2", "c3")
    compact, _result = compact_pruner.apply(plan)
    assert compact.out_features == 124


def test_cpd_group_batch_shrinks_as_an_axis_narrows(execution_device) -> None:
    """Below 32 remaining channels the source hook drops to a single channel."""
    model = nn.Linear(2, 20, bias=False, dtype=torch.float64, device=execution_device)
    graph, pruner = build(model, torch.randn(3, 2, dtype=torch.float64, device=execution_device))
    axis = graph.parameter("weight").axis(0)
    candidates = tuple(Candidate(f"c{index}", (axis.select([index]),), axis) for index in range(20))
    metric = CPDMetric({axis: tuple(float(index) for index in range(20))})

    compact_pruner = Pruner(pruner.model, graph=pruner.graph, preserve_io=False)
    plan = compact_pruner.plan(
        CandidateSpace(candidates, (axis,)),
        budget=ChannelCount((19,), (axis,)),
        strategy=CPDLeastFirst(metric),
    )

    assert plan.selected == ("c0",)


def test_cpd_ranks_group_prefixes_not_the_global_minimum():
    model = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 64), nn.Linear(64, 2))
    graph, pruner = build(model, torch.randn(2, 2))
    axes = (graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(0))
    candidates = tuple(
        Candidate(f"{group}:{i:03}", (axis.select([i]),), axis)
        for group, axis in enumerate(axes)
        for i in range(64)
    )
    metric = CPDMetric({axes[0]: (0.1,) + (100.0,) * 63, axes[1]: (1.0,) * 64})
    plan = pruner.plan(
        CandidateSpace(candidates, axes),
        budget=ChannelCount((64, 60), axes),
        strategy=CPDLeastFirst(metric),
    )
    assert plan.selected == ("1:000", "1:001", "1:002", "1:003")
    compact, _ = pruner.apply(plan)
    assert compact[0].out_features == 64
    assert compact[1].out_features == compact[2].in_features == 60
    assert compact(torch.randn(2, 2)).shape == (2, 2)


def test_cpd_rejects_multichannel_candidate_without_mutation():
    model = nn.Linear(2, 4)
    original = model.weight.detach().clone()
    graph, _ = build(model, torch.randn(2, 2))
    axis = graph.parameter("weight").axis(0)
    candidate = Candidate("pair", (axis.select([0, 1]),), axis)
    with pytest.raises(PlanningError, match="single-channel"):
        Pruner(model, graph=graph, preserve_io=False).plan(
            CandidateSpace((candidate,), (axis,)),
            budget=ChannelCount((2,), (axis,)),
            strategy=CPDLeastFirst(CPDMetric({axis: (1.0,) * 4})),
        )
    torch.testing.assert_close(model.weight, original)


@pytest.mark.parametrize(("width", "expected"), [(1, 0), (20, 1), (32, 2), (64, 4)])
def test_cpd_channel_count_boundaries(width, expected):
    model = nn.Linear(2, width)
    graph, _ = build(model, torch.randn(2, 2))
    axis = graph.parameter("weight").axis(0)
    assert CPDLeastFirst(CPDMetric({axis: (1.0,) * width})).channel_count(axis) == expected


def _tiny_resnet():
    return nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.AdaptiveAvgPool2d(4),
        BasicBlock(4, 4),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(4, 6),
    )


def test_cpd_interval_consumes_exactly_requested_batches_and_averages_bn_group():
    """Independently check all three group members, including the BN bias."""
    model = _tiny_resnet().eval()
    teacher = copy.deepcopy(model).eval()
    graph, _ = build(model, torch.randn(2, 3, 4, 4))
    expected = torch.zeros(4, dtype=torch.float64)
    handles = []
    for name in ("2.conv1.weight", "2.bn1.weight", "2.bn1.bias"):
        parameter = model.get_parameter(name)

        def accumulate(gradient, parameter=parameter):
            products = parameter.detach().double() * gradient.detach().double()
            # Two images, four output channels, 4x4 spatial; mean of three
            # members, divided by six source output/output-input memberships.
            expected.add_(products.reshape(4, -1).sum(1).square() / (128 * 3 * 6))

        handles.append(parameter.register_hook(accumulate))
    consumed = []

    def batches():
        for index in range(4):
            consumed.append(index)
            yield torch.randn(2, 3, 4, 4), torch.tensor([0, 1])

    metric, interval = workflow.train_and_collect_cpd_metric(
        model,
        teacher,
        graph,
        ["2.conv1"],
        batches(),
        "cpu",
        2,
        torch.optim.SGD(model.parameters(), lr=0),
        1.0,
        1.0,
    )
    for handle in handles:
        handle.remove()
    assert consumed == [0, 1]
    assert interval["interval_samples"] == 4
    torch.testing.assert_close(
        torch.tensor(next(iter(metric.scores.values())), dtype=torch.float64), expected
    )
    assert workflow.scoring_groups(model, ["2.conv1"])["2.conv1"][1] == 6


def test_cpd_ffn_group_includes_producer_bias_and_consumer_cardinality():
    model = nn.Sequential(EncoderBlock(2, 8, 16, 0.0, 0.0))
    assert workflow.scoring_groups(model, ["0.mlp.0"]) == {
        "0.mlp.0": (("0.mlp.0.weight", "0.mlp.0.bias"), 4)
    }


@pytest.mark.parametrize("width", [80, 16])
def test_cpd_ffn_scores_drive_a_physical_plan_and_checkpoint(tmp_path, width):
    model = nn.Sequential(
        EncoderBlock(2, 8, width, 0.0, 0.0), nn.Flatten(1), nn.Linear(24, 6)
    ).eval()
    skeleton = copy.deepcopy(model)
    teacher = copy.deepcopy(model)
    images, labels = torch.randn(2, 3, 8), torch.tensor([0, 1])
    graph, pruner = build(model, images)
    expected = torch.zeros(width, dtype=torch.float64)
    handles = []
    for parameter in model[0].mlp[0].parameters():

        def accumulate(gradient, parameter=parameter):
            products = parameter.detach().double() * gradient.detach().double()
            expected.add_(products.reshape(width, -1).sum(1).square() / (2 * 3 * width * 2 * 4))

        handles.append(parameter.register_hook(accumulate))
    metric, _ = workflow.train_and_collect_cpd_metric(
        model,
        teacher,
        graph,
        ["0.mlp.0"],
        [(images, labels)],
        "cpu",
        1,
        torch.optim.SGD(model.parameters(), lr=0),
        1.0,
        1.0,
    )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        torch.tensor(next(iter(metric.scores.values())), dtype=torch.float64), expected
    )
    plan = pruner.plan(
        pruner.discover_candidates(targets=["0.mlp.0"]),
        budget=ParameterBudget(sum(p.numel() for p in model.parameters())),
        # Even a CNN cap of one must not reduce the FFN fixed count.
        strategy=workflow.make_cpd_strategy(model, metric, cnn_cap=1),
    )
    compact, _ = pruner.apply(plan)
    remaining = 48 if width == 80 else 1
    assert len(plan.selected) == width - remaining
    assert compact[0].mlp[0].out_features == compact[0].mlp[3].in_features == remaining
    save_checkpoint(compact, tmp_path / "ffn.pt")
    restored = load_checkpoint(skeleton, tmp_path / "ffn.pt").eval()
    torch.testing.assert_close(restored(images), compact.eval()(images))
    # Recalibrate the physically compact model: reference denominator must use
    # its new width, never the original width or the token axis as channels.
    graph, pruner = build(compact, images)
    expected = torch.zeros(remaining, dtype=torch.float64)
    handles = []
    for parameter in compact[0].mlp[0].parameters():

        def accumulate_compact(gradient, parameter=parameter):
            products = parameter.detach().double() * gradient.detach().double()
            expected.add_(products.reshape(remaining, -1).sum(1).square() / (2 * 3 * remaining * 8))

        handles.append(parameter.register_hook(accumulate_compact))
    metric, _ = workflow.train_and_collect_cpd_metric(
        compact,
        teacher,
        graph,
        ["0.mlp.0"],
        [(images, labels)],
        "cpu",
        1,
        torch.optim.SGD(compact.parameters(), lr=0),
        1.0,
        1.0,
    )
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(
        torch.tensor(next(iter(metric.scores.values())), dtype=torch.float64), expected
    )
    # Literal source default axis=1 on a masked [2, 3, original_width]
    # output would divide by 2*original_width*remaining, not 2*3*remaining.
    assert expected.abs().sum() > 0
    source_default_axis_scores = expected * (3 / width)
    assert not torch.allclose(expected, source_default_axis_scores, rtol=1e-10, atol=0)
    if remaining == 1:
        original = {key: value.clone() for key, value in compact.state_dict().items()}
        with pytest.raises(
            PlanningError, match=r"no candidates remaining|no structurally complete group"
        ):
            pruner.plan(
                pruner.discover_candidates(targets=["0.mlp.0"]),
                budget=ParameterBudget(sum(p.numel() for p in compact.parameters())),
                strategy=workflow.make_cpd_strategy(compact, metric, cnn_cap=4),
            )
        for key, value in compact.state_dict().items():
            torch.testing.assert_close(value, original[key])


@pytest.mark.parametrize(
    ("width", "expected"), [(256, 32), (64, 32), (33, 32), (32, 31), (20, 19), (1, 0)]
)
def test_cpd_ffn_fixed_count_is_independent_of_cnn_cap(width, expected):
    model = nn.Linear(2, width)
    graph, _ = build(model, torch.randn(2, 2))
    axis = graph.parameter("weight").axis(0)
    metric = CPDMetric({axis: (1.0,) * width})
    assert (
        CPDLeastFirst(metric, candidates_per_interval=1, ffn_axes=(axis,)).channel_count(axis)
        == expected
    )


def test_cpd_nlc_normalization_checks_axis_and_ignores_original_width():
    module = nn.Linear(8, 80)
    # A post-compaction output has 48 channels; no stale module width is used.
    assert workflow.cpd_activation_elements(module, torch.zeros(2, 3, 48), 48) == 288
    assert workflow.cpd_activation_elements(module, torch.zeros(5, 7, 48), 48) == 1680
    with pytest.raises(PlanningError, match="aligned NLC"):
        workflow.cpd_activation_elements(module, torch.zeros(2, 48, 3), 48)
    with pytest.raises(PlanningError, match="aligned NLC"):
        workflow.cpd_activation_elements(module, torch.zeros(2, 48), 48)


def test_cpd_resume_rejects_changed_ffn_policy():
    config = {
        "deletion_policy": {"ffn": "ffn_fixed_32_keep_one"},
        "activation_normalization": "surviving_output_elements_NLC_or_NCHW",
    }
    workflow._check_resume_config(config, copy.deepcopy(config))
    with pytest.raises(ValueError, match="deletion_policy"):
        workflow._check_resume_config(
            config, {**config, "deletion_policy": {"ffn": "cnn_adaptive_keep_one"}}
        )


def test_cpd_mixed_cnn_ffn_prefixes_use_their_own_counts():
    model = nn.Sequential(nn.Linear(2, 64), nn.Linear(64, 64), nn.Linear(64, 2))
    graph, pruner = build(model, torch.randn(2, 2))
    cnn, ffn = graph.parameter("0.weight").axis(0), graph.parameter("1.weight").axis(0)
    candidates = tuple(
        Candidate(f"{group}:{i:03}", (axis.select([i]),), axis)
        for group, axis in enumerate((cnn, ffn))
        for i in range(64)
    )
    # FFN's individual score is lower, but its 32-prefix costs 32 versus
    # CNN's 4-prefix costing 8: the CNN group must win.
    metric = CPDMetric({cnn: (2.0,) * 64, ffn: (1.0,) * 64})
    plan = pruner.plan(
        CandidateSpace(candidates, (cnn, ffn)),
        budget=ChannelCount((60, 64), (cnn, ffn)),
        strategy=CPDLeastFirst(metric, ffn_axes=(ffn,)),
    )
    assert plan.selected == ("0:000", "0:001", "0:002", "0:003")
    compact, _ = pruner.apply(plan)
    assert compact[0].out_features == compact[1].in_features == 60
    assert compact[1].out_features == 64
    assert compact(torch.randn(2, 2)).shape == (2, 2)


def test_cpd_manifest_failure_preserves_last_committed_model(tmp_path, monkeypatch):
    model = nn.Linear(2, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(1)
    stream = CyclingBatches([], generator)

    def save():
        save_progress(
            tmp_path,
            model=model,
            optimizer=optimizer,
            pruning_steps=1,
            current_params=sum(p.numel() for p in model.parameters()),
            records=[],
            config={},
            batch_stream=stream,
            worker_generator=generator,
        )

    save()
    first = load_progress(tmp_path / "progress.pt")
    save()
    second = load_progress(tmp_path / "progress.pt")
    assert first["model_checkpoint"] != second["model_checkpoint"]
    original = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(100)

    def fail(*args):
        raise OSError("manifest write failed")

    monkeypatch.setattr(workflow, "_atomic_torch_save", fail)
    with pytest.raises(OSError, match="manifest write failed"):
        save()
    committed = load_progress(tmp_path / "progress.pt")
    assert committed["model_checkpoint"] == second["model_checkpoint"]
    restored = load_checkpoint(nn.Linear(2, 4), tmp_path / committed["model_checkpoint"])
    torch.testing.assert_close(restored.weight, original)


@pytest.mark.parametrize("ratio", [0.0, 0.001])
@pytest.mark.parametrize("interrupt_phase", ["online", "recovery"])
def test_cpd_main_recovery_resume_matches_uninterrupted(
    tmp_path, monkeypatch, ratio, interrupt_phase
):
    """Exercise real build/plan/apply/save/load and resume after recovery epoch one."""
    with torch.device("cpu"):
        dataset = TensorDataset(torch.randn(6, 3, 8, 8), torch.arange(6))
    monkeypatch.setattr(workflow, "make_model", lambda *a, **kw: _tiny_resnet())
    monkeypatch.setattr(workflow, "load_images", lambda *a, **kw: (dataset, dataset, {}))
    monkeypatch.setattr(workflow, "measure_model", lambda *a: {})

    def run(output, resume=False, diagnostic=0):
        argv = [
            "cpd_pruning.py",
            "--device",
            "cpu",
            "--train_workers",
            "0",
            "--val_workers",
            "0",
            "--train_batch_size",
            "2",
            "--val_batch_size",
            "2",
            "--pruning_interval_batches",
            "2",
            "--pruning_ratio",
            str(ratio),
            "--recovery_epochs",
            "2",
            "--output",
            str(output),
            "--diagnostic_every",
            str(diagnostic),
        ]
        if ratio == 0:
            argv.extend(["--control_intervals", "2"])
        if resume:
            argv.extend(["--resume", str(output / "progress.pt")])
        monkeypatch.setattr(sys, "argv", argv)
        workflow.main()

    reference = tmp_path / "reference"
    interrupted = tmp_path / "interrupted"
    run(reference)
    original = workflow.save_progress

    def stop_after_epoch(*args, **kwargs):
        original(*args, **kwargs)
        if (interrupt_phase == "recovery" and kwargs.get("recovery_epochs_completed") == 1) or (
            interrupt_phase == "online" and kwargs["pruning_steps"] == 1
        ):
            raise RuntimeError("test interruption")

    monkeypatch.setattr(workflow, "save_progress", stop_after_epoch)
    with pytest.raises(RuntimeError, match="test interruption"):
        run(interrupted, diagnostic=1)
    monkeypatch.setattr(workflow, "save_progress", original)
    run(interrupted, resume=True, diagnostic=1)
    expected = load_checkpoint(_tiny_resnet(), reference / "model.pt", map_location="cpu")
    actual = load_checkpoint(_tiny_resnet(), interrupted / "model.pt", map_location="cpu")
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    progress = load_progress(interrupted / "progress.pt")
    assert progress["recovery_epochs_completed"] == 2
    last = progress["records"][-1]
    assert last["recovery_optimizer_steps"] == 6
    assert last["recovery_samples"] == 12
    assert last["total_optimizer_steps"] == last["online_optimizer_steps"] + 6
    assert progress["config"]["stopping_metric"] == "whole_model_parameter_count"
    assert progress["config"]["training_budget"]["recovery_steps_per_epoch"] == 3
    assert last["parameter_target_reached"]
    assert last["parameter_reduction_ratio"] == pytest.approx(
        (progress["config"]["initial_params"] - last["#Params"])
        / progress["config"]["initial_params"]
    )
    assert last["target_overshoot_params"] == progress["config"]["max_params"] - last["#Params"]
    # Resuming a completed run must not execute recovery again.
    run(interrupted, resume=True)
    assert load_progress(interrupted / "progress.pt")["records"] == progress["records"]
