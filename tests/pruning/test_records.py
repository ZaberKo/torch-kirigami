"""Caller-owned collections cannot alter candidates or execution reports."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from tests.support.pruning import build
from torch_kirigami import CandidateAxis, TensorRef
from torch_kirigami.pruning import Candidate, ChannelRatio, PruningResult


def test_candidates_freeze_seeds_and_reject_invalid_domains():
    axis = TensorRef("channels", (6,)).axis(0)
    seeds = [axis.select([1, 4])]
    candidate = Candidate("pair", seeds, axis)
    seeds.clear()
    assert candidate.remove == (axis.select([1, 4]),)
    with pytest.raises(FrozenInstanceError):
        candidate.key = "changed"
    for factory in (
        lambda: Candidate("", [axis.select([1])]),
        lambda: Candidate("empty", []),
        lambda: Candidate("empty", [axis.select([])]),
        lambda: Candidate("bad_axis", [axis.select([1])], "channels"),
        lambda: CandidateAxis("", axis),
        lambda: CandidateAxis("channels", "axis"),
        lambda: CandidateAxis("channels", axis, True),
        lambda: CandidateAxis("channels", axis, 0),
        lambda: ChannelRatio(float("nan")),
        lambda: ChannelRatio(1),
        lambda: ChannelRatio(-0.1),
        lambda: ChannelRatio(0.5, scope="typo"),
        lambda: ChannelRatio(0.5, axes=["axis"]),
    ):
        with pytest.raises((ValueError, TypeError)):
            factory()


def test_execution_report_detaches_maps_and_preserves_original_coordinate_segments(
    execution_device,
):
    model = nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 2))
    original = model[0].weight
    graph, pruner = build(model, torch.randn(2, 4))
    ref = graph.parameter("0.weight")
    _, result = pruner.apply(pruner.plan(remove=[ref.axis(0).select([1, 4])]))
    assert result.parameter_map[original] is model[0].weight
    torch.testing.assert_close(model[0].weight, original[[0, 2, 3, 5]])
    parameters = dict(result.parameter_map)
    coordinates = {ref: list(segments) for ref, segments in result.coordinate_maps.items()}
    report = ["original report"]
    detached = PruningResult(result.plan, result.structure, parameters, coordinates, report)
    parameters.clear()
    for segments in coordinates.values():
        segments.clear()
    coordinates.clear()
    report.clear()
    assert dict(detached.parameter_map) == dict(result.parameter_map)
    assert dict(detached.coordinate_maps) == dict(result.coordinate_maps)
    assert detached.report == ("original report",)
    with pytest.raises(TypeError):
        detached.parameter_map[original] = original
    with pytest.raises(TypeError):
        detached.coordinate_maps[ref] = ()
