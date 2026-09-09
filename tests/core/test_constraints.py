"""core / constraints contracts."""

import pytest
import torch
from torch import nn

from torch_kirigami import (
    AxisPort,
    Balanced,
    BlockBalance,
    CandidateAxis,
    DependencyGraph,
    Divisible,
    IndexSet,
    LayoutConstraint,
    Region,
    Selection,
    TensorRef,
)
from torch_kirigami.operation import PartitionedLayout


def test_divisibility_reports_choices_not_automatic_completion():
    graph = DependencyGraph.build(nn.Linear(4, 8), args=(torch.randn(2, 4),))
    axis = graph.parameter("weight").axis(0)
    impact = graph.propagate(remove=[axis.select([0])], constraints=[Divisible(axis, 4)])
    assert impact.status == "unresolved"
    assert set(impact.selection(axis.tensor).fully_selected_indices(0)) == {0}
    assert (
        graph.propagate(remove=[axis.select([0, 1, 2, 3])], constraints=[Divisible(axis, 4)]).status
        == "resolved"
    )


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


def test_partition_contracts_preserve_subset_and_local_coordinates():
    ref = TensorRef("weight", (4, 3))
    row0, row1 = IndexSet.span(0, 2), IndexSet.span(2, 4)
    full = IndexSet.span(0, 3)
    parts = [Region([row0, full]), Region([row1, full])]
    layout = PartitionedLayout(ref, parts)
    parts.clear()
    removed = Selection(ref, (Region((row0, IndexSet.of([0]))), Region((row1, IndexSet.of([2])))))
    assert layout.retained_regions(removed) == (
        Region((row0, IndexSet.of([1, 2]))),
        Region((row1, IndexSet.of([0, 1]))),
    )
    with pytest.raises(ValueError):
        layout.retained_regions(Selection(TensorRef("foreign", ref.shape)))
    with pytest.raises(ValueError):
        PartitionedLayout(ref, (layout.partitions[0],) * 2)
    with pytest.raises(ValueError):
        LayoutConstraint(ref, (AxisPort(TensorRef("foreign", ref.shape).axis(0)),))
    # Balancing a declared subset is legal; it need not cover all original positions.
    assert Balanced(ref.axis(0), [IndexSet.of([0]), IndexSet.of([3])]).check({}) is None
    for partitions in ((), (IndexSet(),), (IndexSet.of([4]),), (row0, row0)):
        with pytest.raises(ValueError):
            Balanced(ref.axis(0), partitions)
    with pytest.raises(ValueError):
        CandidateAxis("channel", ref.axis(0), block_size=1.0)
    with pytest.raises(ValueError):
        BlockBalance(ref.axis(0), ref.axis(0), 1.0)
