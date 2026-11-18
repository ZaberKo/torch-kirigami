"""Composable building blocks for sparse training before structural pruning.

The package offers scalar regularizers over parameter groups (`GroupLasso`,
`GroupSquaredL2`, and `ScaleL1`), in-place parameter operations that scale,
zero, or renormalize groups (`scale_groups_`, `zero_groups_`, and
`set_group_norms_`), explicit activation gates with gate-based importance
scoring (`ChannelGate`, `GateBinding`, and `GateMagnitude`), a
`CumulativeChannelBudget` that measures removal across pruning rounds against
the original widths, and stateless scalar schedules with selection-stability
statistics. These parts do not form a training framework: the caller owns the
task loss, optimizer, loss scaling, training loop, and pruning policy, and the
complete sparse-training algorithms live in the examples.
"""

from .budget import CumulativeChannelBudget
from .gates import ChannelGate, GateBinding, GateMagnitude, register_gate_operators
from .operations import scale_groups_, set_group_norms_, zero_groups_
from .regularizers import GroupLasso, GroupSquaredL2, ScaleL1
from .schedules import (
    Constant,
    Linear,
    Piecewise,
    Polynomial,
    SelectionWindow,
    selection_similarity,
)

__all__ = [
    "ChannelGate",
    "Constant",
    "CumulativeChannelBudget",
    "GateBinding",
    "GateMagnitude",
    "GroupLasso",
    "GroupSquaredL2",
    "Linear",
    "Piecewise",
    "Polynomial",
    "ScaleL1",
    "SelectionWindow",
    "register_gate_operators",
    "scale_groups_",
    "selection_similarity",
    "set_group_norms_",
    "zero_groups_",
]
