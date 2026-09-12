"""Composable sparse losses and operations; algorithms live in examples."""

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
