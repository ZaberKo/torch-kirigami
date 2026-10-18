"""Shared optimization eligibility checks for native structural computations.

Ordinary analysis still accepts extension types. Passing this check permits
consumers to apply their own narrower reuse proofs; it does not prove that any
arbitrary computation on the record is cacheable.
"""

from dataclasses import fields

import torch
from torch import nn

from .contracts import (
    AttributeBinding,
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Constraint,
    Divisible,
    Fixed,
    LayoutConstraint,
    NonEmpty,
    Requirement,
    ShapeExpr,
)
from .graph_state import DependencyGraphState
from .operation import PartitionedLayout
from .operators.defaults import default_rules
from .operators.rules import is_native_rule
from .operators.shapes import CallArgumentConstraint
from .relations import (
    AxisPort,
    AxisRelation,
    BlockMap,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
)
from .selection import AxisRef, IndexSet, Region, Selection, TensorRef

_NATIVE_RECORD_TYPES = (
    AttributeBinding,
    Requirement,
    AxisRef,
    TensorRef,
    IndexSet,
    Region,
    Selection,
    AxisPort,
    BlockMap,
    AxisRelation,
    BroadcastRelation,
    PermuteRelation,
    ReshapeRelation,
    SliceRelation,
    AxisBarrier,
    Balanced,
    Barrier,
    BlockBalance,
    Divisible,
    Fixed,
    LayoutConstraint,
    NonEmpty,
    ShapeExpr,
    PartitionedLayout,
    CallArgumentConstraint,
)


def has_native_semantics(value: object) -> bool:
    """Require native behavior at every depth before bypassing repeated analysis.

    Frozen containers and value equality do not prove native behavior: an integer
    subclass can override arithmetic inside an otherwise built-in mapping. Reject
    such extensions without invoking their methods. They remain valid inputs to
    ordinary analysis; this predicate only controls optimization eligibility.
    """
    if type(value) in (str, int, bool, type(None)):
        return True
    if type(value) is tuple:
        return all(has_native_semantics(item) for item in value)
    if type(value) in _NATIVE_RECORD_TYPES:
        return all(has_native_semantics(getattr(value, field.name)) for field in fields(value))
    return False


def has_native_query_semantics(
    state: DependencyGraphState, constraints: tuple[Constraint, ...]
) -> bool:
    """Prove that structural queries cannot invoke extension behavior.

    This predicate inspects model accessors, captured arguments and operator
    registrations. The caller still validates snapshot freshness, registered
    tensor types, and its own scoring or filtering callbacks.
    """
    if not all(
        has_native_semantics(value)
        for value in (
            *state.relations,
            *state.constraints,
            *constraints,
            *state.expressions.values(),
            *state.requirements,
        )
    ):
        return False
    if not _has_standard_module_access(state.model):
        return False
    if not all(
        _has_native_arguments((operation.args, operation.kwargs))
        for operation in state.operations.values()
    ):
        return False
    defaults = {target: type(rule) for target, rule, _ in default_rules()}
    return all(
        type(rule) is defaults.get(target) and is_native_rule(rule)
        for table in (state.registry.modules, state.registry.functions, state.registry.methods)
        for target, rule in table.items()
    )


def _has_native_arguments(value: object) -> bool:
    """Recognize inert call arguments before reusing metadata-query validation."""
    # Dispatch containers first so nested tuples need only one recursive pass.
    if type(value) in (tuple, list, torch.Size):
        return all(_has_native_arguments(item) for item in value)
    if type(value) is dict:
        return all(
            _has_native_arguments(key) and _has_native_arguments(item)
            for key, item in value.items()
        )
    if type(value) is slice:
        return _has_native_arguments((value.start, value.stop, value.step))
    return has_native_semantics(value) or type(value) in (
        float,
        torch.dtype,
        torch.device,
        torch.layout,
        torch.memory_format,
        range,
        type(Ellipsis),
    )


def _has_standard_module_access(model: nn.Module) -> bool:
    """Reject binding accessors and registered descriptors that can run user code.

    Inspect module dictionaries directly. This eligibility check does not invoke
    overridden module traversal methods or require a particular model class.
    """
    names = (
        "__getattribute__",
        "__getattr__",
        "get_submodule",
        "named_modules",
        "named_parameters",
        "named_buffers",
        "modules",
        "parameters",
        "buffers",
        "_named_members",
    )
    pending, seen = [model], set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        cls = type(module)
        if any(getattr(cls, name, None) is not getattr(nn.Module, name) for name in names):
            return False
        state = object.__getattribute__(module, "__dict__")
        if "forward" in state or any(name in state for name in names):
            return False
        bindings = tuple(state.get(name) for name in ("_modules", "_parameters", "_buffers"))
        if any(type(table) is not dict for table in bindings):
            return False
        registered = {name for table in bindings for name in table}
        if any(registered.intersection(vars(base)) for base in cls.__mro__):
            return False
        pending.extend(child for child in bindings[0].values() if child is not None)
    return True
