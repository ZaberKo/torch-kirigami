"""Completed analysis records shared by graph construction and queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import fx, nn

from .contracts import Constraint, Diagnostic, Requirement, ShapeExpr
from .operation import CallRef, OperationContext, OperatorSpec, TensorFacts
from .registry import OperatorRegistry
from .relations import Relation
from .selection import TensorRef


def resolve_attribute(module: nn.Module, path: str) -> object:
    """Resolve a dotted attribute path on the source model or captured module."""
    value: object = module
    for component in path.split("."):
        value = getattr(value, component)
    return value


def validate_reference(refs: Mapping[str, TensorRef], tensor: TensorRef) -> None:
    """Reject foreign or altered references without inspecting model state."""
    if not isinstance(tensor, TensorRef) or refs.get(tensor.id) != tensor:
        raise ValueError("Tensor belongs to another graph or has altered metadata")


@dataclass(frozen=True)
class GraphData:
    """Store a completed analysis, without construction scratch data or query caches.

    The builder supplies tuples and read-only mappings. Model objects, FX nodes,
    and extension callbacks retain their identities; they are not deep-frozen.
    The public graph controls inspection copies and model freshness checks.
    """

    id: str
    model: nn.Module
    registry: OperatorRegistry
    fingerprint: tuple[object, ...]
    context: Mapping[str, object]
    fx_graph: fx.Graph
    capture_signature: tuple[object, ...]
    refs: Mapping[str, TensorRef]
    tensor_facts: Mapping[str, TensorFacts]
    parameters: Mapping[str, TensorRef]
    buffers: Mapping[str, TensorRef]
    expressions: Mapping[fx.Node, ShapeExpr]
    literal_values: Mapping[str, tuple[int, ...]]
    relations: tuple[Relation, ...]
    constraints: tuple[Constraint, ...]
    requirements: tuple[Requirement, ...]
    diagnostics: tuple[Diagnostic, ...]
    calls: tuple[CallRef, ...]
    operations: Mapping[str, OperationContext]
    specs: Mapping[str, OperatorSpec]
    constant_refs: tuple[TensorRef, ...]
    interfaces: frozenset[str]
    constants: tuple[tuple[TensorRef, str], ...]
