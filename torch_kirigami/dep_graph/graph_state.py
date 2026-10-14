"""One internal owner for structural analysis tables, shared by build and queries."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

from torch import fx, nn

from ..settings import config
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


@dataclass
class DependencyGraphState:
    """Own analysis tables populated during construction and then read internally.

    Only the analyzer writes these tables. DependencyGraph accepts completed data
    and exposes tuples, read-only views, or inspection copies. This private record
    retains original model identities, but no capture activations, scratch tables,
    or solver caches. It is not a model checkpoint or a second copy of the graph.
    """

    model: nn.Module
    registry: OperatorRegistry
    fingerprint: tuple[object, ...]
    context: Mapping[str, object]
    id: str = field(default_factory=lambda: uuid4().hex)
    settings_key: tuple[int, ...] = field(default_factory=config.analysis_key)
    fx_graph: fx.Graph | None = None
    capture_signature: tuple[object, ...] = ()
    refs: dict[str, TensorRef] = field(default_factory=dict)
    tensor_facts: dict[str, TensorFacts] = field(default_factory=dict)
    parameters: dict[str, TensorRef] = field(default_factory=dict)
    buffers: dict[str, TensorRef] = field(default_factory=dict)
    expressions: dict[fx.Node, ShapeExpr] = field(default_factory=dict)
    literal_values: dict[str, tuple[int, ...]] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    requirements: list[Requirement] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    calls: list[CallRef] = field(default_factory=list)
    operations: dict[str, OperationContext] = field(default_factory=dict)
    specs: dict[str, OperatorSpec] = field(default_factory=dict)
    constant_refs: tuple[TensorRef, ...] = ()
    interfaces: set[str] = field(default_factory=set)
    constants: tuple[tuple[TensorRef, str], ...] = ()
    complete: bool = False
