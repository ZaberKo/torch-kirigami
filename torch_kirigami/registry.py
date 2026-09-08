"""Unified local operator definitions shared by analysis and physical lowering."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from torch import fx, nn

from .contracts import Constraint, Requirement, ShapeExpr
from .relations import Relation
from .selection import AxisRef, TensorRef

if TYPE_CHECKING:
    from .capture import TensorFacts


def tensors(value):
    """Yield tensor references from nested tuples, lists, and dictionaries.

    Args:
        value: A normalized argument or result tree.

    Yields:
        TensorRef leaves in deterministic container traversal order.
    """
    if isinstance(value, TensorRef):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


@dataclass(frozen=True)
class OperationContext:
    """Inputs and capture facts provided to an operation semantics rule.

    Attributes:
        node: Original FX operation node.
        args: Positional arguments with tensor values replaced by references.
        kwargs: Keyword arguments with tensor values replaced by references.
        output: Result tree containing references and scalar metadata.
        module: Called module, or None for functions and methods.
        module_path: An original module alias; an empty string denotes the root.
        bindings: Module-local parameter and buffer paths mapped to references.
        expressions: Supported shape expressions available before this operation.
        metadata: Captured tensor facts indexed by reference ID.
        constants: Small captured integer tensors, used only with declared value guards.
    """

    node: fx.Node
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any
    module: nn.Module | None
    module_path: str | None
    bindings: dict[str, TensorRef]
    expressions: dict[fx.Node, ShapeExpr]
    metadata: Mapping[str, TensorFacts]
    constants: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def inputs(self):
        """Return tensor arguments in flattened container traversal order."""
        return tuple(tensors((self.args, self.kwargs)))

    @property
    def outputs(self):
        """Return tensor results in flattened container traversal order."""
        return tuple(tensors(self.output))

    def argument(self, name: str, position: int, default=None):
        """Resolve an argument supplied by keyword or position.

        Args:
            name: Canonical keyword name.
            position: Corresponding positional argument index.
            default: Value to use if neither spelling was supplied.

        Returns:
            The normalized argument, preferring an explicitly supplied keyword.
        """
        return self.kwargs.get(name, self.args[position] if position < len(self.args) else default)

    def parameter(self, name: str) -> TensorRef | None:
        """Return a module-local parameter or buffer binding, or None."""
        return self.bindings.get(name)


@dataclass(frozen=True)
class OperatorSpec:
    """Structural facts emitted by a pure operation rule.

    Attributes:
        relations: Deterministic index correspondences to propagate.
        constraints: Conditions that must hold for a proposed change.
        requirements: Descriptions of later model edits; never surgery callbacks.
        candidates: Logical axes and block sizes for optional default discovery.
        layouts: Shared physical partitions used by generic compaction lowering.
        contract: Optional original-call argument/layout behavior.
        expression: Scalar dimension provenance for a dimension-producing call.
        constants: Integer references whose values must remain unchanged.
    """

    relations: tuple[Relation, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    candidates: tuple[CandidateAxis, ...] = ()
    layouts: tuple[Any, ...] = ()
    contract: Any = None
    expression: ShapeExpr | None = None
    constants: tuple[TensorRef, ...] = ()


@dataclass(frozen=True)
class CandidateAxis:
    """Declare a logical budget axis and its default contiguous removal blocks.

    The stable key names a structural domain, independently of FX call identity.
    Seeds use the logical axis; registered relations map them to parameter regions.
    """

    key: str
    axis: AxisRef
    block: int = 1


class OperatorRule:
    """One definition for capture checks, structural analysis, and lowering.

    Args:
        analyze: Pure callback producing an OperatorSpec from capture metadata.
        lower: Optional pure callback producing declarative rewrite descriptions.
        preflight: Optional callback accepting an FX node and its called module
            before metadata execution. Raise CaptureError for recognized writes.
        effects: Optional callback with the same arguments, returning CallEffects.
        evaluate: Whether the native operation is safe to evaluate on meta tensors.
            Third-party callbacks default to declared output facts instead.
    """

    def __init__(self, analyze=None, *, lower=None, preflight=None, effects=None, evaluate=False):
        self._analyze = analyze
        self._lower = lower
        self._preflight = preflight
        self._effects = effects
        self.evaluate = evaluate

    def analyze(self, context):
        """Produce shared structural facts without modifying model state."""
        if self._analyze is None:
            raise NotImplementedError("Implement OperatorRule.analyze")
        return self._analyze(context)

    def preflight(self, node, module):
        """Check effects using call arguments and configuration, before ShapeProp."""
        if self._preflight is not None:
            self._preflight(node, module)

    def effects(self, node, module):
        """Describe writes/copies without requiring output metadata."""
        return self._effects(node, module) if self._effects is not None else CallEffects()

    def lower(self, context):
        """Lower shared descriptors or delegate to a pure extension callback."""
        if self._lower is not None:
            return self._lower(context)
        from .pruning.rewrite import lower_spec

        return lower_spec(context)


@dataclass(frozen=True)
class CallEffects:
    """Effects relevant to parameter isolation and downstream alias safety."""

    mutates_input: bool = False
    fresh_output: bool = False


@dataclass
class OperatorRegistry:
    """A local registry matching exact module types, functions, and Tensor methods.

    Builds copy the registry tables, while retaining the same rule callables.
    Rules must be deterministic and must not mutate the model or shared state.
    """

    modules: dict[type[nn.Module], OperatorRule] = field(default_factory=dict)
    functions: dict[Callable, OperatorRule] = field(default_factory=dict)
    methods: dict[str, OperatorRule] = field(default_factory=dict)
    opaque_modules: set[type[nn.Module]] = field(default_factory=set)
    opaque_functions: set[Callable] = field(default_factory=set)

    @classmethod
    def default(cls):
        """Create a fresh registry containing the built-in operation rules."""
        from .operators.native import register_defaults

        registry = cls()
        register_defaults(registry)
        return registry

    def copy(self):
        """Copy registration tables and leaf sets without cloning rule callables."""
        return OperatorRegistry(
            dict(self.modules),
            dict(self.functions),
            dict(self.methods),
            set(self.opaque_modules),
            set(self.opaque_functions),
        )

    def register(self, target, rule: OperatorRule, *, opaque: bool = True):
        """Register semantics for an exact module type or function.

        Args:
            target: An nn.Module class or a function object.
            rule: Unified OperatorRule instance.
            opaque: Whether FX should preserve the target as a leaf where supported.

        Returns:
            This registry, allowing registrations to be chained.

        Raises:
            TypeError: target is neither a module class nor a callable.
            ValueError: A rule is already registered for target.
        """
        if not isinstance(rule, OperatorRule):
            raise TypeError("Register an OperatorRule instance")
        if isinstance(target, type) and issubclass(target, nn.Module):
            table = self.modules
            opaque_set = self.opaque_modules if opaque else None
        elif callable(target):
            table = self.functions
            opaque_set = self.opaque_functions if opaque else None
        else:
            raise TypeError("Register an exact nn.Module type or function object")
        if target in table:
            raise ValueError(f"Rule already registered for {target}")
        table[target] = rule
        if opaque_set is not None:
            opaque_set.add(target)
        return self

    def register_method(self, name: str, rule: OperatorRule):
        """Register semantics for a captured Tensor method.

        Args:
            name: Method name, such as reshape.
            rule: Callable implementing the operation's structural semantics.

        Returns:
            This registry.

        Raises:
            ValueError: The method already has a rule.
        """
        if not isinstance(rule, OperatorRule):
            raise TypeError("Register an OperatorRule instance")
        if name in self.methods:
            raise ValueError(f"Rule already registered for Tensor.{name}")
        self.methods[name] = rule
        return self

    def lookup(self, node, module):
        """Return the exact matching rule for a captured operation, or None."""
        if node.op == "call_module":
            return self.modules.get(type(module))
        if node.op == "call_function":
            return self.functions.get(node.target)
        if node.op == "call_method":
            return self.methods.get(node.target)
        return None
