"""Local operator registration; shared contracts live in operation.py."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import nn

from .operation import OperatorRule
from .operators.defaults import register_defaults


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
            rule: Unified OperatorRule instance implementing structural semantics.

        Returns:
            This registry.

        Raises:
            ValueError: The method already has a rule.
            TypeError: rule is not an OperatorRule instance.
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
