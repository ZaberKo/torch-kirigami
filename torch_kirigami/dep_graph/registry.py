"""Local operator registration; shared contracts live in operation.py."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from torch import fx, nn

from .operation import OperatorRule
from .operators.defaults import default_rules


class OperatorRegistry:
    """A local registry matching exact module types, functions, and Tensor methods.

    `DependencyGraph.build` copies the registration tables but shares the rule
    objects. Rules must be deterministic and must not mutate the model or
    shared state.
    """

    def __init__(self) -> None:
        self._modules: dict[type[nn.Module], OperatorRule] = {}
        self._functions: dict[Callable, OperatorRule] = {}
        self._methods: dict[str, OperatorRule] = {}
        self._opaque_modules: set[type[nn.Module]] = set()
        self._opaque_functions: set[Callable] = set()

    @property
    def modules(self) -> Mapping[type[nn.Module], OperatorRule]:
        """Read-only view of rules keyed by exact module type.

        Use `register` to add entries, so duplicate registrations are rejected.
        """
        return MappingProxyType(self._modules)

    @property
    def functions(self) -> Mapping[Callable, OperatorRule]:
        """Read-only view of rules keyed by function object."""
        return MappingProxyType(self._functions)

    @property
    def methods(self) -> Mapping[str, OperatorRule]:
        """Read-only view of rules keyed by Tensor method name."""
        return MappingProxyType(self._methods)

    @property
    def opaque_modules(self) -> frozenset[type[nn.Module]]:
        """Module types that FX capture keeps as leaves."""
        return frozenset(self._opaque_modules)

    @property
    def opaque_functions(self) -> frozenset[Callable]:
        """Function objects that FX capture keeps as leaves."""
        return frozenset(self._opaque_functions)

    @classmethod
    def default(cls) -> OperatorRegistry:
        """Create a fresh registry containing the built-in operator rules.

        Returns:
            A new registry that the caller can extend.
        """
        registry = cls()
        for target, (rule_type, fresh, opaque) in default_rules().items():
            rule = rule_type(fresh=fresh)
            if isinstance(target, str):
                registry.register_method(target, rule)
            else:
                registry.register(target, rule, opaque=opaque)
        return registry

    def copy(self) -> OperatorRegistry:
        """Copy registration tables and leaf sets without cloning rule objects.

        Returns:
            An independent registry that shares this registry's rules.
        """
        result = OperatorRegistry()
        result._modules = self._modules.copy()
        result._functions = self._functions.copy()
        result._methods = self._methods.copy()
        result._opaque_modules = self._opaque_modules.copy()
        result._opaque_functions = self._opaque_functions.copy()
        return result

    def register(
        self, target: type[nn.Module] | Callable, rule: OperatorRule, *, opaque: bool = True
    ) -> OperatorRegistry:
        """Register semantics for an exact module type or function.

        Args:
            target: An `nn.Module` subclass or a function object.
            rule: `OperatorRule` instance implementing structural semantics.
            opaque: Whether FX should preserve the target as a leaf where supported.

        Returns:
            This registry, allowing registrations to be chained.

        Raises:
            TypeError: `rule` is not an `OperatorRule`, or `target` is neither a
                module class nor a callable.
            ValueError: A rule is already registered for `target`.
        """
        if not isinstance(rule, OperatorRule):
            raise TypeError("Register an OperatorRule instance")
        if isinstance(target, type) and issubclass(target, nn.Module):
            if target in self._modules:
                raise ValueError(f"Rule already registered for {target}")
            self._modules[target] = rule
            if opaque:
                self._opaque_modules.add(target)
        elif callable(target):
            if target in self._functions:
                raise ValueError(f"Rule already registered for {target}")
            self._functions[target] = rule
            if opaque:
                self._opaque_functions.add(target)
        else:
            raise TypeError("Register an exact nn.Module type or function object")
        return self

    def register_method(self, name: str, rule: OperatorRule) -> OperatorRegistry:
        """Register semantics for a captured Tensor method.

        Args:
            name: Method name, such as `reshape`.
            rule: `OperatorRule` instance implementing structural semantics.

        Returns:
            This registry, allowing registrations to be chained.

        Raises:
            TypeError: `rule` is not an `OperatorRule` instance.
            ValueError: The method already has a rule.
        """
        if not isinstance(rule, OperatorRule):
            raise TypeError("Register an OperatorRule instance")
        if name in self._methods:
            raise ValueError(f"Rule already registered for Tensor.{name}")
        self._methods[name] = rule
        return self

    def lookup(self, node: fx.Node, module: nn.Module | None) -> OperatorRule | None:
        """Return the exact matching rule for a captured operation.

        Args:
            node: Captured FX node.
            module: Called module for `call_module` nodes; ignored otherwise.

        Returns:
            The rule registered for the module's exact type, the function
            object, or the Tensor method name, or None if there is no match or
            the node is not a call.
        """
        if node.op == "call_module":
            return self._modules.get(type(module))
        if node.op == "call_function":
            return self._functions.get(node.target)
        if node.op == "call_method":
            return self._methods.get(node.target)
        return None
