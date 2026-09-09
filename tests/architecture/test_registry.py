"""Registry ownership, exact matching and shared rule callback contracts."""

import pytest
from torch import fx, nn

from torch_kirigami import CallEffects, OperatorRegistry, OperatorRule, OperatorSpec


def custom_identity(value):
    return value


def test_registry_copies_own_tables_and_matches_exact_types():
    class Child(nn.Linear):
        pass

    registry = OperatorRegistry.default()
    original_keys = set(registry.modules), set(registry.functions), set(registry.methods)
    clone = registry.copy()
    rule = OperatorRule(lambda _: OperatorSpec())
    clone.register(Child, rule).register(custom_identity, rule).register_method(
        "custom_method", rule
    )
    assert (set(registry.modules), set(registry.functions), set(registry.methods)) == original_keys
    assert Child in clone.opaque_modules and custom_identity in clone.opaque_functions
    assert Child not in registry.opaque_modules and custom_identity not in registry.opaque_functions
    graph = fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_module("layer", (x,))
    assert registry.lookup(node, Child(2, 2)) is None
    assert clone.lookup(node, Child(2, 2)) is rule
    assert clone.lookup(graph.call_function(custom_identity, (x,)), None) is rule
    assert clone.lookup(graph.call_method("custom_method", (x,)), None) is rule
    assert clone.lookup(x, None) is None
    assert clone.modules[nn.Linear] is registry.modules[nn.Linear]
    for target in (Child, custom_identity):
        with pytest.raises(ValueError, match="already"):
            clone.register(target, rule)
    with pytest.raises(ValueError, match="already"):
        clone.register_method("custom_method", rule)
    with pytest.raises(TypeError):
        clone.register(12, rule)
    with pytest.raises(TypeError):
        clone.register(nn.Identity, object())


def test_rule_callbacks_receive_exact_context_and_defaults_are_readonly():
    context, node, module, lowered = object(), fx.Graph().placeholder("x"), nn.Identity(), object()
    events = []

    def analyze(actual):
        assert actual is context
        events.append("analyze")
        return OperatorSpec()

    def preflight(actual_node, actual_module):
        assert actual_node is node and actual_module is module
        events.append("preflight")

    rule = OperatorRule(
        analyze,
        preflight=preflight,
        lower=lambda value: lowered if value is context else None,
        effects=lambda actual_node, actual_module: CallEffects(fresh_output=True),
    )
    rule.preflight(node, module)
    assert isinstance(rule.analyze(context), OperatorSpec)
    assert rule.lower(context) is lowered
    assert rule.effects(node, module) == CallEffects(fresh_output=True)
    assert events == ["preflight", "analyze"]
    default = OperatorRule(analyze)
    assert default.preflight(node, module) is None
    assert default.lower(context) is None and default.effects(node, module) == CallEffects()
    assert not default.evaluate_on_meta
    with pytest.raises(NotImplementedError):
        OperatorRule().analyze(context)
