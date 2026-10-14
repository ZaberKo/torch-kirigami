"""Registry ownership, exact matching and shared rule callback contracts."""

import pytest
from torch import fx, nn

from tests.support.operators import make_rule
from torch_kirigami import OperatorRegistry, OperatorRule, OperatorSpec, TensorEffects


def custom_identity(value):
    return value


def test_registry_copies_own_tables_and_matches_exact_types():
    class Child(nn.Linear):
        pass

    registry = OperatorRegistry.default()
    original_keys = set(registry.modules), set(registry.functions), set(registry.methods)
    clone = registry.copy()
    rule = make_rule(lambda _: OperatorSpec())
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


def test_rule_subclass_contract_and_readonly_defaults():
    operation, spec, impact = object(), OperatorSpec(), object()
    node, module = fx.Graph().placeholder("x"), nn.Identity()
    events = []

    class Rule(OperatorRule):
        def validate_capture(self, actual_node, actual_module):
            assert actual_node is node and actual_module is module
            events.append("capture")

        def build_spec(self, actual):
            assert actual is operation
            events.append("spec")
            return spec

        def build_recipe(self, actual, actual_spec, actual_impact):
            assert (actual, actual_spec, actual_impact) == (operation, spec, impact)
            events.append("recipe")
            return None

    rule = Rule()
    rule.validate_capture(node, module)
    assert rule.build_spec(operation) is spec
    assert rule.build_recipe(operation, spec, impact) is None
    assert rule.tensor_effects(node, module) == TensorEffects(writes_inputs=())
    assert events == ["capture", "spec", "recipe"]
    default = make_rule(lambda _: spec)
    assert default.validate_capture(node, module) is None
    assert default.build_recipe(operation, spec, impact) is None
    assert not default.evaluate_on_meta and not default.cache_meta_output
    with pytest.raises(TypeError, match="abstract"):
        OperatorRule()


def test_registry_views_cannot_bypass_registration():
    registry = OperatorRegistry.default()
    rule = registry.modules[nn.Linear]
    for table, key in (
        (registry.modules, nn.Linear),
        (registry.functions, custom_identity),
        (registry.methods, "view"),
    ):
        with pytest.raises(TypeError):
            table[key] = rule
