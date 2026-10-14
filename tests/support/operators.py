"""Small rule fixtures; production extensions implement OperatorRule directly."""

from torch_kirigami import OperatorRule


def make_rule(
    build_spec,
    *,
    build_recipe=None,
    validate_capture=None,
    tensor_effects=None,
    evaluate_on_meta=False,
    cache_meta_output=False,
):
    """Create a test subclass with explicitly supplied hook implementations."""
    namespace = {"build_spec": staticmethod(build_spec)}
    for name, function in (
        ("build_recipe", build_recipe),
        ("validate_capture", validate_capture),
        ("tensor_effects", tensor_effects),
    ):
        if function is not None:
            namespace[name] = staticmethod(function)
    rule_type = type("TestRule", (OperatorRule,), namespace)
    return rule_type(evaluate_on_meta=evaluate_on_meta, cache_meta_output=cache_meta_output)


def with_rule(registry, target, rule):
    """Build an isolated test registry with one deliberately instrumented rule."""
    result = type(registry)()
    for cls, existing in registry.modules.items():
        result.register(
            cls, rule if cls == target else existing, opaque=cls in registry.opaque_modules
        )
    for function, existing in registry.functions.items():
        result.register(
            function,
            rule if function == target else existing,
            opaque=function in registry.opaque_functions,
        )
    for name, existing in registry.methods.items():
        result.register_method(name, rule if name == target else existing)
    return result
