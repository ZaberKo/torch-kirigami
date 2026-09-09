"""Each registered native entry is exercised with a concrete semantic contract."""

import pytest
import torch
from torch import fx

from tests.support.operator_cases import make_case
from torch_kirigami import OperationContext, OperatorRegistry, Selection, TensorFacts, TensorRef

REGISTRY = OperatorRegistry.default()
ENTRIES = [
    pytest.param(
        kind,
        target,
        id=f"{kind}:{target if isinstance(target, str) else target.__module__ + '.' + target.__name__}",
    )
    for kind, table in (
        ("module", REGISTRY.modules),
        ("function", REGISTRY.functions),
        ("method", REGISTRY.methods),
    )
    for target in table
]


def native_context(kind, target, case):
    graph, refs, metadata, constants, raw = fx.Graph(), {}, {}, {}, {}

    def record(tensor, *, path=None):
        identity = id(tensor)
        if identity not in refs:
            kind = "parameter" if isinstance(tensor, torch.nn.Parameter) else "buffer"
            ref = TensorRef(
                f"native:{len(refs)}",
                tuple(tensor.shape),
                kind if path else "input",
                (path,) if path else (),
            )
            refs[identity] = ref
            metadata[ref.id] = TensorFacts(ref.shape, tensor.stride(), tensor.dtype, tensor.device)
            if tensor.dtype in (torch.int32, torch.int64) and tensor.ndim == 1:
                constants[ref.id] = tuple(tensor.cpu().tolist())
            raw[ref.id] = graph.placeholder(f"arg_{len(raw)}")
        return refs[identity]

    bindings = {}
    if case.module is not None:
        for name, tensor in (*case.module.named_parameters(), *case.module.named_buffers()):
            bindings[name] = record(tensor, path=name)

    def tree(value, *, symbolic=False):
        if isinstance(value, torch.Tensor):
            ref = record(value)
            return raw[ref.id] if symbolic else ref
        if isinstance(value, (tuple, list)):
            return type(value)(tree(v, symbolic=symbolic) for v in value)
        if isinstance(value, dict):
            return {k: tree(v, symbolic=symbolic) for k, v in value.items()}
        return value

    args, kwargs = tree(case.args), tree(case.kwargs)
    raw_args, raw_kwargs = tree(case.args, symbolic=True), tree(case.kwargs, symbolic=True)
    node = graph.create_node(
        f"call_{kind}", "op" if kind == "module" else target, raw_args, raw_kwargs
    )
    REGISTRY.lookup(node, case.module).preflight(node, case.module)
    # Native execution validates the case signature and supplies metadata only.
    # Expected structural axes and indices come exclusively from make_case.
    if kind == "module":
        result = case.module(*case.args, **case.kwargs)
    elif kind == "function":
        result = target(*case.args, **case.kwargs)
    else:
        result = getattr(case.args[0], target)(*case.args[1:], **case.kwargs)

    def output_tree(value):
        if isinstance(value, torch.Tensor):
            ref = TensorRef(f"output:{len(metadata)}", tuple(value.shape))
            metadata[ref.id] = TensorFacts(ref.shape, value.stride(), value.dtype, value.device)
            return ref
        if isinstance(value, (tuple, list)):
            return tuple(output_tree(v) for v in value)
        return value

    context = OperationContext(
        node,
        args,
        kwargs,
        output_tree(result),
        case.module,
        "op" if kind == "module" else None,
        bindings,
        {},
        metadata,
        constants,
    )
    return context, refs


@pytest.mark.parametrize("kind,target", ENTRIES)
def test_native_entry_matches_and_propagates_declared_axis(kind, target, execution_device):
    if kind == "method" and target == "cuda" and not torch.cuda.is_available():
        pytest.skip("Tensor.cuda requires an accessible CUDA build")
    case = make_case(kind, target)
    context, refs = native_context(kind, target, case)
    table = {
        "module": REGISTRY.modules,
        "function": REGISTRY.functions,
        "method": REGISTRY.methods,
    }[kind]
    rule = REGISTRY.lookup(context.node, context.module)
    assert rule is table[target]
    rule.preflight(context.node, context.module)
    spec = rule.analyze(context)
    seed = refs[id(case.seed)].axis(case.axis).select(case.removed)
    if case.expression:
        assert spec.expression is not None
        assert refs[id(case.seed)] in spec.expression.refs or target == "dim"
        return
    selections = {seed.tensor.id: seed}
    # Exercise the registered rule's full relation closure, with expectations
    # specified independently by native API semantics in the case table.
    for _ in range(100):
        changed = False
        for relation in spec.relations:
            for ref in relation.refs:
                source = selections.get(ref.id, Selection(ref))
                for target_selection in relation.propagate(source):
                    before = selections.get(
                        target_selection.tensor.id, Selection(target_selection.tensor)
                    )
                    after = before.union(target_selection)
                    if after != before:
                        selections[after.tensor.id] = after
                        changed = True
        if not changed:
            break
    else:
        pytest.fail("Native relation closure failed to converge")
    assert len(context.outputs) == len(case.output_axes)
    for output, axis, expected in zip(
        context.outputs, case.output_axes, case.output_removed, strict=True
    ):
        selection = selections.get(output.id, Selection(output))
        if axis is None:
            assert not selection
        else:
            assert selection == output.axis(axis).select(expected)
    companion_selections = {}
    for tensor, axis, expected in case.companions:
        ref = refs[id(tensor)]
        companion_selections[ref.id] = companion_selections.get(ref.id, Selection(ref)).union(
            ref.axis(axis).select(expected)
        )
    for ref_id, expected in companion_selections.items():
        assert selections.get(ref_id, Selection(expected.tensor)) == expected
    assert not [
        diagnostic
        for constraint in spec.constraints
        if (diagnostic := constraint.check(selections))
    ]
