"""The fixed public FX capture pipeline and metadata execution boundary."""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx, nn
from torch.fx.passes.shape_prop import ShapeProp

from .bindings import reference_edits, reference_signature, storage_key
from .configuration import attributes, forward_hook_paths, has_registration_hooks
from .errors import CaptureError
from .operation import TensorFacts
from .operators.effects import named_inplace
from .registry import OperatorRegistry


def tree_map(fn, value):
    """Apply a function to leaves of nested tuple, list, and dict containers."""
    if isinstance(value, tuple):
        return tuple(tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [tree_map(fn, item) for item in value]
    if isinstance(value, dict):
        return {key: tree_map(fn, item) for key, item in value.items()}
    return fn(value)


def tensor_leaves(value):
    """Collect tensor leaves from supported input containers."""
    result = []
    tree_map(lambda item: result.append(item) if isinstance(item, torch.Tensor) else None, value)
    return result


def fingerprint(model):
    """Record detectable structural properties without hashing parameter values.

    Args:
        model: Source model to inspect.

    Returns:
        Registered tensor identities/layouts and recognizable module configuration.
        This is not a complete guard for arbitrary Python or external state.
    """
    tensor_state = tuple(
        (
            kind,
            name,
            id(t),
            tuple(t.shape),
            tuple(t.stride()),
            str(t.dtype),
            str(t.device),
            storage_key(t),
        )
        for kind, entries in (
            ("parameter", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        )
        for name, t in entries
    )

    modules = tuple(
        (
            name,
            id(m),
            type(m),
            attributes(m),
        )
        for name, m in model.named_modules(remove_duplicate=False)
    )
    return tensor_state, modules, forward_hook_paths(model), reference_signature(model)


@contextmanager
def isolated_execution(model, args, kwargs):
    """Isolate example inputs and buffers while preserving supported shared storage.

    Args:
        model: Source model; parameter storage is not copied.
        args: Example positional arguments.
        kwargs: Example keyword arguments.

    Yields:
        Copied args, copied kwargs, and clone-ID-to-original-buffer bindings.

    Raises:
        CaptureError: Inputs cannot be copied or alias parameter storage.

    Notes:
        Restores buffer bindings, training flags, and CPU/initialized CUDA RNG
        state on both success and failure. Arbitrary forward side effects and
        concurrent use of the same model are outside this contract.
    """
    if has_registration_hooks():
        raise CaptureError("Module registration callbacks are unsupported during capture")
    reference_signature(model)
    buffers = [
        (module, name, tensor)
        for module in model.modules()
        for name, tensor in module.named_buffers(recurse=False, remove_duplicate=False)
    ]
    modes = [(module, module.training) for module in model.modules()]
    protected = {storage_key(p) for p in model.parameters() if p.numel()}
    if any(
        storage_key(t) in protected for t in tensor_leaves((args, kwargs, [b[2] for b in buffers]))
    ):
        raise CaptureError(
            "Inputs or buffers alias parameter storage; safe isolation is unsupported"
        )
    try:
        copied_args, copied_kwargs, copies = copy.deepcopy((args, kwargs, [b[2] for b in buffers]))
    except Exception as error:
        raise CaptureError("Cannot safely copy example inputs and registered buffers") from error
    aliases = {
        id(clone): original for (_, _, original), clone in zip(buffers, copies, strict=False)
    }
    edits = reference_edits(
        model, {id(old): new for (_, _, old), new in zip(buffers, copies, strict=True)}
    )
    restored = []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for edit in edits:
                parent, _, name = edit.path.rpartition(".")
                owner = model.get_submodule(parent)
                restored.append((owner, name, getattr(owner, name)))
                object.__setattr__(owner, name, edit.value)
            for (module, name, _), clone in zip(buffers, copies, strict=False):
                module._buffers[name] = clone
            yield copied_args, copied_kwargs, aliases
    finally:
        for owner, name, original in reversed(restored):
            object.__setattr__(owner, name, original)
        for module, name, original in buffers:
            # Restore exact bindings without re-running registration callbacks.
            module._buffers[name] = original
        for module, mode in modes:
            object.__setattr__(module, "training", mode)


class _LeafTracer(fx.Tracer):
    """Extend only the public FX leaf policy and function autowrap configuration."""

    def __init__(self, registry):
        super().__init__(autowrap_functions=tuple(registry.opaque_functions))
        self.registry = registry

    def is_leaf_module(self, module, module_qualified_name):
        """Preserve explicitly registered opaque modules and standard FX leaves."""
        return type(module) in self.registry.opaque_modules or super().is_leaf_module(
            module, module_qualified_name
        )


def _root_graph(model, signature):
    """Represent an opaque root through a single public FX call_module node."""
    graph = fx.Graph()
    args, kwargs = [], {}
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise CaptureError("Opaque root modules with variadic signatures are unsupported")
        node = graph.placeholder(parameter.name)
        if parameter.kind == parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = node
        else:
            args.append(node)
    output = graph.call_module("_root", tuple(args), kwargs)
    graph.output(output)
    return fx.GraphModule({"_root": model}, graph)


def trace_module(model, registry):
    """Use the same public FX leaf policy for capture and configuration validation."""
    tracer = _LeafTracer(registry)
    if tracer.is_leaf_module(model, ""):
        return _root_graph(model, inspect.signature(model.forward))
    graph = tracer.trace(model)
    for node in graph.nodes:
        if node.op == "placeholder":
            # Metadata execution always supplies fully bound isolated arguments.
            # Do not retain defaults or ask FX codegen to embed Tensor defaults.
            node.args = ()
    return fx.GraphModule(model, graph)


def validate_attribute_changes(model, registry, original_graph, updates):
    """Reject attribute edits that change the captured Python computation.

    Trace an isolated configuration copy with proposed attributes, sharing the
    original parameters without allocating compact weights or running ShapeProp.
    A changed constant, operator, edge or output is not justified by old metadata.
    Opaque leaf internals remain governed by their declared operator contracts.
    """
    if not updates:
        return

    def same(left, right):
        if isinstance(left, fx.Node) or isinstance(right, fx.Node):
            return (
                isinstance(left, fx.Node) and isinstance(right, fx.Node) and left.name == right.name
            )
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            return (
                isinstance(left, torch.Tensor)
                and isinstance(right, torch.Tensor)
                and torch.equal(left, right)
            )
        if type(left) is not type(right):
            return False
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(
                same(a, b) for a, b in zip(left, right, strict=True)
            )
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, slice):
            return same((left.start, left.stop, left.step), (right.start, right.stop, right.step))
        return left is right or left == right

    try:
        with isolated_execution(model, (), {}):
            # Copy configuration without invoking Module.__deepcopy__, which may
            # return the source object. Preallocate module shells to preserve shared
            # submodules and ordinary references; registered tensors remain shared.
            shells = {id(m): object.__new__(type(m)) for m in model.modules()}
            memo = {**shells, **{id(t): t for t in (*model.parameters(), *model.buffers())}}
            for module in model.modules():
                object.__setattr__(
                    shells[id(module)], "__dict__", copy.deepcopy(vars(module), memo)
                )
            prepared = shells[id(model)]
            for path, value in updates:
                parent, _, name = path.rpartition(".")
                object.__setattr__(prepared.get_submodule(parent), name, value)
            revised = trace_module(prepared, registry)
            old_nodes, new_nodes = tuple(original_graph.nodes), tuple(revised.graph.nodes)
            if len(old_nodes) != len(new_nodes) or any(
                not same((a.op, a.target, a.args, a.kwargs), (b.op, b.target, b.args, b.kwargs))
                for a, b in zip(old_nodes, new_nodes, strict=False)
            ):
                raise CaptureError(
                    "Attribute updates change captured forward structure or constants"
                )
    except CaptureError:
        raise
    except Exception as error:
        raise CaptureError(
            f"Cannot validate attribute updates against original forward: {error}"
        ) from error


def _reject_parameter_writes(gm, registry):
    """Reject recognized parameter-alias writes before metadata execution."""
    tainted = set()
    params = {id(p) for p in gm.parameters()}
    for node in gm.graph.nodes:
        module = gm.get_submodule(str(node.target)) if node.op == "call_module" else None
        rule = registry.lookup(node, module)
        if rule is not None:
            rule.preflight(node, module)
        if node.op == "get_attr":
            value = gm
            for component in str(node.target).split("."):
                value = getattr(value, component)
            if id(value) in params:
                tainted.add(node)
        inputs = set(node.all_input_nodes)
        mutates = (
            (rule.effects(node, module).mutates_input if rule is not None else False)
            or named_inplace(node)
            or node.kwargs.get("inplace") is True
        )
        if node.op == "call_module":
            mutates = mutates or getattr(gm.get_submodule(str(node.target)), "inplace", False)
        if mutates and inputs & tainted:
            raise CaptureError(f"Parameter/alias write at {node.name}: {node.target}")
        out = node.kwargs.get("out")
        if node.op == "call_function":
            try:
                bound = inspect.signature(node.target).bind_partial(*node.args, **node.kwargs)
            except (TypeError, ValueError):
                pass  # Many native functions expose only keyword-only out=, without a signature.
            else:
                out = bound.arguments.get("out", out)
        if out is not None:
            raise CaptureError(f"out= mutation is unsupported at {node.name}")
        fresh = rule.effects(node, module).fresh_output if rule is not None else False
        if inputs & tainted and not fresh:
            # Conservative over-approximation: no unsafe alias write is assumed harmless.
            tainted.add(node)


class _MetadataPropagator(ShapeProp):
    """Execute ShapeProp while retaining facts instead of intermediate activations."""

    def __init__(self, module, bound):
        super().__init__(module)
        self.bound = bound
        self.facts = {}

    def placeholder(self, target, args, kwargs):
        """Resolve a placeholder from the bound original forward signature."""
        name = target.lstrip("*")
        if name not in self.bound:
            raise CaptureError(f"No bound input for FX placeholder {target}")
        return self.bound[name]

    def run_node(self, node):
        """Execute one node and save only supported metadata leaves."""
        result = super().run_node(node)

        def facts(value):
            if isinstance(value, torch.Tensor):
                storage_key(value)
                if value.numel() == 0:
                    raise CaptureError(
                        f"Zero-element tensor at {node.name}: structural axis intent on empty "
                        "tensors is unsupported; provide nonempty examples"
                    )
                return TensorFacts(
                    tuple(value.shape), tuple(value.stride()), value.dtype, value.device
                )
            if value is None or isinstance(
                value, (bool, int, float, str, torch.dtype, torch.device)
            ):
                return value
            raise CaptureError(f"Unsupported metadata value at {node.name}: {type(value).__name__}")

        self.facts[node] = tree_map(facts, result)
        return result


@dataclass
class CaptureResult:
    """Internal capture result and original-buffer provenance.

    Attributes:
        module: FX GraphModule produced by the fixed capture path.
        facts: Result metadata indexed by FX nodes.
        buffer_aliases: Temporary buffer object IDs mapped to original tensors.
    """

    module: fx.GraphModule
    facts: dict[fx.Node, Any]
    buffer_aliases: dict[int, torch.Tensor]


def capture(
    model: nn.Module, args: tuple, kwargs: dict, registry: OperatorRegistry
) -> CaptureResult:
    """Trace and execute metadata under the library's state-isolation contract.

    Args:
        model: Source module.
        args: Example positional arguments.
        kwargs: Example keyword arguments.
        registry: Local operation and leaf registrations.

    Returns:
        The FX graph, result facts, and source-buffer provenance.

    Raises:
        CaptureError: Argument binding, tracing, write checks, or execution fail.
    """
    hooks = forward_hook_paths(model)
    if hooks:
        raise CaptureError(f"Forward hooks are outside captured semantics: {hooks}")
    signature = inspect.signature(model.forward)
    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError as error:
        raise CaptureError(f"Invalid forward arguments: {error}") from error
    with isolated_execution(model, bound.args, bound.kwargs) as (safe_args, safe_kwargs, aliases):
        try:
            gm = trace_module(model, registry)
            _reject_parameter_writes(gm, registry)
            bound = signature.bind(*safe_args, **safe_kwargs)
            metadata = _MetadataPropagator(gm, bound.arguments)
            metadata.propagate()
        except CaptureError:
            raise
        except Exception as error:
            raise CaptureError(
                f"FX capture/metadata execution failed: {error}. "
                "Tensor-dependent Python control flow is not specialized from examples."
            ) from error
    return CaptureResult(gm, metadata.facts, aliases)
