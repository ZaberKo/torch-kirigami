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

from .errors import CaptureError
from .registry import OperatorRegistry


@dataclass(frozen=True)
class TensorFacts:
    """Tensor metadata detached from activation storage.

    Attributes:
        shape: Logical dimensions observed during metadata execution.
        stride: Strides measured in tensor elements.
        dtype: PyTorch element type.
        device: Device on which the sample tensor was observed.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


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


def storage_key(tensor):
    """Identify nonempty dense storage for conservative alias checks.

    Args:
        tensor: Tensor whose storage should be inspected.

    Returns:
        A device/pointer pair, or None for an empty tensor.

    Raises:
        CaptureError: The tensor is quantized or does not have strided layout.
    """
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise CaptureError("Only dense, non-quantized strided tensors are supported")
    return (str(tensor.device), tensor.untyped_storage().data_ptr()) if tensor.numel() else None


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

    def stable(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (tuple, list)) and all(
            x is None or isinstance(x, (bool, int, float, str)) for x in value
        ):
            return tuple(value)
        return None

    modules = tuple(
        (
            name,
            id(m),
            type(m),
            tuple(
                (key, stable(value))
                for key, value in sorted(vars(m).items())
                if key != "_kirigami_structure" and stable(value) is not None
            ),
        )
        for name, m in model.named_modules(remove_duplicate=False)
    )
    return tensor_state, modules


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
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            for (module, name, _), clone in zip(buffers, copies, strict=False):
                setattr(module, name, clone)
            yield copied_args, copied_kwargs, aliases
    finally:
        for module, name, original in buffers:
            setattr(module, name, original)
        for module, mode in modes:
            module.training = mode


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
        if parameter.default is inspect.Parameter.empty:
            node = graph.placeholder(parameter.name)
        else:
            node = graph.placeholder(parameter.name, default_value=parameter.default)
        if parameter.kind == parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = node
        else:
            args.append(node)
    output = graph.call_module("_root", tuple(args), kwargs)
    graph.output(output)
    return fx.GraphModule({"_root": model}, graph)


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
        target_name = getattr(node.target, "__name__", str(node.target))
        mutates = (
            (rule.effects(node, module).mutates_input if rule is not None else False)
            or (node.op in ("call_method", "call_function") and target_name.endswith("_"))
            or node.kwargs.get("inplace") is True
        )
        if node.op == "call_module":
            mutates = mutates or getattr(gm.get_submodule(str(node.target)), "inplace", False)
        if mutates and inputs & tainted:
            raise CaptureError(f"Parameter/alias write at {node.name}: {node.target}")
        if "out" in node.kwargs:
            raise CaptureError(f"out= mutation is unsupported at {node.name}")
        fresh = rule.effects(node, module).fresh_output if rule is not None else False
        if inputs & tainted and not fresh:
            # Conservative over-approximation: no unsafe alias write is assumed harmless.
            tainted.add(node)


class _Metadata(ShapeProp):
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
class Captured:
    """Internal capture result and original-buffer provenance.

    Attributes:
        module: FX GraphModule produced by the fixed capture path.
        facts: Result metadata indexed by FX nodes.
        root_leaf: Whether the root was wrapped as a single opaque call.
        buffer_aliases: Temporary buffer object IDs mapped to original tensors.
    """

    module: fx.GraphModule
    facts: dict[fx.Node, Any]
    root_leaf: bool
    buffer_aliases: dict[int, torch.Tensor]


def capture(model: nn.Module, args: tuple, kwargs: dict, registry: OperatorRegistry) -> Captured:
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
    signature = inspect.signature(model.forward)
    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        raise CaptureError(f"Invalid forward arguments: {error}") from error
    with isolated_execution(model, args, kwargs) as (safe_args, safe_kwargs, aliases):
        tracer = _LeafTracer(registry)
        root_leaf = tracer.is_leaf_module(model, "")
        try:
            gm = (
                _root_graph(model, signature)
                if root_leaf
                else fx.GraphModule(model, tracer.trace(model))
            )
            _reject_parameter_writes(gm, registry)
            bound = signature.bind(*safe_args, **safe_kwargs)
            bound.apply_defaults()
            metadata = _Metadata(gm, bound.arguments)
            metadata.propagate()
        except CaptureError:
            raise
        except Exception as error:
            raise CaptureError(
                f"FX capture/metadata execution failed: {error}. "
                "Tensor-dependent Python control flow is not specialized from examples."
            ) from error
    return Captured(gm, metadata.facts, root_leaf, aliases)
