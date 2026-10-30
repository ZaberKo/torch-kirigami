"""Shared input preparation and isolated inference execution."""

import copy
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from torch import nn

from ..bindings import ordinary_tensors, storage_key
from ..dep_graph.capture import isolated_execution, tensor_leaves


def move_inputs_to_device(
    value: object, device: torch.device, memo: dict[object, object]
) -> object:
    """Move isolated inputs without changing container types or unchanged references."""
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, torch.Tensor):
        if value.device != device and value.numel():
            key = ("storage", storage_key(value))
            if key in memo:
                raise ValueError(
                    "Move storage-sharing input views to the measurement device together"
                )
            memo[key] = True
        result = value.to(device)
    elif isinstance(value, (tuple, list, dict)):
        items = tuple(value.values()) if isinstance(value, dict) else value
        moved = tuple(move_inputs_to_device(item, device, memo) for item in items)
        # Keep unchanged containers: the isolated model can refer to them too.
        # For changed containers, deepcopy reuses moved children from the memo
        # and preserves types such as named tuples and defaultdict.
        result = (
            value
            if all(old is new for old, new in zip(items, moved, strict=True))
            else copy.deepcopy(value, memo)
        )
    else:
        result = value
    memo[id(value)] = result
    return result


@contextmanager
def isolated_evaluation(
    target: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None,
    device: torch.device | str | None,
) -> Iterator[tuple[tuple[object, ...], dict[str, object], torch.device]]:
    """Prepare an isolated model/input pair on one supported execution device."""
    if not isinstance(target, nn.Module):
        raise TypeError("Measurement requires an nn.Module")
    tensors = (*target.parameters(), *target.buffers())
    device = torch.device(device if device is not None else tensors[0].device if tensors else "cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Measurement supports CPU and CUDA only")
    if device.type == "cpu":
        device = torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        device = torch.device(
            "cuda", device.index if device.index is not None else torch.cuda.current_device()
        )
    if any(t.device != device for t in tensors):
        raise ValueError("Move all model parameters and buffers to the measurement device first")
    args = tuple(input_args) if isinstance(input_args, (tuple, list)) else (input_args,)
    kwargs = input_kwargs or {}
    constants = tuple(ordinary_tensors(target))
    identities = {id(t) for t in constants}
    storages = {storage_key(t) for t in constants if t.numel()}
    if any(
        t.device != device and (id(t) in identities or (t.numel() and storage_key(t) in storages))
        for t in tensor_leaves((args, kwargs))
    ):
        raise ValueError("Move model-shared inputs and ordinary attributes to the device together")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context, isolated_execution(target, args, kwargs) as (args, kwargs, _):
        if any(t.device != device for t in tensor_leaves((args, kwargs))):
            args, kwargs = move_inputs_to_device((args, kwargs), device, {})
        target.eval()
        yield args, kwargs, device
