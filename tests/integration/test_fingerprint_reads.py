"""Freshness reads are shared within a check and discarded before the next one."""

from collections import Counter

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph, StaleGraphError, bindings, capture, configuration
from torch_kirigami.pruning import ExecutionError, Pruner


class SlottedBase(nn.Module):
    """Provide an inherited slot whose state must remain guarded."""

    __slots__ = ("base_option",)


class SlottedModel(SlottedBase):
    """Exercise registered aliases and mutable configuration/reference slots."""

    __slots__ = ("optional", "references", "settings")

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 6)
        self.last = nn.Linear(6, 2)
        self.alias = self.first
        self.base_option = 1
        self.settings = {"items": [1, {"enabled": True}]}
        self.references = [self.first.weight, {"bias": self.first.bias}]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a chain with one removable hidden width."""
        return self.last(self.first(x).relu())


def test_fingerprint_reads_each_object_and_type_once_per_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated checks must refresh shared reads without multiplying alias work."""
    model = SlottedModel()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    reads: Counter[int] = Counter()
    types: Counter[type] = Counter()
    original_read = configuration.object_attributes
    original_slots = configuration.slot_names

    def read(value: object, *, slots: tuple[str, ...] | None = None) -> dict[str, object]:
        """Count dictionary/slot reads across all fingerprint consumers."""
        reads[id(value)] += 1
        return original_read(value, slots=slots)

    def slots(cls: type) -> tuple[str, ...]:
        """Count descriptor inspection for each exact module type."""
        types[cls] += 1
        return original_slots(cls)

    # Count every consumer, so a fallback attribute read cannot hide duplication.
    monkeypatch.setattr(capture, "object_attributes", read)
    monkeypatch.setattr(configuration, "object_attributes", read)
    monkeypatch.setattr(bindings, "object_attributes", read)
    monkeypatch.setattr(capture, "slot_names", slots)
    monkeypatch.setattr(configuration, "slot_names", slots)
    for check in (1, 2):
        graph.validate()
        assert reads == {id(module): check for module in model.modules()}
        assert types == {type(module): check for module in model.modules()}


def unchanged_output(
    module: nn.Module, args: tuple[object, ...], output: torch.Tensor
) -> torch.Tensor:
    """Leave execution unchanged while adding an unmodeled forward hook."""
    return output


def change_configuration(model: SlottedModel, change: str) -> None:
    """Mutate one independent premise of the captured model."""
    if change == "inherited_slot":
        model.base_option = 2
    elif change == "nested_slot":
        model.settings["items"][1]["enabled"] = False
    elif change == "initialize_slot":
        model.optional = 3
    elif change == "delete_slot":
        del model.base_option
    elif change == "slot_reference":
        model.references[1]["bias"] = model.last.bias
    elif change == "mode":
        model.eval()
    elif change == "buffer":
        model.register_buffer("extra", torch.ones(1))
    elif change == "module_alias":
        model.alias_again = model.first
    elif change == "requires_grad":
        model.first.weight.requires_grad_(False)
    elif change == "hook":
        model.first.register_forward_hook(unchanged_output)
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "inherited_slot",
        "nested_slot",
        "initialize_slot",
        "delete_slot",
        "slot_reference",
        "mode",
        "buffer",
        "module_alias",
        "requires_grad",
        "hook",
    ],
)
def test_freshness_rechecks_mutations_and_rejects_without_edits(
    change: str, execution_device: str
) -> None:
    """Live queries and portable apply must reject changed premises atomically."""
    model = SlottedModel()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    pruner = Pruner(model, graph=graph)
    selection = graph.parameter("first.weight").axis(0).select([1])
    plan = pruner.plan_remove([selection])
    change_configuration(model, change)
    parameters = tuple(model.parameters())
    before = tuple(parameter.detach().clone() for parameter in parameters)

    with pytest.raises(StaleGraphError):
        graph.validate()
    with pytest.raises(StaleGraphError):
        pruner.plan_remove([selection])
    with pytest.raises(ExecutionError):
        Pruner(model).apply(plan)

    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    for parameter, expected in zip(parameters, before, strict=True):
        torch.testing.assert_close(parameter, expected)
    assert model.first.out_features == 6 and model.last.in_features == 6


def test_freshness_allows_weight_updates_and_preserves_slot_aliases(execution_device: str) -> None:
    """Weight updates remain valid, and physical pruning preserves alias bindings."""
    model = SlottedModel()
    x = torch.randn(2, 4)
    graph = DependencyGraph.build(model, args=(x,))
    with torch.no_grad():
        model.first.weight.add_(0.25)
    graph.validate()
    pruner = Pruner(model, graph=graph)
    plan = pruner.plan_remove([graph.parameter("first.weight").axis(0).select([1, 4])])
    keep = [0, 2, 3, 5]
    expected = F.linear(
        F.linear(x, model.first.weight[keep], model.first.bias[keep]).relu(),
        model.last.weight[:, keep],
        model.last.bias,
    )
    pruner.apply(plan)
    assert model.alias is model.first
    assert model.references[0] is model.first.weight
    assert model.references[1]["bias"] is model.first.bias
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()
    assert model.first.weight.grad is not None


def test_new_fingerprint_reinspects_slot_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Descriptor changes between validations cannot reuse an obsolete type readout."""
    model = SlottedModel()
    graph = DependencyGraph.build(model, args=(torch.randn(2, 4),))
    # Replacing a descriptor with a class constant removes that instance field
    # from the supported slot readout. A cross-check type cache would miss it.
    monkeypatch.setattr(SlottedBase, "base_option", 9)
    with pytest.raises(StaleGraphError):
        graph.validate()
