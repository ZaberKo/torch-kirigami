"""Capture and execution consume one allocation/effect contract."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    ArgumentRef,
    AxisRelation,
    CaptureError,
    DependencyGraph,
    Operation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    TensorEffects,
)
from torch_kirigami.pruning import PlanningError, Pruner


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("normalization", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_fresh_producers_support_inplace_consumers(
    spatial, functional, normalization, training, execution_device
):
    affine = (lambda a, b: nn.Conv2d(a, b, 1)) if spatial else nn.Linear
    norm = nn.BatchNorm2d if spatial else nn.BatchNorm1d
    call = F.conv2d if spatial else F.linear

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = affine(4, 6)
            self.norm = norm(6) if normalization else nn.Identity()
            self.last = affine(6, 2)

        def forward(self, x):
            y = call(x, self.first.weight, self.first.bias) if functional else self.first(x)
            if normalization:
                y = self.norm(y)
            return self.last(F.relu(y, inplace=True))

    model = Model().train(training)
    reference = copy.deepcopy(model)
    kept = [0, 2, 3, 4, 5]
    reference.first = affine(4, 5)
    reference.last = affine(5, 2)
    if normalization:
        reference.norm = norm(5)
    reference.train(training)
    with torch.no_grad():
        reference.first.weight.copy_(model.first.weight[kept])
        reference.first.bias.copy_(model.first.bias[kept])
        reference.last.weight.copy_(model.last.weight[:, kept])
        reference.last.bias.copy_(model.last.bias)
    x = torch.randn((3, 4, 3, 3) if spatial else (3, 4))
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph).apply(
        Pruner(model, graph=graph).plan_remove(
            [graph.parameter("first.weight").axis(0).select([1])]
        )
    )
    y, expected = model(x), reference(x)
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(model.first.weight.grad, reference.first.weight.grad)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(("vector", "bias"), [(False, False), (True, False), (False, True)])
def test_functional_linear_vector_and_scalar_bias(batched, vector, bias, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn((4,) if vector else (3, 4)))
            self.bias = nn.Parameter(torch.tensor(0.5)) if bias else None

        def forward(self, x):
            return F.linear(x, self.w, self.bias)

    model = Model()
    x = torch.randn((2, 4) if batched else (4,))
    expected = F.linear(x[..., [0, 2, 3]], model.w[..., [0, 2, 3]], model.bias)
    graph = DependencyGraph.build(model, args=(x,))
    Pruner(model, graph=graph, preserve_io=False).apply(
        Pruner(model, graph=graph, preserve_io=False).plan_remove(
            [graph.parameter("w").axis(-1).select([1])]
        )
    )
    torch.testing.assert_close(model(x[..., [0, 2, 3]]), expected)


@pytest.mark.parametrize("branch", [False, True])
def test_folded_conv_identity_chain_checks_earlier_uses(
    branch: bool, execution_device: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Conv2d(3, 6, 1)
            self.bn = nn.BatchNorm2d(6)
            self.identity = nn.Identity()
            self.relu = nn.ReLU(inplace=True)
            self.head = nn.Conv2d(6, 2, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.first(x)
            other = y.clone() if branch else 0
            return self.head(self.relu(self.identity(self.bn(y))) + other)

    model = Model().to(execution_device).eval()
    x = torch.randn(2, 3, 4, 4, device=execution_device)
    before = model(x)
    model.first = nn.utils.fuse_conv_bn_eval(model.first, model.bn)
    model.bn = nn.Identity()
    torch.testing.assert_close(model(x), before)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    request = [graph.parameter("first.weight").axis(0).select([1, 4])]
    if branch:
        with pytest.raises(PlanningError, match="'first' also feeds earlier consumer 'clone'"):
            pruner.plan_remove(request)
        assert model.first.out_channels == 6
        return
    keep = [0, 2, 3, 5]
    expected = F.conv2d(
        F.conv2d(x, model.first.weight[keep], model.first.bias[keep]).relu(),
        model.head.weight[:, keep],
        model.head.bias,
    )
    pruner.apply(pruner.plan_remove(request))
    torch.testing.assert_close(model(x), expected)
    model(x).sum().backward()


@pytest.mark.parametrize(
    "form",
    [
        "positive",
        "dropout-eval",
        "dropout-zero",
        "dropout-all",
        "native-dropout-eval",
        "native-dropout-all",
        "copy",
        "positional-copy",
        "reference-no-copy",
        "device-no-copy",
    ],
)
def test_value_write_after_proven_identity_or_copy(form: str, execution_device: str) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 6)
            self.head = nn.Linear(6, 2)
            self.reference = nn.Parameter(torch.ones(1))
            self.dropout = nn.Dropout(p=0 if form == "dropout-zero" else 0.5, inplace=True)
            self.dropout.train(form == "dropout-zero")

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.first(x)
            if form == "positive":
                y = torch.positive(+y).positive()
            elif form in ("dropout-eval", "dropout-zero"):
                y = self.dropout(y)
            elif form == "dropout-all":
                y = F.dropout(y, p=1, training=True)
            elif form in ("native-dropout-eval", "native-dropout-all"):
                y = torch.dropout(input=y, p=1.0, train=form == "native-dropout-all")
            elif form == "copy":
                y = y.to(self.reference, copy=True)
            elif form == "positional-copy":
                y = y.to(self.reference, False, True)
            elif form == "reference-no-copy":
                y = y.to(self.reference)
            else:
                y = y.to(self.reference.device, torch.float32, True, False)
            width = y.size(-1)
            y = y.add_(0.5)
            y.relu_()
            return self.head(y.reshape(y.size(0), width))

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    keep = [0, 2, 3, 5]
    hidden = F.linear(x, model.first.weight[keep], model.first.bias[keep])
    if form in ("dropout-all", "native-dropout-all"):
        hidden = torch.zeros_like(hidden)
    expected = F.linear((hidden + 0.5).relu(), model.head.weight[:, keep], model.head.bias)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    request = [graph.parameter("first.weight").axis(0).select([1, 4])]
    if form in ("reference-no-copy", "device-no-copy"):
        # The reference supplies type or device. Capture can proceed, but
        # planning cannot assume the conversion allocated independent storage.
        with pytest.raises(PlanningError, match="in-place alias/consumer"):
            pruner.plan_remove(request)
        assert model.first.out_features == 6
        return
    pruner.apply(pruner.plan_remove(request))
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(model.reference, torch.ones_like(model.reference))
    model(x).sum().backward()


@pytest.mark.parametrize(
    "name",
    [
        "dropout",
        "dropout_",
        "alpha_dropout",
        "alpha_dropout_",
        "feature_dropout",
        "feature_dropout_",
        "feature_alpha_dropout",
        "feature_alpha_dropout_",
    ],
)
def test_native_dropout_train_keyword_cannot_hide_parameter_write(
    name: str, execution_device: str
) -> None:
    dropout = getattr(torch, name)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[[-1.0, 2.0]]]), requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = dropout(input=self.weight, p=0.5, train=False)
            return x + y.relu_()

    model = Model().to(execution_device)
    sample = torch.zeros_like(model.weight)
    original = model.weight.clone()
    with pytest.raises(CaptureError, match="Parameter/alias write"):
        DependencyGraph.build(model, args=(sample,))
    torch.testing.assert_close(model.weight, original)


def test_custom_rule_declares_exact_input_return(execution_device: str) -> None:
    class PassThrough(nn.Module):
        def forward(self, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
            return value

    class Rule(OperatorRule):
        def build_spec(self, operation: Operation) -> OperatorSpec:
            value = operation.kwargs["value"]
            output = operation.outputs[0]
            return OperatorSpec(
                relations=tuple(
                    AxisRelation.equal(value.axis(i), output.axis(i))
                    for i in range(len(value.shape))
                ),
            )

        def tensor_effects(self, node: torch.fx.Node, module: nn.Module | None) -> TensorEffects:
            return TensorEffects(
                writes_inputs=(), aliases_input=ArgumentRef("value", 0), returns_input=True
            )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(3, 4)
            self.head = nn.Linear(4, 2)
            self.reference = nn.Parameter(torch.ones(1))
            self.passthrough = PassThrough()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.passthrough(value=self.first(x), reference=self.reference)
            return self.head(y.relu_())

    model = Model().to(execution_device)
    x = torch.randn(2, 3, device=execution_device)
    keep = [0, 2, 3]
    expected = F.linear(
        F.linear(x, model.first.weight[keep], model.first.bias[keep]).relu(),
        model.head.weight[:, keep],
        model.head.bias,
    )
    registry = OperatorRegistry.default().register(PassThrough, Rule())
    graph = DependencyGraph.build(model, args=(x,), operators=registry)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("first.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("function", [False, True])
def test_shuffle_cannot_hide_weight_mutation(function: bool, execution_device: str) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(-8.0, 8.0).reshape(1, 4, 2, 2), requires_grad=False
            )
            self.shuffle = nn.ChannelShuffle(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = torch.channel_shuffle(self.weight, 1) if function else self.shuffle(self.weight)
            return x + y.relu_()

    model = Model().to(execution_device)
    x = torch.zeros_like(model.weight)
    original = model.weight.detach().clone()
    with pytest.raises(CaptureError, match="Parameter/alias write"):
        DependencyGraph.build(model, args=(x,))
    torch.testing.assert_close(model.weight, original)


@pytest.mark.parametrize("clone", [False, True])
def test_shuffle_that_can_alias_after_compaction_requires_proof(
    clone: bool, execution_device: str
) -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Conv2d(3, 4, 1)
            self.head = nn.Conv2d(4, 1, 1)
            self.other = nn.Conv2d(4, 1, 1)
            self.shuffle = nn.ChannelShuffle(2)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            y = self.first(x).contiguous()
            z = self.shuffle(y)
            if clone:
                z = z.clone()
            return self.head(z.relu_()), self.other(y)

    model = Model().to(execution_device)
    x = torch.randn(2, 3, 3, 3, device=execution_device)
    graph = DependencyGraph.build(model, args=(x,))
    pruner = Pruner(model, graph=graph)
    request = [graph.parameter("first.weight").axis(0).select([1, 3])]
    if not clone:
        with pytest.raises(PlanningError, match="'shuffle' does not guarantee fresh storage"):
            pruner.plan_remove(request)
        assert model.first.out_channels == 4
        return
    y = F.conv2d(x, model.first.weight[[0, 2]], model.first.bias[[0, 2]])
    expected = (
        F.conv2d(y.relu(), model.head.weight[:, :2], model.head.bias),
        F.conv2d(y, model.other.weight[:, [0, 2]], model.other.bias),
    )
    pruner.apply(pruner.plan_remove(request))
    torch.testing.assert_close(model(x), expected)
