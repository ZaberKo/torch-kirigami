import pytest
import torch
from torch import nn

from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import (
    ParameterGroup,
)
from torch_kirigami.sparsity import (
    scale_groups_,
    set_group_norms_,
    zero_groups_,
)


def test_region_union_scaling_norm_targets_and_atomic_rejection(execution_device):
    model = nn.Linear(2, 3, bias=False).to(device=execution_device, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[3, 4], [0, 2], [0, 0]]))
    graph = DependencyGraph.build(
        model, args=(torch.ones(1, 2, device=execution_device, dtype=torch.float64),)
    )
    ref = graph.parameter("weight")
    rows = [ParameterGroup(graph, (ref.axis(0).select([i]),)) for i in range(3)]
    column = ParameterGroup(graph, (ref.axis(1).select([0]),))
    model.weight.grad = torch.ones_like(model.weight)
    scale_groups_((rows[0], column), 0.5)
    torch.testing.assert_close(model.weight, model.weight.new_tensor([[1.5, 2], [0, 2], [0, 0]]))
    torch.testing.assert_close(model.weight.grad, torch.ones_like(model.weight))
    set_group_norms_(rows[:2], [5, 1])
    torch.testing.assert_close(model.weight, model.weight.new_tensor([[3, 4], [0, 1], [0, 0]]))
    before = model.weight.detach().clone()
    for groups, targets in [
        ((rows[0], column), (1, 1)),
        ((rows[0], rows[2]), (1, 2)),
        ((rows[0], rows[0]), (1, 2)),
    ]:
        with pytest.raises(ValueError):
            set_group_norms_(groups, targets)
        torch.testing.assert_close(model.weight, before)
    zero_groups_((rows[0], rows[0]))
    assert not model.weight[0].any()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    model(torch.ones(1, 2, device=execution_device, dtype=torch.float64)).sum().backward()
    optimizer.step()
    assert model.weight[0].ne(0).all()  # Soft zeroing explicitly permits regrowth.


@pytest.mark.parametrize("operation", ["scale", "zero", "norm"])
@pytest.mark.parametrize("include_second", [False, True])
def test_parameter_operations_reject_storage_aliases_atomically(
    operation, include_second, execution_device
):
    class SharedStorage(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], device=execution_device))
            self.b = nn.Parameter(self.a.detach())

        def forward(self, x):
            return x * self.a + x * self.b

    model = SharedStorage()
    graph = DependencyGraph.build(model, args=(torch.ones(1, 3, device=execution_device),))
    group = ParameterGroup(graph, (graph.parameter("a").axis(0).select([0]),))
    groups = (
        (group,)
        if not include_second
        else (group, ParameterGroup(graph, (graph.parameter("b").axis(0).select([1]),)))
    )
    before = model.a.detach().clone()
    version = model.a._version
    model.a.grad = torch.ones_like(model.a)
    with pytest.raises(ValueError, match="sharing storage"):
        if operation == "scale":
            scale_groups_(groups, 2)
        elif operation == "zero":
            zero_groups_(groups)
        else:
            set_group_norms_(groups, [2] * len(groups))
    torch.testing.assert_close(model.a, before)
    torch.testing.assert_close(model.b, before)
    torch.testing.assert_close(model.a.grad, torch.ones_like(model.a))
    assert model.a._version == version


@pytest.mark.parametrize(
    "dtype,value", [(torch.float16, 1e-7), (torch.float32, 1e-40), (torch.float64, 1e-320)]
)
def test_norm_projection_of_nonzero_subnormals(dtype, value, execution_device):
    model = nn.Linear(1, 2, bias=False, device=execution_device, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(model.weight.new_tensor([[value], [-value]]))
    graph = DependencyGraph.build(
        model, args=(torch.ones(1, 1, device=execution_device, dtype=dtype),)
    )
    group = ParameterGroup(graph, (graph.parameter("weight").axis(0).select([0, 1]),))
    set_group_norms_((group,), (1.0,))
    reference = torch.tensor([[2**-0.5], [-(2**-0.5)]], device=execution_device, dtype=dtype)
    torch.testing.assert_close(model.weight, reference)
    with torch.no_grad():
        model.weight.fill_(2)
    before = model.weight.detach().clone()
    version = model.weight._version
    with pytest.raises(ValueError, match="nonfinite"):
        scale_groups_((group,), float("1.7976931348623157e308"))
    torch.testing.assert_close(model.weight, before)
    assert model.weight._version == version
