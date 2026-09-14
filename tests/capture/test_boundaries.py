"""capture / boundaries contracts."""

import pytest
import torch
from torch import fx, nn
from torch.nn import functional as F

from tests.support.models import Chain
from torch_kirigami import (
    CaptureError,
    DependencyGraph,
)
from torch_kirigami.pruning import (
    PlanningError,
    Pruner,
)


@pytest.mark.parametrize("method", [False, True])
def test_invalid_scalar_repeat_interleave_reports_capture_error(method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 6)

        def forward(self, x):
            scalar = self.fc(x).sum()
            return (
                scalar.repeat_interleave(3, dim=0)
                if method
                else torch.repeat_interleave(scalar, 3, dim=-1)
            )

    model = Model()
    sample = torch.randn(2, 4)
    with pytest.raises(IndexError, match="Dimension out of range"):
        model(sample)
    with pytest.raises(CaptureError, match="repeat_interleave"):
        DependencyGraph.build(model, args=(sample,))


@pytest.mark.parametrize("kind", ["data", "shape", "loop"])
def test_dynamic_python_control_flow_fails(kind):
    class Dynamic(nn.Module):
        def forward(self, x):
            if kind == "data":
                return x if x.sum() > 0 else -x
            if kind == "shape":
                return x if x.shape[0] > 2 else -x
            while x.sum() > 0:
                x = x - 1
            return x

    with pytest.raises(CaptureError):
        DependencyGraph.build(Dynamic(), args=(torch.ones(3, 4),))


def test_native_conv_preserves_fx_version_boundary(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Conv1d(3, 2, 1)
            self.w = nn.Parameter(torch.ones(1, 1, 3))

        def forward(self, x, z):
            c = self.fc(z).size(1)
            return F.conv1d(x, self.w, stride=c, padding=c)

    model = Model()
    args = (torch.ones(1, 1, 3), torch.ones(1, 3, 1))
    try:
        fx.symbolic_trace(model)
    except TypeError:
        with pytest.raises(CaptureError, match="FX capture"):
            DependencyGraph.build(model, args=args)
    else:
        graph = DependencyGraph.build(model, args=args)
        with pytest.raises(PlanningError, match="argument"):
            Pruner(model, graph=graph).plan_remove(
                [graph.parameter("fc.weight").axis(0).select([0])]
            )


@pytest.mark.parametrize("intermediate", [False, True])
def test_zero_element_metadata_rejected(intermediate):
    class Model(Chain):
        def forward(self, x):
            return self.b(self.a(x[:0] if intermediate else x))

    with pytest.raises(CaptureError, match=r"[Zz]ero|empty"):
        DependencyGraph.build(Model(), args=(torch.empty(2 if intermediate else 0, 4),))
