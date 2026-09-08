import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--require-cuda",
        action="store_true",
        help="Fail immediately if this run cannot execute CUDA tests.",
    )


def pytest_sessionstart(session):
    if session.config.getoption("--require-cuda") and not torch.cuda.is_available():
        raise pytest.UsageError("--require-cuda needs a CUDA PyTorch build and an accessible GPU")


@pytest.fixture(params=["cpu", "cuda"])
def execution_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA PyTorch build and accessible GPU required")
    # Opt-in tests construct both models and reference tensors in this context.
    # This also avoids mistaking CPU tests under a CUDA wheel for GPU coverage.
    with torch.device(request.param):
        assert torch.empty(()).device.type == request.param
        yield request.param
        if request.param == "cuda":
            torch.cuda.synchronize()
