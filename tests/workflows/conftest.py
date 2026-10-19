"""Keep small workflow models from spawning large CPU thread pools."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def bounded_cpu_threads() -> Iterator[None]:
    """Use one CPU thread per test and restore the caller's setting on exit."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
