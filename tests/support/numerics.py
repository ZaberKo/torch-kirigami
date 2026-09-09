"""support / numerics contracts."""

from math import prod

import torch


def _sample(noncontiguous, *, sequence=False):
    shape = (2, 3, 8) if sequence else (2, 6)
    storage = (torch.arange(prod(shape), dtype=torch.float64) / 13).reshape(shape)
    sample = storage[..., ::2]
    if not noncontiguous:
        sample = sample.contiguous()
    assert sample.is_contiguous() != noncontiguous
    return sample.requires_grad_()


def _initialize(model):
    # Distinct signed weights make an accidental permutation numerically visible.
    with torch.no_grad():
        for number, parameter in enumerate(model.parameters(), 1):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype)
            parameter.copy_(((values % 11 - 5) / (number + 7)).reshape(parameter.shape))
    return model


def _assert_value_and_input_gradient(actual, expected, actual_input, reference_input):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    # A nonuniform cotangent also detects permutations hidden by output summation.
    cotangent = (torch.arange(actual.numel(), dtype=actual.dtype) + 1).reshape(actual.shape)
    actual_gradient = torch.autograd.grad(actual, actual_input, cotangent)[0]
    reference_gradient = torch.autograd.grad(expected, reference_input, cotangent)[0]
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-10, atol=1e-10)
