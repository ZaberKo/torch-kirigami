"""Explicit elementwise surface coverage independent of registry enumeration."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import DependencyGraph, OperatorRegistry
from torch_kirigami.pruning import PlanningError, Pruner

TORCH_UNARY = (
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "tan",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "erfc",
    "erfinv",
    "exp2",
    "log2",
    "log10",
    "trunc",
    "fix",
    "frac",
    "deg2rad",
    "rad2deg",
    "sinc",
    "signbit",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "logical_not",
    "sgn",
    "nan_to_num",
    "i0",
    "lgamma",
    "digamma",
    "positive",
    "hardshrink",
    "logit",
)
TORCH_BINARY = (
    "atan2",
    "arctan2",
    "copysign",
    "fmod",
    "fmax",
    "fmin",
    "hypot",
    "lerp",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "nextafter",
    "xlogy",
    "isclose",
    "heaviside",
    "float_power",
    "addcdiv",
    "addcmul",
    "bitwise_not",
    "true_divide",
    "rsub",
    "gcd",
    "lcm",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "igamma",
    "igammac",
)
TORCH_DEGREE = ("polygamma", "mvlgamma")
SPECIAL_UNARY = (
    "airy_ai",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "digamma",
    "entr",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "exp2",
    "expit",
    "expm1",
    "gammaln",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "log1p",
    "log_ndtr",
    "logit",
    "modified_bessel_i0",
    "modified_bessel_i1",
    "modified_bessel_k0",
    "modified_bessel_k1",
    "ndtr",
    "ndtri",
    "psi",
    "round",
    "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1",
    "sinc",
    "spherical_bessel_j0",
)
SPECIAL_BINARY = ("gammainc", "gammaincc", "xlog1py", "xlogy", "zeta")
SPECIAL_DEGREE = (
    "chebyshev_polynomial_t",
    "chebyshev_polynomial_u",
    "chebyshev_polynomial_v",
    "chebyshev_polynomial_w",
    "hermite_polynomial_h",
    "hermite_polynomial_he",
    "laguerre_polynomial_l",
    "legendre_polynomial_p",
    "shifted_chebyshev_polynomial_t",
    "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v",
    "shifted_chebyshev_polynomial_w",
    "polygamma",
    "multigammaln",
)
CASTS = ("byte", "char", "short", "cfloat", "cdouble")
INPLACE = (
    "sub_",
    "subtract_",
    "div_",
    "divide_",
    "negative_",
    "neg_",
    "absolute_",
    "abs_",
    "square_",
    "sqrt_",
    "reciprocal_",
    "rsqrt_",
    "exp_",
    "expm1_",
    "log_",
    "log1p_",
    "log2_",
    "log10_",
    "sin_",
    "cos_",
    "tan_",
    "sinh_",
    "cosh_",
    "tanh_",
    "sigmoid_",
    "floor_",
    "ceil_",
    "round_",
    "trunc_",
    "fix_",
    "frac_",
    "erf_",
    "erfc_",
    "erfinv_",
    "acos_",
    "acosh_",
    "asin_",
    "asinh_",
    "atan_",
    "atanh_",
    "arccos_",
    "arccosh_",
    "arcsin_",
    "arcsinh_",
    "arctan_",
    "arctanh_",
    "deg2rad_",
    "rad2deg_",
    "nan_to_num_",
    "sign_",
    "sgn_",
    "sinc_",
    "exp2_",
    "clamp_",
    "clip_",
    "clamp_min_",
    "clamp_max_",
    "floor_divide_",
    "remainder_",
    "fmod_",
    "pow_",
    "multiply_",
    "xlogy_",
    "polygamma_",
    "mvlgamma_",
)
ACTIVATIONS = (
    ("hardshrink", nn.Hardshrink),
    ("logsigmoid", nn.LogSigmoid),
    ("softshrink", nn.Softshrink),
    ("alpha_dropout", nn.AlphaDropout),
    ("feature_alpha_dropout", nn.FeatureAlphaDropout),
    ("rrelu", nn.RReLU),
)


def invoke(namespace, name, x):
    """Supply valid scalar/tensor arguments using each documented public signature."""
    if name in ("acosh", "arccosh", "zeta"):
        x = x + 1.1
    if name in ("gcd", "lcm", "bitwise_left_shift", "bitwise_right_shift"):
        x = (x * 4).int()
    if namespace == "method":

        def target(*args):
            return getattr(x, name)(*args)

        prefix = ()
    else:
        target = getattr(torch.special if namespace == "special" else torch, name)
        prefix = (x,)
    if name in SPECIAL_DEGREE + TORCH_DEGREE:
        if name == "polygamma":
            return target(1) if namespace == "method" else target(1, x)
        return target(*prefix, 1 if name in ("multigammaln", "mvlgamma") else 2)
    if name in ("gcd", "lcm", "bitwise_left_shift", "bitwise_right_shift"):
        return target(*prefix, x + 1)
    if name == "ldexp":
        return target(*prefix, (x * 2).int() + 1)
    if name in ("addcdiv", "addcmul"):
        return target(*prefix, x + 0.3, x + 0.7)
    if name == "lerp":
        return target(*prefix, x + 0.7, 0.3)
    if name == "bitwise_not":
        return x.int().bitwise_not() if namespace == "method" else target(x.int())
    if name in TORCH_BINARY + SPECIAL_BINARY:
        return target(*prefix, x + 0.7)
    return target(*prefix)


CASES = (
    [("torch", name) for name in TORCH_UNARY + TORCH_BINARY + TORCH_DEGREE]
    + [
        ("method", name)
        for name in TORCH_UNARY + TORCH_BINARY + TORCH_DEGREE
        if hasattr(torch.Tensor, name)
    ]
    + [("special", name) for name in SPECIAL_UNARY + SPECIAL_BINARY + SPECIAL_DEGREE]
)


@pytest.mark.parametrize("namespace,name", CASES)
def test_elementwise_surface_public_compaction(namespace, name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            return self.head(invoke(namespace, name, self.pre(x).sigmoid()).float())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,), operators=OperatorRegistry.default())
    assert not graph.diagnostics
    keep = [0, 2, 3, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]).sigmoid()
    expected = F.linear(
        invoke(namespace, name, hidden).float(), dense.head.weight[:, keep], dense.head.bias
    )
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1, 4])]))
    actual = model(sample)
    assert bool(torch.isfinite(expected).all())
    torch.testing.assert_close(actual, expected)
    # Some native special functions lack input derivatives. This checks a
    # supported gradient without inventing derivatives for those PyTorch calls.
    gradient = torch.autograd.grad(actual.sum(), model.head.weight)[0]
    assert bool(torch.isfinite(gradient).all())


@pytest.mark.parametrize("name", CASTS)
def test_additional_cast_methods_public_compaction(name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            return self.head(getattr(self.pre(x), name)().float())

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3)
    graph = DependencyGraph.build(model, args=(sample,), operators=OperatorRegistry.default())
    keep = [0, 2, 3, 4, 5]
    hidden = getattr(F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]), name)().float()
    expected = F.linear(hidden, dense.head.weight[:, keep], dense.head.bias)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(sample), expected)


def invoke_inplace(name, value):
    if name in ("acosh_", "arccosh_"):
        value = value + 1.1
    target = getattr(value, name)
    if name in ("polygamma_", "mvlgamma_"):
        return target(1)
    if name in ("clamp_", "clip_"):
        return target(0.1, 0.8)
    if name == "clamp_min_":
        return target(0.1)
    if name == "clamp_max_":
        return target(0.8)
    if name == "pow_":
        return target(2)
    if name in (
        "sub_",
        "subtract_",
        "div_",
        "divide_",
        "multiply_",
        "floor_divide_",
        "remainder_",
        "fmod_",
        "xlogy_",
    ):
        return target(0.7)
    return target()


@pytest.mark.parametrize("name", INPLACE)
def test_elementwise_inplace_surface_compacts_fresh_producers(name, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)
            with torch.no_grad():
                self.pre.weight.fill_(0.05)
                self.pre.bias.fill_(0.3)

        def forward(self, x):
            return self.head(invoke_inplace(name, self.pre(x)))

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.full((2, 3), 0.3)
    graph = DependencyGraph.build(model, args=(sample,), operators=OperatorRegistry.default())
    assert not graph.diagnostics
    keep = [0, 2, 3, 4, 5]
    hidden = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    expected = F.linear(invoke_inplace(name, hidden), dense.head.weight[:, keep], dense.head.bias)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(sample), expected)


def test_new_inplace_forms_still_reject_unproved_view_writes(execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            return self.head(self.pre(x).detach().sub_(0.3))

    model = Model().eval()
    before = copy.deepcopy(model.state_dict())
    graph = DependencyGraph.build(
        model, args=(torch.randn(2, 3),), operators=OperatorRegistry.default()
    )
    with pytest.raises(PlanningError, match="alias/consumer"):
        Pruner(model, graph=graph).plan_remove([graph.parameter("pre.weight").axis(0).select([1])])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("name,module_type", ACTIVATIONS)
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_activation_forms_use_compact_domain_reference(
    name, module_type, functional, training, execution_device
):
    class Model(nn.Module):
        def __init__(self, width=6):
            super().__init__()
            self.pre, self.head = nn.Linear(3, width), nn.Linear(width, 2)
            self.activation = module_type()

        def forward(self, x):
            value = self.pre(x)
            if functional:
                kwargs = {"training": self.training} if "dropout" in name or name == "rrelu" else {}
                value = getattr(F, name)(value, **kwargs)
            else:
                value = self.activation(value)
            return self.head(value)

    model = Model().train(training)
    reference = Model(5).train(training)
    keep = [0, 2, 3, 4, 5]
    with torch.no_grad():
        reference.pre.weight.copy_(model.pre.weight[keep])
        reference.pre.bias.copy_(model.pre.bias[keep])
        reference.head.weight.copy_(model.head.weight[:, keep])
        reference.head.bias.copy_(model.head.bias)
    sample = torch.randn(3, 3)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if execution_device == "cuda" else None
    graph = DependencyGraph.build(model, args=(sample,), operators=OperatorRegistry.default())
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    # Match random draws on two equally compact models, not a dense zero mask.
    torch.manual_seed(17)
    actual = model(sample)
    torch.manual_seed(17)
    expected = reference(sample)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert model.pre.weight.grad is not None


@pytest.mark.parametrize("name", ["angle", "conj", "conj_physical", "resolve_conj", "resolve_neg"])
@pytest.mark.parametrize("method", [False, True])
def test_complex_elementwise_views_keep_coordinates(name, method, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6, dtype=torch.complex64)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            value = self.pre(x).conj()
            value = getattr(value, name)() if method else getattr(torch, name)(value)
            return self.head(value.real)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3, dtype=torch.complex64)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    keep = [0, 2, 3, 4, 5]
    projected = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep]).conj()
    expected = F.linear(
        getattr(torch, name)(projected).real, dense.head.weight[:, keep], dense.head.bias
    )
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()


@pytest.mark.parametrize("name", ["real", "imag"])
@pytest.mark.parametrize("function", [False, True])
def test_complex_component_views_keep_coordinates(name, function, execution_device):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = nn.Linear(3, 6, dtype=torch.complex64)
            self.head = nn.Linear(6, 2)

        def forward(self, x):
            value = self.pre(x)
            value = getattr(torch, name)(value) if function else getattr(value, name)
            return self.head(value)

    model = Model().eval()
    dense = copy.deepcopy(model)
    sample = torch.randn(2, 3, dtype=torch.complex64)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    keep = [0, 2, 3, 4, 5]
    projected = F.linear(sample, dense.pre.weight[keep], dense.pre.bias[keep])
    expected = F.linear(getattr(projected, name), dense.head.weight[:, keep], dense.head.bias)
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.testing.assert_close(model(sample), expected)
    model(sample).sum().backward()


@pytest.mark.parametrize(
    "name", ["alpha_dropout", "feature_alpha_dropout", "rrelu", "celu", "selu", "threshold"]
)
@pytest.mark.parametrize("training", [False, True])
def test_raw_torch_activation_forms_match_compact_module_reference(
    name, training, execution_device
):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre, self.head = nn.Linear(3, 6), nn.Linear(6, 2)

        def forward(self, x):
            value = self.pre(x)
            if name in ("alpha_dropout", "feature_alpha_dropout"):
                value = getattr(torch, name)(value, 0.5, self.training)
            elif name == "rrelu":
                value = torch.rrelu(value, 0.125, 1 / 3, self.training)
            elif name == "threshold":
                value = torch.threshold(value, 0.25, -0.1)
            else:
                value = getattr(torch, name)(value)
            return self.head(value)

    activation = {
        "alpha_dropout": nn.AlphaDropout(),
        "feature_alpha_dropout": nn.FeatureAlphaDropout(),
        "rrelu": nn.RReLU(),
        "celu": nn.CELU(),
        "selu": nn.SELU(),
        "threshold": nn.Threshold(0.25, -0.1),
    }[name]
    model = Model().train(training)
    reference = nn.Sequential(nn.Linear(3, 5), activation, nn.Linear(5, 2)).train(training)
    keep = [0, 2, 3, 4, 5]
    with torch.no_grad():
        reference[0].weight.copy_(model.pre.weight[keep])
        reference[0].bias.copy_(model.pre.bias[keep])
        reference[2].weight.copy_(model.head.weight[:, keep])
        reference[2].bias.copy_(model.head.bias)
    sample = torch.randn(3, 3)
    graph = DependencyGraph.build(model, args=(sample,))
    assert not graph.diagnostics
    pruner = Pruner(model, graph=graph)
    pruner.apply(pruner.plan_remove([graph.parameter("pre.weight").axis(0).select([1])]))
    torch.manual_seed(19)
    actual = model(sample)
    torch.manual_seed(19)
    expected = reference(sample)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
