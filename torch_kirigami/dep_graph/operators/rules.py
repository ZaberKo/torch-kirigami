"""Built-in operator rule classes that pair spec builders with tensor effects."""

import operator

import torch
from torch import fx, nn
from torch.nn import functional as F

from ..contracts import ArgumentRef
from ..operation import OperatorRule, TensorEffects, argument
from .attention import addmm, einsum, multihead_attention, sdpa
from .effects import INPUT, native_effects, normalization_effects, to_copy
from .extended import (
    cast,
    channel_operator,
    embedding,
    instance_norm,
    normalize,
    padding,
    prelu,
)
from .indexing import (
    channel_shuffle,
    getitem,
    glu,
    index_select,
    narrow,
    pixel_shuffle,
    repeat,
    select,
    stack,
    tensor_unfold,
    unfold_fold,
)
from .native import (
    _identity,
    batch_norm,
    concatenate,
    convolution,
    getattr_rule,
    group_norm,
    layer_norm,
    linear,
    matmul,
    permute,
    pointwise,
    reduction,
    reshape,
    shape_only,
    softmax,
    split,
)
from .spectral import fourier


class NativeRule(OperatorRule):
    """Base class for built-in rules with meta evaluation and native effects.

    Built-in rules enable meta evaluation with cached outputs. By default they
    derive tensor effects from in-place method names and `inplace` arguments.

    Args:
        fresh: Whether nonmutating calls guarantee independent output storage.
            Calls with native in-place arguments do not receive this guarantee.
    """

    def __init__(self, *, fresh: bool) -> None:
        super().__init__(evaluate_on_meta=True, cache_meta_output=True)
        self.fresh = fresh

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Resolve native mutation arguments and the family's allocation guarantee."""
        return native_effects(node, module, fresh_output=self.fresh)


class IdentityRule(NativeRule):
    """Pass coordinates through `nn.Identity` and return the input Tensor itself."""

    build_spec = staticmethod(_identity)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare the returned Tensor to be the input itself."""
        return TensorEffects(writes_inputs=(), aliases_input=INPUT, returns_input=True)


class AddmmRule(NativeRule):
    """Map `addmm` and `baddbmm` as a matrix product plus a broadcast additive input."""

    build_spec = staticmethod(addmm)


class BatchNormRule(NativeRule):
    """Map batch-norm channels to affine parameters and running statistics."""

    build_spec = staticmethod(batch_norm)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare native running-statistic and counter updates during training."""
        return normalization_effects(node, module, instance=False)


class CastRule(NativeRule):
    """Preserve coordinates through dtype, device, and `Tensor.to` conversions."""

    build_spec = staticmethod(cast)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare that a conversion can share only the receiver's storage.

        `Tensor.to(..., copy=True)` instead guarantees fresh output storage.
        """
        copy = node.target == "to" and to_copy(node.args, node.kwargs) is True
        return TensorEffects(
            writes_inputs=(), outputs_are_fresh=copy, aliases_input=None if copy else INPUT
        )


class ChannelOperatorRule(NativeRule):
    """Map batch and channel axes of pooling, unpooling, and interpolation; spatial axes stay fixed."""

    build_spec = staticmethod(channel_operator)


class ChannelShuffleRule(NativeRule):
    """Map channel shuffle as a fixed permutation with tied group-local removals."""

    build_spec = staticmethod(channel_shuffle)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare read-only inputs and allow a degenerate shuffle to alias its input."""
        return TensorEffects(writes_inputs=(), aliases_input=INPUT)


class ConcatenateRule(NativeRule):
    """Map each concatenated input to its offset region in the output."""

    build_spec = staticmethod(concatenate)


class ConvolutionRule(NativeRule):
    """Map ordinary and transposed convolution channels and groups; spatial and kernel axes stay fixed."""

    build_spec = staticmethod(convolution)


class DropoutRule(NativeRule):
    """Describe dropout coordinates and effects for fixed mode and probability."""

    build_spec = staticmethod(pointwise)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare effects that depend on the dropout mode and probability.

        Inactive `nn.Dropout`, alpha dropout, and direct dropout functions
        return their input Tensor. Active training dropout with `0 < p <= 1`
        allocates a fresh output unless it writes in place. Every other case,
        including the inactive `dropout1d`, `dropout2d`, and `dropout3d`
        wrappers, can alias the input.
        """
        native = module is None and getattr(node.target, "__module__", None) == "torch"
        direct_function = native or node.target in (
            F.dropout,
            F.alpha_dropout,
            F.feature_alpha_dropout,
        )
        alpha_module = type(module) in (nn.AlphaDropout, nn.FeatureAlphaDropout)
        if module is not None:
            training, p = module.training, module.p
        else:
            if native:
                training = argument(node.args, node.kwargs, "train", 2, target=node.target)
            else:
                default = node.target not in (F.alpha_dropout, F.feature_alpha_dropout)
                training = argument(node.args, node.kwargs, "training", 2, default)
            p = argument(node.args, node.kwargs, "p", 1, 0.5, target=node.target)
        inactive = training is False or (type(p) in (int, float) and p == 0)
        if inactive and (type(module) is nn.Dropout or alpha_module or direct_function):
            return TensorEffects(writes_inputs=(), aliases_input=INPUT, returns_input=True)
        active = training is True and type(p) in (int, float) and 0 < p <= 1
        effects = native_effects(
            node, module, fresh_output=active, returns_written_input=not alpha_module
        )
        if effects.outputs_are_fresh:
            return effects
        return TensorEffects(
            writes_inputs=effects.writes_inputs,
            aliases_input=INPUT,
            returns_input=effects.returns_input,
        )


class EinsumRule(NativeRule):
    """Map einsum operands and result through their shared labels."""

    build_spec = staticmethod(einsum)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Distinguish products from single-operand views and reductions.

        Products allocate a result. Single-operand calls receive no allocation
        guarantee because this family includes identity and diagonal views.
        Unrecognized equation formats also receive no allocation guarantee.
        """
        operands = node.args[1:] if node.args and isinstance(node.args[0], str) else ()
        if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
            operands = operands[0]
        return TensorEffects(writes_inputs=(), outputs_are_fresh=len(operands) > 1)


class EmbeddingRule(NativeRule):
    """Map the embedding feature axis; vocabulary rows stay fixed."""

    build_spec = staticmethod(embedding)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare the weight update performed by native `max_norm` renormalization."""
        if node.target is torch.embedding:
            # Unlike F.embedding, this weight-first primitive has no max_norm.
            return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
        max_norm = (
            module.max_norm
            if module is not None
            else argument(node.args, node.kwargs, "max_norm", 3, target=node.target)
        )
        if max_norm is None:
            return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
        if module is not None:
            return TensorEffects(
                writes_inputs=(), outputs_are_fresh=True, mutable_parameters=("weight",)
            )
        weight = ArgumentRef("weight", 1)
        return TensorEffects(
            writes_inputs=(weight,), outputs_are_fresh=True, mutable_parameters=(weight,)
        )


class AttributeReadRule(NativeRule):
    """Map Tensor attribute reads: transposes, `real`/`imag`, and shape queries."""

    build_spec = staticmethod(getattr_rule)


class IndexRule(NativeRule):
    """Map bracket indexing with basic slices, one fixed vector or mask, or container keys."""

    build_spec = staticmethod(getitem)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Guarantee fresh storage for integer or boolean list indices; others may view."""
        index = node.args[1]
        items = index if isinstance(index, tuple) else (index,)
        if any(
            isinstance(item, list)
            and item
            and type(item[0]) in (int, bool)
            and all(type(i) is type(item[0]) for i in item)
            for item in items
        ):
            return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
        return TensorEffects(writes_inputs=(), aliases_input=INPUT)


class GluRule(NativeRule):
    """Tie both GLU input halves to the same output positions."""

    build_spec = staticmethod(glu)


class GroupNormRule(NativeRule):
    """Map group-norm channels to affine parameters and keep the groups balanced."""

    build_spec = staticmethod(group_norm)


class IndexSelectRule(NativeRule):
    """Map `index_select` through a fixed index registered as a buffer."""

    build_spec = staticmethod(index_select)


class InstanceNormRule(NativeRule):
    """Map instance-norm channels to affine parameters and running statistics."""

    build_spec = staticmethod(instance_norm)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare native statistic updates when input statistics are enabled."""
        return normalization_effects(node, module, instance=True)


class LayerNormRule(NativeRule):
    """Map layer-norm and RMS-norm normalized axes to their affine parameters."""

    build_spec = staticmethod(layer_norm)


class LinearRule(NativeRule):
    """Map linear input and output features to the weight and bias axes."""

    build_spec = staticmethod(linear)


class MatmulRule(NativeRule):
    """Map contracting, free, and broadcast batch axes of matrix and vector products."""

    build_spec = staticmethod(matmul)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Treat `@=` as Python's non-mutating fallback when Tensor lacks `__imatmul__`."""
        if node.target is operator.imatmul and not hasattr(torch.Tensor, "__imatmul__"):
            return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
        return super().tensor_effects(node, module)


class MultiheadAttentionRule(NativeRule):
    """Shrink the embedding width of `nn.MultiheadAttention` with a fixed head count."""

    build_spec = staticmethod(multihead_attention)


class NarrowRule(NativeRule):
    """Map a fixed `narrow` interval back to the input."""

    build_spec = staticmethod(narrow)


class NormalizeRule(NativeRule):
    """Preserve coordinates of norm-based normalization over a recomputed domain."""

    build_spec = staticmethod(normalize)


class PaddingRule(NativeRule):
    """Map axes that padding leaves unchanged; padded or cropped axes stay fixed."""

    build_spec = staticmethod(padding)


class PermuteRule(NativeRule):
    """Map transposes, permutations, and moved dimensions."""

    build_spec = staticmethod(permute)


class PixelShuffleRule(NativeRule):
    """Map pixel shuffle and unshuffle channel blocks; spatial axes stay fixed."""

    build_spec = staticmethod(pixel_shuffle)


class PointwiseRule(NativeRule):
    """Map elementwise calls through broadcasting, including scalar dimension arithmetic."""

    build_spec = staticmethod(pointwise)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare that unary plus and in-place calls return their input Tensor."""
        if node.target in (operator.pos, torch.positive, "positive"):
            return TensorEffects(writes_inputs=(), aliases_input=INPUT, returns_input=True)
        return native_effects(node, module, fresh_output=self.fresh, returns_written_input=True)


class PreluRule(NativeRule):
    """Map per-channel PReLU weights; a single shared slope stays fixed."""

    build_spec = staticmethod(prelu)


class ReductionRule(NativeRule):
    """Map the axes that a reduction keeps and record the reduced domain."""

    build_spec = staticmethod(reduction)


class RepeatRule(NativeRule):
    """Map `repeat`, `tile`, `repeat_interleave`, and broadcast expansions."""

    build_spec = staticmethod(repeat)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Let view-producing expansions alias the input; other arguments are read-only."""
        if not self.fresh:
            return TensorEffects(writes_inputs=(), aliases_input=INPUT)
        return super().tensor_effects(node, module)


class ReshapeRule(NativeRule):
    """Map reshape, view, flatten, squeeze, unsqueeze, and unflatten coordinates."""

    build_spec = staticmethod(reshape)


class AttentionRule(NativeRule):
    """Map query, key, value, and mask axes of scaled dot-product attention."""

    build_spec = staticmethod(sdpa)


class ShapeOnlyRule(NativeRule):
    """Describe dimension reads such as `size` and `numel`, which have no tensor relations."""

    build_spec = staticmethod(shape_only)


class SoftmaxRule(NativeRule):
    """Preserve softmax-family coordinates and record the normalized axis."""

    build_spec = staticmethod(softmax)


class SplitRule(NativeRule):
    """Map `split`, `chunk`, and `unbind` outputs to their input partitions."""

    build_spec = staticmethod(split)


class StackRule(NativeRule):
    """Align stacked inputs; the new stack axis stays fixed."""

    build_spec = staticmethod(stack)


class SelectRule(NativeRule):
    """Map a fixed scalar index while retaining independent axes."""

    build_spec = staticmethod(select)


class FourierRule(NativeRule):
    """Preserve independent axes while protecting transformed frequency domains."""

    build_spec = staticmethod(fourier)


class UnfoldFoldRule(NativeRule):
    """Map channel blocks of `nn.Unfold` and `nn.Fold`; spatial and kernel axes stay fixed."""

    build_spec = staticmethod(unfold_fold)


class TensorUnfoldRule(NativeRule):
    """Map nonwindow axes of `Tensor.unfold`, whose output can view the input."""

    build_spec = staticmethod(tensor_unfold)
