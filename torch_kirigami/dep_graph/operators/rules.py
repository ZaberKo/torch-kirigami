"""Concrete built-in operator rules, sharing existing semantic implementations."""

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
    chunk,
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
    """Share metadata execution and native write declarations.

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
    """Provide identity structural semantics."""

    build_spec = staticmethod(_identity)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare the returned Tensor to be the input itself."""
        return TensorEffects(writes_inputs=(), aliases_input=INPUT, returns_input=True)


class AddmmRule(NativeRule):
    """Provide addmm structural semantics."""

    build_spec = staticmethod(addmm)


class BatchNormRule(NativeRule):
    """Provide batch norm structural semantics."""

    build_spec = staticmethod(batch_norm)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare native running-statistic and counter updates during training."""
        return normalization_effects(node, module, instance=False)


class CastRule(NativeRule):
    """Provide cast structural semantics."""

    build_spec = staticmethod(cast)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """A conversion can share only the receiver's storage."""
        copy = node.target == "to" and to_copy(node.args, node.kwargs) is True
        return TensorEffects(
            writes_inputs=(), outputs_are_fresh=copy, aliases_input=None if copy else INPUT
        )


class ChannelOperatorRule(NativeRule):
    """Provide channel operator structural semantics."""

    build_spec = staticmethod(channel_operator)


class ChannelShuffleRule(NativeRule):
    """Provide channel shuffle structural semantics."""

    build_spec = staticmethod(channel_shuffle)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Allow a degenerate shuffle to share its input storage."""
        return TensorEffects(writes_inputs=(), aliases_input=INPUT)


class ChunkRule(NativeRule):
    """Provide chunk structural semantics."""

    build_spec = staticmethod(chunk)


class ConcatenateRule(NativeRule):
    """Provide concatenate structural semantics."""

    build_spec = staticmethod(concatenate)


class ConvolutionRule(NativeRule):
    """Provide convolution structural semantics."""

    build_spec = staticmethod(convolution)


class DropoutRule(NativeRule):
    """Describe dropout coordinates and effects for fixed mode and probability."""

    build_spec = staticmethod(pointwise)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Account for inactive dropout and the unbatched spatial wrappers."""
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
    """Provide einsum structural semantics."""

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
    """Provide embedding structural semantics."""

    build_spec = staticmethod(embedding)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare native max_norm renormalization without changing its behavior."""
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
    """Provide getattr rule structural semantics."""

    build_spec = staticmethod(getattr_rule)


class IndexRule(NativeRule):
    """Provide getitem structural semantics."""

    build_spec = staticmethod(getitem)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Separate native list-index copies from possible views."""
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
    """Provide glu structural semantics."""

    build_spec = staticmethod(glu)


class GroupNormRule(NativeRule):
    """Provide group norm structural semantics."""

    build_spec = staticmethod(group_norm)


class IndexSelectRule(NativeRule):
    """Provide index select structural semantics."""

    build_spec = staticmethod(index_select)


class InstanceNormRule(NativeRule):
    """Provide instance norm structural semantics."""

    build_spec = staticmethod(instance_norm)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Declare native statistic updates when input statistics are enabled."""
        return normalization_effects(node, module, instance=True)


class LayerNormRule(NativeRule):
    """Provide layer norm structural semantics."""

    build_spec = staticmethod(layer_norm)


class LinearRule(NativeRule):
    """Provide linear structural semantics."""

    build_spec = staticmethod(linear)


class MatmulRule(NativeRule):
    """Provide matmul structural semantics."""

    build_spec = staticmethod(matmul)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Use Python's non-mutating matmul fallback for native Tensors."""
        if node.target is operator.imatmul and not hasattr(torch.Tensor, "__imatmul__"):
            return TensorEffects(writes_inputs=(), outputs_are_fresh=True)
        return super().tensor_effects(node, module)


class MultiheadAttentionRule(NativeRule):
    """Provide multihead attention structural semantics."""

    build_spec = staticmethod(multihead_attention)


class NarrowRule(NativeRule):
    """Provide narrow structural semantics."""

    build_spec = staticmethod(narrow)


class NormalizeRule(NativeRule):
    """Provide normalize structural semantics."""

    build_spec = staticmethod(normalize)


class PaddingRule(NativeRule):
    """Provide padding structural semantics."""

    build_spec = staticmethod(padding)


class PermuteRule(NativeRule):
    """Provide permute structural semantics."""

    build_spec = staticmethod(permute)


class PixelShuffleRule(NativeRule):
    """Provide pixel shuffle structural semantics."""

    build_spec = staticmethod(pixel_shuffle)


class PointwiseRule(NativeRule):
    """Provide pointwise structural semantics."""

    build_spec = staticmethod(pointwise)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Record exact Tensor returns for unary plus and value-only writes."""
        if node.target in (operator.pos, torch.positive, "positive"):
            return TensorEffects(writes_inputs=(), aliases_input=INPUT, returns_input=True)
        return native_effects(node, module, fresh_output=self.fresh, returns_written_input=True)


class PreluRule(NativeRule):
    """Provide prelu structural semantics."""

    build_spec = staticmethod(prelu)


class ReductionRule(NativeRule):
    """Provide reduction structural semantics."""

    build_spec = staticmethod(reduction)


class RepeatRule(NativeRule):
    """Provide repeat structural semantics."""

    build_spec = staticmethod(repeat)

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Expand can borrow the input's storage. Its shape template is read-only."""
        if not self.fresh:
            return TensorEffects(writes_inputs=(), aliases_input=INPUT)
        return super().tensor_effects(node, module)


class ReshapeRule(NativeRule):
    """Provide reshape structural semantics."""

    build_spec = staticmethod(reshape)


class AttentionRule(NativeRule):
    """Provide sdpa structural semantics."""

    build_spec = staticmethod(sdpa)


class ShapeOnlyRule(NativeRule):
    """Provide shape only structural semantics."""

    build_spec = staticmethod(shape_only)


class SoftmaxRule(NativeRule):
    """Provide softmax structural semantics."""

    build_spec = staticmethod(softmax)


class SplitRule(NativeRule):
    """Provide split structural semantics."""

    build_spec = staticmethod(split)


class StackRule(NativeRule):
    """Provide stack structural semantics."""

    build_spec = staticmethod(stack)


class SelectRule(NativeRule):
    """Map a fixed scalar index while retaining independent axes."""

    build_spec = staticmethod(select)


class FourierRule(NativeRule):
    """Preserve independent axes while protecting transformed frequency domains."""

    build_spec = staticmethod(fourier)


class UnfoldFoldRule(NativeRule):
    """Provide unfold fold structural semantics."""

    build_spec = staticmethod(unfold_fold)


class TensorUnfoldRule(NativeRule):
    """Map nonwindow axes of Tensor.unfold and retain its view allocation contract."""

    build_spec = staticmethod(tensor_unfold)


NATIVE_RULE_TYPES = (
    FourierRule,
    AddmmRule,
    AttentionRule,
    AttributeReadRule,
    BatchNormRule,
    CastRule,
    ChannelOperatorRule,
    ChannelShuffleRule,
    ChunkRule,
    ConcatenateRule,
    ConvolutionRule,
    DropoutRule,
    EinsumRule,
    EmbeddingRule,
    GluRule,
    GroupNormRule,
    IdentityRule,
    IndexRule,
    IndexSelectRule,
    InstanceNormRule,
    LayerNormRule,
    LinearRule,
    MatmulRule,
    MultiheadAttentionRule,
    NarrowRule,
    NormalizeRule,
    PaddingRule,
    PermuteRule,
    PixelShuffleRule,
    PointwiseRule,
    PreluRule,
    ReductionRule,
    RepeatRule,
    ReshapeRule,
    ShapeOnlyRule,
    SoftmaxRule,
    SplitRule,
    StackRule,
    SelectRule,
    TensorUnfoldRule,
    UnfoldFoldRule,
)


def is_native_rule(rule: OperatorRule) -> bool:
    """Recognize exact built-ins; subclasses and instance hook overrides are extensions."""
    return type(rule) in NATIVE_RULE_TYPES and not any(
        name in rule.__dict__
        for name in ("build_spec", "build_recipe", "validate_capture", "tensor_effects")
    )
