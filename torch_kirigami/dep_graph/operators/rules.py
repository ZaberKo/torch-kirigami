"""Concrete built-in operator rules, sharing existing semantic implementations."""

from torch import fx, nn

from ..operation import OperatorRule, TensorEffects
from .attention import addmm, einsum, multihead_attention, sdpa
from .effects import native_effects
from .extended import (
    cast,
    channel_operator,
    embedding,
    embedding_preflight,
    instance_norm,
    normalize,
    padding,
    prelu,
)
from .indexing import (
    channel_shuffle,
    chunk,
    glu,
    index_select,
    narrow,
    pixel_shuffle,
    repeat,
    stack,
    unfold_fold,
)
from .native import (
    _identity,
    batch_norm,
    concatenate,
    convolution,
    getattr_rule,
    getitem,
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


class NativeRule(OperatorRule):
    """Share metadata execution and native write declarations, without callbacks."""

    def __init__(self, *, fresh: bool) -> None:
        super().__init__(evaluate_on_meta=True, cache_meta_output=True)
        self.fresh = fresh

    def tensor_effects(self, node: fx.Node, module: nn.Module | None) -> TensorEffects:
        """Resolve native mutation arguments and the family's allocation guarantee."""
        return native_effects(node, module, fresh_output=self.fresh)


class IdentityRule(NativeRule):
    """Provide identity structural semantics."""

    build_spec = staticmethod(_identity)


class AddmmRule(NativeRule):
    """Provide addmm structural semantics."""

    build_spec = staticmethod(addmm)


class BatchNormRule(NativeRule):
    """Provide batch norm structural semantics."""

    build_spec = staticmethod(batch_norm)


class CastRule(NativeRule):
    """Provide cast structural semantics."""

    build_spec = staticmethod(cast)


class ChannelOperatorRule(NativeRule):
    """Provide channel operator structural semantics."""

    build_spec = staticmethod(channel_operator)


class ChannelShuffleRule(NativeRule):
    """Provide channel shuffle structural semantics."""

    build_spec = staticmethod(channel_shuffle)


class ChunkRule(NativeRule):
    """Provide chunk structural semantics."""

    build_spec = staticmethod(chunk)


class ConcatenateRule(NativeRule):
    """Provide concatenate structural semantics."""

    build_spec = staticmethod(concatenate)


class ConvolutionRule(NativeRule):
    """Provide convolution structural semantics."""

    build_spec = staticmethod(convolution)


class EinsumRule(NativeRule):
    """Provide einsum structural semantics."""

    build_spec = staticmethod(einsum)


class EmbeddingRule(NativeRule):
    """Provide embedding structural semantics."""

    build_spec = staticmethod(embedding)
    validate_capture = staticmethod(embedding_preflight)


class AttributeReadRule(NativeRule):
    """Provide getattr rule structural semantics."""

    build_spec = staticmethod(getattr_rule)


class IndexRule(NativeRule):
    """Provide getitem structural semantics."""

    build_spec = staticmethod(getitem)


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


class LayerNormRule(NativeRule):
    """Provide layer norm structural semantics."""

    build_spec = staticmethod(layer_norm)


class LinearRule(NativeRule):
    """Provide linear structural semantics."""

    build_spec = staticmethod(linear)


class MatmulRule(NativeRule):
    """Provide matmul structural semantics."""

    build_spec = staticmethod(matmul)


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


class PreluRule(NativeRule):
    """Provide prelu structural semantics."""

    build_spec = staticmethod(prelu)


class ReductionRule(NativeRule):
    """Provide reduction structural semantics."""

    build_spec = staticmethod(reduction)


class RepeatRule(NativeRule):
    """Provide repeat structural semantics."""

    build_spec = staticmethod(repeat)


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


class UnfoldFoldRule(NativeRule):
    """Provide unfold fold structural semantics."""

    build_spec = staticmethod(unfold_fold)


NATIVE_RULE_TYPES = (
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
    UnfoldFoldRule,
)


def is_native_rule(rule: OperatorRule) -> bool:
    """Recognize exact built-ins; subclasses and instance hook overrides are extensions."""
    return type(rule) in NATIVE_RULE_TYPES and not any(
        name in rule.__dict__
        for name in ("build_spec", "build_recipe", "validate_capture", "tensor_effects")
    )
