"""One opaque GQA definition supplies analysis, candidates, lowering, and persistence."""

import io

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AxisPort,
    AxisRelation,
    BlockMap,
    CandidateAxis,
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    Requirement,
    ShapeExpr,
)
from torch_kirigami.pruning import ChannelRatio, Magnitude, Pruner, load_checkpoint, save_checkpoint


class FusedGQA(nn.Module):
    """A fused projection/attention block with explicit whole-KV-group pruning."""

    def __init__(self):
        super().__init__()
        self.q_heads, self.kv_heads, self.head_dim = 4, 2, 4
        self.q = nn.Linear(8, 16, bias=False)
        self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 8, bias=False)
        self.out = nn.Linear(16, 8, bias=False)

    def forward(self, x):
        q = self.q(x).reshape(x.size(0), x.size(1), self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return self.out(y.transpose(1, 2).reshape(x.size(0), x.size(1), -1))


def fused_gqa(ctx):
    """Declare group linkage and attribute bindings once; no rewrite/save callback."""
    x, y = ctx.inputs[0], ctx.outputs[0]
    q, k, v, out = (ctx.binding(f"{name}.weight") for name in ("q", "k", "v", "out"))
    d = ctx.module.head_dim
    multiplier = ctx.module.q_heads // ctx.module.kv_heads
    relations = [AxisRelation.equal(x.axis(-1), w.axis(1)) for w in (q, k, v)]
    relations.extend(
        (
            AxisRelation.equal(q.axis(0), out.axis(1)),
            AxisRelation.equal(k.axis(0), v.axis(0)),
            AxisRelation.equal(out.axis(0), y.axis(-1)),
            AxisRelation(
                AxisPort(k.axis(0)),
                AxisPort(q.axis(0)),
                (BlockMap(0, 0, ctx.module.kv_heads, d, multiplier * d),),
            ),
        )
    )
    relations.extend(AxisRelation.equal(x.axis(i), y.axis(i)) for i in range(len(x.shape) - 1))
    requirements = []
    for name, tensor in (("q_heads", q), ("kv_heads", k)):
        requirements.append(
            Requirement(
                "attribute",
                f"{ctx.module_path}.{name}".lstrip("."),
                (tensor,),
                "Update declared head count",
                (
                    (
                        "expression",
                        ShapeExpr(
                            "floordiv",
                            args=(
                                ShapeExpr("dimension", (tensor, 0)),
                                ShapeExpr("constant", d),
                            ),
                        ),
                    ),
                ),
            )
        )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        for attr, dim in (("out_features", 0), ("in_features", 1)):
            requirements.append(
                Requirement(
                    "attribute",
                    f"{ctx.module_path}.{name}.{attr}".lstrip("."),
                    (tensor,),
                    "Update explicitly bound projection dimensions",
                    (("axis", tensor.axis(dim)),),
                )
            )
    return OperatorSpec(
        tuple(relations),
        requirements=tuple(requirements),
        candidates=(CandidateAxis(f"{k.paths[0]}:0", k.axis(0), d),),
    )


def main():
    model = FusedGQA()
    x = torch.randn(2, 3, 8)
    operators = OperatorRegistry.default().register(FusedGQA, OperatorRule(fused_gqa))
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    model, result = Pruner(model, graph=graph).prune(metric=Magnitude(), budget=ChannelRatio(0.5))
    assert (model.q_heads, model.kv_heads) == (2, 1)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(FusedGQA(), stream)
    torch.testing.assert_close(model(x), restored(x))
    print(result.plan.explain())


if __name__ == "__main__":
    main()
