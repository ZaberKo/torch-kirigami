"""Prune and restore a small fused grouped-query-attention module.

The custom operator rule describes Q/K/V and output-projection relations,
whole-KV-group candidate units, and head-count/projection attributes. The
registered rule feeds `DependencyGraph.build` and `Pruner` discovery;
`Pruner.prune` then plans, verifies and applies magnitude-ranked removals
under a `ChannelRatio` budget. Finally, `save_checkpoint` and
`load_checkpoint` exercise structural checkpoint restoration against a fresh
module skeleton. This example covers the declared GQA layout and its supported
rewrite; it does not promise arbitrary fused attention or mask semantics.
"""

import io

import torch
from torch import nn
from torch.nn import functional as F

from torch_kirigami import (
    AttributeBinding,
    AxisPort,
    AxisRelation,
    BlockMap,
    CandidateAxis,
    DependencyGraph,
    Operation,
    OperatorRegistry,
    OperatorRule,
    OperatorSpec,
    ShapeExpr,
)
from torch_kirigami.pruning import (
    ChannelRatio,
    Greedy,
    Magnitude,
    Pruner,
    load_checkpoint,
    save_checkpoint,
)


class FusedGQA(nn.Module):
    """A fused projection/attention block with explicit whole-KV-group pruning."""

    def __init__(self) -> None:
        super().__init__()
        self.q_heads, self.kv_heads, self.head_dim = 4, 2, 4
        self.q = nn.Linear(8, 16, bias=False)
        self.k = nn.Linear(8, 8, bias=False)
        self.v = nn.Linear(8, 8, bias=False)
        self.out = nn.Linear(16, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped-query attention to a batch of token sequences."""
        q = self.q(x).reshape(x.size(0), x.size(1), self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return self.out(y.transpose(1, 2).reshape(x.size(0), x.size(1), -1))


class FusedGQARule(OperatorRule):
    """Bind the fused attention structure without custom mutation callbacks."""

    def build_spec(self, operation: Operation) -> OperatorSpec:
        """Declare group linkage and attribute bindings once."""
        x, y = operation.inputs[0], operation.outputs[0]
        q, k, v, out = (operation.bindings[f"{name}.weight"] for name in ("q", "k", "v", "out"))
        head_dim = operation.module.head_dim
        multiplier = operation.module.q_heads // operation.module.kv_heads
        relations = [AxisRelation.equal(x.axis(-1), w.axis(1)) for w in (q, k, v)]
        relations.extend(
            (
                AxisRelation.equal(q.axis(0), out.axis(1)),
                AxisRelation.equal(k.axis(0), v.axis(0)),
                AxisRelation.equal(out.axis(0), y.axis(-1)),
                AxisRelation(
                    AxisPort(k.axis(0)),
                    AxisPort(q.axis(0)),
                    (BlockMap(0, 0, operation.module.kv_heads, head_dim, multiplier * head_dim),),
                ),
            )
        )
        relations.extend(AxisRelation.equal(x.axis(i), y.axis(i)) for i in range(len(x.shape) - 1))
        requirements = []
        for name, tensor in (("q_heads", q), ("kv_heads", k)):
            requirements.append(
                AttributeBinding(
                    f"{operation.module_path}.{name}".lstrip("."),
                    ShapeExpr(
                        "floordiv",
                        args=(
                            ShapeExpr("dimension", (tensor, 0)),
                            ShapeExpr("constant", head_dim),
                        ),
                    ),
                    "Update declared head count",
                )
            )
        for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
            for attr, dim in (("out_features", 0), ("in_features", 1)):
                requirements.append(
                    AttributeBinding(
                        f"{operation.module_path}.{name}.{attr}".lstrip("."),
                        ShapeExpr("dimension", (tensor, dim)),
                        "Update explicitly bound projection dimensions",
                    )
                )
        return OperatorSpec(
            tuple(relations),
            requirements=tuple(requirements),
            candidates=(CandidateAxis(f"{k.paths[0]}:0", k.axis(0), head_dim),),
        )


def main() -> None:
    """Prune whole KV groups and verify checkpoint reconstruction."""
    model = FusedGQA()
    x = torch.randn(2, 3, 8)
    operators = OperatorRegistry.default().register(FusedGQA, FusedGQARule())
    graph = DependencyGraph.build(model, args=(x,), operators=operators)
    pruner = Pruner(model, graph=graph)
    space = pruner.discover_candidates()
    model, result = pruner.prune(space, budget=ChannelRatio(0.5), strategy=Greedy(Magnitude()))
    assert (model.q_heads, model.kv_heads) == (2, 1)
    stream = io.BytesIO()
    save_checkpoint(model, stream)
    stream.seek(0)
    restored = load_checkpoint(FusedGQA(), stream)
    torch.testing.assert_close(model(x), restored(x))
    print(result.plan.explain())


if __name__ == "__main__":
    main()
