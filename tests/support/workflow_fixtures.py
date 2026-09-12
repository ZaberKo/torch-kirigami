"""Small fixtures for independent numerical and persistence regression tests."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from examples.fused_attention import FusedGQA, fused_gqa
from torch_kirigami import (
    AxisPort,
    AxisRelation,
    BlockMap,
    DependencyGraph,
    OperatorRegistry,
    OperatorRule,
    Requirement,
)
from torch_kirigami.pruning import CandidateSpace, load_checkpoint, save_checkpoint
from torch_kirigami.sparsity import ChannelGate, register_gate_operators


class GatedGQA(FusedGQA):
    def __init__(self):
        super().__init__()
        self.gate = ChannelGate(self.q_heads, 1)

    def forward(self, x):
        q = self.q(x).reshape(x.size(0), x.size(1), self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(x.size(0), x.size(1), self.kv_heads, self.head_dim).transpose(1, 2)
        heads = self.gate(F.scaled_dot_product_attention(q, k, v, enable_gqa=True))
        return self.out(heads.transpose(1, 2).reshape(x.size(0), x.size(1), -1))


def gated_gqa_rule(ctx):
    spec = fused_gqa(ctx)
    q, gate, mask = (ctx.binding(path) for path in ("q.weight", "gate.weight", "gate.mask"))
    return replace(
        spec,
        relations=(
            *spec.relations,
            AxisRelation(
                AxisPort(q.axis(0)),
                AxisPort(gate.axis(0)),
                (BlockMap(0, 0, ctx.module.q_heads, ctx.module.head_dim, 1),),
            ),
            AxisRelation.equal(gate.axis(0), mask.axis(0)),
        ),
        requirements=(
            *spec.requirements,
            Requirement(
                "attribute",
                f"{ctx.module_path}.gate.size".lstrip("."),
                (gate,),
                "Compact the explicit head gate",
                (("axis", gate.axis(0)),),
            ),
        ),
    )


class MLP(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self.hidden = nn.Linear(8, 12)
        self.gate = ChannelGate(12, -1) if gated else nn.Identity()
        self.out = nn.Linear(12, 3)

    def forward(self, x):
        return self.out(self.gate(self.hidden(x).relu()))


class ResidualCNN(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.bn = nn.BatchNorm2d(8)
        self.gate = ChannelGate(8, 1) if gated else nn.Identity()
        self.block = nn.Conv2d(8, 8, 3, padding=1, groups=2)
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        y = self.gate(self.bn(self.stem(x)).relu())
        return self.out((y + self.block(y)).relu().mean((2, 3)))


class Transformer(nn.Module):
    def __init__(self, gated=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
        self.attn = GatedGQA() if gated else FusedGQA()
        self.fc1, self.fc2 = nn.Linear(8, 12), nn.Linear(12, 8)
        self.ffn_gate = ChannelGate(12, -1) if gated else nn.Identity()
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(self.ffn_gate(F.gelu(self.fc1(self.norm2(x)))))
        return self.out(x.mean(1))


def registry():
    return (
        register_gate_operators(OperatorRegistry.default())
        .register(FusedGQA, OperatorRule(analyze=fused_gqa))
        .register(GatedGQA, OperatorRule(analyze=gated_gqa_rule))
    )


def build_space(model, x):
    return CandidateSpace(DependencyGraph.build(model, args=(x,), operators=registry()))


def train_steps(model, x, y, optimizer, steps, regularizer=None, strength=0.0):
    for _ in range(steps):
        optimizer.zero_grad()
        task_loss = F.cross_entropy(model(x), y)
        sparse_loss = regularizer() if regularizer is not None else task_loss.new_zeros(())
        (task_loss + strength * sparse_loss).backward()
        optimizer.step()
    return {"task_loss": task_loss.item(), "sparse_loss": sparse_loss.item()}


def task_gradients(model, x, y):
    model.zero_grad()
    F.cross_entropy(model(x), y).backward()


def report(model, plan, metrics):
    print(
        {
            **metrics,
            "target": plan.budget.targets,
            "removed": plan.budget.removed,
            "shortfall": plan.budget.shortfall,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )


def save_training(directory, model, optimizer, *, algorithm, schedule):
    directory.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, directory / "model.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "algorithm": algorithm,
            "schedule": schedule,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        directory / "training.pt",
    )


def restore_training(directory, factory, optimizer_factory, *, device="cpu"):
    model = load_checkpoint(factory(), directory / "model.pt", map_location=device)
    state = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    optimizer = optimizer_factory(model.parameters())
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return model, optimizer, state["algorithm"], state["schedule"]
