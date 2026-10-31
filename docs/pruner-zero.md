# Fixed Pruner-Zero metric

`EvolvedMetric` uses the best expression published by
[Pruner-Zero (ICML 2024)](https://arxiv.org/abs/2406.02924).
It does not run evolution, search expressions, or load executable formulas.
The reference expression is available in the authors'
[`data/best_tree.json`](https://github.com/pprp/Pruner-Zero/blob/main/data/best_tree.json).

For each complete original matrix or convolution kernel, define

```text
g = abs(parameter.grad)
importance = parameter² * (g - min(g)) / (max(g) - min(g))
```

The extrema are per tensor, across every dimension. Constant gradient magnitudes
produce zero scores. This explicit guard differs from the reference's unguarded
division. Computation promotes values before multiplication. Nonfinite values,
complex weights, missing gradients and non-dense gradients fail explicitly.

## Structural adaptation

The original work ranks individual LLM weights. Here, each candidate receives
the sum of importance over all newly removed matrix/kernel weight regions in
its complete dependency closure. Shared parameters and overlapping regions
count once. Biases, buffers and one-dimensional normalization scales do not
contribute to scores, but structural planning still removes their coupled
coordinates. A new proposal with no associated matrix/kernel weights fails.

Conditional scoring subtracts already accepted coordinates after determining
the joint dependency closure. Normalization always uses the complete original
weight tensor, including accepted positions. It never renormalizes individual
slices. Scores are independent of candidate batching. Cache entries are bounded
axis vectors tied to current weight/gradient identities and mutation versions.
Irregular regions use full-tensor normalization as well. Evaluation can allocate
temporary float64 tensors as large as a weight tensor.

```python
from torch_kirigami import DependencyGraph
from torch_kirigami.pruning import EvolvedMetric, Greedy, ParameterBudget, Pruner

# model and example are supplied by the caller.
graph = DependencyGraph.build(model, args=(example,))
pruner = Pruner(model, graph=graph)
space = pruner.discover_candidates(targets=targets)
# Calibrate dense unscaled parameter.grad values before planning.
plan = pruner.plan(
    space,
    budget=ParameterBudget.from_ratio(model, pruning_ratio=0.15),
    strategy=Greedy(EvolvedMetric()),
)
model, result = pruner.apply(plan)
```

The metric owns no loss, optimizer or data loader. The caller owns gradient
collection, reduction and AMP unscaling. The example collects the gradient of
sample-mean cross-entropy over training calibration batches, in evaluation mode,
without optimizer updates. Unequal batch sizes are sample weighted. Gradients
are cleared on calibration failure; mixed module modes are restored.

## Workflow and recovery

Run from the repository root after installing the existing workflow requirements:

```bash
python examples/workflows/pruner_zero_pruning.py \
  --model vit_b_16 --device cuda --data_dir /path/to/imagenet-snapshot \
  --pruning_ratio 0.15 --calibration_batches 8 --train_batch_size 16 \
  --val_batch_size 32 --finetune_epochs 10 --lr 0.0001 \
  --output runs/pruner_zero_vit
```

Supported entries are the existing ResNet-18/34/50 and ViT-B/16/B/32 catalog.
ResNet pruning targets block-internal convolution widths. ViT targets MLP hidden
widths. The residual interfaces and classifier are protected. The ratio is a
whole-model parameter reduction, not the paper's per-weight sparsity. Granularity
can overshoot the requested reduction; the planner reports failure if the target
cannot be reached in the eligible scope.

The workflow records pretrained, pruned and optionally recovered validation
accuracy, parameter counts, supported MACs/FLOPs and latency in `metrics.json`.
`--finetune_epochs 0` skips recovery. Positive values use ordinary labeled CE,
AdamW and an epoch cosine schedule. A new optimizer is created after physical
pruning. There is no teacher or distillation objective. Preprocessing uses the
existing deterministic weight transforms; full paper training is not reproduced.

`pruned.pt` preserves the immediate compact model. `model.pt` contains the final
compact model; the workflow loads it into a fresh model and compares outputs.
`training.pt` stores optimizer/scheduler state and configuration, not a complete
resume command. Large ImageNet runs and CUDA performance must be validated on
the intended data and hardware; small offline lifecycle tests do not establish
published accuracy.
