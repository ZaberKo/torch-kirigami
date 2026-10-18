# Structural pruning workflow examples

These eleven examples use ImageNet classification to demonstrate structural pruning, evaluation, and checkpoint restoration.
Each entry owns its CLI and training stages.
The examples implement selected research ideas, with explicit changes to scope and training.
They do not claim the papers' published accuracy.

Three modules provide shared infrastructure:

- `imagenet_data.py` loads local data and evaluates accuracy.
- `imagenet_models.py` loads official weights and supplies ViT adapters.
- `model_metrics.py` measures complexity and inference latency.

## Prepare the environment and data

Run these commands from the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --torch-backend=auto \
  -r examples/workflows/requirements-examples.txt
cd examples/workflows
```

Create the environment only if it does not exist.
Use `uv pip install` for additional packages.
Do not use `uv sync` or `uv run` in this environment.
Those commands can remove separately installed workflow dependencies.

The entries default to CUDA.
Use `--device cpu` for a CPU run.
Choose training and validation batch sizes that fit the device.
The entries do not adjust batch size or CPU thread count automatically.

### ImageNet data

Download the local training and validation shards:

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
  --include 'data/validation-*.parquet' --include 'data/train-*.parquet'
```

Hugging Face may require authentication and acceptance of the dataset's access conditions.
The workflows read local shards and do not download missing dataset files.
Omit `--data_dir` to use the cached snapshot.
Alternatively, supply a snapshot directory that contains `data/` with the Parquet shards.

Every recorded stage evaluates the validation split.
Training, Taylor calibration, VBP, and OSSCAR also require the training split.
Magnitude pruning with zero fine-tuning epochs needs only validation data.

### Caches and preprocessing

The first use converts local Parquet shards into a memory-mapped Arrow cache.
Later runs reuse that cache.
Cache preparation needs additional disk space, even when a run selects few images.
Explicit sample limits create reproducible subsets with their own contiguous Arrow cache.
Arrow stores encoded image bytes.
DataLoader workers decode images and apply the selected weights' preprocessing.

Dataset source shards, Arrow data, and model weights use separate caches:

- `--data_dir` selects the dataset snapshot.
- `HF_DATASETS_CACHE` selects the Arrow cache directory.
- `TORCH_HOME` changes the PyTorch model-weight cache root.
- `HF_HOME` changes Hugging Face's default cache root.

Torchvision downloads missing model weights on first use.
It then reuses the PyTorch Hub checkpoint cache.
The run configuration records the weight enum and actual dataset cache paths.
All supported weight configurations use 224 × 224 crops.
Training uses deterministic weight preprocessing without the papers' complete augmentation recipes.

## Models and shared controls

### Supported models

| Model | Official weights |
| --- | --- |
| `resnet18` | `ResNet18_Weights.IMAGENET1K_V1` |
| `resnet34` | `ResNet34_Weights.IMAGENET1K_V1` |
| `resnet50` | `ResNet50_Weights.IMAGENET1K_V2` |
| `vit_b_16` | `ViT_B_16_Weights.IMAGENET1K_V1` |
| `vit_b_32` | `ViT_B_32_Weights.IMAGENET1K_V1` |
| `convnext_tiny` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |

BN sparsity supports the three ResNets.
VBP supports both ViTs and ConvNeXt-Tiny.
Head pruning supports both ViTs.
The remaining entries support the three ResNets and both ViTs.
Default models are ResNet-18, ConvNeXt-Tiny for VBP, and ViT-B/16 for head pruning.

All entries protect external input and output dimensions, including the classifier.
Each method defines its candidate scope below.
A selected channel can also remove dependent parameters outside its entry layer.
Candidate discovery does not guarantee executable pruning.

### Size targets and selection

Except for Isomorphic, `--pruning_ratio` specifies the fraction of initial whole-model parameters to remove.
The default is `0.05`.
The parameter count includes unique registered parameter elements, including frozen parameters and inserted gates.
It excludes buffers.
The final cap is `floor(initial_params * (1 - pruning_ratio))`.

`--granularity 8` requires retained widths to be divisible by eight.
It can require extra removals.
It does not combine adjacent channels into candidate blocks.
Planning raises an error if the search cannot reach the target.
That planning call changes no model tensors.
A search limit does not prove that no valid selection exists.

Magnitude entries use normalized `GroupMagnitude(p=2)` scores over all affected Parameters, including biases and custom operator parameters.
Buffers do not contribute to this metric.
Shared regions count once.
Static `Greedy` ranks once per plan.
The basic entry also offers dynamic rescoring for magnitude and Taylor.
Dynamic rescoring does not collect fresh gradients or activations.

Isomorphic uses a family-action ratio and its own selection strategy.
OSSCAR selects channels through reconstruction and local search.
Their method sections describe these procedures.

### Data and execution controls

Use each entry's `--help` output for its complete CLI.
Option names use underscores and do not accept abbreviations.

| Options | Default | Effect |
| --- | --- | --- |
| `--train_samples`, `--val_samples` | `0` | Use the full split. Positive values select reproducible subsets. |
| `--train_batch_size`, `--val_batch_size` | `256` | Control training or calibration batches and accuracy batches separately. |
| `--train_workers`, `--val_workers` | `8` | Control DataLoader workers. Zero uses the main process. |
| `--seed` | `7` | Control shuffled sample selection and training randomness. |
| `--finetune_epochs` | `0` | Add task-only training after pruning. Iterative pruning applies this after each round. |
| `--lr` | `0.001` | Set SGD learning rate. VBP defaults to AdamW at `0.000015`. |
| `--compile_latency` | Disabled | Compile only the inference forward used for latency measurement. |
| `--latency_warmup`, `--latency_repetitions` | `5`, `20` | Set untimed warmup and timed inference repetitions. |
| `--output` | `runs/<entry>` | Select the result directory. |

Isomorphic has separate calibration batch controls, with a default batch size of 64.
Training, accuracy evaluation, and MAC counting remain eager.
Latency excludes compilation, its first invocation, input transfer, and warmup.
Compilation errors propagate instead of silently selecting eager execution.

CUDA runs still use CPU work for graph analysis, image decoding, and compilation.
Progress bars identify the current stage and device.
JSON summaries remain on stdout, and progress bars use stderr.
The `cpu_threads` field records configuration, not active threads or model placement.
More workers do not guarantee higher throughput.

## Workflows

Run the following commands from this directory after data preparation.
They use full splits, default loader settings, and compiled latency measurement.
Every command saves and checks a compact checkpoint.
The training durations are examples, not tuned recovery recipes.

For a small execution check, set explicit sample limits and suitable batch sizes.
Also reduce each method's calibration limit when needed.
Small checks establish execution behavior, not ImageNet accuracy.

### Magnitude, Taylor, and FPGM

#### Research basis

[Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2019_paper.html) introduced FPGM at CVPR 2019.
Magnitude and Taylor scoring are generic baselines. This entry does not reproduce one paper.

#### Implemented

The entry targets every ResNet BasicBlock `conv1`, Bottleneck `conv1` and `conv2`, or ViT EncoderBlock `mlp.0`.
Stems, residual interfaces, attention widths, and classifier dimensions stay unchanged.
It scores candidates, plans to the parameter cap, compacts the model, and optionally fine-tunes.

Use `--finetune_epochs 1 --lr 0.001` to add one recovery epoch.
Use `--metric taylor` to collect task-only gradients from one training batch in evaluation mode.
Calibration does not update weights or BatchNorm statistics.
Taylor sums `abs(weight * gradient)` over affected parameter regions, including biases.

`--selection static` is the default.
`--selection dynamic` rescores after each accepted joint addition.
It does not repeat the forward or refresh Taylor gradients.
Neither policy guarantees better task accuracy.

Use `--metric geometric_median --selection static` for FPGM's redundancy criterion.
For signed flattened filters `w_i`, the score is:

```text
score(i) = sum_j ||w_i - w_j||_2
```

This criterion has quadratic work in output width.
The example extends it to ViT FFN rows and uses global parameter budgeting with one-shot deletion.

#### Differences and omissions

This example omits FPGM's per-epoch filter zeroing and regrowth.
The CLI rejects dynamic selection for this initial snapshot.

#### Run

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --metric magnitude \
  --finetune_epochs 0 --output runs/resnet18_prune_only
```

### Iterative pruning

#### Research basis

[DepGraph: Towards Any Structural Pruning](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) provides CVPR 2023 background on coupled-coordinate deletion. This generic iterative baseline does not reproduce one paper.

#### Implemented

Iterative pruning uses the same internal widths as the basic entry.
It interpolates absolute parameter caps from the initial count to the final cap.
Each later round uses a new graph and fresh scores for the compact model.
It does not apply the reduction ratio again to each shrinking model.

The default is three rounds with no recovery training.
This command uses two rounds with one recovery epoch after each.
Alignment can make a later round require no additional removal.
Earlier rounds remain applied if a later round fails.

#### Differences and omissions

The baseline omits DepGraph's exponential sparsity penalties and Top-N score normalization.

#### Run

```bash
python iterative_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --rounds 2 \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_iterative_pruning
```

### BN-scale sparsity

#### Research basis

[Learning Efficient Convolutional Networks through Network Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf) introduced BN-scale sparsity at ICCV 2017.

#### Implemented

This entry applies L1 to every supported ResNet block's `bn1.weight`.
It ranks matching `conv1` channels by learned BN-scale magnitude, then plans and compacts them.
`BNScaleMagnitude` implements this scoring rule and uses the library's `parameter_regions` query.
Conv weights and BN offsets do not contribute to its scores. A request without associated BN scales raises an error.
The default is one sparse-training epoch and no recovery epochs.
`--sparse_epochs 0` skips sparse training.

#### Differences and omissions

This entry retains BN-scale L1, magnitude selection, and channel deletion.
Greedy parameter budgeting replaces global percentile thresholding.
The example omits multi-pass slimming and the paper's pre-activation adaptations.

#### Run

```bash
python bn_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 \
  --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_bn_sparsity
```

### Dependency-group sparsity

#### Research basis

[Learning Structured Sparsity in Deep Neural Networks](https://proceedings.neurips.cc/paper/2016/hash/41bfd20a38bb1b0bec75acf0845530a7-Abstract.html) studies structured group penalties (NeurIPS 2016).
[Neural Pruning via Growing Regularization](https://arxiv.org/abs/2012.09243) introduces GReg-1 and GReg-2 (ICLR 2021).

#### Implemented

This entry targets ResNet block `conv1` outputs or ViT `mlp.0` outputs.
Each regularization group contains the parameters linked to one candidate's removal.
The workflow calls `torch_kirigami.pruning.unique_groups` before constructing the regularizer.
Groups with the same parameter regions then contribute one penalty.
The default `--penalty lasso` penalizes all candidate groups with a constant coefficient.

For `--penalty squared`, selected groups receive a growing squared-L2 coefficient.
The entry reselects groups every `--selection_interval_steps` updates, which defaults to 100.
For step `s` and total updates `T`, it uses one of these schedules:

```text
cosine:  weight * (1 - cos(pi * s / T)) / 2
linear:  weight * s / T
```

Cosine is the default schedule.
The ramp continues across epochs and reselections.
`--sparse_loss_weight` sets the final coefficient in squared mode.
It sets the constant coefficient in Lasso mode.
Zero sparse epochs skip training.

#### Differences and omissions

**Structured Sparsity Learning:** This entry omits joint filter/channel, filter-shape, and depth learning.
**Growing Regularization:** It uses one shared coefficient and dependency-group magnitude ranking instead of per-filter schedules. It omits GReg-2's curvature phase and separate kept/pruned penalties.

#### Run

```bash
python group_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --penalty lasso \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 \
  --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_group_sparsity
```

### Soft projection and norm decay

#### Research basis

[Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://www.ijcai.org/proceedings/2018/309) introduced SFP at IJCAI 2018.
[Decay Pruning Method: Smooth Pruning With a Self-Rectifying Procedure](https://openaccess.thecvf.com/content/ICCV2025W/ECLR/html/Yang_Decay_Pruning_Method_Smooth_Pruning_With_a_Self-Rectifying_Procedure_ICCVW_2025_paper.html) presented norm decay at ICCV Workshops 2025.

#### Implemented

This entry targets ResNet block `conv1` outputs or ViT `mlp.0` outputs.
It uses one warmup epoch, projection cycles, and one unprojected recovery epoch between cycles.
Selection stays unchanged within each cycle.
The entry checks the final exact selection before physical deletion.

`--operation zero` projects one selected dependency union after each minibatch.
It uses trainable zeroed regions and recovery.
`--operation decay` rescales one selected union along a linear norm path to zero per cycle.
Zeroing or norm projection changes values until physical pruning changes dimensions.

Before physical deletion, training lasts `1 + cycles * projection_epochs + (cycles - 1)` epochs.
Defaults give four epochs.
This command adds one post-pruning epoch.

#### Differences and omissions

Zero mode omits SFP's per-epoch filter reselection.
Decay mode omits DPM's gradient-based self-rectification and separate per-structure trajectories.

#### Run

```bash
python soft_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --operation decay --cycles 2 --projection_epochs 1 \
  --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_soft_pruning
```

### Gate training

#### Research basis

[Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2019/file/b51a15f382ac914391a58850ab343b00-Paper.pdf) introduced trainable gates and L1 training at NeurIPS 2019.

#### Implemented

This entry inserts unit-valued gates after ResNet `bn1` or ViT FFN activation.
It trains ordinary weights and gates together with L1 on gate scales.
`GateMagnitude` sums `abs(gate_weight * gate_mask)` over affected gate positions.
Static planning chooses removals under the parameter cap.
Retained gates remain explicit modules with their learned values.

#### Differences and omissions

The example adopts gates and L1 training, but replaces `|gate * d(loss)/d(gate)|` with magnitude scoring.
It restricts candidates to the gated internal widths.
It omits Tick-Tock phases, the paper's shortcut Group Pruning procedure, and gate folding.
The ViT path extends the gating idea beyond that CNN paper.

#### Run

```bash
python gate_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 \
  --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_gate_pruning
```

### Selection stability

#### Research basis

[One-Cycle Structured Pruning via Stability-Driven Subnetwork Search](https://openaccess.thecvf.com/content/WACV2026/html/Ghimire_One-Cycle_Structured_Pruning_via_Stability-Driven_Subnetwork_Search_WACV_2026_paper.html) studies stability-driven search (WACV 2026).

#### Implemented

This entry alternates parameter-budget selection and squared-L2 training on selected dependency groups.
It calls `torch_kirigami.pruning.unique_groups` before constructing the regularizer.
Groups with the same parameter regions then contribute one penalty.
It compares retained channel indices with mean per-axis Jaccard similarity.
`--window 2` needs three selections for two adjacent comparisons.
Search stops at `--threshold 0.99` or the configured check limit.

The maximum training duration is `(max_selection_checks - 1) * selection_interval_steps` updates.
Defaults give 1,000 updates, with the earliest stability stop after 200.
One check skips training.
A limit below `window + 1` cannot establish stability.
The sparse coefficient follows the same step-based ramp as squared group sparsity.
Early stopping does not rescale that ramp or force the final coefficient.

#### Differences and omissions

The example starts from pretrained weights. It omits normalized saliency, delayed start, a distinct stability cutoff, and the full one-cycle schedule.

#### Run

```bash
python stability_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --max_selection_checks 11 --selection_interval_steps 100 \
  --window 2 --threshold 0.99 \
  --sparse_loss_weight 0.0001 --sparsity_schedule cosine \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_stability_pruning
```

### Variance-Based Pruning

#### Research basis

[Variance-Based Pruning for Accelerating and Compressing Trained Networks](https://openaccess.thecvf.com/content/ICCV2025/html/Berisha_Variance-Based_Pruning_for_Accelerating_and_Compressing_Trained_Networks_ICCV_2025_paper.html) introduced VBP at ICCV 2025.

#### Implemented

This entry discovers eligible dense MLP chains in ViT and ConvNeXt.
It records consumer-input means and sample variances in evaluation mode.
Calibration pools batch, spatial, and token positions, including the ViT class token.
It uses training images without labels or backward.

The default calibration reads 16 batches, up to 4,096 images with the default batch size.
`--calibration_batches 0` uses the full loader.
One initial snapshot supplies the static global ranking by raw variance.
The entry excludes branches, shared calls, and unsupported transforms.
Padded spatial convolution can need position-dependent correction, so this entry does not support ResNet.

For removed hidden positions `R`, the consumer bias correction is:

```text
new_bias = old_bias + old_weight[:, R] @ calibration_mean[R]
```

The workflow computes correction values before applying the plan and installs them afterward.
This is exact for replacing removed activations with their means, not for their original varying values.
Earlier pruning can change later activation distributions.
The workflow does not recalibrate during selection.

#### Differences and omissions

Whole-model budgeting and width alignment replace neuron-ratio experiments.
The [official implementation](https://github.com/boschresearch/variance-based-pruning) uses the full loader by default.
Optional recovery uses AdamW, weight decay `0.01`, and a cosine schedule per optimizer step.
It omits teacher distillation and the complete augmentation recipe.

#### Run

```bash
python variance_pruning.py \
  --model convnext_tiny --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --calibration_batches 16 \
  --finetune_epochs 0 --output runs/convnext_tiny_variance
```

### Isomorphic Pruning

#### Research basis

[Isomorphic Pruning for Vision Models](https://arxiv.org/abs/2407.04616) compares structurally similar dependency groups with Taylor scores (ECCV 2024).

#### Implemented

The entry groups complete dependency impacts by ordered typed structural signatures.
These signatures are conservative comparisons, not a general graph-isomorphism solver.
Equivalent closures count once.
The entry inspects graph-declared domains under interface protections and execution checks.
Proved identity components reuse per-position Taylor statistics, including ordinary
CNN channel chains with full-tensor scopes. Partial partitions and nested extension
behavior retain per-candidate analysis. This reuse does not alter family quotas or
skip final joint verification.

Calibration accumulates batch-mean cross-entropy gradients without updating weights or BatchNorm statistics.
Defaults use 100 batches of 64 training images.
Reduce the batch limit explicitly when a short split cannot supply the requested calibration batches.
`PaperTaylor` sums per-parameter L2 norms of removed `weight * gradient` regions.
It excludes biases and buffers, and counts shared regions once.
This follows the displayed equations rather than the [official code](https://github.com/VainF/Isomorphic-Pruning/blob/main/pbench/isomorphic_pruner.py), which uses normalized absolute products.

Each family requests its lowest-ranked `floor(family_pruning_ratio * action_count)` actions.
Alignment can add removals beyond that quota.
Rounding scans each axis's ordered contributors once as selected positions grow;
it preserves the original action order and revisits coupled axes.
The ratio is not a whole-model parameter reduction and is not derived through target search.
The original parameter count supplies only a non-increase guard.
An invalid joint allocation fails before mutation.

Native attention candidates remove the same local dimension from every head.
Head count stays unchanged while per-head width decreases.
Optional fine-tuning uses task loss only.

#### Differences and omissions

Independent whole-head deletion, architecture-specific family ratios, and distilled recovery are absent.

#### Run

```bash
python isomorphic_pruning.py \
  --model vit_b_32 --device cuda --compile_latency \
  --family_pruning_ratio 0.05 --granularity 8 \
  --calibration_batches 100 --calibration_batch_size 64 \
  --finetune_epochs 0 --output runs/vit_b_32_isomorphic
```

### OSSCAR reconstruction

#### Research basis

[OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization](https://proceedings.mlr.press/v235/meng24a.html) combines reconstruction with grouped deletion and local search (ICML 2024).

#### Implemented

The entry discovers supported unbranched Conv2d and Linear producer/consumer chains.
It excludes grouped or depthwise convolution, attention, branches, shared uses, and unsupported cross-channel transforms.

First, a common channel-removal fraction allocates aligned widths that meet the parameter cap.
Consumers then follow forward order:

1. Pair current compact-model inputs with original dense-teacher targets.
2. Form damped Gram and cross-product systems.
3. Search grouped deletions and bounded remove/restore swaps.
4. Fit retained weights.
5. Apply the exact structural request.
6. Copy the refitted values. Rebuild the graph for the next consumer.

For design rows `X`, bias-adjusted targets `Y`, original coefficients `B0`, and ridge strength `lambda`:

```text
H = X.T @ X / observations + lambda * I
G = X.T @ Y / observations + lambda * B0
B_retained = solve(H[retained, retained], G[retained])
```

Calibration defaults to two training batches per consumer.
Zero uses the full training loader for each consumer.
`--calibration_rows 4096` caps evenly spaced spatial or token positions per batch.
Student and teacher use matching sampled positions.
Statistics require memory quadratic in the consumer's input width times kernel area.

`--damping 0.01` scales ridge strength by the mean Gram diagonal and anchors reconstruction toward original weights.
`--prune_batch 8` controls deletion updates independently of retained-width alignment.
`--swap_steps 5` bounds local swaps and can stop earlier.
These settings do not guarantee a global optimum or exhaustive local search.
The sampled calibration, common-width allocation, and ViT FFN scope adapt the paper's experiments.
Width allocation uses integer arithmetic and binary lookup in aligned-width
breakpoints. Each tested allocation still uses a joint structural plan to measure
the actual parameter count; layer costs are not approximated as additive.

Each consumer is a separate operation, so earlier changes remain if a later consumer fails.

#### Differences and omissions

Common-width allocation does not optimize reconstruction error across layers.
Sampled calibration, ViT FFN scope, and bounded search adapt the paper's experiments.
Teacher targets remain part of reconstruction even when users disable fine-tuning. The workflow omits distillation recovery.

#### Run

```bash
python osscar_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --calibration_batches 2 --calibration_rows 4096 --damping 0.01 \
  --prune_batch 8 --swap_steps 5 --finetune_epochs 0 \
  --output runs/resnet18_osscar
```

### ViT heads and FFN channels

#### Research basis

This magnitude baseline does not reproduce one paper.

#### Implemented

The adapter exposes packed Q/K/V projections, SDPA, and an output projection.
It supports unmasked self-attention and preserves pretrained values and attention dropout.

One head candidate removes matching Q/K/V rows, biases, and output-projection columns.
Head count can decrease independently in each block.
Per-head width, attention scale, embedding width, and residual interfaces stay unchanged.
FFN candidates remove hidden positions and matching consumer columns.
Granularity applies only to FFN widths.
Every block retains at least one head and a nonempty FFN.

Static `Greedy` uses normalized affected-weight energy for one mixed candidate space.
Stage reports record actual head counts and FFN widths.
Isomorphic pruning instead preserves head count and prunes dimensions within each head.

#### Differences and omissions

The adapter omits cross-attention, GQA, and token pruning.
No quota forces both heads and FFNs, or every block, to shrink.

#### Run

```bash
python vit_head_pruning.py \
  --model vit_b_16 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --finetune_epochs 0 --output runs/vit_b_16_heads_and_ffn
```

## Outputs, restoration, and comparison

Each output directory contains:

- `metrics.json`: effective settings, data metadata, stage measurements, and method reports.
- `model.pt`: final compact structure and values, including learned gates, corrected biases, or refitted weights.
- `training.pt`: caller-owned optimizer, algorithm, configuration, and random-number state.

The saved training state does not provide a general resume CLI.
Each successful run checks checkpoint outputs against the compact model in memory.
Each entry recreates its optimizer after physical parameter replacement.
VBP and OSSCAR restoration must use the corrected or refitted checkpoint, not only structural plans.

For a completed head-pruning run, restore the matching adapter from this directory:

```python
from imagenet_models import make_head_prunable_model
from torch_kirigami.pruning import load_checkpoint

model = load_checkpoint(
    make_head_prunable_model("vit_b_16", pretrained=False),
    "runs/vit_b_16_heads_and_ffn/model.pt",
    map_location="cuda",
).eval()
```

For other entries, construct the matching original skeleton, including any inserted gates.
The library does not replace native MHA automatically during load.

Stage records include validation loss, top-1/top-5 accuracy, and accuracy change in percentage points.
They also include parameter elements, MACs, latency, and measurement settings.
MACs and latency describe one image, with batch size one.
Accuracy uses `--val_batch_size`, which does not change the measurement batch.
Check `unsupported_ops` before assuming complete MAC coverage.
Neither MACs nor latency is a pruning target in these entries.

Compare methods with matched candidate scope, parameter target, calibration, and recovery settings.
Parameter reduction does not guarantee better latency or preserved accuracy.
Synthetic tests and small execution checks do not establish ImageNet accuracy.
Accuracy claims require pretrained weights, the actual dataset, and recorded evaluation settings.
