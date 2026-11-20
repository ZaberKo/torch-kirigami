# Structural pruning workflow examples

These examples use ImageNet classification to demonstrate structural pruning, evaluation, and checkpoint restoration.
Each entry owns its CLI and training stages.
The examples implement selected research ideas, with explicit changes to scope and training.
They do not claim the papers' published accuracy.

Three modules provide shared infrastructure:

- `imagenet_data.py` loads local data and evaluates accuracy.
- `imagenet_models.py` loads official weights, supplies ViT adapters, and discovers dense MLP pairs.
- `model_metrics.py` uses Torch Prisma to measure complexity and inference latency.

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
Training, Taylor calibration, VBP, DVBP, and OSSCAR also require the training split.
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
Training uses deterministic weight preprocessing by default.
`prune_finetune.py --train_augmentation` enables random resized crop and horizontal flip for fine-tuning only.
Its calibration and validation loaders retain deterministic weight preprocessing.
The training view shares the same Arrow rows and encoded image cache.
Other entries retain deterministic preprocessing and omit the papers' complete augmentation recipes.

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

BN-guided pruning supports the three ResNets.
VBP and DVBP support both ViTs and ConvNeXt-Tiny.
Head pruning supports both ViTs.
The remaining entries support the three ResNets and both ViTs.
Default models are ResNet-18, ConvNeXt-Tiny for VBP, and ViT-B/16 for DVBP and head pruning.

All entries protect external input and output dimensions, including the classifier.
Each method defines its candidate scope below.
A selected channel can also remove dependent parameters outside its entry layer.
Candidate discovery does not guarantee executable pruning.

### Size targets and selection

`--pruning_ratio` specifies the fraction of initial whole-model parameters to remove in entries that use that option.
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
DVBP uses a global MLP hidden-neuron ratio.
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
| `--lr` | `0.001` | Set SGD learning rate. VBP and DVBP default to AdamW at `0.000015`. |
| `--compile_latency` | Disabled | Compile only the inference forward used for latency measurement. |
| `--latency_warmup`, `--latency_repetitions` | `5`, `20` | Set untimed warmup and timed inference repetitions. |
| `--output` | `runs/<entry>` | Select the result directory. |

Isomorphic has separate calibration batch controls, with a default batch size of 64.
Training, accuracy evaluation, and MAC counting remain eager.
Latency excludes compilation preparation and warmup. The workflow prepares the model and input on the selected device before measurement.
Compilation errors propagate instead of silently selecting eager execution.
Measurement temporarily selects evaluation mode and restores each module's previous mode, including after failure.
Other forward side effects persist, including input or buffer writes and random-number consumption.
See the [measurement guide](../../docs/guides/measurement.md) for the Torch Prisma API and execution contract.

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

### Magnitude, Taylor, FPGM, and Pruner-Zero

#### Research basis

- `geometric_median` adapts *Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration* (CVPR 2019).
- `pruner_zero` applies the expression from *Pruner-Zero: Evolving Symbolic Pruning Metric From Scratch for Large Language Models* (ICML 2024).

Magnitude and Taylor scoring are generic baselines.

#### Implemented

The entry targets every ResNet BasicBlock `conv1`, Bottleneck `conv1` and `conv2`, or ViT EncoderBlock `mlp.0`.
Stems, residual interfaces, attention widths, and classifier dimensions stay unchanged.
It scores candidates, plans to the parameter cap, compacts the model, and optionally fine-tunes.

Use `--finetune_epochs 1 --lr 0.001` to add one recovery epoch.
Use `--metric taylor` to collect task-only gradients in evaluation mode.
`--calibration_batches` defaults to 16. Zero uses the full training loader.
Calibration accumulates summed cross-entropy gradients and divides by the total sample count.
Unequal batch sizes therefore contribute by sample count.
This computes the dataset-mean loss gradient, not an average of absolute batch Taylor scores or per-sample Fisher statistics.
Calibration does not update weights or BatchNorm statistics and restores all train/eval flags.
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

Pruner-Zero takes the L2 norm of each weight's batch-mean cross-entropy gradients in evaluation mode.
It min-max scales these norms over each entire weight tensor and multiplies them by the squared weights.
Each candidate sums the scores of the newly removed matrix and kernel entries across its dependency closure.
A constant gradient statistic gives zero importance. Biases and normalization vectors still follow the plan.
The calibration limit and batch size use `--calibration_batches` and `--train_batch_size`.
Batch size can change the scores because each gradient is reduced before squaring.
Dynamic selection reuses these fixed per-weight scores and skips entries already removed by the plan.

#### Differences and omissions

The FPGM option omits per-epoch filter zeroing and regrowth and uses static selection.
Pruner-Zero here uses ImageNet batch gradients and physical channel deletion. The original LLM pipeline calibrates on sequences and masks individual weights in unstructured or N:M patterns. This entry applies the discovered expression and omits its genetic search.

#### Run

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --metric magnitude \
  --finetune_epochs 0 --output runs/resnet18_prune_only

python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --metric pruner_zero --calibration_batches 16 \
  --finetune_epochs 0 --output runs/resnet18_pruner_zero
```

### Iterative pruning

#### Research basis

*DepGraph: Towards Any Structural Pruning* (CVPR 2023) provides background on coupled-coordinate deletion.

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

### BN-guided channel pruning

#### Research basis

*Learning Efficient Convolutional Networks through Network Slimming* (ICCV 2017) introduced BN-scale sparsity.

#### Implemented

This entry applies L1 to every supported ResNet block's `bn1.weight`.
It ranks matching `conv1` channels by learned BN-scale magnitude, then plans and compacts them.
The candidate axis is `conv1.weight` axis 0. BN scales supply scores, not candidate entry axes.
`BNScaleMagnitude` implements this scoring rule and uses the library's `parameter_regions` query.
It sums absolute values across affected, explicitly selected BN scales, including multiple BN modules when present.
Shared parameter coordinates count once. The score has no per-BN averaging.
Conv weights and BN offsets do not contribute to its scores. A request without associated BN scales raises an error.
The default is one sparse-training epoch and no recovery epochs.
`--sparse_epochs 0` skips sparse training.

#### Differences and omissions

This entry retains BN-scale L1, magnitude selection, and channel deletion.
Greedy parameter budgeting replaces global percentile thresholding.
The example omits multi-pass slimming and the paper's pre-activation adaptations.

#### Run

```bash
python bn_guided_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 \
  --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_bn_guided_pruning
```

### Dependency-group sparsity

#### Research basis

*Learning Structured Sparsity in Deep Neural Networks* (NeurIPS 2016) studies structured group penalties.
*Neural Pruning via Growing Regularization* (ICLR 2021) introduces GReg-1 and GReg-2.

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

*Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks* (IJCAI 2018) introduced SFP.
*Decay Pruning Method: Smooth Pruning With a Self-Rectifying Procedure* (ICCV Workshops 2025) presented norm decay.

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

*Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks* (NeurIPS 2019) introduced trainable gates and L1 training.

#### Implemented

This entry inserts unit-valued gates after ResNet `bn1` or ViT FFN activation.
It trains ordinary weights and gates together with L1 on gate scales.
`GateMagnitude` sums `abs(gate_weight * gate_mask)` over affected gate positions.
Static planning chooses removals under the parameter cap.
Retained gates remain explicit modules with their learned values.
With `--sparse_epochs 0`, the workflow warns that unit-valued gates provide no learned ranking.
This setting remains available for smoke tests.

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

### Comb, Prune, Distill

*Comb, Prune, Distill: Towards Unified Pruning for Vision Model Compression* (ITSC 2024) interleaves training, importance scoring, and group removal. The public implementation supplies a semantic segmentation workflow. [CPD code](https://github.com/InSAI-Lab/CPD)

This classification adaptation uses a frozen copy of the original model as the teacher. After each backward pass, it sums the signed weight-gradient products within each output channel, squares the sum, and divides it by the output activation count. It accumulates these scores over a fixed number of training batches. It applies the source code's normalization across group members and ranks each group by the sum of its selected channels' scores.

Each interval removes one executable group. The entry considers dense ResNet block interiors and ViT MLP hidden widths. The width determines the removal size for a convolution group, up to four channels. A ViT MLP group removes up to 32 channels. At least one channel remains. Kirigami analyzes the dependency closure and physically compacts the model. The entry also compacts SGD momentum buffers using the same retained coordinates. Each interval starts with fresh scores.

The run stops when the total number of remaining parameters meets the cap derived from `--pruning_ratio`. The final complete group can reduce the count below the cap. Raise `--max_intervals` if the interval limit stops the run before it reaches the cap. Optional recovery uses the same cross-entropy and logit distillation objective as pruning. Set `--distillation_weight 0` to train with cross-entropy alone.

```bash
python cpd_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --pruning_interval_batches 50 \
  --recovery_epochs 1 --output runs/resnet18_cpd
```

### Selection stability

#### Research basis

*One-Cycle Structured Pruning via Stability-Driven Subnetwork Search* (WACV 2026) studies stability-driven search.

#### Implemented

This entry alternates parameter-budget selection and squared-L2 training on selected dependency groups.
It calls `torch_kirigami.pruning.unique_groups` before constructing the regularizer.
Groups with the same parameter regions then contribute one penalty.
It compares retained channel indices with mean per-axis Jaccard similarity.
`--window 2` needs three selections for two adjacent comparisons.
Search stops at `--threshold 0.99` or the configured check limit. The example applies the plan from that final check.

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

Variance-Based Pruning for Accelerating and Compressing Trained Networks (ICCV 2025) introduced VBP.

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

### Denoised Variance-Based Pruning and OB2C

*Denoised Variance-Based Pruning with Optimal Brain Bias Compensation* (ECCV 2026).

Calibration collects a mean and centered covariance from the inputs of each MLP's second linear layer.
All token or spatial positions contribute activation rows.
The Marchenko–Pastur noise threshold uses the number of calibration images.
With the default batch settings, calibration reads up to 8,192 training images.
`--calibration_batches 0` uses the full loader.

`--hidden_pruning_ratio` has two uses.
For a ratio of 0.20, the score is 80% raw variance plus 20% denoised variance, and the target is to remove 20% of the discovered MLP hidden neurons across the model.
The selection keeps at least one aligned block per MLP.
`--granularity 8` ranks blocks of eight lowest remaining scores by their summed score and can increase the actual reduction.
Use `--granularity 1` for per-neuron ranking.
The dependency graph checks the complete selection before any weights change.

OB2C fits the surviving columns of each second linear layer against the original layer under the damped denoised covariance. It adjusts the bias to preserve the original calibration mean.
Calibration and denoising use the selected device. Compensation solves run in float64 on CPU.
The workflow computes corrected values before structural apply, copies them into the compact model, and verifies the saved checkpoint.

The example uses one covariance per MLP for all three torchvision models. The authors' ConvNeXt code uses a joint covariance per stage, which can change selection and compensation.
Torchvision weights, width alignment, and optional cross-entropy fine-tuning are workflow choices.
The default run is training-free.

```bash
python dvbp_pruning.py \
  --model vit_b_16 --device cuda --compile_latency \
  --hidden_pruning_ratio 0.20 --granularity 8 --calibration_batches 32 \
  --finetune_epochs 0 --output runs/vit_b_16_dvbp
```

### Isomorphic Pruning

#### Research basis

*Isomorphic Pruning for Vision Models* (ECCV 2024) compares structurally similar dependency groups with Taylor scores.

#### Implemented

The entry groups complete dependency impacts by ordered typed structural signatures.
These signatures are conservative comparisons, not a general graph-isomorphism solver.
Candidates that remove the same coordinates throughout the graph count as one action toward the family quota, even when their seeds appear in a different order.
The entry inspects graph-declared domains under interface protections and execution checks.
Proved identity components reuse per-position Taylor statistics, including ordinary
CNN channel chains with full-tensor scopes. Partial partitions and nested extension
behavior retain per-candidate analysis. This reuse does not change family quotas or
skip final joint verification.

Calibration accumulates batch-mean cross-entropy gradients without updating weights or BatchNorm statistics.
Defaults use 100 batches of 64 training images.
Reduce the batch limit explicitly when a short split cannot supply the requested calibration batches.
`PaperTaylor` sums per-parameter L2 norms of removed `weight * gradient` regions.
It excludes biases and buffers, and counts shared regions once.
This follows the displayed equations rather than the [official code](https://github.com/VainF/Isomorphic-Pruning/blob/main/pbench/isomorphic_pruner.py), which uses normalized absolute products.

Each family requests its lowest-ranked `floor(family_pruning_ratio * action_count)` actions.
Alignment can add removals beyond that quota.
Rounding scans each axis's ordered contributors once as selected positions grow.
It preserves the original action order and revisits coupled axes.
The ratio is not a whole-model parameter reduction and is not derived through target search.
The original parameter count supplies only a non-increase guard.
An invalid joint allocation fails before mutation.

Discovery builds each native attention candidate from the same local dimension in every head.
This applies to graph-declared producer axes that share those coordinates, including ViT patch embeddings.
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

*OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization* (ICML 2024) combines reconstruction with grouped deletion and local search.

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
the actual parameter count. Layer costs are not approximated as additive.

Each consumer is a separate operation, so earlier changes remain if a later consumer fails.
The `pruned` stage in `metrics.json` records the retained positions, calibration observations, and search objectives for each reconstructed layer.

#### Differences and omissions

Common-width allocation does not optimize reconstruction error across layers.
Sampled calibration, ViT FFN scope, and bounded search adapt the paper's experiments.
The paper's displayed dense-layer objective uses identical inputs for the dense and pruned weights.
This workflow pairs current-model inputs with original-model outputs after earlier pruning, and adds damping toward original weights.
The frozen dense model supplies layer-output targets for channel selection and least-squares weight refitting.
These steps belong to pruning and still run with `--finetune_epochs 0`.
The workflow performs no distillation training. Optional fine-tuning uses labeled cross-entropy only.
Reconstruction keeps the current consumer bias unchanged and subtracts that bias from dense-model targets before fitting weights.

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

Each successful run writes:

- `metrics.json`: effective settings, data metadata, stage measurements, and method reports. Each completed evaluation updates this file and emits the same stage record as a JSON line on stdout.
- `model.pt`: final compact structure and values, including learned gates, corrected biases, or refitted weights.

Some entries also write:

- `pruned.pt`: CPD and DVBP save this model checkpoint at the end of pruning when post-pruning training is enabled. DVBP saves it after OB2C compensation. Load it with the same model constructor as `model.pt` to evaluate the compact model before recovery.
- `training.pt`: caller-owned optimizer, algorithm, configuration, and random-number state.

Model checkpoints support evaluation and fresh fine-tuning. CPD and DVBP keep optimizer state in memory for the current run.
The workflows do not provide a general resume CLI.
Each successful run checks checkpoint outputs against the compact model in memory.
Fine-tuning creates its optimizer after physical parameter replacement.
Restore VBP, DVBP, and OSSCAR with `model.pt`, which includes the corrected or refitted values as well as the structure.

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
They also include parameter elements, MACs, FLOPs, latency, and measurement settings.
MACs, FLOPs, and latency describe one image, with batch size one.
Accuracy uses `--val_batch_size`, which does not change the measurement batch.
Check `uncovered_ops["macs"]` before assuming complete MAC coverage.
Check `uncovered_ops["flops"]` separately for incomplete FLOP coverage.
Neither MACs nor latency is a pruning target in these entries.

Compare methods with matched candidate scope, parameter target, calibration, and recovery settings.
Parameter reduction does not guarantee better latency or preserved accuracy.
Synthetic tests and small execution checks do not establish ImageNet accuracy.
Accuracy claims require pretrained weights, the actual dataset, and recorded evaluation settings.
