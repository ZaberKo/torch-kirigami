# Pretrained ImageNet pruning workflows

Eleven standalone entries demonstrate structural pruning, sparse training, fine-tuning, and checkpoint restoration. Each file owns its CLI and training/pruning steps. Three infrastructure modules are shared: `imagenet_data.py` (data and accuracy), `imagenet_models.py` (official weights and the ViT adapter), and `model_metrics.py` (complexity and latency). Common CLI options have the same meaning across entries; they are documented below rather than hidden in a shared parser. See [method selection and adaptations](../../docs/workflow-methods.md) for the research sources and implemented scope.

## Environment, data, and weight caches

From the repository root, use the existing environment, install the workflow dependencies, and enter this directory:

```bash
source .venv/bin/activate
uv pip install --torch-backend=auto -r examples/workflows/requirements-examples.txt
cd examples/workflows
```

If no environment exists, first create it with `uv venv .venv`. Use `python` from the activated environment; do not use `uv sync` or `uv run`. CUDA is the default device. A CPU run requires explicit `--device cpu`; the scripts do not change PyTorch's CPU thread settings.

Download the ImageNet validation and training shards once:

```bash
hf download ILSVRC/imagenet-1k --repo-type dataset \
  --include 'data/validation-*.parquet' --include 'data/train-*.parquet'
```

Access to this dataset may require accepting its access conditions and authenticating with Hugging Face. Each experiment reads the local shards without downloading dataset files. Omit `--data_dir` to resolve the cached HF snapshot, or pass a snapshot directory whose `data/` subdirectory contains these Parquet files. The dataset cache and the model-weight cache are separate.

Both training and validation use Hugging Face's memory-mapped **PyArrow cache**.
The first use converts the requested split's local Parquet shards to Arrow;
subsequent runs reuse it. Training no longer streams Parquet during each epoch.
Arrow files normally live under `~/.cache/huggingface/datasets`; set
`HF_DATASETS_CACHE=/path/to/arrow-cache` before launching Python to use another
disk. `HF_HOME` also changes the default Hugging Face cache root. The run's
`config.dataset` records the actual Arrow files, separately from source shards.

Cache preparation is a one-time CPU/I/O task and needs additional disk space
alongside the original shards. Even a limited training run prepares the source
split's Arrow cache on first use. Explicit sample limits then choose a reproducible
random subset and materialize it as a separate contiguous Arrow cache; full splits
do not require that extra copy. Arrow contains encoded image bytes, **not decoded
or normalized pixels**: PIL decoding and torchvision preprocessing still happen
in DataLoader workers. This preserves the pretrained weights' exact preprocessing.
See [HF caching](https://huggingface.co/docs/datasets/about_cache) and
[Arrow-backed datasets](https://huggingface.co/docs/datasets/about_arrow).

Torchvision downloads the selected weights on first use and reuses them thereafter. Weights are stored in `torch.hub.get_dir()/checkpoints`, normally `~/.cache/torch/hub/checkpoints`. Setting `TORCH_HOME=/path/to/torch-cache` changes this to `/path/to/torch-cache/hub/checkpoints`; an existing `XDG_CACHE_HOME` also affects the default. Inspect the effective directory with:

```bash
python -c 'from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / "checkpoints")'
```

See the official [torchvision weight-loading documentation](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights) and [PyTorch Hub cache documentation](https://docs.pytorch.org/docs/stable/hub.html#where-are-my-downloaded-models-saved).

## Supported models and pruning scope

| `--model` | Official weights |
| --- | --- |
| `resnet18` (default) | `ResNet18_Weights.IMAGENET1K_V1` |
| `resnet34` | `ResNet34_Weights.IMAGENET1K_V1` |
| `resnet50` | `ResNet50_Weights.IMAGENET1K_V2` |
| `vit_b_16` | `ViT_B_16_Weights.IMAGENET1K_V1` |
| `vit_b_32` | `ViT_B_32_Weights.IMAGENET1K_V1` |
| `convnext_tiny` (VBP only; its default) | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |

BN sparsity accepts the three ResNet models. VBP accepts ConvNeXt-Tiny and both ViTs. Head pruning accepts both ViTs; the remaining entries accept the three ResNets and both ViTs.

| Workflows | Candidate scope |
| --- | --- |
| Basic and iterative | Every BasicBlock's `conv1`, every Bottleneck's `conv1` and `conv2`, and every ViT EncoderBlock's `mlp.0`. Stem, block input/output widths, attention widths and classifier dimensions stay fixed. |
| ViT head pruning | Complete attention heads and FFN hidden channels in every encoder block. Residual embedding width and per-head feature width stay fixed. |
| Isomorphic | All graph-declared candidates, including residual-linked widths, subject to dependency and execution checks. |
| VBP and OSSCAR | Chains satisfying the algorithm's mathematical requirements; excluded positions have explicit reasons. |
| Sparse training | Each ResNet block's `conv1` output or each ViT block's FFN intermediate width; BN sparsity uses ResNet only. |

The basic examples use internal widths to keep block interfaces unchanged and
make a controlled baseline. This is an experimental scope, not a restriction of
the library's discovery API or a guarantee of better accuracy. An unmet parameter
target does not silently expand that scope. Iterative pruning keeps it across
every rebuild. Compare algorithms with matched candidate scopes, calibration and
recovery settings, not just the same parameter ratio.

All workflows preserve external input/output dimensions, including the classifier. ViT uses an explicit forward adapter preserving torchvision's weights and computation while exposing data flow to FX. Attention and hidden-width changes in broader scopes remain subject to their operator constraints and execution support; discovery is not a guarantee of executability. Preprocessing comes from the selected weight enum; all listed configurations use 224 × 224 crops.

There is no layer-selection CLI or first-layer default. Every workflow examines its declared scope throughout the model; this does not mean that every parameter is eligible for every method. The dependency graph covers the entire model, and a selected channel can affect other parameters through dependencies.

Granularity constrains retained widths in the chosen scope; it does not package
adjacent channels. `--pruning_ratio` is a reduction fraction of the entire model's
parameter count, including fixed parts, rather than a per-layer channel ratio.
Whole-model ViT candidate spaces can take substantially longer to plan than
ResNets; reducing dataset samples does not reduce dependency-planning work.
Planning fails explicitly if bounded search cannot reach the parameter target.

### Scoring and selection

Ranking-based entries default to the library's static `Greedy`: scores are computed once per plan,
then joint dependencies, constraints, and the parameter target determine which
additions are accepted. Explicit training or pruning rounds produce fresh scores
on the next planning call. None of these scripts silently recalibrates during
selection. The basic entry additionally exposes `--selection dynamic` for
magnitude/Taylor; the other entries retain their explicit static selection steps.
Isomorphic uses independent family quotas and its own ratio search, not `Greedy`.
OSSCAR uses its own quadratic deletion/swap search and submits exact selections
through `plan_remove`; it is not a magnitude metric for `Greedy`.

Magnitude-based entries use `GroupMagnitude(p=2)`. It includes affected recognized
Conv, Linear, and normalization weights, excludes bias, and normalizes their
combined squared magnitude by the average single-channel score in the declared
channel domain. Shared parameter regions count once. This replaces the previous
producer-weight-only formula; resulting selections can therefore differ.
The head-pruning entry uses an explicit logical head axis, so its normalization
compares complete heads within the same attention block. FFN channels are
normalized within their own hidden axis. This heuristic does not guarantee
comparable task sensitivity across attention and FFN.
`WeightTaylor(mode="elementwise_abs")` instead sums `abs(weight * grad)` over the
affected parameter union, including bias and normalization parameters. BN and
gate training retain their algorithm-specific learned-scale signals:
`Magnitude(p=1, parameter_filter=...)` and `GateMagnitude`, respectively.
The basic entry also offers FPGM's signed-filter distance criterion. VBP uses
post-activation variance and consumer-bias compensation. These two criteria are
method-specific signals, rather than alternative implementations of magnitude.

Custom model scoring implements `score(context, candidates, *, selected)`;
`selected` is the already accepted dependency impact. The library also offers
`DynamicGreedy`, which rescores after accepting each feasible addition. It updates
conditional parameter-region scores, not the model's activations or gradients.
Fresh task statistics require explicit execution and calibration between rounds.

## Common CLI options

All multiword workflow options use underscores. The CLI accepts complete option names, without abbreviations or aliases for old spellings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `resnet18`; VBP: `convnext_tiny`; head pruning: `vit_b_16` | Model choices depend on the method, as described above |
| `--data_dir` | HF cache | Local dataset snapshot directory |
| `--device` | `cuda` | Training, accuracy evaluation, graph capture, and measurement device |
| `--train_samples` | `0` | Available training images per dataset traversal; `0` makes the full split available. Taylor reads one batch; VBP/OSSCAR have calibration-batch limits; stability search may stop before completing a traversal |
| `--val_samples` | `0` | Validation images per recorded stage; `0` evaluates the complete validation split |
| `--train_batch_size` | `256` | Training and calibration batch size |
| `--val_batch_size` | `256` | Accuracy-evaluation batch size; independent of MAC/latency measurement |
| `--train_workers` | `8` | Training/calibration DataLoader worker processes; `0` performs loading in the main process |
| `--val_workers` | `8` | Validation DataLoader worker processes; `0` performs loading in the main process |
| `--seed` | `7` | Model/training RNG and shuffled data selection seed |
| `--pruning_ratio` | `0.05` | Fraction of initial whole-model parameters to remove; includes fixed/frozen parameters and inserted gates. Iterative pruning uses the final cumulative reduction |
| `--granularity` | `8` | Retained producer widths must be divisible by this factor; `1` adds no alignment constraint |
| `--finetune_epochs` | `0` | Task-only epochs after physical pruning; iterative pruning applies this after every round |
| `--lr` | `0.001`; VBP: `0.000015` | SGD learning rate with momentum `0.9`; VBP uses AdamW and a cosine schedule |
| `--compile_latency` | Disabled | Apply `torch.compile` **only to the forward used for latency measurement** |
| `--latency_warmup` | `5` | Untimed inference warmup iterations; matches `measure_module_latency()` |
| `--latency_repetitions` | `20` | Timed inference repetitions; matches `measure_module_latency()` |
| `--output` | `runs/<script>` | Directory for metrics and checkpoints |

Training and accuracy evaluation remain eager even when `--compile_latency` is enabled. MAC counting also uses the eager model. Compilation is performed before the timed measurements; reported latency excludes initial compilation. Compile errors are surfaced, not silently replaced with eager results. Omitting the flag measures eager latency. There is no training compilation option.

### Understanding CPU activity during CUDA runs

`--device cuda` places model parameters, forward/backward inputs, and latency
measurements on CUDA; it does not move Python graph analysis or image decoding to
the GPU. `prune_finetune.py` first evaluates the pretrained model, then measures
its complexity and latency, builds the dependency graph, plans and applies
pruning, evaluates the compact model, and finally fine-tunes. The default baseline
evaluation covers all 50,000 validation images before pruning begins.

Dependency propagation and constraint search run on the CPU. Image decoding and
preprocessing also use the CPU. Both loaders default to eight persistent worker
processes, launched with `spawn`; PyTorch limits each worker's intra-op thread
pool to one. This avoids using the main process's full CPU thread pool for every
small per-image normalization. Workers are reused across dataset traversals;
training uses a new seeded permutation each traversal, and full validation stays
sequential. Batched dataset fetching reduces HF row-formatting calls. CUDA loaders
pin batches and transfers request `non_blocking=True`; this does not promise
complete overlap with model execution. Parent-process thread settings are unchanged.

Setting either worker count to zero is useful for debugging, but per-image tensor
operations then use the main process's thread pool and can again occupy all cores.
First-time Arrow preparation and initial `torch.compile` work can also occupy the
CPU before inference timing. CPU utilization alone therefore cannot identify the
device used by the model. Compare GPU utilization and completed images per second;
more workers are not automatically faster. See the official
[PyTorch data-loading guidance](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html).

`cpu_threads` records `torch.get_num_threads()`: the configured intra-op CPU
thread count. It is retained for both CPU and CUDA measurements as environment
metadata, not as a count of active threads or an indication of model placement.
It does not report DataLoader workers or compiler subprocesses.

The magnitude/Taylor entry prints stage boundaries, actual parameter and input
devices, planning duration, and training progress. Evaluation and measurement
helpers also report their devices. Inspect the latest stage and the `device`
field in `metrics.json` when diagnosing a slow run. Library magnitude and Taylor
metrics reduce parameter values on their device and transfer score batches to the
CPU; dependency propagation and ranking remain Python work on the CPU.

Every evaluation and training pass displays a `tqdm` progress bar on stderr,
including sparse training, projection/recovery epochs, iterative fine-tuning,
and the single-batch Taylor calibration. Bars identify the stage and device and
show completed batches, throughput, and estimated time remaining. The postfix
reports processed images and running sample-weighted loss; evaluation also shows
top-1 accuracy. Partial final batches count their actual images. For 50,000 images
with a batch size of 256, evaluation finishes at `196/196` with `images=50000`.
Bars close on completion or exceptions; JSON stage/epoch summaries remain on
stdout. In redirected logs, carriage-return progress updates may appear as
multiple lines, depending on the log viewer.

Both splits default to full data. Explicit sample limits enable small checks without downloading a different dataset. Training uses reproducible per-traversal shuffling with the weight enum's deterministic preprocessing, without a full augmentation recipe. A batch of 256 is a default, not a memory guarantee for every model: set the two batch sizes independently to fit the device, especially when training ViT or ResNet-50. Workers and their prefetched batches also consume host memory.

## End-to-end workflows

Run these commands from `examples/workflows` after the downloads above. Every command uses real pretrained weights, covers physical pruning and checkpoint restoration, and writes a separate output directory. Dataset sample limits are disabled: epoch-based training reads the full training split, Taylor calibration reads one batch, VBP calibration reads 16 batches, OSSCAR reads two batches per consumer by default, and stability search reads only as many batches as its stopping rule permits. Every recorded stage evaluates the full validation split. Training and validation each use their default batch size of 256. MACs and latency use the single-image example supplied by each workflow.

The commands use the default HF cache, data-loader settings, seed, and latency iteration counts. To use a separate snapshot, append `--data_dir /absolute/path/to/snapshot`. Each command enables `--compile_latency` to measure compiled inference; allow compilation time at each recorded stage. Training and accuracy evaluation remain eager.

### Default training duration and command overrides

| Entry | CLI default before physical pruning | CLI default after pruning | README command |
| --- | --- | --- | --- |
| `prune_finetune.py` | No training; magnitude scoring | 0 epochs | Separate prune-only and 1-epoch fine-tuning commands |
| `iterative_pruning.py` | 3 pruning rounds | 0 epochs per round | 2 rounds with 1 epoch after each round |
| `bn_sparsity.py` | 1 sparse-training epoch | 0 epochs | Adds 1 fine-tuning epoch |
| `group_sparsity.py` | 1 Group-Lasso epoch | 0 epochs | Adds 1 fine-tuning epoch; squared is an explicit alternative |
| `gate_pruning.py` | 1 gate-training epoch | 0 epochs | Adds 1 fine-tuning epoch |
| `soft_pruning.py` | 4 epochs: 1 warmup, 2 projection, 1 recovery | 0 epochs | Adds 1 fine-tuning epoch, for 5 total |
| `stability_pruning.py` | Up to 1,000 optimizer updates, with early stopping | 0 epochs | Adds 1 fine-tuning epoch |
| `variance_pruning.py` | Forward-only calibration on 16 training batches | 0 epochs | Prune-only ConvNeXt and ViT commands |
| `isomorphic_pruning.py` | Static within-family magnitude rankings; no training | 0 epochs | Prune-only ResNet and ViT commands |
| `osscar_pruning.py` | Two calibration batches per consumer, followed by quadratic reconstruction/search | 0 epochs | Prune-only ResNet and ViT commands |
| `vit_head_pruning.py` | Static head/FFN group magnitude; no training | 0 epochs | Prune-only ViT command |

These are bounded workflow demonstrations, not tuned accuracy-recovery recipes.
The 5% parameter target, alignment of 8, SGD learning rate of `0.001`, and sparse-loss
weight of `1e-4` are starting settings. In particular, L1 scales, group norms, and
squared group norms have different magnitudes; the common coefficient does not
make their regularization effects comparable. Adjust it using validation results
and the task/sparse loss contributions. Batch sizes remain 256 by default as an
explicit throughput setting, rather than adapting silently to each model or GPU.

### 1. Magnitude or Taylor pruning and fine-tuning

`prune_finetune.py` ranks dependency-group magnitudes, prunes, and optionally fine-tunes. Taylor uses task-only gradients from one training batch in evaluation mode; it does not update weights or BN statistics. Change `--metric magnitude` to `--metric taylor` to check this path.

`--selection static` is the default. `--selection dynamic` rescores the remaining
candidates after every accepted feasible addition using conditional parameter
regions; it can take considerably longer. It does not rerun the model or refresh
Taylor gradients. Neither policy is universally more accurate.

For a single magnitude-pruning pass **without training**:

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --metric magnitude \
  --finetune_epochs 0 --output runs/resnet18_prune_only
```

This evaluates the pretrained and pruned models, measures their complexity and
compiled latency, and saves and verifies the compact checkpoint. Only the
validation split is needed. It still evaluates all 50,000 validation images at
each stage; omitting training does not omit evaluation or latency compilation.

The basic example requests a 5% reduction from ResNet-18's 11,689,512 parameters,
giving a cap of 11,105,036. Each entry uses
`ParameterBudget.from_ratio(model, options.pruning_ratio)` to count unique
Parameters once and compute `floor(initial_params * (1 - pruning_ratio))`.
The converted `max_params` is saved in the run configuration. Alignment can
require additional removal. The policy allocates removals across eligible blocks instead of prescribing an
equal ratio per block. This is a demonstration target, not an accuracy guarantee.
Compare full-validation scores and fine-tune when needed.

To follow the pruning pass with one epoch of fine-tuning:

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --metric magnitude \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_prune_finetune
```

For a classic redundancy-based alternative, FPGM (CVPR 2019) ranks each output
filter by the sum of its Euclidean distances to the other filters in that layer.
Distances use the original **signed** weights. The local `GeometricMedian` metric
shows how to provide a model-specific signal through the library metric contract:

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --metric geometric_median --selection static \
  --finetune_epochs 0 --output runs/resnet18_geometric_median
```

`geometric_median` requires static selection. It also accepts ViT FFN rows as
filters, but this extension and global parameter budgeting are adaptations;
the original FPGM layerwise soft-pruning/training schedule is not reproduced.
Pairwise distances require quadratic work in each layer's output width. The
implementation bounds temporary distance matrices with tiles; it does not
normalize away differences between layers or promise universally comparable scores.

### 2. Iterative pruning

`iterative_pruning.py` interpolates absolute parameter caps from the initial model
to the cap converted from `--pruning_ratio` over `--rounds`. The initial parameter
count is fixed; the ratio is not reapplied to each shrinking model. It rebuilds
the graph and recomputes scores after each round. An alignment-induced overshoot
may make the next round a no-op. An
unmet round target raises an error; earlier completed rounds remain applied.

```bash
python iterative_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --rounds 2 \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_iterative_pruning
```

### 3. BN-scale sparse training

`bn_sparsity.py` regularizes every residual block's `bn1` scales with L1, ranks channels by those scales, then prunes. `--sparse_epochs 0` skips sparse training; `--sparse_loss_weight` is the fixed sparse-loss multiplier (default: `1e-4`).

```bash
python bn_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_bn_sparsity
```

### 4. Dependency-group sparse training

`group_sparsity.py` regularizes each candidate channel's complete dependent parameter group. `--penalty lasso` (default) applies Group Lasso with a fixed `--sparse_loss_weight`. `--penalty squared` reselects low-magnitude groups before training and every `--selection_interval_steps` optimizer updates (default: `100`), and increases their squared-L2 penalty each update. Selection uses the library's normalized `GroupMagnitude` after any explicit training updates.

For squared penalties, `--sparse_loss_weight` is the final coefficient (default: `1e-4`). With one-based optimizer step `s` and `T = sparse_epochs * len(train_loader)`, `--sparsity_schedule cosine` (default) uses `weight * (1 - cos(pi * s / T)) / 2`; `linear` uses `weight * s / T`. Cosine starts and ends more gradually; this is a scheduling choice, not a claim of better accuracy. Both counters continue across epoch boundaries and reselections. A one-batch run uses the final coefficient; zero sparse epochs skip the schedule. Schedule and reselection options apply only to `squared`.

All sparse workflows optimize `cross_entropy + current_weight * sparse_loss`. Cross-entropy is averaged over samples; sparse penalties are unnormalized sums. Group and stability logs include the last `sparse_loss_weight`, raw `sparse_loss`, and mean `weighted_sparse_loss` so the actual contribution remains visible as the weight changes. Post-pruning fine-tuning uses cross-entropy alone.

```bash
python group_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --penalty lasso \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_group_sparsity
```

To exercise progressive regularization, replace `--penalty lasso` with
`--penalty squared --sparsity_schedule cosine --selection_interval_steps 100`.

### 5. Soft pruning and gradual norm decay

`soft_pruning.py` first trains for one epoch to establish momentum. Each projection cycle then zeros selected regions (`--operation zero`) or reduces their L2 norm (`--operation decay`, default) after optimizer steps. One recovery epoch between cycles permits regrowth. Before physical pruning, the total is `1 + cycles * projection_epochs + (cycles - 1)` epochs: 4 with the defaults `cycles=2`, `projection_epochs=1`. The final selection is revalidated before physical pruning. `--cycles` and `--projection_epochs` must be positive.

```bash
python soft_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --operation decay --cycles 2 --projection_epochs 1 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_soft_pruning
```

### 6. Gate training and pruning

`gate_pruning.py` inserts unit-valued gates after ResNet `bn1` or the ViT FFN activation, trains the model with an L1 penalty on gate scales, and uses `GateMagnitude` to rank their affected regions. Both ordinary model weights and trainable gates participate in optimization. Final selection is static `Greedy`, separate from gate training. Retained gate values and placements are preserved by checkpoint restoration.

```bash
python gate_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --sparse_epochs 1 --sparse_loss_weight 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_gate_pruning
```

### 7. Stability-driven pruning

`stability_pruning.py` alternates magnitude selection and squared-L2 regularization. `--window 2` requires three selections to obtain two consecutive Jaccard similarities. `--max_selection_checks` (default: `11`) includes the initial check, with `--selection_interval_steps` optimizer updates between checks (default: `100`). The maximum sparse-training duration is therefore `(max_selection_checks - 1) * selection_interval_steps`: `1,000` updates by default. Stability can first stop training after 200 updates, before reaching the limit. Setting one check skips sparse training; a limit smaller than `window + 1` cannot establish stability and stops only at the limit.

The sparse weight increases each optimizer update using the same `cosine` (default) or `linear` formula as above, with `T` set to the maximum training duration. Search ends when the mean of the latest `window` adjacent-selection similarities reaches `--threshold` (default: `0.99`) or the check limit. This measures selection stability, not accuracy recovery. Early stopping does not rescale the schedule or force it to reach `--sparse_loss_weight` (default: `1e-4`). Reports distinguish `selection_checks` from actual `training_steps`. Checks do not restart the data iterator; the training split is reopened only when exhausted. No training occurs after the final check. The interval and check limit are illustrative settings, not tuned ImageNet hyperparameters.

```bash
python stability_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --max_selection_checks 11 --selection_interval_steps 100 --window 2 --threshold 0.99 \
  --sparse_loss_weight 0.0001 --sparsity_schedule cosine \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_stability_pruning
```

### 8. Variance-Based Pruning (ICCV 2025)

`variance_pruning.py` implements VBP's activation-variance criterion and bias
compensation for ViT and ConvNeXt MLPs. It records the input to each MLP's second
Linear layer in evaluation mode, pools batch/token/spatial positions, and ranks
hidden positions by their sample variance. Calibration uses the **training**
split without labels or backward; validation is reserved for reporting accuracy.
Streaming moments retain channel vectors, not complete activations.

For each removed hidden position, its calibration mean multiplied by the old
consumer weight column is added to that consumer's bias. This represents replacing
the removed activation with its mean. It is not an assertion of equivalence to
the original network. ResNet convolutions with spatial padding are deliberately
outside this workflow: their missing mean contribution is generally not a spatially
constant bias. The CNN example therefore uses ConvNeXt's dense MLPs.

```bash
python variance_pruning.py \
  --model convnext_tiny --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --calibration_batches 16 \
  --finetune_epochs 0 --output runs/convnext_tiny_variance
```

```bash
python variance_pruning.py \
  --model vit_b_32 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 --calibration_batches 16 \
  --finetune_epochs 0 --output runs/vit_b_32_variance
```

`--calibration_batches 16` reads up to 4,096 training images at the default batch
size; `0` calibrates on the full training loader. This bounds example runtime and
differs from the official full-loader default. `--train_samples` still defaults
to the full Arrow-backed split; calibration does not change its cache policy.
The report records the actual observation count for every MLP, including all
ViT tokens. Calibration sample size affects the ranking and must be reported.

Optional fine-tuning uses `--finetune_epochs`, AdamW with `--lr 0.000015`,
`--weight_decay 0.01`, and a cosine schedule updated each optimizer step.
Training is task-only; teacher distillation and the paper's full augmentation
recipe are not included. The framework's whole-model parameter target and width
alignment also differ from the paper's neuron-ratio experiments.

The saved compact checkpoint includes corrected biases. A structural plan alone
does **not** encode this value correction; replaying only that plan would omit
part of VBP. Use the workflow's `model.pt` for restoration.

### 9. Isomorphic Pruning

`isomorphic_pruning.py` discovers graph-declared candidates and groups dependency
structures by operator types, affected dimensions, connectivity and parameter
sharing. It ranks magnitude scores independently within each family, then selects
each family's least-important fraction. Scores from different families are not
mixed into a global ranking. Equivalent dependency closures count once.

An outer search increases the common channel ratio until a jointly executable
proposal meets the whole-model parameter target. Retained widths round down to
the configured granularity while preserving a nonempty domain. Actual removals
may therefore exceed the unrounded family quotas. Rejected proposals do not
mutate the model. A homogeneous candidate universe legitimately has one family;
the example does not manufacture extra families from layer names or stages.
`--max_trials` defaults to 10,000 ratio breakpoints, including repeated rounded
requests; `planning_trials` separately counts actual joint attempts. A limit
failure reports the recent blockers and does not apply a partial allocation.

```bash
python isomorphic_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --finetune_epochs 0 --output runs/resnet18_isomorphic
```

```bash
python isomorphic_pruning.py \
  --model vit_b_32 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --finetune_epochs 0 --output runs/vit_b_32_isomorphic
```

The example exercises the custom `Strategy` interface. Its metric is replaceable;
magnitude scoring is not the paper's Taylor-calibrated experimental recipe.
Optional fine-tuning is task-only. Family reports and exclusions describe the
actual scope; attention pruning and arbitrary custom operators are not assumed
supported merely because their candidates can be discovered.

### 10. OSSCAR reconstruction and local search

`osscar_pruning.py` first allocates aligned hidden widths using a common channel
removal fraction that reaches the whole-model parameter cap. This allocation uses
actual dependency plans and is separate from reconstruction-error optimization.
It then processes each consumer in forward order: collect current-model inputs
and original dense-teacher targets, solve grouped least squares, delete channels,
try bounded remove/restore swaps, physically prune and copy reconstructed weights.
The next consumer is calibrated against the already compacted model.

```bash
python osscar_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --calibration_batches 2 --calibration_rows 4096 --damping 0.01 \
  --prune_batch 8 --swap_steps 5 --finetune_epochs 0 \
  --output runs/resnet18_osscar
```

```bash
python osscar_pruning.py \
  --model vit_b_32 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --calibration_batches 2 --calibration_rows 4096 --damping 0.01 \
  --prune_batch 8 --swap_steps 5 --finetune_epochs 0 \
  --output runs/vit_b_32_osscar
```

Calibration uses the Arrow-backed training split without labels or backward.
`--calibration_batches` is a **per-consumer** limit; `0` traverses the full loader
for every consumer. `--calibration_rows` caps uniformly spaced spatial/token
observations per batch and uses identical positions for student and teacher.
The full image dataset remains available; this is not a replacement cache or a
validation-data calibration. Statistics use float64; memory is quadratic in
the consumer's input width times kernel area. Only one consumer's statistics are
kept at a time, and convolution unfolding processes one image at a time.

`--damping` regularizes toward original weights, with a default scale of 0.01
times the average Gram diagonal. `--prune_batch` controls channels removed per
search update; it is independent of retained-width `--granularity`.
`--swap_steps` bounds attempts to exchange a retained and a removed channel;
search stops early without an objective improvement. The reconstruction report
records observations, retained indices, fixed-cardinality objective history and
swap attempts. This is a bounded local optimizer, not a globally optimal solver.

Eligible ordinary Conv2d and Linear chains are discovered from actual dataflow,
including adjacent overlapping pairs; ResNet and ViT are test models, not path
matching rules. Excluded producers and their reasons are saved in the report.
Grouped/depthwise consumers and attention
heads are outside this example. The CNN sequential teacher-target formulation
is retained, while width allocation, sampling and search settings are explicit
experimental adaptations. Optional task-only SGD fine-tuning uses the common
`--finetune_epochs` and `--lr` options.

The checkpoint contains **refitted weights**. Structural plans alone omit these
value changes. Each layer is a separate validated apply operation; if a later
layer fails, completed earlier layers remain changed in memory. Optimizers are
created only after the full reconstruction sequence.

### 11. ViT attention heads and FFN channels

`vit_head_pruning.py` explicitly replaces torchvision's native self-attention
with packed Q/K/V Linear projections, SDPA, and an output Linear projection.
Before pruning, it preserves the original parameters and computation up to
floating-point differences. This conversion is part of the example, not an
automatic change to the library's native MultiheadAttention rule.

Each attention candidate removes one complete head: the corresponding Q/K/V
rows, their biases, and output-projection columns. Head count is inferred from
the compact width; per-head width, attention scale, embedding width, residual
connections and token positions remain unchanged. FFN candidates remove hidden
neurons and their downstream input columns. `--granularity` applies only to FFN
widths. At least one head and a nonempty FFN remain in every block.

```bash
python vit_head_pruning.py \
  --model vit_b_16 --device cuda --compile_latency \
  --pruning_ratio 0.05 --granularity 8 \
  --finetune_epochs 0 --output runs/vit_b_16_heads_and_ffn
```

Static Greedy ranks normalized group energy toward the whole-model
parameter cap. Both kinds of candidate are available; no quota forces every
layer or both kinds to shrink. `metrics.json` records actual head counts and FFN
widths for every stage. This is an explicit magnitude baseline, not a paper
reproduction or a demonstrated improvement over FFN-only pruning. Add
`--finetune_epochs 1 --lr 0.001` for optional task-loss SGD recovery.

The adapter implements unmasked ViT self-attention, not general cross-attention,
GQA or token pruning. It preserves training-mode attention dropout. Restore using
the same adapted skeleton, rather than the original native-MHA model:

```python
from imagenet_models import make_head_prunable_model
from torch_kirigami.pruning import load_checkpoint

model = load_checkpoint(
    make_head_prunable_model("vit_b_16", pretrained=False),
    "runs/vit_b_16_heads_and_ffn/model.pt",
    map_location="cuda",
).eval()
```

## Additional models

For ResNet-34/50, change `--model`; the parameter cap is computed from that model's
initial count. The ViT command below includes every encoder block and requests a
0.3% parameter reduction (ViT-B/32 starts at 88,224,232). Even a small reduction
analyzes all candidates and may take considerably longer to plan than the CNN
workflows:

```bash
python prune_finetune.py \
  --model vit_b_32 --device cuda --compile_latency \
  --pruning_ratio 0.003 --granularity 8 --metric magnitude \
  --finetune_epochs 1 --lr 0.001 --output runs/vit_b_32_prune_finetune
```

## Results and verification

Each stage records validation cross-entropy, top-1/top-5 accuracy, change from
baseline in percentage points, parameter count, MACs, and `latency_ms`. MACs and
latency describe **one image (batch=1)**. Each workflow passes a batch=1 example
to the shared measurement helper, which uses it unchanged for both measurements.
`input_shape` records their common input shape.
Baseline and compact measurements use identical batch size, dtype, device, and
compilation settings. Check `unsupported_ops` before treating MAC counts as
complete. Neither MACs nor latency is a pruning target in these workflows.
Pruning stages report `max_params`, `before_params`, `after_params` and `target_met`.
Greedy-based workflows also report `planning_trials` and `planning_limit_reached`;
Isomorphic reports family quotas, ratio trials and rejected allocations; OSSCAR
reports its per-consumer reconstruction history. Counts include all unique
Parameters, including retained learned gates; buffers are excluded. Meeting the
parameter cap does not imply a particular latency or accuracy improvement.

The default Greedy limit is 10,000 tentative joint dependency queries, including
fallback completion attempts. It does not count training steps or removed
channels; scoring queries are separate. The strategy first combines candidates
using the known divisibility and balance constraints, then checks the complete
batch jointly. A width of 64 aligned to 8 can submit eight selected channels in
one trial. Unpredicted joint effects still require further checks, and proven
channel-budget violations are skipped without a query. Parameter-budget search
allows intermediate requests above the final cap and stops at the first verified
combination meeting it. If the cap remains unmet, planning raises an error before
any new model mutation or compact checkpoint is produced. The error reports the
count reached on this greedy path, the limit, and recent blockers; it is not a
proof that no other selection could succeed. Increasing
`Greedy(..., max_trials=...)` in the example permits more search, but cannot make
an unsupported structural change executable.

Each output directory contains:

- `metrics.json`: effective CLI settings, data/weight metadata, and stage results.
- `model.pt`: the compact model checkpoint.
- `training.pt`: optimizer, algorithm, configuration, and RNG state; iterative pruning also saves initial and target parameter counts.

A successful run ends with a checkpoint-verification message. Restoration is checked against the in-memory compact model. Optimizers are recreated after parameter replacement. The saved training state is not a general resume CLI. These examples demonstrate component composition; their optimizer, preprocessing, and regularization choices do not constitute complete paper reproductions or established accuracy/speedup claims.

For infrastructure and lifecycle regression tests, run from the repository root:

```bash
uv pip install --group dev
.venv/bin/pytest tests/integration/test_pretrained_examples.py \
  tests/integration/test_geometric_metric.py tests/integration/test_variance_workflow.py \
  tests/integration/test_variance_operators.py \
  tests/integration/test_isomorphic_workflow.py tests/integration/test_osscar_workflow.py \
  tests/integration/test_vit_head_workflow.py \
  --require-cuda -q
```

These tests use local synthetic datasets and controlled model fixtures. The commands above additionally exercise installed pretrained weights and real ImageNet shards. Neither a short smoke run nor the test suite guarantees accuracy at an arbitrary pruning ratio.
