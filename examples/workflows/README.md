# Pretrained ImageNet pruning workflows

Seven standalone entries demonstrate structural pruning, sparse training, fine-tuning, and checkpoint restoration. Each file owns its CLI and training/pruning steps. Three infrastructure modules are shared: `imagenet_data.py` (data and accuracy), `imagenet_models.py` (official weights and the ViT adapter), and `model_metrics.py` (complexity and latency). Common CLI options have the same meaning across entries; they are documented below rather than hidden in a shared parser.

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

| `--model` | Official weights | Candidate channels |
| --- | --- | --- |
| `resnet18` (default) | `ResNet18_Weights.IMAGENET1K_V1` | Every BasicBlock's `conv1` output / `conv2` input |
| `resnet34` | `ResNet34_Weights.IMAGENET1K_V1` | Every BasicBlock's `conv1` output / `conv2` input |
| `resnet50` | `ResNet50_Weights.IMAGENET1K_V2` | Every Bottleneck's `conv1` output / `conv2` input |
| `vit_b_16` | `ViT_B_16_Weights.IMAGENET1K_V1` | Every encoder block's FFN intermediate width |
| `vit_b_32` | `ViT_B_32_Weights.IMAGENET1K_V1` | Every encoder block's FFN intermediate width |

BN sparsity accepts the three ResNet models; the other entries accept all five. Residual output widths, ViT hidden width and attention heads, and the 1,000-class classifier remain fixed. ViT uses an explicit forward adapter preserving torchvision's weights and computation while exposing data flow to FX. Preprocessing comes from the selected weight enum; all listed configurations use 224 × 224 crops.

Every workflow enumerates supported pruning positions throughout the entire model. There is no layer-selection CLI and no first-layer default. Whole-model coverage means all positions in the table participate; protected dimensions and axes outside the method's scope remain unchanged. The dependency graph covers the entire model, and a selected channel can affect other parameters through dependencies.

Each entry passes every supported producer path to `Granularity` and
`discover_candidates`. Granularity constrains retained widths; it does not package
adjacent channels. `--pruning_ratio` is a reduction fraction of the entire model's
parameter count, including fixed parts, rather than a per-layer channel ratio.
Whole-model ViT candidate spaces can take substantially longer to plan than
ResNets; reducing dataset samples does not reduce dependency-planning work.
Planning fails explicitly if bounded search cannot reach the parameter target.

## Common CLI options

All multiword workflow options use underscores. The CLI accepts complete option names, without abbreviations or aliases for old spellings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `resnet18` | Model from the table above; BN sparsity restricts choices to ResNets |
| `--data_dir` | HF cache | Local dataset snapshot directory |
| `--device` | `cuda` | Training, accuracy evaluation, graph capture, and measurement device |
| `--train_samples` | `0` | Available training images per dataset traversal; `0` makes the full split available. Taylor reads one batch; stability search may stop before completing a traversal |
| `--val_samples` | `0` | Validation images per recorded stage; `0` evaluates the complete validation split |
| `--train_batch_size` | `256` | Training batch size; also the Taylor calibration batch size |
| `--val_batch_size` | `256` | Accuracy-evaluation batch size and the fixed synthetic inference batch for MAC/latency measurement |
| `--train_workers` | `8` | Training/Taylor DataLoader worker processes; `0` performs loading in the main process |
| `--val_workers` | `8` | Validation DataLoader worker processes; `0` performs loading in the main process |
| `--seed` | `7` | Model/training RNG and shuffled data selection seed |
| `--pruning_ratio` | `0.05` | Fraction of initial whole-model parameters to remove; includes fixed/frozen parameters and inserted gates. Iterative pruning uses the final cumulative reduction |
| `--granularity` | `8` | Retained producer widths must be divisible by this factor; `1` adds no alignment constraint |
| `--finetune_epochs` | `0` | Task-only epochs after physical pruning; iterative pruning applies this after every round |
| `--lr` | `0.001` | SGD learning rate; momentum is `0.9` |
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
field in `metrics.json` when diagnosing a slow run. Producer scores are transferred
to the CPU once per axis, rather than synchronizing CUDA for each candidate.

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

Run these commands from `examples/workflows` after the downloads above. Every command uses real pretrained weights, covers physical pruning and checkpoint restoration, and writes a separate output directory. Dataset sample limits are disabled: epoch-based training reads the full training split, Taylor calibration reads one batch, and stability search reads only as many batches as its stopping rule permits. Every recorded stage evaluates the full validation split. Training and validation each use their default batch size of 256. Latency measurement uses the same validation batch size.

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

These are bounded workflow demonstrations, not tuned accuracy-recovery recipes.
The 5% parameter target, alignment of 8, SGD learning rate of `0.001`, and sparse-loss
weight of `1e-4` are starting settings. In particular, L1 scales, group norms, and
squared group norms have different magnitudes; the common coefficient does not
make their regularization effects comparable. Adjust it using validation results
and the task/sparse loss contributions. Batch sizes remain 256 by default as an
explicit throughput setting, rather than adapting silently to each model or GPU.

### 1. Magnitude or Taylor pruning and fine-tuning

`prune_finetune.py` ranks producer channels, prunes, and optionally fine-tunes. Taylor uses task-only gradients from one training batch in evaluation mode; it does not update weights or BN statistics. Change `--metric magnitude` to `--metric taylor` to check this path.

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

`group_sparsity.py` regularizes each candidate channel's complete dependent parameter group. `--penalty lasso` (default) applies Group Lasso with a fixed `--sparse_loss_weight`. `--penalty squared` reselects low-magnitude groups before training and every `--selection_interval_steps` optimizer updates (default: `100`), and increases their squared-L2 penalty each update. Final selection uses producer-weight magnitude.

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

`gate_pruning.py` inserts unit-valued gates after ResNet `bn1` or the ViT FFN activation, regularizes their scales, and ranks channels by gate magnitude. Retained gate values and placements are preserved by checkpoint restoration.

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
latency describe the **whole configured validation batch**, not one image.
Baseline and compact measurements use identical batch size, dtype, device, and
compilation settings. Check `unsupported_ops` before treating MAC counts as
complete. Neither MACs nor latency is a pruning target in these workflows.
Pruning stages report `max_params`, `before_params`, `after_params`, `target_met`,
`planning_trials`, and `planning_limit_reached`. Counts include all unique
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

A successful run ends with `Checkpoint verified; results: ...`. Restoration is checked against the in-memory compact model. Optimizers are recreated after parameter replacement. The saved training state is not a general resume CLI. These examples demonstrate component composition; their simple SGD, preprocessing, and regularization choices do not constitute complete paper reproductions or established accuracy/speedup claims.

For infrastructure and lifecycle regression tests, run from the repository root:

```bash
uv pip install --group dev
.venv/bin/pytest tests/integration/test_pretrained_examples.py --require-cuda -q
```

These tests use local synthetic datasets and controlled model fixtures. The commands above additionally exercise installed pretrained weights and real ImageNet shards. Neither a short smoke run nor the test suite guarantees accuracy at an arbitrary pruning ratio.
