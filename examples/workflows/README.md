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

Each entry passes every supported producer path to `Granularity` and `discover_candidates`. Granularity constrains retained widths; it does not package adjacent channels. Ratios count channels in those domains, not whole-model parameters or MACs. Whole-model ViT candidate spaces can take substantially longer to plan than ResNets; reducing dataset samples does not reduce dependency-planning work. The strategy reports any shortfall when its bounded search cannot reach the target.

## Common CLI options

All multiword workflow options use underscores. The CLI accepts complete option names, without abbreviations or aliases for old spellings.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `resnet18` | Model from the table above; BN sparsity restricts choices to ResNets |
| `--data_dir` | HF cache | Local dataset snapshot directory |
| `--device` | `cuda` | Training, accuracy evaluation, graph capture, and measurement device |
| `--train_samples` | `0` | Training images per epoch; `0` reads the complete training split. Used only when training or Taylor calibration is needed |
| `--val_samples` | `0` | Validation images per recorded stage; `0` evaluates the complete validation split |
| `--train_batch_size` | `256` | Training batch size; also the Taylor calibration batch size |
| `--val_batch_size` | `256` | Accuracy-evaluation batch size and the fixed synthetic inference batch for MAC/latency measurement |
| `--val_workers` | `0` | Validation DataLoader worker processes. Training streams in the main process to avoid duplicating the iterable dataset |
| `--seed` | `7` | Model/training RNG and shuffled data selection seed |
| `--channel_pruning_ratio` | `0.125` for basic pruning; `0.25` otherwise | Maximum fraction of candidate-domain channels to delete, not a parameter or MAC ratio; iterative pruning uses the final cumulative target |
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
preprocessing also use the CPU; with the default zero validation workers, they
run in the main process and can leave the GPU waiting. `--val_workers 4` enables
parallel validation loading. Training currently streams in the main process.
Initial `torch.compile` work can also occupy the CPU before GPU latency timing.
CPU utilization alone therefore cannot identify the device used by the model.

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

Both splits default to full data. Explicit sample limits enable small checks without downloading a different dataset. Training uses a fixed shuffled stream with the weight enum's preprocessing, without a full augmentation recipe. A batch of 256 is a default, not a memory guarantee for every model: set the two batch sizes independently to fit the device, especially when training ViT or ResNet-50.

## End-to-end workflows

Run these commands from `examples/workflows` after the downloads above. Every command uses real pretrained weights and the full ImageNet splits needed by the task, covers physical pruning and checkpoint restoration, and writes a separate output directory. Training and validation each use their default batch size of 256. Latency measurement uses the same validation batch size.

The commands use the default HF cache, data-loader settings, seed, and latency iteration counts. To use a separate snapshot, append `--data_dir /absolute/path/to/snapshot`. Each command enables `--compile_latency` to measure compiled inference; allow compilation time at each recorded stage. Training and accuracy evaluation remain eager.

### 1. Magnitude or Taylor pruning and fine-tuning

`prune_finetune.py` ranks producer channels, prunes, and optionally fine-tunes. Taylor uses task-only gradients from one training batch in evaluation mode; it does not update weights or BN statistics. Change `--metric magnitude` to `--metric taylor` to check this path.

For a single magnitude-pruning pass **without training**:

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.125 --granularity 8 --metric magnitude \
  --finetune_epochs 0 --output runs/resnet18_prune_only
```

This evaluates the pretrained and pruned models, measures their complexity and
compiled latency, and saves and verifies the compact checkpoint. Only the
validation split is needed. It still evaluates all 50,000 validation images at
each stage; omitting training does not omit evaluation or latency compilation.

The basic example uses a 12.5% channel cap instead of 25% to make the first
pruning pass less aggressive. On ResNet-18 this permits aligned removals in every
candidate domain, including width 64 with granularity 8. This is a demonstration
setting, not an accuracy guarantee: magnitude-only pruning can still cause a
substantial immediate accuracy loss. Compare the reported full-validation scores
and fine-tune when needed.

To follow the pruning pass with one epoch of fine-tuning:

```bash
python prune_finetune.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.125 --granularity 8 --metric magnitude \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_prune_finetune
```

### 2. Iterative pruning

`iterative_pruning.py` increases the cumulative target over `--rounds`. It rebuilds the graph, recomputes scores, and deducts actual previous removals from the original-width budget. An underfilled round is not counted as completed pruning.

```bash
python iterative_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 --rounds 2 \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_iterative_pruning
```

### 3. BN-scale sparse training

`bn_sparsity.py` regularizes every residual block's `bn1` scales with L1, ranks channels by those scales, then prunes. `--sparse_epochs 0` skips sparse training; `--strength` controls the regularizer.

```bash
python bn_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 \
  --sparse_epochs 1 --strength 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_bn_sparsity
```

### 4. Dependency-group sparse training

`group_sparsity.py` regularizes each candidate channel's complete dependent parameter group. `--penalty lasso` applies Group Lasso; `--penalty squared` reselects low-magnitude groups each epoch and increases their squared-L2 penalty. Final selection uses producer-weight magnitude.

```bash
python group_sparsity.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 --penalty lasso \
  --sparse_epochs 1 --strength 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_group_sparsity
```

### 5. Soft pruning and gradual norm decay

`soft_pruning.py` first trains for one epoch to establish momentum. Each projection cycle then zeros selected regions (`--operation zero`) or reduces their L2 norm (`--operation decay`) after optimizer steps. Recovery between cycles permits regrowth. The final selection is revalidated before physical pruning. `--cycles` and `--projection_epochs` must be positive.

```bash
python soft_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 \
  --operation decay --cycles 2 --projection_epochs 1 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_soft_pruning
```

### 6. Gate training and pruning

`gate_pruning.py` inserts unit-valued gates after ResNet `bn1` or the ViT FFN activation, regularizes their scales, and ranks channels by gate magnitude. Retained gate values and placements are preserved by checkpoint restoration.

```bash
python gate_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 \
  --sparse_epochs 1 --strength 0.0001 --finetune_epochs 1 --lr 0.001 \
  --output runs/resnet18_gate_pruning
```

### 7. Stability-driven pruning

`stability_pruning.py` alternates magnitude selection and squared-L2 regularization. `--window 2` requires three selections to compare two adjacent Jaccard similarities. Search ends at `--threshold` or `--search_steps`, then generates the final plan. Training occurs between selection checks; reports distinguish checks from epochs.

```bash
python stability_pruning.py \
  --model resnet18 --device cuda --compile_latency \
  --channel_pruning_ratio 0.25 --granularity 8 \
  --search_steps 3 --window 2 --threshold 0.99 --strength 0.0001 \
  --finetune_epochs 1 --lr 0.001 --output runs/resnet18_stability_pruning
```

## Additional models

For ResNet-34/50, replace `--model` in a workflow command with `resnet34` or `resnet50`. The ViT command below includes every encoder block. Its small ratio caps deletion at eight FFN channels per block; inspect the actual per-axis removals and shortfall rather than assuming the search reaches every cap. Even this small-ratio run analyzes all candidates and may take considerably longer to plan than the CNN workflows:

```bash
python prune_finetune.py \
  --model vit_b_32 --device cuda --compile_latency \
  --channel_pruning_ratio 0.003 --granularity 8 --metric magnitude \
  --finetune_epochs 1 --lr 0.001 --output runs/vit_b_32_prune_finetune
```

## Results and verification

Each stage records validation cross-entropy, top-1/top-5 accuracy, change from baseline in percentage points, parameter count, MACs, and `latency_ms`. MACs and latency describe the **whole configured validation batch**, not one image. Baseline and compact measurements use identical batch size, dtype, device, and compilation settings. Check `unsupported_ops` before treating MAC counts as complete. Pruning stages also report requested/actual channel reductions, shortfall, `planning_trials`, and `planning_limit_reached`. A reached strategy limit can leave some blocks unchanged even though every supported block was included in discovery; inspect the per-axis `removed` values instead of assuming the requested ratio was achieved everywhere.

The default Greedy limit is 10,000 tentative joint dependency queries, including
fallback completion attempts. It does not count training steps or removed
channels; scoring queries are separate. The strategy first combines candidates
using the known divisibility and balance constraints, then checks the complete
batch jointly. A width of 64 aligned to 8 can submit eight selected channels in
one trial. Unpredicted joint effects still require further checks, and proven
budget violations are skipped without a query. A limit hit retains only complete,
executable combinations. Increasing
`Greedy(..., max_trials=...)` in the example permits more search, but cannot make
an incompatible ratio and granularity feasible.

Each output directory contains:

- `metrics.json`: effective CLI settings, data/weight metadata, and stage results.
- `model.pt`: the compact model checkpoint.
- `training.pt`: optimizer, algorithm, configuration, and RNG state; iterative pruning also saves cumulative accounting.

A successful run ends with `Checkpoint verified; results: ...`. Restoration is checked against the in-memory compact model. Optimizers are recreated after parameter replacement. The saved training state is not a general resume CLI. These examples demonstrate component composition; their simple SGD, preprocessing, and regularization choices do not constitute complete paper reproductions or established accuracy/speedup claims.

For infrastructure and lifecycle regression tests, run from the repository root:

```bash
uv pip install --group dev
.venv/bin/pytest tests/integration/test_pretrained_examples.py --require-cuda -q
```

These tests use local synthetic datasets and controlled model fixtures. The commands above additionally exercise installed pretrained weights and real ImageNet shards. Neither a short smoke run nor the test suite guarantees accuracy at an arbitrary pruning ratio.
