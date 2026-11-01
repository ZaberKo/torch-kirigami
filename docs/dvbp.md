# Denoised variance pruning and OB2C

This implementation adapts
[Denoised Variance-Based Pruning with Optimal Brain Bias Compensation](https://arxiv.org/abs/2608.17657)
to the library's current plan/apply lifecycle. See the
[official implementation](https://github.com/geontackee/DVBP_OB2C) for the reference
denoising and compensation pipeline.

## Calibration and spectral ranking

`ActivationCovariance` merges centered second moments in bounded activation-row
chunks. It retains a mean vector and full covariance accumulator, not activation
history. Storage remains O(d²) per hidden width d. `count` counts flattened batch,
token and spatial observations; `images` separately counts independent images.
The Marchenko-Pastur aspect ratio is d/images, not d/token-count.

`collect_covariances` observes actual consumer inputs through temporary hooks,
in evaluation mode, without reading labels or changing gradients. Every consumer
must execute once per batch. Empty/short calibration and nonfinite inputs fail;
hooks and mixed train/eval flags are restored on failure.

`denoise_covariance` computes eigenvalues, estimates noise from the observed
median and theoretical MP continuous-bulk median, and retains components above
the MP upper edge. At least the largest component is retained, as in the
reference. The theoretical median uses deterministic quadrature without SciPy.
Numerically negative eigenvalues are clamped only within the precision tolerance;
materially non-PSD covariance fails. Inputs containing NaN/Inf are rejected rather
than silently sanitized.

The score is `(1 - p) * diag(C) + p * diag(C_denoised)`.
`DenoisedVariance` sums fixed scores over newly removed hidden positions in
original coordinates, without layer normalization or weight-magnitude weighting.

## Compensation

Given retained indices K and removed indices P, `solve_ob2c` computes

```text
delta = W[:,P] H[P,K] inverse(H[K,K])
W_compact = W[:,K] + delta
b_compact = b + W[:,P] mean[P] - delta mean[K]
```

The implementation uses a Cholesky solve rather than explicit inverses. Singular
retained systems fail explicitly. Default H is the denoised covariance with
relative diagonal damping, matching the authors' code. `repair_covariance="raw"`
instead uses the centered sample covariance from the paper's local reconstruction
objective. An explicit absolute ridge floor handles all-constant activations.
Both regularization terms and spectral diagnostics are reported.

`prepare_ob2c_updates` computes compact values before structural application and
rejects unrelated parameter deletions. `Pruner.apply` then performs structural
changes, and `apply_ob2c_updates` validates all values before copying them.
The structural plan alone does not store corrections: save the corrected compact
checkpoint. Unit tests compare the solve with independently constructed augmented
least-squares and ridge systems, including preservation of the calibration mean.

## Scope and differences from the reference

The workflow uses the current VBP pair discovery and evaluation-only operator
registry for torchvision ViT and ConvNeXt dense MLPs. It preserves block input,
output, residual interfaces and classifier. It does not support attention-head
pruning or Swin models.

Statistics and reconstruction are per MLP pair. The official ConvNeXt code uses
stage-concatenated statistics; this implementation intentionally retains the
library's independent-pair scope and does not reproduce that cross-block
covariance construction. Earlier pruning may change later activation statistics;
all pairs use one fixed calibration snapshot, without an end-to-end equivalence
claim. These adaptations must be retained when reporting comparisons.

`--pruning_ratio` is a whole-model parameter reduction. `--score_ratio` is a
separate coefficient p for hybrid scoring, default 0.5. In the paper p corresponds
to the MLP pruning ratio; do not equate that ratio with whole-model parameter
reduction. Use explicit channel budgets through the public APIs for experiments
requiring the paper's hidden-channel target. The workflow uses aligned widths
and parameter budgeting, not the paper's exact global hidden-channel allocation.

## Run and recover with task loss

```bash
python examples/workflows/dvbp_pruning.py \
  --model vit_b_16 --device cuda --data_dir /path/to/imagenet-snapshot \
  --pruning_ratio 0.15 --score_ratio 0.5 --calibration_batches 32 \
  --train_batch_size 16 --val_batch_size 32 --statistics_device cpu \
  --finetune_epochs 10 --output runs/dvbp_vit
```

The same entry supports `convnext_tiny` and `vit_b_32`. Float32 accumulators save
memory; `--statistics_dtype float64` improves precision. Spectral calculations and
reconstruction run in float64 on CPU. ImageNet loading, weight preprocessing,
evaluation and complexity/latency measurements use existing infrastructure.
Small batch limits are practical demonstration settings, not the paper's 8192
calibration images. Record the actual image count from `metrics.json`.

Optional recovery uses labeled cross-entropy, AdamW and cosine scheduling after
new parameters are installed. There is no KD. `--finetune_epochs 0` skips training.
The workflow saves immediate corrected `pruned.pt`, final `model.pt`, metrics and
optimizer/scheduler state in `training.pt`. It reloads the final compact model and
checks outputs. These examples and offline tests do not claim published accuracy.
