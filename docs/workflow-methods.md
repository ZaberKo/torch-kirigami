# Workflow methods and research adaptations

This document records the paper and official-repository review behind the
ImageNet workflows. It distinguishes implemented algorithm steps from experimental
recipes. Published state-of-the-art results depend on the model, compression
target, data and recovery training; this repository does not claim a universal
best method or reproduce those accuracy results.

## Research selection

| Method | What the method adds | Decision for these examples |
| --- | --- | --- |
| **Variance-Based Pruning (VBP), ICCV 2025** | Rank MLP neurons by activation variance; compensate their mean contribution in the following layer's bias | Implemented in `variance_pruning.py` for ViT and ConvNeXt MLPs. Provides a recent data-dependent method with explicit calibration and correction steps |
| **Isomorphic Pruning, ECCV 2024** | Compare importance within families of structurally similar dependency groups, rather than globally mixing heterogeneous structures | Independent within-family rankings and quotas; an outer search chooses a common channel ratio for the whole-model parameter cap |
| **SnapViT, NeurIPS 2025** | Training-free elastic ViTs using gradient information, an evolutionary architecture search and optional weight correction | Deferred. Its ViT-specific search and attention/head decisions are substantially more than a metric adapter and are outside the current example scope |
| **OSSCAR, ICML 2024** | Structured output reconstruction through least-squares weight updates and combinatorial local search | Implemented in `osscar_pruning.py`: sequential dense-teacher reconstruction, grouped deletion, remove/restore local search and retained-weight refitting |
| **DVBP + OB²C, 2026 preprint** | Covariance-based denoising and correction of both retained weights and bias, extending variance-based pruning | Deferred. Full covariance storage, spectral estimation and reconstruction solves need a separate numerical design and validation effort |
| **FPGM, CVPR 2019** | Rank filters by their summed pairwise Euclidean distances within a layer | Implemented as a classic comparison criterion in `prune_finetune.py`, not presented as a recent SOTA method or full FPGM training reproduction |

Primary sources:

- VBP: [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Berisha_Variance-Based_Pruning_for_Accelerating_and_Compressing_Trained_Networks_ICCV_2025_paper.html), [official implementation](https://github.com/boschresearch/variance-based-pruning).
- Isomorphic Pruning: [paper](https://arxiv.org/abs/2407.04616), [official implementation](https://github.com/VainF/Isomorphic-Pruning/blob/main/pbench/isomorphic_pruner.py).
- SnapViT: [official repository and paper links](https://github.com/WalterSimoncini/SnapViT).
- OSSCAR: [official solver](https://github.com/mazumder-lab/OSSCAR/blob/master/prune_algo.py).
- DVBP + OB²C: [preprint](https://arxiv.org/html/2608.17657v1), [official implementation](https://github.com/geontackee/DVBP_OB2C).
- FPGM: [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.pdf).

The implementations here use independently written formulas and the library's
public APIs. Reference repositories are not runtime dependencies.

Paper-named examples must retain the method's pruning criterion, selection
procedure and required corrections or reconstruction. Optional post-pruning
training is separate: knowledge distillation trains a compact student to match a
teacher's predictions or features, often alongside the supervised task loss.
Omitting that recovery stage does not justify omitting a pruning step, and these
examples do not claim the papers' recovered accuracy. OSSCAR's teacher targets
are part of its reconstruction objective and are retained even with no fine-tuning.

## VBP implementation

For a dense MLP, let `h` denote its intermediate activation and let the second
Linear layer compute `y = W h + b`. Calibration runs the pretrained model in
evaluation mode and records each channel's mean and **sample variance** at the
input to that second Linear layer. Dropout is inactive; all batch and spatial/token
positions are pooled, including ViT's class token. Labels and gradients are unused.

The workflow ranks hidden positions globally by their raw variance, without
magnitude weighting or per-layer normalization. `Greedy` performs one static
ranking, completing the requested deletions to satisfy granularity and structural
constraints. Candidate scores are a ranking signal, not a joint reconstruction
loss; no independence of hidden activations is assumed.

For removed hidden indices `R`, the bias correction is:

```text
new_bias = old_bias + old_weight[:, R] @ calibration_mean[R]
```

The producer loses output rows and the consumer loses matching input columns.
The correction represents substituting the calibration mean for each removed
activation. This is locally exact for that substitution, not equivalent to the
original varying activation. Earlier pruned blocks can also change the input
distribution of later blocks; this workflow intentionally uses one initial
calibration, rather than silently recalibrating during selection.

Implementation boundaries:

- **Models:** CLI examples use torchvision ViT-B/16, ViT-B/32 and ConvNeXt-Tiny.
  Eligible dense producer/consumer chains are discovered from the captured graph,
  not selected by model class or fixed module paths. Branches, shared calls and
  unsupported intervening operations require additional algorithm-specific proofs;
  their exclusion reasons are reported. This method targets MLP intermediate
  activations, not attention heads or arbitrary convolution channels.
- **CNN scope:** ConvNeXt has dense channels-last MLPs, so the same formula applies.
  A padded spatial convolution generally has a position-dependent missing mean
  contribution; adding one constant bias is insufficient. ResNet is therefore
  not advertised for this VBP workflow.
- **Calibration:** streaming per-channel moments use the batch-merge variance
  identity. Per-batch reductions promote low precision to float32; stored means
  and accumulated squared deviations use float64. Hooks are removed and mixed
  train/eval modes restored even if calibration fails. Existing gradients remain
  unchanged. Calibration intentionally consumes the caller's shuffled training
  iterator.
- **Runtime:** 16 training batches by default; `--calibration_batches 0` uses the
  entire loader. The official implementation defaults to a full traversal.
- **Target:** whole-model `ParameterBudget` and retained-width granularity, rather
  than the paper's neuron-ratio experiments. This changes which deletion threshold
  is reached and must be reported when comparing results.
- **Recovery:** optional AdamW with learning rate `1.5e-5`, weight decay `0.01` and
  a cosine schedule per optimizer step. The example omits teacher distillation
  and the full training/augmentation recipe. Default recovery duration is zero.
- **Persistence:** prepare correction values from the old weights before applying
  the structural plan; copy the corrected biases afterward. Save the resulting
  compact checkpoint. A structural plan alone does not include these value updates.

## FPGM criterion implementation

For each supported producer, flatten every output filter to a signed vector
`w_i`. The score is the sum of Euclidean distances to all filters in that layer:

```text
score(i) = sum_j ||w_i - w_j||₂
```

Low-scoring filters are more central/redundant under this criterion. The workflow
uses raw signed weights, not squared magnitudes or absolute-value vectors. It
computes one score snapshot with bounded pairwise-distance tiles; memory is
bounded per tile, but arithmetic remains quadratic in output width. No additional
layer normalization is applied.

`--metric geometric_median --selection static` uses this snapshot with the
framework's dependency analysis, constraints and whole-model parameter target.
ResNet convolution filters and ViT FFN rows are supported. The ViT extension,
global allocation and one-shot physical deletion are adaptations: the original
FPGM layerwise soft-pruning/training procedure is not reproduced. Dynamic
selection is rejected instead of returning stale snapshot scores under a dynamic
label.

## Separation from the pruning core

The workflows reuse `Pruner`, structural constraints, plan execution and checkpoint
APIs. VBP supplies an activation-statistics metric. Isomorphic supplies a custom
strategy that does not invoke `Greedy`. OSSCAR explicitly solves each layer and
submits its exact result through `plan_remove`. No algorithm-specific scoring or
reconstruction is added to dependency analysis.

`PlanningContext` exposes complete-impact queries, validated metric evaluation,
joint recipe compilation and exact remaining-parameter counts. These services
allow a strategy to own its selection algorithm; it need not imitate greedy
ranking. The planner independently verifies the returned registered keys,
dependencies, execution support and final budget. Calibration and reconstruction
values remain outside immutable structural plans and are saved in checkpoints.

The basic and iterative workflows target internal widths in every supported
block: BasicBlock `conv1`, Bottleneck `conv1` and `conv2`, and EncoderBlock
`mlp.0`. Stem, residual interfaces, attention widths and classifier dimensions
remain unchanged. Block types identify the scope; stage names and a selected
single block do not. Iterative pruning reuses the same scope after every rebuild.
Isomorphic retains broader discovery to demonstrate heterogeneous structures.
Sparse-training examples retain their specialized BN/gate or regularization
scopes. These are experimental choices, not restrictions of the discovery API.
An unmet parameter target fails rather than implicitly widening the scope.
Comparisons between methods must control candidate scope as well as parameter
target, calibration and recovery training.

The VBP calibration, metric and compensation live together in its entry file;
the FPGM metric lives in the basic entry. The only shared additions are model
loading infrastructure. ConvNeXt's opaque torchvision stochastic-depth function
is handled through a local operator registration with an evaluation-mode identity
contract. It does not change the original module's training behavior.

The basic entry separately exposes `--selection dynamic` for existing magnitude
and Taylor metrics. This rescores conditional parameter regions after accepted
deletions; it does not refresh activations, gradients or trained weights. Dynamic
ranking is a policy choice, not a new paper-specific pruning method.

## Isomorphic Pruning implementation

`isomorphic_pruning.py` uses all graph-declared candidate domains except proven
interface protections. The strategy classifies complete dependency impacts using
operator types, affected axes, parameter sharing and ordered dataflow edges.
Module names, tensor widths and channel indices do not define a family. This
ordered signature is conservative, not a general graph-isomorphism solver.
Equivalent complete selections are counted once, regardless of entry aliases.

Static metric scores are compared only within a family. For a common channel
ratio, each family selects its own lowest-ranked quota. Retained-width alignment
rounds down, adding that root's next lowest-ranked positions; nonempty and joint
execution checks still apply. The strategy does not convert ranks to percentiles
and feed them to `Greedy`. One family is a valid homogeneous case, not a reason
to invent stage-based families or suppress the method.

The example's `ParameterBudget` is a whole-model target. An outer search visits
exact rational quota breakpoints and checks actual joint recipes until the cap
is reached. It does not assume that execution feasibility is monotone. Families
keep their original populations throughout this search; failed execution checks
do not silently shrink denominators or replace the ranking algorithm. Reports
separate unrounded quotas, actual selected actions, ratio trials and rejected
proposals. A trial limit or unreachable target produces no applied partial plan.

The default metric is `GroupMagnitude`; custom metrics use the same `Metric`
contract. This is not the paper's Taylor-calibrated and distilled accuracy recipe.
Optional task-only fine-tuning is a separate stage. Unsupported structures remain
explicit exclusions rather than presumed supported attention/head operations.

## OSSCAR implementation

`osscar_pruning.py` separates whole-model width allocation from within-layer
reconstruction. It first searches aligned width breakpoints for a common channel
removal fraction whose **actual joint plan recipes** meet the requested parameter
cap. This supplies per-consumer retained cardinalities; it is not an optimized
global allocation of reconstruction error. No model weights are changed during
allocation, and an unreachable target is rejected before sequential pruning.

Consumers are then processed in forward order. For each consumer:

1. Run the fixed original dense teacher and the current compact model on the same
   training images. Pair current consumer-input rows with original teacher
   preactivation outputs, subtracting the fixed current consumer bias.
2. Accumulate a design Gram matrix `H` and teacher cross-product `G`, normalized
   by the number of observations. Add diagonal damping to `H` and the matching
   original-weight anchor to `G`.
3. Delete channel groups using their quadratic reconstruction costs and jointly
   update the remaining coefficients and inverse with Schur-complement formulas.
4. Attempt bounded fixed-cardinality swaps: remove one inexpensive retained group,
   evaluate restoring each absent group, refit the proposed support, and accept
   only an objective improvement. Newly removed groups may also be restored.
5. Submit the exact chosen producer channels to `plan_remove`, validate matching
   consumer-column deletion, apply the structural plan and copy refitted weights.
   Rebuild the graph for the next consumer.

For design rows `X`, bias-adjusted teacher targets `Y`, original coefficients `B0`
and ridge strength `lambda`, the system is:

```text
H = X.T @ X / observations + lambda * I
G = X.T @ Y / observations + lambda * B0
B_retained = solve(H[retained, retained], G[retained])
objective = -0.5 * sum(B_retained * G[retained])
```

The omitted target-squared term is constant. Objective values can be negative;
only differences for the same layer/system are meaningful. Damping defaults to
`0.01` times the mean Gram diagonal, with a tiny positive scale floor for completely
dead inputs. It regularizes toward original weights, not toward zero. With zero
damping, singular systems fail explicitly. Support membership is stored as
indices, never inferred from zero coefficients or signed weight sums.

For ordinary Conv2d, `unfold` uses the consumer's real kernel, stride, dilation
and zero padding; all kernel coefficients of one input channel form a group.
For Linear, one input feature forms a group. Discovery follows actual unbranched
dataflow between eligible Conv2d or Linear endpoints, including supported
coordinatewise transforms and channelwise pooling/normalization. It does not
require ResNet/ViT class names or fixed paths. Adjacent pairs may overlap: a
consumer can become the next producer, with calibration and graph rebuilding
after each change. Branches, shared uses, unproved reindexing and cross-channel
mixing are excluded with reasons. Grouped/depthwise convolution reconstruction
is outside this example; attention operations are not treated as ordinary MLPs.

Runtime and experimental boundaries:

- Process one consumer's covariance at a time and unfold at most one image at a
  time. Memory remains quadratic in `input_channels * kernel_area`; sampling
  fewer observations does not reduce this matrix dimension.
- Default calibration is two training batches per consumer, with at most 4,096
  uniformly spaced spatial/token positions per batch. Identical position IDs
  pair student inputs with teacher targets. This is an explicitly sampled
  calibration adaptation; actual observation counts are recorded. The Arrow-backed
  training dataset remains available in full.
- `--prune_batch 8` controls deletion updates, independently of retained-width
  `--granularity 8`. `--swap_steps 5` bounds local remove/restore attempts and may
  stop earlier. Neither the batched deletion heuristic nor the one-group swap
  neighborhood guarantees the global optimum or exhaustive local optimality.
- The current model already includes earlier reconstructions when collecting the
  next layer's statistics. This follows the sequential teacher-target formulation,
  rather than pretending the changed input covariance always satisfies `G=H@B0`.
- The ViT FFN scope, width allocation and calibration/search settings differ from
  published experiments. Optional task-only SGD fine-tuning defaults to zero.
- Completed earlier layers remain changed if a later layer fails; this is an
  explicit sequence of pruning operations, not one atomic whole-model transaction.
- Save the **refitted compact checkpoint**. Structural plans alone cannot recover
  the reconstructed values. Optimizers are created after all replacements.

## ViT attention-head and FFN baseline

`vit_head_pruning.py` exercises custom candidates and scoring without adding core
operator rules. It is a baseline, not an implementation of a research paper.
The example explicitly converts torchvision self-attention to packed Q/K/V
Linear projections, reshape/permute/unbind, SDPA and an output Linear projection.
All structural relationships and physical changes use existing operator rules.

An attention candidate selects one position on the SDPA query's head axis.
Propagation selects the complete matching Q/K/V head features and removes their
projection rows, biases and matching output-projection columns.
The inferred head count can shrink independently in each encoder block; per-head
feature width and scale remain fixed. FFN candidates use the existing hidden
channel domain. Residual width, embeddings, normalization widths and classifier
interfaces stay fixed. Nonempty constraints prevent deleting the final head or
entire FFN. Hardcoded per-head width is checked by the existing reshape execution
contract; unsupported feature-level requests are rejected before mutation.

Scoring uses the library's static `GroupMagnitude(p=2)`: joint affected-weight
energy relative to the logical axis's mean singleton energy. For attention, one
singleton is a complete head; for FFN, it is a hidden neuron. No separate score
implementation or scalar Q/K/V-row normalization is needed. This is an explicit
cross-domain heuristic, not a sensitivity estimate or a guarantee that those
units are equally important. Greedy accepts feasible additions toward the
whole-model parameter cap; it does not force a head/FFN quota. Stage reports
record actual head counts and FFN widths.

The attention adapter is limited to torchvision's unmasked, batch-first,
equal-width self-attention. It preserves attention dropout and pretrained
projection values. Checkpoint restoration constructs the same adapter around an
unpruned skeleton using `imagenet_models.make_head_prunable_model`; native MHA
is not automatically replaced during load. Adapter classes live in the shared
model infrastructure so CLI execution and subsequent imports use identical
class names in structural checkpoints.
Independent dense matmul/softmax references, input gradients, joint head/FFN
deletions, invalid partial-head requests, automatic planning, training and
structural checkpoint restoration are covered on CPU and CUDA. A separate
subprocess test checks restoration after executing the entry as `__main__`.

## Verification and interpretation

CPU and CUDA regressions cover signed-distance formulas, independent sample
variance references, streaming batch partitions, calibration cleanup on failure,
physical compaction, backward and compact-checkpoint restoration. VBP's numerical
reference explicitly replaces removed hidden activations with their means in an
independent dense model. Tests use real torchvision ConvNeXt/ViT block structures
at reduced widths, in addition to small algebraic fixtures.

These checks establish implementation behavior. Accuracy claims still require
running the [documented commands](../examples/workflows/README.md) with pretrained
weights and ImageNet, recording calibration size, compression target, training
settings and full validation results.
