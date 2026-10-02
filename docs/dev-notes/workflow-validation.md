# Workflow validation — 2026-10-03

Review scope: workflow model adapters, candidate declarations, selection settings,
physical execution, training, and structural checkpoint restoration. No new
production defect was confirmed in this review; coverage was extended where
existing entry tests did not exercise the relevant setting.

## Automated regression coverage

The workflow integration suite passed 689 tests on CPU/CUDA with
`--require-cuda`. It includes all 11 entries, short-data calibration,
training/projection schedules, multiple pruning rounds, invalid configurations,
unreachable targets, and checkpoint restoration. Most entry tests use reduced
architectures and synthetic data; full-size models were additionally exercised
in the smoke runs below.

New tests cover:

- Complete ResNet/ViT entry runs with dynamic magnitude, dynamic Taylor and static
  FPGM: 12 CPU/CUDA cases, including fine-tuning and checkpoint verification.
- Head-first and FFN-first sequential pruning with a rebuilt graph; retaining one
  head; rejecting its removal without changing bindings or values; a subsequent
  valid request using the same graph; independent masked references and restored
  backward: four CPU/CUDA cases.
- Attention capture and physical pruning with active dropout, followed by
  backward, deterministic evaluation and checkpoint restoration: two CPU/CUDA
  cases.

The final attention-workflow/architecture rerun passed 76 tests. The new settings
and attention cases also passed nine CPU tests on Python 3.10 / PyTorch 2.6 with
torchvision 0.21; nine CUDA cases were skipped in that isolated CPU environment.
GPU validation above used the development environment with `--require-cuda`.
Ruff, formatting and diff checks passed.

## Full-size pretrained CLI smoke runs

Host: RTX 5070 Ti, Python 3.12, PyTorch 2.14.0+cu130. Every run used official
cached pretrained weights, local ImageNet Arrow data, four validation images,
batch size two, and retained-width granularity eight. Training/calibration used
four available training images where needed. Latency warmup was zero with one
repetition; those values trigger the code path, not a reliable latency benchmark.

| Entry | Model | Selection settings | Additional paths verified |
| --- | --- | --- | --- |
| `prune_finetune.py` | ResNet18 | Parameter ratio 0.005; dynamic Taylor | Task-gradient calibration, one fine-tuning epoch, checkpoint |
| `vit_head_pruning.py` | ViT-B/32 | Parameter ratio 0.005; static group magnitude | Whole head 12→11, FFN compaction, checkpoint |
| `vit_head_pruning.py` | ViT-B/16 | Parameter ratio 0.005; static group magnitude | Whole heads 12→10, FFN compaction, one fine-tuning epoch, default Inductor latency compilation, two spawned train/validation workers, checkpoint |
| `isomorphic_pruning.py` | ViT-B/32 | Family ratios 0.01 and 0.05; one calibration batch | Independent family quotas, alignment and checkpoint; the 0.05 run also changes native attention widths |
| `variance_pruning.py` | ViT-B/32 | Parameter ratio 0.005; one calibration batch | Activation statistics, bias compensation, checkpoint |
| `osscar_pruning.py` | ResNet18 | Parameter ratio 0.001; one calibration batch; 64 rows; one swap attempt | Sampled convolution patches, grouped reconstruction, refitting and checkpoint |

All parameter-target runs reached their reported cap. Isomorphic uses independent
family quotas and an original-parameter safety ceiling; its family ratio is not a
whole-model parameter reduction target. Its 0.05 run restored with embedding
width 720 and 12 heads of width 60 in every block. A fresh process loaded this
checkpoint and successfully ran backward on nonzero random inputs.

CLI artifacts are under `/tmp/kirigami-workflow-review-20261003/`. These tiny data
runs establish execution paths, not ImageNet accuracy, recovery quality, or
representative throughput.

## Additional architecture smoke runs

Full torchvision ResNet34/50 architectures with random weights passed CUDA
planning, application, backward and checkpoint roundtrips. ResNet34 declared 16
block-internal axes; ResNet50 declared both bottleneck widths, 32 axes. Both used
a 0.005 parameter ratio and granularity eight, with nonzero 64×64 inputs.

Full ConvNeXt-Tiny with random weights discovered all 18 VBP MLP pairs. Pruning
eight positions in its first and last MLP passed calibration, an independent
mean-substitution output reference, bias compensation, training backward and
checkpoint restoration. This separately exercises the evaluation-only
stochastic-depth rule and channels-last MLP paths.

The matrix is deliberately finite. Unsupported architectures, partial-head
requests and infeasible resource/constraint combinations retain their explicit
diagnostics; passing these tests does not promise every model/setting combination.
