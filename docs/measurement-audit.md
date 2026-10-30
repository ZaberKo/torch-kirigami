# Measurement operation audit

## Scope and evidence

The 2026-10-30 audit checks measurement, not dependency-graph operator semantics.
The development environment uses PyTorch 2.14.0+cu130 and an RTX 5070 Ti.
The minimum environment uses Python 3.10 and PyTorch 2.6 on CPU.

The [inventory script](../experiments/measurement_surface_audit.py) enumerates public names and registered ATen schemas.
The [recorded inventory](../experiments/measurement_surface_results.json) contains each name, overload name, and measurement declaration.
It also records names and overloads that exist only in the minimum environment.

| Inventory | PyTorch 2.14 |
| --- | ---: |
| Public callable entries | 1,760 |
| Public Tensor properties | 38 |
| ATen operator packets | 1,731 |
| ATen overloads | 3,757 |

Public namespaces include `torch`, `Tensor`, `nn`, `nn.functional`, `special`, `linalg`, `fft`, `sparse`, `masked`, and `signal.windows`.
Names include aliases and utilities. They are not distinct mathematical operations.
The script does not invoke arbitrary utilities such as serialization or global-state setters.

The [execution probe](../experiments/measurement_opinfo_probe.py) uses up to three float32 samples from each installed PyTorch OpInfo entry.
It executes each sample normally before measurement, so baseline failures remain separate from measurement failures.
The [probe results](../experiments/measurement_probe_results.json) retain each entry's counts, diagnostics, or failure.
The audit checks 703 entries on each device, with 1,941 result records per device.
Entries without float32 support produce one availability record instead of executing samples.

| Result | CPU | CUDA |
| --- | ---: | ---: |
| Measurement completed | 1,894 | 1,902 |
| No float32 sample | 25 | 25 |
| Ordinary execution failed | 15 | 7 |
| Measurement rejected or failed | 6 | 6 |
| Input exceeded the audit size limit | 1 | 1 |

Completed measurement does not mean complete FLOP coverage.
The result can contain explicit MAC or FLOP coverage gaps.
The CPU baseline failures are CUDA-only Jiterator samples.
CUDA baseline failures include a CPU-only attention kernel, an unavailable sparse reduction kernel, and one sample with incompatible gradient dtypes.

This probe uses private PyTorch test data for the audit only.
The library and regular regression tests do not import OpInfo or depend on `expecttest`.

## Defects and corrections

| Defect | Correction | Regression evidence |
| --- | --- | --- |
| A counted call hid an uncounted call with the same operator name. | Match formula execution to individual profiler calls and their children. | Hidden `deg2rad` multiplication remains unverified when the counter counts another multiplication. |
| Raw efficient attention interpreted sequence length as the head count. | Use its batch/sequence/head/width layout for dense inputs. Retain the native variable-length formula. | CUDA self-attention and cross-attention with different value widths. |
| Raw CPU flash attention had no dense matrix-product count. | Reuse the native SDPA matrix formula for its matching input layout. | Compare numerical output to public SDPA and independently calculate both matrix products. |
| In-place matrix operations and batch-reduced `addbmm` missed formulas. | Reuse the matching dense matrix formulas. Count bias and scaling separately. | Functional and in-place forms with nonunit alpha, nonunit beta, and zero beta. |
| Arithmetic aliases, in-place operations, and safe softmax had inconsistent coverage. | Reuse the corresponding arithmetic formulas. Add `addcmul` and `addcdiv` arithmetic. | Function, Tensor method, in-place, and `out=` forms with independent counts. |
| Custom namespaces did not match profiler names. Some custom calls lacked a named event. | Normalize namespace separators and use the immediate dispatch scope when needed. | A registered custom FLOP formula covers its own call without claiming a MAC formula. |
| Native backward formulas omitted auxiliary arithmetic without a FLOP diagnostic. | Keep their available count and report partial coverage. | Convolution backward retains a diagnostic for its omitted bias reduction. |
| Some parameterless CUDA tests silently measured on CPU. | Pass the selected device explicitly in those tests. | The arithmetic and vector-product cases now execute on both selected devices. |

These regressions use the public measurement entry point.
The formula markers use the public profiler API. They do not introduce a tracer or another computation graph.
Formula registration remains local to each measurement call.

## Remaining boundaries

The execution probe retains these measurement failures:

- Sparse input tensors: existing execution isolation supports dense, non-quantized strided tensors only.
- CUDA `tensor_split` with a CPU index tensor passed as input: measurement transfers tensor inputs to the selected device, but this overload requires CPU indices.

For static split indices, pass a Python tuple or list instead of a CPU tensor.
The regression tests exercise this valid CUDA alternative.
Measurement does not infer which tensor arguments are host-side metadata.

The inventory distinguishes formulas, zero-cost conventions, composites, and entries without a direct formula.
An entry without a direct formula can still decompose into counted operations.
A registered packet formula does not prove coverage for every overload, dtype, layout, or argument combination.

Unimplemented arithmetic, including pooling, GELU, FFTs, general decompositions, and many losses, retains explicit FLOP diagnostics.
Opaque work without profiler events cannot be discovered by this audit.
The [counting conventions](measurement.md) remain the definition of the reported numbers.

## Repeat the audit

```bash
PYTHONPATH=. .venv/bin/python experiments/measurement_surface_audit.py \
  --output /tmp/measurement_inventory.json
uv pip install expecttest
PYTHONPATH=. .venv/bin/python experiments/measurement_opinfo_probe.py \
  --device cpu --output /tmp/measurement_cpu.jsonl
PYTHONPATH=. .venv/bin/python experiments/measurement_opinfo_probe.py \
  --device cuda --output /tmp/measurement_cuda.jsonl
.venv/bin/pytest tests/measurement tests/architecture/test_coverage_inventory.py --require-cuda
```

The inventory guard detects new public names and ATen overload names.
It requires a review when the visible surface changes. It does not replace numerical regression tests.
