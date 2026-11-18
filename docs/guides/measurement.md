# Measurement

Torch Kirigami uses [Torch Prisma](https://github.com/ZaberKo/torch-prisma) to
count parameters, MACs, and FLOPs, and to time inference. Torch Prisma installs
with Torch Kirigami. Import measurement functions from `torch_prisma` directly.
Measurement needs no dependency graph and has no role in choosing what to prune.

```python
import torch
from torch import nn
from torch_prisma import calculate_model_complexity, measure_module_latency

device = "cpu"
model = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 4)).to(device).eval()
inputs = torch.randn(2, 8, device=device)

complexity = calculate_model_complexity(model, inputs)
latency = measure_module_latency(model, inputs, device, compile=False)
print(complexity.params, complexity.macs, complexity.flops)
print(complexity.uncovered_ops)
print(f"Batch latency: {latency.median_ms:.3f} ms")
```

## How to measure

Set the model mode and move the model and inputs to the target device before you
measure. Neither function changes the mode or moves tensors. The `device`
argument of `measure_module_latency()` only selects the timing device. Both
functions accept `input_kwargs` for keyword arguments. Pass `(items,)` to give a
list as one positional argument.

- Counting runs one eager forward without gradients.
- Latency runs in inference mode and returns `LatencyStats`. Read `median_ms` for
  the median batch latency. Compilation is on by default. Pass `compile=False` for
  eager timing. Compilation and warmup are not timed.

Measurement runs your `forward` on your tensors. Anything `forward` changes, such
as inputs, buffers, parameters, or random-number state, stays changed, even if an
exception occurs. Use copies when the originals must stay untouched. This is
unlike graph capture, which isolates inputs and buffers.

## Reading the results

- `count_parameters(model)` counts unique registered parameters without a forward
  pass. It includes frozen and unused parameters and excludes buffers. It counts
  distinct parameter objects separately, even if they share storage.
  `ParameterBudget.from_ratio()` uses this function.
- MACs and FLOPs are Python integers for the given batch and the executed path.
  They include only work whose count is known, and they do not estimate the rest.
- `uncovered_ops["macs"]` and `uncovered_ops["flops"]` list operations that ran
  without a complete count under that convention. Work outside a convention, such
  as bias additions for MACs, is not listed. These lists describe counting
  coverage, not pruning support.

The [Torch Prisma documentation](https://github.com/ZaberKo/torch-prisma/blob/main/docs/measurement.md)
describes the formulas, latency statistics, and device support.

## Comparing models

Compare the original and compact models with the same input shape, dtype, device,
and timing settings. Fewer parameters or MACs do not guarantee lower latency,
because memory layout, kernel selection, and hardware utilization also matter.
Accuracy needs a separate evaluation on task data.

The [workflow examples](https://github.com/ZaberKo/torch-kirigami/blob/main/examples/workflows/README.md)
report complexity and median latency for one image. Their `model_metrics.py`
adapter switches the model to evaluation mode for the measurement and restores
each module's previous mode, even after a failure. It does not copy weights,
buffers, or inputs. The `--compile_latency` flag controls compilation explicitly.
