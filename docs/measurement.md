# Complexity and latency measurement

Torch Kirigami uses [Torch Prisma](https://github.com/ZaberKo/torch-prisma) to measure parameter counts, MACs, FLOPs, and inference latency. Measurement does not require a dependency graph or select a pruning policy.

Torch Prisma is a runtime dependency and installs with Torch Kirigami.
Import measurement APIs directly from `torch_prisma`.
Torch Kirigami no longer provides the `torch_kirigami.measurement` package.
`ParameterBudget.from_ratio()` uses Prisma's parameter counter without executing a forward.

## Basic use

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

## API and execution contracts

Prepare the model mode and place the model and inputs on the execution device before measurement.
`calculate_model_complexity()` takes no `device` argument.
The latency function's required `device` argument selects the timing device. It does not transfer the model or inputs.
Both functions accept `input_kwargs` for keyword arguments.
Use `(items,)` to pass a list as one positional argument.

`measure_module_latency()` returns `LatencyStats`, rather than a scalar duration.
Read `median_ms` for median batch latency in milliseconds.
Compilation defaults to `True`. Pass `compile=False` for eager timing, as in the example.
Compilation preparation and warmup do not contribute to the timed samples.
Counting always executes an eager forward without gradients. Latency forwards run in inference mode.

Neither API automatically selects evaluation mode or isolates inputs and model state.
Forward changes to inputs, buffers, parameters, and random-number state persist, including changes made before an exception.
Use separate model and input instances when measurement must preserve the originals.
Do not train or change the model concurrently with measurement.
These measurement contracts do not change dependency capture's separate isolation contract.

## Results and workflow integration

`count_parameters(model)` counts unique registered parameter objects without a forward pass.
It includes frozen and unused parameters, excludes buffers, and counts distinct objects separately even when they share storage.
Parameter, MAC, and FLOP counts are Python integers.
MACs and FLOPs describe the supplied batch and executed path.
Each total includes only work with an established count.
Neither total estimates uncounted work.

`uncovered_ops` contains `macs` and `flops` keys with sorted tuples of unique operation names.
Each tuple lists observed operations without a complete count under that convention.
Work outside a convention does not produce an entry solely because of that exclusion.
For example, bias additions contribute no dense MACs and do not create a MAC coverage entry.
These diagnostics describe counting coverage, not pruning support.

See the [Torch Prisma measurement API](https://github.com/ZaberKo/torch-prisma/blob/main/docs/measurement.md) for formulas, latency statistics, and device support.

The [workflow examples](../examples/workflows/README.md) report complexity and median latency for a batch of one image.
Their `model_metrics.measure_model()` adapter temporarily selects evaluation mode and restores each module's previous mode after success or failure.
Other forward side effects persist. The adapter does not copy model weights, buffers, or inputs.
The CLI's `--compile_latency` flag controls timing compilation explicitly, independently of Prisma's default.
Training, accuracy evaluation, and complexity counting remain eager.
Compare baseline and compact models using the same input shape, dtype, device, and timing settings.
Lower parameter counts or MACs do not guarantee lower latency.
