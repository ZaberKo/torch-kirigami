# Complexity and latency measurement

`torch_kirigami.measurement` measures an eager PyTorch model independently of dependency analysis and pruning policy. It reports parameter elements, matrix/convolution multiply-accumulates (MACs), and inference latency on CPU or CUDA. Latency can use `torch.compile`.

## Basic use

Pass the original eager module to both functions, with its parameters and buffers already on the selected device. `compile=True` requests compilation inside the latency function.

```python
import torch
from torch import nn

from torch_kirigami.measurement import (
    calculate_model_complexity,
    measure_module_latency,
)

model = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 4)).eval()
x = torch.randn(2, 8)
complexity = calculate_model_complexity(model, (x,))
latency_ms = measure_module_latency(model, (x,), compile=True, warmup=5, repetitions=20)
print(f"#Params: {complexity.params}")
print(f"#MACs: {complexity.macs}")
print(f"Unsupported operations: {complexity.unsupported_ops}")
print(f"Batch latency: {latency_ms:.3f} ms")
```

Both functions accept a single input or a tuple/list of positional arguments, plus `input_kwargs` for keyword arguments. Nested tensor containers are supported. Inputs are isolated jointly and moved to the chosen device while retaining repeated tensor and container identity across positional and keyword arguments. This preserves identity-sensitive paths such as `MultiheadAttention(x, x, x)`. Distinct storage-sharing views remain supported on the selected device; automatic cross-device transfer of those views is rejected because moving each tensor separately would break their relationship. Move their shared base and recreate the views on the target device before measurement. A list intended as one model argument must itself be wrapped in an outer positional-argument tuple.

When `device` is omitted, the device is inferred from registered model tensors, falling back to CPU for a tensor-free model. All model parameters and buffers must already be on that one device. The functions do not move the model for the caller.

## Two independent execution paths

```mermaid
flowchart LR
    Model["Model"] --> Count["MAC counter"]
    Count --> Complexity["Complexity"]
    Model --> Prepare["Compile / warm up"]
    Prepare --> Time["Time inference"]
    Time --> Latency["Latency"]
```

Counting and timing are separate forwards. Instrumentation and the math attention backend used for counting do not affect the timed path. The measurement functions do not require a dependency graph and do not select a pruning budget.

## Parameter and MAC definitions

`calculate_model_complexity(target, input_args, device=None, input_kwargs=None)` returns the immutable `ModelComplexity` record:

| Field | Meaning |
| --- | --- |
| `params: int` | Number of elements in unique registered parameters, including frozen parameters; buffers are excluded |
| `macs: int` | MACs for the entire supplied input batch and executed path |
| `unsupported_ops: tuple[str, ...]` | Observed operations outside the explicitly supported counting convention |

One MAC is one multiply-accumulate, equivalent to two FLOPs in the counting convention. Matrix products and convolutions contribute, including the matrix products in attention. Bias addition, activation, normalization, and other elementwise work are excluded. Parameter sharing is deduplicated by `Parameter` identity, not by comparing numerical values.

The implementation uses PyTorch's native `FlopCounterMode` formulas divided by two and profiles executed operations to identify unsupported coverage. It temporarily selects the SDPA math backend so fused attention does not silently bypass matrix-operation counting. Counting uses `no_grad` outside inference mode.

A nonempty `unsupported_ops` means the MAC result is a partial count. The result is neither a complete count of all floating-point operations nor a prediction of compiler-generated instructions. Coverage describes the observed execution path; it cannot establish the cost of arbitrary opaque Python or native code. Pass an eager module, not a previously compiled wrapper.

## Latency definition

```python
latency_ms = measure_module_latency(
    model,
    (x,),
    device="cpu",
    repetitions=20,
    warmup=5,
    compile=True,
    compile_kwargs={"mode": "default"},
)
```

`measure_module_latency` returns the median duration of one forward for the entire supplied batch, in milliseconds. `repetitions` must be positive and `warmup` nonnegative.

| Device | Timing mechanism |
| --- | --- |
| CPU | `time.perf_counter()` around each uninstrumented forward |
| CUDA | Preinitialized events on the selected device's current stream, with synchronization before and after measurement |

Input transfer, state isolation, compilation, the initial compiled invocation, and warmup are outside the timed region. Timing uses inference mode and the normal attention backend. It measures model execution, not data loading or end-to-end application latency.

`compile_kwargs` forwards native options such as `backend` and `mode` to `torch.compile`; it requires `compile=True`. Compilation errors propagate instead of silently switching execution mode. A callable wrapper retains module call semantics without permanently adding compilation bookkeeping to the original model. Each measurement call creates its own wrapper, while PyTorch manages any underlying compiler caches.

After physical pruning, call measurement again with the compact eager model. A previously compiled callable is not the artifact to compare against the newly changed structure.

## State preservation and boundaries

Both functions restore per-module training flags, complete registered buffer tables and values (including `None`, added/deleted entries, and persistence flags), isolated input state, and torch RNG state on success and failure. Latency also restores the garbage-collector enabled state. Existing supported isolation contracts apply; incompatible storage aliasing can be rejected.

Forward execution must not modify parameter values. Arbitrary Python side effects are outside the isolation contract, and the same model must not be concurrently trained or mutated during measurement. Only CPU and CUDA devices are supported.

## Interpreting before/after results

Use identical input shapes, batch size, dtype, device, thread count, compilation options, warmup, and repetitions for the baseline and compact model. MACs and latency are batch-level quantities; report batch size alongside them. Lower parameter count or theoretical MACs does not guarantee lower latency because kernel choices, shape alignment, launch overhead, and hardware utilization also change.

The [workflow examples](../examples/workflows/README.md) print and save parameter counts, MACs, unsupported operations, latency, and measurement settings alongside validation accuracy. Their benchmark batch is independent of the training/evaluation batch. They suppress MAC reduction claims when counts have unsupported coverage.

From `examples/workflows`, after installing its requirements and downloading the required data:

```bash
python prune_finetune.py --model resnet18 --device cpu --threads 4
python gate_pruning.py --model vit_b_16 --device cuda --compile \
  --benchmark-batch-size 1 --warmup 10 --repetitions 50
```

These commands also perform the workflow's task-specific evaluation or training; they are not isolated benchmark-only commands. See the workflow README for required ImageNet splits and training controls.

Inputs sharing identity or storage with Tensor leaves of ordinary model attributes
are isolated together and temporarily rebound, including nested containers. Shared input/container identity is retained as well, including empty containers; same-device measurement does not rebuild that tree.
Bindings are restored on both success and failure. Moving only the input to another
device would break this relationship and is rejected; move the model-owned state
and its shared input together before measuring.
