# PyTorch operator audit

## Scope

The audit enumerates visible callables in these namespaces:

- `torch`
- `torch.Tensor`
- `torch.nn.functional`
- Public `torch.nn.Module` classes in `torch.nn`
- `torch.special`, `torch.linalg`, and `torch.fft`

The inventory separately lists public Tensor properties.
Properties such as `T`, `mH`, `real`, and `imag` use the existing attribute rule.
That rule does not support arbitrary properties such as `data` or `grad`.

The recorded environments use PyTorch 2.6 and 2.14.
The [inventory script](../experiments/operator_surface_audit.py) inspects every visible name in these namespaces.
The [recorded inventory](../experiments/operator_surface_results.json) lists each entry and its direct rule.
It also records differences between the two environments.

Each environment exposes 1,717 entries under this definition.
The registry directly matches 813 entries.
The other 904 entries include utilities, wrappers, and operations that need more semantics.
The environments each expose 14 names absent from the other.
These counts include aliases and do not measure distinct mathematical operations.

The inventory includes tensor operations, factories, state controls, and implementation utilities.
It does not execute arbitrary functions such as `torch.load` or `torch.set_rng_state`.
It excludes private names, arbitrary `torch.ops` namespaces, and nested backend APIs.
Custom operations still require their own semantic contract.

## Reading the inventory

| Status | Meaning |
| --- | --- |
| `direct_rule` | The default registry matches this exact entry. Support still depends on arguments, axes, effects, and execution requirements. |
| `no_direct_rule` | The default registry has no direct match. FX may expand a Python wrapper into supported calls. Otherwise, structural analysis needs another rule. |

A registration count is not a model compatibility score.
Function aliases can reference the same Python object.
A Tensor method with the same name can have another signature or coordinate behavior.

For example, `Tensor.unfold` returns window views along one axis.
`F.unfold` extracts and flattens image patches.
Each uses a separate coordinate rule.
Equivalent spellings reuse one existing semantic implementation.

## Additions and corrections

| Family | Changes | Boundary |
| --- | --- | --- |
| Pointwise mathematics | Add missing arithmetic, comparison, special-function, activation, and Tensor spellings. | Native dtype, broadcasting, derivative, and write restrictions still apply. |
| Casts and complex values | Add mature casts, real/imaginary properties, conjugation, and transpose properties. | Alias-capable results remain conservative. The registry excludes experimental complex-half conversion. |
| Normalization | Add raw `torch` forms and preserve their distinct parameter order. Add local response normalization. | Direct Parameter statistics updates produce the existing capture warning. Registered buffers remain isolated. |
| Embedding | Add the raw weight-first `torch.embedding` form. | Vocabulary rows remain fixed. The raw operation has no `max_norm` update. |
| Pooling, padding, and interpolation | Add missing module and function forms, including paired max-pool indices. | Spatial and padded axes remain fixed. |
| Coordinates and matrix operations | Add `ravel`, `select`, `split_with_sizes`, shape aliases, dot products, matrix-vector products, and outer products. | Integer selection keeps the indexed axis fixed. Shape provenance and original-call checks still apply. |
| Fourier operations | Add transform and shift forms from `torch.fft`. | Transform and frequency axes remain fixed. Other axes can shrink. |
| Tensor window views | Add `Tensor.unfold` without treating it as image-patch extraction. | Window axes remain fixed. Alias checks still apply. |

These changes do not add a tracer, a second computation graph, or backend-specific layout guesses.
The public registry and pruning APIs remain unchanged.

## Entries that need more semantics

The audit does not assign pointwise semantics to an operation merely because its output shape matches its input.
The following families have no general default implementation:

| Family | Examples | Additional requirement |
| --- | --- | --- |
| Value-dependent coordinates | `nonzero`, `unique`, sorting, top-k, arbitrary gather/scatter | Describe how values determine retained coordinates and index outputs. |
| Coordinate transforms | Diagonals, `roll`, `flip`, general `tensor_split`, `view_as`, `reshape_as` | Track remapped coordinates, template provenance, output ports, and original-call validity. |
| General storage mutation | `as_strided`, resize, storage replacement, arbitrary scatter writes | Prove alias, layout, and consumer safety. |
| Linear algebra | Decompositions, solves, determinants, matrix powers | Define coupled row/column changes and compact-domain semantics. |
| Losses and distances | Classification losses, pairwise distances, bilinear layers | Define target alignment, reductions, class domains, and parameter bindings. |
| Sampling and specialized modules | Grid sampling, fractional pooling, EmbeddingBag, recurrent and packed models | Define sampling coordinates, offsets, state, or packed layout. |
| Specialized execution | Sparse, quantized, distributed, and compiled operation boundaries | Supply storage and execution contracts beyond ordinary dense tensors. |

Some entries in these families can expand through FX.
The resulting calls still need structural rules before related requests can prune.
These are implementation gaps, not claims that structural pruning is mathematically impossible.

Unknown structural semantics block affected requests.
Independent branches remain usable when capture itself is safe.
Unknown write effects on parameters can reject capture before model execution.
See [operator support](operator-support.md) and [model support](model-support.md) for the full failure boundaries.

## Evidence and future API changes

Independent spelling lists exercise public build, plan, and apply calls on CPU and CUDA:

- `test_entry_aliases.py`: equivalent entry forms and provenance.
- `test_pointwise_surface.py`: mathematical functions, activations, writes, and complex values.
- `test_functional_surface.py`: normalization, embedding, pooling, and local response normalization.
- `test_coordinate_surface.py`: shape, indexing, vector products, and transpose properties.
- `test_spectral.py`: Fourier references, protected axes, and checkpoint restoration.

These files are in `tests/dep_graph/operators/`.
Special-function tests retain PyTorch's derivative limitations.
They do not claim a native backward implementation for every input.
The registered-entry suite separately checks rule matching and propagation.

The audit produced these verification results:

| Environment | Check | Result |
| --- | --- | --- |
| PyTorch 2.14, CPU and CUDA | Full suite with `--require-cuda` | 7,099 passed, one expected skip, 390.61 seconds. |
| PyTorch 2.14, CPU and CUDA | Final pointwise spelling suite | 622 passed. |
| PyTorch 2.14, CPU and CUDA | Final registered-entry and architecture suites | 1,655 passed. |
| Python 3.10, PyTorch 2.6 CPU | Surface and architecture suites | 1,264 passed. CUDA cases skipped in this CPU-only environment. |

The full run started before the final four mathematical spelling additions.
The final pointwise, registered-entry, and architecture runs checked those additions.
The final Tensor-property inventory guard also passed in both environments.
Ruff, documentation lint, and wheel/source builds passed.
These results establish the tested contracts, not task accuracy or universal operator support.

The namespace guard checks new visible names against the recorded inventory.
A new name requires review even when the registry tests remain unchanged.
The guard detects API changes. It does not prove semantic correctness or detect every new overload.
New overloads still need argument-specific lifecycle tests and independent numerical references.
