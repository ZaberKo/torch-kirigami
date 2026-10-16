# PyTorch operator registration audit

Review date: 2026-10-16. Reference: PyTorch 2.14, with PyTorch 2.6 as the minimum supported version.
This note records a review of common forward operations, not exhaustive PyTorch coverage.
The [operator guide](../operator-support.md) defines the current support contract.

## Changes from this review

- Add function and Tensor method forms of `unflatten` through the existing reshape relation and original-call checks.
- Add function and Tensor method forms of `movedim` and `moveaxis` through the existing permutation relation.
- Document all three registration forms and provide a custom function example.

These operations also exist in PyTorch 2.6.
Their absence was a coverage gap, not a new PyTorch 2.14 requirement.
No new tracer, execution IR, or alias framework is necessary.

PyTorch 2.6 FX rejects proxy factors in the function form of `unflatten` before rule analysis.
The Tensor method handles those factors in the tested cases.
The library retains this capture boundary rather than replacing the tracer.

## Remaining registration gaps

The default registry has no direct entries for the following representative operations.
FX can expand some Python wrappers into other operations, so registry absence alone does not establish capture failure.
Support depends on the actual FX target and requested change.

| Category | Examples | Work needed before support |
| --- | --- | --- |
| Arithmetic aliases | `multiply`, `subtract`, `divide`, `true_divide`, `negative`, `absolute`, `clip` | Check exact signatures and allocation behavior before reusing pointwise rules. |
| Additional pointwise operations | `sinc`, `nan_to_num` | Check supported values, overloads, and write variants. |
| Shape and template operations | `ravel`, `reshape_as`, `view_as`, Tensor `broadcast_to`, `broadcast_tensors` | Track template dependencies, output ports, and view or copy behavior. |
| Partition and stacking variants | `tensor_split`, `hsplit`, `vsplit`, `dsplit`, `hstack`, `vstack`, `dstack` | Preserve original partition coordinates and rank-promotion behavior. |
| Static index transforms | `select`, `diagonal`, `flip`, `roll`, `rot90`, `tril`, `triu` | Define exact retained-coordinate conditions. Shape equality alone is insufficient. |
| Statistical reductions | `var`, `std`, `var_mean`, `std_mean` | Handle correction, reduced axes, multiple outputs, and small retained domains. |
| Prefix and neighbor operations | `cumsum`, `cumprod`, `diff` | Describe dependencies across positions. Do not treat these as pointwise operations. |
| Index-producing reductions | `max`, `min`, `topk`, `sort` | Distinguish overloads and preserve index meaning. Existing `amax` and `amin` do not return indices. |
| Data-dependent indexing | `gather`, `take_along_dim`, `scatter`, `masked_select`, `nonzero` | General remapping remains outside the current support boundary. Constant-index cases need explicit proofs. |

Explicit copy variants also need separate allocation declarations before registration.
Do not register every PyTorch function with matching input and output shapes as an identity operation.

## Boundaries retained

Python data-dependent control flow still requires a model adaptation.
Capturing arbitrary Python state changes and undeclared parameter writes remains outside the isolation contract.
Supporting `unflatten` does not infer a binding to an owner's `num_heads` attribute.
Default depthwise candidates still remove complete input/output groups rather than search multiplier reductions.

## Restriction review

The follow-up review examined capture checks, operator rules, planning, physical replacement, persistence, sparse training, and measurement boundaries.
This review does not establish support for every model.

Changes:

- Embedding `max_norm` no longer causes an unconditional capture error. The rule declares native parameter value updates for module and functional calls.
- Capture warns before declared parameter updates. Updates remain in the original model even if a later call fails. Capture allocates no weight copy.
- `einsum` supports implicit string outputs through the existing label relations. Output order follows PyTorch's ellipsis and alphabetical-label rules.
- Softmax and log-softmax accept the native rank-dependent default dimension. PyTorch's deprecation warning remains visible.
- Unsupported analysis retains its reasons in `graph.diagnostics`. Barriers protect affected dependencies. It does not emit a blanket warning.

Restrictions retained:

| Area | Reason |
| --- | --- |
| Python control flow and arbitrary cache mutation | FX does not provide complete execution or rollback semantics. |
| Unknown writes, `out=`, and unproven parameter views | Removing checks can expose caller state to mutations without a declared contract. |
| Empty tensors, sparse or quantized storage, custom tensor replacement | Existing coordinate and replacement representations do not cover these cases. |
| Data-dependent indexing and unsupported shape provenance | These cases require additional coordinate semantics. Matching shapes are insufficient. |
| View strides, partition ports, fixed indices, and original-forward checks | These checks prevent executable shapes with incorrect coordinates or invalid layouts. |
| Distinct tensors sharing storage | Replacement must preserve alias semantics. Equal shapes do not establish a valid shared replacement. |
| Hooks and custom persistence callbacks | Replacement and restoration can bypass their behavior or cause external side effects. |
| Missing gradients, nonfinite scores, incomplete influence, and missed targets | These checks prevent invalid scoring or plans that violate the requested result. |
| Analysis and search limits | These limits bound exact work. Their exhaustion must remain explicit. |
| CPU/CUDA measurement and input-device checks | Timing needs verified synchronization and device ownership. Other backends need separate validation. |

Default depthwise candidate blocks remain policy, not a dependency-analysis restriction.
Manual requests can still express legal multiplier reductions.
No warning converts an unsafe request into an executable plan.

### Parameter-update follow-up

The review found two false rejections in planning:

- Declared parameter updates in container or variadic arguments passed capture, but planning did not resolve all their inputs.
- Functional Embedding with a registered buffer weight passed capture, but planning rejected its declared write.

Capture and planning now share the parameter-update argument resolver.
Declared writes to direct buffer bindings use the existing buffer isolation contract.
Views and unknown writes retain their checks.
The generic warning no longer describes every custom update as Embedding renormalization.
No new class, field, or public method is necessary for these fixes.

### Einsum follow-up

Repeated input labels now use the existing equal-axis relations.
For `ii->i`, input rows, input columns, and output positions retain the same original indices.
For `ii`, the input axes remain coupled while the diagonal reduces to a scalar.
Multiple consumers participate in the same dependency closure.
String equations remain required.

Einsum no longer declares independent output storage for all calls.
Identity, permutation, and diagonal expressions can return views.
Single-operand calls conservatively retain alias checks, including reductions that allocate in the tested runtime.
Multi-operand products retain their independent-storage declaration and metadata-cache support.
The existing alias checks reject parameter writes through those views before metadata execution.
An explicit clone can establish independent storage.
Compact-call validation still checks subsequent view strides.
Tests cover padded input layouts where the original view works but the compact view fails.

## Sources

- [PyTorch 2.14 operation index](https://docs.pytorch.org/docs/2.14/torch.html)
- [Unflatten factors and view semantics](https://docs.pytorch.org/docs/2.14/generated/torch.unflatten.html)
- [Movedim axis ordering](https://docs.pytorch.org/docs/2.14/generated/torch.movedim.html)
- [Embedding parameter renormalization](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)
- [Einsum implicit output labels](https://docs.pytorch.org/docs/2.14/generated/torch.einsum.html)
