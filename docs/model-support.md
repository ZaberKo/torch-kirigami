# Model support

Support depends on one captured call, its arguments, the requested original-coordinate selections, and the resulting compact execution. A model-family name alone does not establish support.

Capture uses FX symbolic tracing and metadata execution. Operator rules describe dependencies. The pruning layer checks whether the resulting edits can run. Physical pruning keeps the module hierarchy and applies declarative tensor and attribute replacements. It does not rewrite Python `forward` code.

Examples provide metadata. They do not specialize tensor-dependent Python control flow. Capture isolates example inputs and registered buffers, then restores buffer bindings, module modes, and protected RNG state on success or failure. It does not copy parameters or roll back arbitrary external side effects. Concurrent model use during capture or application is outside the contract.

## Support outcomes

| Outcome | Meaning |
| --- | --- |
| Supported | Capture, analysis, constraints, and execution rules establish the requested transformation under their stated premises. |
| Requires adaptation | A source change or explicit operator rule can represent the intended transformation. The model author must establish equivalence for that adaptation. |
| Rejected | A stage cannot establish safe capture, complete influence, or executable compaction. The failure scope depends on the stage. |
| Verification limit | Unsupported behavior may escape detection. Successful capture or planning alone does not establish correctness. |

### Failure scope

| Stage | Result |
| --- | --- |
| Capture or metadata execution | Capture raises `CaptureError` and returns no graph. |
| Operator analysis | Missing semantics create barriers over affected dependencies and remain visible in `graph.diagnostics`. Independent subgraphs remain queryable. Unexpected extension errors are not converted to barriers. |
| Manual planning | The full request must be executable. Invalid requests raise `PlanningError`. Planning does not drop requested selections. |
| Automatic planning | The default strategy keeps verified combinations and reports exclusions and search limits. An unmet target raises before mutation. Search limits do not prove mathematical infeasibility. |
| Application | Application checks the model and plan again. It does not choose a new candidate set. |

Unknown operations can use input values as indices or sizes. Their barriers can therefore reach data ancestors across branches. Shared parameters and residual joins can also connect branches. An unsupported operation is not a fixed box that permits unrestricted upstream pruning.

A structural barrier requires successful capture first.
An unregistered call with unknown write effects cannot execute on a parameter or an unisolated constant.
Capture rejects that call before metadata execution.
A rule can declare read-only inputs yet raise `UnsupportedOperation` from `build_spec()`.
That case creates a barrier instead of failing capture.

## Capture and snapshot assumptions

Capture supports FX-traceable modules with supported call signatures and metadata. It binds positional and keyword arguments using the forward signature and applies defaults. Supported nested containers are plain tuples, lists, and dictionaries with supported keys.

Metadata leaves can be tensors, `None`, `bool`, `int`, `float`, `str`, `torch.dtype`, or `torch.device` values. Capture rejects other metadata leaf types.

Capture rejects empty tensor examples and results.

Capture rejects forward hooks, unsupported registration callbacks, undeclared parameter writes, unsupported storage, and metadata values that it cannot represent.
It also rejects inputs or buffers that share storage with parameters.

`nn.Embedding` and `F.embedding` declare their native `max_norm` updates.
Capture warns before execution and permits these updates on directly bound parameters.
Embedding renormalizes only vocabulary rows used by the example.
These changes remain in the original model, including when a later operation causes capture to fail.
Capture preserves existing gradients, which may be stale.
Magnitude metrics read the updated weights.
Recompute gradients before Taylor scoring if the update invalidates prior calibration.
Planning itself remains read-only.
For `F.embedding` with a registered buffer as its weight, capture updates only the isolated buffer copy.
Planning supports this form and does not change the original buffer.
The parameter-update warning does not apply to isolated buffer writes or writes to fresh temporary tensors.

BatchNorm and InstanceNorm normally keep running statistics in registered buffers.
Capture isolates those buffers.
If a model instead registers statistics as Parameters, the same declared-update warning applies when the call updates them.
These native updates persist in the original Parameters.
The declaration does not permit updates through unproved parameter aliases or views.

Warnings use Python's `warnings` module, not `print`.
Callers can record, filter, or promote them to errors.
Promoting the parameter-update warning to an error stops metadata execution before these updates occur.

Some Python tests can run on FX proxies without leaving a graph node. Tensor identity checks, `isinstance(x, torch.Tensor)`, and runtime-state tests such as `x.grad is None` can silently select a branch. Capture does not guarantee an error for every such case. Treat models with these decisions as outside the current contract.

The graph checks detectable structure, tensor identity and layout, module modes, recognized configuration, supported ordinary tensor references, and guarded integer constants. It does not hash ordinary weight values. Rebuild after structural or relevant configuration changes. Recreate optimizers after nonempty pruning because affected parameters become new objects. Optimizer-state migration is not implemented.

## Structural support

The [operator support page](operator-support.md) lists supported families and their main restrictions. This page states cross-cutting conditions.

- Dependency support does not imply a default candidate axis. Default candidate domains come from selected module rules. Explicit requests can still use other supported dependency axes.
- Interface preservation protects external input and output axes by default. Disabling it does not disable operator constraints.
- Shared tensor objects retain their aliases. Distinct tensor objects that share storage are not treated as independently compactable tensors.
- Group balance, divisibility, nonempty axes, layout support, and execution requirements remain active constraints.
- A static plan can use current parameter values from a structurally compatible target without rescoring. Structural configuration and guarded constants must satisfy its preconditions.

Normalization, softmax, reductions, and attention can recompute over a smaller domain. A dense zero mask is not a general numerical reference. Structural resolution does not establish numerical equivalence or task quality.

### Affected requests

The graph rejects only requests that reach unsupported behavior. It can still analyze an independent branch. Shared weights and residual joins can connect branches, so inspect diagnostics before you trust an impact.

Automatic planning can retain independent candidates that pass validation. Manual planning treats the submitted set as one request. It raises if any part of that set cannot execute.

An unresolved impact needs proof or additional choices. Its influence can be complete or incomplete. Check `impact.complete` separately from `impact.status`. Neither an unresolved result nor incomplete influence authorizes physical pruning.

## Layout and view conditions

Convolution, normalization, padding, and other backend-dependent operations can leave output strides unknown. Captured sample strides do not prove compact output strides. The validator does not infer layouts from device, dtype, batch size, or backend settings.

When strides are unknown, accept a view only if it works for every legal input stride. Shape identity, singleton-axis insertion or removal, and splitting one axis into consecutive factors can meet this condition. Merging distinct non-singleton axes requires a layout proof. `contiguous()` establishes the requested layout. `reshape()` can copy when required. Neither operation fixes invalid sizes or coordinate mappings.

Use `size(dim)` when Python needs one changing dimension. A whole-shape read guards the full shape because later Python indexing is outside FX. Reshape and view arguments need supported dimension provenance, fixed algorithmic constants, or valid inferred dimensions. Shape inference alone does not prove axis identity.

## Adaptations

The model author makes an adaptation, then builds a new graph.

| Condition | Adaptation | Preserve this condition |
| --- | --- | --- |
| A view merges axes without a layout proof | Use `reshape(...)` or `contiguous().view(...)`. | Preserve shape and element order. Copying can change alias behavior. |
| A reshape size encodes a changing feature width | Use `size(dim)`, supported integer arithmetic, or valid `-1` inference. | Keep algorithmic constants fixed. |
| Python reads a whole shape but uses one changing dimension | Read that dimension with `size(dim)`. | Any Python decision that uses it must remain valid after compaction. |
| Indexing inserts a new axis with `None` | Use `unsqueeze(dim)` separately. | Remaining indexing must still use supported semantics. |
| In-place behavior lacks an alias and consumer proof | Use an out-of-place call when appropriate. | No consumer may depend on mutation through an alias. |
| `index_select` uses an unregistered constant vector | Register the integer vector as a buffer. | The vector must be static, one-dimensional, and unchanged. |
| A hook supplies forward behavior | Represent that behavior in `forward` or a declared module boundary. | Preserve required computation and gradient behavior. |

An explicit operator rule can describe a third-party operation. Opaque registration creates a capture boundary, but it does not prove internal mathematics, alias behavior, or output strides. The extension author must provide those proofs and the required lowering. See the [operator extension contract](operator-support.md#implementing-an-operator-rule).

## Persistence and rebuilds

A compact checkpoint restores the final model structure. The caller must provide a compatible model definition, including custom modules and gates. The checkpoint does not serialize arbitrary Python forward code.

Ordinary tensor constants keep structural premises but do not automatically save their numeric values. A constructor must restore those values, or the model must register or persist them. Optimizer state, random generators, and data-loader state belong to the training application.

## Verification limits

| Limit | What can happen | Required premise |
| --- | --- | --- |
| Python tests on proxies | Identity, type, or runtime-state checks can select a branch without a recorded graph node. | Do not prune models with these decisions under the current contract. |
| Untracked Python state | Globals, closures, custom object internals, external resources, or callbacks can affect behavior outside the recorded graph. | Keep structural decisions and mutations inside supported model state or declare them through a valid boundary. |
| Changed inputs or context | Example metadata does not cover every input shape, layout, AMP setting, or backend setting. | Recheck changed execution assumptions and rebuild when needed. |
| Extension declarations | Structural checks cannot prove a third-party rule's mathematics or effects. | Test against an independent compact reference and public lifecycle behavior. |
| Concurrent mutation | Other threads, callbacks, or processes can change a model during planning or application. | Do not mutate the model concurrently. |
| Compact-domain changes | Normalization, softmax, attention, and reductions can produce different values on retained domains. | Measure numerical and task-level effects separately. |

Unknown operations and unsupported axes must not authorize scoring from incomplete influence or physical execution. Automatic planning can keep independent verified candidates. Manual requests remain atomic at the planning boundary.

Checkpoint admission has additional rules for hooks, custom state, aliases, and restoration. Ordinary tensor constants do not automatically save their numerical values. Constructors must restore them, or the model must register or persist them explicitly. See [persistence](persistence.md) for checkpoint behavior.
