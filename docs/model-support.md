# Model Support Contract

This document defines model admission, supported structural transformations, required adaptations, rejection scope, and known verification limits. It applies to the dependency graph and physical pruning APIs. Exact operator registrations and argument restrictions are maintained in [Operator support and extension](operator-coverage.md).

## 1. Scope and guarantees

Support is specific to a captured call, its arguments, the requested original-coordinate selections, and the resulting compact execution. A valid request must satisfy capture assumptions, dependency constraints, and execution requirements.

The capture pipeline is fixed:

```text
nn.Module + bound example arguments
    → FX symbolic tracing
    → isolated ShapeProp execution
    → operator rule analysis
    → joint pruning analysis and recipe validation
```

Examples supply execution metadata; they do not specialize data-dependent Python control flow. Capture uses public FX interfaces without export, autograd capture, or a fallback tracer. Physical pruning retains the original module hierarchy and executes declarative tensor and attribute replacements. It does not rewrite Python `forward` source.

| Classification | Contract |
| --- | --- |
| Supported | Existing capture, analysis, and execution rules establish the requested transformation under their declared premises. |
| Requires adaptation | A source rewrite or explicit operator contract can make the intended transformation representable. Semantic equivalence must be established for the adaptation. |
| Explicitly rejected | The current implementation cannot establish capture safety, complete structural influence, or executable compaction. Rejection scope depends on the stage. |
| Known verification limit | Unsupported behavior may escape detection. Successful capture or planning alone does not establish correctness for such models. |

### 1.1 Failure scope

| Stage | Result and scope |
| --- | --- |
| Capture or metadata execution | `CaptureError`; no usable dependency graph is returned. The entire build attempt fails. |
| Rule analysis | Unsupported semantics become barriers over their structural and data dependencies. Independent subgraphs remain queryable. Unexpected extension programming errors are not silently converted into barriers. |
| Manual planning | The complete submitted request must be executable. An invalid request raises `PlanningError`; requested selections are not silently dropped. |
| Default automatic planning | The strategy retains validated combinations and reports excluded candidates, shortfall, and search limits. Temporary imbalance may be resolved by combining candidates. Arbitrary custom metric or strategy failures are not guaranteed to become candidate exclusions. |
| Application | Model and plan preconditions are revalidated before submission. Invalid plans or incompatible targets fail the application attempt; `apply()` does not rerun candidate selection. |

Barrier scope follows dependencies rather than module boundaries. Unknown operations may use input values as indices or dimensions, so their data ancestors can also be protected. Shared parameters and residual joins can extend the affected region across branches. An unsupported operation cannot safely be treated as an isolated fixed box with unrestricted upstream pruning.

### 1.2 Selection and lifecycle invariants

- All selections and recipes use coordinates from the current graph. Shared uses must agree on retained coordinates, ordering, and structural attribute values.
- `preserve_io=True` protects every external input/output tensor axis by default. Disabling it removes interface protection only; operator constraints remain active.
- Group balance, divisibility, nonempty dimensions, execution support, and budget limits constrain admissible selections. Budget shortfall is not proof of mathematical infeasibility.
- A static plan can use the current weights of a structurally compatible target without rescoring. Configuration and guarded constants must still satisfy its preconditions.
- After nonempty pruning, rebuild dependency graphs and graph-bound training objects, and recreate the optimizer. Resized parameters are new objects; optimizer-state migration is not implemented.

## 2. Supported models and transformations

### 2.1 Graph structure, bindings, and execution

| Domain | Supported contract |
| --- | --- |
| Module composition | FX-traceable `nn.Module` hierarchies, Sequential, and custom compositions whose affected calls have applicable rules. |
| Control and data flow | Configuration-constant branches, fixed-iteration loops, residual connections, multiple consumers, and joins within the captured graph. |
| Repeated calls and sharing | Separate call records for repeated module use; identity-based parameter deduplication and preservation of supported registration/reference aliases. |
| Arguments and results | Forward-signature binding for positional arguments, keyword arguments, and defaults; supported tensor/scalar trees in ordinary tuple/list/dict containers. |
| Ordinary tensor attributes | Structural premises and supported aliases for directly addressable tensors and ordinary containers. Unregistered constants remain constructor-owned unless explicitly persisted. |
| Execution modes | Capture under the caller's train/eval and gradient modes, with CPU/CUDA execution where the model and operators support the selected device. |
| Isolation | Supported example inputs and registered buffers are isolated. Buffer bindings, module modes, and protected PyTorch RNG state are restored on success and failure. Parameter values are not copied. |

Binding a scalar argument does not specialize a Python branch that requires evaluating an FX Proxy. Configuration-controlled capture refers to explicit, supported model configuration. Changed structural configuration requires a new graph.

Operator registrations match exact module types, function objects, and Tensor method names. Custom subclasses may be traced through their implementations; inheritance alone does not authorize reuse of a base module's semantics.

### 2.2 Structural domains

| Family | Supported transformations |
| --- | --- |
| Linear and Embedding | Linear input/output features and applicable bias broadcasting, including supported functional rank-one weights; embedding feature width. |
| Conv/ConvTranspose 1D–3D | Ordinary, grouped, and depthwise channel compaction. Per-group local selections may differ when balance and layout requirements hold; original group order is preserved. |
| Normalization and affine operations | Related feature/channel positions, affine tensors, statistics buffers, and explicitly bound attributes for BN, LN, GN, InstanceNorm, RMSNorm, PReLU, and normalize. |
| Elementwise and matrix operations | Registered arithmetic, activations, comparisons, broadcasting, where/masked_fill, matrix contractions, and supported explicit-output einsum equations. |
| Coordinate transformations | Permutations, proven reshape/flatten, squeeze/unsqueeze, static slicing/narrow/index_select, concatenation, partitioning, repetition, and expansion with preserved original-coordinate correspondence. |
| Grouped coordinate operations | GLU half alignment, compatible ChannelShuffle patterns, PixelShuffle/Unshuffle channel blocks, and supported batched 2D Unfold/Fold mappings. |
| Spatial operations | Pooling/interpolation channel correspondence and coordinates on axes unaffected by padding or cropping. |
| Reductions | Supported reductions and softmax variants distinguish reduced axes from retained axes and recompute on the compact domain. |
| Explicit-head attention | SDPA Q/K contraction features, V output width, batch/head broadcasting, masks, and valid GQA group or multiplier changes. |
| Native MultiheadAttention | Balanced embedding-width changes with a fixed head count, including packed and separate projections. |
| Storage-related calls | Coordinate correspondence for casts, device moves, clone, detach, and contiguous; in-place calls only when declared effects and consumer relationships establish safety. |

Default candidate domains come from module Linear, Conv/ConvTranspose, Embedding, and native MHA declarations. Dependency support does not imply an independent candidate domain. Depthwise automatic candidates remove complete logical groups; explicit requests may express legal multiplier contraction.

### 2.3 Deliberately fixed dimensions

These restrictions are defined operator semantics. They do not indicate missing analysis for the entire operator.

| Family | Fixed domain |
| --- | --- |
| Convolution | Spatial and kernel coordinates. |
| GroupNorm | Group count; retained channels must remain balanced across groups. |
| Pooling, interpolation, Unfold/Fold | The relevant spatial and kernel coordinates. |
| Padding/cropping | Transformed axes, including equal-length crop/pad combinations that change coordinate identity. |
| Embedding | Vocabulary/token-ID positions. |
| Stack/unbind | Input or output port count and identity. |
| PixelShuffle/Unshuffle | Spatial positions; channels participate through complete blocks. |
| Native MHA | Head count and batch/token/mask/attention-weight positions. Internal whole-head deletion at unchanged external width is not implemented. |
| Causal SDPA | Q/K token coordinates, preserving the implicit causal mask's meaning. |

Extensions may declare additional fixed axes. These constraints remain active when interface preservation is disabled.

### 2.4 Layout contract

Convolution, transposed convolution, BN, GN, padding, and other backend-dependent operations retain unknown output strides. The validator does not infer output layouts from CPU/CUDA selection, dtype, batch size, or cuDNN settings. Captured sample strides do not establish compact output strides.

Unknown strides permit view transformations proven valid for every legal input stride: shape identity, singleton-axis insertion/removal, and splitting individual axes into consecutive factors. Original size expressions and coordinate constraints must still hold. These transformations propagate stride uncertainty to later consumers.

A view merging distinct non-singleton axes requires a layout proof. Explicit `contiguous()` establishes the requested layout; `reshape()` permits copying where required. Neither operation repairs invalid sizes or index correspondence.

## 3. Required source adaptations and extensions

Adaptations are performed by the model author, followed by graph reconstruction. Diagnostics and automatic-plan exclusion reasons provide applicable guidance.

| Condition | Adaptation | Equivalence obligation |
| --- | --- | --- |
| Merging view with unproved strides | Use `reshape(...)` or `contiguous().view(...)`. | Preserve the intended shape and element order; copying must be acceptable. Storage aliasing and downstream mutation semantics may change. |
| Hardcoded reshape/view size represents a variable feature width | Express the dimension with `size(dim)`, supported integer arithmetic, or valid `-1` inference. | Keep algorithmic constants fixed. Inferred element count alone does not establish axis identity. |
| Eager whole-shape read constrains unused dimensions | Read only the required dimension with `size(dim)`. | The selected dimension and any configuration decision using it must remain valid after compaction. |
| New-axis indexing such as `x[:, None, ...]` | Express singleton insertion with `unsqueeze(dim)` separately from indexing. | Remaining indexing must have supported semantics; this is not a general advanced-index conversion. |
| In-place call lacks an alias/consumer proof | Use an out-of-place operation where appropriate. | No consumer may rely on modifying the original tensor through an alias. |
| Constant index_select vector lacks a registered source | Register the immutable integer vector as a buffer. | The index must satisfy the supported static one-dimensional rule and analysis limits. Registration does not make mutable or generated indices constant. |
| Forward semantics depend on hooks | Represent necessary computation explicitly in forward or through a declared module boundary. | Removing a hook must not silently remove required computation or gradient behavior. |
| Decision can be expressed as fixed model configuration | Capture with an explicit supported configuration. | This defines one configuration; it does not preserve arbitrary data-dependent branch selection. |

### 3.1 Opaque and third-party operators

Use a local `OperatorRegistry` and the unified `OperatorRule` interface. The rule must declare structural relations, constraints, requirements, candidate domains where applicable, and capture-time effects. Use existing region, block, and partition descriptors before introducing operator-specific logic. Supply lowering only when existing declarations cannot express the required physical edits.

Opaque registration establishes a capture boundary. It does not prove the internal implementation's semantics. Extension authors remain responsible for the original forward's validity, alias behavior, and any declared output strides. Opaque root modules must satisfy the supported signature contract; variadic root signatures are rejected.

## 4. Explicit rejection boundaries

### 4.1 Build-wide rejection

The following conditions prevent a usable graph from being returned:

| Condition | Enforcement boundary |
| --- | --- |
| Tensor data or input-shape-dependent Python control flow requiring concrete Proxy values | FX capture fails; no example-selected branch or alternate capture mechanism is substituted. |
| Detected eager tensor data extraction or unrecorded runtime-state reads | Trace guards reject operations such as buffer item/bool/tolist/equal reads. Undetectable Proxy decisions are addressed in Section 5. |
| Detected parameter/alias writes, out= mutation, unisolated constant writes, or unrecorded buffer structure/registration changes | Capture effect and mutation checks. Embedding max_norm is rejected because it writes parameters. |
| Input or buffer storage aliases parameter storage | Isolation cannot preserve the parameter-sharing contract without copying weights. |
| Unsupported storage or input copying | Sparse/quantized/non-dense-strided registered storage, or inability to safely copy example state. |
| Unsupported forward hooks or global registration callbacks | Capture cannot establish their effects through the declared graph. |
| Invalid arguments or metadata execution | Signature mismatch, execution failure, unsupported metadata values, or zero-element tensor examples/results. |
| Unsupported opaque root signature | Variadic opaque root arguments cannot be represented by the current root-call construction. |
| Unsupported ordinary-container keys | Tensor/Module keys can hide bindings outside the supported reference model. |

### 4.2 Rejection of affected pruning requests

A successfully captured graph may still reject requests involving:

- Missing operator semantics or unsupported argument overloads.
- Data-dependent advanced/boolean indexing, dynamic index remapping, or unsupported diagonal/repeated-label einsum.
- Original reshape sizes, slices, split boundaries, or port counts that cannot preserve the compact shape and original-coordinate mapping without editing forward.
- Unproved merging-view layouts or unsafe alias/mutation paths.
- Unregistered constants requiring physical replacement without a supported original binding.
- Distinct tensor objects sharing storage; supported aliases of the same Parameter are handled by identity instead.
- Affected parameters with gradient hooks or custom Tensor/Parameter subclass behavior that replacement cannot preserve.
- Conflicting shared-use layouts, coordinate mappings, or attribute assignments.
- Unresolved structural constraints, an insufficient budget, or exhausted analysis/search limits.

Incomplete influence must not authorize scoring from a partial impact or physical execution. Default automatic planning may preserve independent validated candidates; manual requests remain atomic at the planning boundary.

### 4.3 Mechanisms without general built-in support

RNN/PackedSequence, dynamic MoE routing, dynamic KV-cache maintenance, arbitrary custom autograd/CUDA kernels, and sharded parameters have no general admission guarantee. A particular implementation may be expressible through captured supported operations or an explicit extension. Model-family names alone do not determine support.

## 5. Known verification limits

| Limit | Failure mechanism | Required premise |
| --- | --- | --- |
| Silent FX branch selection | Tests such as `x.grad is None`, `self.weight.grad is None`, `isinstance(x, torch.Tensor)`, or object identity checks may execute on proxies without leaving a usable graph node. | Models containing these decisions are outside the capture contract unless represented by a valid explicit boundary. A CaptureError is not guaranteed. |
| Untracked Python state | Globals, closures, custom object internals, external resources, and arbitrary callback side effects may influence structure without entering the recorded premises. | Structural decisions and mutations must be declared or represented within the supported model state. Isolation and rollback do not cover arbitrary external effects. |
| Changed inputs or execution context | Sample metadata and known configuration checks do not cover every input shape/layout or global AMP/backend setting. | Revalidate changed execution assumptions and rebuild when required. Successful capture is not whole-program validity analysis. |
| Incorrect extension declarations | Structural validation cannot prove a third-party implementation's mathematics, effects, or asserted stride guarantees. | Validate extensions against independent compact references and public lifecycle tests. |
| Concurrent mutation | Planning and commit do not lock all access by threads, callbacks, or other processes. | Do not concurrently mutate a model during analysis or application. |
| Numerical and task-level changes | Normalization, softmax, attention, and reductions may recompute on smaller domains. | Verify compact-domain numerical semantics and task quality separately; resolved structural analysis does not establish equivalence to the unpruned model. |

These limits must remain distinct from enforced fixed dimensions and diagnosed unsupported requests. They cannot be classified as safely protected subgraphs solely because build or tests succeed.

## 6. Persistence and validation requirements

Ordinary tensor constants carry structural compatibility premises, but their numerical values are not automatically included in checkpoints. Constructors must restore them consistently, or they must be registered or explicitly persisted. Checkpoint admission additionally constrains hooks, custom state, storage aliases, and restoration behavior; see [Persistence](persistence.md).

Changes to this contract require tests at the boundary where users observe the behavior:

| Contract | Implementation and regression references |
| --- | --- |
| Capture, isolation, and ordinary references | [capture.py](../torch_kirigami/capture.py), [bindings.py](../torch_kirigami/bindings.py), [Dependency graph design](dependency-graph-design.md) |
| Operator domains and fixed axes | [Operator support](operator-coverage.md), [registered-entry tests](../tests/operators/test_registered_entries.py) |
| Request rejection and diagnostic propagation | [validation.py](../torch_kirigami/pruning/validation.py), [planning diagnostics](../tests/integration/test_planning_diagnostics.py) |
| Unknown layouts and valid adaptations | [view guards](../tests/integration/test_view_guards.py), [backend layout tests](../tests/integration/test_backend_layouts.py) |
| Metadata observations and capture gaps | [metadata guards](../tests/integration/test_metadata_guards.py), [verification coverage](testing-coverage.md) |

Behavioral regressions should pair rejection with a valid alternative and preserved-state assertions. Cross-layer changes require public build/plan/apply or save/load coverage. Numerical references must be independent of the implementation; zero masking is not a universal reference for normalization or attention.
