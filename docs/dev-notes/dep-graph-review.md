# Dependency graph package review

Review date: 2026-10-14. This note records implementation decisions; the public
contract is defined in [the dependency guide](../dependency-graph.md).

## Package boundary

Capture, coordinates, relations, constraints, operator definitions, registration,
and graph queries now live in `torch_kirigami/dep_graph/`. The package initializer
has no imports. The root package continues to export the public classes.
Internal consumers use concrete submodule imports; old module paths have no shims.
`bindings.py`, `configuration.py`, and `errors.py` remain shared infrastructure.

## Class review

All 50 classes in the package were checked for responsibility, construction state,
ownership, extension behavior, and unnecessary abstraction. The rows below group
classes with the same decision; each named class was inspected.

| Classes | Decision and reasoning |
| --- | --- |
| `DependencyGraph` | Keep the public query and freshness boundary established by the preceding refactor. It delegates construction and solving. Correct attribute-update input typing to an iterable of `(path, value)` pairs, matching its caller and implementation. |
| `GraphBuilder` | Keep one build owner. Its mutable tables serve one construction lifecycle; splitting them into services would require shared mutable state across services. |
| `GraphData` | Keep the completed record separate from construction scratch data and solver caches. |
| `DependencySolver` | Keep indexes, closure computation, and constraint evaluation together. Fix the nested-extension cache eligibility defect described below. |
| `IndexSet` | Keep normalized interval arithmetic together. Interval limits and exact set operations are required, not incidental control flow. |
| `Region`, `Selection` | Keep Cartesian operations and union semantics with their records. Existing independent coordinate tests cover overlap, axes, and scalar domains. |
| `TensorRef`, `AxisRef` | Keep reference identity and canonical axis validation together. |
| `TensorRefMap` | Correct the key and value annotations: queries take `TensorRef`; values are tuples of a generic item type. Runtime lookup behavior is unchanged. |
| `Relation`, `AxisPort`, `BlockMap` | Keep the extension contract, scoped coordinates, and block mapping as separate existing responsibilities. |
| `AxisRelation`, `BroadcastRelation`, `ReshapeRelation`, `PermuteRelation`, `SliceRelation` | Keep one relation per mapping law. Their propagation methods are cohesive and do not select pruning policy. |
| `Constraint`, `NonEmpty`, `Fixed`, `Balanced`, `BlockBalance`, `Divisible` | Keep constraint checks separate from choosing a completion. Nested extension fields must not receive built-in cache assumptions. |
| `Barrier`, `AxisBarrier`, `LayoutConstraint` | Keep distinct support and layout conditions; combining them would obscure completeness versus legality. |
| `Diagnostic`, `Provenance`, `Impact` | Keep reporting and query results as records. They do not execute pruning. |
| `ShapeExpr`, `ArgumentRef`, `Requirement` | Keep dimension provenance, call-argument identity, and required changes explicit. They do not form a new execution IR. |
| `CallRef`, `TensorFacts`, `OperationContext` | Keep captured call facts together. Context accessors resolve arguments and bindings without storing scoring state. |
| `CandidateAxis`, `PartitionedLayout`, `OutputContract`, `OperatorSpec`, `CallEffects` | Keep declarative operator records. Frozen containers do not imply arbitrary extension objects are deeply immutable. |
| `OperatorRegistrar`, `OperatorRegistry`, `OperatorRule` | Keep the minimal registration protocol, local lookup tables, and unified operator definition. No singleton or second execution registry is needed. |
| `_TraceDataGuard` | Keep trace-time data-access checks in one mode. Correct its registered-object table annotation to `dict[int, str]`: values are model paths, not tensors. |
| `_LeafTracer`, `_MetadataPropagator`, `CaptureResult` | Keep the public FX hook, metadata extraction, and capture result separate. None requires another abstraction layer. |
| `CallArgumentConstraint`, `_PendingLayout` | Keep scalar-argument checking with shape evaluation. The internal exception distinguishes a layout needing further completion from unknown provenance. |

No additional class split was justified by this review. The preceding graph
refactor already separated the principal state lifetimes. Long files containing
several small records do not by themselves imply oversized classes.

## Confirmed defect and regression coverage

The constraint index recognized an exact `Balanced` or `Divisible` as reusable
without checking nested field types. A subclassed partition or integer factor can
execute custom behavior. Reusing the original empty-selection result could then
report `resolved` even when exhaustive evaluation reported a violation.

Index preparation now requires recursively verified built-in records. Nested
extensions are evaluated on every query. The same immutability predicate serves
relation caching and constraint indexing. This adds no cache or solver layer.
Tests cover nested `IndexSet` and integer subclasses, empty and unrelated requests,
changes in callback results, rejected planning without model mutation, a valid
alternative, independent compact outputs, and backward on CPU and CUDA.

An exhaustive-reference test still overrode a private method on `DependencyGraph`
after that method moved to the solver. The override was no longer called. The test
now intercepts the solver method and asserts that the exhaustive path executed.
Its public plan and applied-output comparisons remain in place.

The attribute-update input, portable-reference mapping, and trace guard issues were annotation defects;
they are not reported as numerical pruning failures.

## Follow-up optimization fixes

The independent review found that the same nested-extension assumption also
affected identity proofs in planning. Native-record validation now lives in
`dep_graph/_records.py`; propagation, identity indexing, and scoring eligibility
share it. Candidate discovery also uses the shared identity index instead of
maintaining a second proof implementation.

Explicit full-tensor ports now qualify for identity proofs. Partial partitions
still require ordinary analysis. This enables the existing CNN scoring
optimization without changing its formula. Isomorphic templates retain their
complete-closure witness and final execution checks. Activation assertions now
accompany optimized-versus-generic plan comparisons, so using the generic path on
both sides can no longer hide a disabled optimization.
