# Configuration

The library reads resource limits and cache sizes from one process-wide object,
`torch_kirigami.config`. Graph, pruner, and metric constructors take no separate
configuration. Most programs never change it. Raise a limit when a diagnostic or
error names it, or shrink caches when memory is tight.

```python
from torch_kirigami import config

saved = config.as_dict()
config.update(score_batch_size=64, max_literal_elements=8192)
config.impact_cache_size = 64  # Assignment is validated the same way.
# ... build graphs, plan, and apply ...
config.update(**saved)  # Or config.reset() to restore the defaults.
```

`update()` validates every name and value before it changes anything. It rejects
unknown names, booleans, non-integers, and values below the minimum. Settings are
not stored in plans or checkpoints, and loading either one uses the current
settings.

## Settings

### Structural limits

These limits bound how much structure the analysis may expand. Each requires a
positive integer.

| Setting | Default | Bounds |
| --- | ---: | --- |
| `max_parts` | 4,096 | Intervals or regions in one compressed selection. |
| `max_literal_elements` | 4,096 | Elements of an integer tensor kept as a structural literal, such as an index buffer. |
| `max_shuffle_channels` | 4,096 | Channels that ChannelShuffle analysis expands into an explicit permutation. |
| `max_container_depth` | 50 | Nesting in module configuration, requirement data, and checkpoint extra state. |
| `max_record_depth` | 100 | Nesting in pruning records and their encoded form. |

Hitting a limit never produces an approximate answer. Depending on where it
happens, the affected operation gets an `analysis_limit` diagnostic, the
operation is treated as unsupported (for an oversized literal), or an error is
raised. Raising a limit allows more work, but it adds no operator support and
does not guarantee that memory or the Python stack will suffice.

### Scoring and caches

Cache sizes control retained work. Set a cache size to zero to disable that
cache. Taylor scoring has a floating-point rounding caveat described below.

| Setting | Default | Controls |
| --- | ---: | --- |
| `score_batch_size` | 32 | Candidates per batch in generic metric scoring. Minimum 1. |
| `impact_cache_size` | 32 | Dependency queries kept during planning, and equivalent analyses shared by one `parameter_groups()` call. |
| `candidate_cache_size` | 4,096 | Base capacity of each candidate filter cache. With a known candidate count, the capacity is the larger of this setting and twice the count. Zero disables these caches and reuse of the last filtered space. |
| `recipe_cache_size` | 4 | Compiled recipe sets kept during planning. |
| `attribute_cache_size` | 32 | Attribute-validation results kept during planning. |
| `normalization_cache_size` | 32 | `GroupMagnitude` normalization entries. |
| `taylor_cache_size` | 128 | `WeightTaylor` per-axis statistic vectors. |
| `taylor_cache_positions` | 262,144 | Total scalar positions across cached Taylor vectors. |
| `taylor_chunk_elements` | 1,048,576 | Elements per chunk when computing Taylor statistics. Larger slices use ordinary region scoring. Zero disables per-axis statistics. |

`DynamicGreedy` scans the remaining candidates after each accepted step. A
cache smaller than one scan can evict a proof before the next scan reuses it.
The filter therefore scales its capacity with the candidate count. Entries are
created on demand. Memory use depends on how many proofs are stored and how
many intervals describe each selection. An existing filter applies a changed
capacity on its next query.

Custom metrics must give the same scores regardless of batch size.
`score_batch_size` does not limit a custom strategy that calls `context.score()`
itself. Taylor chunk size changes the order of floating-point reductions, so
near-tied rankings can differ between chunk sizes. Whether `WeightTaylor` uses
its per-axis statistics also changes the reduction order. It skips them when
`taylor_cache_size` or `taylor_chunk_elements` is zero, or when the axis is
wider than `taylor_cache_positions`. The statistics sum per-position products,
while ordinary scoring reduces each gathered region, so near-tied rankings can
also differ between these settings.

Scoring gathers each parameter region before it reduces the region. The most
selective axis is gathered first, but each step still allocates the gathered
positions times the remaining full axes. Temporary memory can therefore exceed
the selected elements, for example a `[2, n]` slice for a `[2, 2]` region.

## When to change settings

Change settings only **between** library operations. All threads share the
object. Do not change it concurrently or from inside an operator rule, metric, or
strategy.

A dependency graph records the five structural limits when it is built. If they
differ when a dependency query checks freshness, it raises `StaleGraphError`,
even for a cached result. Recorded metadata remains available for inspection.
Rebuild the graph to analyze requests under the new limits. Scoring and cache
settings can change at any time between operations. Existing planning contexts
and metrics drop the affected caches on their next use.

`Greedy.max_trials` and `DynamicGreedy.max_trials` are algorithm arguments, not
global settings.
