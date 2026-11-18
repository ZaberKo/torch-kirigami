"""Process-wide resource limits and cache sizes for analysis and pruning."""

from dataclasses import dataclass, field, fields


@dataclass
class Config:
    """Validated resource limits and cache sizes shared by the whole process.

    Use the exported `config` instance rather than creating new instances; the
    library reads only that instance. `update()` validates every change before
    it commits any of them, and plain attribute assignment uses the same
    validation. Unknown names, booleans, non-integers, and values below a
    field's minimum are rejected. Change settings only between library
    operations: updates during an operation or concurrent use from several
    threads are unsupported. Settings are not saved in plans.

    The structural `max_*` limits are recorded when a `DependencyGraph` is
    built. If they differ when the graph is used later, the graph raises
    `torch_kirigami.StaleGraphError` and must be rebuilt. Cache sizes and `score_batch_size`
    can change between operations without rebuilding a graph.

    Attributes:
        max_parts: Maximum number of intervals or regions in a compressed
            selection, and the expansion bound for identity proofs.
        max_literal_elements: Maximum number of integer tensor elements that
            analysis retains as structural literals, such as index tensors.
        max_shuffle_channels: Maximum channel count that channel-shuffle
            analysis expands into an explicit permutation.
        max_container_depth: Maximum nesting depth for module configuration,
            requirements, and checkpoint extra state.
        max_record_depth: Maximum nesting depth for static and encoded pruning
            records.
        score_batch_size: Number of candidates per batch in generic scoring.
            Custom strategies can choose their own batching.
        impact_cache_size: Maximum number of retained planning impact queries.
            Zero disables retention.
        candidate_cache_size: Base capacity of each proof cache in a candidate
            filter. When enabled, each cache holds up to the larger of this
            setting and twice the known candidate count. This allows
            `DynamicGreedy` to reuse proofs across scans. Zero disables these
            caches and reuse of the latest filtered candidate space.
        recipe_cache_size: Maximum number of retained compiled recipe sets.
        attribute_cache_size: Maximum number of retained attribute-validation
            results.
        normalization_cache_size: Maximum number of `GroupMagnitude`
            normalization entries.
        taylor_cache_size: Maximum number of `WeightTaylor` per-axis statistic
            vectors.
        taylor_cache_positions: Maximum total number of scalar positions across
            the cached Taylor statistic vectors.
        taylor_chunk_elements: Maximum number of elements per chunk when Taylor
            statistics are computed. Larger slices fall back to ordinary region
            scoring. Zero disables per-axis statistics.
    """

    max_parts: int = field(default=4096, metadata={"minimum": 1})
    max_literal_elements: int = field(default=4096, metadata={"minimum": 1})
    max_shuffle_channels: int = field(default=4096, metadata={"minimum": 1})
    max_container_depth: int = field(default=50, metadata={"minimum": 1})
    max_record_depth: int = field(default=100, metadata={"minimum": 1})
    score_batch_size: int = field(default=32, metadata={"minimum": 1})
    impact_cache_size: int = 32
    candidate_cache_size: int = 4096
    recipe_cache_size: int = 4
    attribute_cache_size: int = 32
    normalization_cache_size: int = 32
    taylor_cache_size: int = 128
    taylor_cache_positions: int = 262_144
    taylor_chunk_elements: int = 1_048_576

    def __setattr__(self, name: str, value: int) -> None:
        self.update(**{name: value})

    def update(self, **changes: int) -> None:
        """Validate every supplied setting, then apply all changes together.

        Args:
            **changes: New values keyed by field name. Each value must be a
                Python `int` (not `bool`) at or above the field's minimum,
                which is one for limits and the scoring batch size and zero
                for cache sizes.

        Raises:
            TypeError: A name is not a configuration field.
            ValueError: A value is not an integer or is below its minimum.
                No setting changes when an error is raised.
        """
        definitions = {item.name: item for item in fields(self)}
        for name, value in changes.items():
            if name not in definitions:
                raise TypeError(f"Unknown configuration field: {name}")
            minimum = definitions[name].metadata.get("minimum", 0)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a Python integer >= {minimum}")
        self.__dict__.update(changes)

    def reset(self) -> None:
        """Restore every setting to its declared default.

        Call this only between library operations.
        """
        self.update(**{item.name: item.default for item in fields(self)})

    def as_dict(self) -> dict[str, int]:
        """Return the current settings as a new dictionary.

        Returns:
            A copy keyed by field name. Pass it to `update(**settings)` to
            restore these settings later.
        """
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def analysis_key(self) -> tuple[int, ...]:
        """Return the settings that a dependency graph records when it is built.

        Returns:
            The structural `max_*` limits in declaration order. A graph whose
            recorded key differs from the current key must be rebuilt.
        """
        return (
            self.max_parts,
            self.max_literal_elements,
            self.max_shuffle_channels,
            self.max_container_depth,
            self.max_record_depth,
        )


config = Config()
"""Process-wide resource settings for dependency analysis and pruning."""
