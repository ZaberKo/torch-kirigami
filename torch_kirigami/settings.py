"""Process-wide resource configuration for analysis and pruning."""

from dataclasses import dataclass, field, fields


@dataclass
class Config:
    """Validated global resource settings, updated between library operations.

    Use the exported `config` instance. `update()` validates all changes before
    committing them; unknown names, booleans, and invalid sizes are rejected.
    Attribute assignments use the same validation. Updates during an operation
    or concurrent library use are unsupported. Settings are not saved in plans.

    Attributes:
        max_parts: Compressed interval/region and identity-proof expansion bound.
        max_literal_elements: Integer tensor elements retained as structural literals.
        max_shuffle_channels: Channels expanded by ChannelShuffle analysis.
        max_container_depth: Configuration, requirement, and extra-state nesting bound.
        max_record_depth: Static and encoded pruning-record nesting bound.
        score_batch_size: Generic scoring batch size; custom strategies may batch explicitly.
        impact_cache_size: Maximum retained planning impact queries; zero disables retention.
        candidate_cache_size: Maximum native rejection proofs per filter. Zero also
            disables reuse of the latest filtered candidate space.
        recipe_cache_size: Maximum retained compiled recipe sets.
        attribute_cache_size: Maximum retained attribute-validation results.
        normalization_cache_size: Maximum GroupMagnitude normalization entries.
        taylor_cache_size: Maximum WeightTaylor axis-statistic vectors.
        taylor_cache_positions: Maximum scalar positions across Taylor vectors.
        taylor_chunk_elements: Taylor statistics chunk bound; oversized slices
            use ordinary region scoring. Zero disables axis statistics.
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
        """Validate every supplied field, then apply the changes together."""
        definitions = {item.name: item for item in fields(self)}
        for name, value in changes.items():
            if name not in definitions:
                raise TypeError(f"Unknown configuration field: {name}")
            minimum = definitions[name].metadata.get("minimum", 0)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a Python integer >= {minimum}")
        self.__dict__.update(changes)

    def reset(self) -> None:
        """Restore all declared defaults; call only between library operations."""
        self.update(**{item.name: item.default for item in fields(self)})

    def as_dict(self) -> dict[str, int]:
        """Return an independent copy for inspection or later restoration."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def analysis_key(self) -> tuple[int, ...]:
        """Identify the settings that require rebuilding structural analysis."""
        return (
            self.max_parts,
            self.max_literal_elements,
            self.max_shuffle_channels,
            self.max_container_depth,
            self.max_record_depth,
        )


config = Config()
