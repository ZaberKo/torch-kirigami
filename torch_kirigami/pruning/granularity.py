"""Translate exact module alignment settings into ordinary graph constraints."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import nn

from ..contracts import Divisible
from ..graph import DependencyGraph
from .types import PlanningError


@dataclass(frozen=True)
class Granularity:
    """Require retained logical widths to be multiples of configured factors.

    Args:
        default: Positive integer factor used without an override. One adds no
            alignment constraint and never disables an operator's own constraints.
        by_type: Overrides for exact Module types; subclasses do not match.
        by_path: Overrides for exact original module paths, ahead of type rules.
            The empty path denotes the root. Prefixes and wildcards are unsupported.

    Mappings are copied and frozen. Requirements also apply to unchanged axes and
    manual requests; alignment never permits exceeding the channel budget.
    """

    default: int = 1
    by_type: Mapping[type[nn.Module], int] = field(default_factory=dict)
    by_path: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        types, paths = dict(self.by_type), dict(self.by_path)
        if any(not isinstance(t, type) or not issubclass(t, nn.Module) for t in types):
            raise TypeError("Granularity by_type requires Module types")
        if any(not isinstance(p, str) or any(c in p for c in "*?[]") for p in paths):
            raise ValueError("Granularity by_path requires exact module paths")
        if any(
            type(n) is not int or n <= 0 for n in (self.default, *types.values(), *paths.values())
        ):
            raise ValueError("Granularity factors must be positive integers")
        object.__setattr__(self, "by_type", MappingProxyType(types))
        object.__setattr__(self, "by_path", MappingProxyType(paths))


def alignment_constraints(
    graph: DependencyGraph, config: Granularity
) -> tuple[tuple[Divisible, ...], tuple[str, ...]]:
    """Resolve aliases before deduplication; return constraints and readable notes."""
    if not isinstance(config, Granularity):
        raise TypeError("Expected a Granularity configuration")
    modules = dict(graph.model.named_modules(remove_duplicate=False))
    declarations = {}
    for operation in graph.operations():
        if operation.module is not None:
            domains = graph.operator_spec(operation).candidates
            if domains:
                entry = declarations.setdefault(id(operation.module), [])
                entry.extend(domains)
    for path in config.by_path:
        if path not in modules:
            raise PlanningError(f"Granularity module path does not exist: {path!r}")
        if id(modules[path]) not in declarations:
            raise PlanningError(f"Granularity module {path!r} declares no logical channel axes")
    aliases = {}
    for path, module in modules.items():
        if id(module) in declarations:
            aliases.setdefault(id(module), []).append(path)
    constraints, notes, matched_types = [], [], set()
    for identity, paths in aliases.items():
        module_type = type(modules[paths[0]])
        if module_type in config.by_type:
            matched_types.add(module_type)
        base = config.by_type.get(module_type, config.default)
        explicit = {config.by_path[p] for p in paths if p in config.by_path}
        if len(explicit) > 1:
            raise PlanningError(
                f"Conflicting granularity overrides for shared module aliases: {paths}"
            )
        # One explicitly addressed alias configures the module object, including
        # its other aliases. Unspecified aliases are not competing overrides.
        factor = next(iter(explicit)) if explicit else base
        if explicit:
            source = "path " + ", ".join(repr(p) for p in paths if p in config.by_path)
        elif module_type in config.by_type:
            source = f"type {module_type.__module__}.{module_type.__qualname__}"
        else:
            source = "default"
        for domain in dict.fromkeys(declarations[identity]):
            if factor != 1:
                constraints.append(Divisible(domain.alignment_axis or domain.axis, factor))
            if factor != 1 or explicit or module_type in config.by_type:
                notes.append(
                    f"Granularity {domain.key}: multiple of {factor} ({source}; module {paths[0]!r})"
                )
    for module_type in config.by_type.keys() - matched_types:
        notes.append(
            f"Granularity type {module_type.__module__}.{module_type.__qualname__}: no declared axes matched"
        )
    return tuple(dict.fromkeys(constraints)), tuple(sorted(set(notes)))
