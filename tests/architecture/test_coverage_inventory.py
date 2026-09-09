"""Keep coverage navigation aligned with public exports, internal classes and entries."""

import ast
import importlib
import json
from pathlib import Path

import torch_kirigami
import torch_kirigami.pruning
from torch_kirigami import OperatorRegistry

ROOT = Path(__file__).resolve().parents[2]


def test_every_export_and_internal_class_has_contract_test_links():
    inventory = json.loads((ROOT / "docs/testing-coverage.json").read_text())
    rows = inventory["objects"]
    names = {row["object"] for row in rows}
    assert len(names) == len(rows)
    exports = {
        f"{module.__name__}.{name}"
        for module in (torch_kirigami, torch_kirigami.pruning)
        for name in module.__all__
    }
    assert not exports - names, f"Missing public contracts: {exports - names}"
    definitions = set()
    for path in (ROOT / "torch_kirigami").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        definitions.update(
            f"{module}.{node.name}"
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef)
        )
    covered_definitions = set()
    for name in names:
        module, field = name.rsplit(".", 1)
        obj = getattr(importlib.import_module(module), field)
        covered_definitions.add(f"{obj.__module__}.{obj.__qualname__}")
    assert not definitions - covered_definitions, (
        f"Missing class contracts: {definitions - covered_definitions}"
    )
    for row in rows:
        assert row["contracts"], row["object"]
        for contract in row["contracts"]:
            assert contract["behavior"] and contract["conditions"] and contract["tests"]
            for node_id in contract["tests"]:
                path, name = node_id.split("::")
                tree = ast.parse((ROOT / path).read_text())
                # This guard checks navigation only. Numerical assertions may
                # live in shared or example helpers and are reviewed separately.
                assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body), (
                    node_id
                )


def test_registered_entry_inventory_has_no_missing_or_obsolete_targets():
    inventory = json.loads((ROOT / "docs/testing-coverage.json").read_text())
    registry = OperatorRegistry.default()
    expected = {
        f"{kind}:{target if isinstance(target, str) else target.__module__ + '.' + target.__name__}"
        for kind, table in (
            ("module", registry.modules),
            ("function", registry.functions),
            ("method", registry.methods),
        )
        for target in table
    }
    entries = inventory["operators"]
    assert len(entries) == len({entry["entry"] for entry in entries})
    assert {entry["entry"] for entry in entries} == expected
    for entry in entries:
        assert (
            entry["test"]
            == "tests/operators/test_registered_entries.py::test_native_entry_matches_and_propagates_declared_axis"
        )
        assert entry["conditions"] and entry["contract"]
