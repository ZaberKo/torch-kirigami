"""architecture / imports contracts."""

import ast
import inspect
import subprocess
import sys
from dataclasses import is_dataclass
from graphlib import TopologicalSorter
from importlib.util import resolve_name
from pathlib import Path
from typing import get_type_hints

import torch_kirigami
import torch_kirigami.pruning


def test_package_import_dependencies_are_explicit_and_acyclic():
    package = Path(torch_kirigami.__file__).parent
    modules = {}
    for path in package.rglob("*.py"):
        parts = path.relative_to(package.parent).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        modules[name] = (path, ast.parse(path.read_text()))

    dependencies = {name: set() for name in modules}
    for name, (path, tree) in modules.items():
        parent = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            assert not isinstance(node, ast.Name) or node.id != "TYPE_CHECKING", path
            assert not isinstance(node, ast.Attribute) or node.attr != "TYPE_CHECKING", path
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            # Hidden local/conditional imports must not conceal a reverse edge.
            assert node in tree.body, f"Non-top-level import: {path}:{node.lineno}"
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                target = resolve_name("." * node.level + (node.module or ""), parent)
                targets = [target, *(f"{target}.{alias.name}" for alias in node.names)]
            dependencies[name].update(t for t in targets if t in modules)

    # Include package export modules in the declared dependency graph. Ordinary
    # Python parent-package initialization is not an extra source-level edge.
    assert set(TopologicalSorter(dependencies).static_order()) == set(modules)


def test_dependency_core_does_not_import_pruning():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import torch_kirigami; "
            "assert not any(n == 'torch_kirigami.pruning' or "
            "n.startswith('torch_kirigami.pruning.') for n in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_public_record_annotations_resolve_at_runtime():
    for package in (torch_kirigami, torch_kirigami.pruning):
        for name in package.__all__:
            value = getattr(package, name)
            if inspect.isclass(value) and is_dataclass(value):
                get_type_hints(value)

    hints = get_type_hints(torch_kirigami.OperationContext)
    assert hints["metadata"].__args__[1] is torch_kirigami.TensorFacts
