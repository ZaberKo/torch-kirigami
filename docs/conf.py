"""Sphinx configuration for the user guides and static API reference."""

import sys
from pathlib import Path

import tomllib

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS))

project = "Torch Kirigami"
copyright = "2026, zaber"
release = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "autodoc2",
]

exclude_patterns = ["_build", "dev-notes", "dev-notes/**", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "dollarmath", "fieldlist"]
myst_fence_as_directive = ["mermaid"]
myst_heading_anchors = 4

# HTML diagrams render in the browser. Keep the renderer fixed to the version
# checked against every diagram in the published guides.
mermaid_version = "12.3.0"
mermaid_height = "auto"

# Static analysis keeps the docs build independent of PyTorch installation.
autodoc2_packages = ["../torch_kirigami"]
autodoc2_render_plugin = "myst"
autodoc2_module_all_regexes = [
    r"torch_kirigami",
    r"torch_kirigami\.pruning",
    r"torch_kirigami\.sparsity",
]
autodoc2_docstring_parser_regexes = [(r"torch_kirigami(?:\..*)?", "google_docstrings")]
autodoc2_hidden_objects = ["private", "dunder", "inherited"]
autodoc2_class_docstring = "both"
autodoc2_sort_names = True
napoleon_use_ivar = True
html_theme = "shibuya"
html_title = f"Torch Kirigami {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
