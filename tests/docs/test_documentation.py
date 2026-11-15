"""Build and inspect the published documentation without installing PyTorch."""

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

pytest.importorskip("sphinx", reason="Install the docs dependency group")
pytest.importorskip("myst_parser", reason="Install the docs dependency group")
pytest.importorskip("autodoc2", reason="Install the docs dependency group")
pytest.importorskip("shibuya", reason="Install the docs dependency group")
pytest.importorskip("sphinxcontrib.mermaid", reason="Install the docs dependency group")

from bs4 import BeautifulSoup
from sphinx.util.inventory import InventoryFile

yaml = pytest.importorskip("yaml", reason="Install the docs dependency group")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def built_docs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sphinx")
    shutil.copytree(
        ROOT / "docs",
        root / "docs",
        ignore=shutil.ignore_patterns("_build", "apidocs", "dev-notes", "__pycache__"),
    )
    shutil.copytree(
        ROOT / "torch_kirigami",
        root / "torch_kirigami",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    package = root / "torch_kirigami/__init__.py"
    package_text = package.read_text(encoding="utf-8")
    doc_end = package_text.find('"""', 3)
    assert package_text.startswith('"""') and doc_end >= 0
    package.write_text(
        package_text[:doc_end]
        + "\n\nNotes:\n    Package metadata is analyzed without importing PyTorch.\n"
        + package_text[doc_end:],
        encoding="utf-8",
    )
    (root / "torch_kirigami" / "doc_fixture.py").write_text(
        textwrap.dedent(
            '''\
            from dataclasses import dataclass

            class MarkdownExample:
                """A **bold** label and a [guide](https://example.com/guide).

                Use `item` in a Markdown table:

                | Input | Output |
                | --- | --- |
                | one | two |

                ```python
                result = "example"
                ```

                :::{note}
                An explicit MyST note.
                :::

                    indented_example = True

                Args:
                    item: A `str` documented in the [API](https://example.com/api).

                Attributes:
                    value: A **stored** result.

                Returns:
                    The `item` after processing.

                Raises:
                    ValueError: If `item` is invalid.

                Notes:
                    Read [more](https://example.com/notes) before calling.

                    The second paragraph remains in the notes section.

                Note:
                    A [short note](https://example.com/note).

                Warning:
                    Use **care** with this operation.

                    ```python
                    checked = True
                    ```
                """

            def fenced_parameter(value):
                """Describe a fenced block within a parameter.

                Args:
                    value: Follow this example:

                        ```python
                        value = 1
                        ```
                """

            @dataclass
            class AttributeExample:
                """A dataclass with an explicitly documented field.

                Attributes:
                    size: The number of elements.
                """

                size: int
                """The field's own description."""

            '''
        ),
        encoding="utf-8",
    )
    notes = root / "docs" / "dev-notes"
    notes.mkdir()
    (notes / "excluded.md").write_text(
        "# Private note\n\nDO_NOT_PUBLISH_THIS_NOTE\n", encoding="utf-8"
    )
    api_index = root / "docs" / "api" / "index.md"
    with api_index.open("a", encoding="utf-8") as file:
        file.write(
            "\nPublic: {py:class}`torch_kirigami.pruning.Pruner`.\n"
            "\nDefinition: {py:class}`torch_kirigami.pruning.pruner.Pruner`.\n"
        )
    output = root / "html"
    env = os.environ.copy()
    if os.name == "posix":
        env["LC_ALL"] = "C"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(root / "docs"),
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return output


def test_google_docstrings_render_parameters_attributes_and_examples(built_docs):
    package = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.html").read_text(encoding="utf-8"),
        "html.parser",
    )
    assert package.find("p", class_="rubric", string="Notes") is not None
    assert "Package metadata is analyzed without importing PyTorch" in package.get_text(" ")
    pruning = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.pruning.pruner.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    pruner = pruning.find(id="torch_kirigami.pruning.pruner.Pruner")
    assert pruner is not None
    pruner = pruner.find_parent("dl")
    assert "Parameters" in pruner.get_text(" ")
    assert "preserve_io" in pruner.get_text(" ")
    assert pruner.select_one("code") is not None

    sparsity = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.sparsity.regularizers.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    lasso = sparsity.find(id="torch_kirigami.sparsity.regularizers.GroupLasso")
    assert lasso is not None
    lasso = lasso.find_parent("dl")
    assert "Variables" in lasso.get_text(" ")
    assert "Examples" in lasso.get_text(" ")
    example = lasso.select_one("div.highlight-python pre")
    assert example is not None
    assert "unique_groups((group, group))" in example.get_text()
    assert lasso.select_one("code.literal") is not None
    assert ".. rubric::" not in lasso.get_text(" ")
    constraints = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.dep_graph.contracts.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    balanced = constraints.find(id="torch_kirigami.dep_graph.contracts.Balanced")
    assert balanced is not None
    balanced = balanced.find_parent("dl")
    assert balanced.find("p", class_="rubric", string="Notes") is not None
    assert "Unequal counts remain unresolved" in balanced.get_text(" ")
    for page in (built_docs / "apidocs").rglob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert "<p>Args:" not in html
        assert "<p>Attributes:" not in html
        assert ".. rubric::" not in html


def test_google_docstrings_can_contain_markdown(built_docs):
    page = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.doc_fixture.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    content = page.find(id="torch_kirigami.doc_fixture.MarkdownExample")
    assert content is not None
    content = content.find_parent("dl")
    assert content.find("strong", string="bold") is not None
    assert content.find("a", href="https://example.com/guide").get_text(strip=True) == "guide"
    assert content.find("a", href="https://example.com/api").get_text(strip=True) == "API"
    assert content.find("a", href="https://example.com/notes").get_text(strip=True) == "more"
    notes_heading = content.find("p", class_="rubric", string="Notes")
    assert notes_heading is not None
    assert "second paragraph remains" in notes_heading.find_next_sibling("p").find_next_sibling(
        "p"
    ).get_text(" ")
    assert content.find("a", href="https://example.com/note").get_text(strip=True) == "short note"
    assert content.find("code", string="item") is not None
    assert content.select_one("table") is not None
    assert content.select_one("div.highlight-python pre").get_text().strip() == 'result = "example"'
    code = [block.get_text() for block in content.select("pre")]
    assert any("indented_example = True" in c for c in code)
    assert "An explicit MyST note." in content.get_text(" ")
    assert content.select_one("div.admonition.note") is not None
    assert content.select_one("div.admonition.warning strong").get_text(strip=True) == "care"
    warning_code = content.select_one("div.admonition.warning div.highlight-python pre")
    assert warning_code is not None
    assert "checked = True" in warning_code.get_text()
    assert "Parameters" in content.get_text(" ")
    assert "Variables" in content.get_text(" ")
    assert "Returns" in content.get_text(" ")
    assert "Raises" in content.get_text(" ")
    assert "Notes" in content.get_text(" ")
    assert content.find("p", string=re.compile(r"^\.\. rubric::")) is None
    parameter = page.select_one("dl.py.function")
    assert parameter is not None
    assert "value = 1" in parameter.select_one("pre").get_text()


def test_dataclass_attributes_do_not_produce_duplicate_descriptions(built_docs):
    page = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.doc_fixture.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    attribute_example = page.find(id="torch_kirigami.doc_fixture.AttributeExample")
    assert attribute_example is not None
    attribute_example = attribute_example.find_parent("dl")
    assert "The number of elements" in attribute_example.get_text(" ")
    assert len(attribute_example.select("dl.py.attribute")) == 1


def test_public_class_and_method_indices_use_their_real_names(built_docs):
    page = BeautifulSoup(
        (built_docs / "apidocs/torch_kirigami/torch_kirigami.pruning.pruner.html").read_text(
            encoding="utf-8"
        ),
        "html.parser",
    )
    for name in (
        "torch_kirigami.pruning.pruner.Pruner",
        "torch_kirigami.pruning.pruner.Pruner.apply",
    ):
        assert page.find(id=name) is not None
    index = (built_docs / "genindex.html").read_text(encoding="utf-8")
    assert "torch_kirigami.pruning.pruner.torch_kirigami" not in index


def test_public_and_defining_paths_resolve_in_documentation(built_docs):
    page = BeautifulSoup((built_docs / "api/index.html").read_text(encoding="utf-8"), "html.parser")
    for name in ("torch_kirigami.pruning.Pruner", "torch_kirigami.pruning.pruner.Pruner"):
        link = page.find("a", title=name)
        assert link is not None
        assert urlsplit(link["href"]).fragment == name


def test_public_dataclass_fields_have_link_targets(built_docs):
    with (built_docs / "objects.inv").open("rb") as file:
        inventory = InventoryFile.load(file, "", os.path.join)
    attributes = inventory["py:attribute"]
    for name in ("torch_kirigami.ArgumentRef.name", "torch_kirigami.pruning.Candidate.key"):
        target = urlsplit(attributes[name].uri)
        page = BeautifulSoup(
            (built_docs / unquote(target.path)).read_text(encoding="utf-8"), "html.parser"
        )
        assert page.find(id=unquote(target.fragment)) is not None


def test_all_public_exports_have_reference_entries(built_docs):
    exported = {}
    for module in ("torch_kirigami", "torch_kirigami.pruning", "torch_kirigami.sparsity"):
        path = ROOT.joinpath(*module.split("."), "__init__.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            )
        )
        imports = {
            alias.asname or alias.name: (
                f"{importlib.util.resolve_name('.' * node.level + (node.module or ''), module)}"
                f".{alias.name}"
            )
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        for name in ast.literal_eval(assignment.value):
            exported[f"{module}.{name}"] = imports.get(name, f"{module}.{name}")

    assert exported
    with (built_docs / "objects.inv").open("rb") as file:
        inventory = InventoryFile.load(file, "", os.path.join)
    documented = {
        name: item
        for kind, objects in inventory.items()
        if kind.startswith("py:")
        for name, item in objects.items()
    }
    missing = {
        public: source
        for public, source in exported.items()
        if public not in documented or source not in documented
    }
    assert not missing
    pages = {}
    for name in set(exported) | set(exported.values()):
        target = urlsplit(documented[name].uri)
        path = built_docs / unquote(target.path)
        assert path.is_file(), name
        if path not in pages:
            pages[path] = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        assert pages[path].find(id=unquote(target.fragment)) is not None, name


def test_notes_are_excluded_and_mermaid_is_pinned(built_docs):
    assert not list(built_docs.rglob("excluded.html"))
    assert "DO_NOT_PUBLISH_THIS_NOTE" not in (built_docs / "searchindex.js").read_text(
        encoding="utf-8"
    )
    diagram = (built_docs / "reference/dependency-graph.html").read_text(encoding="utf-8")
    assert 'class="mermaid"' in diagram
    assert "block-beta" in diagram
    assert "mermaid@12.3.0/dist/mermaid.esm.min.mjs" in diagram


def test_read_the_docs_configuration_uses_the_checked_sources():
    config = yaml.safe_load((ROOT / ".readthedocs.yaml").read_text(encoding="utf-8"))
    assert config["version"] == 2
    assert config["build"]["tools"]["python"] == "3.12"
    assert config["sphinx"] == {"configuration": "docs/conf.py", "fail_on_warning": True}
    assert config["build"]["jobs"]["install"] == [
        '"$READTHEDOCS_VIRTUALENV_PATH/bin/python" -m pip install uv==0.9.11',
        '"$READTHEDOCS_VIRTUALENV_PATH/bin/python" -m uv pip install '
        '--python "$READTHEDOCS_VIRTUALENV_PATH/bin/python" --group docs',
    ]
    assert (ROOT / config["sphinx"]["configuration"]).is_file()


def test_internal_html_links_and_anchors_resolve(built_docs):
    pages = {
        path: BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        for path in built_docs.rglob("*.html")
    }
    failures = []
    for source, soup in pages.items():
        for link in soup.select("a[href]"):
            url = urlsplit(link["href"])
            if url.scheme or url.netloc or (not url.path and not url.fragment):
                continue
            path = unquote(url.path)
            target = (source.parent / path).resolve() if path else source
            if target.is_dir():
                target /= "index.html"
            if not target.exists():
                failures.append((source.relative_to(built_docs), link["href"]))
                continue
            if (
                url.fragment
                and target in pages
                and pages[target].find(id=unquote(url.fragment)) is None
            ):
                failures.append((source.relative_to(built_docs), link["href"]))
    assert not failures, failures
