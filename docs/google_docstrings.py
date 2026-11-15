"""Render Google-style sections and Markdown within autodoc2 docstrings."""

import re
import textwrap

from docutils import nodes
from myst_parser.parsers.sphinx_ import MystParser
from sphinx.ext.napoleon.docstring import GoogleDocstring

_DIRECTIVE = re.compile(
    r"^([ \t]*)\.\. (rubric|admonition|attention|caution|danger|error|hint|"
    r"important|note|seealso|tip|todo|warning)::(?:[ \t]*(.*))?$"
)


def _convert_napoleon_directives(content: str, delimiter: str) -> str:
    """Convert Napoleon's reStructuredText directives to MyST directives."""
    lines = content.splitlines()
    converted = []
    index = 0
    while index < len(lines):
        match = _DIRECTIVE.match(lines[index])
        if match is None:
            converted.append(lines[index])
            index += 1
            continue

        indent, name, argument = match.groups()
        argument = argument or ""
        index += 1
        if name == "rubric":
            converted.extend((f"{indent}{delimiter}{{rubric}} {argument}", f"{indent}{delimiter}"))
            continue

        body = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
                break
            body.append(line)
            index += 1
        body_text = textwrap.dedent("\n".join(body)).strip("\n")
        if name in {"admonition", "todo"}:
            title = argument if name == "admonition" else "Todo"
            directive = f"{{admonition}} {title}"
            if name == "todo" and argument:
                body_text = "\n".join(filter(None, (argument, body_text)))
        else:
            directive = f"{{{name}}}"
            body_text = "\n".join(filter(None, (argument, body_text)))
        converted.append(f"{indent}{delimiter}{directive}")
        converted.extend(indent + line for line in body_text.splitlines())
        converted.extend((f"{indent}{delimiter}", ""))
    return "\n".join(converted)


class GoogleDocstringParser(MystParser):
    """Parse Napoleon's Google-style sections with the MyST parser."""

    def parse(self, input_string: str, document: nodes.document) -> None:
        """Parse a docstring in the current Sphinx document context."""
        content = str(GoogleDocstring(input_string, document.settings.env.config))
        # Keep generated directives longer than any nested MyST directive.
        longest_delimiter = max(
            (len(match) for match in re.findall(r"(?m)^\s*(:{3,})", input_string)), default=2
        )
        content = _convert_napoleon_directives(content, ":" * (longest_delimiter + 1))
        super().parse(content, document)


# Autodoc2 loads the module's `Parser` attribute as its custom parser.
Parser = GoogleDocstringParser
