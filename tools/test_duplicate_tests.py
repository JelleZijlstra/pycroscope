from __future__ import annotations

import ast
import re
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tooling tests require Python 3.11+"
)


def _iter_test_functions(
    tree: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    tests: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and node.name.startswith("test"):
            tests.append(node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    child.name.startswith("test")
                ):
                    tests.append(child)
    return tests


def _normalize_test_source(source: str) -> str:
    normalized = textwrap.dedent(source).strip()
    normalized = re.sub(
        r"^(async\s+def|def)\s+test[^(]*\(",
        lambda match: f"{match.group(1)} TEST_NAME(",
        normalized,
        count=1,
        flags=re.MULTILINE,
    )
    return "\n".join(line.rstrip() for line in normalized.splitlines())


def _get_test_source(
    lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef
) -> str | None:
    if node.end_lineno is None:
        return None
    start_lineno = node.lineno
    if node.decorator_list:
        start_lineno = min(decorator.lineno for decorator in node.decorator_list)
    return "\n".join(lines[start_lineno - 1 : node.end_lineno])


def _find_duplicate_tests(repo_root: Path) -> dict[str, list[str]]:
    files = sorted((repo_root / "pycroscope").glob("test_*.py")) + sorted(
        (repo_root / "tools").glob("test_*.py")
    )
    by_source: defaultdict[str, list[str]] = defaultdict(list)

    for path in files:
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        tree = ast.parse(src, filename=str(path))
        for node in _iter_test_functions(tree):
            segment = _get_test_source(lines, node)
            if segment is None:
                continue
            by_source[_normalize_test_source(segment)].append(
                f"{path.relative_to(repo_root)}:{node.lineno}"
            )

    return {
        source: locations
        for source, locations in by_source.items()
        if len(locations) > 1
    }


def test_no_duplicate_test_bodies() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    duplicates = _find_duplicate_tests(repo_root)
    if not duplicates:
        return

    groups = [
        "Duplicate test bodies found:\n"
        + "\n".join(f"  {location}" for location in sorted(locations))
        for locations in duplicates.values()
    ]
    pytest.fail("\n\n".join(sorted(groups)))
