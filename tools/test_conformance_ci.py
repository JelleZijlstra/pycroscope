from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="conformance tooling tests require Python 3.11+"
)

if sys.version_info >= (3, 11):
    from tools.conformance_ci import (
        diff_expected_errors,
        get_expected_errors,
        parse_pycroscope_concise_errors,
    )


def test_get_expected_errors(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        x: int = "x"  # E
        y: int = "y"  # E? # E?
        @final  # E[tag]
        def f() -> None: ...  # E[tag]
        # x: int = "x"  # E
        """
    ).strip()
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    try:
        line_markers, groups = get_expected_errors(path)
    finally:
        path.unlink()

    assert line_markers == {1: (1, 0), 2: (0, 2)}
    assert groups == {"tag": ([3, 4], False)}


def test_get_expected_errors_rejects_single_tagged_line(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text('x: int = "x"  # E[one]\n', encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="only appears on a single line"):
            get_expected_errors(path)
    finally:
        path.unlink()


def test_diff_expected_errors_and_parse_output(tmp_path: Path) -> None:
    test_case = tmp_path / "foo.py"
    test_case.write_text(
        textwrap.dedent(
            """
            a: int = "a"  # E
            @final  # E[tag]
            def f() -> None: ...  # E[tag]
            b: int = "b"  # E?
            """
        ).strip(),
        encoding="utf-8",
    )
    output_lines = [
        f"{test_case}:1:1: Bad assignment [incompatible_assignment]",
        f"{test_case}:2:1: Bad final [bad_final_decorator]",
        f"{test_case}:4:1: Revealed type is \"Literal['x']\" [reveal_type]",
    ]
    errors = parse_pycroscope_concise_errors(output_lines)
    assert diff_expected_errors(test_case, errors[test_case.name]) == []

    unexpected_lines = [
        f"{test_case}:1:1: Bad assignment [incompatible_assignment]",
        f"{test_case}:4:1: Unexpected [incompatible_assignment]",
    ]
    unexpected_errors = parse_pycroscope_concise_errors(unexpected_lines)
    differences = diff_expected_errors(test_case, unexpected_errors[test_case.name])
    assert differences == ["Lines 2, 3: Expected error (tag 'tag')"]
