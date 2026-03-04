from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="conformance tooling tests require Python 3.11+"
)

if sys.version_info >= (3, 11):
    from tools.conformance_ci import (
        check_conformance,
        diff_expected_errors,
        get_expected_errors,
        parse_pycroscope_concise_errors,
        parse_pycroscope_internal_error_cases,
        run_pycroscope,
    )


def test_get_expected_errors(tmp_path: Path) -> None:
    source = textwrap.dedent("""
        x: int = "x"  # E
        y: int = "y"  # E? # E?
        @final  # E[tag]
        def f() -> None: ...  # E[tag]
        # x: int = "x"  # E
        """).strip()
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
        textwrap.dedent("""
            a: int = "a"  # E
            @final  # E[tag]
            def f() -> None: ...  # E[tag]
            b: int = "b"  # E?
            """).strip(),
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


def test_parse_pycroscope_internal_error_cases(tmp_path: Path) -> None:
    test_case = tmp_path / "dataclasses_order.py"
    output_lines = [
        f"{test_case}:12:5: Internal error; please report this as a bug [internal_error]",
        f"{test_case}:14:1: Some other error [incompatible_assignment]",
    ]
    assert parse_pycroscope_internal_error_cases(output_lines) == {"dataclasses_order"}


def test_parse_pycroscope_internal_error_cases_traceback_tail(tmp_path: Path) -> None:
    test_case = tmp_path / "dataclasses_order.py"
    output_lines = [
        f"{test_case}:53:3: Traceback (most recent call last):",
        '  File "/repo/pycroscope/name_check_visitor.py", line 1518, in visit',
        "Internal error: NotAGradualType('...') [internal_error]",
    ]
    assert parse_pycroscope_internal_error_cases(output_lines) == {"dataclasses_order"}


def test_run_pycroscope_disables_must_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    seen_command: list[str] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command
        seen_command = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.conformance_ci.subprocess.run", fake_run)
    errors, internal_error_cases = run_pycroscope(tests_dir)

    assert errors == {}
    assert internal_error_cases == set()
    must_use_index = seen_command.index("must_use")
    assert seen_command[must_use_index - 1] == "--disable"


def test_check_conformance_fails_on_internal_error_in_known_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    typing_repo = tmp_path / "typing"
    case_path = typing_repo / "conformance" / "tests" / "dataclasses_order.py"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "tools.conformance_ci.get_test_cases", lambda _typing_repo: [case_path]
    )
    monkeypatch.setattr(
        "tools.conformance_ci.run_pycroscope",
        lambda _tests_dir: (
            {
                case_path.name: {
                    1: [f"{case_path}:1:1: Unexpected [incompatible_assignment]"]
                }
            },
            {"dataclasses_order"},
        ),
    )

    result = check_conformance(typing_repo, {"dataclasses_order"})
    assert result == 1


def test_check_conformance_passes_when_outcomes_match_known_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    typing_repo = tmp_path / "typing"
    case_path = typing_repo / "conformance" / "tests" / "dataclasses_order.py"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        "tools.conformance_ci.get_test_cases", lambda _typing_repo: [case_path]
    )
    monkeypatch.setattr(
        "tools.conformance_ci.run_pycroscope",
        lambda _tests_dir: (
            {
                case_path.name: {
                    1: [f"{case_path}:1:1: Unexpected [incompatible_assignment]"]
                }
            },
            set(),
        ),
    )

    result = check_conformance(typing_repo, {"dataclasses_order"})
    assert result == 0
