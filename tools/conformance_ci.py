import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib

EXPECTED_ERROR_RE = re.compile(r"# E\??(?=:|$| )")
TAGGED_ERROR_RE = re.compile(r"# E\[([^\]]+)\]")
CONCISE_OUTPUT_RE = re.compile(r"^(.+?):(\d+)(?::\d+)?:\s(.*)$")


def load_known_failures(path: Path) -> set[str]:
    failures = set()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(r"\s", stripped):
            raise ValueError(
                f"Invalid known failure name at {path}:{lineno}: {stripped!r}"
            )
        failures.add(stripped)
    return failures


def load_test_group_names(typing_repo: Path) -> set[str]:
    test_groups_file = typing_repo / "conformance" / "src" / "test_groups.toml"
    data = tomllib.loads(test_groups_file.read_text(encoding="utf-8"))
    return set(data)


def get_test_cases(typing_repo: Path) -> list[Path]:
    group_names = load_test_group_names(typing_repo)
    tests_dir = typing_repo / "conformance" / "tests"
    cases = [
        path
        for path in sorted([*tests_dir.glob("*.py"), *tests_dir.glob("*.pyi")])
        if path.name.split("_")[0] in group_names
    ]
    return cases


def get_expected_errors(
    test_case: Path,
) -> tuple[dict[int, tuple[int, int]], dict[str, tuple[list[int], bool]]]:
    """Return expected errors in the same format as typing/conformance/src/main.py."""
    lines = test_case.read_text(encoding="utf-8").splitlines()
    output: dict[int, tuple[int, int]] = {}
    groups: dict[str, tuple[list[int], bool]] = {}

    for i, line in enumerate(lines, start=1):
        line_without_comment, *_ = line.split("#")
        if not line_without_comment.strip():
            continue

        required = 0
        optional = 0
        for match in EXPECTED_ERROR_RE.finditer(line):
            if match.group() == "# E":
                required += 1
            else:
                optional += 1
        if required or optional:
            output[i] = (required, optional)

        for match in TAGGED_ERROR_RE.finditer(line):
            tag = match.group(1)
            if tag.endswith("+"):
                allow_multiple = True
                tag = tag[:-1]
            else:
                allow_multiple = False
            if tag not in groups:
                groups[tag] = ([i], allow_multiple)
            else:
                existing_lines, existing_allow_multiple = groups[tag]
                if existing_allow_multiple != allow_multiple:
                    raise ValueError(
                        f"Error group {tag} has inconsistent allow_multiple value in {test_case}"
                    )
                existing_lines.append(i)

    for group, (linenos, _) in groups.items():
        if len(linenos) == 1:
            raise ValueError(
                f"Error group {group} only appears on a single line in {test_case}"
            )

    return output, groups


def parse_pycroscope_concise_errors(
    lines: Sequence[str],
) -> dict[str, dict[int, list[str]]]:
    errors_by_file: dict[str, dict[int, list[str]]] = {}
    for line in lines:
        if not line.strip():
            continue
        if "[reveal_type]" in line or "Revealed type is " in line:
            continue
        match = CONCISE_OUTPUT_RE.match(line)
        if match is None:
            continue
        file_name = Path(match.group(1)).name
        lineno = int(match.group(2))
        file_errors = errors_by_file.setdefault(file_name, {})
        file_errors.setdefault(lineno, []).append(line)
    return errors_by_file


def diff_expected_errors(test_case: Path, errors: dict[int, list[str]]) -> list[str]:
    expected_errors, error_groups = get_expected_errors(test_case)
    differences: list[str] = []

    for expected_lineno, (expected_count, _) in expected_errors.items():
        if expected_lineno not in errors and expected_count > 0:
            differences.append(
                f"Line {expected_lineno}: Expected {expected_count} errors"
            )

    linenos_used_by_groups: set[int] = set()
    for group, (linenos, allow_multiple) in error_groups.items():
        num_errors = sum(1 for lineno in linenos if lineno in errors)
        if num_errors == 0:
            joined_linenos = ", ".join(map(str, linenos))
            differences.append(
                f"Lines {joined_linenos}: Expected error (tag {group!r})"
            )
        elif num_errors == 1 or allow_multiple:
            linenos_used_by_groups.update(linenos)
        else:
            joined_linenos = ", ".join(map(str, linenos))
            differences.append(
                f"Lines {joined_linenos}: Expected exactly one error (tag {group!r})"
            )

    for actual_lineno, actual_errors in sorted(errors.items()):
        if (
            actual_lineno not in expected_errors
            and actual_lineno not in linenos_used_by_groups
        ):
            differences.append(
                f"Line {actual_lineno}: Unexpected errors {actual_errors}"
            )

    return differences


def run_pycroscope(tests_dir: Path) -> dict[str, dict[int, list[str]]]:
    command = [
        sys.executable,
        "-m",
        "pycroscope",
        ".",
        "--output-format",
        "concise",
        "--disable",
        "import_failed",
        "--disable",
        "unused_variable",
        "--disable",
        "unused_assignment",
    ]
    proc = subprocess.run(
        command,
        cwd=tests_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    output_lines = [*proc.stdout.splitlines(), *proc.stderr.splitlines()]
    if proc.returncode not in (0, 1):
        summarized_output = "\n".join(output_lines[-40:])
        raise RuntimeError(
            f"pycroscope failed with exit code {proc.returncode}:\n{summarized_output}"
        )
    return parse_pycroscope_concise_errors(output_lines)


def _default_known_failures_path() -> Path:
    return Path(__file__).resolve().parent / "conformance_known_failures.txt"


def _default_typing_repo() -> Path:
    return Path(os.environ.get("TYPING_REPO", "~/py/typing")).expanduser()


def _validate_case_names(test_cases: list[Path]) -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    duplicates: list[str] = []
    for case in test_cases:
        name = case.stem
        if name in by_name:
            duplicates.append(name)
        by_name[name] = case
    if duplicates:
        dup_list = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate conformance case stems: {dup_list}")
    return by_name


def check_conformance(typing_repo: Path, known_failures: set[str]) -> int:
    test_cases = get_test_cases(typing_repo)
    tests_dir = typing_repo / "conformance" / "tests"
    cases_by_name = _validate_case_names(test_cases)

    missing_cases = sorted(known_failures - set(cases_by_name))
    if missing_cases:
        print("Known-failure list includes cases that do not exist:")
        for case in missing_cases:
            print(f"  - {case}")
        return 1

    errors_by_file = run_pycroscope(tests_dir)

    actual_failures: set[str] = set()
    differences_by_case: dict[str, list[str]] = {}
    for case_name, test_case in sorted(cases_by_name.items()):
        errors = errors_by_file.get(test_case.name, {})
        differences = diff_expected_errors(test_case, errors)
        if differences:
            actual_failures.add(case_name)
            differences_by_case[case_name] = differences

    unexpected_passes = sorted(known_failures - actual_failures)
    unexpected_failures = sorted(actual_failures - known_failures)
    total_cases = len(cases_by_name)
    total_passes = total_cases - len(actual_failures)
    print(
        f"Conformance summary: {total_cases} total, {total_passes} pass, "
        f"{len(actual_failures)} fail."
    )

    if not unexpected_passes and not unexpected_failures:
        print("Conformance outcomes match the known-failure list.")
        return 0

    if unexpected_passes:
        print("Known failing cases that now pass:")
        for case in unexpected_passes:
            print(f"  - {case}")

    if unexpected_failures:
        print("Cases that now fail but are not in known-failures:")
        for case in unexpected_failures:
            print(f"  - {case}")
            for diff in differences_by_case.get(case, []):
                print(f"    {diff}")

    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pycroscope on typing conformance tests and validate outcomes against "
            "a known-failing case list."
        )
    )
    parser.add_argument(
        "--typing-repo",
        type=Path,
        default=_default_typing_repo(),
        help="Path to the python/typing checkout (default: ~/py/typing or $TYPING_REPO).",
    )
    parser.add_argument(
        "--known-failures",
        type=Path,
        default=_default_known_failures_path(),
        help="Path to newline-delimited known failing case names.",
    )
    args = parser.parse_args(argv)

    typing_repo = args.typing_repo.resolve()
    known_failures = load_known_failures(args.known_failures.resolve())
    return check_conformance(typing_repo, known_failures)


if __name__ == "__main__":
    raise SystemExit(main())
