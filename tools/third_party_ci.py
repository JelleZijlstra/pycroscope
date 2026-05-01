from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import run_third_party

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # static analysis: ignore[import_failed]


@dataclass(frozen=True)
class ThirdPartyCheck:
    name: str
    repo_url: str
    targets: tuple[str, ...]
    ref: str | None = None
    config_file: str = "pyproject.toml"
    run_from: str = "."
    install_path: str = "."
    pythonpath_entries: tuple[str, ...] = ()
    pycroscope_args: tuple[str, ...] = ()
    install: bool = True
    expected_to_fail: bool = False


THIRD_PARTY_CHECKS: Final[tuple[ThirdPartyCheck, ...]] = (
    ThirdPartyCheck(
        name="stubdefaulter",
        repo_url="https://github.com/JelleZijlstra/stubdefaulter.git",
        targets=("stubdefaulter/",),
    ),
    ThirdPartyCheck(
        name="taxonomy",
        repo_url="https://github.com/JelleZijlstra/taxonomy.git",
        targets=("taxonomy/",),
    ),
)

LOCAL_CONFIG_PATH: Final = Path(__file__).with_name("third_party_ci_local.toml")


def _checks_by_name() -> dict[str, ThirdPartyCheck]:
    by_name: dict[str, ThirdPartyCheck] = {}
    duplicates: list[str] = []
    for check in THIRD_PARTY_CHECKS:
        if check.name in by_name:
            duplicates.append(check.name)
        by_name[check.name] = check
    if duplicates:
        joined = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate third-party check names: {joined}")
    return by_name


def select_checks(names: list[str]) -> list[ThirdPartyCheck]:
    checks_by_name = _checks_by_name()
    if not names:
        return list(THIRD_PARTY_CHECKS)

    selected: list[ThirdPartyCheck] = []
    missing: list[str] = []
    for name in names:
        check = checks_by_name.get(name)
        if check is None:
            missing.append(name)
            continue
        selected.append(check)
    if missing:
        joined = ", ".join(sorted(set(missing)))
        raise ValueError(f"Unknown third-party checks: {joined}")
    return selected


def parse_local_overrides(values: list[str], known_checks: set[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        name, sep, path = value.partition(":")
        if not sep or not name or not path:
            raise ValueError(f"Invalid --local value {value!r}; expected NAME:PATH.")
        if name not in known_checks:
            raise ValueError(f"Unknown third-party check in --local: {name}")
        overrides[name] = path
    return overrides


def read_local_config(
    path: Path | None = None, known_checks: set[str] | None = None
) -> dict[str, str]:
    if path is None:
        path = LOCAL_CONFIG_PATH
    if not path.exists():
        return {}

    known_checks = known_checks or {check.name for check in THIRD_PARTY_CHECKS}
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)

    local = data.get("local", {})
    if not isinstance(local, dict):
        raise ValueError(f"{path}: [local] must be a table.")

    overrides: dict[str, str] = {}
    for name, value in local.items():
        if name not in known_checks:
            raise ValueError(f"{path}: unknown third-party check in [local]: {name}")
        if not isinstance(value, str):
            raise ValueError(f"{path}: [local].{name} must be a string path.")
        overrides[name] = value
    return overrides


def build_run_third_party_argv(
    check: ThirdPartyCheck, local_overrides: dict[str, str], *, verbose: bool = False
) -> list[str]:
    repo_ref = local_overrides.get(check.name, check.repo_url)
    argv = [repo_ref, *check.targets, "--config-file", check.config_file]
    if check.ref is not None:
        argv.extend(["--ref", check.ref])
    if check.run_from != ".":
        argv.extend(["--run-from", check.run_from])
    if check.install_path != ".":
        argv.extend(["--install-path", check.install_path])
    for entry in check.pythonpath_entries:
        argv.extend(["--pythonpath", entry])
    for arg in check.pycroscope_args:
        argv.append(f"--pycroscope-arg={arg}")
    if check.install:
        argv.append("--install")
    if verbose:
        argv.append("--verbose")
    return argv


def run_check(
    check: ThirdPartyCheck, local_overrides: dict[str, str], *, verbose: bool = False
) -> int:
    print(f"Running pycroscope on {check.name}...")
    exit_code = run_third_party.main(
        build_run_third_party_argv(check, local_overrides, verbose=verbose)
    )
    if exit_code == 0:
        if check.expected_to_fail:
            print(f"{check.name}: unexpectedly clean")
            return 1
        print(f"{check.name}: clean")
    else:
        if check.expected_to_fail:
            print(f"{check.name}: expected failure (exit code {exit_code})")
            return 0
        print(f"{check.name}: failed with exit code {exit_code}")
        return exit_code
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pycroscope against configured third-party repositories and fail "
            "if any are not clean."
        )
    )
    parser.add_argument(
        "checks",
        nargs="*",
        help="Optional configured check names to run. Defaults to all configured checks.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured third-party checks and exit.",
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        help=(
            "Use a local checkout for a configured check, in NAME:PATH form. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show dependency installation output from each third-party check.",
    )
    args = parser.parse_args(argv)

    try:
        selected_checks = select_checks(args.checks)
        known_checks = {check.name for check in THIRD_PARTY_CHECKS}
        local_overrides = {
            **read_local_config(known_checks=known_checks),
            **parse_local_overrides(args.local, known_checks),
        }
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.list:
        for check in selected_checks:
            print(check.name)
        return 0

    failures: list[str] = []
    for check in selected_checks:
        if run_check(check, local_overrides, verbose=args.verbose) != 0:
            failures.append(check.name)

    if failures:
        joined = ", ".join(failures)
        print(f"Third-party checks failed: {joined}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
