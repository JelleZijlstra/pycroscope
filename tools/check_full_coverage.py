"""Check that pinned fully covered files remain fully covered."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _is_package_module(path: str) -> bool:
    return (
        path.startswith("pycroscope/")
        and not path.startswith("pycroscope/test_")
        and path not in {"pycroscope/tests.py", "pycroscope/asynq_tests.py"}
    )


def _load_pinned_files(path: Path) -> list[str]:
    pinned = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pinned.append(line)
    return pinned


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: python tools/check_full_coverage.py COVERAGE_JSON PINNED_FILES",
            file=sys.stderr,
        )
        return 2

    coverage_path = Path(sys.argv[1])
    pinned_path = Path(sys.argv[2])
    data = json.loads(coverage_path.read_text())
    pinned_files = _load_pinned_files(pinned_path)

    failures = []
    files = data["files"]
    for relpath in pinned_files:
        info = files.get(relpath)
        if info is None:
            failures.append(f"{relpath}: missing from coverage report")
            continue
        if info["summary"]["missing_lines"] != 0:
            missing = ", ".join(str(line) for line in info["missing_lines"])
            failures.append(f"{relpath}: no longer fully covered (missing: {missing})")

    if failures:
        print("Pinned fully covered files lost full coverage:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    current_full = [
        path
        for path, info in sorted(files.items())
        if _is_package_module(path) and info["summary"]["missing_lines"] == 0
    ]
    new_full = [path for path in current_full if path not in pinned_files]
    if new_full:
        print("Additional fully covered files you may want to pin:")
        for path in new_full:
            print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
