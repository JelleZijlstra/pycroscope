from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import run_third_party

PUBLIC_TAXONOMY_REPO_URL = "https://github.com/JelleZijlstra/taxonomy.git"


def _build_run_third_party_argv(args: argparse.Namespace) -> list[str]:
    repo = str(args.local_path) if args.local_path is not None else args.taxonomy_url
    translated = [repo, "taxonomy/", "--config-file", "pyproject.toml"]
    if args.taxonomy_ref is not None:
        translated.extend(["--ref", args.taxonomy_ref])
    if args.install:
        translated.append("--install")
    elif args.skip_install:
        translated.append("--skip-install")
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    if sys.version_info < (3, 14):
        print(
            "tools/taxonomy_ci.py requires Python 3.14+ because taxonomy uses "
            "Python 3.14 syntax and its own CI runs pycroscope under Python 3.14."
        )
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Run pycroscope against taxonomy using the same config-file and target "
            "path that taxonomy's own CI uses."
        )
    )
    parser.add_argument(
        "--local-path",
        type=Path,
        help=(
            "Path to a local taxonomy checkout. If omitted, the script clones the "
            "public taxonomy repository."
        ),
    )
    parser.add_argument(
        "--taxonomy-url",
        default=PUBLIC_TAXONOMY_REPO_URL,
        help="Git URL to clone when no local checkout is provided.",
    )
    parser.add_argument(
        "--taxonomy-ref", help="Optional git ref to check out after cloning taxonomy."
    )
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--install",
        action="store_true",
        help="Install taxonomy into the current interpreter before running pycroscope.",
    )
    install_group.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install taxonomy before running, even for a freshly cloned checkout.",
    )
    args = parser.parse_args(argv)
    return run_third_party.main(_build_run_third_party_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
