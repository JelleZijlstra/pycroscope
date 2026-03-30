from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

PUBLIC_TAXONOMY_REPO_URL = "https://github.com/JelleZijlstra/taxonomy.git"


def _validate_taxonomy_repo(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Taxonomy checkout does not exist: {resolved}")
    if not (resolved / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Taxonomy checkout is missing pyproject.toml: {resolved}"
        )
    if not (resolved / "taxonomy").is_dir():
        raise FileNotFoundError(
            f"Taxonomy checkout is missing the taxonomy package directory: {resolved}"
        )
    return resolved


def install_taxonomy(taxonomy_repo: Path) -> None:
    if importlib.util.find_spec("pip") is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(taxonomy_repo)],
            check=True,
        )
        return

    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "-e", str(taxonomy_repo)],
        check=True,
    )


@contextmanager
def clone_taxonomy(repo_url: str, ref: str | None = None) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="taxonomy-") as temp_dir:
        clone_dir = Path(temp_dir) / "taxonomy"
        clone_command = ["git", "clone"]
        if ref is None:
            clone_command.extend(["--depth", "1"])
        clone_command.extend([repo_url, str(clone_dir)])
        subprocess.run(clone_command, check=True)
        if ref is not None:
            subprocess.run(["git", "-C", str(clone_dir), "checkout", ref], check=True)
        yield clone_dir


def run_pycroscope(taxonomy_repo: Path) -> int:
    pycroscope_repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    pythonpath = [str(taxonomy_repo), str(pycroscope_repo)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycroscope",
            "--config-file",
            "pyproject.toml",
            "taxonomy/",
        ],
        cwd=taxonomy_repo,
        env=env,
        check=False,
    )
    return proc.returncode


@contextmanager
def _taxonomy_repo_context(
    local_path: Path | None, repo_url: str, ref: str | None
) -> Iterator[tuple[Path, bool]]:
    if local_path is not None:
        yield _validate_taxonomy_repo(local_path), False
        return

    with clone_taxonomy(repo_url, ref) as taxonomy_repo:
        yield _validate_taxonomy_repo(taxonomy_repo), True


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

    try:
        with _taxonomy_repo_context(
            args.local_path, args.taxonomy_url, args.taxonomy_ref
        ) as (taxonomy_repo, cloned):
            print(f"Using taxonomy checkout: {taxonomy_repo}")
            should_install = args.install or (cloned and not args.skip_install)
            if should_install:
                print("Installing taxonomy into the current interpreter...")
                install_taxonomy(taxonomy_repo)
            return run_pycroscope(taxonomy_repo)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(exc, file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
