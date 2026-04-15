from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


def _looks_like_git_url(repo_ref: str) -> bool:
    return repo_ref.startswith(
        ("http://", "https://", "ssh://", "git://", "git@")
    ) or repo_ref.endswith(".git")


def _sanitize_repo_name(repo_ref: str) -> str:
    name = repo_ref.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return sanitized or "repo"


def _resolve_local_repo(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Repository checkout does not exist: {resolved}")
    return resolved


def _resolve_repo_relative_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_workdir_relative_path(workdir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workdir / path
    return path.resolve()


def install_editable_package(package_dir: Path) -> None:
    if importlib.util.find_spec("pip") is not None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(package_dir)], check=True
        )
        return

    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "-e", str(package_dir)],
        check=True,
    )


@contextmanager
def clone_repo(repo_url: str, ref: str | None = None) -> Iterator[Path]:
    repo_name = _sanitize_repo_name(repo_url)
    with tempfile.TemporaryDirectory(prefix=f"{repo_name}-") as temp_dir:
        clone_dir = Path(temp_dir) / repo_name
        clone_command = ["git", "clone"]
        if ref is None:
            clone_command.extend(["--depth", "1"])
        clone_command.extend([repo_url, str(clone_dir)])
        subprocess.run(clone_command, check=True)
        if ref is not None:
            subprocess.run(["git", "-C", str(clone_dir), "checkout", ref], check=True)
        yield clone_dir


@contextmanager
def repo_context(repo_ref: str, ref: str | None = None) -> Iterator[tuple[Path, bool]]:
    if Path(repo_ref).expanduser().exists():
        yield _resolve_local_repo(repo_ref), False
        return
    if _looks_like_git_url(repo_ref):
        with clone_repo(repo_ref, ref) as repo_root:
            yield repo_root, True
        return
    raise FileNotFoundError(
        f"Repository checkout does not exist and does not look like a git URL: {repo_ref}"
    )


def validate_repo_checkout(
    repo_root: Path,
    *,
    run_from: str,
    config_file: str,
    targets: Sequence[str],
    install_path: str,
    pythonpath_entries: Sequence[str],
) -> tuple[Path, Path, Path, list[Path]]:
    resolved_repo = repo_root.expanduser().resolve()
    if not resolved_repo.is_dir():
        raise FileNotFoundError(f"Repository checkout does not exist: {resolved_repo}")

    workdir = _resolve_repo_relative_path(resolved_repo, run_from)
    if not workdir.is_dir():
        raise FileNotFoundError(f"Working directory does not exist: {workdir}")

    config_path = _resolve_workdir_relative_path(workdir, config_file)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    install_dir = _resolve_repo_relative_path(resolved_repo, install_path)
    if not install_dir.is_dir():
        raise FileNotFoundError(f"Install path does not exist: {install_dir}")

    for target in targets:
        target_path = _resolve_workdir_relative_path(workdir, target)
        if not target_path.exists():
            raise FileNotFoundError(f"Target path does not exist: {target_path}")

    resolved_pythonpath = [
        _resolve_repo_relative_path(resolved_repo, entry)
        for entry in pythonpath_entries
    ]
    return resolved_repo, workdir, install_dir, resolved_pythonpath


def run_pycroscope(
    repo_root: Path,
    *,
    workdir: Path,
    config_file: str,
    targets: Sequence[str],
    pycroscope_args: Sequence[str] = (),
    pythonpath_entries: Sequence[Path] = (),
) -> int:
    pycroscope_repo = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    pythonpath = [*(str(path) for path in pythonpath_entries), str(pycroscope_repo)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        pythonpath.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycroscope",
            *pycroscope_args,
            "--config-file",
            config_file,
            *targets,
        ],
        cwd=workdir,
        env=env,
        check=False,
    )
    return proc.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run pycroscope against a local checkout or a temporary clone of a "
            "third-party repository."
        )
    )
    parser.add_argument(
        "repo", help="Path to a local checkout, or a git-compatible URL to clone."
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="One or more files or directories to analyze, relative to --run-from.",
    )
    parser.add_argument(
        "--ref", help="Optional git ref to check out after cloning a remote repository."
    )
    parser.add_argument(
        "--config-file",
        default="pyproject.toml",
        help="Pycroscope config file path, relative to --run-from.",
    )
    parser.add_argument(
        "--run-from",
        default=".",
        help="Directory inside the checkout to use as the working directory.",
    )
    parser.add_argument(
        "--install-path",
        default=".",
        help="Directory inside the checkout to install editable when installation is enabled.",
    )
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        help="Additional path to prepend to PYTHONPATH, relative to the repo root.",
    )
    parser.add_argument(
        "--pycroscope-arg",
        action="append",
        default=[],
        help=(
            "Additional argument to pass through to pycroscope. May be repeated. "
            "For values starting with '-', use --pycroscope-arg=VALUE."
        ),
    )
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--install",
        action="store_true",
        help="Install the checkout into the current interpreter before running pycroscope.",
    )
    install_group.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install the checkout before running, even for a freshly cloned repo.",
    )
    args = parser.parse_args(argv)

    try:
        with repo_context(args.repo, args.ref) as (repo_root, cloned):
            repo_root, workdir, install_dir, pythonpath_entries = (
                validate_repo_checkout(
                    repo_root,
                    run_from=args.run_from,
                    config_file=args.config_file,
                    targets=args.targets,
                    install_path=args.install_path,
                    pythonpath_entries=args.pythonpath,
                )
            )
            print(f"Using checkout: {repo_root}")
            should_install = args.install or (cloned and not args.skip_install)
            if should_install:
                print(f"Installing editable package from: {install_dir}")
                install_editable_package(install_dir)
            return run_pycroscope(
                repo_root,
                workdir=workdir,
                config_file=args.config_file,
                targets=args.targets,
                pycroscope_args=args.pycroscope_arg,
                pythonpath_entries=[repo_root, *pythonpath_entries],
            )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(exc, file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
