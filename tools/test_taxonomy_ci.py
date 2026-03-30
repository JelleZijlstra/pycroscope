from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="tooling tests require Python 3.11+"
)

if sys.version_info >= (3, 11):
    import tools.taxonomy_ci as taxonomy_ci


def _make_taxonomy_repo(root: Path) -> Path:
    (root / "taxonomy").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[tool.pycroscope]\npaths=['.']\n", encoding="utf-8"
    )
    return root


def test_install_taxonomy_uses_editable_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_taxonomy_repo(tmp_path / "taxonomy")
    seen_command: list[str] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command
        seen_command = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "tools.taxonomy_ci.importlib.util.find_spec", lambda name: object()
    )
    monkeypatch.setattr("tools.taxonomy_ci.subprocess.run", fake_run)

    taxonomy_ci.install_taxonomy(repo)

    assert seen_command == [sys.executable, "-m", "pip", "install", "-e", str(repo)]


def test_install_taxonomy_falls_back_to_uv_without_pip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_taxonomy_repo(tmp_path / "taxonomy")
    seen_command: list[str] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command
        seen_command = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.taxonomy_ci.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("tools.taxonomy_ci.subprocess.run", fake_run)

    taxonomy_ci.install_taxonomy(repo)

    assert seen_command == [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "-e",
        str(repo),
    ]


def test_clone_taxonomy_uses_shallow_clone_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_commands: list[list[str]] = []

    @contextmanager
    def fake_tempdir(*, prefix: str) -> Iterator[str]:
        assert prefix == "taxonomy-"
        yield str(tmp_path)

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.taxonomy_ci.tempfile.TemporaryDirectory", fake_tempdir)
    monkeypatch.setattr("tools.taxonomy_ci.subprocess.run", fake_run)

    with taxonomy_ci.clone_taxonomy("https://example.com/taxonomy.git") as repo:
        assert repo == tmp_path / "taxonomy"

    assert seen_commands == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://example.com/taxonomy.git",
            str(tmp_path / "taxonomy"),
        ]
    ]


def test_clone_taxonomy_checks_out_ref_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_commands: list[list[str]] = []

    @contextmanager
    def fake_tempdir(*, prefix: str) -> Iterator[str]:
        assert prefix == "taxonomy-"
        yield str(tmp_path)

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("tools.taxonomy_ci.tempfile.TemporaryDirectory", fake_tempdir)
    monkeypatch.setattr("tools.taxonomy_ci.subprocess.run", fake_run)

    with taxonomy_ci.clone_taxonomy(
        "https://example.com/taxonomy.git", ref="master"
    ) as repo:
        assert repo == tmp_path / "taxonomy"

    assert seen_commands == [
        [
            "git",
            "clone",
            "https://example.com/taxonomy.git",
            str(tmp_path / "taxonomy"),
        ],
        ["git", "-C", str(tmp_path / "taxonomy"), "checkout", "master"],
    ]


def test_run_pycroscope_uses_taxonomy_ci_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy_repo = _make_taxonomy_repo(tmp_path / "taxonomy")
    seen_command: list[str] = []
    seen_cwd: Path | None = None
    seen_env: dict[str, str] | None = None

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command, seen_cwd, seen_env
        seen_command = command
        seen_cwd = kwargs["cwd"]
        seen_env = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYTHONPATH", "existing-path")
    monkeypatch.setattr("tools.taxonomy_ci.subprocess.run", fake_run)

    result = taxonomy_ci.run_pycroscope(taxonomy_repo)

    assert result == 0
    assert seen_command == [
        sys.executable,
        "-m",
        "pycroscope",
        "--config-file",
        "pyproject.toml",
        "taxonomy/",
    ]
    assert seen_cwd == taxonomy_repo
    assert seen_env is not None
    assert seen_env["PYTHONPATH"].split(taxonomy_ci.os.pathsep) == [
        str(taxonomy_repo),
        str(Path(taxonomy_ci.__file__).resolve().parent.parent),
        "existing-path",
    ]


def test_main_uses_default_local_repo_without_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy_repo = _make_taxonomy_repo(tmp_path / "taxonomy")
    seen_repo: Path | None = None
    install_called = False

    def fake_run_pycroscope(repo: Path) -> int:
        nonlocal seen_repo
        seen_repo = repo
        return 7

    def fake_install_taxonomy(repo: Path) -> None:
        nonlocal install_called
        install_called = True

    monkeypatch.setattr("tools.taxonomy_ci.run_pycroscope", fake_run_pycroscope)
    monkeypatch.setattr("tools.taxonomy_ci.install_taxonomy", fake_install_taxonomy)

    result = taxonomy_ci.main(["--local-path", str(taxonomy_repo)])

    assert result == 7
    assert seen_repo == taxonomy_repo
    assert not install_called


def test_main_clones_and_installs_public_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taxonomy_repo = _make_taxonomy_repo(tmp_path / "cloned-taxonomy")
    seen_repo: Path | None = None
    install_called = False
    seen_clone_args: tuple[str, str | None] | None = None

    @contextmanager
    def fake_clone_taxonomy(repo_url: str, ref: str | None = None) -> Iterator[Path]:
        nonlocal seen_clone_args
        seen_clone_args = (repo_url, ref)
        yield taxonomy_repo

    def fake_run_pycroscope(repo: Path) -> int:
        nonlocal seen_repo
        seen_repo = repo
        return 0

    def fake_install_taxonomy(repo: Path) -> None:
        nonlocal install_called
        install_called = True

    monkeypatch.setattr("tools.taxonomy_ci.clone_taxonomy", fake_clone_taxonomy)
    monkeypatch.setattr("tools.taxonomy_ci.run_pycroscope", fake_run_pycroscope)
    monkeypatch.setattr("tools.taxonomy_ci.install_taxonomy", fake_install_taxonomy)

    result = taxonomy_ci.main(["--taxonomy-ref", "master"])

    assert result == 0
    assert seen_clone_args == (taxonomy_ci.PUBLIC_TAXONOMY_REPO_URL, "master")
    assert seen_repo == taxonomy_repo
    assert install_called


def test_main_requires_python_314(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("tools.taxonomy_ci.sys.version_info", (3, 13, 9))

    result = taxonomy_ci.main([])

    assert result == 1
    assert "requires Python 3.14+" in capsys.readouterr().out
