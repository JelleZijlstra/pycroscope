from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

from tools import run_third_party, third_party_ci


def test_read_local_config(tmp_path: Path) -> None:
    config = tmp_path / "third_party_ci_local.toml"
    config.write_text(
        textwrap.dedent("""
            [local]
            taxonomy = "/path/to/taxonomy"
        """),
        encoding="utf-8",
    )

    assert third_party_ci.read_local_config(config, {"taxonomy"}) == {
        "taxonomy": "/path/to/taxonomy"
    }


def test_read_local_config_rejects_unknown_check(tmp_path: Path) -> None:
    config = tmp_path / "third_party_ci_local.toml"
    config.write_text(
        textwrap.dedent("""
            [local]
            unknown = "/tmp/unknown"
        """),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown third-party check"):
        third_party_ci.read_local_config(config, {"taxonomy"})


def test_build_run_third_party_argv_uses_local_override() -> None:
    check = third_party_ci.ThirdPartyCheck(
        name="taxonomy",
        repo_url="https://example.com/taxonomy.git",
        targets=("taxonomy/",),
    )
    argv = third_party_ci.build_run_third_party_argv(
        check, {"taxonomy": "/path/to/taxonomy"}, verbose=True
    )

    assert argv[0] == "/path/to/taxonomy"
    assert "--verbose" in argv


def test_main_reads_local_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "third_party_ci_local.toml"
    config.write_text(
        textwrap.dedent("""
            [local]
            taxonomy = "/path/to/taxonomy"
        """),
        encoding="utf-8",
    )
    seen_overrides: dict[str, str] = {}

    def fake_run_check(
        check: third_party_ci.ThirdPartyCheck,
        local_overrides: dict[str, str],
        *,
        verbose: bool = False,
    ) -> int:
        nonlocal seen_overrides
        seen_overrides = local_overrides
        assert not verbose
        return 0

    monkeypatch.setattr(third_party_ci, "LOCAL_CONFIG_PATH", config)
    monkeypatch.setattr(third_party_ci, "run_check", fake_run_check)

    assert third_party_ci.main(["taxonomy"]) == 0
    assert seen_overrides == {"taxonomy": "/path/to/taxonomy"}


def test_main_command_line_local_overrides_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "third_party_ci_local.toml"
    config.write_text(
        textwrap.dedent("""
            [local]
            taxonomy = "/path/to/taxonomy"
        """),
        encoding="utf-8",
    )
    seen_overrides: dict[str, str] = {}
    seen_verbose = False

    def fake_run_check(
        check: third_party_ci.ThirdPartyCheck,
        local_overrides: dict[str, str],
        *,
        verbose: bool = False,
    ) -> int:
        nonlocal seen_overrides, seen_verbose
        seen_overrides = local_overrides
        seen_verbose = verbose
        return 0

    monkeypatch.setattr(third_party_ci, "LOCAL_CONFIG_PATH", config)
    monkeypatch.setattr(third_party_ci, "run_check", fake_run_check)

    assert (
        third_party_ci.main(["taxonomy", "--local", "taxonomy:/cli/taxonomy", "-v"])
        == 0
    )
    assert seen_overrides == {"taxonomy": "/cli/taxonomy"}
    assert seen_verbose is True


def test_install_editable_package_hides_success_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_command: list[str] = []
    seen_kwargs: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command, seen_kwargs
        seen_command = command
        seen_kwargs = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_third_party.subprocess, "run", fake_run)

    run_third_party.install_editable_package(tmp_path)

    assert seen_command == [
        "uv",
        "pip",
        "install",
        "--python",
        run_third_party.sys.executable,
        "-e",
        str(tmp_path),
    ]
    assert seen_kwargs["stdout"] is subprocess.PIPE
    assert seen_kwargs["stderr"] is subprocess.STDOUT
    assert seen_kwargs["text"] is True


def test_install_editable_package_verbose_streams_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_kwargs
        seen_kwargs = kwargs
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_third_party.subprocess, "run", fake_run)

    run_third_party.install_editable_package(tmp_path, verbose=True)

    assert "stdout" not in seen_kwargs
    assert "stderr" not in seen_kwargs
