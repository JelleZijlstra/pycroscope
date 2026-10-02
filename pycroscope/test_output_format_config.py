import json
import subprocess
import sys
from pathlib import Path

import pytest

from .name_check_visitor import NameCheckVisitor, OutputFormatOption
from .options import InvalidConfigOption


def _write_config(path: Path, output_format: str) -> None:
    path.write_text(
        f'[tool.pycroscope]\noutput_format = "{output_format}"\n', encoding="utf-8"
    )


def test_output_format_can_be_set_in_config_file(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, "concise")
    kwargs = NameCheckVisitor.prepare_constructor_kwargs({"config_file": config_path})
    assert "output_format" not in kwargs
    assert kwargs["checker"].options.get_value_for(OutputFormatOption) == "concise"


def test_output_format_uses_config_when_not_set_on_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, "concise")
    parser = NameCheckVisitor._get_argument_parser()
    args = parser.parse_args(["--config-file", str(config_path)])
    kwargs = NameCheckVisitor.prepare_constructor_kwargs(vars(args))
    assert "output_format" not in kwargs
    assert kwargs["checker"].options.get_value_for(OutputFormatOption) == "concise"


def test_output_format_command_line_overrides_config(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, "concise")
    kwargs = NameCheckVisitor.prepare_constructor_kwargs(
        {"config_file": config_path, "output_format": "detailed"}
    )
    assert "output_format" not in kwargs
    assert kwargs["checker"].options.get_value_for(OutputFormatOption) == "detailed"


def test_output_format_rejects_invalid_config_value(tmp_path: Path) -> None:
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, "compact")
    with pytest.raises(InvalidConfigOption, match="output_format"):
        NameCheckVisitor.prepare_constructor_kwargs({"config_file": config_path})


@pytest.mark.parametrize("output_format", ["concise", "detailed"])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("ignore_comment", ["", "  # static analysis: ignore"])
def test_unused_call_pattern_output(
    tmp_path: Path, output_format: str, json_output: bool, ignore_comment: str
) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "def f(a: bool = False) -> int:\n"
        "    return 1 if a else 0\n"
        "\n"
        f"def g(b: bool) -> int:{ignore_comment}\n"
        "    return f(b)\n"
        "\n"
        "def h() -> int:\n"
        "    return g(False)\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, output_format)
    report = tmp_path / "findings.json"
    args = ["--json-output", str(report)] if json_output else []
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycroscope",
            "--config-file",
            str(config_path),
            "--find-unused-call-patterns",
            *args,
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout == ""
    if ignore_comment:
        assert result.returncode == 0, result.stderr
        assert result.stderr == ""
        assert not report.exists()
        return
    assert result.returncode == 1, result.stderr
    assert result.stderr.count("Unused call pattern:") == 1
    assert "parameter 'b' is only called with literal False" in result.stderr
    if output_format == "concise":
        assert f"{source}:4:6: Unused call pattern:" in result.stderr
    else:
        assert f"In {source} at line 4" in result.stderr
        assert "def g(b: bool) -> int:" in result.stderr
    if json_output:
        findings = json.loads(report.read_text(encoding="utf-8"))
        assert len(findings) == 1
        assert findings[0]["lineno"] == 4
        assert findings[0]["col_offset"] == 6
        assert findings[0]["description"] in result.stderr


@pytest.mark.skipif(sys.version_info < (3, 12), reason="PEP 695 syntax")
@pytest.mark.parametrize("include_alias_call", [False, True])
def test_recursive_alias_unused_call_pattern_json(
    tmp_path: Path, include_alias_call: bool
) -> None:
    source = tmp_path / "recursive.py"
    source.write_text(
        "type A = int | list[A]\n"
        "\n"
        "def f(a: A | bytes) -> None:\n"
        "    pass\n"
        "\n"
        "def g() -> None:\n"
        "    f(bytes(1))\n" + ("    f([1, [2]])\n" if include_alias_call else ""),
        encoding="utf-8",
    )
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("value: int = 'wrong'\n", encoding="utf-8")
    config_path = tmp_path / "pyproject.toml"
    _write_config(config_path, "concise")
    report = tmp_path / "findings.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pycroscope",
            "--config-file",
            str(config_path),
            "--find-unused-call-patterns",
            "--json-output",
            str(report),
            str(source),
            str(unrelated),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    findings = json.loads(report.read_text(encoding="utf-8"))
    assert any(finding["filename"] == str(unrelated) for finding in findings)
    alias_findings = [
        finding for finding in findings if finding["filename"] == str(source)
    ]
    if include_alias_call:
        assert alias_findings == []
    else:
        assert len(alias_findings) == 1
        description = alias_findings[0]["description"]
        assert "parameter 'a' never receives" in description
        assert ".A" in description
        assert "(observed bytes)" in description
        assert description in result.stderr
