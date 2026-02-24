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
