from . import name_check_visitor  # noqa: F401
from .options import render_config_options_markdown


def test_render_config_options_markdown() -> None:
    rendered = render_config_options_markdown(
        excluded_options=frozenset({"paths", "import_paths"}), include_error_codes=False
    )
    lines = rendered.splitlines()
    assert any(line.startswith("- `enforce_no_unused`") for line in lines)
    assert not any(line.startswith("- `add_import`") for line in lines)
    assert not any(line.startswith("- `paths`") for line in lines)
    assert not any(line.startswith("- `import_paths`") for line in lines)
    assert ":class:" not in rendered
    assert ":term:" not in rendered
    assert ":data:" not in rendered
    assert "{py:class}`pycroscope.value.VariableNameValue`" in rendered
    assert "{term}`impl`" in rendered
    assert (
        "If any of these functions returns True, we will exclude this object from the "
        "unused object check."
    ) in rendered
