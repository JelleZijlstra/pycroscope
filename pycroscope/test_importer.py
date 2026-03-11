import sys
from pathlib import Path
from typing import cast, final

from . import importer


def test_import_module_pyi(tmp_path: Path) -> None:
    stub = tmp_path / "final_stub.pyi"
    stub.write_text(
        "from typing import final\n\n@final\nclass C:\n    ...\n", encoding="utf-8"
    )
    module_name = "final_stub_for_test"

    # Python 3.10's typing.final does not set __final__; newer versions do.
    @final
    class _Probe: ...

    expected_final_marker = getattr(_Probe, "__final__", False)
    try:
        module = importer.import_module(module_name, stub)
        assert module.__name__ == module_name
        assert getattr(module.C, "__final__", False) is expected_final_marker
    finally:
        sys.modules.pop(module_name, None)


def test_load_module_from_file_pyi(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    stub = pkg / "mod.pyi"
    stub.write_text("VALUE = 42\n", encoding="utf-8")

    cast(list[str], sys.path).insert(0, str(tmp_path))
    try:
        module, is_compiled = importer.load_module_from_file(
            str(stub), import_paths=[str(tmp_path)]
        )
        assert module is not None
        assert module.VALUE == 42
        assert is_compiled is False
    finally:
        sys.modules.pop("pkg.mod", None)
        sys.modules.pop("pkg", None)
        cast(list[str], sys.path).remove(str(tmp_path))
