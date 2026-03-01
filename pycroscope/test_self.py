"""

Runs pycroscope on itself.

"""

import ast
import sys
import textwrap
from pathlib import Path

import pycroscope
from pycroscope.error_code import ErrorCode
from pycroscope.test_node_visitor import skip_if_not_installed


class PycroscopeVisitor(pycroscope.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False
    config_filename = "../pyproject.toml"


def _files_for_self_check() -> list[str]:
    files = ["pycroscope"]
    if sys.version_info >= (3, 11):
        conformance_ci = (
            Path(__file__).resolve().parent.parent / "tools" / "conformance_ci.py"
        )
        files.append(str(conformance_ci))
    return files


def _check_all_files_with_annotations() -> None:
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {"settings": settings, "files": _files_for_self_check()}
    kwargs = dict(PycroscopeVisitor.prepare_constructor_kwargs(kwargs))
    files = PycroscopeVisitor.get_files_to_check(False, **kwargs)
    failures = []
    missing_annotations = []
    for filename in files:
        with open(filename, encoding="utf-8") as f:
            contents = f.read()
        tree = ast.parse(contents.encode("utf-8"), filename)
        visitor = PycroscopeVisitor(filename, contents, tree, annotate=True, **kwargs)
        failures += visitor.check()
        for node in ast.walk(tree):
            if (
                hasattr(node, "lineno")
                and hasattr(node, "col_offset")
                and not hasattr(node, "inferred_value")
                and not isinstance(node, (ast.keyword, ast.arg))
            ):
                missing_annotations.append((filename, node))
    failures += PycroscopeVisitor.perform_final_checks(kwargs)
    assert not failures, "".join(failure["message"] for failure in failures)
    if missing_annotations:
        for filename, node in missing_annotations:
            print(f"{filename}:{node.lineno}:{node.col_offset}: {ast.dump(node)}")
        assert False, f"found no annotations on {len(missing_annotations)} expressions"


def _missing_annotations_for_tree(tree: ast.AST) -> list[ast.AST]:
    return [
        node
        for node in ast.walk(tree)
        if (
            hasattr(node, "lineno")
            and hasattr(node, "col_offset")
            and not hasattr(node, "inferred_value")
            and not isinstance(node, (ast.keyword, ast.arg))
        )
    ]


def test_typealiastype_subscript_annotation(tmp_path: Path) -> None:
    code = textwrap.dedent("""
        from typing import List, TypeVar
        from typing_extensions import TypeAliasType, assert_type

        T = TypeVar("T")
        MyType = TypeAliasType("MyType", List[T], type_params=(T,))

        def f(x: MyType[int]) -> None:
            assert_type(x, MyType[int])
    """)
    filename = tmp_path / "type_alias_annotation.py"
    filename.write_text(code, encoding="utf-8")
    tree = ast.parse(code.encode("utf-8"), str(filename))
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {"settings": settings, "files": [str(filename)]}
    kwargs = dict(PycroscopeVisitor.prepare_constructor_kwargs(kwargs))
    visitor = PycroscopeVisitor(str(filename), code, tree, annotate=True, **kwargs)
    failures = visitor.check()
    assert not any(failure["code"].name == "internal_error" for failure in failures)
    assert not _missing_annotations_for_tree(tree)


@skip_if_not_installed("asynq")
def test_all() -> None:
    _check_all_files_with_annotations()


if __name__ == "__main__":
    PycroscopeVisitor.main()
