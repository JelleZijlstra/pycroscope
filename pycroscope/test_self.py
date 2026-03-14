"""

Runs pycroscope on itself.

"""

import ast
import sys
import textwrap
from pathlib import Path

import pycroscope
from pycroscope import node_visitor
from pycroscope.error_code import ErrorCode
from pycroscope.find_unused import UnusedObjectFinder
from pycroscope.shared_options import EnforceNoUnused, EnforceNoUnusedAttributes
from pycroscope.test_name_check_visitor import _make_module
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
    failures, missing_annotations = _check_files_with_annotations(
        _files_for_self_check()
    )
    assert not failures, "".join(
        failure["message"] for failure in failures if "message" in failure
    )
    if missing_annotations:
        for filename, node in missing_annotations:
            print(
                f"{filename}:{getattr(node, 'lineno')}:{getattr(node, 'col_offset')}:"
                f" {ast.dump(node)}"
            )
        assert False, f"found no annotations on {len(missing_annotations)} expressions"


def _check_files_with_annotations(
    files: list[str],
) -> tuple[list[node_visitor.Failure], list[tuple[str, ast.AST]]]:
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {"settings": settings, "files": files}
    kwargs = dict(PycroscopeVisitor.prepare_constructor_kwargs(kwargs))
    files_to_check = PycroscopeVisitor.get_files_to_check(False, **kwargs)
    failures: list[node_visitor.Failure] = []
    missing_annotations: list[tuple[str, ast.AST]] = []
    should_check_unused_attributes = kwargs["checker"].options.get_value_for(
        EnforceNoUnusedAttributes
    )
    unused_finder = UnusedObjectFinder(
        kwargs["checker"].options,
        enabled=kwargs["checker"].options.get_value_for(EnforceNoUnused),
        print_output=False,
    )
    attribute_checker_enabled = kwargs[
        "checker"
    ].options.is_error_code_enabled_anywhere(ErrorCode.attribute_is_never_set)
    with pycroscope.name_check_visitor.ClassAttributeChecker(
        enabled=attribute_checker_enabled or should_check_unused_attributes,
        should_check_unused_attributes=should_check_unused_attributes,
        options=kwargs["checker"].options,
        ts_finder=kwargs["checker"].ts_finder,
    ) as attribute_checker:
        for filename in files_to_check:
            with open(filename, encoding="utf-8") as f:
                contents = f.read()
            tree = ast.parse(contents.encode("utf-8"), filename)
            visitor = PycroscopeVisitor(
                filename,
                contents,
                tree,
                annotate=True,
                attribute_checker=attribute_checker,
                unused_finder=unused_finder,
                **kwargs,
            )
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
    failures += _unused_object_failures(unused_finder)
    failures += _unused_attribute_failures(attribute_checker)
    return failures, missing_annotations


def _unused_object_failures(
    unused_finder: UnusedObjectFinder,
) -> list[node_visitor.Failure]:
    failures: list[node_visitor.Failure] = []
    for unused_object in unused_finder.get_unused_objects():
        failure = str(unused_object)
        failures.append(
            {
                "filename": node_visitor.UNUSED_OBJECT_FILENAME,
                "absolute_filename": node_visitor.UNUSED_OBJECT_FILENAME,
                "message": failure + "\n",
                "description": failure,
            }
        )
    return failures


def _unused_attribute_failures(
    attribute_checker: pycroscope.name_check_visitor.ClassAttributeChecker | None,
) -> list[node_visitor.Failure]:
    if attribute_checker is None:
        return []
    failures: list[node_visitor.Failure] = []
    for unused_attribute in attribute_checker.unused_attributes:
        failure = str(unused_attribute)
        failures.append(
            {
                "filename": node_visitor.UNUSED_OBJECT_FILENAME,
                "absolute_filename": node_visitor.UNUSED_OBJECT_FILENAME,
                "message": failure + "\n",
                "description": failure,
            }
        )
    return failures


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


def test_self_check_reports_unused_objects() -> None:
    code = textwrap.dedent("""
        def unused() -> None:
            pass
    """)
    filename = "capybara_module.py"
    tree = ast.parse(code.encode("utf-8"), filename)
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {"settings": settings, "files": [filename]}
    kwargs = dict(PycroscopeVisitor.prepare_constructor_kwargs(kwargs))
    unused_finder = UnusedObjectFinder(
        kwargs["checker"].options,
        enabled=kwargs["checker"].options.get_value_for(EnforceNoUnused),
        print_output=False,
    )
    visitor = PycroscopeVisitor(
        filename,
        code,
        tree,
        annotate=True,
        module=_make_module(code),
        unused_finder=unused_finder,
        **kwargs,
    )
    failures = visitor.check()
    failures += PycroscopeVisitor.perform_final_checks(kwargs)
    failures += _unused_object_failures(unused_finder)
    assert any(
        failure["description"].endswith(".unused: unused") for failure in failures
    )


def test_self_check_reports_unused_attributes() -> None:
    code = textwrap.dedent("""
        class Config:
            enabled = True

            def __init__(self) -> None:
                self.cache = 0

            def is_enabled(self) -> bool:
                return self.enabled
    """)
    filename = "capybara_module.py"
    tree = ast.parse(code.encode("utf-8"), filename)
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {
        "settings": settings,
        "files": [filename],
        "enforce_no_unused_attributes": True,
    }
    kwargs = dict(PycroscopeVisitor.prepare_constructor_kwargs(kwargs))
    with pycroscope.name_check_visitor.ClassAttributeChecker(
        enabled=True,
        should_check_unused_attributes=True,
        options=kwargs["checker"].options,
        ts_finder=kwargs["checker"].ts_finder,
    ) as attribute_checker:
        visitor = PycroscopeVisitor(
            filename,
            code,
            tree,
            annotate=True,
            module=_make_module(code),
            attribute_checker=attribute_checker,
            **kwargs,
        )
        failures = visitor.check()
    failures += PycroscopeVisitor.perform_final_checks(kwargs)
    failures += _unused_attribute_failures(attribute_checker)
    assert any(
        failure["description"].endswith(".Config'>.cache") for failure in failures
    )


@skip_if_not_installed("asynq")
def test_all() -> None:
    _check_all_files_with_annotations()


if __name__ == "__main__":
    PycroscopeVisitor.main()
