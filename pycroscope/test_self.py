"""

Runs pycroscope on itself.

"""

import ast

import pycroscope
from pycroscope.error_code import ErrorCode
from pycroscope.test_node_visitor import skip_if_not_installed


class PycroscopeVisitor(pycroscope.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False
    config_filename = "../pyproject.toml"


def _check_all_files_with_annotations() -> None:
    settings = PycroscopeVisitor._get_default_settings()
    if settings is not None:
        settings[ErrorCode.implicit_any] = False
    kwargs: dict[str, object] = {"settings": settings}
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


@skip_if_not_installed("asynq")
def test_all() -> None:
    _check_all_files_with_annotations()


if __name__ == "__main__":
    PycroscopeVisitor.main()
