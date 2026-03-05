# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestSysPlatform(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import os
        import sys

        from typing_extensions import assert_type

        def capybara() -> None:
            if sys.platform == "win32":
                assert_type(os.P_DETACH, int)
            else:
                os.P_DETACH  # E: undefined_attribute


class TestSysVersion(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import ast
        import sys

        from typing_extensions import assert_type

        if sys.version_info >= (3, 10):

            def capybara(m: ast.Match) -> None:
                assert_type(m, ast.Match)

        if sys.version_info >= (3, 12):

            def pacarana(m: ast.TypeVar) -> None:
                assert_type(m, ast.TypeVar)


class TestTypeCheckingDirective(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import_from(self):
        from typing import TYPE_CHECKING

        from typing_extensions import assert_type

        if not TYPE_CHECKING:
            a: int = ""

        if TYPE_CHECKING:
            b: list[int] = [1, 2, 3]
        else:
            b: list[str] = ["a", "b", "c"]

        assert_type(b, list[int])

    @assert_passes()
    def test_module_attribute(self):
        import typing

        from typing_extensions import assert_type

        if typing.TYPE_CHECKING:
            c: int = 1
        else:
            c: str = ""

        assert_type(c, int)
