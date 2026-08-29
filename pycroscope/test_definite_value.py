# static analysis: ignore
import sys

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

    def test_membership(self):
        platform = repr(sys.platform)
        self.assert_passes(f"""
            import sys

            tuple_platforms = ({platform}, "other")
            set_platforms = {{{platform}, "other"}}
            list_platforms = [{platform}, "other"]

            if sys.platform in tuple_platforms:
                tuple_result = 1
            else:
                1 + "x"

            if sys.platform in set_platforms:
                set_result = 1
            else:
                1 + "x"

            if sys.platform in list_platforms:
                list_result = 1
            else:
                1 + "x"

            if sys.platform not in ("definitely-not-a-platform",):
                negative_result = 1
            else:
                1 + "x"

            tuple_result + set_result + list_result + negative_result
            """)

    def test_startswith(self):
        prefix = repr(sys.platform[: max(1, len(sys.platform) // 2)])
        self.assert_passes(f"""
            import sys

            if sys.platform.startswith({prefix}):
                result = 1
            else:
                1 + "x"

            if sys.platform.startswith("definitely-not-a-platform"):
                1 + "x"
            else:
                other_result = 1

            result + other_result
            """)

    def test_boolean_combinations(self):
        platform = repr(sys.platform)
        self.assert_passes(f"""
            import sys

            if sys.platform == {platform} and sys.version_info >= (0, 0):
                and_result = 1
            else:
                1 + "x"

            if (
                sys.platform == "definitely-not-a-platform"
                or sys.version_info < (0, 0)
            ):
                1 + "x"
            else:
                or_result = 1

            and_result + or_result
            """)

    @assert_passes()
    def test_dynamic_and_mixed_containers_are_not_definite(self):
        import sys

        def capybara(platforms: list[str], prefix: str) -> None:
            if sys.platform in platforms:
                1 + "x"  # E: unsupported_operation
            else:
                1 + "x"  # E: unsupported_operation

            if sys.platform in ("darwin", 1):
                1 + "x"  # E: unsupported_operation
            else:
                1 + "x"  # E: unsupported_operation

            if sys.platform.startswith(prefix):
                1 + "x"  # E: unsupported_operation
            else:
                1 + "x"  # E: unsupported_operation


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


class TestSysImplementation(TestNameCheckVisitorBase):
    def test_name(self):
        implementation_name = repr(sys.implementation.name)
        self.assert_passes(f"""
            import sys

            if sys.implementation.name == {implementation_name}:
                equality_result = 1
            else:
                1 + "x"

            if sys.implementation.name in ({implementation_name}, "other"):
                membership_result = 1
            else:
                1 + "x"

            if sys.implementation.name != "definitely-not-an-implementation":
                inequality_result = 1
            else:
                1 + "x"

            equality_result + membership_result + inequality_result
            """)

    def test_version(self):
        implementation_version = repr(sys.implementation.version[:2])
        self.assert_passes(f"""
            import sys

            if sys.implementation.version >= {implementation_version}:
                result = 1
            else:
                1 + "x"

            if sys.implementation.version < (0, 0):
                1 + "x"
            else:
                other_result = 1

            result + other_result
            """)


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
