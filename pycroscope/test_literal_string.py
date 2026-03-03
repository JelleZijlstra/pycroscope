# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import TypedValue, assert_is_value


class TestLiteralString(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing_extensions import LiteralString

        def f(x: LiteralString) -> LiteralString:
            return "x"

        def capybara(x: str, y: LiteralString):
            f(x)  # E: incompatible_argument
            f(y)
            f("x")
            assert_is_value(f("x"), TypedValue(str, literal_only=True))

    @assert_passes()
    def test_f_strings(self):
        from typing_extensions import LiteralString

        def capybara(a: LiteralString, b: LiteralString, non_literal: str):
            assert_is_value(f"{a} {b}", TypedValue(str, literal_only=True))
            assert_is_value(f"{a} {non_literal}", TypedValue(str))
            assert_is_value(f"{1}", TypedValue(str))
