# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import KnownValue


class TestInferenceHelpers(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pycroscope import assert_is_value, dump_value
        from pycroscope.value import Value

        def capybara(val: Value) -> None:
            reveal_type(dump_value)  # E: reveal_type
            dump_value(reveal_type)  # E: reveal_type
            assert_is_value([], KnownValue([]))
            assert_is_value([], KnownValue(()))  # E: inference_failure
            assert_is_value([], val)  # E: inference_failure

    @assert_passes()
    def test_return_value(self) -> None:
        from pycroscope import assert_is_value, dump_value

        def capybara():
            x = dump_value([])  # E: reveal_type
            y = reveal_type([])  # E: reveal_type
            assert_is_value(x, KnownValue([]))
            assert_is_value(y, KnownValue([]))

    @assert_passes()
    def test_assert_type(self) -> None:
        from typing import Any

        from pycroscope.extensions import assert_type

        def capybara(x: int) -> None:
            assert_type(x, int)
            assert_type(x, "int")
            assert_type(x, Any)  # E: inference_failure
            assert_type(x, str)  # E: inference_failure


class TestAssertError(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                f("x")

            with assert_error():  # E: inference_failure
                f(1)


class TestRevealLocals(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pycroscope.extensions import reveal_locals

        def capybara(a: object, b: str) -> None:
            c = 3
            if b == "x":
                reveal_locals()  # E: reveal_type
            print(a, b, c)
