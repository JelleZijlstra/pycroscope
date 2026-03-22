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

    @assert_passes()
    def test_nested_control_flow(self) -> None:
        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara(flag: bool) -> None:
            with assert_error():
                if flag:
                    f("x")
                else:
                    f("y")

    @assert_passes()
    def test_nested_with_block(self) -> None:
        import contextlib

        from typing_extensions import assert_type

        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                with contextlib.nullcontext(None) as value:
                    assert_type(value, None)
                    f("x")

    @assert_passes()
    def test_multiple_errors_in_block(self) -> None:
        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def g(x: str) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                f("x")
                g(1)


class TestRevealLocals(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        from pycroscope.extensions import reveal_locals

        def capybara(a: object, b: str) -> None:
            c = 3
            if b == "x":
                reveal_locals()  # E: reveal_type
            print(a, b, c)


class TestGetMro(TestNameCheckVisitorBase):

    @assert_passes()
    def test_get_mro(self) -> None:
        from typing import Generic, NamedTuple, TypeVar

        from pycroscope import get_mro
        from pycroscope.extensions import assert_type

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Child(Base[int]):
            pass

        class L(list[int]):
            pass

        class Tup(tuple[int, str]):
            pass

        class NT(NamedTuple):
            x: int
            y: str

        assert_type(get_mro(object), tuple[object])
        assert_type(get_mro(int), tuple[int, object])
        assert_type(get_mro(Base), tuple[Base[T], Generic, object])
        assert_type(get_mro(Child), tuple[Base[int], Generic, object])
        assert_type(get_mro(L), tuple[list[int], object])
        assert_type(get_mro(Tup), tuple[tuple[int, str], object])
        assert_type(get_mro(NT), tuple[tuple[int, str], object])

    @assert_passes(run_in_both_module_modes=True)
    def test_get_mro_multiple_inheritance(self) -> None:
        from pycroscope import get_mro
        from pycroscope.extensions import assert_type

        class O:
            pass

        class A(O):
            pass

        class B(O):
            pass

        class C(O):
            pass

        class D(O):
            pass

        class E(O):
            pass

        class K1(A, B, C):
            pass

        class K2(D, B, E):
            pass

        class K3(D, A):
            pass

        class Z(K1, K2, K3):
            pass

        assert_type(get_mro(K1), tuple[K1, A, B, C, O, object])
        assert_type(get_mro(K2), tuple[K2, D, B, E, O, object])
        assert_type(get_mro(K3), tuple[K3, D, A, O, object])
        assert_type(get_mro(Z), tuple[Z, K1, K2, K3, D, A, B, C, E, O, object])

    @assert_passes(run_in_both_module_modes=True)
    def test_get_mro_multiple_inheritance_with_generics(self) -> None:
        from typing import Generic, TypeVar

        from pycroscope import get_mro
        from pycroscope.extensions import assert_type

        T = TypeVar("T")
        U = TypeVar("U")

        class Base(Generic[T]):
            pass

        class Left(Base[T]):
            pass

        class Right(Generic[U]):
            pass

        class Child(Left[int], Right[str]):
            pass

        assert_type(get_mro(Left), tuple[Left[T], Base[T], Generic, object])
        assert_type(get_mro(Right), tuple[Right[U], Generic, object])
        assert_type(
            get_mro(Child),
            tuple[Child, Left[int], Right[str], Base[int], Generic, object],
        )
