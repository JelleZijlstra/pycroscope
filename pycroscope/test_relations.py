# static analysis: ignore

from pycroscope.test_name_check_visitor import TestNameCheckVisitorBase
from pycroscope.test_node_visitor import assert_passes, skip_before


class TestRelations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_mixed_input_sig_generic_relation_does_not_crash(self):
        from typing import Generic

        from typing_extensions import ParamSpec

        P = ParamSpec("P")

        class C(Generic[P]):
            pass

        def takes_int(x: C[[int]]) -> None:
            pass

        def g(x: C[P]) -> None:
            takes_int(x)  # E: incompatible_argument

    @assert_passes()
    def test_single_paramspec_list_and_flat_forms_are_equivalent(self):
        from typing import Generic

        from typing_extensions import ParamSpec, assert_type

        P = ParamSpec("P")

        class Z(Generic[P]):
            pass

        def takes_one_list(x: Z[[int]]) -> None:
            assert_type(x, Z[int])

        def takes_one_flat(x: Z[int]) -> None:
            assert_type(x, Z[[int]])

        def takes_two_list(x: Z[[int, str]]) -> None:
            assert_type(x, Z[int, str])

        def takes_two_flat(x: Z[int, str]) -> None:
            assert_type(x, Z[[int, str]])

        def capybara(
            one_list: Z[[int]],
            one_flat: Z[int],
            two_list: Z[[int, str]],
            two_flat: Z[int, str],
            paramspec: Z[P],
        ) -> None:
            takes_one_list(one_flat)
            takes_one_flat(one_list)
            takes_two_list(two_flat)
            takes_two_flat(two_list)
            takes_one_list(paramspec)  # E: incompatible_argument

    @assert_passes()
    def test_paramspec_after_typevar_uses_list_form(self):
        from typing import Generic, TypeVar

        from typing_extensions import ParamSpec

        T = TypeVar("T")
        P = ParamSpec("P")

        class X(Generic[T, P]):
            pass

        def takes_params(x: X[int, [int]]) -> None:
            pass

        def capybara(good: X[int, [int]], paramspec: X[int, P]) -> None:
            takes_params(good)
            takes_params(paramspec)  # E: incompatible_argument

    @assert_passes()
    def test_paramspec_specialization_requires_list_form_in_mixed_generics(self):
        from typing import Any, Callable, Concatenate, Generic, TypeVar, cast

        from typing_extensions import ParamSpec

        T = TypeVar("T")
        P = ParamSpec("P")

        class A(Generic[T, P]):
            f: Callable[P, int] = cast(Any, "")

        def capybara(
            ok_list: A[int, [int]],
            ok_paramspec: A[int, P],
            ok_concatenate: A[int, Concatenate[str, P]],
            ok_ellipsis: A[int, ...],
            bad_scalar: A[int, int],  # E: invalid_annotation
        ) -> None:
            print(ok_list, ok_paramspec, ok_concatenate, ok_ellipsis, bad_scalar)

    @assert_passes()
    def test_paramspec_specialization_applies_to_callable_attribute(self):
        from typing import Any, Callable, Generic, cast

        from typing_extensions import ParamSpec

        P = ParamSpec("P")

        class C(Generic[P]):
            f: Callable[P, int] = cast(Any, "")

        def takes_list_form(x: C[[int, str, bool]]) -> None:
            x.f(0, "", True)
            x.f("", "", True)  # E: incompatible_argument
            x.f(0, "", "")  # E: incompatible_argument

        def takes_flat_form(x: C[int, str, bool]) -> None:
            x.f(0, "", True)
            x.f("", "", True)  # E: incompatible_argument
            x.f(0, "", "")  # E: incompatible_argument

    @assert_passes()
    def test_paramspec_constructor_inference_matches_list_specialization(self):
        from typing import Callable, Generic, TypeVar

        from typing_extensions import ParamSpec, assert_type

        U = TypeVar("U")
        P = ParamSpec("P")

        class Y(Generic[U, P]):
            f: Callable[P, str]
            prop: U

            def __init__(self, f: Callable[P, str], prop: U) -> None:
                self.f = f
                self.prop = prop

        def callback_a(q: int, /) -> str:
            raise NotImplementedError

        def capybara(x: int) -> None:
            y1 = Y(callback_a, x)
            assert_type(y1, Y[int, [int]])
            y2 = y1.f
            assert_type(y2, Callable[[int], str])

    @assert_passes()
    def test_paramspec_specialization_through_inherited_generic_bases(self):
        from typing import Any, Callable, Generic, TypeVar, cast

        from typing_extensions import ParamSpec

        T = TypeVar("T")
        P1 = ParamSpec("P1")
        P2 = ParamSpec("P2")

        class A(Generic[T, P1]):
            f: Callable[P1, int] = cast(Any, "")
            x: T = cast(Any, "")

        class B(A[T, P1], Generic[T, P1, P2]):
            g: Callable[P2, int] = cast(Any, "")
            x: T = cast(Any, "")

        def capybara(x: B[int, [int, bool], [str]]) -> None:
            x.f(0, True)
            x.f("", True)  # E: incompatible_argument
            x.g("x")
            x.g(0)  # E: incompatible_argument

    @assert_passes()
    def test_paramspec_concatenate_specialization_in_generic_class(self):
        from typing import Any, Callable, Concatenate, Generic, TypeVar, cast

        from typing_extensions import ParamSpec

        T = TypeVar("T")
        P1 = ParamSpec("P1")
        P2 = ParamSpec("P2")

        class A(Generic[T, P1]):
            f: Callable[P1, int] = cast(Any, "")
            x: T = cast(Any, "")

        def takes(x: A[int, Concatenate[int, P2]]) -> None:
            x.f(0, "x")

    @skip_before((3, 12))
    def test_unbounded_tuple_unions(self):
        self.assert_passes("""
            from typing import assert_type

            type Eq0 = tuple[()]
            type Eq1 = tuple[int]
            Ge0 = tuple[int, ...]
            type Ge1 = tuple[int, *Ge0]

            def capybara(eq0: Eq0, eq1: Eq1, ge0: Ge0, ge1: Ge1) -> None:
                eq0_ge1__eq0: Eq0 | Ge1 = eq0
                eq0_ge1__eq1: Eq0 | Ge1 = eq1
                eq0_ge1__ge0: Eq0 | Ge1 = ge0
                eq0_ge1__ge1: Eq0 | Ge1 = ge1
                print(eq0_ge1__eq0, eq0_ge1__eq1, eq0_ge1__ge0, eq0_ge1__ge1)

                assert_type(ge0, Eq0 | Ge1)
            """)

    @assert_passes()
    def test_literal_tuple_equivalent_to_tuple_of_literals(self):
        from typing_extensions import Literal, assert_type

        def capybara(
            literal_tuple: Literal[(("x",),)], tuple_of_literals: tuple[Literal["x"]]
        ) -> None:
            assert_type(literal_tuple, tuple[Literal["x"]])
            assert_type(tuple_of_literals, Literal[(("x",),)])


class TestIntersections(TestNameCheckVisitorBase):
    @assert_passes()
    def test_equivalence(self):
        from typing_extensions import Any, Literal, Never, assert_type

        from pycroscope.extensions import Intersection

        class A:
            x: Any

        class B:
            x: int

        def capybara(
            x: Intersection[Literal[1], Literal[2]], y: Intersection[A, B]
        ) -> None:
            assert_type(x, Never)

            assert_type(y, Intersection[A, B])
            assert_type(y.x, Intersection[int, Any])

    @assert_passes()
    def test_nested(self):
        from typing_extensions import Any, Literal, Never, assert_type

        from pycroscope.extensions import Intersection

        def func() -> None:
            class A:
                x: Any

            class B:
                x: int

            def capybara(
                x: Intersection[Literal[1], Literal[2]], y: Intersection[A, B]
            ) -> None:
                assert_type(x, Never)

                assert_type(y, Intersection[A, B])
                assert_type(y.x, Intersection[int, Any])

    @assert_passes()
    def test_nested_annotation_only_attribute(self):
        from typing_extensions import assert_type

        def func() -> None:
            class A:
                x: int

            a = A()
            assert_type(a.x, int)

    @assert_passes()
    def test_nested_protocol_with_annotation_only_member(self):
        from typing import Protocol

        def func() -> None:
            class P(Protocol):
                x: int

            class Good:
                x: int

            class Bad:
                pass

            def takes_p(arg: P) -> None:
                pass

            takes_p(Good())
            takes_p(Bad())  # E: incompatible_argument

    @assert_passes()
    def test_typed_value_intersections(self):
        from typing_extensions import Never, assert_type, final

        from pycroscope.extensions import Intersection

        class A:
            x: int

        @final
        class B:
            x: str

        def capybara(
            ab: Intersection[A, B],
            int_str: Intersection[int, str],
            a_int: Intersection[A, int],
        ) -> None:
            assert_type(ab, Never)
            assert_type(int_str, Never)
            assert_type(a_int, Never)  # E: inference_failure
