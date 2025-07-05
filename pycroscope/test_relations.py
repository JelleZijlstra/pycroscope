# static analysis: ignore

from pycroscope.test_name_check_visitor import TestNameCheckVisitorBase
from pycroscope.test_node_visitor import assert_passes, skip_before


class TestRelations(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_unbounded_tuple_unions(self):
        self.assert_passes(
            """
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

                assert_type(ge0, Eq0 | Ge1)
            """
        )


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
