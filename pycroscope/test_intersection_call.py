# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestIntersectionCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test_overlapping_domains_intersect_returns(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class C: ...

        def capybara(
            func: Intersection[Callable[[int], A], Callable[[object], C]],
        ) -> None:
            assert_type(func(1), Intersection[A, C])

    @assert_passes()
    def test_union_argument_uses_regions(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        def capybara(
            func: Intersection[Callable[[int], A], Callable[[str], B]], value: int | str
        ) -> None:
            assert_type(func(value), A | B)

    @assert_passes()
    def test_uncovered_argument_part_is_invalid(self):
        from collections.abc import Callable

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        def capybara(
            func: Intersection[Callable[[int], A], Callable[[str], B]],
            value: int | bytes,
        ) -> None:
            func(value)  # E: incompatible_call

    @assert_passes()
    def test_same_return_type_erases_gradual_uncertainty(self):
        from collections.abc import Callable
        from typing import Any

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class Base: ...

        def capybara(
            func: Intersection[Callable[[int], Base], Callable[[str], Base]], value: Any
        ) -> None:
            assert_type(func(value), Base)

    @assert_passes()
    def test_any_parameter_kept_subset_result(self):
        from collections.abc import Callable
        from typing import Any

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        def capybara(
            func: Intersection[Callable[[int], A], Callable[[Any], B]],
        ) -> None:
            assert_type(func(1), A)

    @assert_passes()
    def test_list_any_top_materialization_accepts_specific_list(self):
        from collections.abc import Callable
        from typing import Any

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class C: ...

        def capybara(
            func: Intersection[Callable[[list[Any]], A], Callable[[str], C]],
            value: list[int],
        ) -> None:
            assert_type(func(value), A)

    @assert_passes()
    def test_invariant_generic_bottom_materialization_keeps_shape(self):
        from collections.abc import Callable
        from typing import Any

        from pycroscope.extensions import Intersection

        class B: ...

        class C: ...

        def capybara(
            func: Intersection[Callable[[str], B], Callable[[bytes], C]],
            value: list[Any],
        ) -> None:
            func(value)  # E: incompatible_call
