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

        def overlaps_int_and_object(
            func: Intersection[Callable[[int], A], Callable[[object], C]],
        ) -> None:
            assert_type(func(1), Intersection[A, C])

        def overlaps_bool_and_int(
            func: Intersection[Callable[[bool], A], Callable[[int], C]],
        ) -> None:
            assert_type(func(True), Intersection[A, C])

        def overlaps_list_and_object(
            func: Intersection[Callable[[list[int]], A], Callable[[object], C]],
            value: list[int],
        ) -> None:
            assert_type(func(value), Intersection[A, C])

    @assert_passes()
    def test_zero_argument_call_intersects_returns(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        def zero_arg_intersection(
            func: Intersection[Callable[[], A], Callable[[], B]],
        ) -> None:
            assert_type(func(), Intersection[A, B])

    @assert_passes()
    def test_multi_argument_rejection_regions_use_any_rejecting_argument(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class C: ...

        def rejects_second_argument(
            func: Intersection[Callable[[int, int], A], Callable[[object, object], C]],
            y: int | str,
        ) -> None:
            assert_type(func(1, y), C)

        def rejects_either_argument(
            func: Intersection[Callable[[int, str], A], Callable[[object, object], C]],
            x: int | bytes,
            y: str | bytes,
        ) -> None:
            assert_type(func(x, y), C)

    @assert_passes()
    def test_star_args_tuple_shape_splits_argument_list(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        class C: ...

        def tuple_shape_union(
            func: Intersection[
                Callable[[int], A],
                Callable[[int, int], B],
                Callable[[int, int, int], C],
            ],
            args: tuple[int] | tuple[int, int] | tuple[int, int, int],
        ) -> None:
            assert_type(func(*args), A | B | C)

    @assert_passes()
    def test_overload_member_uses_overload_algorithm_for_any_region(self):
        from collections.abc import Callable
        from typing import Any, Protocol, overload

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        class C: ...

        class Overloaded(Protocol):
            @overload
            def __call__(self, value: int) -> A: ...

            @overload
            def __call__(self, value: str) -> B: ...

        def overloaded_any_argument(
            func: Intersection[Overloaded, Callable[[object], C]], value: Any
        ) -> None:
            assert_type(func(value), Intersection[Any, C])

    @assert_passes()
    def test_union_argument_uses_regions(self):
        from collections.abc import Callable

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        class C: ...

        def disjoint_union_argument(
            func: Intersection[
                Callable[[int], A], Callable[[str], B], Callable[[bytes], C]
            ],
            value: int | str | bytes,
        ) -> None:
            assert_type(func(value), A | B | C)

    @assert_passes()
    def test_partitioned_subclass_argument_simplifies_gradual_result(self):
        from collections.abc import Callable
        from typing import Any, Union

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection, Not

        class A: ...

        class B(A): ...

        class C(A): ...

        def partitioned_int_argument(
            func: Intersection[
                Callable[[Intersection[int, Not[bool]]], B], Callable[[bool], C]
            ],
            value: int,
            any_val: Any,
        ) -> None:
            assert_type(func(value), B | C)
            assert_type(func(True), C)
            assert_type(
                func(any_val),
                Union[Intersection[B, C], Intersection[Any, B], Intersection[Any, C]],
            )

    @assert_passes()
    def test_uncovered_argument_part_is_invalid(self):
        from collections.abc import Callable

        from pycroscope.extensions import Intersection

        class A: ...

        class B: ...

        def partially_uncovered_union_argument(
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

        def any_over_simple_domains(
            func: Intersection[Callable[[int], Base], Callable[[str], Base]], value: Any
        ) -> None:
            assert_type(func(value), Base)

        def any_over_container_domains(
            func: Intersection[
                Callable[[list[int]], Base], Callable[[tuple[str]], Base]
            ],
            value: Any,
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

        def keeps_specific_int_parameter(
            func: Intersection[Callable[[int], A], Callable[[Any], B]],
        ) -> None:
            assert_type(func(1), A)

        def keeps_specific_str_parameter(
            func: Intersection[Callable[[str], A], Callable[[Any], B]],
        ) -> None:
            assert_type(func("x"), A)

        def keeps_specific_list_parameter(
            func: Intersection[Callable[[list[int]], A], Callable[[list[Any]], B]],
            value: list[int],
        ) -> None:
            assert_type(func(value), A)

        def keeps_specific_tuple_parameter(
            func: Intersection[
                Callable[[tuple[int, str]], A], Callable[[tuple[Any, Any]], B]
            ],
            value: tuple[int, str],
        ) -> None:
            assert_type(func(value), A)

    @assert_passes()
    def test_list_any_top_materialization_accepts_specific_list(self):
        from collections.abc import Callable
        from typing import Any

        from typing_extensions import assert_type

        from pycroscope.extensions import Intersection

        class A: ...

        class C: ...

        def accepts_specific_list(
            func: Intersection[Callable[[list[Any]], A], Callable[[str], C]],
            value: list[int],
        ) -> None:
            assert_type(func(value), A)

        def accepts_specific_dict(
            func: Intersection[Callable[[dict[str, Any]], A], Callable[[str], C]],
            value: dict[str, int],
        ) -> None:
            assert_type(func(value), A)

    @assert_passes()
    def test_invariant_generic_bottom_materialization_keeps_shape(self):
        from collections.abc import Callable
        from typing import Any

        from pycroscope.extensions import Intersection

        class B: ...

        class C: ...

        def rejects_gradual_list(
            func: Intersection[Callable[[str], B], Callable[[bytes], C]],
            value: list[Any],
        ) -> None:
            func(value)  # E: incompatible_call

        def rejects_gradual_dict(
            func: Intersection[Callable[[str], B], Callable[[bytes], C]],
            value: dict[str, Any],
        ) -> None:
            func(value)  # E: incompatible_call
