# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestPatma(TestNameCheckVisitorBase):
    @assert_passes()
    def test_singletons(self):
        from typing import Literal

        def capybara(x: Literal[True, False, None]):
            match x:
                case True:
                    assert_type(x, Literal[True])
                case _:
                    assert_type(x, Literal[False] | None)

    @assert_passes()
    def test_value(self):
        from typing import Literal

        from pycroscope.tests import assert_never

        def capybara(x: int):
            match x:
                case None:  # E: impossible_pattern
                    assert_never(x)
                case "x":  # E: impossible_pattern
                    assert_is_value(x, NO_RETURN_VALUE)
                    assert_never(x)
                case 3:
                    assert_type(x, Literal[3])
                case _ if x == 4:
                    assert_type(x, Literal[4])
                case _:
                    assert_type(x, int)

    @assert_passes()
    def test_sequence(self):
        import collections.abc
        from typing import Tuple

        def capybara(seq: Tuple[int, ...], obj: object):
            match seq:
                case [1, 2, 3]:
                    assert_is_value(
                        seq,
                        make_simple_sequence(
                            tuple, [TypedValue(int), TypedValue(int), TypedValue(int)]
                        ),
                    )
                case [1, *x]:
                    assert_type(x, list[int])

            match obj:
                case [*x]:
                    assert_is_value(
                        obj, TypedValue(collections.abc.Sequence), skip_annotated=True
                    )
                    assert_is_value(
                        x, GenericValue(list, [AnyValue(AnySource.generic_argument)])
                    )

            assert_type(seq[0], int)
            match seq[0]:
                case [1, 2, 3]:  # E: impossible_pattern
                    pass

    @assert_passes()
    def test_sequence_length_narrowing(self):
        from typing import TypeAlias

        from typing_extensions import Unpack

        Input: TypeAlias = (
            tuple[int] | tuple[str, str] | tuple[int, Unpack[tuple[str, ...]], int]
        )

        def capybara(val: Input) -> None:
            match val:
                case (x,):
                    assert_type(val, tuple[int])
                case (x, y):
                    assert_type(val, tuple[str, str] | tuple[int, int])
                case (x, y, z):
                    assert_type(val, tuple[int, str, int])

    @assert_passes()
    def test_or(self):
        from typing import Literal

        def capybara(obj: object):
            match obj:
                case 1 | 2:
                    assert_type(obj, Literal[1, 2])
                case (3 as x) | (4 as x):
                    assert_type(x, Literal[3, 4])

    @assert_passes()
    def test_mapping(self):
        from typing import Literal

        def capybara(obj: object):
            match {1: 2, 3: 4, 5: 6}:
                case {1: x}:
                    assert_type(x, Literal[2])
                case {3: 4, **x}:
                    assert_is_value(
                        x,
                        DictIncompleteValue(
                            dict,
                            [
                                KVPair(KnownValue(1), KnownValue(2)),
                                KVPair(KnownValue(5), KnownValue(6)),
                            ],
                        ),
                    )

    @assert_passes()
    def test_class_pattern(self):
        from typing import Literal

        class NotMatchable:
            x: str

        class MatchArgs:
            __match_args__ = ("x", "y")
            x: str
            y: int

        def capybara(obj: object):
            match obj:
                case int(1, 2):  # E: bad_match
                    pass
                case int(2):
                    assert_type(obj, Literal[2])
                case int("x"):  # E: impossible_pattern
                    pass
                case int():
                    assert_type(obj, int)
                case NotMatchable(x="x"):
                    pass
                case NotMatchable("x"):  # E: bad_match
                    pass
                case NotMatchable():
                    pass
                case MatchArgs("x", 1 as y):
                    assert_type(y, Literal[1])
                case MatchArgs(x) if x == "x":
                    assert_type(x, Literal["x"])
                case MatchArgs(x):
                    assert_type(x, str)
                case MatchArgs("x", x="x"):  # E: bad_match
                    pass
                case MatchArgs(1, 2, 3):  # E: bad_match
                    pass

    @assert_passes()
    def test_bool_narrowing(self):
        from typing import Literal

        class X:
            true = True

        def capybara(b: bool):
            match b:
                # Make sure we hit the MatchValue case, not MatchSingleton
                case X.true:
                    assert_type(b, Literal[True])
                case _ as b2:
                    assert_type(b, Literal[False])
                    assert_type(b2, Literal[False])

    @assert_passes()
    def test_bool_narrowing_singleton(self):
        from typing import Literal

        def capybara(b: bool):
            match b:
                case True:
                    assert_type(b, Literal[True])
                case _ as b2:
                    assert_type(b, Literal[False])
                    assert_type(b2, Literal[False])

    @assert_passes()
    def test_enum_narrowing(self):
        from enum import Enum
        from typing import Literal

        class Planet(Enum):
            mercury = 1
            venus = 2
            earth = 3

        def capybara(p: Planet):
            match p:
                case Planet.mercury:
                    assert_type(p, Literal[Planet.mercury])
                case Planet.venus:
                    assert_type(p, Literal[Planet.venus])
                case _ as p2:
                    assert_type(p2, Literal[Planet.earth])
                    assert_type(p, Literal[Planet.earth])

    @assert_passes()
    def test_exhaustive(self):
        def f(x: object) -> int:
            match x:
                case _:
                    return 1

        def g(x: object) -> int:  # E: missing_return
            match x:
                case _ if x == 2:
                    return 1

        def some_func() -> object:
            return 1

        def h() -> int:
            match some_func():
                case _:
                    return 1

        def i(x: bool) -> int:
            match x:
                case True:
                    return 1
                case False:
                    return 2

    @assert_passes()
    def test_match_narrows_cases_and_guard_bindings(self):
        from typing_extensions import assert_type

        def capybara(x: int | str | None) -> None:
            match x:
                case int() as i if i > 0:
                    assert_type(x, int)
                    i.bit_length()
                case str() as s:
                    assert_type(x, str)
                    assert_type(s, str)
                case None:
                    assert_type(x, None)

    @assert_passes()
    def test_match_sequence_and_mapping_patterns(self):
        from typing_extensions import assert_type

        def capybara(x: tuple[int, str] | dict[str, int]) -> None:
            match x:
                case (a, b):
                    assert_type(x, tuple[int, str])
                    assert_type(a, int)
                    assert_type(b, str)
                case {"count": count}:
                    assert_type(x, dict[str, int])
                    assert_type(count, int)
                case _:
                    assert_type(x, dict[str, int])

    @assert_passes()
    def test_match_class_pattern_fallthrough_after_irrefutable_keyword(self):
        from typing_extensions import assert_type

        class Box:
            value: int

        def capybara(x: int | Box) -> None:
            match x:
                case Box(value=value):
                    assert_type(x, Box)
                    assert_type(value, int)
                case _:
                    assert_type(x, int)

    @assert_passes()
    def test_reassign_in_tuple(self):
        def f(x: int | str) -> None:
            match (x,):
                case (int() as x,):
                    assert_type(x, int)

    @assert_passes()
    def test_match_body_narrows_after_prior_case(self):
        class E:
            pass

        def capybara(result: int | E) -> list[tuple[bool, int]]:
            match result:
                case E():
                    return []
                case _:
                    assert_type(result, int)
                    return [(True, result)]

    @assert_passes()
    def test_exhaustive_match_does_not_leave_scope(self):
        def capybara(x: object | None) -> int:
            if x is not None:
                match x:
                    case int():
                        val = 1
                    case _:
                        val = 2
            else:
                val = 3
            return val

    @assert_passes()
    def test_class_pattern_with_subpatterns_does_not_negatively_narrow(self):
        class E:
            x: int

        def capybara(v: int | E) -> int:
            match v:
                case E(x=1):
                    return 0
                case _:
                    assert_type(v, int | E)
                    return 1
