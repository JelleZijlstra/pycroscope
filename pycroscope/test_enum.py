# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import AnySource, AnyValue, KnownValue, assert_is_value


class TestEnum(TestNameCheckVisitorBase):
    @assert_passes()
    def test_functional(self):
        from enum import Enum, IntEnum

        def capybara():
            X = Enum("X", ["a", "b", "c"])
            assert_type(X, type[Enum])

            IE = IntEnum("X", ["a", "b", "c"])
            assert_type(IE, type[Enum])

    @assert_passes()
    def test_call(self):
        from enum import Enum

        class X(Enum):
            a = 1
            b = 2

        def capybara():
            assert_type(X(1), X)
            # This should be an error, but the typeshed
            # stubs are too lenient.
            assert_type(X(None), X)

    @assert_passes()
    def test_iteration(self):
        from enum import Enum, IntEnum
        from typing import Type

        class X(Enum):
            a = 1
            b = 2

        class MySubclass(str, Enum):
            pass

        def capybara(
            enum_t: Type[Enum], int_enum_t: Type[IntEnum], subclass_t: Type[MySubclass]
        ):
            for x in X:
                assert_type(x, X)

            for et in enum_t:
                assert_type(et, Enum)

            for iet in int_enum_t:
                assert_type(iet, IntEnum)

            for st in subclass_t:
                assert_type(st, MySubclass)

    @assert_passes()
    def test_duplicate_enum_member(self):
        import enum

        class Foo(enum.Enum):
            a = 1
            b = 1  # E: duplicate_enum_member

    @assert_passes()
    def test_value_assignment_with_nonstandard_receiver_name(self):
        import enum

        class Foo(enum.Enum):
            _value_: int
            a = 1

            def __init__(this, value: object) -> None:  # E: method_first_arg
                this._value_ = value  # E: invalid_annotation


class TestNarrowing(TestNameCheckVisitorBase):
    @assert_passes()
    def test_exhaustive(self):
        from enum import Enum

        from typing_extensions import Literal, assert_never

        class X(Enum):
            a = 1
            b = 2

        def capybara_eq(x: X):
            if x == X.a:
                assert_type(x, Literal[X.a])
            else:
                assert_type(x, Literal[X.b])

        def capybara_is(x: X):
            if x is X.a:
                assert_type(x, Literal[X.a])
            else:
                assert_type(x, Literal[X.b])

        def capybara_in_list(x: X):
            if x in [X.a]:
                assert_type(x, Literal[X.a])
            else:
                assert_type(x, Literal[X.b])

        def capybara_in_tuple(x: X):
            if x in (X.a,):
                assert_type(x, Literal[X.a])
            else:
                assert_type(x, Literal[X.b])

        def test_multi_in(x: X):
            if x in (X.a, X.b):
                assert_type(x, Literal[X.a, X.b])
            else:
                assert_never(x)

        def whatever(x):
            if x == X.a:
                assert_is_value(x, KnownValue(X.a))
                return
            assert_is_value(x, AnyValue(AnySource.unannotated))


class TestEnumName(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import enum

        from pycroscope.extensions import EnumName

        class Rodent(enum.IntEnum):
            capybara = 1
            agouti = 2

        def capybara(x: EnumName[Rodent]):
            pass

        def needs_str(s: str):
            pass

        def caller(r: Rodent, s: str):
            capybara(s)  # E: incompatible_argument
            capybara(r)  # E: incompatible_argument
            needs_str(r.name)  # OK
            capybara(r.name)
