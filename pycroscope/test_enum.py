# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .value import AnySource, AnyValue, assert_is_value


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

    @skip_before((3, 11))
    @assert_passes(run_in_both_module_modes=True)
    def test_member_and_nonmember_helpers(self):
        from enum import Enum, member, nonmember

        from typing_extensions import Literal, assert_type

        class Example(Enum):
            a = member(1)
            b = nonmember(2)

            @member
            def c(self) -> None:
                raise NotImplementedError

        assert_type(Example.a, Literal[Example.a])
        x: int = Example.b
        assert_type(Example.c, Literal[Example.c])

    @assert_passes(run_in_both_module_modes=True)
    def test_annotated_nonmember_attributes_can_be_assigned_in_init(self):
        from enum import Enum

        class Pet(Enum):
            genus: str
            species: str

            CAT = "felis", "catus"
            DOG = "canis", "lupus"

            def __init__(self, genus: str, species: str) -> None:
                self.genus = genus
                self.species = species

        def capybara(pet: Pet):
            assert_type(pet.genus, str)
            assert_type(pet.species, str)

            # TODO: this should be an error, the attribute doesn't exist on the class
            assert_type(Pet.genus, str)

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
                assert_type(x, Literal[X.a])
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
            capybara("capybara")
            capybara("agouti")
            capybara("porcupine")  # E: incompatible_argument
