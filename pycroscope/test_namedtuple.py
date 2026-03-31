# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestNamedTuple(TestNameCheckVisitorBase):
    @assert_passes(allow_import_failures=True)
    def test_namedtuple_after_import_failure(self):
        from typing import Generic, NamedTuple, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Point(NamedTuple):
            x: int
            y: int
            units: str = "meters"

        p = Point(1, 2)
        assert_type(p.x, int)
        assert_type(p.units, str)

        class Property(NamedTuple, Generic[T]):
            name: str
            value: T

        def capybara(x: float) -> None:
            pr = Property("", x)
            assert_type(pr, Property[float])
            assert_type(pr[1], float)
            assert_type(pr.value, float)
            Property[str]("", 3.1)  # E: incompatible_argument

        class DefaultProperty(NamedTuple, Generic[T]):
            name: str
            value: T
            units: str = "meters"

        DefaultProperty[int]("", 3)
        default_pr = DefaultProperty("", 3)
        assert_type(default_pr, DefaultProperty[int])
        assert_type(default_pr.units, str)
        DefaultProperty[int]("")  # E: incompatible_call

        class PointWithName(Point):
            name: str = ""

        pn = PointWithName(1, 2, "")
        assert_type(pn.name, str)

        Point(1)  # E: incompatible_call

        class Point3(NamedTuple):
            _y: int  # E: invalid_namedtuple

        class Location(NamedTuple):
            altitude: float = 0.0
            latitude: float  # E: invalid_namedtuple

        class BadPointWithName(Point):
            x: int = 0  # E: incompatible_override

        class Unit(NamedTuple, object):  # E: invalid_base
            name: str

    @assert_passes(allow_import_failures=True)
    def test_generic_namedtuple_specialization_uses_synthetic_new_signature(self):
        from typing import Generic, NamedTuple, TypeVar

        T = TypeVar("T")

        class Property(NamedTuple, Generic[T]):
            name: str
            value: T

        Property[str]("", "")
        Property[str]("", 3.1)  # E: incompatible_argument

    @assert_passes(run_in_both_module_modes=True)
    def test_namedtuple_subclass_after_import_failure_uses_base_constructor(self):
        from typing import NamedTuple

        from typing_extensions import assert_type

        class Base(NamedTuple):
            x: int
            y: int

        class Child(Base):
            label: str = ""

        def capybara(x: int, y: int):
            child = Child(x, y)
            assert_type(child, Child)
            assert_type(child.x, int)
            child.label.upper()
            assert_type(child[0], int)
            assert_type(child[1], int)

        def g(value: Child) -> None:
            x, y = value
            assert_type(x, int)
            assert_type(y, int)

        def f() -> None:
            child = Child(1, 2)
            child[2]  # E: incompatible_call
            _x, _y, _label = child  # E: bad_unpack
            Child("")  # E: incompatible_call
            Child(1, 2, 3)  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_specialized_namedtuple_base_after_import_failure(self):
        from typing import Generic, NamedTuple, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Base(NamedTuple, Generic[T]):
            value: T

        class Child(Base[int]):
            label: str = ""

        def capybara(value: int) -> None:
            child = Child(value)
            assert_type(child.value, int)
            assert_type(child.label, str)

        Child("x")  # E: incompatible_argument

    @assert_passes()
    def test_namedtuple_attribute_is_immutable(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        p = Point(1)

        def mutate() -> None:
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes()
    def test_namedtuple_attribute_is_immutable_for_intersections(self):
        from typing import NamedTuple

        from pycroscope.extensions import Intersection

        class PointA(NamedTuple):
            x: int

        class PointB(NamedTuple):
            x: int

        def mutate(p: Intersection[PointA, PointB]) -> None:
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_namedtuple_attribute_is_immutable_after_import_failure(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        p = Point(1)

        def capybara() -> None:
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes()
    def test_namedtuple_attribute_is_immutable_for_unions(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        class Mutable:
            x: int

        def mutate(p: Point | Mutable) -> None:
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes()
    def test_local_namedtuple_attribute_is_immutable(self):
        import collections

        def capybara() -> None:
            Point = collections.namedtuple("Point", ["x"])
            p = Point(1)
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes()
    def test_namedtuple_subclass_classmethod_forward_ref(self):
        from collections import namedtuple

        class BasicAuth(namedtuple("BasicAuth", ["login", "password", "encoding"])):
            @classmethod
            def decode(cls, auth_header: str, encoding: str = "latin1") -> "BasicAuth":
                return cls(auth_header, "", encoding)

    @assert_passes()
    def test_local_namedtuple_signature_lookup_does_not_crash(self):
        import inspect
        from typing import NamedTuple

        from typing_extensions import assert_type

        class Params(NamedTuple):
            x: int
            y: float = 0.0

        def capybara() -> None:
            params = Params(1)
            inspect.signature(Params)
            assert_type(params.x, int)
            assert_type(params.y, float)

    @assert_passes(allow_import_failures=True)
    def test_except_local_namedtuple_signature_lookup_does_not_crash(self):
        import inspect
        from typing import NamedTuple

        try:
            from definitely_missing_namedtuple_module import Params
        except ImportError:

            class Params(NamedTuple):  # type: ignore[no-redef]
                x: int
                y: float = 0.0

        def capybara() -> None:
            params = Params(1)
            inspect.signature(Params)
            params.x
            params.y

    @assert_passes(run_in_both_module_modes=True)
    def test_namedtuple_tuple_operations(self):
        from typing import NamedTuple

        from typing_extensions import assert_type

        class Point(NamedTuple):
            x: int
            y: int
            units: str = "meters"

        def f(x: int, y: int, units: str) -> None:
            p = Point(x, y, units)
            assert_type(p[0], int)
            assert_type(p[1], int)
            assert_type(p[2], str)
            assert_type(p[-1], str)
            assert_type(p[-2], int)
            assert_type(p[-3], int)

            p[3]  # E: incompatible_call
            p[-4]  # E: incompatible_call
            p[0] = x  # E: unsupported_operation
            del p[0]  # E: unsupported_operation

            x1, y1, units1 = p
            assert_type(x1, int)
            assert_type(units1, str)
            _x2, _y2 = p  # E: bad_unpack
            _x3, _y3, _units3, _other = p  # E: bad_unpack

        class PointWithName(Point):
            name: str = ""

        def g(x: int, y: int, units: str) -> None:
            pn = PointWithName(x, y, units)
            x4, y4, units4 = pn
            assert_type(x4, int)
            assert_type(units4, str)

    @assert_passes(run_in_both_module_modes=True)
    def test_namedtuple_is_assignable_to_exact_tuple(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int
            y: int
            units: str = "meters"

        def capybara(p: Point) -> tuple[int, int, str]:
            exact: tuple[int, int, str] = p
            return exact

    @assert_passes()
    def test_namedtuple(self):
        import collections

        typ = collections.namedtuple("typ", "foo bar")

        def fn():
            t = typ(1, 2)
            print(t.baz)  # E: undefined_attribute

    @assert_passes()
    def test_local_namedtuple(self):
        import collections

        from typing_extensions import Literal

        from pycroscope.value import SyntheticClassObjectValue, TypedValue

        def capybara():
            typ = collections.namedtuple("typ", "foo bar")
            assert_is_value(
                typ,
                SyntheticClassObjectValue(
                    "typ", TypedValue(f"{__name__}.capybara.<locals>.typ")
                ),
            )
            t = typ(1, 2)
            assert_type(t.foo, Literal[1])
            assert_type(t.bar, Literal[2])
            print(t.baz)  # E: undefined_attribute
            typ(1, 2, 3)  # E: incompatible_call
            typ(1)  # E: incompatible_call
