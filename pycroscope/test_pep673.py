# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestPEP673(TestNameCheckVisitorBase):
    @assert_passes()
    def test_instance_attribute(self):
        from typing_extensions import Self

        class X:
            parent: Self

            @property
            def prop(self) -> Self:
                raise NotImplementedError

        class Y(X):
            pass

        def capybara(x: X, y: Y):
            assert_type(x.parent, X)
            assert_type(y.parent, Y)

            assert_type(x.prop, X)
            assert_type(y.prop, Y)

    @assert_passes()
    def test_method(self):
        from typing_extensions import Self

        class X:
            def ret(self) -> Self:
                return self

            @classmethod
            def from_config(cls) -> Self:
                return cls()

        class Y(X):
            pass

        def capybara(x: X, y: Y):
            assert_type(x.ret(), X)
            assert_type(y.ret(), Y)

            assert_type(X.from_config(), X)
            assert_type(Y.from_config(), Y)

    @assert_passes()
    def test_invalid_self_return(self):
        from typing_extensions import Self

        class Shape:
            def method2(self) -> Self:
                return Shape()  # E: incompatible_return_value

            @classmethod
            def cls_method2(cls) -> Self:
                return Shape()  # E: incompatible_return_value

    @assert_passes()
    def test_self_return_via_typevar_function(self):
        from dataclasses import dataclass, replace

        from typing_extensions import Self

        @dataclass
        class Box:
            value: int

            def with_value(self, value: int) -> Self:
                return replace(self, value=value)

    @assert_passes()
    def test_parameter_type(self):
        from typing import Callable

        from typing_extensions import Self

        class Shape:
            def difference(self, other: Self) -> float:
                raise NotImplementedError

            def apply(self, f: Callable[[Self], None]) -> None:
                raise NotImplementedError

        class Circle(Shape):
            pass

        def difference():
            s = Shape()
            s.difference(s)
            s.difference(1.0)  # E: incompatible_argument
            s.difference(Circle())

            c = Circle()
            c.difference(c)
            c.difference(s)  # E: incompatible_argument
            c.difference("x")  # E: incompatible_argument

        def takes_shape(s: Shape) -> None:
            pass

        def takes_circle(c: Circle) -> None:
            pass

        def takes_int(i: int) -> None:
            pass

        def apply():
            s = Shape()
            c = Circle()
            s.apply(takes_shape)
            s.apply(takes_circle)  # E: incompatible_argument
            s.apply(takes_int)  # E: incompatible_argument
            c.apply(takes_shape)
            c.apply(takes_circle)
            c.apply(takes_int)  # E: incompatible_argument

    @assert_passes()
    def test_linked_list(self):
        from dataclasses import dataclass
        from typing import Generic, Optional, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        @dataclass
        class LinkedList(Generic[T]):
            value: T
            next: Optional[Self] = None

        @dataclass
        class OrdinalLinkedList(LinkedList[int]):
            pass

        def capybara(o: OrdinalLinkedList):
            # Unfortunately we don't fully support the example in
            assert_type(o.next, OrdinalLinkedList | None)

    @assert_passes()
    def test_generic(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Container(Generic[T]):
            value: T

            def set_value(self, value: T) -> Self:
                return self

        def capybara(c: Container[int]):
            assert_type(c.value, int)
            assert_type(c.set_value(3), Container[int])

    @assert_passes()
    def test_classvar(self):
        from typing import ClassVar, List

        from typing_extensions import Self

        class Registry:
            children: ClassVar[List[Self]]

        def capybara():
            assert_type(Registry.children, list[Registry])

    @assert_passes(run_in_both_module_modes=True)
    def test_invalid_self_contexts(self):
        from typing import Any, TypeAlias, TypeVar
        from typing import TypeAlias as TAlias

        from typing_extensions import Self

        def foo(bar: Self) -> Self: ...  # E: invalid_annotation

        _bar: Self  # E: invalid_annotation
        TupleSelf: TypeAlias = tuple[Self]  # E: invalid_annotation
        AliasTupleSelf: TAlias = tuple[Self]  # E: invalid_annotation

        TFoo = TypeVar("TFoo", bound="Foo")

        class Foo:
            # E: invalid_annotation
            def has_existing_self_annotation(self: TFoo) -> Self:
                raise NotImplementedError

            @staticmethod
            def make() -> Self:  # E: invalid_annotation
                raise NotImplementedError

            @staticmethod
            def return_parameter(foo: Self) -> Self:  # E: invalid_annotation
                raise NotImplementedError

        class Meta(type):
            def __new__(cls, *args: Any) -> Self:  # E: invalid_annotation
                raise NotImplementedError

    @assert_passes(run_in_both_module_modes=True)
    def test_invalid_self_in_nested_class_body_method(self):
        from typing_extensions import Self

        class Foo:
            try:

                @staticmethod
                def make() -> Self:  # E: invalid_annotation
                    raise NotImplementedError

            finally:
                pass

    @assert_passes(allow_import_failures=True)
    def test_invalid_self_bases(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Box(Generic[T]):
            pass

        class BadGeneric(Box[Self]): ...  # E: invalid_base

        class BadSelf(Self): ...  # E: invalid_base

    @assert_passes()
    def test_stub(self):
        def capybara():
            from _pycroscope_tests.self import X, Y

            x = X()
            y = Y()
            assert_type(x, X)
            assert_type(y, Y)

            def want_x(x: X):
                pass

            def want_y(y: Y):
                pass

            want_x(x.ret())
            want_y(y.ret())

            want_x(X.from_config())
            want_y(Y.from_config())

    @assert_passes()
    def test_typeshed_self(self):
        def capybara():
            from _pycroscope_tests.tsself import X

            x = X()
            assert_type(x, X)
