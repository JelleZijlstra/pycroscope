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
    def test_implicit_receiver_is_bound_self_in_method_body(self):
        from typing_extensions import Self, assert_type

        class Shape:
            def set_scale(self, scale: float) -> Self:
                assert_type(self, Self)
                self.scale = scale
                return self

            @classmethod
            def from_config(cls) -> Self:
                assert_type(cls, type[Self])
                return cls()

    @assert_passes()
    def test_self_annotated_receiver_attribute_assignment(self):
        from typing_extensions import Self

        class Box:
            def set_value(self: Self, value: int) -> None:
                self.value = value

        def capybara(box: Box):
            assert_type(box.value, int)

    @assert_passes()
    def test_self_annotated_receiver_setattr(self):
        from typing_extensions import Self

        class Box:
            def set_value(self: Self, value: int) -> None:
                setattr(self, "value", value)

        def capybara(box: Box):
            assert_type(box.value, int)

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
    def test_self_attribute_assignment_specializes_for_subclass(self):
        from dataclasses import dataclass
        from typing import Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        @dataclass
        class LinkedList(Generic[T]):
            value: T
            next: Self | None = None

        @dataclass
        class OrdinalLinkedList(LinkedList[int]):
            def ordinal_value(self) -> str:
                return str(self.value)

        xs = OrdinalLinkedList(
            value=1, next=LinkedList[int](value=2)  # E: incompatible_argument
        )

        if xs.next is not None:
            xs.next = OrdinalLinkedList(value=3, next=None)
            xs.next = LinkedList[int](value=3, next=None)  # E: incompatible_assignment

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
    def test_self_preserved_inside_generic_receiver_method(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self, assert_type

        T = TypeVar("T", bound="Model")

        class Query(Generic[T]):
            def filter(self) -> "Query[T]":
                raise NotImplementedError

        class Model:
            @classmethod
            def select(cls) -> Query[Self]:
                raise NotImplementedError

            @classmethod
            def select_valid(cls) -> Query[Self]:
                q = cls.select()
                assert_type(q, Query[Self])
                q2 = q.filter()
                assert_type(q2, Query[Self])
                return q2

        class SubModel(Model):
            pass

        def capybara() -> None:
            assert_type(SubModel.select_valid(), Query[SubModel])

    @assert_passes()
    def test_self_preserved_inside_generic_receiver_method_without_self_return(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self, assert_type

        T = TypeVar("T", bound="Model")

        class Getter(Generic[T]):
            def get_one(self) -> T | None:
                raise NotImplementedError

        class Model:
            @classmethod
            def getter(cls) -> Getter[Self]:
                raise NotImplementedError

            @classmethod
            def get_one(cls) -> Self | None:
                getter = cls.getter()
                assert_type(getter, Getter[Self])
                result = getter.get_one()
                assert_type(result, Self | None)
                return result

        class SubModel(Model):
            pass

        def capybara() -> None:
            assert_type(SubModel.get_one(), SubModel | None)

    @assert_passes()
    def test_self_preserved_inside_imported_generic_receiver_method(self):
        import sys
        import types

        from typing_extensions import Self, assert_type

        mod = types.ModuleType("_runtime_self_inline")
        exec(
            "\n".join(
                [
                    "from typing import Generic, TypeVar",
                    "",
                    'T = TypeVar("T")',
                    "",
                    "class Query(Generic[T]):",
                    '    def filter(self) -> "Query[T]":',
                    "        raise NotImplementedError",
                    "",
                    "class Getter(Generic[T]):",
                    "    def get_one(self) -> T | None:",
                    "        raise NotImplementedError",
                ]
            ),
            mod.__dict__,
        )
        sys.modules[mod.__name__] = mod

        from _runtime_self_inline import Getter, Query

        class Model:
            @classmethod
            def select(cls) -> Query[Self]:
                raise NotImplementedError

            @classmethod
            def getter(cls) -> Getter[Self]:
                raise NotImplementedError

            @classmethod
            def select_valid(cls) -> Query[Self]:
                q = cls.select()
                assert_type(q.filter(), Query[Self])
                return q.filter()

            @classmethod
            def get_one(cls) -> Self | None:
                getter = cls.getter()
                assert_type(getter.get_one(), Self | None)
                return getter.get_one()

        class SubModel(Model):
            pass

        def capybara() -> None:
            assert_type(SubModel.select_valid(), Query[SubModel])
            assert_type(SubModel.get_one(), SubModel | None)

    @assert_passes()
    def test_qualified_self(self):
        import typing_extensions as typing_ext

        class Node:
            next: typing_ext.Self | None = None

            def get(self) -> typing_ext.Self | None:
                return self.next

        class Child(Node):
            pass

        def capybara(node: Node, child: Child):
            assert_type(node.next, Node | None)
            assert_type(node.get(), Node | None)
            assert_type(child.next, Child | None)
            assert_type(child.get(), Child | None)

    @assert_passes()
    def test_generic_classmethod_preserves_type_arguments(self):
        from typing import Any, Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Box(Generic[T]):
            @classmethod
            def make(cls) -> Self:
                return cls()

            @classmethod
            def get_class(cls) -> type[Self]:
                return cls

        def capybara() -> None:
            assert_type(Box.make(), Box[Any])
            assert_type(Box[int].make(), Box[int])
            assert_type(Box[type[int]].make(), Box[type[int]])
            assert_type(Box[int].get_class(), type[Box[int]])

    @assert_passes()
    def test_classmethod_self_attribute_assignment(self):
        from typing_extensions import Self, assert_type

        class Base:
            current: Self | None = None

            @classmethod
            def set_current(cls, value: Self | None) -> None:
                cls.current = value

            @classmethod
            def get_current(cls) -> Self | None:
                return cls.current

        class Child(Base):
            pass

        def capybara(child: Child) -> None:
            Child.set_current(child)
            assert_type(Child.get_current(), Child | None)

    @assert_passes()
    def test_self_union_is_stable_in_method_body(self):
        from typing_extensions import Self, assert_type

        class Base:
            def echo(self, value: Self | None) -> Self | None:
                assert_type(value, Self | None)
                return value

        class Child(Base):
            pass

        def capybara(child: Child) -> None:
            assert_type(child.echo(child), Child | None)

    @assert_passes()
    def test_generic_descriptor_preserves_owner_self(self):
        from typing import Any, Generic, TypeVar, cast, overload

        from typing_extensions import Self, assert_type

        T = TypeVar("T")

        class Field(Generic[T]):
            def __init__(self, name: str | None = None) -> None:
                pass

            @overload
            def __get__(self, obj: None, objtype: object = None) -> Self: ...

            @overload
            def __get__(self, obj: object, objtype: object = None) -> T: ...

            def __get__(self, obj: object | None, objtype: object = None) -> T | Self:
                return self if obj is None else cast(Any, obj)

        class Model:
            parent = Field[Self | None]("parent_id")

            def get(self) -> Self | None:
                return self.parent

        def capybara(model: Model) -> None:
            assert_type(model.parent, Model | None)

    @assert_passes(allow_import_failures=True)
    def test_generic_descriptor_preserves_owner_self_without_runtime_module(self):
        from typing import Any, Generic, TypeVar, cast, overload

        from typing_extensions import Self

        T = TypeVar("T")

        class Field(Generic[T]):
            def __init__(self, name: str | None = None) -> None:
                pass

            @overload
            def __get__(self, obj: None, objtype: object = None) -> Self: ...

            @overload
            def __get__(self, obj: object, objtype: object = None) -> T: ...

            def __get__(self, obj: object | None, objtype: object = None) -> T | Self:
                return self if obj is None else cast(Any, obj)

        class Model:
            parent = Field[Self | None]("parent_id")

            def get(self) -> Self | None:
                return self.parent

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

        import typing_extensions as typing_ext
        from typing_extensions import Self

        def foo(bar: Self) -> Self: ...  # E: invalid_self_usage

        # E: invalid_self_usage
        def qualified_foo(bar: typing_ext.Self) -> typing_ext.Self: ...

        _bar: Self  # E: invalid_self_usage
        TupleSelf: TypeAlias = tuple[Self]  # E: invalid_self_usage
        AliasTupleSelf: TAlias = tuple[Self]  # E: invalid_self_usage

        TFoo = TypeVar("TFoo", bound="Foo")

        class Foo:
            # E: invalid_self_usage
            def has_existing_self_annotation(self: TFoo) -> Self:
                raise NotImplementedError

            @staticmethod
            def make() -> Self:  # E: invalid_self_usage
                raise NotImplementedError

            @staticmethod
            def return_parameter(foo: Self) -> Self:  # E: invalid_self_usage
                raise NotImplementedError

        class Meta(type):
            def __new__(cls, *args: Any) -> Self:  # E: invalid_self_usage
                raise NotImplementedError

    @assert_passes(run_in_both_module_modes=True)
    def test_invalid_self_in_nested_class_body_method(self):
        from typing_extensions import Self

        class Foo:
            try:

                @staticmethod
                def make() -> Self:  # E: invalid_self_usage
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
