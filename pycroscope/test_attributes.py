# static analysis: ignore
from typing import Dict, Union

import pytest

from .extensions import LiteralOnly
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import (
    assert_passes,
    only_before,
    skip_before,
    skip_if_not_installed,
)
from .type_object import normalize_synthetic_descriptor_attribute
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CustomCheckExtension,
    GenericValue,
    KnownValue,
    SelfT,
    TypeVarMap,
    assert_is_value,
)

_global_dict: Dict[Union[int, str], bytes] = {}
EXPECTED_ASCII_TYPEVARS = TypeVarMap(
    typevars={
        SelfT: AnnotatedValue(
            KnownValue("ascii"), [CustomCheckExtension(LiteralOnly())]
        )
    }
)


def test_normalize_synthetic_descriptor_attribute_empty_args() -> None:
    assert normalize_synthetic_descriptor_attribute(
        GenericValue(staticmethod, [])
    ) == AnyValue(AnySource.inference)
    assert normalize_synthetic_descriptor_attribute(
        GenericValue(classmethod, [])
    ) == AnyValue(AnySource.inference)


class TestAttributes(TestNameCheckVisitorBase):
    @skip_if_not_installed("attr")
    @assert_passes()
    def test_attrs(self):
        import attr

        @attr.s(frozen=True)
        class Capybara(object):
            value = attr.ib()
            int_value = attr.ib(type=int)

        def kerodon():
            c = Capybara(42, 43)
            assert_is_value(c.value, AnyValue(AnySource.unannotated))
            assert_type(c.int_value, int)

    @assert_passes()
    def test_attribute_in_annotations(self):
        class Capybara:
            capybara_id: int
            kerodon_id: object = None

        def capybara():
            assert_type(Capybara.kerodon_id, object)
            c = Capybara()
            return c.capybara_id

    @assert_passes()
    def test_attribute_in_base_class(self):
        from typing import Optional

        from typing_extensions import Literal

        class Capybara:
            capybara_id: Optional[int] = None

            @classmethod
            def clsmthd(cls):
                assert_type(cls.capybara_id, int | None)

        class DefiniteCapybara(Capybara):
            capybara_id = 3

            @classmethod
            def clsmthd(cls):
                assert_type(cls.capybara_id, Literal[3])

        def capybara():
            assert_type(Capybara().capybara_id, int | None)
            assert_type(Capybara.capybara_id, int | None)
            assert_type(DefiniteCapybara().capybara_id, Literal[3])
            assert_type(DefiniteCapybara.capybara_id, Literal[3])

    @assert_passes()
    def test_subclass_special_attributes(self):
        class Base:
            pass

        def capybara(cls: type[Base]) -> None:
            assert_is_value(cls.__class__, KnownValue(type))
            assert_type(cls.__bases__, tuple[type[object], ...])

    @assert_passes()
    def test_readonly_attribute_assignment_and_deletion(self):
        from typing import ClassVar

        from typing_extensions import Final, ReadOnly

        class Base:
            id: ReadOnly[int]

            def __init__(self, value: int) -> None:
                self.id = value
                self.id = value + 1

        class Child(Base):
            def __init__(self, value: int) -> None:
                super().__init__(value)
                self.id = value  # E: incompatible_assignment

        class Inline:
            def __init__(self, name: str) -> None:
                self.name: ReadOnly[str] = name
                self.name = name.upper()

            def rename(self, name: str) -> None:
                self.name = name  # E: incompatible_assignment
                del self.name  # E: incompatible_assignment

        class Config:
            version: ReadOnly[int] = 1
            token: ClassVar[ReadOnly[str]]

            def __init_subclass__(cls) -> None:
                cls.token = cls.__name__

            def bump(self) -> None:
                self.version = 2  # E: incompatible_assignment

        class FinalAttr:
            def __init__(self) -> None:
                self.value: Final[int] = 1

            def clear(self) -> None:
                del self.value  # E: incompatible_assignment

        def mutate(inline: Inline) -> None:
            inline.name = "y"  # E: incompatible_assignment
            del inline.name  # E: incompatible_assignment

        inline = Inline("x")
        Config.version = 2  # E: incompatible_assignment
        Config.token = "override"  # E: incompatible_assignment
        mutate(inline)
        print(Base, Child, Config, FinalAttr, inline)

    @assert_passes()
    def test_known_value_hook(self):
        from typing_extensions import Literal

        from pycroscope.test_config import SPECIAL_STRING

        def capybara():
            assert_type(SPECIAL_STRING.special, Literal["special"])

    @assert_passes()
    def test_default_known_attribute_hook(self):
        import sys
        import types
        from typing import Any

        def capybara() -> None:
            assert_is_value(Any.whatever, AnyValue(AnySource.explicit))
            assert_type(sys.modules, dict[str, types.ModuleType])

    @assert_passes()
    def test_generic(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class X(Generic[T]):
            x: T

        class Child1(X[str]):
            pass

        class Child2(X[U]):
            pass

        def capybara(obj: X[int], c1: Child1, c2: Child2[bool]) -> None:
            assert_type(obj.x, int)
            assert_type(c1.x, str)
            assert_type(c2.x, bool)

    @assert_passes()
    def test_attribute_union(self):
        class A:
            x: int

        class B:
            x: str

        class C(B):
            y: bytes

        def capybara() -> None:
            assert_type(A().x, int)
            assert_type(C().y, bytes)
            assert_type(C().x, str)

    @assert_passes()
    def test_name_py3(self):
        from typing_extensions import Literal

        def capybara():
            assert_type(KnownValue.__name__, Literal["KnownValue"])

    @assert_passes()
    def test_attribute_type_inference(self):
        from pycroscope.tests import PropertyObject

        class Capybara(object):
            def init(self, aid):
                self.answer = PropertyObject(aid)

            def tree(self):
                assert_type(self.answer, PropertyObject)
                return []

    @assert_passes()
    def test_property_on_unhashable_object(self):
        class CustomDescriptor(object):
            __hash__ = None

            def __get__(self, obj, typ):
                if obj is None:
                    return self
                return 3

        class Unhashable(object):
            __hash__ = None

            prop = CustomDescriptor()

        def use_it():
            assert_is_value(Unhashable().prop, AnyValue(AnySource.inference))

    @assert_passes(run_in_both_module_modes=True)
    def test_property_on_class_object(self):
        from typing_extensions import assert_type

        class Capybara:
            @property
            def p(self) -> int:
                return 3

        def use_it() -> None:
            assert_type(Capybara().p, int)
            prop: property = Capybara.p
            prop2: int = Capybara.p  # E: incompatible_assignment
            print(prop, prop2)

    @assert_passes()
    def test_custom_descriptor_on_class_object(self):
        class CustomDescriptor:
            def __get__(self, obj, typ):
                if obj is None:
                    return self
                return 3

        class Capybara:
            prop = CustomDescriptor()

        def use_it(cls: type[Capybara]) -> None:
            assert_is_value(cls.prop, AnyValue(AnySource.inference))

    @assert_passes()
    def test_tuple_subclass_with_getattr(self):
        # Inspired by pyspark.sql.types.Row
        class Row(tuple):
            def __getattr__(self, attr):
                if attr.startswith("__"):
                    raise AttributeError(attr)
                return attr.upper()

        def capybara():
            x = Row()
            return x.capybaras

    @skip_if_not_installed("pydantic")
    @assert_passes()
    def test_only_known_attributes_pydantic(self):
        from pydantic import BaseModel

        class BM(BaseModel):
            a: int

        def capybara(bm: BM) -> None:
            assert_type(bm.a, int)

            bm.b  # E: undefined_attribute

    @assert_passes()
    def test_annotated_known(self):
        from typing_extensions import Annotated, Literal

        from pycroscope.extensions import LiteralOnly
        from pycroscope.test_attributes import EXPECTED_ASCII_TYPEVARS
        from pycroscope.value import KnownValueWithTypeVars

        def capybara():
            encoding: Annotated[Literal["ascii"], LiteralOnly()] = "ascii"
            assert_is_value(
                encoding.encode,
                KnownValueWithTypeVars(encoding.encode, EXPECTED_ASCII_TYPEVARS),
            )

    @skip_before((3, 12))
    def test_annotated_plus_alias(self):
        self.assert_passes("""
            from typing import Annotated, Literal, assert_type

            type X = Annotated[str, "hi"]

            def capybara(x: X):
                assert_type(x.isnumeric(), bool)
            """)

    @assert_passes()
    def test_optional_operation(self):
        from typing import Optional

        def capybara(x: Optional[str]):
            print(x[1:])  # E: unsupported_operation

    @assert_passes()
    def test_optional(self):
        from typing import Optional

        def capybara(x: Optional[str]):
            x.split()  # E: undefined_attribute

    @assert_passes()
    def test_typeshed(self):
        # missing_generic_parameters is a bit questionable here, but the
        # class really is defined as generic in typeshed.
        def capybara(c: staticmethod):  # E: missing_generic_parameters
            assert_type(c.__isabstractmethod__, bool)

    @assert_passes()
    def test_no_attribute_for_typeshed_class():
        def capybara(c: staticmethod):  # E: missing_generic_parameters
            c.no_such_attribute  # E: undefined_attribute

    @assert_passes()
    def test_typeshed_getattr(self):
        # has __getattr__
        from codecs import StreamWriter

        # has __getattribute__ in typeshed
        from types import SimpleNamespace

        def capybara(sn: SimpleNamespace, sw: StreamWriter):
            assert_is_value(sn.whatever, AnyValue(AnySource.inference))
            assert_is_value(sw.whatever, AnyValue(AnySource.inference))

    @assert_passes()
    def test_allow_function(self):
        def decorator(f):
            return f

        def capybara():
            @decorator
            def f():
                pass

            f.attr = 42
            print(f.attr)

    # TODO: Doesn't trigger incompatible_override on 3.11 for some reason.
    @only_before((3, 11))
    @assert_passes()
    def test_enum_name(self):
        import enum

        class E(enum.Enum):
            name = 1  # E: incompatible_override
            no_name = 2

        def capybara():
            assert_is_value(E.no_name, KnownValue(E.no_name))
            assert_is_value(E.name, KnownValue(E.name))
            E.what_is_this  # E: undefined_attribute

    @assert_passes()
    def test_module_annotations(self):
        from typing import Optional

        from pycroscope import test_attributes

        annotated_global: Optional[str] = None

        def capybara():
            assert_type(test_attributes._global_dict, dict[int | str, bytes])
            assert_type(annotated_global, str | None)

    @assert_passes()
    def test_enum_class_name_is_undefined(self):
        import enum

        class Rodent(enum.Enum):
            capybara = 1

        class Cavy(enum.Enum):
            paca = 1

        def capybara(flag: bool) -> None:
            Rodent.name  # E: undefined_attribute
            enum_type = Rodent if flag else Cavy
            enum_type.name  # E: undefined_attribute

    @assert_passes()
    def test_unwrap_mvv(self):
        def render_task(name: str):
            if not (name or "").strip():
                name = "x"
            assert_type(name, str)

    @assert_passes()
    def test_raising_prop(self):
        class HasProp:
            @property
            def does_it_really(self) -> int:
                raise Exception("fooled you")

        has_prop = HasProp()

        def capybara():
            assert_is_value(has_prop.does_it_really, AnyValue(AnySource.inference))

    @assert_passes()
    def test_super_descriptor_binding(self):
        class Base:
            @classmethod
            def make_name(cls) -> str:
                return cls.__name__

            @property
            def payload(self) -> bytes:
                return b"x"

            def scale(self, x: float) -> float:
                return x

        class Child(Base):
            @classmethod
            def read_classmethod(cls) -> None:
                assert_type(super().make_name(), str)

            def read_property(self) -> None:
                assert_type(super().payload, bytes)

            def read_method(self) -> None:
                assert_type(super().scale(1.0), float)

    @assert_passes()
    def test_direct_super_objects(self):
        class Base:
            @classmethod
            def make_name(cls) -> str:
                return cls.__name__

            def scale(self, x: float) -> float:
                return x

        class Child(Base):
            @classmethod
            def read_classmethod(cls) -> None:
                bound = super(Child, cls).make_name
                assert_type(bound(), str)

            def read_method(self) -> None:
                bound = super(Child, self).scale
                assert_type(bound(1.0), float)

    def test_super_special_attributes(self):
        class Base:
            pass

        class Child(Base):
            def read_instance_super(self) -> None:
                receiver: object = super().__self__
                owner: object = super().__thisclass__
                print(receiver, owner)

            @classmethod
            def read_class_super(cls) -> None:
                receiver: object = super().__self__
                print(receiver)

    @assert_passes()
    def test_explicit_super_receiver_variants(self):
        class Base:
            @classmethod
            def make_name(cls) -> str:
                return cls.__name__

            def scale(self, x: float) -> float:
                return x

        class Child(Base):
            pass

        instance_super = super(Child, Child())
        class_super = super(Child, Child)

        def capybara() -> None:
            receiver1: object = instance_super.__self__
            receiver2: object = class_super.__self__
            assert_type(instance_super.scale(1.0), float)
            assert_type(class_super.make_name(), str)
            print(receiver1, receiver2)

    @assert_passes()
    def test_super_staticmethod(self):
        class Base:
            @staticmethod
            def label() -> int:
                return 1

        class Child(Base):
            def read_staticmethod(self) -> None:
                assert_type(super().label(), int)

    @assert_passes()
    def test_super_new_attribute(self):
        class Base:
            def __new__(cls) -> "Base":
                return super().__new__(cls)

        class Child(Base):
            @classmethod
            def make(cls) -> "Child":
                ctor = super().__new__
                # TODO: This should be inferred as Child once super().__new__ is specialized.
                return ctor(cls)  # E: incompatible_return_value

    @assert_passes()
    def test_function_annotations_and_unhashable_hash(self):
        from typing import Any

        def f(x: int) -> str:
            return str(x)

        class Unhashable:
            __hash__ = None

        def capybara(u: Unhashable) -> None:
            assert_type(f.__annotations__, dict[str, Any])
            assert_is_value(u.__hash__, KnownValue(None))

    @assert_passes(ignore_none_attributes=True)
    def test_ignore_none_attributes_option(self):
        value = None

        def capybara() -> None:
            print(value.missing)

    @assert_passes()
    def test_runtime_type_alias_attributes(self):
        from typing_extensions import Literal, TypeAliasType

        Alias = TypeAliasType("Alias", int)

        def capybara() -> None:
            assert_type(Alias.__name__, Literal["Alias"])
            Alias.unknown  # E: undefined_attribute

    @assert_passes()
    def test_metatype_instance_attributes(self):
        from typing import Any

        def capybara(typ: type) -> None:
            assert_type(typ.__name__, str)
            assert_type(typ.__qualname__, str)
            assert_type(typ.__module__, str)
            assert_type(typ.__annotations__, dict[str, Any])
            assert_type(typ.__bases__, tuple[type, ...])
            assert_type(typ.__mro__, tuple[type, ...])

    @assert_passes()
    def test_enum_value_type(self):
        import enum

        from typing_extensions import Literal

        class Mixed(enum.Enum):
            INT = 1
            STR = "x"

        def capybara(value: Mixed) -> None:
            assert_type(value.value, Literal[1, "x"])

    @assert_passes(allow_import_failures=True)
    def test_synthetic_enum_value_type(self):
        import enum

        import does_not_exist  # noqa: F401

        class Mixed(enum.Enum):
            INT = 1
            STR = "x"

        def capybara(value: Mixed) -> None:
            print(value.value)

    @pytest.mark.filterwarnings(
        "ignore:private variables, such as .* will be normal attributes in 3\\.11:"
        "DeprecationWarning"
    )
    @assert_passes()
    def test_private_enum_nonmember_attribute(self):
        import enum
        import sys

        from typing_extensions import Literal, assert_type

        class Example2(enum.Enum):
            __B = 2

            def method(self) -> None:
                if sys.version_info < (3, 11):
                    assert_type(Example2.__B, Literal[Example2._Example2__B])
                else:
                    assert_type(Example2.__B, Literal[2])

    @assert_passes(allow_import_failures=True)
    def test_private_synthetic_enum_nonmember_attribute(self):
        import enum

        import does_not_exist  # noqa: F401

        class Example2(enum.Enum):
            __B = 2

            def method(self) -> None:
                print(Example2.__B)

    @assert_passes(allow_import_failures=True)
    def test_synthetic_typevar_bound_class_attribute_after_import_failure(self):
        from typing import TypeVar

        import does_not_exist  # noqa: F401
        from typing_extensions import assert_type

        class Base:
            value: int

            @classmethod
            def build(cls) -> type["Base"]:
                return cls

        T = TypeVar("T", bound="Base")

        def capybara(cls: type[T]) -> None:
            assert_type(cls.value, int)
            print(cls.build())

    @assert_passes(allow_import_failures=True)
    def test_synthetic_sequence_base_attributes_after_import_failure(self):
        from collections.abc import Sequence

        import does_not_exist  # noqa: F401
        from typing_extensions import assert_type

        class Base(Sequence[int]):
            def __getitem__(self, index: int) -> int:
                return index

            def __len__(self) -> int:
                return 0

        class Child(Base):
            pass

        def capybara(child: Child) -> None:
            print(child.__dict__)
            assert_type(child.count(1), int)

    @assert_passes(allow_import_failures=True)
    def test_synthetic_inherited_call_after_import_failure(self):
        from typing import Any

        import does_not_exist  # noqa: F401
        from typing_extensions import assert_type

        class Base:
            def __call__(self, x: int) -> int:
                return x

        class Child(Base):
            pass

        def capybara(child: Child) -> None:
            assert_type(child(1), Any)

    @assert_passes(allow_import_failures=True)
    def test_synthetic_module_attribute_lookup_after_import_failure(self):
        import does_not_exist

        def capybara() -> None:
            print(does_not_exist.__name__)
            print(does_not_exist.no_such_attribute)

    @assert_passes(allow_import_failures=True)
    def test_synthetic_static_and_class_method_wrappers_after_import_failure(self):
        import does_not_exist  # noqa: F401
        from typing_extensions import Self, assert_type

        class Base:
            @staticmethod
            def parse(x: int) -> int:
                return x

            @classmethod
            def make(cls, x: int) -> Self:
                return cls()

        class Child(Base):
            pass

        def capybara(child: Child, cls: type[Child]) -> None:
            assert_type(child.parse(1), int)
            assert_type(Child.parse(1), int)
            assert_type(cls.parse(1), int)
            assert_type(child.make(1), Child)
            assert_type(Child.make(1), Child)
            print(cls.make(1))

    @assert_passes(allow_import_failures=True)
    def test_synthetic_descriptor_signature_matching_after_import_failure(self):
        from typing import Annotated, Any

        import does_not_exist  # noqa: F401

        class DefaultDescriptor:
            def __get__(
                self, instance: object | None, owner: Any, extra: int = 0
            ) -> int:
                return extra

        class VariadicDescriptor:
            def __get__(
                self, instance: object | None, owner: Any, *extra: object
            ) -> int:
                return len(extra)

        class BrokenDescriptor:
            def __get__(self, instance: object | None, owner: Any, extra: int) -> int:
                return extra

        class Box:
            default: DefaultDescriptor = DefaultDescriptor()
            variadic: VariadicDescriptor = VariadicDescriptor()
            annotated: Annotated[DefaultDescriptor, "x"] = DefaultDescriptor()
            broken: BrokenDescriptor = BrokenDescriptor()

        def capybara(box: Box) -> None:
            print(box.default)
            print(box.variadic)
            print(box.annotated)
            print(box.broken)

    @assert_passes(run_in_both_module_modes=True)
    def test_private_attribute_lookup(self):
        from typing_extensions import Literal, assert_type

        class Base:
            def __init__(self) -> None:
                self.__secret = 1

            def reveal(self) -> None:
                assert_type(self.__secret, Literal[1])

    @assert_passes(run_in_both_module_modes=True)
    def test_private_dataclass_attribute_lookup(self):
        from dataclasses import dataclass

        from typing_extensions import assert_type

        @dataclass
        class Base:
            __secret: int

            def reveal(self) -> None:
                assert_type(self.__secret, int)

        def capybara(base: Base) -> None:
            assert_type(base.reveal(), None)

    @assert_passes()
    def test_descriptor_instance_access_strips_descriptor_self(self):
        from typing import Any, Generic, TypeVar, cast

        from typing_extensions import Self, assert_type

        T = TypeVar("T")

        class Descriptor(Generic[T]):
            def __get__(self, instance: object | None, owner: Any) -> T | Self:
                return self if instance is None else cast(Any, instance)

        class Related:
            name: str

        class Model:
            related = Descriptor[Related | None]()

            def get_name(self) -> str | None:
                if self.related is not None:
                    return self.related.name
                return None

        def capybara(model: Model) -> None:
            assert_type(model.related, Related | None)

    @assert_passes(allow_import_failures=True)
    def test_synthetic_generic_descriptor_and_private_attributes(self):
        from typing import Any, Generic, TypeVar, overload

        import does_not_exist  # noqa: F401
        from typing_extensions import Self

        T = TypeVar("T")

        class Descriptor(Generic[T]):
            @overload
            def __get__(self, instance: None, owner: Any) -> "Descriptor[T]": ...

            @overload
            def __get__(self, instance: object, owner: Any) -> T: ...

            def __get__(
                self, instance: object | None, owner: Any
            ) -> "Descriptor[T] | T":
                raise NotImplementedError

        class Base(Generic[T]):
            data: T
            __secret: int
            desc: Descriptor[T] = Descriptor()

            def __init__(self, data: T, secret: int) -> None:
                self.data = data
                self.__secret = secret

            @property
            def payload(self) -> T:
                return self.data

            @classmethod
            def make(cls, data: T, secret: int) -> Self:
                return cls(data, secret)

            def reveal(self) -> int:
                return self.__secret

        class Child(Base[int]):
            pass

        def capybara(cls: type[Child], inst: Child) -> None:
            print(Child.__dict__)
            print(cls.make(1, 2))
            print(inst.data)
            print(inst.desc)
            print(inst.payload)
            print(inst.reveal())

    @assert_passes(allow_import_failures=True)
    def test_synthetic_dataclass_descriptor_generic_base_attributes(self):
        from dataclasses import dataclass
        from typing import Any, Generic, TypeVar, cast, overload

        import does_not_exist  # noqa: F401

        T = TypeVar("T")

        class ReadDescriptor(Generic[T]):
            @overload
            def __get__(self, instance: None, owner: Any) -> "ReadDescriptor[T]": ...

            @overload
            def __get__(self, instance: object, owner: Any) -> T: ...

            def __get__(
                self, instance: object | None, owner: Any
            ) -> "ReadDescriptor[T] | T":
                if instance is None:
                    return self
                return cast(T, 0)

        @dataclass
        class Base(Generic[T]):
            value: T
            __secret: int
            extra: ReadDescriptor[T] = ReadDescriptor()

            @property
            def payload(self) -> T:
                return self.value

            def reveal(self) -> int:
                return self.__secret

        @dataclass
        class Child(Base[int]):
            label: str = "x"

        def capybara(inst: Child) -> None:
            print(Child.__class__)
            print(Child.__dict__)
            print(Child.extra)
            print(inst.value)
            print(inst.extra)
            print(inst.payload)
            print(inst.reveal())

    @assert_passes(run_in_both_module_modes=True)
    def test_module_mode_dataclass_descriptor_generic_base_attributes(self):
        from dataclasses import dataclass
        from typing import Any, Generic, TypeVar, cast, overload

        T = TypeVar("T")

        class ReadDescriptor(Generic[T]):
            @overload
            def __get__(self, instance: None, owner: Any) -> "ReadDescriptor[T]": ...

            @overload
            def __get__(self, instance: object, owner: Any) -> T: ...

            def __get__(
                self, instance: object | None, owner: Any
            ) -> "ReadDescriptor[T] | T":
                if instance is None:
                    return self
                return cast(T, 0)

        @dataclass
        class Base(Generic[T]):
            value: T
            __secret: int
            extra: ReadDescriptor[T] = ReadDescriptor()

            @property
            def payload(self) -> T:
                return self.value

            def reveal(self) -> int:
                return self.__secret

        @dataclass
        class Child(Base[int]):
            label: str = "x"

        def capybara(inst: Child) -> None:
            print(Child.__class__)
            print(Child.__dict__)
            print(Child.extra)
            print(inst.value)
            print(inst.extra)
            print(inst.payload)
            print(inst.reveal())

    @skip_if_not_installed("qcore")
    @assert_passes()
    def test_cached_per_instance(self):
        from qcore.caching import cached_per_instance

        class C:
            @cached_per_instance()
            def f(self) -> int:
                return 42

        def capybara():
            c = C()
            assert_type(c.f(), int)


class TestHasAttr(TestNameCheckVisitorBase):
    @assert_passes()
    def test_hasattr(self):
        from typing import Any

        from typing_extensions import Literal, Never, assert_type

        def capybara(x: Literal[1], y: object) -> None:
            if hasattr(x, "x"):
                assert_type(x, Never)
            if hasattr(y, "x"):
                assert_type(y.x, Any)

    @assert_passes()
    def test_multi_hasattr(self):
        from typing import Any, Union

        from typing_extensions import assert_type

        class A:
            pass

        class B:
            pass

        def capybara(x: Union[A, B]):
            if hasattr(x, "a") and hasattr(x, "b"):
                assert_type(x.a, Any)
                assert_type(x.b, Any)

    @assert_passes()
    def test_hasattr_plus_call(self):
        class X:
            @classmethod
            def types(cls):
                return []

        def capybara(x: X) -> None:
            cls = X
            if hasattr(cls, "types"):  # E: value_always_true
                assert_is_value(cls.types(), AnyValue(AnySource.unannotated))


class TestClassAttributeTransformer(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from pycroscope.test_config import StringField

        class Capybara:
            foo = StringField()

        def capybara(c: Capybara, cls: type[Capybara]):
            assert_type(c.foo, str)
            assert_type(cls.foo, str)


class TestAttributeWrites(TestNameCheckVisitorBase):
    @assert_passes()
    def test_unannotated_in_body(self):
        from typing_extensions import Literal, assert_type

        class Capybara:
            flag = False

            def __init__(self) -> None:
                self.flag = True

            def method(self) -> None:
                # TODO should be bool
                assert_type(self.flag, Literal[False])
