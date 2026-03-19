# static analysis: ignore
from typing import Dict, Union

from .attributes import normalize_synthetic_descriptor_attribute
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import (
    assert_passes,
    only_before,
    skip_before,
    skip_if_not_installed,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    assert_is_value,
)

_global_dict: Dict[Union[int, str], bytes] = {}


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
        from pycroscope.value import CustomCheckExtension, KnownValueWithTypeVars, SelfT

        def capybara():
            encoding: Annotated[Literal["ascii"], LiteralOnly()] = "ascii"
            assert_is_value(
                encoding.encode,
                KnownValueWithTypeVars(
                    encoding.encode,
                    {
                        SelfT: AnnotatedValue(
                            KnownValue("ascii"), [CustomCheckExtension(LiteralOnly())]
                        )
                    },
                ),
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

    @assert_passes()
    def test_super_staticmethod_current_behavior(self):
        class Base:
            @staticmethod
            def label() -> int:
                return 1

        class Child(Base):
            def read_staticmethod(self) -> None:
                # TODO: This should be accepted once super() staticmethod binding is fixed.
                super().label()  # E: incompatible_call

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

    @assert_passes()
    def test_runtime_type_alias_attributes(self):
        from typing_extensions import Literal, TypeAliasType

        Alias = TypeAliasType("Alias", int)

        def capybara() -> None:
            assert_type(Alias.__name__, Literal["Alias"])
            Alias.unknown  # E: undefined_attribute

    @assert_passes()
    def test_enum_value_type(self):
        import enum

        from typing_extensions import Literal

        class Mixed(enum.Enum):
            INT = 1
            STR = "x"

        def capybara(value: Mixed) -> None:
            assert_type(value.value, Literal[1, "x"])

    @assert_passes()
    def test_private_attribute_lookup(self):
        from typing_extensions import Literal

        class Base:
            def __init__(self) -> None:
                self.__secret = 1

            def reveal(self) -> None:
                assert_type(self.__secret, Literal[1])

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


class TestHasAttrExtension(TestNameCheckVisitorBase):
    @assert_passes()
    def test_hasattr(self):
        from typing_extensions import Literal

        def capybara(x: Literal[1]) -> None:
            if hasattr(x, "x"):
                assert_is_value(x.x, AnyValue(AnySource.inference))

    @assert_passes()
    def test_user_hasattr(self):
        from typing import Any, TypeVar

        from typing_extensions import Annotated, Literal

        from pycroscope.extensions import HasAttrGuard

        T = TypeVar("T", bound=str)

        def my_hasattr(
            obj: object, name: T
        ) -> Annotated[bool, HasAttrGuard["obj", T, Any]]:
            return hasattr(obj, name)

        def has_int_attr(
            obj: object, name: T
        ) -> Annotated[bool, HasAttrGuard["obj", T, int]]:
            val = getattr(obj, name, None)
            return isinstance(val, int)

        def capybara(x: Literal[1]) -> None:
            if my_hasattr(x, "x"):
                assert_is_value(x.x, AnyValue(AnySource.explicit))

        def inty_capybara(x: Literal[1]) -> None:
            if has_int_attr(x, "inty"):
                assert_type(x.inty, int)

    @assert_passes()
    def test_multi_hasattr(self):
        from typing import Union

        class A:
            pass

        class B:
            pass

        def capybara(x: Union[A, B]):
            if hasattr(x, "a") and hasattr(x, "b"):
                assert_is_value(x.a, AnyValue(AnySource.inference))
                assert_is_value(x.b, AnyValue(AnySource.inference))

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

        def capybara(c: Capybara):
            assert_type(c.foo, str)
