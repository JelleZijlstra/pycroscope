# static analysis: ignore

import sys
from types import ModuleType
from typing import TypeVar

from .checker import Checker
from .test_config import TEST_OPTIONS
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .type_object import (
    DataclassFieldRecord,
    NamedTupleField,
    _class_key_from_value,
    _should_use_permissive_dunder_hash,
    lookup_declared_symbol_with_owner,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    ClassSymbol,
    DataclassFieldInfo,
    DataclassInfo,
    FunctionDecorator,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    SubclassValue,
    SyntheticClassObjectValue,
    TypedValue,
    TypeVarParam,
    TypeVarTupleBindingValue,
    TypeVarTupleParam,
    TypeVarTupleValue,
    TypeVarValue,
    assert_is_value,
)

T = TypeVar("T")


def _make_runtime_property_class(module_name: str) -> type:
    class C:
        @property
        def name(self):
            return 1

    package_path = ""
    for package_piece in module_name.split(".")[:-1]:
        package_path = (
            f"{package_path}.{package_piece}" if package_path else package_piece
        )
        package = sys.modules.get(package_path)
        if package is None:
            package = ModuleType(package_path)
            package.__path__ = []
            sys.modules[package_path] = package
    C.__module__ = module_name
    C.__qualname__ = "C"
    module = ModuleType(module_name)
    module.__package__ = module_name.rpartition(".")[0]
    module.C = C
    sys.modules[module_name] = module
    if module.__package__:
        parent = sys.modules[module.__package__]
        setattr(parent, module_name.rsplit(".", maxsplit=1)[1], module)
    return C


def test_protocol_member_str_order_is_deterministic() -> None:
    from typing_extensions import Protocol

    class HasMembers(Protocol):
        def f(self) -> int: ...

        def m(self) -> int: ...

    checker = Checker()
    type_object = checker.make_type_object(HasMembers)
    assert str(type_object).endswith("(Protocol with members 'f', 'm')")


def test_class_key_from_subclass_generic_value() -> None:
    value = SubclassValue(GenericValue("mod.Base", [TypedValue(int)]))
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_union_with_consistent_key() -> None:
    value = MultiValuedValue(
        [
            SubclassValue(TypedValue("mod.Base")),
            GenericValue("mod.Base", [TypedValue(int)]),
        ]
    )
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_intersection_with_consistent_key() -> None:
    value = IntersectionValue(
        (TypedValue("mod.Base"), SubclassValue(TypedValue("mod.Base")))
    )
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_annotated_value() -> None:
    value = AnnotatedValue(TypedValue("mod.Base"), ())
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_known_generic_alias() -> None:
    assert _class_key_from_value(KnownValue(list[int])) is list


def test_lookup_declared_symbol_with_owner_handles_synthetic_base() -> None:
    checker = Checker()
    base = SyntheticClassObjectValue("Base", TypedValue("mod.Base"))
    child = SyntheticClassObjectValue("Child", TypedValue("mod.Child"))
    checker.register_synthetic_class(base)
    checker.register_synthetic_class(child)
    checker.make_type_object("mod.Base").set_declared_symbol(
        "x", ClassSymbol(annotation=TypedValue(int))
    )
    checker.make_type_object("mod.Child").set_direct_bases((TypedValue("mod.Base"),))

    match = lookup_declared_symbol_with_owner("mod.Child", "x", checker)
    assert match is not None
    owner, symbol = match
    assert owner == "mod.Base"
    assert symbol.annotation == TypedValue(int)


def test_lookup_declared_symbol_with_owner_merges_typeshed_property_info() -> None:
    module_name = "_pycroscope_tests.stub_property"
    try:
        runtime_class = _make_runtime_property_class(module_name)
        checker = Checker(raw_options=TEST_OPTIONS)
        match = lookup_declared_symbol_with_owner(runtime_class, "name", checker)

        assert match is not None
        owner, symbol = match
        assert owner is runtime_class
        assert symbol.property_info is not None
        assert symbol.property_info.getter_type == TypedValue(str)
    finally:
        sys.modules.pop(module_name, None)
        sys.modules.pop("_pycroscope_tests", None)


def test_runtime_mro_marks_typeshed_only_bases_virtual() -> None:
    from collections.abc import MutableSequence

    checker = Checker()
    entries = {
        entry.tobj.typ: entry
        for entry in checker.make_type_object(list).get_mro()
        if entry.tobj is not None
    }

    assert entries[list].is_virtual is False
    assert entries[MutableSequence].is_virtual is True
    assert entries[object].is_virtual is False


def test_synthetic_mro_prefers_non_virtual_duplicate_base() -> None:
    from collections.abc import MutableSequence, Sequence

    checker = Checker()
    checker.make_type_object("mod.A").set_direct_bases((TypedValue(list),))
    checker.make_type_object("mod.B").set_direct_bases((TypedValue(Sequence),))
    checker.make_type_object("mod.Child").set_direct_bases(
        (TypedValue("mod.A"), TypedValue("mod.B"))
    )

    entries = {
        entry.tobj.typ: entry
        for entry in checker.make_type_object("mod.Child").get_mro()
        if entry.tobj is not None
    }

    assert entries["mod.A"].is_virtual is False
    assert entries["mod.B"].is_virtual is False
    assert entries[Sequence].is_virtual is False
    assert entries[MutableSequence].is_virtual is True


def test_permissive_dunder_hash_class_object_detection() -> None:
    assert _should_use_permissive_dunder_hash(TypedValue(type))
    assert _should_use_permissive_dunder_hash(GenericValue(type, [TypedValue(int)]))
    assert not _should_use_permissive_dunder_hash(TypedValue(list))
    assert not _should_use_permissive_dunder_hash(
        MultiValuedValue([TypedValue(type), TypedValue(list)])
    )
    assert _should_use_permissive_dunder_hash(
        IntersectionValue((TypedValue(type), TypedValue(list)))
    )


def test_runtime_declared_symbol_uses_annotation_expr_parsing() -> None:
    from typing import Annotated, ClassVar

    from typing_extensions import ReadOnly

    class Runtime:
        attr: Annotated[ClassVar[ReadOnly[int]], "meta"]

    checker = Checker()
    symbol = checker.make_type_object(Runtime).get_declared_symbol("attr")
    assert symbol is not None
    assert symbol.is_classvar
    assert symbol.is_readonly
    assert not symbol.is_instance_only
    assert symbol.annotation == TypedValue(int)
    assert symbol.initializer is None


def test_runtime_declared_symbol_tracks_decorator_and_deprecation_metadata() -> None:
    from abc import abstractmethod
    from typing import final

    from .extensions import deprecated

    class Runtime:
        @classmethod
        @abstractmethod
        def build(cls) -> "Runtime":
            raise NotImplementedError

        @classmethod
        @final
        def done(cls) -> int:
            return 1

        @staticmethod
        @deprecated("old")
        def old() -> int:
            return 1

    checker = Checker()
    type_object = checker.make_type_object(Runtime)

    build = type_object.get_declared_symbol("build")
    assert build is not None
    assert build.is_method
    assert FunctionDecorator.classmethod in build.function_decorators
    assert FunctionDecorator.abstractmethod in build.function_decorators

    done = type_object.get_declared_symbol("done")
    assert done is not None
    assert FunctionDecorator.classmethod in done.function_decorators
    if getattr(getattr(Runtime.__dict__["done"], "__func__", None), "__final__", False):
        assert done.is_final
        assert FunctionDecorator.final in done.function_decorators
    else:
        assert not done.is_final
        assert FunctionDecorator.final not in done.function_decorators

    old = type_object.get_declared_symbol("old")
    assert old is not None
    assert old.deprecation_message == "old"
    assert FunctionDecorator.staticmethod in old.function_decorators


def test_runtime_property_symbol_tracks_accessor_deprecations() -> None:
    from .extensions import deprecated

    class Runtime:
        @property
        @deprecated("getter")
        def value(self) -> int:
            return 1

        @value.setter
        @deprecated("setter")
        def value(self, new_value: int) -> None:
            pass

    checker = Checker()
    symbol = checker.make_type_object(Runtime).get_declared_symbol("value")

    assert symbol is not None
    assert symbol.property_info is not None
    assert symbol.property_info.getter_deprecation == "getter"
    assert symbol.property_info.setter_deprecation == "setter"


def test_runtime_type_object_tracks_dataclass_fields() -> None:
    from dataclasses import InitVar, dataclass
    from typing import ClassVar

    @dataclass
    class Base:
        a: int

    @dataclass
    class Child(Base):
        b: int
        c: InitVar[str]
        d: ClassVar[int] = 0

    checker = Checker()
    assert checker.make_type_object(Child).get_dataclass_fields() == (
        DataclassFieldRecord("a", DataclassFieldInfo()),
        DataclassFieldRecord("b", DataclassFieldInfo()),
        DataclassFieldRecord("c", DataclassFieldInfo()),
        DataclassFieldRecord("d", DataclassFieldInfo(has_default=True, kw_only=False)),
    )


def test_synthetic_type_object_tracks_dataclass_fields_without_initializers() -> None:
    checker = Checker()
    dataclass_info = DataclassInfo(
        init=True,
        eq=True,
        frozen=False,
        unsafe_hash=False,
        match_args=True,
        order=False,
        slots=False,
        kw_only_default=False,
        field_specifiers=(),
    )
    base = SyntheticClassObjectValue("Base", TypedValue("mod.Base"))
    child = SyntheticClassObjectValue("Child", TypedValue("mod.Child"))
    checker.register_synthetic_class(base)
    checker.register_synthetic_class(child)
    checker.make_type_object("mod.Base").set_dataclass_info(dataclass_info)
    checker.make_type_object("mod.Child").set_dataclass_info(dataclass_info)
    checker.make_type_object("mod.Child").set_direct_bases((TypedValue("mod.Base"),))
    checker.make_type_object("mod.Base").set_declared_symbol(
        "a",
        ClassSymbol(annotation=TypedValue(int), dataclass_field=DataclassFieldInfo()),
    )
    checker.make_type_object("mod.Child").set_declared_symbol(
        "b",
        ClassSymbol(annotation=TypedValue(str), dataclass_field=DataclassFieldInfo()),
    )

    assert checker.make_type_object("mod.Child").get_dataclass_fields() == (
        DataclassFieldRecord("a", DataclassFieldInfo()),
        DataclassFieldRecord("b", DataclassFieldInfo()),
    )


def test_runtime_type_object_direct_dataclass_fields_ignore_mangled_duplicates() -> (
    None
):
    from dataclasses import dataclass

    @dataclass
    class Base:
        value: int
        __secret: int
        extra: int = 0

    checker = Checker()
    type_object = checker.make_type_object(Base)
    type_object.set_dataclass_info(
        DataclassInfo(
            init=True,
            eq=True,
            frozen=False,
            unsafe_hash=False,
            match_args=True,
            order=False,
            slots=False,
            kw_only_default=False,
            field_specifiers=(),
        )
    )
    type_object.set_declared_symbol(
        "value", ClassSymbol(dataclass_field=DataclassFieldInfo())
    )
    type_object.set_declared_symbol(
        "__secret", ClassSymbol(dataclass_field=DataclassFieldInfo())
    )
    type_object.set_declared_symbol(
        "extra", ClassSymbol(dataclass_field=DataclassFieldInfo(has_default=True))
    )

    assert type_object.get_direct_dataclass_fields() == (
        DataclassFieldRecord("value", DataclassFieldInfo()),
        DataclassFieldRecord("__secret", DataclassFieldInfo()),
        DataclassFieldRecord("extra", DataclassFieldInfo(has_default=True)),
    )


def test_runtime_namedtuple_field_is_readonly() -> None:
    from typing import NamedTuple

    class Point(NamedTuple):
        x: int

    checker = Checker()
    symbol = checker.make_type_object(Point).get_declared_symbol("x")
    assert symbol is not None
    assert symbol.is_readonly


def test_type_object_exposes_synthetic_namedtuple_metadata() -> None:
    checker = Checker()
    checker.make_type_object("mod.Base").set_namedtuple_fields(
        [NamedTupleField("x", TypedValue(int), None)]
    )
    checker.make_type_object("mod.Child").set_direct_bases((TypedValue("mod.Base"),))
    checker.make_type_object("mod.Child").set_declared_symbol(
        "label", ClassSymbol(annotation=TypedValue(str), is_instance_only=True)
    )

    type_object = checker.make_type_object("mod.Child")
    assert type_object.is_namedtuple_like()
    assert type_object.get_namedtuple_fields() == (
        NamedTupleField("x", TypedValue(int), None),
    )
    assert type_object.get_namedtuple_field("x") == NamedTupleField(
        "x", TypedValue(int), None
    )


def test_synthetic_declared_symbol_overrides_raw_attribute_value() -> None:
    checker = Checker()
    synthetic = SyntheticClassObjectValue("Impl", TypedValue("mod.Impl"))
    checker.register_synthetic_class(synthetic)
    checker.make_type_object("mod.Impl").set_declared_symbol(
        "attr",
        ClassSymbol(
            annotation=TypedValue(object),
            is_instance_only=True,
            initializer=TypedValue(str),
        ),
    )

    symbol = checker.make_type_object("mod.Impl").get_declared_symbol("attr")
    assert symbol is not None
    assert symbol.is_instance_only
    assert symbol.annotation == TypedValue(object)
    assert symbol.initializer == TypedValue(str)


def test_type_object_declared_symbols_are_canonical_for_synthetic_class() -> None:
    checker = Checker()
    synthetic = SyntheticClassObjectValue("Impl", TypedValue("mod.Impl"))
    checker.register_synthetic_class(synthetic)
    type_object = checker.make_type_object("mod.Impl")
    type_object.set_declared_symbol("attr", ClassSymbol(annotation=TypedValue(int)))

    first = type_object.get_declared_symbols()
    second = type_object.get_declared_symbols()
    assert first is second
    assert type_object.get_synthetic_declared_symbols() is first

    checker.register_synthetic_protocol_members("mod.Impl", {"extra"})
    assert "extra" in first
    assert (
        type_object.get_synthetic_declared_symbols()
        is type_object.get_declared_symbols()
    )


def test_runtime_type_object_tracks_declared_type_params_and_specialized_mro() -> None:
    from typing import Generic

    class Base(Generic[T]):
        pass

    class Child(Base[int]):
        pass

    checker = Checker()
    base_object = checker.make_type_object(Base)
    child_object = checker.make_type_object(Child)

    assert tuple(base_object.get_declared_type_params()) == tuple(
        checker.get_type_parameters(Base)
    )
    assert [entry.get_mro_value() for entry in base_object.get_mro()] == [
        GenericValue(Base, [TypeVarValue(TypeVarParam(T))]),
        TypedValue(Generic),
        TypedValue(object),
    ]
    assert tuple(child_object.get_declared_type_params()) == ()
    assert [entry.get_mro_value() for entry in child_object.get_mro()] == [
        TypedValue(Child),
        GenericValue(Base, [TypedValue(int)]),
        TypedValue(Generic),
        TypedValue(object),
    ]


def test_variadic_runtime_type_object_keeps_unpacked_self_mro_args() -> None:
    if sys.version_info < (3, 11):
        return

    ns: dict[str, object] = {}
    exec(
        """
from typing import Generic, TypeVarTuple

Shape = TypeVarTuple("Shape")

class Array(Generic[*Shape]):
    pass
""",
        ns,
        ns,
    )
    Shape = ns["Shape"]
    Array = ns["Array"]

    checker = Checker()
    array_object = checker.make_type_object(Array)

    mro_value = array_object.get_mro()[0].get_mro_value()
    assert isinstance(mro_value, GenericValue)
    assert mro_value.typ is Array
    assert len(mro_value.args) == 1
    assert isinstance(mro_value.args[0], TypeVarTupleValue)
    assert mro_value.args[0].typevar is Shape
    substitutions = array_object.get_substitutions_for_base(
        Array, [TypedValue(int), TypedValue(str)]
    )
    assert substitutions.get_typevartuple(TypeVarTupleParam(Shape)) == (
        (False, TypedValue(int)),
        (False, TypedValue(str)),
    )
    assert substitutions.get_value(
        TypeVarTupleParam(Shape)
    ) == TypeVarTupleBindingValue(((False, TypedValue(int)), (False, TypedValue(str))))


def test_variadic_runtime_type_object_preserves_packed_base_substitutions() -> None:
    if sys.version_info < (3, 11):
        return

    ns: dict[str, object] = {}
    exec(
        """
from typing import Generic, TypeVarTuple

Shape = TypeVarTuple("Shape")

class Base(Generic[*Shape]):
    pass

class Child(Base[*Shape], Generic[*Shape]):
    pass
""",
        ns,
        ns,
    )
    Shape = ns["Shape"]
    Base = ns["Base"]
    Child = ns["Child"]

    checker = Checker()
    child_object = checker.make_type_object(Child)

    substitutions = child_object.get_substitutions_for_base(
        Base, [TypedValue(int), TypedValue(str)]
    )
    assert substitutions.get_typevartuple(TypeVarTupleParam(Shape)) == (
        (False, TypedValue(int)),
        (False, TypedValue(str)),
    )
    assert substitutions.get_value(
        TypeVarTupleParam(Shape)
    ) == TypeVarTupleBindingValue(((False, TypedValue(int)), (False, TypedValue(str))))


def test_synthetic_type_object_tracks_declared_type_params_and_specialized_mro() -> (
    None
):
    checker = Checker()
    base = "test.Base"
    child = "test.Child"
    grandchild = "test.GrandChild"
    type_param = TypeVarParam(T)
    checker.register_synthetic_type_bases(base, [], declared_type_params=[type_param])
    checker.register_synthetic_type_bases(
        child,
        [GenericValue(base, [GenericValue(list, [TypeVarValue(type_param)])])],
        declared_type_params=[type_param],
    )
    checker.register_synthetic_type_bases(
        grandchild, [GenericValue(child, [TypedValue(int)])]
    )

    base_tobj = checker.make_type_object(base)
    child_tobj = checker.make_type_object(child)
    child_tobj.set_direct_bases(
        [GenericValue(base, [GenericValue(list, [TypeVarValue(type_param)])])]
    )

    grandchild_tobj = checker.make_type_object(grandchild)
    grandchild_tobj.set_direct_bases([GenericValue(child, [TypedValue(int)])])

    assert base_tobj.get_declared_type_params() == (type_param,)
    assert [entry.get_mro_value() for entry in base_tobj.get_mro()] == [
        GenericValue(base, [TypeVarValue(type_param)]),
        TypedValue(object),
    ]
    assert [entry.get_mro_value() for entry in grandchild_tobj.get_mro()] == [
        TypedValue(grandchild),
        GenericValue(child, [TypedValue(int)]),
        GenericValue(base, [GenericValue(list, [TypedValue(int)])]),
        TypedValue(object),
    ]


def test_synthetic_type_object_infers_declared_type_params_from_bases() -> None:
    checker = Checker()
    base = "test.Base"
    child = "test.Child"
    type_param = TypeVarParam(T)
    checker.register_synthetic_type_bases(base, [], declared_type_params=[type_param])
    checker.register_synthetic_type_bases(
        child, [GenericValue(base, [TypeVarValue(type_param)])]
    )

    assert checker.make_type_object(child).get_declared_type_params() == (type_param,)


def test_direct_synthetic_declared_symbol_mutation_updates_type_object_view() -> None:
    checker = Checker()
    synthetic = SyntheticClassObjectValue("Impl", TypedValue("mod.Impl"))
    checker.register_synthetic_class(synthetic)

    type_object = checker.make_type_object("mod.Impl")
    assert type_object.get_declared_symbol("attr") is None

    type_object.add_declared_symbol("attr", ClassSymbol(annotation=TypedValue(int)))
    symbol = type_object.get_declared_symbol("attr")
    assert symbol is not None
    assert symbol.annotation == TypedValue(int)


def test_runtime_and_string_type_objects_share_declared_symbols() -> None:
    class Impl:
        pass

    checker = Checker()
    synthetic = SyntheticClassObjectValue("Impl", TypedValue(Impl))
    checker.register_synthetic_class(synthetic)
    checker.make_type_object(Impl).set_declared_symbol(
        "attr", ClassSymbol(annotation=TypedValue(int))
    )

    runtime_type_object = checker.make_type_object(Impl)
    string_type_object = checker.make_type_object(
        f"{Impl.__module__}.{Impl.__qualname__}"
    )

    assert runtime_type_object is string_type_object
    assert (
        runtime_type_object.get_declared_symbols()
        is string_type_object.get_declared_symbols()
    )
    assert runtime_type_object.get_declared_symbol("attr") is not None


def test_inherited_symbol_lookup_returns_declaring_class() -> None:
    class Base:
        x: int = 1

    class Child(Base):
        pass

    checker = Checker()
    match = lookup_declared_symbol_with_owner(Child, "x", checker)
    assert match is not None
    owner, symbol = match
    assert owner is Base
    assert symbol.annotation == TypedValue(int)


def test_runtime_declared_symbol_includes_plain_class_dict_entry() -> None:
    class Meta(type):
        answer = 1

    checker = Checker()
    symbol = checker.make_type_object(Meta).get_declared_symbol("answer")
    assert symbol is not None
    assert not symbol.is_method
    assert not symbol.is_property
    assert symbol.annotation is None
    assert symbol.initializer == KnownValue(1)


def test_get_attribute_substitutes_direct_declared_type_params() -> None:
    from typing import Generic

    class Box(Generic[T]):
        value: T

        @property
        def prop(self) -> T:
            raise NotImplementedError

    checker = Checker()
    type_object = checker.make_type_object(Box)

    value_attr = type_object.get_attribute(
        "value",
        checker,
        on_class=False,
        receiver_value=GenericValue(Box, [TypedValue(str)]),
    )
    assert value_attr is not None
    assert value_attr.value == TypedValue(str)

    prop_attr = type_object.get_attribute(
        "prop",
        checker,
        on_class=False,
        receiver_value=GenericValue(Box, [TypedValue(str)]),
    )
    assert prop_attr is not None
    assert prop_attr.value == TypedValue(str)

    class_prop_attr = type_object.get_attribute(
        "prop",
        checker,
        on_class=True,
        receiver_value=GenericValue(Box, [TypedValue(str)]),
    )
    assert class_prop_attr is not None
    assert class_prop_attr.value == TypedValue(property)


def test_get_attribute_substitutes_inherited_generic_base_args() -> None:
    from typing import Generic

    class Base(Generic[T]):
        value: T

    class Child(Base[int]):
        pass

    checker = Checker()
    attribute = checker.make_type_object(Child).get_attribute(
        "value", checker, on_class=False, receiver_value=TypedValue(Child)
    )

    assert attribute is not None
    assert attribute.owner.typ is Base
    assert attribute.value == TypedValue(int)


def test_get_attribute_substitutes_receiver_args_through_generic_mro() -> None:
    from typing import Generic

    class Base(Generic[T]):
        value: T

    class Child(Base[list[T]], Generic[T]):
        pass

    checker = Checker()
    attribute = checker.make_type_object(Child).get_attribute(
        "value",
        checker,
        on_class=False,
        receiver_value=GenericValue(Child, [TypedValue(int)]),
    )

    assert attribute is not None
    assert attribute.owner.typ is Base
    assert attribute.value == GenericValue(list, [TypedValue(int)])


def test_get_enum_value_type_uses_runtime_members() -> None:
    import enum

    class Color(enum.Enum):
        RED = 1
        CRIMSON = RED
        BLUE = 2

    checker = Checker()
    assert_is_value(
        checker.make_type_object(Color).get_enum_value_type(),
        KnownValue(1) | KnownValue(2),
    )


def test_get_enum_value_type_falls_back_to_declared_value_annotation() -> None:
    import enum

    checker = Checker()
    synthetic = checker.make_synthetic_class("mod.Color")
    type_object = checker.make_type_object("mod.Color")

    type_object.set_direct_bases((TypedValue(enum.Enum),))
    type_object.set_declared_symbol("_value_", ClassSymbol(annotation=TypedValue(int)))

    assert synthetic.class_type == TypedValue("mod.Color")
    assert type_object.get_enum_value_type() == TypedValue(int)


def test_get_attribute_applies_classmethod_descriptor_protocol() -> None:
    class Box:
        @classmethod
        def make(cls) -> int:
            return 1

    checker = Checker()
    attribute = checker.make_type_object(Box).get_attribute(
        "make", checker, on_class=False, receiver_value=TypedValue(Box)
    )

    assert attribute is not None
    assert isinstance(attribute.value, CallableValue)
    assert attribute.value.signature.return_value == TypedValue(int)
    assert not attribute.value.signature.parameters


def test_get_attribute_special_cases_instance_class_and_dict_for_runtime_types() -> (
    None
):
    class Box:
        pass

    checker = Checker()
    type_object = checker.make_type_object(Box)

    class_attr = type_object.get_attribute(
        "__class__", checker, on_class=False, receiver_value=TypedValue(Box)
    )
    assert class_attr is not None
    assert class_attr.owner.typ is Box
    assert isinstance(class_attr.value, AnyValue)
    assert class_attr.value.source is AnySource.inference

    dict_attr = type_object.get_attribute(
        "__dict__", checker, on_class=False, receiver_value=TypedValue(Box)
    )
    assert dict_attr is not None
    assert dict_attr.owner.typ is Box
    assert dict_attr.value == TypedValue(dict)


def test_get_attribute_uses_metaclass_members_for_class_object_access() -> None:
    class Meta(type):
        answer = 42

        def make(cls) -> int:
            return 1

    class Box(metaclass=Meta):
        pass

    checker = Checker()
    type_object = checker.make_type_object(Box)

    answer_attr = type_object.get_attribute(
        "answer", checker, on_class=True, receiver_value=TypedValue(Box)
    )
    assert answer_attr is not None
    assert answer_attr.owner.typ is Meta
    assert answer_attr.value == KnownValue(42)

    instance_answer_attr = type_object.get_attribute(
        "answer", checker, on_class=False, receiver_value=TypedValue(Box)
    )
    assert instance_answer_attr is None

    make_attr = type_object.get_attribute(
        "make", checker, on_class=True, receiver_value=TypedValue(Box)
    )
    assert make_attr is not None
    assert make_attr.owner.typ is Meta
    assert isinstance(make_attr.value, CallableValue)
    assert make_attr.value.signature.return_value == TypedValue(int)
    assert not make_attr.value.signature.parameters


def test_get_attribute_prefers_metaclass_property_on_class_access() -> None:
    class Meta(type):
        @property
        def value(cls) -> int:
            return 1

    class Box(metaclass=Meta):
        value = "class value"

    checker = Checker()
    attribute = checker.make_type_object(Box).get_attribute(
        "value", checker, on_class=True, receiver_value=TypedValue(Box)
    )

    assert attribute is not None
    assert attribute.owner.typ is Meta
    assert attribute.is_property
    assert attribute.value == TypedValue(int)


def test_get_attribute_resolves_runtime_custom_descriptor() -> None:
    from typing import Any

    class Descriptor:
        def __get__(self, obj: object | None, owner: Any) -> int:
            return 1

    class Box:
        value = Descriptor()

    checker = Checker()
    type_object = checker.make_type_object(Box)

    instance_attr = type_object.get_attribute(
        "value", checker, on_class=False, receiver_value=TypedValue(Box)
    )
    assert instance_attr is not None
    assert instance_attr.is_property
    assert instance_attr.value == TypedValue(int)


def test_get_attribute_resolves_runtime_generic_descriptor_instance_type() -> None:
    from typing import Any, Generic, TypeVar, overload

    T = TypeVar("T")

    class Descriptor(Generic[T]):
        @overload
        def __get__(self, obj: None, owner: Any) -> "Descriptor[T]": ...

        @overload
        def __get__(self, obj: object, owner: Any) -> T: ...

        def __get__(self, obj: object | None, owner: Any) -> "Descriptor[T] | T":
            return self if obj is None else 1

    class Box:
        value = Descriptor[int]()

    checker = Checker()
    attribute = checker.make_type_object(Box).get_attribute(
        "value", checker, on_class=False, receiver_value=TypedValue(Box)
    )

    assert attribute is not None
    assert attribute.is_property
    assert attribute.value == TypedValue(int)


def test_get_attribute_prefers_metaclass_data_descriptor_on_class_access() -> None:
    from typing import Any, overload

    class Descriptor:
        @overload
        def __get__(self, obj: None, owner: Any) -> "Descriptor": ...

        @overload
        def __get__(self, obj: object, owner: Any) -> int: ...

        def __get__(self, obj: object | None, owner: Any) -> "int | Descriptor":
            return self if obj is None else 1

        def __set__(self, obj: object, value: int) -> None:
            pass

    class Meta(type):
        value = Descriptor()

    class Box(metaclass=Meta):
        value = "class value"

    checker = Checker()
    attribute = checker.make_type_object(Box).get_attribute(
        "value", checker, on_class=True, receiver_value=TypedValue(Box)
    )

    assert attribute is not None
    assert attribute.is_metaclass_owner
    assert attribute.is_property
    assert attribute.property_has_setter
    assert attribute.value == TypedValue(int)


def test_get_attribute_uses_typeshed_types_for_type_descriptors() -> None:
    checker = Checker()
    type_object = checker.make_type_object(int)

    name_attr = type_object.get_attribute(
        "__name__", checker, on_class=True, receiver_value=TypedValue(int)
    )
    assert name_attr is not None
    assert name_attr.is_metaclass_owner
    assert name_attr.is_property
    assert name_attr.value == TypedValue(str)

    qualname_attr = type_object.get_attribute(
        "__qualname__", checker, on_class=True, receiver_value=TypedValue(int)
    )
    assert qualname_attr is not None
    assert qualname_attr.is_metaclass_owner
    assert qualname_attr.is_property
    assert qualname_attr.value == TypedValue(str)

    module_attr = type_object.get_attribute(
        "__module__", checker, on_class=True, receiver_value=TypedValue(int)
    )
    assert module_attr is not None
    assert module_attr.is_metaclass_owner
    assert module_attr.is_property
    assert module_attr.value == TypedValue(str)

    mro_attr = type_object.get_attribute(
        "__mro__", checker, on_class=True, receiver_value=TypedValue(int)
    )
    assert mro_attr is not None
    assert mro_attr.is_metaclass_owner
    assert mro_attr.is_property
    assert mro_attr.value == GenericValue(tuple, [TypedValue(type)])

    bases_attr = type_object.get_attribute(
        "__bases__", checker, on_class=True, receiver_value=TypedValue(int)
    )
    assert bases_attr is not None
    assert bases_attr.is_metaclass_owner
    assert bases_attr.is_property
    assert bases_attr.value == GenericValue(tuple, [TypedValue(type)])


class TestNumerics(TestNameCheckVisitorBase):
    @assert_passes()
    def test_float(self):
        from typing import NewType

        NT = NewType("NT", int)

        def take_float(x: float) -> None:
            pass

        class IntSubclass(int):
            pass

        def capybara(nt: NT, i: int, f: float) -> None:
            take_float(nt)
            take_float(i)
            take_float(f)
            take_float(3.0)
            take_float(3)
            take_float(1 + 1j)  # E: incompatible_argument
            take_float("string")  # E: incompatible_argument
            # bool is a subclass of int, which is treated as a subclass of float
            take_float(True)
            take_float(IntSubclass(3))

    @assert_passes()
    def test_complex(self):
        from typing import NewType

        NTI = NewType("NTI", int)
        NTF = NewType("NTF", float)

        def take_complex(c: complex) -> None:
            pass

        def capybara(nti: NTI, ntf: NTF, i: int, f: float, c: complex) -> None:
            take_complex(ntf)
            take_complex(nti)
            take_complex(i)
            take_complex(f)
            take_complex(c)
            take_complex(3.0)
            take_complex(3)
            take_complex(1 + 1j)
            take_complex("string")  # E: incompatible_argument
            take_complex(True)  # bool is an int, which is a float, which is a complex


class TestSyntheticType(TestNameCheckVisitorBase):
    def test_protocol_uses_typeshed_type_for_runtime_property(self):
        module_name = "_pycroscope_tests.stub_property"
        try:
            _make_runtime_property_class(module_name)
            self.assert_passes("""
                from typing import Protocol
                from _pycroscope_tests.stub_property import C

                class WantsIntName(Protocol):
                    @property
                    def name(self) -> int: ...

                def takes(value: WantsIntName) -> None:
                    pass

                def capybara(path: C) -> None:
                    takes(path)  # E: incompatible_argument
                """)
        finally:
            sys.modules.pop(module_name, None)
            sys.modules.pop("_pycroscope_tests", None)

    @assert_passes(run_in_both_module_modes=True)
    def test_synthetic_namedtuple_field_is_readonly_without_runtime_class() -> None:
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        def capybara(p: Point) -> None:
            p.x = 42  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_generic_property_specializes_in_protocol_check(self):
        from typing import Generic, Protocol, TypeVar

        T = TypeVar("T")

        class Base(Generic[T]):
            @property
            def value(self) -> T:
                raise NotImplementedError

        class Child(Base[str]):
            pass

        class WantsInt(Protocol):
            @property
            def value(self) -> int: ...

        def takes(value: WantsInt) -> None:
            pass

        def capybara(child: Child) -> None:
            takes(child)  # E: incompatible_argument

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_self_typevar_map_handles_classmethod_and_staticmethod(self):
        from typing import Protocol, TypeVar

        from typing_extensions import Self

        T_co = TypeVar("T_co", covariant=True)

        class Proto(Protocol[T_co]):
            def clone(self: T_co) -> T_co: ...

            @classmethod
            def make(cls: type[T_co]) -> T_co: ...

            @staticmethod
            def marker() -> None: ...

        class Good:
            def clone(self) -> Self:
                return self

            @classmethod
            def make(cls) -> Self:
                return cls()

            @staticmethod
            def marker() -> None:
                pass

        class BadClone:
            def clone(self) -> int:
                return 0

            @classmethod
            def make(cls) -> Self:
                return cls()

            @staticmethod
            def marker() -> None:
                pass

        good: Proto[Good] = Good()
        bad: Proto[BadClone] = BadClone()  # E: incompatible_assignment
        print(good, bad)

    @assert_passes()
    def test_callable_protocol_nonstandard_receiver_name(self):
        from typing import Protocol

        class CallableProto(Protocol):
            def __call__(receiver, x: int) -> int: ...  # E: method_first_arg

        def identity(x: int) -> int:
            return x

        def capybara() -> None:
            fn: CallableProto = identity
            print(fn)

    @assert_passes()
    def test_protocol_class_object_call_member_with_annotated_receiver(self):
        from typing import Protocol

        class Concrete:
            def __init__(self, x: int) -> None:
                self.x = x

        class Factory(Protocol):
            def __call__(self: "type[Concrete]", x: int) -> Concrete: ...

        factory: Factory = Concrete
        created: Concrete = factory(1)  # E: not_callable
        print(created)

    @assert_passes()
    def test_protocol_class_object_rejects_instance_property_data_member(self):
        from typing import Protocol

        class WantsValue(Protocol):
            value: int

        class Concrete:
            @property
            def value(self) -> int:
                return 1

        bad: WantsValue = Concrete  # E: incompatible_assignment
        print(bad)

    @assert_passes()
    def test_dunder_protocol_nonstandard_receiver_name(self):
        from collections.abc import Iterator
        from typing import Protocol

        class IterableProto(Protocol):
            def __iter__(receiver) -> Iterator[int]: ...  # E: method_first_arg

        class WithIter:
            def __iter__(self) -> Iterator[int]:
                yield 1

        def capybara() -> None:
            value: IterableProto = WithIter()
            print(value)

    @assert_passes()
    def test_overloaded_callable_protocols(self):
        from typing import Protocol

        from pycroscope.extensions import overload

        class OverloadedNarrow(Protocol):
            @overload
            def __call__(self, x: int) -> int: ...

            @overload
            def __call__(self, x: str) -> str: ...

        class FloatArg(Protocol):
            def __call__(self, x: float) -> float: ...

        class OverloadedWide(Protocol):
            @overload
            def __call__(self, x: int, y: str) -> float: ...

            @overload
            def __call__(self, x: str) -> complex: ...

        class IntStrArg(Protocol):
            def __call__(self, x: int | str, y: str = "") -> int: ...

        class StrArg(Protocol):
            def __call__(self, x: str) -> complex: ...

        def capybara(
            overloaded_narrow: OverloadedNarrow, int_str_arg: IntStrArg, str_arg: StrArg
        ) -> None:
            bad: FloatArg = overloaded_narrow  # E: incompatible_assignment
            ok: OverloadedWide = int_str_arg
            bad2: OverloadedWide = str_arg  # E: incompatible_assignment
            print(bad, ok, bad2)

    @assert_passes()
    def test_callable_protocol_instance_call_not_treated_as_instantiation(self):
        from typing import Protocol

        class IdentityFunction(Protocol):
            def __call__(self, x: int) -> int: ...

        def apply_identity(identity: IdentityFunction) -> int:
            return identity(1)

    @assert_passes()
    def test_paramspec_callable_protocol_equivalence(self):
        from typing import Callable, ParamSpec, Protocol, TypeAlias

        P = ParamSpec("P")

        class ProtocolWithP(Protocol[P]):
            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

        TypeAliasWithP: TypeAlias = Callable[P, None]

        def capybara(proto: ProtocolWithP[P], ta: TypeAliasWithP[P]) -> None:
            as_callable: TypeAliasWithP[P] = proto
            as_protocol: ProtocolWithP[P] = ta
            print(as_callable, as_protocol)

    @skip_before((3, 11))
    def test_protocol_receiver_constraints_with_typevartuple(self):
        self.assert_passes(
            """
            from typing import Generic, Protocol, TypeVar, TypeVarTuple

            T_co = TypeVar("T_co", covariant=True)
            Ts = TypeVarTuple("Ts")

            class Box(Generic[T_co, *Ts]):
                def head(self) -> T_co:
                    raise NotImplementedError

                def tail(self) -> tuple[*Ts]:
                    raise NotImplementedError

            class Proto(Protocol[T_co, *Ts]):
                def head(self: Box[T_co, *Ts]) -> T_co: ...

                def tail(self: Box[T_co, *Ts]) -> tuple[*Ts]: ...

            def wants_proto(proto: Proto[int, str, bool]) -> None:
                print(proto)

            def capybara(box: Box[int, str, bool]) -> None:
                as_proto: Proto[int, str, bool] = box
                wants_proto(box)
                wants_proto(as_proto)
            """,
            run_in_both_module_modes=True,
        )

    @assert_passes(run_in_both_module_modes=True)
    def test_callable_annotation_protocol_interop(self):
        from typing import Any, Callable, ParamSpec, Protocol, TypeVar

        T_contra = TypeVar("T_contra", contravariant=True)
        P = ParamSpec("P")

        class ProtoAnyTail(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class ProtoFixedTail(Protocol):
            def __call__(self, a: int, *args: Any, **kwargs: Any) -> None: ...

        class ProtoParamSpecAny(Protocol[P]):
            def __call__(self, a: int, *args: P.args, **kwargs: P.kwargs) -> None: ...

        class ProtoStrict(Protocol):
            def __call__(self, a: float, b: int, *, k: str, m: str) -> None: ...

        class ProtoZero(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Proto5(Protocol[T_contra]):
            def __call__(self, *args: T_contra, **kwargs: T_contra) -> None: ...

        class Proto8(Protocol):
            def __call__(self) -> None: ...

        def capybara(
            p_any: ProtoAnyTail,
            p_ps: ProtoParamSpecAny[...],
            p5: Proto5[Any],
            p_strict: ProtoStrict,
            p8: Proto8,
            c1: Callable[..., None],
            c2: ProtoFixedTail,
        ) -> None:
            ok1: ProtoAnyTail = c2
            ok2: ProtoFixedTail = p_ps
            ok3: ProtoFixedTail = p_strict
            ok4: ProtoParamSpecAny[...] = p_any  # keep reverse direction covered
            ok5: Proto5[Any] = c1
            err1: Proto5[Any] = p8  # E: incompatible_assignment
            print(ok1, ok2, ok3, ok4, ok5, err1)

    @assert_passes()
    def test_functools(self):
        import functools
        import types

        from pycroscope.signature import ELLIPSIS_PARAM, Signature

        sig = Signature.make([ELLIPSIS_PARAM], return_annotation=TypedValue(int))

        def f() -> int:
            return 0

        def capybara():
            c = functools.singledispatch(f)
            assert_is_value(
                c, GenericValue("functools._SingleDispatchCallable", [TypedValue(int)])
            )
            assert_is_value(
                c.registry,
                GenericValue(
                    types.MappingProxyType,
                    [AnyValue(AnySource.explicit), CallableValue(sig)],
                ),
            )
            assert_type(c._clear_cache(), None)
            assert_type(c(), int)
            c.doesnt_exist  # E: undefined_attribute

    @assert_passes()
    def test_protocol(self):
        # Note that csv.writer expects this protocol:
        # class _Writer(Protocol):
        #    def write(self, s: str) -> Any: ...
        import csv
        import io

        class BadWrite:
            def write(self, s: int) -> object:
                return object()

        class GoodWrite:
            def write(self, s: str) -> object:
                return object()

        class BadArgKind:
            def write(self, *, s: str) -> object:
                return object()

        def capybara(s: str):
            writer = io.StringIO()
            # Ideally we'd test the return type but it's a private type
            # and the exact one changes between typeshed versions.
            csv.writer(writer)

            csv.writer(1)  # E: incompatible_argument
            csv.writer(s)  # E: incompatible_argument
            csv.writer(BadWrite())  # E: incompatible_argument
            csv.writer(GoodWrite())
            csv.writer(BadArgKind())  # E: incompatible_argument

    @assert_passes(allow_import_failures=True)
    def test_protocol_generic_base_after_import_failure(self):
        from typing import Hashable, Iterable, Protocol

        class P0(Protocol):
            pass

        P0()  # E: incompatible_call

        class HashableFloats(Iterable[float], Hashable, Protocol):
            pass

        def cached_func(args: HashableFloats) -> float:
            return 0.0

        cached_func((1, 2, 3))

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_subtyping_after_import_failure(self):
        from typing import Protocol, Sequence, TypeVar

        class Proto1(Protocol):
            pass

        class Proto2(Protocol):
            def method1(self) -> None: ...

        class Concrete2:
            def method1(self) -> None:
                pass

        def func1(p2: Proto2, c2: Concrete2) -> None:
            v1: Proto2 = c2
            v2: Concrete2 = p2  # E: incompatible_assignment
            print(v1, v2)

        class Proto3(Protocol):
            def method1(self) -> None: ...

            def method2(self) -> None: ...

        def func2(p2: Proto2, p3: Proto3) -> None:
            v1: Proto2 = p3
            v2: Proto3 = p2  # E: incompatible_assignment
            print(v1, v2)

        S = TypeVar("S")
        T = TypeVar("T")

        class Proto4(Protocol[S, T]):
            def method1(self, a: S, b: T) -> tuple[S, T]: ...

        class Proto5(Protocol[T]):
            def method1(self, a: T, b: T) -> tuple[T, T]: ...

        def func3(p4_int: Proto4[int, int], p5_int: Proto5[int]) -> None:
            v1: Proto4[int, int] = p5_int
            v2: Proto5[int] = p4_int
            v3: Proto4[int, float] = p5_int  # E: incompatible_assignment
            v4: Proto5[float] = p4_int  # E: incompatible_assignment
            print(v1, v2, v3, v4)

        S_co = TypeVar("S_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class Proto6(Protocol[S_co, T_contra]):
            def method1(self, a: T_contra) -> Sequence[S_co]: ...

        class Proto7(Protocol[S_co, T_contra]):
            def method1(self, a: T_contra) -> Sequence[S_co]: ...

        def func4(p6: Proto6[float, float]) -> None:
            v1: Proto7[object, int] = p6
            v2: Proto7[float, float] = p6
            v3: Proto7[int, float] = p6  # E: incompatible_assignment
            v4: Proto7[float, object] = p6  # E: incompatible_assignment
            print(v1, v2, v3, v4)

        def capybara() -> None:
            Proto1()  # E: incompatible_call

    @assert_passes()
    def test_protocol_shorthand_type_parameter_order(self):
        from typing import Iterator, Protocol, TypeVar

        S = TypeVar("S")
        T_co = TypeVar("T_co", covariant=True)

        class Iterable(Protocol[T_co]):
            def __iter__(self) -> Iterator[T_co]: ...

        class Proto1(Iterable[T_co], Protocol[S, T_co]):
            def method1(self, x: S) -> S: ...

        class Concrete1:
            def __iter__(self) -> Iterator[int]:
                return iter((1, 2, 3))

            def method1(self, x: str) -> str:
                return x

        ok: Proto1[str, int] = Concrete1()
        bad: Proto1[int, str] = Concrete1()  # E: incompatible_assignment
        print(ok, bad)

    @assert_passes()
    def test_protocol_member_constraints_must_be_satisfiable(self):
        from typing import Callable, Protocol, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class HasPropertyProto(Protocol):
            @property
            def f(self: T) -> T: ...

            def m(self, item: T, callback: Callable[[T], str]) -> str: ...

        class ConcreteHasProperty1:
            @property
            def f(self: T) -> T:
                return self

            def m(self, item: T, callback: Callable[[T], str]) -> str:
                return ""

        class ConcreteHasProperty2:
            @property
            def f(self) -> Self:
                return self

            def m(self, item: int, callback: Callable[[int], str]) -> str:
                return ""

        class ConcreteHasProperty4:
            @property
            def f(self) -> Self:
                return self

            def m(self, item: str, callback: Callable[[int], str]) -> str:
                return ""

        hp1: HasPropertyProto = ConcreteHasProperty1()
        hp2: HasPropertyProto = ConcreteHasProperty2()  # E: incompatible_assignment
        hp4: HasPropertyProto = ConcreteHasProperty4()  # E: incompatible_assignment
        print(hp1, hp2, hp4)

    @assert_passes(allow_import_failures=True)
    def test_recursive_protocol_classmethod_constraints(self):
        from typing import Never, Protocol, Self, TypeVar, assert_type

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class ProtoA(Protocol[T_co, T_contra]):
            def method1(self) -> "ProtoA[T_co, T_contra]": ...

            @classmethod
            def method2(cls, value: T_contra) -> None: ...

        class ProtoB(Protocol[T_co, T_contra]):
            def method3(self) -> ProtoA[T_co, T_contra]: ...

        class ImplA:
            def method1(self) -> Self:
                return self

            @classmethod
            def method2(cls, value: int) -> None:
                pass

        class ImplB:
            def method3(self) -> ImplA:
                return ImplA()

            def method1(self) -> Self:
                return self

            @classmethod
            def method2(cls: type[ProtoB[object, T]], value: list[T]) -> None:
                pass

        def func1(x: ProtoA[Never, T]) -> T:
            raise NotImplementedError

        v1 = func1(ImplB())
        assert_type(v1, list[int])

    @assert_passes(allow_import_failures=True)
    def test_protocol_member_semantics_after_import_failure(self):
        from dataclasses import dataclass
        from typing import ClassVar, NamedTuple, Protocol, Sequence

        class WantsClassVar(Protocol):
            val: ClassVar[Sequence[int]]

        class HasInstanceVal:
            val: Sequence[int] = [0]

        bad_classvar: WantsClassVar = HasInstanceVal()  # E: incompatible_assignment

        class WantsDataMember(Protocol):
            val: Sequence[int]

        class HasClassVar:
            val: ClassVar[Sequence[int]] = [0]

        class HasReadOnlyProperty:
            @property
            def val(self) -> Sequence[int]:
                return [0]

        class HasMutableList:
            val: list[int] = [0]

        bad_data1: WantsDataMember = HasClassVar()  # E: incompatible_assignment
        bad_data2: WantsDataMember = HasReadOnlyProperty()  # E: incompatible_assignment
        bad_data3: WantsDataMember = HasMutableList()  # E: incompatible_assignment

        class WantsReadOnlyProperty(Protocol):
            @property
            def val(self) -> Sequence[float]: ...

        class PlainAttrForReadOnly:
            val: Sequence[float] = [0]

        ok_read_only: WantsReadOnlyProperty = PlainAttrForReadOnly()

        class WantsSettableProperty(Protocol):
            @property
            def val(self) -> Sequence[float]: ...

            @val.setter
            def val(self, value: Sequence[float]) -> None: ...

        class SettablePropertyImpl:
            @property
            def val(self) -> Sequence[float]:
                return [0]

            @val.setter
            def val(self, value: Sequence[float]) -> None:
                pass

        class PlainAttrImpl:
            val: Sequence[float] = [0]

        class ReadOnlyPropertyImpl:
            @property
            def val(self) -> Sequence[float]:
                return [0]

        class NamedTupleImpl(NamedTuple):
            val: Sequence[float] = [0]

        @dataclass(frozen=True)
        class FrozenDataclassImpl:
            val: Sequence[float] = [0]

        ok_settable1: WantsSettableProperty = SettablePropertyImpl()
        ok_settable2: WantsSettableProperty = PlainAttrImpl()
        bad_settable1: WantsSettableProperty = (  # E: incompatible_assignment
            ReadOnlyPropertyImpl()
        )
        bad_settable2: WantsSettableProperty = (  # E: incompatible_assignment
            NamedTupleImpl()
        )
        bad_settable3: WantsSettableProperty = (  # E: incompatible_assignment
            FrozenDataclassImpl()
        )
        print(
            bad_classvar,
            bad_data1,
            bad_data2,
            bad_data3,
            ok_read_only,
            ok_settable1,
            ok_settable2,
            bad_settable1,
            bad_settable2,
            bad_settable3,
        )

    @assert_passes()
    def test_protocol_readonly_data_member(self):
        from functools import cached_property
        from typing import Protocol, Sequence

        from typing_extensions import ReadOnly

        class WantsReadOnlyData(Protocol):
            val: ReadOnly[Sequence[float]]

        class PlainAttrImpl:
            val: Sequence[float] = [0]

        class ReadOnlyAttrImpl:
            val: ReadOnly[Sequence[float]]

            def __init__(self) -> None:
                self.val = [0]

        class PropertyImpl:
            @property
            def val(self) -> Sequence[float]:
                return [0]

        class CachedPropertyImpl:
            @cached_property
            def val(self) -> Sequence[float]:
                return [0]

        class BadTypeImpl:
            val: Sequence[str] = ["x"]

        ok1: WantsReadOnlyData = PlainAttrImpl()
        ok2: WantsReadOnlyData = ReadOnlyAttrImpl()
        ok3: WantsReadOnlyData = PropertyImpl()
        ok4: WantsReadOnlyData = CachedPropertyImpl()
        bad: WantsReadOnlyData = BadTypeImpl()  # E: incompatible_assignment
        print(ok1, ok2, ok3, ok4, bad)

    @assert_passes()
    def test_protocol_with_runtime_property_without_getter(self):
        from typing import Protocol

        class WantsSettable(Protocol):
            @property
            def value(self) -> int: ...

            @value.setter
            def value(self, new_value: int) -> None: ...

        class WeirdProperty:
            value = property(None, object())  # E: incompatible_argument

        maybe_ok: WantsSettable = WeirdProperty()
        print(maybe_ok)

    @assert_passes()
    def test_base_value_conversion_covers_runtime_shape_helpers(self):
        from typing_extensions import TypeIs

        class A:
            pass

        class B:
            pass

        class Base:
            pass

        HomTuple = tuple[int, ...]
        FixedTuple = tuple[int, str]

        class LoopBase:
            def __mro_entries__(self, bases):
                return (self,)

        loop = LoopBase()

        def is_a_class(cls: object) -> TypeIs[type[A]]:
            return True

        def capybara(flag: bool, cls: type[Base], narrowed: type[A] | type[B]) -> None:
            class FromUnion(A if flag else B):
                pass

            class FromSubclassValue(cls):
                pass

            class FromDirectTuple(tuple[int, str]):
                pass

            class FromFixedTupleAlias(FixedTuple):
                pass

            class FromHomTupleAlias(HomTuple):
                pass

            class FromBitOr(A | B):
                pass

            class FromLoop(loop):
                pass

            if is_a_class(narrowed):

                class FromIntersection(narrowed):
                    pass

                print(FromIntersection)

            print(
                FromUnion,
                FromSubclassValue,
                FromDirectTuple,
                FromFixedTupleAlias,
                FromHomTupleAlias,
                FromBitOr,
                FromLoop,
            )

    @assert_passes(run_in_both_module_modes=True)
    def test_runtime_type_object_handles_odd_runtime_metadata_shapes(self):
        from dataclasses import dataclass
        from types import GenericAlias
        from typing import Any, Generic, NamedTuple, TypeVar, cast, final

        from typing_extensions import override

        T = TypeVar("T")

        class Box(Generic[T]):
            pass

        class EmptyOrig:
            __orig_class__ = list[int]

        class MismatchOrig:
            __orig_class__ = GenericAlias(Box, (int, str))

        @dataclass
        class Data:
            x: int

        cast(Any, Data.__dataclass_fields__)[1] = Data.__dataclass_fields__["x"]
        cast(Any, Data.__dataclass_fields__["x"]).init = 0

        class NT(NamedTuple):
            x: int

        cast(Any, NT)._fields = 1
        setattr(NT, "__annotations__", 1)

        class Runtime:
            x = 1
            empty = EmptyOrig()
            mismatch = MismatchOrig()

            @final
            def plain(self) -> int:
                return 1

            def chosen(self) -> int:
                return 1

        class Child(Runtime):
            @override
            def chosen(self) -> int:
                return 2

            @property
            def value(self) -> int:
                return 1

            @value.setter
            def value(self, new_value) -> None:
                pass

        ok1 = Data(1).x
        ok2 = Runtime.x
        ok3 = NT(1)
        ok4 = Child().value
        print(ok1, ok2, ok3, ok4)

    @assert_passes(run_in_both_module_modes=True)
    def test_namedtuple_with_non_mapping_annotations_loses_class_attribute(self):
        from typing import Any, NamedTuple, cast

        from typing_extensions import assert_type

        class NT(NamedTuple):
            x: int

        cast(Any, NT)._fields = 1
        setattr(NT, "__annotations__", 1)

        print(NT.x)  # E: undefined_attribute
        assert_type(NT.x, object)  # E: undefined_attribute  # E: inference_failure

    @assert_passes(run_in_both_module_modes=True)
    def test_runtime_generic_type_param_defaults_use_runtime_types(self):
        from dataclasses import dataclass
        from typing import Generic

        from typing_extensions import TypeVar, assert_type

        T = TypeVar("T", default=int)
        U = TypeVar("U", default=str)

        @dataclass
        class Box(Generic[T, U]):
            first: T
            second: U

        i: int = 1
        s: str = "x"

        assert_type(Box(i, s).first, int)
        assert_type(Box(i, s).second, str)
        assert_type(Box[bool](True, s).first, bool)
        assert_type(Box[bool](True, s).second, str)

    @assert_passes()
    def test_protocol_hash_method_accepts_class_object_metaclass_hash(self):
        from typing import Protocol

        class HashProto(Protocol):
            def __hash__(self) -> int: ...

        class Meta(type):
            def __hash__(self, extra: int = 0) -> int:
                return extra

        class Concrete(metaclass=Meta):
            pass

        ok: HashProto = Concrete
        print(ok)

    @assert_passes()
    def test_protocol_writable_data_member_rejects_method_member(self):
        from typing import Protocol

        class WantsWritable(Protocol):
            value: int

        class HasMethod:
            def value(self) -> int:
                return 1

        bad: WantsWritable = HasMethod()  # E: incompatible_assignment
        print(bad)

    @assert_passes()
    def test_runtime_type_object_skips_non_string_class_entries(self):
        from typing import Protocol

        class WantsX(Protocol):
            x: int

        class Weird:
            locals()[1] = 2  # E: incompatible_argument
            __annotations__ = {1: int, "x": int}
            x = 1

        ok: WantsX = Weird()
        print(ok)

    @assert_passes()
    def test_runtime_type_object_falls_back_when_annotations_access_raises(self):
        from typing import Protocol

        class WantsX(Protocol):
            x: int

        class Meta(type):
            @property
            def __annotations__(self):
                raise RuntimeError("boom")

        class C(metaclass=Meta):
            x = 1

        ok: WantsX = C()
        print(ok)

    @assert_passes()
    def test_protocol_accepts_property_subclass_that_raises_on_class_access(self):
        from typing import Protocol

        class WantsX(Protocol):
            @property
            def x(self) -> int: ...

        class RaisingProperty(property):
            def __get__(self, obj, objtype=None):
                if obj is None:
                    raise RuntimeError("boom")
                return super().__get__(obj, objtype)

        def get_x(self) -> int:
            return 1

        class C:
            x = RaisingProperty(get_x)

        ok: WantsX = C()
        print(ok)

    @assert_passes()
    def test_protocol_accepts_annotated_runtime_property(self):
        from typing import Protocol

        class WantsX(Protocol):
            @property
            def x(self) -> int: ...

        class C:
            @property
            def x(self) -> int:
                return 1

            __annotations__ = {"x": int}

        ok: WantsX = C()
        print(ok)

    @assert_passes()
    def test_namedtuple_with_non_tuple_fields_still_builds_type_object(self):
        from typing import NamedTuple, Protocol

        class WantsX(Protocol):
            x: int

        class Point(NamedTuple):
            x: int

        Point._fields = 1

        ok: WantsX = Point(1)
        print(ok)

    @assert_passes()
    def test_custom_subclasscheck(self):
        class _ThriftEnumMeta(type):
            def __subclasscheck__(self, subclass):
                return hasattr(subclass, "_VALUES_TO_NAMES")

        class ThriftEnum(metaclass=_ThriftEnumMeta):
            pass

        class IsOne:
            _VALUES_TO_NAMES = {}

        class IsntOne:
            _NAMES_TO_VALUES = {}

        def want_enum(te: ThriftEnum) -> None:
            pass

        def capybara(good_instance: IsOne, bad_instance: IsntOne, te: ThriftEnum):
            want_enum(good_instance)
            want_enum(bad_instance)  # E: incompatible_argument
            want_enum(IsOne())
            want_enum(IsntOne())  # E: incompatible_argument
            want_enum(te)

    @assert_passes()
    def test_generic_stubonly(self):
        import pkgutil

        # pkgutil.read_code requires SupportsRead[bytes]

        class Good:
            def read(self, length: int = 0) -> bytes:
                return b""

        class Bad:
            def read(self, length: int = 0) -> str:
                return ""

        def capybara():
            pkgutil.read_code(1)  # E: incompatible_argument
            pkgutil.read_code(Good())
            pkgutil.read_code(Bad())  # E: incompatible_argument

    @assert_passes()
    def test_protocol_inheritance(self):
        import operator

        # operator.getitem requires SupportsGetItem[K, V]

        class Good:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: str) -> str:
                raise KeyError(k)

        class Bad:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: bytes) -> str:
                raise KeyError(k)

        def capybara():
            operator.getitem(Good(), "hello")
            operator.getitem(Bad(), "hello")  # E: incompatible_call
            operator.getitem(1, "hello")  # E: incompatible_argument

    @assert_passes()
    def test_iterable(self):
        from typing import Iterable, Iterator

        class Bad:
            def __iter__(self, some, random, args):
                pass

        class Good:
            def __iter__(self) -> Iterator[int]:
                raise NotImplementedError

        class BadType:
            def __iter__(self) -> Iterator[str]:
                raise NotImplementedError

        def want_iter_int(f: Iterable[int]) -> None:
            pass

        def capybara():
            want_iter_int(Bad())  # E: incompatible_argument
            want_iter_int(Good())
            want_iter_int(BadType())  # E: incompatible_argument

    @assert_passes()
    def test_self_iterator(self):
        from typing import Iterator

        class MyIter:
            def __iter__(self) -> "MyIter":
                return self

            def __next__(self) -> int:
                return 42

        def want_iter(it: Iterator[int]):
            pass

        def capybara():
            want_iter(MyIter())

    @assert_passes()
    def test_container(self):
        from typing import Any, Container

        class Good:
            def __contains__(self, whatever: object) -> bool:
                return False

        class Bad:
            def __contains__(self, too, many, arguments) -> bool:
                return True

        def want_container(c: Container[Any]) -> None:
            pass

        def capybara() -> None:
            want_container(Bad())  # E: incompatible_argument
            want_container(Good())
            want_container([1])
            want_container(1)  # E: incompatible_argument

    @assert_passes()
    def test_runtime_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            a: int

            def b(self) -> int:
                raise NotImplementedError

        class Q(P, Protocol):
            c: str

        class NotAProtocol(P):
            c: str

        def want_p(x: P):
            print(x.a + x.b())

        def want_q(q: Q):
            pass

        def want_not_a_proto(nap: NotAProtocol):
            pass

        class GoodP:
            a: int

            def b(self) -> int:
                return 3

        class BadP:
            def a(self) -> int:
                return 5

            def b(self) -> int:
                return 4

        class GoodQ(GoodP):
            c: str

        class BadQ(GoodP):
            c: float

        def capybara():
            want_p(GoodP())
            want_p(BadP())  # E: incompatible_argument
            want_q(GoodQ())
            want_q(BadQ())  # E: incompatible_argument
            want_not_a_proto(GoodQ())  # E: incompatible_argument

    @assert_passes()
    def test_callable_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            def __call__(self, x: int) -> str:
                return str(x)

        def want_p(p: P) -> str:
            return p(1)

        def good(x: int) -> str:
            return "hello"

        def bad(x: str) -> str:
            return x

        def capybara():
            want_p(good)
            want_p(bad)  # E: incompatible_argument


class TestHashable(TestNameCheckVisitorBase):
    @assert_passes()
    def test_type(self):
        from typing import Hashable, Type

        from typing_extensions import Protocol

        class MyHashable(Protocol):
            def __hash__(self) -> int:
                raise NotImplementedError

        def want_hash(h: Hashable):
            pass

        def want_myhash(h: MyHashable):
            pass

        class A:
            pass

        class B:
            def __hash__(self) -> int:
                return 42

        def capybara(t1: Type[int], t2: type, x: list[int]):
            want_hash(t1)
            want_hash(t2)
            want_hash(int)
            want_hash(A)
            want_hash(B)

            {t1: 0}
            {t2: 0}
            {int: 0}
            {A: 0}

            want_hash(x)  # E: incompatible_argument
            want_hash([x])  # E: incompatible_argument
            want_hash([])  # E: incompatible_argument
            want_myhash(x)  # E: incompatible_argument
            want_myhash([x])  # E: incompatible_argument
            want_myhash([])  # E: incompatible_argument


class TestIO(TestNameCheckVisitorBase):
    @assert_passes()
    def test_text(self):
        import io
        from typing import TextIO

        def want_io(x: TextIO):
            x.write("hello")

        def capybara():
            with open("x") as f:
                assert_type(f, io.TextIOWrapper)
                want_io(f)

    @assert_passes()
    def test_binary(self):
        import io
        from typing import BinaryIO

        def want_io(x: BinaryIO):
            x.write(b"hello")

        def capybara():
            with open("x", "rb") as f:
                assert_type(f, io.BufferedReader)
                want_io(f)

        def pacarana():
            with open("x", "w+b") as f:
                assert_type(f, io.BufferedRandom)
                want_io(f)
