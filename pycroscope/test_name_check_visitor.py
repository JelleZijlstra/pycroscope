# static analysis: ignore
import ast
import collections
import os
import types

from . import test_node_visitor
from .analysis_lib import make_module
from .checker import Checker
from .error_code import DISABLED_IN_TESTS, ErrorCode
from .implementation import assert_is_value, dump_value
from .name_check_visitor import ClassAttributeChecker, NameCheckVisitor, _static_hasattr
from .test_config import CONFIG_PATH
from .test_node_visitor import (
    assert_fails,
    assert_passes,
    skip_before,
    skip_if_not_installed,
)
from .tests import make_simple_sequence
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    UNRESOLVED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    AsyncTaskIncompleteValue,
    CallableValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    KVPair,
    MultiValuedValue,
    NewTypeValue,
    ReferencingValue,
    SequenceValue,
    SubclassValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    UnboundMethodValue,
    VariableNameValue,
)

# ===================================================
# Base classes for test_scope tests.
#
# These are also used in scope_lib files.
# ===================================================


class ConfiguredNameCheckVisitor(NameCheckVisitor):
    """NameCheckVisitor configured for testing.

    Should be called TestNameCheckVisitor but that would cause it to be confused with test classes.

    """

    config_filename = str(CONFIG_PATH)


class TestNameCheckVisitorBase(test_node_visitor.BaseNodeVisitorTester):
    visitor_cls = ConfiguredNameCheckVisitor

    def _run_tree(
        self,
        code_str,
        tree,
        check_attributes=True,
        apply_changes=False,
        settings=None,
        allow_runtime_module_load_failure=False,
        **kwargs,
    ):
        # This can happen in Python 2.
        if isinstance(code_str, bytes):
            code_str = code_str.decode("utf-8")
        default_settings = {code: code not in DISABLED_IN_TESTS for code in ErrorCode}
        if settings is not None:
            default_settings.update(settings)
        verbosity = int(os.environ.get("ANS_TEST_SCOPE_VERBOSITY", 0))
        try:
            mod = _make_module(code_str)
        except Exception:
            if not allow_runtime_module_load_failure:
                raise
            mod = None
        kwargs["settings"] = default_settings
        kwargs = self.visitor_cls.prepare_constructor_kwargs(kwargs)
        new_code = ""
        with ClassAttributeChecker(
            enabled=check_attributes, options=kwargs["checker"].options
        ) as attribute_checker:
            visitor_kwargs = dict(
                attribute_checker=attribute_checker, verbosity=verbosity, **kwargs
            )
            if mod is None:
                visitor = self.visitor_cls(
                    "<test input>", code_str, tree, **visitor_kwargs
                )
            else:
                visitor = self.visitor_cls(
                    mod.__name__, code_str, tree, module=mod, **visitor_kwargs
                )
            result = visitor.check_for_test(apply_changes=apply_changes)
            if apply_changes:
                result, new_code = result
            result += visitor.perform_final_checks(kwargs)
        if apply_changes:
            return result, new_code
        return result


class TestAnnotatingNodeVisitor(test_node_visitor.BaseNodeVisitorTester):
    """Base class for testing AnnotatingNodeVisitor subclasses."""

    def _run_tree(self, code_str, tree, **kwargs):
        """Runs the visitor on this tree."""
        kwargs["module"] = _make_module(code_str)
        return super(TestAnnotatingNodeVisitor, self)._run_tree(
            code_str, tree, **kwargs
        )


def _make_module(code_str: str) -> types.ModuleType:
    """Creates a Python module with the given code."""
    # make helpers for value inference checking available to all tests
    extra_scope = dict(
        assert_is_value=assert_is_value,
        AsyncTaskIncompleteValue=AsyncTaskIncompleteValue,
        CallableValue=CallableValue,
        DictIncompleteValue=DictIncompleteValue,
        KVPair=KVPair,
        TypedDictEntry=TypedDictEntry,
        GenericValue=GenericValue,
        KnownValue=KnownValue,
        MultiValuedValue=MultiValuedValue,
        AnnotatedValue=AnnotatedValue,
        SequenceValue=SequenceValue,
        TypedValue=TypedValue,
        UnboundMethodValue=UnboundMethodValue,
        AnySource=AnySource,
        AnyValue=AnyValue,
        UNRESOLVED_VALUE=UNRESOLVED_VALUE,
        VariableNameValue=VariableNameValue,
        ReferencingValue=ReferencingValue,
        SubclassValue=SubclassValue,
        NewTypeValue=NewTypeValue,
        TypedDictValue=TypedDictValue,
        TypeVarValue=TypeVarValue,
        dump_value=dump_value,
        make_simple_sequence=make_simple_sequence,
        UNINITIALIZED_VALUE=UNINITIALIZED_VALUE,
        NO_RETURN_VALUE=NO_RETURN_VALUE,
    )
    return make_module(code_str, extra_scope)


# ===================================================
# Tests for specific functionality.
# ===================================================


def test_annotation():
    tree = ast.Call(ast.Name("int", ast.Load()), [], [])
    checker = Checker()
    ConfiguredNameCheckVisitor(
        "<test input>", "int()", tree, module=ast, annotate=True, checker=checker
    ).check()
    assert TypedValue(int) == tree.inferred_value


class TestImportFailureHandling(TestNameCheckVisitorBase):
    @assert_passes(allow_import_failures=True)
    def test_import_failure_points_to_failing_line_and_continues(self):
        a = 1
        b = 1 / 0

        def f():
            return missing_name  # E: undefined_name

    @assert_passes(allow_runtime_module_load_failure=True)
    def test_import_failure_is_ignorable(self):
        a = 1  # static analysis: ignore[import_failed]
        b = 1 / 0

        def f():
            return missing_name  # E: undefined_name

    @assert_passes(allow_import_failures=True)
    def test_typeddict_fallback_after_import_failure(self):
        from typing import TypedDict

        from typing_extensions import NotRequired, ReadOnly, Required

        class F1(TypedDict):
            a: Required[int]
            b: ReadOnly[NotRequired[int]]
            c: ReadOnly[Required[int]]

        boom = 1 / 0

        class F3(F1):
            a: ReadOnly[int]  # E: invalid_annotation

        class F4(F1):
            a: NotRequired[int]  # E: invalid_annotation

        class F5(F1):
            b: ReadOnly[Required[int]]

        class F6(F1):
            c: ReadOnly[NotRequired[int]]  # E: invalid_annotation

        class TD_A1(TypedDict):
            x: int
            y: ReadOnly[int]

        class TD_A2(TypedDict):
            x: float
            y: ReadOnly[float]

        class TD_A(TD_A1, TD_A2): ...  # E: invalid_annotation

        class TD_B1(TypedDict):
            x: ReadOnly[NotRequired[int]]
            y: ReadOnly[Required[int]]

        class TD_B2(TypedDict):
            x: ReadOnly[Required[int]]
            y: ReadOnly[NotRequired[int]]

        class TD_B(TD_B1, TD_B2): ...  # E: invalid_annotation

    @assert_passes(allow_import_failures=True)
    def test_typeddict_extra_items_and_unpack_after_import_failure(self):
        from typing_extensions import TypedDict, Unpack

        class Movie(TypedDict, extra_items=bool):
            name: str

        MovieFunctional = TypedDict("MovieFunctional", {"name": str}, extra_items=bool)

        boom = {}.popitem()

        a: Movie = {"name": "Blade Runner", "year": 1982}  # E: incompatible_assignment
        # E: incompatible_assignment
        b: MovieFunctional = {"name": "Blade Runner", "year": 1982}

        class MovieNoExtra(TypedDict):
            name: str

        class MovieExtra(TypedDict, extra_items=int):
            name: str

        def unpack_no_extra(**kwargs: Unpack[MovieNoExtra]) -> None: ...

        def unpack_extra(**kwargs: Unpack[MovieExtra]) -> None: ...

        # E: incompatible_call
        unpack_no_extra(name="No Country for Old Men", year=2007)
        unpack_extra(name="No Country for Old Men", year=2007)

    @assert_passes(allow_import_failures=True)
    def test_constructor_explicit_self_annotation_after_import_failure(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class C(Generic[T]):
            def __init__(self: "C[int]") -> None:
                pass

        boom = 1 / 0

        C[int]()
        C[str]()  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_nominal_class_fallback_after_import_failure(self):
        from typing import Any, overload

        boom = 1 / 0

        class Desc:
            value: int = 0

            @overload
            def __get__(self, obj: None, owner: Any) -> "Desc": ...

            @overload
            def __get__(self, obj: object, owner: Any) -> int: ...

            def __get__(self, obj: object | None, owner: Any) -> "int | Desc":
                return 1

    @assert_passes(allow_import_failures=True)
    def test_inherited_class_attribute_after_import_failure(self):
        boom = 1 / 0

        class Base:
            @staticmethod
            def foo() -> int:
                return 1

        class Child(Base):
            pass

        x: int = Child.foo()

    @assert_passes(allow_import_failures=True)
    def test_any_base_class_after_import_failure(self):
        from typing import Any

        class ClassA(Any):
            def method1(self) -> int:
                return 1

        a = ClassA()
        x: int = a.method1()
        y = a.method2()
        z = ClassA.method3()
        y.nonexistent_attribute
        z.nonexistent_attribute

    @assert_passes(allow_import_failures=True)
    def test_typevar_annotations_after_import_failure(self):
        from typing import TypeVar

        from typing_extensions import assert_type

        class User: ...

        class TeamUser(User): ...

        U = TypeVar("U", bound=User)

        def func3(user_class: type[U]) -> U:
            return user_class()

        assert_type(func3(TeamUser), TeamUser)
        type.unknown  # E: undefined_attribute

    @assert_passes(allow_import_failures=True)
    def test_type_union_annotation_after_import_failure(self):
        class User: ...

        class BasicUser(User): ...

        class ProUser(User): ...

        class TeamUser(User): ...

        def func4(user_class: type[BasicUser | ProUser]) -> User:
            return user_class()

        func4(TeamUser)  # E: incompatible_argument
        type.unknown  # E: undefined_attribute

    @assert_passes(allow_import_failures=True)
    def test_type_arity_and_typing_alias_attrs_after_import_failure(self):
        from typing import Any, Type, TypeAlias

        _bad_type1: type[int, str]  # E: invalid_annotation

        TA1: TypeAlias = Type
        TA2: TypeAlias = Type[Any]
        TA1.unknown  # E: undefined_attribute
        TA2.unknown  # E: undefined_attribute
        type.unknown  # E: undefined_attribute

    @assert_passes(allow_import_failures=True)
    def test_type_object_name_attribute_after_import_failure(self):
        from typing import Type

        from typing_extensions import assert_type

        def f(a: type[object], b: Type[object]) -> None:
            assert_type(a.__name__, str)
            assert_type(b.__name__, str)

        type.unknown  # E: undefined_attribute


class TestPartialValueInference(TestNameCheckVisitorBase):
    @assert_passes()
    def test_specialform_getitem_infers_partial_value(self):
        import ast
        import typing

        from pycroscope.value import (
            AnySource,
            AnyValue,
            KnownValue,
            PartialValue,
            PartialValueOperation,
            SubclassValue,
            TypedValue,
        )

        expected = PartialValue(
            PartialValueOperation.SUBSCRIPT,
            KnownValue(typing.Union),
            ast.parse("typing.Union", mode="eval").body,
            (SubclassValue(TypedValue(int)),),
            AnyValue(AnySource.inference),
        )

        def f(x: type[int]):
            y = typing._SpecialForm.__getitem__(typing.Union, x)
            assert_is_value(y, expected)
            return y


class TestImportFailureHandlingCodeSamples(TestNameCheckVisitorBase):
    @assert_passes(allow_import_failures=True)
    def test_overload_consistency_after_import_failure(self):
        from typing import overload

        boom = 1 / 0

        @overload
        def return_type(x: int, /) -> int: ...

        @overload
        def return_type(x: str, /) -> str:  # E: inconsistent_overload
            ...

        def return_type(x: int | str, /) -> int:
            return 1

        @overload
        def parameter_type(x: int, /) -> int: ...

        @overload
        def parameter_type(x: str, /) -> str:  # E: inconsistent_overload
            ...

        def parameter_type(x: int, /) -> int | str:
            return 1

    @assert_passes(allow_import_failures=True)
    def test_dict_subclass_assignable_to_dict_after_import_failure(self):
        boom = 1 / 0

        class CustomDict(dict[str, int]):
            pass

        def takes_dict(x: dict[str, int]) -> None:
            return None

        takes_dict(CustomDict({"num": 1}))

    @assert_passes(allow_import_failures=True)
    def test_overload_fallback_after_import_failure(self):
        from typing import assert_type, overload

        boom = 1 / 0

        @overload
        def f(x: int, /) -> int: ...

        @overload
        def f(x: str, /) -> str: ...

        def f(x: int | str, /) -> int | str:
            return x

        class B:
            @overload
            def __getitem__(self, x: int, /) -> int: ...

            @overload
            def __getitem__(self, x: str, /) -> bytes: ...

            def __getitem__(self, x: int | str, /) -> int | bytes:
                raise NotImplementedError

        b = B()
        assert_type(f(1), int)
        assert_type(f("x"), str)
        assert_type(b[0], int)
        assert_type(b["x"], bytes)

    @assert_passes(allow_import_failures=True)
    def test_typeddict_class_syntax_after_import_failure(self):
        boom = 1 / 0

        from typing import TypedDict

        class Movie(TypedDict):
            director: "Person"

        class Person(TypedDict):
            name: str

        class BadTypedDict1(TypedDict):
            name: str

            def method(self):  # E: invalid_annotation
                pass

        class BadTypedDict2(TypedDict, metaclass=type):  # E: invalid_annotation
            name: str

        class BadTypedDict3(TypedDict, other=True):  # E: invalid_annotation
            name: str

    @assert_passes(allow_import_failures=True)
    def test_typeddict_extra_items_and_unpack_after_import_failure(self):
        boom = 1 / 0

        from typing_extensions import TypedDict, Unpack

        class Movie(TypedDict, extra_items=bool):
            name: str

        MovieFunctional = TypedDict("MovieFunctional", {"name": str}, extra_items=bool)

        a: Movie = {"name": "Blade Runner", "year": 1982}  # E: incompatible_assignment
        # E: incompatible_assignment
        b: MovieFunctional = {"name": "Blade Runner", "year": 1982}

        class MovieNoExtra(TypedDict):
            name: str

        class MovieExtra(TypedDict, extra_items=int):
            name: str

        def unpack_no_extra(**kwargs: Unpack[MovieNoExtra]) -> None: ...

        def unpack_extra(**kwargs: Unpack[MovieExtra]) -> None: ...

        # E: incompatible_call
        unpack_no_extra(name="No Country for Old Men", year=2007)
        unpack_extra(name="No Country for Old Men", year=2007)

    @assert_passes(allow_import_failures=True)
    def test_property_setter_in_synthetic_class_after_import_failure(self):
        boom = 1 / 0

        class C:
            @property
            def value(self) -> int:
                return 1

            @value.setter
            def value(self, new_value: int) -> None:
                pass

    @assert_passes(allow_import_failures=True)
    def test_synthetic_instance_attrs_and_forward_methods_after_import_failure(self):
        from typing import Generic, TypeVar

        boom = 1 / 0

        T = TypeVar("T")

        class LoggedVar(Generic[T]):
            def __init__(self, value: T, name: str) -> None:
                self.name = name
                self.value = value

            def set(self, new: T) -> None:
                self.log("Set " + repr(self.value))
                self.value = new

            def get(self) -> T:
                self.log("Get " + repr(self.value))
                return self.value

            def log(self, message: str) -> None:
                print(f"{self.name}: {message}")

        def capybara(v: LoggedVar[int]) -> int:
            v.set(1)
            return v.get()

    @assert_passes(allow_import_failures=True)
    def test_zero_arg_super_in_synthetic_class_after_import_failure(self):
        boom = 1 / 0

        class Base:
            def method(self) -> int:
                return 1

        class Child(Base):
            def other(self) -> int:
                return super().method()

    @assert_passes(allow_import_failures=True)
    def test_overloaded_override_and_final_after_import_failure(self):
        from typing import final, overload, override

        boom = 1 / 0

        class Base:
            @overload
            @staticmethod
            def final_method(x: int) -> int: ...

            @overload
            @staticmethod
            def final_method(x: str) -> str: ...

            @staticmethod
            @final
            def final_method(x: int | str) -> int | str:
                return x

            @overload
            @staticmethod
            def good_override(x: int) -> int: ...

            @overload
            @staticmethod
            def good_override(x: str) -> str: ...

            @staticmethod
            def good_override(x: int | str) -> int | str:
                return x

        class Child(Base):
            @overload
            @staticmethod
            def final_method(x: int) -> int: ...

            @overload
            @staticmethod
            def final_method(x: str) -> str: ...

            @staticmethod
            def final_method(x: int | str) -> int | str:  # E: invalid_annotation
                return x

            @overload
            @staticmethod
            def good_override(x: int) -> int: ...

            @overload
            @staticmethod
            def good_override(x: str) -> str: ...

            @staticmethod
            @override
            def good_override(x: int | str) -> int | str:
                return x

    @assert_passes(allow_import_failures=True)
    def test_namedtuple_after_import_failure(self):
        boom = 1 / 0

        from typing import Generic, NamedTuple, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Point(NamedTuple):
            x: int
            y: int
            units: str = "meters"

        p = Point(1, 2)
        assert_type(p, Point)
        assert_type(p.x, int)
        assert_type(p[2], str)
        a, b, c = p
        assert_type(a, int)
        assert_type(c, str)
        p[3]  # E: incompatible_call
        p[-4]  # E: incompatible_call
        Point(1)  # E: incompatible_call

        class Point3(NamedTuple):
            _y: int  # E: invalid_annotation

        class Location(NamedTuple):
            altitude: float = 0.0
            latitude: float  # E: invalid_annotation

        class Property(NamedTuple, Generic[T]):
            name: str
            value: T

        pr = Property("", 3.4)
        assert_type(pr, Property[float])
        assert_type(pr[1], float)
        assert_type(pr.value, float)
        Property[str]("", 3.1)  # E: incompatible_argument

        class PointWithName(Point):
            name: str = ""

        pn = PointWithName(1, 2, "")
        assert_type(pn.name, str)

        class BadPointWithName(Point):
            x: int = 0  # E: incompatible_override

        class Unit(NamedTuple, object):  # E: invalid_base
            name: str

    @assert_passes(allow_import_failures=True)
    def test_dataclass_comparison_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import dataclass

        @dataclass(order=True)
        class DC1:
            a: str
            b: int

        @dataclass(order=True)
        class DC2:
            a: str
            b: int

        dc1_1 = DC1("", 0)
        dc2_1 = DC2("hi", 2)

        if dc1_1 < dc2_1:  # E: unsupported_operation
            pass

        if dc1_1 != dc2_1:
            pass

    @assert_passes()
    def test_frozen_dataclass_disallows_instance_attribute_assignment(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Frozen:
            value: int

        def mutate() -> None:
            frozen = Frozen(1)
            frozen.value = 2  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_frozen_dataclass_checks_after_import_failure(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Frozen:
            value: int

        frozen = Frozen(1)
        frozen.value = 2  # E: incompatible_assignment

        @dataclass
        class NonFrozenChild(Frozen):  # E: invalid_base
            pass

        @dataclass
        class Mutable:
            value: int

        @dataclass(frozen=True)
        class FrozenChild(Mutable):  # E: invalid_base
            pass

    @assert_passes()
    def test_dataclass_hashability(self):
        from dataclasses import dataclass
        from typing import Hashable

        @dataclass
        class Unhashable:
            value: int

        @dataclass(frozen=True)
        class Frozen:
            value: int

        @dataclass(eq=False)
        class NoEq:
            value: int

        @dataclass(unsafe_hash=True)
        class UnsafeHash:
            value: int

        @dataclass
        class ExplicitHash:
            value: int

            def __hash__(self) -> int:
                return 0

        @dataclass(eq=False)
        class ExplicitlyUnhashable:
            value: int
            __hash__ = None

        bad_unhashable: Hashable = Unhashable(1)  # E: incompatible_assignment
        ok_frozen: Hashable = Frozen(1)
        ok_no_eq: Hashable = NoEq(1)
        ok_unsafe_hash: Hashable = UnsafeHash(1)
        ok_explicit_hash: Hashable = ExplicitHash(1)
        # E: incompatible_assignment
        bad_explicit_none: Hashable = ExplicitlyUnhashable(1)

    @assert_passes()
    def test_hashability_respects_hash_none_for_typed_values(self):
        from typing import Hashable

        class Unhashable:
            __hash__ = None

        obj: Unhashable = Unhashable()
        bad: Hashable = obj  # E: incompatible_assignment

    @assert_passes()
    def test_dataclass_kw_only_marker_is_allowed(self):
        from dataclasses import KW_ONLY, dataclass

        @dataclass
        class DC:
            a: str
            _: KW_ONLY
            b: int = 0

        DC("hi")
        DC("hi", b=1)

    @assert_passes()
    def test_dataclass_slots_semantics(self):
        from dataclasses import dataclass

        @dataclass(slots=True)
        class Slotted:
            x: int

            def set_bad(self) -> None:
                self.y = 3  # E: incompatible_assignment

        Slotted.__slots__
        Slotted(1).__slots__

        @dataclass
        class NotSlotted:
            x: int

        @dataclass(slots=False)
        class ExplicitSlots:
            x: int
            __slots__ = ("x",)

            def set_bad(self) -> None:
                self.y = 3  # E: incompatible_assignment

        def check_errors() -> None:
            NotSlotted.__slots__  # E: undefined_attribute
            NotSlotted(1).__slots__  # E: undefined_attribute

            @dataclass(slots=True)
            class DataclassWithSlotsAttribute:  # E: invalid_annotation
                x: int
                __slots__ = ()

    @assert_passes(allow_import_failures=True)
    def test_dataclass_init_and_match_args_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import dataclass

        @dataclass(init=False)
        class InitDisabled:
            x: int
            y: int

        InitDisabled()
        InitDisabled(1, 2)  # E: incompatible_call

        def match_init_disabled(value: InitDisabled) -> None:
            match value:
                case InitDisabled(1, 2):
                    pass

        @dataclass(match_args=False)
        class NoMatchArgs:
            x: int

        def reject_positional_patterns(value: NoMatchArgs) -> None:
            match value:
                case NoMatchArgs(1):  # E: bad_match
                    pass

    @assert_passes(allow_import_failures=True)
    def test_dataclass_kw_only_checks_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import KW_ONLY, dataclass, field

        @dataclass
        class DC1:
            a: str
            _: KW_ONLY
            b: int = 0

        DC1("hi")
        DC1(a="hi")
        DC1(a="hi", b=1)
        DC1("hi", b=1)
        DC1("hi", 1)  # E: incompatible_call

        @dataclass
        class DC2:
            b: int = field(kw_only=True, default=3)
            a: str

        DC2("hi")
        DC2(a="hi")
        DC2(a="hi", b=1)
        DC2("hi", b=1)
        DC2("hi", 1)  # E: incompatible_call

        @dataclass(kw_only=True)
        class DC3:
            a: str = field(kw_only=False)
            b: int = 0

        DC3("hi")
        DC3(a="hi")
        DC3(a="hi", b=1)
        DC3("hi", b=1)
        DC3("hi", 1)  # E: incompatible_call

        @dataclass
        class DC4(DC3):
            c: float

        DC4("", 0.2, b=3)
        DC4(a="", b=3, c=0.2)

    @assert_passes(allow_import_failures=True)
    def test_dataclass_constructor_field_metadata_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import dataclass, field

        @dataclass
        class InventoryItem:
            x = 0
            name: str
            unit_price: float
            quantity_on_hand: int = 0

        InventoryItem("soap", 2.3)
        InventoryItem("name")  # E: incompatible_call

        @dataclass
        class WithInitFalse:
            a: int = field(init=False)
            b: int

        WithInitFalse(1)
        WithInitFalse(a=1, b=2)  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_dataclass_usage_features_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import dataclass, field
        from typing import (
            Any,
            Callable,
            ClassVar,
            Generic,
            Protocol,
            TypeVar,
            assert_type,
        )

        T = TypeVar("T")

        @dataclass(order=True)
        class InventoryItem:
            name: str
            unit_price: float
            quantity_on_hand: int = 0

        class InventoryItemInitProto(Protocol):
            def __call__(
                self, name: str, unit_price: float, quantity_on_hand: int = ...
            ) -> None: ...

        item = InventoryItem("soap", 2.3)
        init_proto: InventoryItemInitProto = item.__init__
        item.__repr__
        item.__eq__
        item.__ne__
        item.__lt__
        item.__le__
        item.__gt__
        item.__ge__

        def parser(s: str) -> int:
            return int(s)

        @dataclass
        class WithCallableDefault:
            c: Callable[[str], int] = parser

        with_callable_default = WithCallableDefault()
        assert_type(with_callable_default.c, Callable[[str], int])

        @dataclass
        class WithBadFactory:
            a: int = field(default_factory=str)  # E: incompatible_assignment

        class DataclassProto(Protocol):
            __dataclass_fields__: ClassVar[dict[str, Any]]

        proto: DataclassProto = item

        @dataclass
        class Box(Generic[T]):
            value: T

        class StrBox(Box[str]):
            pass

        StrBox("")

    @assert_passes(allow_import_failures=True)
    def test_dataclass_default_order_validation(self):
        from dataclasses import InitVar, dataclass, field

        @dataclass  # E: invalid_annotation
        class DC1:
            a: int = 0
            b: int

        @dataclass  # E: invalid_annotation
        class DC2:
            a: int = field(default=1)
            b: int

        @dataclass  # E: invalid_annotation
        class DC3:
            a: InitVar[int] = 0
            b: int

        @dataclass
        class DC4:
            a: int = field(repr=False)
            b: int

        @dataclass
        class DC5:
            a: int = 0
            b: int = field(init=False)

            def method(self) -> None:
                local: int = 0
                _ = local

        DC4(1, 2)

    @assert_passes()
    def test_dataclass_post_init_initvar_semantics(self):
        from dataclasses import InitVar, dataclass, field

        @dataclass
        class DC1:
            a: int
            b: int
            x: InitVar[int]
            c: int
            y: InitVar[str]

            def __post_init__(self, x: int, y: int) -> None:  # E: incompatible_override
                pass

        def f(dc1: DC1) -> None:
            dc1.x  # E: undefined_attribute
            dc1.y  # E: undefined_attribute

        @dataclass
        class DC2:
            x: InitVar[int]
            y: InitVar[str]

            def __post_init__(self, x: int) -> None:  # E: incompatible_override
                pass

        @dataclass
        class DC3:
            _name: InitVar[str] = field()
            name: str = field(init=False)

            def __post_init__(self, _name: str): ...

        @dataclass
        class DC4(DC3):
            _age: InitVar[int] = field()
            age: int = field(init=False)

            def __post_init__(self, _name: str, _age: int): ...

    @assert_passes(allow_import_failures=True)
    def test_dataclass_post_init_initvar_semantics_after_import_failure(self):
        boom = 1 / 0

        from dataclasses import InitVar, dataclass

        @dataclass
        class DC:
            x: InitVar[int]
            y: InitVar[str]

            def __post_init__(self, x: int) -> None:  # E: incompatible_override
                pass

        def f(dc: DC) -> None:
            dc.x  # E: undefined_attribute
            dc.y  # E: undefined_attribute

    @assert_passes()
    def test_final_attribute_assignment_on_instance(self):
        from typing import Final

        class C:
            x: Final[int] = 1

        c = C()
        c.x = 2  # E: incompatible_assignment

    @assert_passes()
    def test_classvar_attribute_assignment_on_instance(self):
        from typing import ClassVar

        class C:
            x: ClassVar[int] = 1

        class D(C):
            pass

        c = C()
        c.x = 2  # E: incompatible_assignment
        C.x = 3

        d = D()
        d.x = 4  # E: incompatible_assignment
        D.x = 5

    @assert_passes()
    def test_namedtuple_attribute_is_immutable(self):
        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        p = Point(1)

        def mutate() -> None:
            p.x = 2  # E: incompatible_assignment
            del p.x  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_namedtuple_attribute_is_immutable_after_import_failure(self):
        boom = 1 / 0

        from typing import NamedTuple

        class Point(NamedTuple):
            x: int

        p = Point(1)
        p.x = 2  # E: incompatible_assignment
        del p.x  # E: incompatible_assignment

    @assert_passes()
    def test_incompatible_annotated_attribute_assignment(self):
        class C:
            x: int

            def __init__(self) -> None:
                self.x = 1

        c = C()
        c.x = "x"  # E: incompatible_assignment

    @assert_passes()
    def test_assign_attribute_on_function_object(self):
        def decorator(func):
            func.__is_type_evaluation__ = True
            return func

        @decorator
        def f() -> int:
            return 1

        assert f() == 1


class TestNameCheckVisitor(TestNameCheckVisitorBase):
    @assert_passes(allow_import_failures=True)
    def test_undefined_class_decorator_does_not_internal_error(self):
        @decorator1(0)  # E: undefined_name
        class C: ...

    @assert_passes()
    def test_synthetic_class_methods_from_stub_import(self):
        def run():
            from _pycroscope_tests.self import X, Y

            from pycroscope.value import SyntheticClassObjectValue

            assert_is_value(
                X,
                SyntheticClassObjectValue("X", TypedValue("_pycroscope_tests.self.X")),
            )
            assert_is_value(
                Y,
                SyntheticClassObjectValue("Y", TypedValue("_pycroscope_tests.self.Y")),
            )
            assert_is_value(X.from_config(), TypedValue("_pycroscope_tests.self.X"))
            assert_is_value(Y.from_config(), TypedValue("_pycroscope_tests.self.Y"))
            assert_is_value(X().ret(), TypedValue("_pycroscope_tests.self.X"))
            assert_is_value(Y().ret(), TypedValue("_pycroscope_tests.self.Y"))

    @assert_passes()
    def test_function_local_class_uses_synthetic_class_object(self):
        from pycroscope.value import SyntheticClassObjectValue

        def outer():
            class Local:
                @staticmethod
                def static_method() -> int:
                    return 1

                @staticmethod
                def plus_one(x: int) -> int:
                    return x + 1

            assert_is_value(
                Local,
                SyntheticClassObjectValue(
                    "Local", TypedValue(f"{__name__}.outer.<locals>.Local")
                ),
            )
            assert_is_value(Local.static_method(), TypedValue(int))
            assert_is_value(Local.plus_one(1), TypedValue(int))
            return Local

        assert_is_value(
            outer(),
            SyntheticClassObjectValue(
                "Local", TypedValue(f"{__name__}.outer.<locals>.Local")
            ),
        )

    @assert_passes()
    def test_synthetic_class_inherits_synthetic_base_attributes(self):
        def outer():
            class Base:
                @staticmethod
                def base_method() -> int:
                    return 1

            class Child(Base):
                pass

            assert_is_value(Child.base_method(), TypedValue(int))
            return Child

        assert_is_value(outer().base_method(), TypedValue(int))

    @assert_passes()
    def test_synthetic_class_inherits_runtime_base_attributes(self):
        class RuntimeBase:
            @staticmethod
            def runtime_method() -> str:
                return ""

        def outer():
            class Child(RuntimeBase):
                pass

            assert_is_value(Child.runtime_method(), TypedValue(str))
            return Child

        assert_is_value(outer().runtime_method(), TypedValue(str))

    @assert_passes()
    def test_typedvalue_accepts_function_local_synthetic_class(self):
        from pycroscope.value import TypedValue

        def outer():
            class Local:
                pass

            TypedValue(Local)

    @assert_passes()
    def test_function_local_generic_class_subscript(self):
        from typing import Generic, Literal, TypeVar

        from typing_extensions import assert_type

        A = TypeVar("A", bound=int)
        B = TypeVar("B", bound=int)

        def outer() -> None:
            class Matrix(Generic[A, B]):
                pass

            def func(a: Matrix[Literal[2], Literal[3]]) -> None:
                assert_type(a, Matrix[Literal[2], Literal[3]])

    @assert_passes()
    def test_function_scope_typeddict_readonly_inheritance(self):
        from typing import TypedDict

        from typing_extensions import NotRequired, ReadOnly, Required

        def run() -> None:
            class F1(TypedDict):
                a: Required[int]
                b: ReadOnly[NotRequired[int]]
                c: ReadOnly[Required[int]]

            class F3(F1):
                a: ReadOnly[int]  # E: invalid_annotation

            class F4(F1):
                a: NotRequired[int]  # E: invalid_annotation

            class F5(F1):
                b: ReadOnly[Required[int]]

            class F6(F1):
                c: ReadOnly[NotRequired[int]]  # E: invalid_annotation

            class TD_A1(TypedDict):
                x: int
                y: ReadOnly[int]

            class TD_A2(TypedDict):
                x: float
                y: ReadOnly[float]

            class TD_A(TD_A1, TD_A2): ...  # E: invalid_annotation

            class TD_B1(TypedDict):
                x: ReadOnly[NotRequired[int]]
                y: ReadOnly[Required[int]]

            class TD_B2(TypedDict):
                x: ReadOnly[Required[int]]
                y: ReadOnly[NotRequired[int]]

            class TD_B(TD_B1, TD_B2): ...  # E: invalid_annotation

    @assert_passes()
    def test_function_scope_typeddict_values(self):
        from typing import TypedDict

        from typing_extensions import NotRequired, ReadOnly, Required

        def run() -> None:
            class OptionalName(TypedDict):
                name: ReadOnly[NotRequired[str]]

            class RequiredName(OptionalName):
                name: ReadOnly[Required[str]]

            d: RequiredName = {}  # E: incompatible_assignment
            print(d)

            class Movie(TypedDict):
                title: ReadOnly[str]

            movie: Movie = {"title": ""}
            movie["title"] = "x"  # E: readonly_typeddict

    @assert_passes()
    def test_known_ordered(self):
        from typing_extensions import OrderedDict

        known_ordered = OrderedDict({1: 2})
        bad_ordered = OrderedDict({"a": "b"})

        def capybara(arg: OrderedDict[int, int]) -> None:
            pass

        def caller() -> None:
            capybara(known_ordered)
            capybara(bad_ordered)  # E: incompatible_argument

    @assert_passes()
    def test_undefined_name(self):
        def run():
            print(undefined_variable)  # E: undefined_name

    @assert_passes()
    def test_undefined_attribute(self):
        def run():
            lst = []
            print(lst.coruro)  # E: undefined_attribute

    def test_undefined_name_with_star_import(self):
        # can't use the decorator version because import * isn't allowed with nested functions
        self.assert_fails(
            ErrorCode.undefined_name,
            """
            from typing import *
            def run():
                print(not_in.typing)
            """,
        )

    @assert_passes()
    def test_undefined_name_in_return(self):
        def what_is_it():
            return tucotuco  # E: undefined_name

    @assert_passes()
    def test_undefined_name_in_class_kwarg(self):
        def capybara():
            class Capybara(metaclass=Hutia):  # E: undefined_name
                pass

    @assert_passes()
    def test_no_failure_on_builtin(self):
        def run():
            print(len)

    @assert_passes()
    def test_no_failure_on_global(self):
        capybara = 3

        def run():
            print(capybara)

    @assert_passes()
    def test_no_failure_on_global_return(self):
        tucotuco = "a burrowing rodent"

        def what_is_it():
            return tucotuco

    @assert_passes()
    def test_no_failure_on_arg(self):
        def double_it(x):
            return x * 2

    @assert_passes()
    def test_class_scope(self):
        class Porcupine(object):
            def coendou(self):
                return 1

            sphiggurus = coendou

    @assert_passes()
    def test_class_scope_fails_wrong_order(self):
        def run():
            class Porcupine(object):
                sphiggurus = coendou  # E: undefined_name

                def coendou(self):
                    return 1

    @assert_passes()
    def test_class_scope_is_not_searched(self):
        class Porcupine(object):
            sphiggurus = 1

            def coendou(self):
                return sphiggurus  # E: undefined_name

    @assert_passes()
    def test_getter_decorator(self):
        class Porcupine(object):
            sphiggurus = property()

            @sphiggurus.getter
            def sphiggurus(self):
                pass

    @assert_passes()
    def test_ipython_whitelisting(self):
        def run():
            print(__IPYTHON__)

    @assert_passes()
    def test_mock_attributes(self):
        def cavy():
            pass

        def run():
            print(cavy.call_count)

    @assert_passes()
    def test_mock_attr(self):
        from unittest import mock

        class X:
            a = mock.MagicMock()

        class Y:
            def __init__(self):
                self.x = X()

        def f():
            y = Y()
            assert_is_value(y.x.a, KnownValue(X.a))

    @assert_passes()
    def test_method_mock_attributes(self):
        class Capybara(object):
            def hutia(self):
                pass

            def kerodon(self):
                print(self.hutia.call_count)

    @assert_passes()
    def test_global_assignment(self):
        from typing import get_args

        fn = get_args

        def run():
            assert_is_value(fn, KnownValue(get_args))

    @assert_passes()
    def test_builtin_attribute(self):
        def run():
            print(True.hutia)  # E: undefined_attribute

    @assert_passes()
    def test_module_reassignment(self):
        _std_set = set

        def set(key, value):
            return _std_set([key, value])

        _std_set()

    @assert_passes()
    def test_display(self):
        def run():
            x = [1, 2]
            print(x())  # E: not_callable

    @assert_passes()
    def test_set_display(self):
        def run():
            print({[]})  # E: unhashable_key

            print({*[1, 2, 3], "a", "b"})
            print({*[{}], "a", "b"})  # E: unhashable_key

    @assert_passes()
    def test_multiple_assignment_global(self):
        if False:
            goes_in_set = []
        else:
            goes_in_set = "capybara"
        if False:
            # The assignment actually executed at runtime wins
            assert_is_value(goes_in_set, KnownValue("capybara"))
            print({goes_in_set})

    @assert_passes()
    def test_multiple_assignment_function(self):
        def fn(cond):
            if cond:
                goes_in_set = []
            else:
                goes_in_set = "capybara"
            assert_is_value(goes_in_set, KnownValue([]) | KnownValue("capybara"))
            print({goes_in_set})  # E: unhashable_key

    @assert_passes()
    def test_duplicate_dict_key(self):
        def run():
            print({"capybara": 1, "capybara": 2})  # E: duplicate_dict_key

    @assert_passes()
    def test_unhashable_dict_key(self):
        def run():
            print({[]: 1})  # E: unhashable_key

    @assert_passes()
    def test_inferred_duplicate_dict_key(self):
        key = "capybara"

        def run():
            print({"capybara": 1, key: 1})  # E: duplicate_dict_key

    @assert_passes()
    def test_inferred_unhashable_dict_key(self):
        key = []

        def run():
            print({key: 1})  # E: unhashable_key

    @assert_passes()
    def test_cant_del_tuple(self):
        tpl = (1, 2, 3)

        def run():
            del tpl[1]  # E: unsupported_operation

    @assert_passes()
    def test_cant_del_generator(self):
        tpl = (x for x in (1, 2, 3))

        def run():
            del tpl[1]  # E: unsupported_operation

    @assert_passes()
    def test_cant_assign_tuple(self):
        tpl = (1, 2, 3)

        def run():
            tpl[1] = 1  # E: unsupported_operation

    @assert_passes()
    def test_global_sets_value(self):
        capybara = None

        def set_it():
            global capybara
            capybara = (0,)

        def use_it():
            assert_is_value(capybara, KnownValue((0,)) | KnownValue(None))

    # can't change to assert_passes because it changes between Python 3.6 to 3.7
    @assert_fails(ErrorCode.unsupported_operation)
    def test_self_type_inference(self):
        class Capybara(object):
            def get(self, i):
                assert_is_value(self, TypedValue(Capybara))
                return self[i]

    @assert_passes()
    def test_self_is_subscriptable(self):
        class Capybara(object):
            def get(self, i):
                return self[i]

            def __getitem__(self, i):
                return i

    @assert_passes()
    def test_cls_type_inference(self):
        class OldStyle:
            def __init_subclass__(cls):
                assert_is_value(cls, SubclassValue(TypedValue(OldStyle)))

            def __new__(cls):
                assert_is_value(cls, SubclassValue(TypedValue(OldStyle)))

            @classmethod
            def capybara(cls):
                assert_is_value(cls, SubclassValue(TypedValue(OldStyle)))

    @assert_passes()
    def test_display_type_inference(self):
        UNANNOTATED = AnyValue(AnySource.unannotated)

        def capybara(a, b):
            x = [a, b]
            assert_is_value(x, make_simple_sequence(list, [UNANNOTATED, UNANNOTATED]))
            y = a, 2
            assert_is_value(
                y, make_simple_sequence(tuple, [UNANNOTATED, KnownValue(2)])
            )

            s = {a, b}
            assert_is_value(s, make_simple_sequence(set, [UNANNOTATED, UNANNOTATED]))
            z = {a: b}
            assert_is_value(
                z, DictIncompleteValue(dict, [KVPair(UNANNOTATED, UNANNOTATED)])
            )
            q = {a: 3, b: 4}
            assert_is_value(
                q,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(UNANNOTATED, KnownValue(3)),
                        KVPair(UNANNOTATED, KnownValue(4)),
                    ],
                ),
            )

    @assert_passes()
    def test_if_exp(self):
        def capybara(x):
            y = 3 if x else 4
            assert_is_value(y, MultiValuedValue([KnownValue(3), KnownValue(4)]))

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

        from pycroscope.value import KnownValue, SyntheticClassObjectValue, TypedValue

        def capybara():
            typ = collections.namedtuple("typ", "foo bar")
            assert_is_value(
                typ,
                SyntheticClassObjectValue(
                    "typ", TypedValue(f"{__name__}.capybara.<locals>.typ")
                ),
            )
            t = typ(1, 2)
            assert_is_value(t.foo, KnownValue(1))
            assert_is_value(t.bar, KnownValue(2))
            print(t.baz)  # E: undefined_attribute
            typ(1, 2, 3)  # E: incompatible_call
            typ(1)  # E: incompatible_call

    @assert_passes()
    def test_set_after_get(self):
        def fn():
            capybara = None
            for _ in range(5):
                if capybara:
                    print(capybara[0])
                capybara = "foo"

    @assert_passes()
    def test_multiple_anys(self):
        def fn(item):
            if False:
                item = None
            assert_is_value(item, KnownValue(None) | AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_bad_attribute_of_global(self):
        import os

        path = os.path

        def capybara():
            print(path.joyn)  # E: undefined_attribute

    @assert_passes()
    def test_double_assignment(self):
        from pycroscope.tests import PropertyObject

        def capybara(aid):
            answer = PropertyObject(aid)
            print(answer)
            answer = PropertyObject(aid)
            assert_is_value(answer, TypedValue(PropertyObject))

    @assert_passes()
    def test_duplicate_method(self):
        class Tucotuco(object):
            def __init__(self, fn):
                pass

            def __init__(self, an):  # E: class_variable_redefinition
                pass

    @assert_passes()
    def test_duplicate_attribute(self):
        class Hutia:
            capromys = 1
            capromys = 2  # E: class_variable_redefinition

    @assert_passes()
    def test_duplicate_attribute_augassign(self):
        class Capybara:
            x = 1
            x += 1

    @assert_passes()
    def test_duplicate_property_method(self):
        class Capybara(object):
            @property
            def fur(self):
                return "a lot"

            @fur.setter
            def fur(self, value):
                pass

    @assert_passes()
    def test_bad_global(self):
        global x  # E: bad_global

    @assert_passes()
    def test_undefined_global(self):
        def fn():
            global x
            return x  # E: undefined_name

    @assert_passes()
    def test_global_value(self):
        x = 3

        def capybara():
            global x
            assert_is_value(x, KnownValue(3))


class TestSubclassValue(TestNameCheckVisitorBase):
    @assert_passes()
    def test_annotations_in_arguments(self):
        from typing import Type

        TI = Type[int]

        def capybara(x: TI, y: str):
            assert_is_value(x, SubclassValue(TypedValue(int)))
            assert_is_value(y, TypedValue(str))

    @assert_passes()
    def test_type_any(self):
        from typing import Any, Type

        def f(x) -> Type[Any]:
            return type(x)

        def capybara():
            f(1)
            assert_is_value(f(1), TypedValue(type))

    @assert_passes()
    def test_call_method_through_type(self):
        class A:
            def run(self):
                pass

            @classmethod
            def call_on_instance(cls, instance):
                assert_is_value(cls.run, KnownValue(A.run))
                cls.run(instance)

    @assert_passes()
    def test_metaclass_method(self):
        from typing import Type

        class EnumMeta(type):
            def __getitem__(self, x: str) -> bytes:
                return b"hi"

        class Enum(metaclass=EnumMeta):
            pass

        def capybara(enum: Type[Enum]) -> None:
            assert_is_value(enum["x"], TypedValue(bytes))

    @assert_passes()
    def test_metaclass_call(self):
        from typing import Type

        class Meta(type):
            def __call__(self, *args: object, **kwargs: object) -> bytes:
                return b"hi"

        class C(metaclass=Meta):
            pass

        def capybara(cls: Type[C]) -> None:
            assert_is_value(cls("x"), TypedValue(bytes))

    @assert_passes()
    def test_default_constructor_call(self):
        class C:
            pass

        def capybara(cls: type[C]) -> None:
            cls()
            cls(1)  # E: incompatible_call

    @assert_passes()
    def test_unbounded_typevar_constructor_call(self):
        from typing import TypeVar

        T = TypeVar("T")

        def capybara(cls: type[T]) -> None:
            cls()
            cls(1)  # E: incompatible_call

    @assert_passes()
    def test_bound_typevar_constructor_call(self):
        from typing import TypeVar

        class Base:
            def __init__(self, x: int, y: str) -> None:
                pass

        T = TypeVar("T", bound=Base)

        def capybara(cls: type[T]) -> None:
            cls(x=1, y="")
            cls()  # E: incompatible_call
            cls(1)  # E: incompatible_call
            cls(1, 2)  # E: incompatible_argument

    @assert_passes()
    def test_constructor_inherited_generic_init_with_self_annotation(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Base(Generic[T]):
            def __init__(self, x: Self | None) -> None:
                pass

        class Child(Base[int]):
            pass

        Child(Child(None))
        Child(Base(None))  # E: incompatible_argument

    @assert_passes()
    def test_constructor_respects_explicit_init_self_annotation(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class C(Generic[T]):
            def __init__(self: "C[int]") -> None:
                pass

        C()
        C[int]()
        C[str]()  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_constructor_callable_ignores_init_when_new_returns_proxy(self):
        from typing import Callable, ParamSpec, TypeVar, assert_type

        P = ParamSpec("P")
        R = TypeVar("R")

        def accepts_callable(cb: Callable[P, R]) -> Callable[P, R]:
            return cb

        class Proxy:
            pass

        class C:
            def __new__(cls) -> Proxy:
                return Proxy()

            def __init__(self, x: int) -> None:
                pass

        r = accepts_callable(C)
        assert_type(r(), Proxy)
        r(1)  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_constructor_callable_ignores_init_when_new_returns_any(self):
        from typing import Any, Callable, ParamSpec, TypeVar, assert_type

        P = ParamSpec("P")
        R = TypeVar("R")

        def accepts_callable(cb: Callable[P, R]) -> Callable[P, R]:
            return cb

        class C:
            def __new__(cls) -> Any:
                return super().__new__(cls)

            def __init__(self, x: int) -> None:
                pass

        r = accepts_callable(C)
        assert_type(r(), Any)
        r(1)  # E: incompatible_call

    @assert_passes()
    def test_init_self_annotation_disallows_class_scoped_typevars(self):
        from typing import Generic, TypeVar

        T1 = TypeVar("T1")
        T2 = TypeVar("T2")
        V = TypeVar("V")

        class Bad(Generic[T1, T2]):
            def __init__(self: "Bad[T2, T1]") -> None:  # E: invalid_annotation
                pass

        class Good(Generic[T1]):
            def __init__(self: "Good[V]", value: V) -> None:
                pass

        Good(1)

    @assert_passes()
    def test_incompatible_invariant_typevar_arguments(self):
        from typing import TypeVar

        T = TypeVar("T")

        def f(x: list[T], y: list[T]) -> None:
            pass

        f([1], [2])
        f([1], [""])  # E: incompatible_call

    @assert_passes()
    def test_type_union(self):
        from typing import Type, Union

        def capybara(x: Type[Union[int, str]]) -> None:
            assert_is_value(
                x,
                MultiValuedValue(
                    [SubclassValue(TypedValue(int)), SubclassValue(TypedValue(str))]
                ),
            )

        def caller() -> None:
            capybara(int)
            capybara(str)


class TestReturn(TestNameCheckVisitorBase):
    @assert_passes()
    def test_missing_return(self):
        from abc import abstractmethod

        from typing_extensions import NoReturn, Protocol

        def foo(cond: bool) -> int:  # E: missing_return
            if cond:
                return 3

        def capybara() -> int:  # E: missing_return
            pass

        class Absy:
            @abstractmethod
            def doesnt_return(self, cond: bool) -> int:  # E: missing_return
                if cond:
                    return 1

        class AbsyEllipsis:
            @abstractmethod
            def doesnt_return(self) -> int:  # ok
                ...

        class AbsyEllipsisWithDoc:
            @abstractmethod
            def doesnt_return(self) -> int:  # ok
                """this is intentionally abstract"""
                ...

        class AbsyPass:
            @abstractmethod
            def doesnt_return(self) -> int:  # ok
                pass

        class AbsyPassWithDoc:
            @abstractmethod
            def doesnt_return(self) -> int:  # ok
                """this is intentionally abstract"""
                pass

        class Proto(Protocol):
            def doesnt_return(self) -> int:  # ok
                ...

        class ProtoWithDoc(Protocol):
            def doesnt_return(self) -> int:  # ok
                """this is intentionally abstract"""
                ...

        class ProtoPass(Protocol):
            def doesnt_return(self) -> int:  # ok
                pass

        class ProtoPassWithDoc(Protocol):
            def doesnt_return(self) -> int:  # ok
                """this is intentionally abstract"""
                pass

        def you_can_skip_return_none() -> None:
            pass

        def no_return_but_does_it() -> NoReturn:  # E: no_return_may_return
            pass

        def return_sometimes(cond: bool) -> NoReturn:  # E: no_return_may_return
            if cond:
                raise Exception

        def no_return_returns() -> NoReturn:
            return 42  # E: no_return_may_return

    @assert_passes()
    def test_missing_return_with_invalid_paramspec_return(self):
        from typing import ParamSpec

        P = ParamSpec("P")

        def f() -> P:  # E: invalid_annotation  # E: missing_return
            ...

    @assert_passes()
    def test_list_return(self):
        from typing import List

        class A:
            pass

        def func() -> List[A]:
            return [A]  # E: incompatible_return_value

        def func() -> A:
            return A  # E: incompatible_return_value


class TestYieldFrom(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Iterator

        def f(x) -> Iterator[int]:
            yield from x

    def capybara(x):
        yield from [1, 2]

    @assert_passes()
    def test_bad_yield_from(self):
        def capybara():
            yield from 3  # E: bad_yield_from


class TestClassAttributeChecker(TestNameCheckVisitorBase):
    @assert_passes()
    def test_mangled_attributes(self):
        class Capybara(object):
            def __mangled(self):
                pass

            def other_method(self):
                self.__mangled()

    @assert_passes()
    def test_never_set(self):
        class Capybara(object):
            def method(self):
                return self.doesnt_exist  # E: attribute_is_never_set

    @assert_passes()
    def test_exists_on_class(self):
        class Capybara(object):
            @classmethod
            def type(cls):
                pass

            def method(self):
                return self.__class__.type()

    @assert_passes()
    def test_in_classmethod(self):
        class Capybara(object):
            @classmethod
            def do_stuff(cls):
                return cls.stuff  # E: attribute_is_never_set

    @assert_passes()
    def test_getattribute_overridden(self):
        class GetAttribute(object):
            def __getattribute__(self, attr):
                return 42

            def foo(self):
                return self.answer

    @assert_passes()
    def test_base_attribute(self):
        class Capybara(object):
            def __init__(self, obj):
                self.obj = str(obj)

        class Neochoerus(Capybara):
            def eat(self):
                assert_is_value(self.obj, TypedValue(str))

    @assert_passes()
    def test_unexamined_base(self):
        from pycroscope.tests import PropertyObject

        # this base class was not examined, so we don't know if it has the attribute
        class Capybara(PropertyObject):
            def tree(self):
                return self.this_attribute_does_not_exist

    @skip_if_not_installed("qcore")
    @assert_passes()
    def test_cythonized_unexamined_base(self):
        import qcore

        # this base class was also not examined, but it is Cython so we can still know that the
        # attribute does not exist
        class Capybara(qcore.decorators.DecoratorBase):
            def tree(self):
                return self.this_attribute_does_not_exist  # E: attribute_is_never_set

    @assert_passes()
    def test_setattr(self):
        class Capybara(object):
            def __init__(self, unannotated):
                for k, v in unannotated:
                    assert_is_value(k, AnyValue(AnySource.generic_argument))
                    setattr(self, k, v)
                assert_is_value(self.grass, AnyValue(AnySource.inference))

    @assert_passes()
    def test_setattr_on_base(self):
        class Capybara(object):
            def __init__(self, unannotated):
                for k, v in unannotated:
                    # Make sure we're not smart enough to infer the attribute
                    assert_is_value(k, AnyValue(AnySource.generic_argument))
                    setattr(self, k, v)
                assert_is_value(self.grass, AnyValue(AnySource.inference))

        class Neochoerus(Capybara):
            def eat(self):
                # this doesn't exist, but we shouldn't alert because there is a setattr() on the
                # base
                self.consume(self.grass)


class TestBadRaise(TestNameCheckVisitorBase):
    @assert_passes()
    def test_raise(self):
        def bad_value():
            raise 42  # E: bad_exception

        def bad_type():
            # make sure this isn't inferenced to KnownValue, so this tests what it's supposed to
            # test
            assert_is_value(int("3"), TypedValue(int))
            raise int("3")  # E: bad_exception

        def wrong_type():
            raise bool  # E: bad_exception

        def raise_type():
            raise NotImplementedError

        def reraise():
            try:
                pass
            except OSError:
                raise

        def raise_value():
            raise ValueError("not valuable")

    @assert_passes()
    def test_from(self):
        def none():
            raise Exception() from None

        def other_exc():
            raise Exception() from Exception()

        def not_exc():
            raise Exception() from 42  # E: bad_exception


class TestVariableNameValue(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Any, NewType

        Uid = NewType("Uid", int)

        def name_ends_with_uid(uid):
            return uid

        def some_func() -> Any:
            return 42

        def test(self, uid: Uid):
            assert_is_value(name_ends_with_uid, KnownValue(name_ends_with_uid))
            uid = some_func()
            assert_is_value(uid, VariableNameValue(["uid"]))
            another_uid = "hello"
            assert_is_value(another_uid, KnownValue("hello"))

            d = {"uid": self}
            assert_is_value(d["uid"], VariableNameValue(["uid"]))
            assert_is_value(self.uid, VariableNameValue(["uid"]))


class TestNewType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing import NewType

        Uid = NewType("Uid", int)

        def capybara(uid: Uid):
            assert_is_value(uid, NewTypeValue("Uid", TypedValue(int), Uid))

    @assert_passes()
    def test_operation(self):
        from typing import NewType

        from typing_extensions import assert_type

        Uid = NewType("Uid", int)

        def want_int(x: int) -> int:
            return x

        def capybara(uid: Uid):
            assert_type(uid, Uid)
            assert_type(uid + 1, int)
            assert_type(uid.bit_length(), int)
            assert_type(want_int(uid), int)

    @assert_passes()
    def test_variadic_tuple(self):
        from typing import NewType

        from typing_extensions import assert_type

        NT = NewType("NT", tuple[int, ...])

        def capybara(nt: NT, x: tuple[int, ...]):
            assert_type(nt, NT)
            assert_type(nt * 2, tuple[int, ...])
            assert_type(nt[0], int)

    @assert_passes(allow_import_failures=True)
    def test_static_fallback_behavior(self):
        from typing import Any, Hashable, Literal, NewType, TypedDict, TypeVar

        from typing_extensions import assert_type

        UserId = NewType("UserId", int)

        UserId("user")  # E: incompatible_argument
        u1: UserId = 42  # E: incompatible_assignment
        u2: UserId = UserId(42)
        assert_type(UserId(5) + 1, int)
        _: type = UserId  # E: incompatible_assignment
        isinstance(u2, UserId)  # E: incompatible_argument

        class UserIdDerived(UserId):  # E: invalid_base
            pass

        GoodName = NewType("BadName", int)  # E: incompatible_call

        GoodNewType1 = NewType("GoodNewType1", list)
        GoodNewType2 = NewType("GoodNewType2", GoodNewType1)
        _nt1: GoodNewType1[int]  # E: unsupported_operation
        TypeAlias1 = dict[str, str]
        GoodNewType3 = NewType("GoodNewType3", TypeAlias1)

        BadNewType1 = NewType("BadNewType1", int | str)  # E: incompatible_call
        T = TypeVar("T")
        BadNewType2 = NewType("BadNewType2", list[T])  # E: incompatible_call
        BadNewType3 = NewType("BadNewType3", Hashable)  # E: incompatible_call
        BadNewType4 = NewType("BadNewType4", Literal[7])  # E: incompatible_call

        class TD1(TypedDict):
            a: int

        BadNewType5 = NewType("BadNewType5", TD1)  # E: incompatible_call
        BadNewType6 = NewType("BadNewType6", int, int)  # E: incompatible_call
        BadNewType7 = NewType("BadNewType7", Any)  # E: incompatible_call


class TestTypingConstructNameMatching(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assignment_target_name_mismatch(self):
        from typing import NamedTuple, NewType, TypedDict, TypeVar

        from typing_extensions import ParamSpec, TypeVarTuple

        GoodTypeVar = TypeVar("GoodTypeVar")
        BadTypeVar = TypeVar("WrongTypeVar")  # E: incompatible_call
        GoodTypeVarTuple = TypeVarTuple("GoodTypeVarTuple")
        BadTypeVarTuple = TypeVarTuple("WrongTypeVarTuple")  # E: incompatible_call
        GoodParamSpec = ParamSpec("GoodParamSpec")
        BadParamSpec = ParamSpec("WrongParamSpec")  # E: incompatible_call
        GoodNewType = NewType("GoodNewType", int)
        BadNewType = NewType("WrongNewType", int)  # E: incompatible_call
        GoodNamedTuple = NamedTuple("GoodNamedTuple", [("x", int)])
        BadNamedTuple = NamedTuple(
            "WrongNamedTuple", [("x", int)]  # E: incompatible_call
        )
        GoodTypedDict = TypedDict("GoodTypedDict", {"x": int})
        BadTypedDict = TypedDict("WrongTypedDict", {"x": int})  # E: incompatible_call
        print(
            GoodTypeVar,
            BadTypeVar,
            GoodTypeVarTuple,
            BadTypeVarTuple,
            GoodParamSpec,
            BadParamSpec,
            GoodNewType,
            BadNewType,
            GoodNamedTuple,
            BadNamedTuple,
            GoodTypedDict,
            BadTypedDict,
        )

    @skip_before((3, 11))
    def test_generic_typevartuple_base_validation(self):
        self.assert_passes(
            """
            from typing import Generic, TypeVarTuple

            Shape = TypeVarTuple("Shape")
            Ts1 = TypeVarTuple("Ts1")
            Ts2 = TypeVarTuple("Ts2")

            class Good(Generic[*Shape]):
                ...

            class Bad(Generic[Shape]):  # E: invalid_annotation
                ...

            class Bad2(Generic[*Ts1, *Ts2]):  # E: invalid_annotation
                ...
            """,
            allow_import_failures=True,
        )


class TestImports(TestNameCheckVisitorBase):
    def test_star_import(self):
        self.assert_passes("""
            from typing import *

            get_args(42)
            """)

    @assert_passes()
    def test_local_import(self):
        import inspect as _inspect

        def capybara(foo):
            import inspect

            assert_is_value(inspect.signature, KnownValue(_inspect.signature))
            assert_is_value(inspect.signature(capybara), TypedValue(inspect.Signature))
            assert_is_value(_inspect.signature(capybara), TypedValue(inspect.Signature))

    @assert_passes()
    def test_local_import_from(self):
        from typing import get_args as _get_args

        def capybara(foo):
            from typing import get_args

            assert_is_value(get_args, KnownValue(_get_args))

    @assert_passes()
    def test_transform_globals(self):
        from unittest.mock import ANY

        def f():
            assert_is_value(ANY, AnyValue(AnySource.explicit))


class TestComprehensions(TestNameCheckVisitorBase):
    @assert_passes()
    def test_scoping_in_list_py3(self):
        def capybara(self):
            x = [a for a in (1, 2)]
            return a, x  # E: undefined_name

    @assert_passes()
    def test_scoping_in_set(self):
        def capybara(self):
            x = {a for a in (1, 2)}
            return a, x  # E: undefined_name

    @assert_passes()
    def test_scoping_in_generator(self):
        def capybara(self):
            x = (a for a in (1, 2))
            return a, x  # E: undefined_name

    @assert_passes()
    def test_scoping_in_dict(self):
        def capybara(self):
            x = {a: 3 for a in (1, 2)}
            return a, x  # E: undefined_name

    @assert_passes()
    def test_incomplete_value(self):
        import types

        def capybara(lst):
            a = [int(x) for x in lst]
            assert_is_value(a, SequenceValue(list, [(True, TypedValue(int))]))

            b = (0 for _ in lst)
            assert_is_value(
                b,
                GenericValue(
                    types.GeneratorType,
                    [KnownValue(0), KnownValue(None), KnownValue(None)],
                ),
            )

            c = {int(x): int(x) for x in lst}
            assert_is_value(
                c,
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(int), TypedValue(int), is_many=True)]
                ),
            )

    @assert_passes()
    def test_sequence_iterable(self):
        # this failed previously because str has no __iter__, but is iterable
        def capybara(oid):
            tmp = str(oid)
            return [s for s in tmp]

    @assert_passes()
    def test_comprehension_body_within_class(self):
        def capybara():
            class Capybara(object):
                incisors = [1, 2]
                canines = {incisors[0] for _ in incisors}  # E: undefined_name

    @assert_passes()
    def test_comprehension_within_class(self):
        class Capybara(object):
            incisors = [1, 2]
            canines = {i + 1 for i in incisors}

    @assert_passes()
    def test_hashability(self):
        def capybara(it):
            x = {set() for _ in it}  # E: unhashable_key
            assert_is_value(x, SequenceValue(set, [(True, AnyValue(AnySource.error))]))

            y = {set(): 3 for _ in it}  # E: unhashable_key
            assert_is_value(
                y,
                DictIncompleteValue(
                    dict,
                    [KVPair(AnyValue(AnySource.error), KnownValue(3), is_many=True)],
                ),
            )


class TestIterationTarget(TestNameCheckVisitorBase):
    @assert_passes()
    def test_known(self):
        def capybara():
            for char in "hello":
                assert_is_value(
                    char,
                    MultiValuedValue(
                        [
                            KnownValue("h"),
                            KnownValue("e"),
                            KnownValue("l"),
                            KnownValue("o"),
                        ]
                    ),
                )

            for num in [1, 2, 3]:
                assert_is_value(
                    num, MultiValuedValue([KnownValue(1), KnownValue(2), KnownValue(3)])
                )

            for elt in [1, None, "hello"]:
                assert_is_value(
                    elt,
                    MultiValuedValue(
                        [KnownValue(1), KnownValue(None), KnownValue("hello")]
                    ),
                )

    @assert_passes()
    def test_known_not_iterable(self):
        def capybara():
            for _ in 3:  # E: unsupported_operation
                pass

    @assert_passes()
    def test_typed_not_iterable(self):
        def capybara(x):
            for _ in int(x):  # E: unsupported_operation
                pass

    @assert_passes()
    def test_union_iterable(self):
        from typing import List, Set, Union

        def capybara(x: Union[List[int], Set[str]]) -> None:
            for obj in x:
                assert_is_value(
                    obj, MultiValuedValue([TypedValue(int), TypedValue(str)])
                )

    @assert_passes()
    def test_generic_iterable(self):
        from typing import Iterable, Tuple, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class ItemsView(Iterable[Tuple[T, U]]):
            pass

        def capybara(it: ItemsView[int, str]):
            for k, v in it:
                assert_is_value(k, TypedValue(int))
                assert_is_value(v, TypedValue(str))

    @assert_passes()
    def test_incomplete(self):
        def capybara(x):
            lst = [1, 2, int(x)]
            assert_is_value(
                lst,
                make_simple_sequence(
                    list, [KnownValue(1), KnownValue(2), TypedValue(int)]
                ),
            )
            for elt in lst:
                assert_is_value(
                    elt,
                    MultiValuedValue([KnownValue(1), KnownValue(2), TypedValue(int)]),
                )

    @assert_passes()
    def test_list_comprehension(self):
        from typing import Sequence

        from typing_extensions import Literal

        def capybara(ints: Sequence[Literal[1, 2]]):
            lst = [x for x in ints]
            mvv = KnownValue(1) | KnownValue(2)
            assert_is_value(lst, SequenceValue(list, [(True, mvv)]))
            for y in lst:
                assert_is_value(y, mvv)

            lst2 = [x for x in (1, 2)]
            assert_is_value(
                lst2, make_simple_sequence(list, [KnownValue(1), KnownValue(2)])
            )

            lst3 = [i + j * 10 for i in range(2) for j in range(3)]
            assert_is_value(lst3, SequenceValue(list, [(True, TypedValue(int))]))

    @assert_passes()
    def test_dict_comprehension(self):
        from typing import Sequence

        from typing_extensions import Literal

        def capybara(ints: Sequence[Literal[1, 2, 3]]):
            dct = {x: x for x in ints}
            mvv = KnownValue(1) | KnownValue(2) | KnownValue(3)
            assert_is_value(
                dct, DictIncompleteValue(dict, [KVPair(mvv, mvv, is_many=True)])
            )

            for key in dct:
                assert_is_value(key, mvv)

            dct2 = {x: x for x in (1, 2, 3)}
            assert_is_value(
                dct2,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue(1), KnownValue(1)),
                        KVPair(KnownValue(2), KnownValue(2)),
                        KVPair(KnownValue(3), KnownValue(3)),
                    ],
                ),
            )

    @assert_passes()
    def test_maybe_empty(self):
        def capybara(cond):
            lst = []
            if cond:
                lst.append("x")
            assert_is_value(lst, KnownValue(["x"]) | KnownValue([]))
            for c in lst:
                assert_is_value(c, KnownValue("x"))

    @assert_passes()
    def test_old_style(self):
        class HasGetItem:
            def __getitem__(self, i: int) -> str:
                return str(i)

        class BadGetItem:
            def __getitem__(self, i: int, extra: bool) -> str:
                return str(i) + str(extra)

        def capybara():
            for x in HasGetItem():
                assert_is_value(x, TypedValue(str))

            for x in BadGetItem():  # E: unsupported_operation
                assert_is_value(x, AnyValue(AnySource.error))


class TestYieldInComprehension(TestNameCheckVisitorBase):
    @assert_passes()
    def test_body_only(self):
        def capybara(y):
            [x for x in (yield y)]


class TestUnboundMethodValue(TestNameCheckVisitorBase):
    @assert_passes()
    def test_inference(self):
        from pycroscope.stacked_scopes import Composite
        from pycroscope.tests import PropertyObject

        def capybara(oid):
            assert_is_value(
                PropertyObject(oid).non_async_method,
                UnboundMethodValue(
                    "non_async_method", Composite(TypedValue(PropertyObject))
                ),
            )
            assert_is_value(
                PropertyObject(oid).decorated_method,
                UnboundMethodValue(
                    "decorated_method", Composite(TypedValue(PropertyObject))
                ),
            )
            assert_is_value(
                PropertyObject(1).decorated_method,
                UnboundMethodValue(
                    "decorated_method", Composite(TypedValue(PropertyObject))
                ),
            )
            assert_is_value(
                PropertyObject(1).non_async_method,
                UnboundMethodValue(
                    "non_async_method", Composite(TypedValue(PropertyObject))
                ),
            )
            assert_is_value(
                [oid].append,
                UnboundMethodValue(
                    "append",
                    Composite(
                        make_simple_sequence(list, [AnyValue(AnySource.unannotated)])
                    ),
                ),
            )

    @assert_passes()
    def test_metaclass_super(self):
        from typing import Any, cast
        from unittest.mock import ANY

        from pycroscope.stacked_scopes import Composite, VarnameWithOrigin

        varname = VarnameWithOrigin("self", cast(Any, ANY))

        class Metaclass(type):
            def __init__(self, name, bases, attrs):
                super(Metaclass, self).__init__(name, bases, attrs)
                # TODO: the value is inferred correctly but this test fails because identical super
                # objects don't compare equal
                # assert_is_value(super(Metaclass, self).__init__,
                #                 UnboundMethodValue('__init__', super(Metaclass, Metaclass)))
                assert_is_value(
                    self.__init__,
                    UnboundMethodValue(
                        "__init__", Composite(TypedValue(Metaclass), varname)
                    ),
                )


class TestSubscripting(TestNameCheckVisitorBase):
    @assert_passes()
    def test_list_success(self):
        def capybara():
            return [1, 2][0]

    @assert_passes()
    def test_tuple_success(self):
        def capybara():
            return (1, 2)[0]

    @assert_passes()
    def test_str_success(self):
        def capybara():
            return "12"[0]

    @assert_passes()
    def test_custom_index(self):
        class CustomIndex(object):
            def __index__(self):
                return 0

        def capybara():
            return [1, 2][CustomIndex()]

    @assert_passes()
    def test_permissive_subclass(self):
        from typing import Any

        # Inspired by pyspark.sql.types.Row
        class LetItAllThrough(tuple):
            def __getitem__(self, idx: object) -> Any:
                if isinstance(idx, (int, slice)):
                    return super().__getitem__(idx)
                else:
                    return "whatever"

        def capybara(liat: LetItAllThrough) -> None:
            assert_is_value(liat["x"], AnyValue(AnySource.explicit))
            assert_is_value(liat[0], AnyValue(AnySource.explicit))

    @assert_passes()
    def test_slice(self):
        def capybara():
            return [1, 2][1:]

    @assert_passes()
    def test_failure(self):
        def capybara():
            x = [1, 2]
            return x[3.0]  # E: unsupported_operation

    @assert_passes()
    def test_union(self):
        from typing import Any, Dict, Union

        def capybara(seq: Union[Dict[int, str], Any]) -> None:
            assert_is_value(seq[0], TypedValue(str) | AnyValue(AnySource.from_another))

    @assert_passes()
    def test_weak():
        from typing import Any, Dict, List

        def get_min_max_pk_value(
            min_pks: List[Dict[str, Any]], max_pks: List[Dict[str, Any]]
        ):
            return [r["pk"] for r in [*min_pks, *max_pks]]


class TestNonlocal(TestNameCheckVisitorBase):
    @assert_passes()
    def test_nonlocal(self):
        def capybara():
            x = 3

            def inner_capybara():
                nonlocal x
                assert_is_value(
                    x, MultiValuedValue([KnownValue(4), KnownValue(3), KnownValue(5)])
                )
                x = 4
                assert_is_value(x, KnownValue(4))

            def second_inner():
                nonlocal x
                # this should not throw unused_variable
                x = 5

            return x

    @assert_passes()
    def test_no_unused_var(self):
        def loop():
            running = True

            def handler():
                nonlocal running
                running = False

            return running


class SampleCallSiteCollector(object):
    """Records as string instead of actual reference so can be tested.

    Replaces name_check_visitor.py:CallSiteCollector, since that class records
    real references, which are hard to test against.

    """

    def __init__(self):
        self.map = collections.defaultdict(list)

    def record_call(self, caller, callee):
        try:
            self.map[callee.__qualname__].append(caller.__qualname__)
        except (AttributeError, TypeError):
            # Copied for consistency; see comment in name_check_visitor.py:CallSiteCollector
            pass


class TestCallSiteCollection(TestNameCheckVisitorBase):
    """Base class for testing call site collection."""

    def run_and_get_call_map(self, code_str, **kwargs):
        collector = SampleCallSiteCollector()
        self._run_str(code_str, collector=collector, **kwargs)
        return collector.map

    def test_member_function_call(self):
        call_map = self.run_and_get_call_map("""
            class TestClass(object):
                def __init__(self):
                    self.first_function(5)

                def first_function(self, x):
                    print(x)
                    self.second_function(x, 4)

                def second_function(self, y, z):
                    print(y + z)
            """)

        assert "TestClass.first_function" in call_map["TestClass.second_function"]
        assert "TestClass.__init__" in call_map["TestClass.first_function"]
        assert "TestClass.second_function" in call_map["print"]


class TestUnpacking(TestNameCheckVisitorBase):
    @assert_passes()
    def test_dict_unpacking(self):
        from typing import Dict, Optional

        from typing_extensions import NotRequired, TypedDict

        class FullTD(TypedDict):
            a: int
            b: str

        class PartialTD(TypedDict):
            a: int
            b: NotRequired[str]

        def capybara(
            d: Dict[str, int],
            ftd: FullTD,
            ptd: PartialTD,
            maybe_ftd: Optional[FullTD] = None,
        ):
            d1 = {1: 2}
            d2 = {3: 4, **d1}
            assert_is_value(
                d2,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue(3), KnownValue(4)),
                        KVPair(KnownValue(1), KnownValue(2)),
                    ],
                ),
            )
            assert_is_value(
                {1: 2, **d},
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue(1), KnownValue(2)),
                        KVPair(TypedValue(str), TypedValue(int), is_many=True),
                    ],
                ),
            )
            assert_is_value(
                {1: 2, **ftd},
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue(1), KnownValue(2)),
                        KVPair(KnownValue("a"), TypedValue(int)),
                        KVPair(KnownValue("b"), TypedValue(str)),
                    ],
                ),
            )
            assert_is_value(
                {1: 2, **ptd},
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue(1), KnownValue(2)),
                        KVPair(KnownValue("a"), TypedValue(int)),
                        KVPair(KnownValue("b"), TypedValue(str), is_required=False),
                    ],
                ),
            )
            assert_is_value(
                {**(maybe_ftd or {})},
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(KnownValue("a"), TypedValue(int), is_required=False),
                        KVPair(KnownValue("b"), TypedValue(str), is_required=False),
                    ],
                ),
            )

    @assert_passes()
    def test_minimal_mapping(self):
        from typing import List

        class MyMapping:
            def keys(self) -> List[bool]:
                raise NotImplementedError

            def __getitem__(self, key: bool) -> bytes:
                raise NotImplementedError

        def capybara(m: MyMapping):
            assert_is_value(
                {**m},
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(bool), TypedValue(bytes), is_many=True)]
                ),
            )

    @assert_passes()
    def test_iterable_unpacking(self):
        def capybara(x):
            degu = (1, *x)
            assert_is_value(
                degu,
                SequenceValue(
                    tuple,
                    [
                        (False, KnownValue(1)),
                        (True, AnyValue(AnySource.generic_argument)),
                    ],
                ),
            )

            z = [1, *(2, 3)]
            assert_is_value(z, KnownValue([1, 2, 3]))

    @assert_passes()
    def test_not_iterable(self):
        def capybara(x: int):
            (*x,)  # E: unsupported_operation

    @assert_passes()
    def test_bad_unpack(self):
        def too_many_values():
            a, b = 1, 2, 3  # E: bad_unpack
            return a, b

        def too_few_values():
            a, b, c = 1, 2  # E: bad_unpack
            return a, b, c

        def too_few_values_list():
            [a, b, c] = 1, 2  # E: bad_unpack
            return a, b, c

        def too_short_generalized():
            a, b, *c = (1,)  # E: bad_unpack
            return a, b, c

    @assert_passes()
    def test_correct_unpack(self):
        from typing import Any, List, Tuple, Union

        def run(lst: List[int], union: Union[Any, List[int], Tuple[str, float]]):
            a, b = 1, 2
            assert_is_value(a, KnownValue(1))
            assert_is_value(b, KnownValue(2))

            c, d = lst
            assert_is_value(c, TypedValue(int))
            assert_is_value(d, TypedValue(int))

            e, f = (lst, 42)
            assert_is_value(e, GenericValue(list, [TypedValue(int)]))
            assert_is_value(f, KnownValue(42))

            g, h = union
            assert_is_value(
                g,
                AnyValue(AnySource.generic_argument)
                | TypedValue(int)
                | TypedValue(str),
            )
            assert_is_value(
                h,
                AnyValue(AnySource.generic_argument)
                | TypedValue(int)
                | TypedValue(float),
            )

            long_tuple = (1, 2, 3, 4, 5, 6)
            *i, j, k = long_tuple
            assert_is_value(
                i,
                make_simple_sequence(
                    list, [KnownValue(1), KnownValue(2), KnownValue(3), KnownValue(4)]
                ),
            )
            assert_is_value(j, KnownValue(5))
            assert_is_value(k, KnownValue(6))
            l, m, *n, o, p = long_tuple
            assert_is_value(l, KnownValue(1))
            assert_is_value(m, KnownValue(2))
            assert_is_value(
                n, make_simple_sequence(list, [KnownValue(3), KnownValue(4)])
            )
            assert_is_value(o, KnownValue(5))
            assert_is_value(p, KnownValue(6))

            q, r, *s = (1, 2)
            assert_is_value(q, KnownValue(1))
            assert_is_value(r, KnownValue(2))
            assert_is_value(s, SequenceValue(list, []))

            for sprime in []:
                assert_is_value(sprime, NO_RETURN_VALUE)

            for t, u in []:
                assert_is_value(t, AnyValue(AnySource.unreachable))
                assert_is_value(u, AnyValue(AnySource.unreachable))

            known_list = [1, 2]
            v, w = known_list
            assert_is_value(v, KnownValue(1))
            assert_is_value(w, KnownValue(2))

            if lst:
                known_list.append(3)

            # We allow this unsafe code to avoid false positives
            x, y = known_list
            assert_is_value(
                x, MultiValuedValue([KnownValue(1), KnownValue(2), KnownValue(3)])
            )

    @assert_passes()
    def test_unpack_int(self):
        def run():
            a, b = 1, 2
            a()  # E: not_callable
            return a, b


class TestUnusedIgnore(TestNameCheckVisitorBase):
    @assert_passes()
    def test_unused(self):
        def capybara(condition):
            x = 1
            print(x)  # static analysis: ignore[undefined_name]  # E: unused_ignore
            print(x)  # static analysis: ignore  # E: unused_ignore  # E: bare_ignore

    @assert_passes()
    def test_used(self):
        def capybara(condition):
            print(x)  # static analysis: ignore[undefined_name]


class TestTypeIgnore(TestNameCheckVisitorBase):
    @assert_passes()
    def test_inline_type_ignore(self):
        x: int = ""  # type: ignore
        y: int = ""  # type: ignore - additional text
        z: int = ""  # type: ignore[attr-defined]
        a: int = ""  # type: ignore # comment

    def test_top_of_file_type_ignore(self):
        self.assert_passes("""
            #!/usr/bin/env python

            # type: ignore

            x: int = ""
            """)

    def test_type_ignore_not_file_level_after_docstring(self):
        self.assert_passes('''
            """module doc"""

            # type: ignore

            x: int = ""  # E: incompatible_assignment
            ''')


class TestNestedLoop(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara(x: int):
            v = 1
            while x < 2:
                while True:
                    if x == 0:
                        assert_is_value(v, KnownValue(1) | KnownValue(2))
                        break
                v = 2


class TestMissingF(TestNameCheckVisitorBase):
    @assert_passes()
    def test_no_error(self):
        def capybara(func):
            "x"
            "not valid syntax {}"
            b"no byte {f} {string}"
            "{undefined_name} in this string"
            x = 3
            print("a{x}".format(x=x))
            func("translate {x}", x=x)

    @assert_passes()
    def test_missing_f(self):
        def capybara():
            x = 3
            return "x = {x}"  # E: missing_f

    def test_autofix(self):
        self.assert_is_changed(
            """
            def capybara():
                x = 3
                return "x = {x}"
            """,
            """
            def capybara():
                x = 3
                return f'x = {x}'
            """,
        )


class TestFStrings(TestNameCheckVisitorBase):
    @assert_passes()
    def test_fstr(self):
        def capybara(x):
            y = f"{x} stuff"
            assert_is_value(y, TypedValue(str))

    @assert_passes()
    def test_undefined_name(self):
        def capybara():
            return f"{x}"  # E: undefined_name


_AnnotSettings = {
    ErrorCode.missing_parameter_annotation: True,
    ErrorCode.missing_return_annotation: True,
}


class TestRequireAnnotations(TestNameCheckVisitorBase):
    @assert_passes(settings=_AnnotSettings)
    def test_missing_annotation(self):
        def no_param(x) -> None:  # E: missing_parameter_annotation
            pass

        # E: missing_return_annotation
        def no_return(x: object):
            pass

        class Capybara:
            def f(self, x) -> None:  # E: missing_parameter_annotation
                pass

    @assert_passes(settings=_AnnotSettings)
    def test_dont_annotate_self():
        def f() -> None:
            class X:
                def method(self) -> None:
                    pass

        class X:
            def f(self) -> None:
                pass


class TestAnnAssign(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing_extensions import Final

        def capybara() -> None:
            x: Final = 3
            assert_is_value(x, KnownValue(3))
            y: int = 3
            assert_is_value(y, TypedValue(int))
            z: bytes
            print(z)  # E: undefined_name

            y: bytes = b"ytes"  # E: already_declared
            assert_is_value(y, TypedValue(bytes))

    @assert_passes()
    def test_final(self):
        from typing_extensions import Final

        x: Final = 1
        x = 2  # E: incompatible_assignment

        def capybara():
            y: Final = 1  # E: unused_assignment
            y = 2  # E: incompatible_assignment
            return y

    @assert_passes()
    def test_final_class_attributes(self):
        from typing_extensions import Final

        class Capybara:
            missing: Final[int]  # E: invalid_annotation
            initialized_in_init: Final[int]
            initialized_in_class: Final[int] = 0

            def __init__(self) -> None:
                self.initialized_in_init = 1
                self.initialized_in_class = 1  # E: incompatible_assignment

            def method(self) -> None:
                self.initialized_in_init = 2  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_final_decorator_in_unimportable_module(self):
        from typing import final

        import does_not_exist  # noqa: F401

        @final
        class FinalBase:
            pass

        class FinalChild(FinalBase):  # E: invalid_annotation
            pass

        class Parent:
            @final
            def method(self) -> None:
                pass

        class Child(Parent):
            def method(self) -> None:  # E: invalid_annotation
                pass

        @final
        def f() -> None:  # E: invalid_annotation
            pass

    @assert_passes(allow_import_failures=True)
    def test_protocol_merging_in_unimportable_module(self):
        from abc import abstractmethod
        from collections.abc import Sized
        from typing import Protocol

        import does_not_exist  # noqa: F401

        class SizedAndClosable1(Sized, Protocol):
            def close(self) -> None: ...

        class SizedAndClosable2(Protocol):
            def __len__(self) -> int: ...

            def close(self) -> None: ...

        class SCConcrete1:
            def __len__(self) -> int:
                return 0

            def close(self) -> None:
                pass

        class SCConcrete2:
            def close(self) -> None:
                pass

        s1: SizedAndClosable1 = SCConcrete1()
        s2: SizedAndClosable2 = SCConcrete1()

        bad1: SizedAndClosable1 = SCConcrete2()  # E: incompatible_assignment
        bad2: SizedAndClosable2 = SCConcrete2()  # E: incompatible_assignment

        def accepts_both(p1: SizedAndClosable1, p2: SizedAndClosable2) -> None:
            merged1: SizedAndClosable2 = p1
            merged2: SizedAndClosable1 = p2
            print(merged1, merged2)

        class SizedClosableFlush(SizedAndClosable2, Protocol):
            def flush(self) -> None: ...

        class NotAProtocol(SizedAndClosable1):
            pass

        class BadProto(NotAProtocol, Protocol):  # E: invalid_base
            ...

        class FlushOnly:
            def flush(self) -> None:
                pass

        class WithFlush:
            def __len__(self) -> int:
                return 0

            def close(self) -> None:
                pass

            def flush(self) -> None:
                pass

        class AbstractSized(SizedAndClosable1):
            @abstractmethod
            def close(self) -> None:
                raise NotImplementedError

        x = AbstractSized()  # E: incompatible_call
        f1: SizedClosableFlush = WithFlush()
        f2: SizedClosableFlush = FlushOnly()  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_self_methods_in_unimportable_generic_module(self):
        from typing import Generic, TypeVar

        import does_not_exist  # noqa: F401
        from typing_extensions import Self, assert_type

        T = TypeVar("T")

        class Base:
            def f(self) -> Self:
                return self

        class Child(Base):
            pass

        class Box(Generic[T]):
            def set_value(self, value: T) -> Self:
                return self

        def capybara(child: Child, int_box: Box[int]):
            assert_type(child.f(), Child)
            assert_type(int_box.set_value(1), Box[int])

    @assert_passes(allow_import_failures=True)
    def test_self_classmethod_in_unimportable_module(self):
        import does_not_exist  # noqa: F401
        from typing_extensions import Self, assert_type

        class Shape:
            @classmethod
            def from_config(cls, config: dict[str, float]) -> Self:
                return cls()

        class Circle(Shape):
            pass

        def capybara():
            assert_type(Shape.from_config({}), Shape)
            assert_type(Circle.from_config({}), Circle)

    @assert_passes(allow_import_failures=True)
    def test_typevar_classmethod_in_unimportable_module(self):
        from typing import TypeVar

        import does_not_exist  # noqa: F401

        T = TypeVar("T")

        class Box:
            @classmethod
            def identity(cls, value: T) -> T:
                return value

        def capybara():
            Box.identity(1).bit_length()
            Box.identity("x").upper()

    @assert_passes(allow_import_failures=True)
    def test_generic_alias_constructor_in_unimportable_module(self):
        from typing import Generic, TypeVar, assert_type

        import does_not_exist  # noqa: F401

        T = TypeVar("T")

        class Node(Generic[T]):
            label: T

            def __init__(self, label: T | None = None) -> None:
                if label is not None:
                    self.label = label

        n1 = Node[int]()
        n2 = Node[str]()
        assert_type(n1, Node[int])
        assert_type(n2, Node[str])
        assert_type(Node[int]().label, int)

        Node[int](0)
        Node[int]("")  # E: incompatible_argument
        Node[str]("")
        Node[str](0)  # E: incompatible_argument

    @assert_passes()
    def test_protocol_override_keeps_compatible_self_type(self):
        from abc import abstractmethod
        from collections.abc import Sized
        from typing import Protocol

        class SizedAndClosable(Sized, Protocol):
            def close(self) -> None: ...

        class AbstractSized(SizedAndClosable):
            @abstractmethod
            def close(self) -> None:
                raise NotImplementedError

    @assert_passes()
    def test_inconsistent_type(self):
        def capybara():
            x: int = 1
            assert_is_value(x, TypedValue(int))
            x = "x"  # E: incompatible_assignment

            y: int = "y"  # E: incompatible_assignment
            return (x, y)

    @assert_passes()
    def test_class_scope(self):
        class Capybara:
            x: int = 0
            assert_is_value(x, TypedValue(int))

            def __init__(self) -> None:
                self.y: object = 3

            def method(self):
                assert_is_value(self.y, TypedValue(object))

    @assert_passes()
    def test_loop(self):
        def capybara():
            for i in range(3):
                j: int = i  # E: unused_variable

            j: int = 0  # E: already_declared  # E: unused_variable

    @assert_passes()
    def test_module_scope(self):
        x: int = 3
        assert_is_value(x, TypedValue(int))

        if __name__ == "__main__":
            y: int = 3
            assert_is_value(y, TypedValue(int))


class TestWhile(TestNameCheckVisitorBase):
    @assert_passes()
    def test_while_true_reachability(self):
        def capybara() -> int:  # E: missing_return
            while True:
                break

        def pacarana(i: int) -> int:
            while True:
                if i == 1:
                    return 1


class TestWith(TestNameCheckVisitorBase):
    @assert_passes()
    def test_with(self) -> None:
        class BadCM1:
            pass

        class BadCM2:
            def __enter__(self, extra_arg) -> int:
                return 0

            def __exit__(self, typ, value, tb):
                pass

        class GoodCM:
            def __enter__(self) -> int:
                return 0

            def __exit__(self, typ, value, tb):
                pass

        def capybara():
            with BadCM1() as e:  # E: invalid_context_manager
                assert_is_value(e, AnyValue(AnySource.error))

            with BadCM2() as e:  # E: invalid_context_manager
                assert_is_value(e, AnyValue(AnySource.error))

            with GoodCM() as e:
                assert_is_value(e, TypedValue(int))

    @assert_passes()
    def test_async_with(self) -> None:
        class BadCM1:
            pass

        class BadCM2:
            async def __aenter__(self, extra_arg) -> int:
                return 0

            async def __aexit__(self, typ, value, tb):
                pass

        class GoodCM:
            async def __aenter__(self) -> int:
                return 0

            async def __aexit__(self, typ, value, tb):
                pass

        async def capybara():
            async with BadCM1() as e:  # E: invalid_context_manager
                assert_is_value(e, AnyValue(AnySource.error))

            async with BadCM2() as e:  # E: invalid_context_manager
                assert_is_value(e, AnyValue(AnySource.error))

            async with GoodCM() as e:
                assert_is_value(e, TypedValue(int))


class HasGetattr(object):
    def __getattr__(self, attr):
        return 42

    def real_method(self):
        pass


class HasGetattribute(object):
    def __getattribute__(self, attr):
        return 43

    def real_method(self):
        pass


def test_static_hasattr():
    hga = HasGetattr()
    assert _static_hasattr(hga, "real_method")
    assert _static_hasattr(hga, "__getattr__")
    assert not _static_hasattr(hga, "random_attribute")

    hgat = HasGetattribute()
    assert _static_hasattr(hgat, "real_method")
    assert _static_hasattr(hgat, "__getattribute__")
    assert not _static_hasattr(hgat, "random_attribute")


class TestIncompatibleOverride(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing_extensions import Literal

        class A:
            x: str
            y: int

            def capybara(self, x: int) -> None:
                pass

            def pacarana(self, b: int) -> None:
                pass

        class B(A):
            x: int  # E: incompatible_override
            y: Literal[1]

            def capybara(self, x: str) -> None:  # E: incompatible_override
                pass

            def pacarana(self, b: int) -> None:
                pass

    @assert_passes()
    def test_property_unannotated(self):
        class Unannotated:
            @property
            def f(self):
                pass

        class UnannotatedChild(Unannotated):
            @property
            def f(self):
                pass

    @assert_passes()
    def test_property_annotated(self):
        class Annotated:
            @property
            def f(self) -> int:
                return 0

        class AnnotatedChild(Annotated):
            @property
            def f(self) -> int:
                return 0

        class MoreSpecificChild(Annotated):
            @property
            def f(self) -> bool:
                return True

        class JustPutItInTheClassChild(Annotated):
            f: int

        class BadChild(Annotated):
            @property
            def f(self) -> str:  # E: incompatible_override
                return ""

    @assert_passes()
    def test_property_with_setter(self):
        class Annotated:
            @property
            def f(self) -> int:
                return 0

            @f.setter
            def f(self, value: int) -> None:
                pass

        class AnnotatedChild(Annotated):
            @property
            def f(self) -> int:  # E: incompatible_override
                return 0

        class AnnotatedChildWithSetter(Annotated):
            @property
            def f(self) -> int:
                return 0

            @f.setter
            def f(self, value: int) -> None:
                pass

        class JustPutItInTheClassChild(Annotated):
            f: int

        class JustPutItInTheClassWithMoreSpecificType(Annotated):
            f: bool  # E: incompatible_override

        class AnnotatedChildWithMoreSpecificSetter(Annotated):
            @property
            def f(self) -> bool:  # E: incompatible_override
                return False

            @f.setter
            def f(self, value: bool) -> None:  # E: incompatible_override
                pass


class TestWalrus(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Optional

        def opt() -> Optional[int]:
            return None

        def capybara():
            if x := opt():
                assert_is_value(x, TypedValue(int))
            assert_is_value(x, TypedValue(int) | KnownValue(None))

            if (y := opt()) is not None:
                assert_is_value(y, TypedValue(int))
            assert_is_value(y, TypedValue(int) | KnownValue(None))

    @assert_passes()
    def test_and(self):
        from typing import Optional, Set

        def opt() -> Optional[int]:
            return None

        def capybara(cond):
            if (x := opt()) and cond:
                assert_is_value(x, TypedValue(int))
            assert_is_value(x, TypedValue(int) | KnownValue(None))

        def func(myvar: str, strset: Set[str]) -> None:
            if (encoder_type := myvar) and myvar in strset:
                print(encoder_type)

    @assert_passes()
    def test_and_then_walrus(self):
        from typing_extensions import Literal, assert_type

        def capybara(cond):
            if cond and (x := 1):
                assert_type(x, Literal[1])
            else:
                print(x)  # E: possibly_undefined_name
            print(x)  # E: possibly_undefined_name

    @assert_passes()
    def test_if_exp(self):
        def capybara(cond):
            (x := 2) if cond else (x := 1)
            assert_is_value(x, KnownValue(2) | KnownValue(1))

    @assert_passes()
    def test_comprehension_scope(self):
        from typing import List, Optional

        def capybara(elts: List[Optional[int]]) -> None:
            if any((x := i) is not None for i in elts):
                assert_is_value(x, TypedValue(int) | KnownValue(None))
                print(i)  # E: undefined_name


class TestUnion(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union_to_union(self):
        from typing import Optional

        def capybara(x: Optional[str], y: Optional[str]) -> Optional[str]:
            assert_is_value(x, TypedValue(str) | KnownValue(None))
            assert_is_value(y, TypedValue(str) | KnownValue(None))
            return x or y


class TestContextManagerWithSuppression(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sync(self):
        import contextlib
        from types import TracebackType
        from typing import ContextManager, Iterator, Optional, Type

        class SuppressException:
            def __enter__(self) -> None:
                pass

            def __exit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> bool:
                return isinstance(exn, Exception)

        class EmptyContext(object):
            def __enter__(self) -> None:
                pass

            def __exit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> None:
                pass

        class MaybeSuppressException:
            def __enter__(self) -> None:
                pass

            def __exit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> Optional[bool]:
                return None

        def empty_context_manager() -> ContextManager[None]:
            return EmptyContext()

        @contextlib.contextmanager
        def empty_contextlib_manager() -> Iterator[None]:
            yield

        def use_suppress_exception():
            a = 2
            with SuppressException():
                a = 3
            assert_is_value(a, KnownValue(2) | KnownValue(3))

        def use_suppress_exception_multi_assignment():
            a = 2
            with SuppressException():
                a = 3
                a = 4
            assert_is_value(a, KnownValue(2) | KnownValue(3) | KnownValue(4))

        def use_empty_context():
            a = 2  # E: unused_assignment
            with EmptyContext():
                a = 3  # E: unused_assignment
                a = 4
            assert_is_value(a, KnownValue(4))

        def use_context_manager():
            a = 2  # E: unused_assignment
            with empty_context_manager():
                a = 3
            assert_is_value(a, KnownValue(3))

        def use_builtin_function():
            a = 2  # E: unused_assignment
            with open("test_file.txt"):
                a = 3
            assert_is_value(a, KnownValue(3))

        def use_contextlib_manager():
            a = 2  # E: unused_assignment
            with empty_contextlib_manager():
                a = 3
            assert_is_value(a, KnownValue(3))

        def use_optional_bool_return(x: int | str) -> None:
            if isinstance(x, int):
                with MaybeSuppressException():
                    raise ValueError
            assert_is_value(x, TypedValue(str))

        def use_nested_contexts():
            b = 2
            with SuppressException(), EmptyContext() as b:
                assert_is_value(b, KnownValue(None))
            assert_is_value(b, KnownValue(2) | KnownValue(None))

            c = 2  # E: unused_assignment
            with EmptyContext() as c, SuppressException():
                assert_is_value(c, KnownValue(None))
            assert_is_value(c, KnownValue(None))

    @assert_passes()
    def test_possibly_undefined_with_leaves_scope(self):
        from types import TracebackType
        from typing import Optional, Type

        class SuppressException:
            def __enter__(self) -> None:
                pass

            def __exit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> bool:
                return isinstance(exn, Exception)

        def use_suppress_with_nested_block():
            with SuppressException():
                a = 4
                try:
                    b = 3
                except Exception:
                    return
            print(a)  # E: possibly_undefined_name
            print(b)  # E: possibly_undefined_name

    @assert_passes()
    def test_async(self):
        from types import TracebackType
        from typing import AsyncContextManager, Optional, Type

        class AsyncSuppressException(object):
            async def __aenter__(self) -> None:
                pass

            async def __aexit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> bool:
                return isinstance(exn, Exception)

        class AsyncEmptyContext(object):
            async def __aenter__(self) -> None:
                pass

            async def __aexit__(
                self,
                typ: Optional[Type[BaseException]],
                exn: Optional[BaseException],
                tb: Optional[TracebackType],
            ) -> None:
                pass

        def async_empty_context_manager() -> AsyncContextManager[None]:
            return AsyncEmptyContext()

        async def use_async_suppress_exception():
            a = 2
            async with AsyncSuppressException():
                a = 3
            assert_is_value(a, KnownValue(2) | KnownValue(3))

        async def use_async_empty_context():
            a = 2  # E: unused_assignment
            async with AsyncEmptyContext():
                a = 3
            assert_is_value(a, KnownValue(3))

        async def use_async_context_manager():
            a = 2  # E: unused_assignment
            async with async_empty_context_manager():
                a = 3
            assert_is_value(a, KnownValue(3))

        async def use_async_nested_contexts():
            b = 2
            async with AsyncSuppressException(), AsyncEmptyContext() as b:
                assert_is_value(b, KnownValue(None))
            assert_is_value(b, KnownValue(2) | KnownValue(None))

            c = 2  # E: unused_assignment
            async with AsyncEmptyContext() as c, AsyncSuppressException():
                assert_is_value(c, KnownValue(None))
            assert_is_value(c, KnownValue(None))

    def test_async_contextlib_manager(self):
        import contextlib
        from typing import AsyncIterator

        @contextlib.asynccontextmanager
        async def async_empty_contextlib_manager() -> AsyncIterator[None]:
            yield

        async def use_async_contextlib_manager():
            a = 2  # E: unused_assignment
            async with async_empty_contextlib_manager():
                a = 3
            assert_is_value(a, KnownValue(3))


class TestMustUse(TestNameCheckVisitorBase):
    @assert_passes()
    def test_generator(self):
        from typing import Generator

        def gen() -> Generator[int, None, None]:
            yield 1
            yield 2

        def capybara() -> None:
            gen()  # E: must_use

    @assert_passes()
    def test_async_generator(self):
        from typing import AsyncGenerator

        async def gen() -> AsyncGenerator[int, None]:
            yield 1
            yield 2

        def capybara() -> None:
            gen()  # E: must_use


class TestImplicitAny(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.implicit_any: True})
    def test_class_args(self):
        class Base:
            def __init_subclass__(cls, **kwargs) -> None:
                pass

        class X(Base, a=True):
            pass
