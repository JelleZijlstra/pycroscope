# static analysis: ignore
import ast
import collections
import os
import textwrap
import types

from typing_extensions import assert_type

from . import test_node_visitor
from .analysis_lib import make_module
from .checker import Checker
from .error_code import DISABLED_IN_TESTS, ErrorCode
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
    TypeVarParam,
    TypeVarValue,
    UnboundMethodValue,
    VariableNameValue,
    assert_is_value,
    dump_value,
)

BOX_FLOAT_IN_TEST_INPUT = GenericValue("<test input>.Box", [TypedValue(float)])
BOX_FLOAT_OR_INT_IN_TEST_INPUT = GenericValue(
    "<test input>.Box", [TypedValue(float) | TypedValue(int)]
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

    def assert_passes(self, code_str, run_in_both_module_modes=False, **kwargs):
        if not run_in_both_module_modes:
            return super().assert_passes(code_str, **kwargs)
        if kwargs.get("allow_import_failures"):
            raise AssertionError(
                "run_in_both_module_modes cannot be combined with allow_import_failures"
            )
        if kwargs.get("allow_runtime_module_load_failure"):
            raise AssertionError(
                "run_in_both_module_modes cannot be combined with "
                "allow_runtime_module_load_failure"
            )
        try:
            module = _make_module(textwrap.dedent(code_str))
        except Exception as exc:
            raise AssertionError(
                "run_in_both_module_modes could not execute importable mode: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._assert_passes_in_module_mode(
            "importable", code_str, module=module, **kwargs
        )
        self._assert_passes_in_module_mode(
            "unimportable",
            code_str,
            allow_import_failures=True,
            force_runtime_module_load_failure=True,
            **kwargs,
        )

    def _assert_passes_in_module_mode(self, mode_name, code_str, **kwargs):
        try:
            super().assert_passes(code_str, **kwargs)
        except AssertionError as exc:
            raise AssertionError(f"{mode_name} mode failed:\n{exc}") from exc

    def _run_tree(
        self,
        code_str,
        tree,
        check_attributes=True,
        apply_changes=False,
        settings=None,
        allow_runtime_module_load_failure=False,
        force_runtime_module_load_failure=False,
        module=None,
        **kwargs,
    ):
        # This can happen in Python 2.
        if isinstance(code_str, bytes):
            code_str = code_str.decode("utf-8")
        default_settings = {code: code not in DISABLED_IN_TESTS for code in ErrorCode}
        if settings is not None:
            default_settings.update(settings)
        verbosity = int(os.environ.get("ANS_TEST_SCOPE_VERBOSITY", 0))
        if module is not None:
            mod = module
        elif force_runtime_module_load_failure:
            mod = None
        else:
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
        assert_type=assert_type,
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
        TypeVarParam=TypeVarParam,
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
    @assert_passes(run_in_both_module_modes=True)
    def test_import_failure_points_to_failing_line_and_continues(self):
        a = 1

        def f():
            return missing_name  # E: undefined_name

    @assert_passes(allow_import_failures=True)
    def test_top_level_name_ordering_after_import_failure(self):
        boom = 1 / 0

        print(later_name)  # E: undefined_name
        later_name = 1

        def f() -> int:
            return later_name

    @assert_passes(run_in_both_module_modes=True)
    def test_typing_import_falls_back_to_typing_extensions(self):
        from typing import Generic, TypeVar

        from typing_extensions import Self, assert_type

        T = TypeVar("T")

        class Box(Generic[T]):
            def clone(self) -> Self:
                return self

        assert_type(Box[int]().clone(), Box[int])

    @assert_passes(run_in_both_module_modes=True)
    def test_subscript_value_annotated_with_type_alias(self):
        from typing import TypeAlias

        from typing_extensions import assert_type

        TagsByType: TypeAlias = dict[type[int], list[int]]

        def f(by_type: TagsByType) -> None:
            assert_type(by_type[int], list[int])

    @assert_passes(run_in_both_module_modes=True)
    def test_alias_symbol_still_supports_value_position_specialization(self):
        from typing import TypeVar

        from typing_extensions import TypeAliasType

        T = TypeVar("T")
        ListAlias = TypeAliasType("ListAlias", list[T], type_params=(T,))

        def f() -> None:
            print(ListAlias[int].__value__)

    @assert_passes(allow_runtime_module_load_failure=True)
    def test_import_failure_is_ignorable(self):
        a = 1  # static analysis: ignore[import_failed]
        b = 1 / 0

        def f():
            return missing_name  # E: undefined_name

    @assert_passes(run_in_both_module_modes=True)
    def test_typeddict_fallback_after_import_failure(self):
        from typing import TypedDict

        from typing_extensions import NotRequired, ReadOnly, Required

        class F1(TypedDict):
            a: Required[int]
            b: ReadOnly[NotRequired[int]]
            c: ReadOnly[Required[int]]

        class TD_A1(TypedDict):
            x: int
            y: ReadOnly[int]

        class TD_A2(TypedDict):
            x: float
            y: ReadOnly[float]

        class TD_B1(TypedDict):
            x: ReadOnly[NotRequired[int]]
            y: ReadOnly[Required[int]]

        class TD_B2(TypedDict):
            x: ReadOnly[Required[int]]
            y: ReadOnly[NotRequired[int]]

        def capybara() -> None:
            class F3(F1):
                a: ReadOnly[int]  # E: invalid_annotation

            class F4(F1):
                a: NotRequired[int]  # E: invalid_annotation

            class F5(F1):
                b: ReadOnly[Required[int]]

            class F6(F1):
                c: ReadOnly[NotRequired[int]]  # E: invalid_annotation

            class TD_A(TD_A1, TD_A2): ...  # E: invalid_base

            class TD_B(TD_B1, TD_B2): ...  # E: invalid_base

    @assert_passes(run_in_both_module_modes=True)
    def test_typeddict_extra_items_and_unpack_after_import_failure(self):
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

    @assert_passes(run_in_both_module_modes=True)
    def test_constructor_explicit_self_annotation_after_import_failure(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class C(Generic[T]):
            def __init__(self: "C[int]") -> None:
                pass

        C[int]()
        C[str]()  # E: incompatible_call

    @assert_passes(run_in_both_module_modes=True)
    def test_nominal_class_fallback_after_import_failure(self):
        from typing import Any, overload

        class Desc:
            value: int = 0

            @overload
            def __get__(self, obj: None, owner: Any) -> "Desc": ...

            @overload
            def __get__(self, obj: object, owner: Any) -> int: ...

            def __get__(self, obj: object | None, owner: Any) -> "int | Desc":
                return 1

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_class_attribute_after_import_failure(self):
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
    def test_any_base_class_preserves_declared_method_type_after_import_failure(self):
        from typing import Any

        from typing_extensions import assert_type

        class ClassA(Any):
            def method1(self) -> int:
                return 1

        a = ClassA()
        assert_type(a.method1(), int)
        assert_type(a.method2(), Any)
        assert_type(ClassA.method3(), Any)

    @skip_before((3, 11))
    @assert_passes(run_in_both_module_modes=True)
    def test_any_base_class_preserves_declared_method_type(self):
        from typing import Any

        from typing_extensions import assert_type

        class ClassA(Any):
            def method1(self) -> int:
                return 1

        def capybara(a: ClassA) -> None:
            assert_type(a.method1(), int)
            assert_type(a.method2(), Any)
            assert_type(ClassA.method3(), Any)

    @assert_passes(run_in_both_module_modes=True)
    def test_typevar_annotations_after_import_failure(self):
        from typing import TypeVar

        from typing_extensions import assert_type

        class User: ...

        class TeamUser(User): ...

        U = TypeVar("U", bound=User)

        def func3(user_class: type[U]) -> U:
            return user_class()

        def capybara() -> None:
            assert_type(func3(TeamUser), TeamUser)
            type.unknown  # E: undefined_attribute

    @assert_passes(allow_import_failures=True)
    def test_forward_reference_to_later_dataclass_uses_instance_field_type(self):
        from dataclasses import dataclass

        from typing_extensions import assert_type

        def f(symbol: "C") -> None:
            assert_type(symbol.x, bool)
            if not symbol.x:
                pass

        @dataclass(frozen=True)
        class C:
            x: bool = False

    @assert_passes(allow_import_failures=True)
    def test_forward_reference_to_later_dataclass_member_from_mapping(self):
        from dataclasses import dataclass

        from typing_extensions import assert_type

        def f(symbols: dict[str, "C"]) -> None:
            for symbol in symbols.values():
                assert_type(symbol.is_method, bool)
                if not symbol.is_method:
                    pass

        @dataclass(frozen=True)
        class C:
            is_method: bool = False

    @assert_passes()
    def test_generic_constructor_accepts_known_protocol_value(self):
        import logging
        import sys

        logging.StreamHandler(sys.stderr)

    @assert_passes(allow_import_failures=True)
    def test_callable_protocol_after_import_failure(self):
        from typing import Any, Callable, Protocol

        _Bad: Callable[int]  # E: invalid_annotation

        class Proto(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        def f(p: Proto) -> None:
            cb: Callable[..., None] = p
            cb()

    @assert_passes(run_in_both_module_modes=True)
    def test_type_union_annotation_after_import_failure(self):
        class User: ...

        class BasicUser(User): ...

        class ProUser(User): ...

        class TeamUser(User): ...

        def func4(user_class: type[BasicUser | ProUser]) -> User:
            return user_class()

        def capybara() -> None:
            func4(TeamUser)  # E: incompatible_argument
            type.unknown  # E: undefined_attribute

    @assert_passes(run_in_both_module_modes=True)
    def test_type_arity_and_typing_alias_attrs_after_import_failure(self):
        from typing import Any, Type, TypeAlias

        TA1: TypeAlias = Type
        TA2: TypeAlias = Type[Any]
        TA3: TypeAlias = type
        TA4: TypeAlias = type[Any]

        def capybara() -> None:
            _bad_type1: type[int, str]  # E: invalid_annotation
            TA1.unknown  # E: undefined_attribute
            TA2.unknown  # E: undefined_attribute
            TA3.unknown  # E: undefined_attribute
            TA4.unknown  # E: undefined_attribute
            type.unknown  # E: undefined_attribute

    @assert_passes(allow_import_failures=True)
    def test_explicit_type_alias_callability_after_import_failure(self):
        from typing import TypeAlias

        _bad_type1: type[int, str]  # E: invalid_annotation

        ListAlias: TypeAlias = list
        ListOrSetAlias: TypeAlias = list | set
        _x: list[str] = ListAlias()
        _x2: ListAlias[int]  # E: invalid_annotation
        _x3 = ListOrSetAlias()  # E: not_callable

    @assert_passes(run_in_both_module_modes=True)
    def test_explicit_type_alias_uses_runtime_attribute_semantics(self):
        from typing import TypeAlias

        Alias: TypeAlias = int

        def capybara() -> None:
            print(Alias.bit_count)
            print(Alias.__name__)
            Alias.__value__  # E: undefined_attribute

    @assert_passes(run_in_both_module_modes=True)
    def test_type_object_name_attribute_after_import_failure(self):
        from typing import Type

        from typing_extensions import assert_type

        def f(a: type[object], b: Type[object]) -> None:
            assert_type(a.__name__, str)
            assert_type(b.__name__, str)

        def capybara() -> None:
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
    @assert_passes(run_in_both_module_modes=True)
    def test_overload_consistency_after_import_failure(self):
        from typing import overload

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

    @assert_passes(run_in_both_module_modes=True)
    def test_dict_subclass_assignable_to_dict_after_import_failure(self):
        class CustomDict(dict[str, int]):
            pass

        def takes_dict(x: dict[str, int]) -> None:
            return None

        takes_dict(CustomDict({"num": 1}))

    @assert_passes(allow_import_failures=True)
    def test_generic_base_classes_after_import_failure(self):
        from collections.abc import Container, Iterable, Iterator, Mapping
        from typing import Generic, TypeVar, assert_type

        T = TypeVar("T")
        T1 = TypeVar("T1")
        T2 = TypeVar("T2")
        T3 = TypeVar("T3")

        class Node: ...

        class SymbolTable(dict[str, list[Node]]): ...

        def takes_dict(x: dict): ...  # E: missing_generic_parameters

        def takes_dict_typed(x: dict[str, list[Node]]): ...

        def takes_dict_incorrect(x: dict[str, list[object]]): ...

        def test_symbol_table(s: SymbolTable):
            takes_dict(s)
            takes_dict_typed(s)
            takes_dict_incorrect(s)  # E: incompatible_argument

        def func1(y: Generic[T]):  # E: invalid_annotation
            _x: Generic  # E: invalid_annotation

        class LinkedList(Iterable[T], Container[T]): ...

        def test_linked_list(l: LinkedList[int]):
            assert_type(iter(l), Iterator[int])
            assert_type(l.__contains__(1), bool)

        _linked_list_invalid: LinkedList[int, int]  # E: invalid_annotation

        class MyDict(Mapping[str, T]): ...

        def test_my_dict(d: MyDict[int]):
            assert_type(d["a"], int)

        _my_dict_invalid: MyDict[int, int]  # E: invalid_annotation

        class BadClass1(Generic[T, T]):  # E: invalid_base
            pass

        class Parent1(Generic[T1, T2]): ...

        class Parent2(Generic[T1, T2]): ...

        class Child(Parent1[T1, T3], Parent2[T2, T3]): ...

        def takes_parent1(x: Parent1[int, bytes]): ...

        def takes_parent2(x: Parent2[str, bytes]): ...

        child: Child[int, bytes, str] = Child()
        takes_parent1(child)
        takes_parent2(child)

        class Grandparent(Generic[T1, T2]): ...

        class Parent(Grandparent[T1, T2]): ...

        class BadChild(Parent[T1, T2], Grandparent[T2, T1]): ...  # E: invalid_base

    @assert_passes(allow_import_failures=True)
    def test_generic_instance_attribute_preserves_type_args_after_import_failure(self):
        from typing import Any, Generic, TypeVar, assert_type

        T = TypeVar("T")

        class Node(Generic[T]):
            label: T

            def __init__(self, label: T | None = None) -> None:
                if label is not None:
                    self.label = label

        def check_nodes() -> None:
            assert_type(Node(0).label, int)
            assert_type(Node().label, Any)
            assert_type(Node[int]().label, int)

    @assert_passes(allow_import_failures=True)
    def test_generic_paramspec_attribute_after_import_failure(self):
        from typing import Callable, Generic, ParamSpec, TypeVar, assert_type

        P = ParamSpec("P")
        U = TypeVar("U")

        class Y(Generic[U, P]):
            f: Callable[P, str]
            prop: U

            def __init__(self, f: Callable[P, str], prop: U) -> None:
                self.f = f
                self.prop = prop

        def callback_a(q: int, /) -> str:
            raise NotImplementedError

        def check_paramspec(x: int) -> None:
            y = Y(callback_a, x)
            assert_type(y.prop, int)
            assert_type(y.f, Callable[[int], str])

    @assert_passes(run_in_both_module_modes=True)
    def test_overload_fallback_after_import_failure(self):
        from typing import overload

        from typing_extensions import assert_type

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
                if isinstance(x, int):
                    return x
                return x.encode()

        b = B()
        assert_type(f(1), int)
        assert_type(f("x"), str)

        def capybara(i: int, s: str) -> None:
            assert_type(b[i], int)
            assert_type(b[s], bytes)

    @assert_passes(run_in_both_module_modes=True)
    def test_generic_parameter_order_after_import_failure(self):
        from collections.abc import Iterable, Mapping
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        K = TypeVar("K")
        V = TypeVar("V")

        class LoggedVar(Generic[T]):
            def __init__(self, value: T) -> None:
                self.value = value

            def get(self) -> T:
                return self.value

        def read_vars(vars: Iterable[LoggedVar[int]]) -> None:
            for var in vars:
                assert_type(var.get(), int)

        class ReorderedMap(Mapping[K, V], Generic[V, K]): ...

        def test_reordered_map(m: ReorderedMap[int, str]) -> None:
            assert_type(m["key"], int)
            m[0]  # E: incompatible_argument

    @assert_passes(run_in_both_module_modes=True)
    def test_type_parameter_base_validation_after_import_failure(self):
        from collections.abc import Iterable
        from typing import Generic, Protocol, TypeVar

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)
        S_co = TypeVar("S_co", covariant=True)

        class Good(Iterable[T_co], Protocol): ...

        def capybara() -> None:
            class Bad1(Generic[int]): ...  # E: invalid_base

            class Bad2(Protocol[int]): ...  # E: invalid_base

            class Bad3(Iterable[T_co], Generic[S_co]): ...  # E: invalid_base

            class Bad4(Iterable[T_co], Protocol[S_co]): ...  # E: invalid_base

    @assert_passes(run_in_both_module_modes=True)
    def test_typeddict_class_syntax_after_import_failure(self):
        from typing import TypedDict

        class Movie(TypedDict):
            director: "Person"

        class Person(TypedDict):
            name: str

        def capybara() -> None:
            class BadTypedDict1(TypedDict):
                name: str

                def method(self):  # E: invalid_annotation
                    pass

            class BadTypedDict2(TypedDict, metaclass=type):  # E: invalid_annotation
                name: str

            class BadTypedDict3(TypedDict, other=True):  # E: invalid_annotation
                name: str

    @assert_passes(run_in_both_module_modes=True)
    def test_typeddict_extra_items_and_unpack_after_import_failure(self):
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

    @assert_passes(run_in_both_module_modes=True)
    def test_property_setter_in_synthetic_class_after_import_failure(self):
        class C:
            @property
            def value(self) -> int:
                return 1

            @value.setter
            def value(self, new_value: int) -> None:
                pass

    @assert_passes(run_in_both_module_modes=True)
    def test_synthetic_instance_attrs_and_forward_methods_after_import_failure(self):
        from typing import Generic, TypeVar

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

    @assert_passes(run_in_both_module_modes=True)
    def test_zero_arg_super_in_synthetic_class_after_import_failure(self):
        class Base:
            def method(self) -> int:
                return 1

        class Child(Base):
            def other(self) -> int:
                return super().method()

    @skip_before((3, 11))  # @final doesn't set __final__ in 3.10
    @assert_passes(run_in_both_module_modes=True)
    def test_overloaded_override_and_final_after_import_failure(self):
        from typing import final, overload

        from typing_extensions import override

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

    @assert_passes(run_in_both_module_modes=True)
    def test_synthetic_typeddict_inheritance_edge_cases_after_import_failure(self):
        from typing_extensions import NotRequired, ReadOnly, TypedDict

        class ClosedBase(TypedDict, closed=True):
            name: str

        class BadClosedChild(ClosedBase):
            year: int  # E: invalid_annotation

        class ReadOnlyExtraBase(TypedDict, extra_items=ReadOnly[int]):
            name: str

        class BadReadOnlyExtraChild(ReadOnlyExtraBase):
            year: str  # E: invalid_annotation

        class MutableExtraBase(TypedDict, extra_items=int):
            name: str

        class BadRequiredExtraChild(MutableExtraBase):
            year: int  # E: invalid_annotation

        class BadOptionalExtraChild(MutableExtraBase):
            year: NotRequired[str]  # E: invalid_annotation

        class MutableBase(TypedDict):
            x: int

        class BadMutableOverride(MutableBase):
            x: float  # E: invalid_annotation

        class MutableLeft(TypedDict):
            x: int

        class ReadOnlyRight(TypedDict):
            x: ReadOnly[int]

        class BadBaseMutability(MutableLeft, ReadOnlyRight): ...  # E: invalid_base

        class ReadOnlyLeft(TypedDict):
            x: ReadOnly[int]

        class ReadOnlyWrongType(TypedDict):
            x: ReadOnly[str]

        class BadReadonlyTypes(ReadOnlyLeft, ReadOnlyWrongType): ...  # E: invalid_base

        class OptionalRight(TypedDict):
            x: NotRequired[int]

        class BadBaseRequiredness(MutableLeft, OptionalRight): ...  # E: invalid_base

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
            _y: int  # E: invalid_annotation

        class Location(NamedTuple):
            altitude: float = 0.0
            latitude: float  # E: invalid_annotation

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
    def test_synthetic_instance_annotations_do_not_create_namedtuple_constructor(self):
        class Box:
            value: int

        def f() -> None:
            Box(1)  # E: incompatible_call

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

    @assert_passes()
    def test_hashability_respects_hash_none_for_typed_values(self):
        from typing import Hashable

        class Unhashable:
            __hash__ = None

        obj: Unhashable = Unhashable()
        bad: Hashable = obj  # E: incompatible_assignment

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

    @assert_passes()
    def test_subclass_write_shadows_inherited_classvar(self):
        from typing import ClassVar

        class C:
            x: ClassVar[int] = 1

        class D(C):
            pass

        D.x = 5

        d = D()
        d.x = 4

    @assert_passes()
    def test_union_attribute_assignment_type(self):
        class C:
            x: int

        class D:
            x: int

        def f(obj: C | D) -> None:
            obj.x = "x"  # E: incompatible_assignment

    @assert_passes()
    def test_union_classvar_attribute_assignment_on_instance(self):
        from typing import ClassVar

        class C:
            x: ClassVar[int] = 1

        class D:
            x: ClassVar[int] = 2

        def f(obj: C | D) -> None:
            obj.x = 3  # E: incompatible_assignment

    @assert_passes()
    def test_union_final_attribute_assignment_on_instance(self):
        from typing import Final

        class C:
            x: Final[int] = 1

        class D:
            x: Final[int] = 2

        def f(obj: C | D) -> None:
            obj.x = 3  # E: incompatible_assignment

    @assert_passes()
    def test_union_classvar_attribute_assignment_on_class_object(self):
        from typing import ClassVar

        class C:
            x: ClassVar[int] = 1

        class D:
            x: ClassVar[int] = 2

        def f(cls: type[C] | type[D]) -> None:
            cls.x = "x"  # E: incompatible_assignment

    @assert_passes()
    def test_union_instance_member_assignment_through_class(self):
        class C:
            x: int

        class D:
            x: int

        def f(cls: type[C] | type[D]) -> None:
            cls.x = 1  # E: incompatible_assignment

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
    def test_readonly_attribute_intersection_allows_write_if_one_side_is_writable(self):
        from typing_extensions import ReadOnly

        from pycroscope.extensions import Intersection

        class Readonly:
            x: ReadOnly[int]

        class Mutable:
            x: int

        def mutate(p: Intersection[Readonly, Mutable]) -> None:
            p.x = 2
            # We disallow deleting attributes
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

    @assert_passes(run_in_both_module_modes=True)
    def test_exact_tuple_subclass_operations(self):
        from typing_extensions import assert_type

        class NotANT(tuple[int, str]):
            pass

        def capybara(x: NotANT) -> None:
            assert_type(x[0], int)
            assert_type(x[1], str)
            assert_type(x, tuple[int, str])  # E: inference_failure

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
    def test_len_narrowing_on_tuple_union(self):
        from typing import TypeAlias

        from typing_extensions import Unpack, assert_type

        FuncInput: TypeAlias = (
            tuple[int] | tuple[str, str] | tuple[int, Unpack[tuple[str, ...]], int]
        )

        def capybara(val: FuncInput) -> None:
            if len(val) == 3:
                assert_type(val, tuple[int, str, int])

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_string_annotation_accessed_through_cls(self):
        from typing_extensions import assert_type

        class Base:
            x: "int"

        class Child(Base):
            @classmethod
            def f(cls) -> None:
                assert_type(cls.x, int)

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_generic_annotation_accessed_through_cls(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Base(Generic[T]):
            x: T

        class Child(Base[int]):
            @classmethod
            def f(cls) -> None:
                assert_type(cls.x, int)

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
        from collections.abc import Callable
        from typing import Any

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func.__is_type_evaluation__ = True
            return func

        @decorator
        def f() -> int:
            return 1

        assert f() == 1

    @assert_passes()
    def test_unknown_attribute_assignment_on_protocol_typed_callable(self):
        from typing import Callable, ParamSpec, Protocol, TypeVar, cast

        P = ParamSpec("P")
        R = TypeVar("R", covariant=True)

        class CallableWithAttr(Protocol[P, R]):
            other_attribute: int

            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

        def decorator(func: Callable[P, R]) -> CallableWithAttr[P, R]:
            converted = cast(CallableWithAttr[P, R], func)
            converted.other_attribute = 1
            converted.missing = 2  # E: undefined_attribute
            return converted

        @decorator
        def wrapped(x: int) -> str:
            return str(x)

    @assert_passes()
    def test_protocol_with_function_metadata_members_accepts_function(self):
        from typing import Any, Protocol

        from typing_extensions import assert_type

        class FunctionLike(Protocol):
            __name__: str
            __module__: str
            __qualname__: str
            __annotations__: dict[str, Any]

            def __call__(self) -> None: ...

        def f() -> None:
            pass

        wrapped: FunctionLike = f
        assert_type(f.__name__, str)
        assert_type(f.__module__, str)
        assert_type(f.__qualname__, str)
        assert_type(wrapped.__name__, str)


class TestNameCheckVisitor(TestNameCheckVisitorBase):
    @assert_passes(run_in_both_module_modes=True)
    def test_undefined_class_decorator_does_not_internal_error(self):
        def run() -> None:
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
            assert_type(X.from_config(), X)
            assert_type(Y.from_config(), Y)
            assert_type(X().ret(), X)
            assert_type(Y().ret(), Y)

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
            assert_type(Local.static_method(), int)
            assert_type(Local.plus_one(1), int)
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

            assert_type(Child.base_method(), int)
            return Child

        assert_type(outer().base_method(), int)

    @assert_passes()
    def test_synthetic_instance_inherits_generic_base_method(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        def outer() -> None:
            class Base(Generic[T]):
                def identity(self, value: T) -> T:
                    return value

            class Child(Base[int]):
                pass

            child = Child()
            assert_type(child.identity(1), int)
            child.identity("x")  # E: incompatible_argument

        outer()

    @assert_passes()
    def test_synthetic_class_inherits_runtime_base_attributes(self):
        class RuntimeBase:
            @staticmethod
            def runtime_method() -> str:
                return ""

        def outer():
            class Child(RuntimeBase):
                pass

            assert_type(Child.runtime_method(), str)
            return Child

        assert_type(outer().runtime_method(), str)

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

            class TD_A(TD_A1, TD_A2): ...  # E: invalid_base

            class TD_B1(TypedDict):
                x: ReadOnly[NotRequired[int]]
                y: ReadOnly[Required[int]]

            class TD_B2(TypedDict):
                x: ReadOnly[Required[int]]
                y: ReadOnly[NotRequired[int]]

            class TD_B(TD_B1, TD_B2): ...  # E: invalid_base

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

    def test_star_import_snapshots_module_dict(self):
        self.assert_passes("""
            import sys
            import types
            from typing import Final, Literal

            module = types.ModuleType("test_star_import_snapshots_module_dict")

            class MutatingAnnotations(dict):
                def __getitem__(self, key):
                    module.extra = 3
                    return super().__getitem__(key)

            module.exported = 1
            module.__annotations__ = MutatingAnnotations({"exported": Final[int]})
            sys.modules[module.__name__] = module

            from test_star_import_snapshots_module_dict import *

            assert_type(exported, Literal[1])
            """)

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
        from typing_extensions import Literal

        if False:
            goes_in_set = []
        else:
            goes_in_set = "capybara"
        if False:
            # The assignment actually executed at runtime wins
            assert_type(goes_in_set, Literal["capybara"])
            print({goes_in_set})

    @assert_passes()
    def test_multiple_assignment_function(self):
        def fn(cond):
            if cond:
                goes_in_set = []
            else:
                goes_in_set = "capybara"
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
                assert_type(self, Capybara)
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
                assert_type(cls, type[OldStyle])

            def __new__(cls):
                assert_type(cls, type[OldStyle])

            @classmethod
            def capybara(cls):
                assert_type(cls, type[OldStyle])

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
        from typing_extensions import Literal

        def capybara(x):
            y = 3 if x else 4
            assert_type(y, Literal[3, 4])

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
            assert_type(answer, PropertyObject)

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
        from typing_extensions import Literal

        x = 3

        def capybara():
            global x
            assert_type(x, Literal[3])


class TestSubclassValue(TestNameCheckVisitorBase):
    @assert_passes()
    def test_annotations_in_arguments(self):
        from typing import Type

        TI = Type[int]

        def capybara(x: TI, y: str):
            assert_is_value(x, SubclassValue(TypedValue(int)))
            assert_type(y, str)

    @assert_passes()
    def test_type_any(self):
        from typing import Any, Type

        def f(x) -> Type[Any]:
            return type(x)

        def capybara():
            f(1)
            assert_type(f(1), type)

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
            assert_type(enum["x"], bytes)

    @assert_passes()
    def test_metaclass_data_attribute(self):
        from typing import Literal

        class Meta(type):
            answer = 1

        class C(metaclass=Meta):
            pass

        assert_type(C.answer, Literal[1])

    @assert_passes()
    def test_metaclass_property_overrides_class_attribute(self):
        from typing import Literal

        class Meta(type):
            @property
            def value(self) -> int:
                return 1

            @value.setter
            def value(self, new_value: int) -> None:
                pass

        class C(metaclass=Meta):
            value = "class value"

        def capybara() -> None:
            assert_type(C.value, Literal[1])
            C.value = 3

    @assert_passes()
    def test_metaclass_call(self):
        from typing import Type

        class Meta(type):
            def __call__(self, *args: object, **kwargs: object) -> bytes:
                return b"hi"

        class C(metaclass=Meta):
            pass

        def capybara(cls: Type[C]) -> None:
            assert_type(cls("x"), bytes)

    @assert_passes()
    def test_invalid_metaclass(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class GenericMeta(type, Generic[T]):
            pass

        class GenericMetaInstance(metaclass=GenericMeta[T]):  # E: invalid_metaclass
            pass

        class Why(
            metaclass=print,  # E: invalid_metaclass
            flush="complicated" == "to make sure error is on right line",
        ):
            pass

    @assert_passes()
    def test_type_form_union_with_metaclass_type(self):
        from typing import TypeVar

        from typing_extensions import Self, assert_type

        T = TypeVar("T")

        class Meta2(type):
            def __call__(self, *args, **kwargs) -> "int | Meta2":
                return 1

        class Class2(metaclass=Meta2):
            def __new__(cls, x: int) -> Self:
                return super().__new__(cls)

        class Meta3(type):
            def __call__(self: type[T], *args, **kwargs) -> T:
                return super().__call__(*args, **kwargs)

        class Class3(metaclass=Meta3):
            def __new__(cls, x: int) -> Self:
                return super().__new__(cls)

        def capybara() -> None:
            assert_type(Class2(), int | Meta2)
            assert_type(Class3(1), Class3)

    @assert_passes(run_in_both_module_modes=True)
    def test_constructor_metaclass_passthrough_call_uses_constructor_signature(self):
        from typing import TypeVar

        from typing_extensions import Self

        T = TypeVar("T")

        class Meta(type):
            def __call__(self: type[T], *args: object, **kwargs: object) -> T:
                return super().__call__(*args, **kwargs)

        class C(metaclass=Meta):
            def __new__(cls, x: int) -> Self:
                return super().__new__(cls)

        def capybara() -> None:
            C()  # E: incompatible_call
            C(1)

    @assert_passes(run_in_both_module_modes=True)
    def test_constructor_custom_metaclass_call_still_overrides(self):
        from typing_extensions import Self, assert_type

        class Meta(type):
            def __call__(self, *args: object, **kwargs: object) -> int:
                return 1

        class C(metaclass=Meta):
            def __new__(cls, x: int) -> Self:
                return super().__new__(cls)

        def capybara() -> None:
            assert_type(C(), int)

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

    @assert_passes(run_in_both_module_modes=True)
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

    @assert_passes(allow_import_failures=True)
    def test_uninitialized_classvar_is_available_on_class_object(self):
        from typing import Any, ClassVar

        from typing_extensions import assert_type

        class C:
            x: ClassVar

        assert_type(C.x, Any)

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

    @assert_passes(run_in_both_module_modes=True)
    def test_constructor_callable_ignores_init_when_new_returns_proxy(self):
        from typing import Callable, ParamSpec, TypeVar

        from typing_extensions import assert_type

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

        def run() -> None:
            r = accepts_callable(C)
            assert_type(r(), Proxy)
            r(1)  # E: incompatible_call

    @assert_passes(allow_import_failures=True)
    def test_constructor_callable_widens_overloaded_literal_returns(self):
        from typing import Callable, Generic, ParamSpec, TypeVar, overload

        from typing_extensions import assert_type

        P = ParamSpec("P")
        R = TypeVar("R")
        T = TypeVar("T")

        def accepts_callable(cb: Callable[P, R]) -> Callable[P, R]:
            return cb

        class Crash:
            def __init__(self, value: int) -> None:
                pass

        accepts_callable(Crash)()  # E: incompatible_call

        class Box(Generic[T]):
            @overload
            def __init__(self: "Box[int]", value: int) -> None: ...

            @overload
            def __init__(self: "Box[str]", value: str) -> None: ...

            def __init__(self, value: int | str) -> None:
                pass

        ctor = accepts_callable(Box)
        assert_type(ctor(0), Box[int])
        assert_type(ctor(""), Box[str])

    @assert_passes(run_in_both_module_modes=True)
    def test_constructor_callable_ignores_init_when_new_returns_any(self):
        from typing import Any, Callable, ParamSpec, TypeVar

        from typing_extensions import assert_type

        P = ParamSpec("P")
        R = TypeVar("R")

        def accepts_callable(cb: Callable[P, R]) -> Callable[P, R]:
            return cb

        class C:
            def __new__(cls) -> Any:
                return super().__new__(cls)

            def __init__(self, x: int) -> None:
                pass

        def run() -> None:
            r = accepts_callable(C)
            assert_type(r(), Any)
            r(1)  # E: incompatible_call

    @assert_passes()
    def test_constructor_ignores_init_when_new_returns_any(self):
        from typing import Any

        class C:
            def __new__(cls) -> Any:
                return super().__new__(cls)

            def __init__(self, x: int) -> None:
                pass

        def capybara() -> None:
            C()
            C(1)  # E: incompatible_call

    @assert_passes()
    def test_constructor_ignores_init_when_new_returns_noreturn(self):
        from typing_extensions import Never, assert_type

        class C:
            def __new__(cls) -> Never:
                raise NotImplementedError

            def __init__(self, x: int) -> None:
                pass

        def capybara() -> None:
            assert_type(C(), Never)
            C(1)  # E: incompatible_call

    @assert_passes()
    def test_constructor_ignores_init_when_new_may_return_non_instance(self):
        from typing import Any

        class C:
            def __new__(cls) -> "C | Any":
                return 0

            def __init__(self, x: int) -> None:
                pass

        def capybara() -> None:
            C()
            C(1)  # E: incompatible_call

    @assert_passes()
    def test_constructor_subscript_respects_new_cls_annotation(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class C(Generic[T]):
            def __new__(cls: "type[C[int]]") -> "C[int]":
                return super().__new__(cls)

        C()
        C[int]()
        C[str]()  # E: incompatible_call

    @assert_passes()
    def test_constructor_subscript_preserves_explicit_new_return(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Box(Generic[T]):
            def __new__(cls, *args, **kwargs) -> "Box[list[T]]":
                return super().__new__(cls)

        assert_type(Box[int](), Box[list[int]])
        assert_type(Box[str](), Box[list[str]])

    @assert_passes(allow_import_failures=True)
    def test_unimportable_constructor_subscript_preserves_explicit_new_return(self):
        import does_not_exist  # noqa: F401
        from typing_extensions import Generic, Self, TypeVar

        from pycroscope.test_name_check_visitor import BOX_FLOAT_OR_INT_IN_TEST_INPUT
        from pycroscope.value import assert_is_value

        T = TypeVar("T")

        class Box(Generic[T]):
            def __new__(cls, value: T) -> Self:
                return super().__new__(cls)

        assert_is_value(Box[float](1), BOX_FLOAT_OR_INT_IN_TEST_INPUT)

    @assert_passes(allow_import_failures=True)
    def test_overloaded_constructor_prefers_synthetic_signature(self):
        from typing import Any, Generic, TypeVar, overload

        import does_not_exist  # noqa: F401
        from typing_extensions import assert_type

        T = TypeVar("T")

        class Box(Generic[T]):
            @overload
            def __init__(self: "Box[list[int]]", value: int) -> None: ...

            @overload
            def __init__(self: "Box[set[str]]", value: str) -> None: ...

            @overload
            def __init__(self, value: T) -> None: ...

            def __init__(self, value: Any) -> None:
                pass

        assert_type(Box(0), Box[list[int]])
        assert_type(Box(""), Box[set[str]])
        assert_type(Box(3.0), Box[float])

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
                assert_type(self.obj, str)

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

    @assert_passes()
    def test_attribute_checker_respects_isinstance_narrowing_for_methods(self):
        class Value:
            pass

        class AnnotationExpr:
            def unqualify(self, values: set[object]) -> tuple[int, set[object]]:
                return 1, values

        def f(x: AnnotationExpr | Value) -> int:
            if isinstance(x, AnnotationExpr):
                inner, _ = x.unqualify(set())
                return inner
            return 0


class TestBadRaise(TestNameCheckVisitorBase):
    @assert_passes()
    def test_raise(self):
        def bad_value():
            raise 42  # E: bad_exception

        def bad_type():
            # make sure this isn't inferenced to KnownValue, so this tests what it's supposed to
            # test
            assert_type(int("3"), int)
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

        from typing_extensions import Literal

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
            assert_type(another_uid, Literal["hello"])

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

    @assert_passes(run_in_both_module_modes=True)
    def test_static_fallback_behavior(self):
        from typing import Any, Hashable, Literal, NewType, TypedDict, TypeVar

        from typing_extensions import assert_type

        UserId = NewType("UserId", int)

        u2: UserId = UserId(42)
        assert_type(UserId(5) + 1, int)

        def check_static_errors() -> None:
            UserId("user")  # E: incompatible_argument
            u1: UserId = 42  # E: incompatible_assignment
            print(u1)
            _: type = UserId  # E: incompatible_assignment
            isinstance(u2, UserId)  # E: incompatible_argument

            class UserIdDerived(UserId):  # E: invalid_base
                pass

        GoodName = NewType("BadName", int)  # E: incompatible_call

        GoodNewType1 = NewType("GoodNewType1", list)
        GoodNewType2 = NewType("GoodNewType2", GoodNewType1)
        TypeAlias1 = dict[str, str]
        GoodNewType3 = NewType("GoodNewType3", TypeAlias1)

        def check_newtype_operations() -> None:
            _nt1: GoodNewType1[int]  # E: unsupported_operation
            NewType("BadNewType6", int, int)  # E: incompatible_call

        BadNewType1 = NewType("BadNewType1", int | str)  # E: incompatible_call
        T = TypeVar("T")
        BadNewType2 = NewType("BadNewType2", list[T])  # E: incompatible_call
        BadNewType3 = NewType("BadNewType3", Hashable)  # E: incompatible_call
        BadNewType4 = NewType("BadNewType4", Literal[7])  # E: incompatible_call

        class TD1(TypedDict):
            a: int

        BadNewType5 = NewType("BadNewType5", TD1)  # E: incompatible_call
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

    @assert_passes()
    def test_assignment_target_name_mismatch_with_keywords(self):
        from typing_extensions import ParamSpec

        BadParamSpec = ParamSpec(name="WrongParamSpec")  # E: incompatible_call
        print(BadParamSpec)

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

            def define_bad() -> None:
                class Bad(Generic[Shape]):  # E: invalid_base
                    ...

            def define_bad2() -> None:
                class Bad2(Generic[*Ts1, *Ts2]):  # E: invalid_base
                    ...
            """,
            run_in_both_module_modes=True,
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
            assert_type(inspect.signature(capybara), inspect.Signature)
            assert_type(_inspect.signature(capybara), inspect.Signature)

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
                assert_type(k, int)
                assert_type(v, str)

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
        from typing_extensions import Literal

        def capybara(cond):
            lst = []
            if cond:
                lst.append("x")
            assert_is_value(lst, KnownValue(["x"]) | KnownValue([]))
            for c in lst:
                assert_type(c, Literal["x"])

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
                assert_type(x, str)

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
        from typing_extensions import Literal

        def capybara():
            x = 3

            def inner_capybara():
                nonlocal x
                assert_is_value(
                    x, MultiValuedValue([KnownValue(4), KnownValue(3), KnownValue(5)])
                )
                x = 4
                assert_type(x, Literal[4])

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
    def test_bad_dict_unpack(self):
        def capybara() -> None:
            {**3}  # E: unsupported_operation

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

        from typing_extensions import Literal

        def run(lst: List[int], union: Union[Any, List[int], Tuple[str, float]]):
            a, b = 1, 2
            assert_type(a, Literal[1])
            assert_type(b, Literal[2])

            c, d = lst
            assert_type(c, int)
            assert_type(d, int)

            e, f = (lst, 42)
            assert_type(e, list[int])
            assert_type(f, Literal[42])

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
            assert_type(j, Literal[5])
            assert_type(k, Literal[6])
            l, m, *n, o, p = long_tuple
            assert_type(l, Literal[1])
            assert_type(m, Literal[2])
            assert_is_value(
                n, make_simple_sequence(list, [KnownValue(3), KnownValue(4)])
            )
            assert_type(o, Literal[5])
            assert_type(p, Literal[6])

            q, r, *s = (1, 2)
            assert_type(q, Literal[1])
            assert_type(r, Literal[2])
            assert_is_value(s, SequenceValue(list, []))

            for sprime in []:
                assert_is_value(sprime, NO_RETURN_VALUE)

            for t, u in []:
                assert_is_value(t, AnyValue(AnySource.unreachable))
                assert_is_value(u, AnyValue(AnySource.unreachable))

            known_list = [1, 2]
            v, w = known_list
            assert_type(v, Literal[1])
            assert_type(w, Literal[2])

            if lst:
                known_list.append(3)

            # We allow this unsafe code to avoid false positives
            x, y = known_list
            assert_type(x, Literal[1, 2, 3])

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
        from typing_extensions import Literal

        def capybara(x: int):
            v = 1
            while x < 2:
                while True:
                    if x == 0:
                        assert_type(v, Literal[1, 2])
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

    @assert_passes()
    def test_invalid_format_spec_falls_back_to_str(self):
        from typing_extensions import assert_type

        def capybara() -> None:
            assert_type(f"{1:zz}", str)

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
            assert_type(y, str)

    @assert_passes()
    def test_formatted_value_conversions_and_dynamic_spec(self):
        from typing_extensions import Literal

        def with_conversions() -> None:
            assert_type(f"{1!r}", str)
            assert_type(f"{'capybara'!s}", Literal["capybara"])
            assert_type(f"{'capybara'!a}", Literal["'capybara'"])

        def with_dynamic_spec(spec: str) -> None:
            assert_type(f"{1:{spec}}", str)

    @assert_passes()
    def test_docstring_with_braces_does_not_require_f_string(self):
        x = 3

        def capybara() -> None:
            """x = {x}"""

    @assert_passes()
    def test_unknown_name_in_braces_does_not_require_f_string(self):
        def capybara() -> str:
            return "x = {missing}"

    @assert_passes()
    def test_conversion_failure_falls_back_to_str(self):
        class BadStr:
            def __str__(self) -> str:
                raise RuntimeError("capybara")

        bad_str = BadStr()

        def capybara() -> None:
            assert_type(f"{bad_str!s}", str)

    @assert_passes()
    def test_undefined_name(self):
        def capybara():
            return f"{x}"  # E: undefined_name


class TestRuntimeTypeExpressions(TestNameCheckVisitorBase):
    @skip_before((3, 11))
    def test_unpack_runtime_type_expression(self):
        self.assert_passes("""
            import typing

            from typing import TypeVarTuple, Unpack

            Ts = TypeVarTuple("Ts")

            def capybara() -> None:
                tuple[*Ts]
                tuple[Unpack[Ts]]
                tuple[typing.Unpack[Ts]]
            """)


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
        from typing_extensions import Final, Literal

        def capybara() -> None:
            x: Final = 3
            assert_type(x, Literal[3])
            y: int = 3
            assert_type(y, int)
            z: bytes
            print(z)  # E: undefined_name

            y: bytes = b"ytes"  # E: already_declared
            assert_type(y, bytes)

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

    @assert_passes()
    def test_final_instance_attributes_with_nonstandard_receiver_name(self):
        from typing_extensions import Final

        class Capybara:
            def __init__(this) -> None:  # E: method_first_arg
                this.initialized_in_init: Final[int]

    @assert_passes()
    def test_attribute_assignment_with_nonstandard_receiver_name(self):
        class Capybara:
            x: int

            def method(this, value: object) -> None:  # E: method_first_arg
                this.x = value  # E: incompatible_assignment

    @skip_before((3, 11))  # @final doesn't set __final__ in 3.10
    @assert_passes(run_in_both_module_modes=True)
    def test_final_decorator_in_unimportable_module(self):
        from typing import final

        @final
        class FinalBase:
            pass

        class FinalChild(FinalBase):  # E: invalid_base
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
    def test_explicit_protocol_abstract_instantiation_in_unimportable_module(self):
        from abc import ABC, abstractmethod
        from typing import ClassVar, Protocol

        class PColor(Protocol):
            @abstractmethod
            def draw(self) -> str: ...

        class BadColor(PColor):
            def draw(self) -> str:
                return super().draw()  # E: bad_super_call

        class RGB(Protocol):
            rgb: tuple[int, int, int]

            @abstractmethod
            def intensity(self) -> int:
                return 1

            def transparency(self) -> int: ...

        class Point(RGB):
            def __init__(self, blue: str) -> None:
                self.rgb = 0, 0, blue  # E: incompatible_assignment

        Point("")  # E: incompatible_call

        class Proto1(Protocol):
            cm1: ClassVar[int]

        class Concrete1(Proto1):
            pass

        Concrete1()  # E: incompatible_call

        class Proto5(Protocol):
            def method1(self) -> int: ...

        class Concrete5(Proto5):
            pass

        Concrete5()  # E: incompatible_call

        class Proto7(Protocol):
            @abstractmethod
            def method1(self) -> None: ...

        class Mixin7(Proto7, ABC):
            def method1(self) -> None:
                pass

        class Concrete7A(Proto7):
            pass

        class Concrete7B(Mixin7, Proto7):
            pass

        Concrete7A()  # E: incompatible_call
        Concrete7B()

    @assert_passes()
    def test_protocol_base_member_assignment_type(self):
        from typing import Protocol

        class RGB(Protocol):
            rgb: tuple[int, int, int]

        class Point(RGB):
            def __init__(self, blue: str) -> None:
                self.rgb = 0, 0, blue  # E: incompatible_assignment

    @assert_passes()
    def test_protocol_assignment_to_declared_self_attribute(self):
        from typing import Protocol

        class Proto(Protocol):
            value: int

            def __init__(self) -> None:
                self.value = 3

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_class_object_method_and_property_shapes(self):
        from typing import Any, Protocol

        class ProtoA1(Protocol):
            def method1(self, x: int) -> int: ...

        class ProtoA2(Protocol):
            def method1(_self, self: Any, x: int) -> int: ...  # E: method_first_arg

        class ConcreteA:
            def method1(self, x: int) -> int:
                return 0

        pa1: ProtoA1 = ConcreteA  # E: incompatible_assignment
        pa2: ProtoA2 = ConcreteA

        class ProtoB1(Protocol):
            @property
            def prop1(self) -> int: ...

        class ConcreteB:
            @property
            def prop1(self) -> int:
                return 0

        pb1: ProtoB1 = ConcreteB  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_class_object_classvar_members(self):
        from typing import ClassVar, Protocol

        class ProtoC1(Protocol):
            attr1: ClassVar[int]

        class ProtoC2(Protocol):
            attr1: int

        class ConcreteC1:
            attr1: ClassVar[int] = 1

        class ConcreteC2:
            attr1: int = 1

        class CMeta(type):
            pass

        class ConcreteC3(metaclass=CMeta):
            pass

        ConcreteC3.attr1 = 1

        pc1: ProtoC1 = ConcreteC1  # E: incompatible_assignment
        pc2: ProtoC2 = ConcreteC1
        pc3: ProtoC1 = ConcreteC2  # E: incompatible_assignment
        pc4: ProtoC2 = ConcreteC2  # E: incompatible_assignment
        pc5: ProtoC1 = ConcreteC3  # E: incompatible_assignment
        pc6: ProtoC2 = ConcreteC3  # E: incompatible_assignment

    @assert_passes()
    def test_protocol_class_object_call_member(self):
        from typing import Protocol

        class Concrete:
            def __init__(self, x: int) -> None:
                self.x = x

        class Factory(Protocol):
            def __call__(self, x: int) -> Concrete: ...

        factory: Factory = Concrete
        created: Concrete = factory(1)

    @assert_passes()
    def test_callable_assignment_from_protocol_instance(self):
        from collections.abc import Callable
        from typing import Protocol

        class GoodCallable(Protocol):
            def __call__(self, x: int) -> str: ...

        class BadCallable(Protocol):
            @property
            def __call__(self) -> int: ...

        def wants_callable(fn: Callable[[int], str]) -> None:
            pass

        def capybara(good: GoodCallable, bad: BadCallable) -> None:
            wants_callable(good)
            wants_callable(bad)  # E: incompatible_argument

    @assert_passes()
    def test_collections_callable_assignment_from_protocol_instance(self):
        from collections.abc import Callable
        from typing import Protocol

        class GoodCallable(Protocol):
            def __call__(self, x: int) -> str: ...

        class NarrowCallable(Protocol):
            def __call__(self, x: int, y: int) -> str: ...

        def capybara(good: GoodCallable, narrow: NarrowCallable) -> None:
            generic_cb: Callable = good
            specific_cb: Callable[[int], str] = good
            bad_cb: Callable[[int], str] = narrow  # E: incompatible_assignment
            print(generic_cb, specific_cb, bad_cb)

    @assert_passes()
    def test_protocol_class_object_metaclass_members(self):
        from typing import Protocol

        class Meta(type):
            answer: int = 1

        class WantsAnswer(Protocol):
            answer: int

        class Concrete(metaclass=Meta):
            pass

        good: WantsAnswer = Concrete
        print(good)

    @assert_passes()
    def test_hashable_class_object_with_custom_metaclass_hash(self):
        from collections.abc import Hashable

        class Meta(type):
            def __hash__(self, extra: int = 0) -> int:
                return extra

        class Concrete(metaclass=Meta):
            pass

        def capybara() -> None:
            hashable: Hashable = Concrete
            print(hashable)

    @assert_passes()
    def test_type_protocol_constructor_call_allows_concrete_implementers(self):
        from typing import Protocol, cast

        class Proto(Protocol):
            def meth(self) -> int: ...

        class Concrete:
            def meth(self) -> int:
                return 1

        def call_it(cls: type[Proto]) -> int:
            return cls().meth()

        call_it(cast(type[Proto], Concrete))
        impl = cast(type[Proto], Concrete)
        impl().meth()

    @assert_passes(run_in_both_module_modes=True)
    def test_self_methods_in_unimportable_generic_module(self):
        from typing import Generic, TypeVar

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

    @assert_passes(run_in_both_module_modes=True)
    def test_self_classmethod_in_unimportable_module(self):
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

    @assert_passes(run_in_both_module_modes=True)
    def test_self_advanced_in_unimportable_module(self):
        from typing_extensions import Self, assert_type

        class ParentA:
            @property
            def prop1(self) -> Self:
                raise NotImplementedError

        class ChildA(ParentA):
            pass

        def f() -> None:
            assert_type(ParentA().prop1, ParentA)
            assert_type(ChildA().prop1, ChildA)

        class ParentB:
            a: list[Self]

            @classmethod
            def method1(cls) -> Self:
                raise NotImplementedError

        class ChildB(ParentB):
            b: int = 0

            def method2(self) -> None:
                assert_type(self, Self)
                assert_type(self.a, list[Self])
                assert_type(self.a[0], Self)
                assert_type(self.method1(), Self)

            @classmethod
            def method3(cls) -> None:
                assert_type(cls, type[Self])
                assert_type(cls.a, list[Self])
                assert_type(cls.a[0], Self)
                assert_type(cls.method1(), Self)

    @assert_passes(run_in_both_module_modes=True)
    def test_self_property_in_method_body_in_unimportable_module(self):
        from typing_extensions import Self, assert_type

        class Base:
            @property
            def prop(self) -> Self:
                raise NotImplementedError

            def method(self) -> None:
                assert_type(self.prop, Self)

        class Child(Base):
            def method(self) -> None:
                assert_type(self.prop, Self)

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_instance_only_member_substitutes_generic_base_args(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Base(Generic[T]):
            value: T

        class Child(Base[int]):
            @classmethod
            def class_method(cls) -> None:
                assert_type(cls.value, int)

            def method(self) -> None:
                assert_type(self.value, int)

    @assert_passes(run_in_both_module_modes=True)
    def test_typevar_classmethod_in_unimportable_module(self):
        from typing import TypeVar

        T = TypeVar("T")

        class Box:
            @classmethod
            def identity(cls, value: T) -> T:
                return value

        def capybara():
            Box.identity(1).bit_length()
            Box.identity("x").upper()

    @assert_passes(run_in_both_module_modes=True)
    def test_generic_alias_constructor_in_unimportable_module(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

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
        assert_type(Node[int](0).label, int)

        Node[int](0)
        Node[int]("")  # E: incompatible_argument
        Node[str]("")
        Node[str](0)  # E: incompatible_argument

        def bad_attribute_access() -> None:
            Node[int].label = 1  # E: incompatible_assignment
            Node[int].label  # E: undefined_attribute
            Node.label = 1  # E: incompatible_assignment
            Node.label  # E: undefined_attribute
            type(n1).label  # E: undefined_attribute

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
    def test_protocol_receiver_assignment_with_nonstandard_receiver_name(self):
        from typing import Protocol

        class Proto(Protocol):
            allowed: int

            def assign(this) -> None:  # E: method_first_arg
                this.disallowed = 1  # E: invalid_annotation
                this.allowed = 1

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_staticmethod_with_receiver_param_is_incompatible(self):
        from typing import Protocol

        class Proto(Protocol):
            def method1(self, a: int, b: int) -> float: ...

        class Good:
            @staticmethod
            def method1(a: int, b: int) -> float:
                return 0

        class Bad:
            @staticmethod
            def method1(self, a: int, b: int) -> float:
                return 0

        ok: Proto = Good()
        bad: Proto = Bad()  # E: incompatible_assignment

    @assert_passes()
    def test_inconsistent_type(self):
        def capybara():
            x: int = 1
            assert_type(x, int)
            x = "x"  # E: incompatible_assignment

            y: int = "y"  # E: incompatible_assignment
            return (x, y)

    @assert_passes()
    def test_parameter_reassignment_uses_parameter_type(self):
        def capybara(x: int) -> None:
            assert_type(x, int)
            x = "x"  # E: incompatible_assignment
            print(x)

    @assert_passes()
    def test_parameter_augmented_reassignment_uses_parameter_type(self):
        from typing import Literal

        def capybara(x: Literal[3, 4, 5]) -> None:
            assert_type(x, Literal[3, 4, 5])
            x += 3  # E: incompatible_assignment
            print(x)

    @assert_passes()
    def test_class_scope(self):
        class Capybara:
            x: int = 0
            assert_type(x, int)

            def __init__(self) -> None:
                self.y: object = 3

            def method(self):
                assert_type(self.y, object)

    @assert_passes()
    def test_loop(self):
        def capybara():
            for i in range(3):
                j: int = i  # E: unused_variable

            j: int = 0  # E: already_declared  # E: unused_variable

    @assert_passes()
    def test_module_scope(self):
        x: int = 3
        assert_type(x, int)

        if __name__ == "__main__":
            y: int = 3
            assert_type(y, int)


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


class TestControlFlow(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assert_message_uses_inverted_constraint(self):
        from typing import Optional

        from typing_extensions import assert_type

        def describe_missing(x: None) -> str:
            return "missing"

        def capybara(x: Optional[int]) -> None:
            assert x is not None, describe_missing(x)
            assert_type(x, int)

    @assert_passes()
    def test_assert_false(self):
        def capybara() -> None:
            assert False

    @assert_passes()
    def test_condition_on_attribute_of_call_result(self):
        class Box:
            def __init__(self, value: bool) -> None:
                self.value = value

        def make_box(value: bool) -> Box:
            return Box(value)

        def capybara() -> None:
            if make_box(True).value:
                pass

    def test_annotate_explicit_type_alias_assignment(self):
        self.assert_passes(
            """
            from typing import TypeAlias
            from typing_extensions import assert_type

            Alias: TypeAlias = tuple[int, str]

            def capybara(x: Alias) -> None:
                a, b = x
                assert_type(a, int)
                assert_type(b, str)
            """,
            annotate=True,
        )

    def test_annotate_skipped_constant_branches(self):
        self.assert_passes(
            """
            from typing_extensions import Literal, assert_type

            def capybara() -> None:
                if True:  # E: value_always_true
                    x = 1
                else:
                    x = 2
                assert_type(x, Literal[1, 2])

                if False:
                    y = 3
                else:
                    y = 4
                assert_type(y, Literal[3, 4])

                z = 5 if True else 6  # E: value_always_true
                assert_type(z, Literal[5, 6])

                w = 7 if False else 8
                assert_type(w, Literal[7, 8])
            """,
            annotate=True,
        )

    @assert_passes()
    def test_if_true_keeps_merged_assignment_type(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            if True:  # E: value_always_true
                x = 1
            else:
                x = "x"
            assert_type(x, Literal[1, "x"])

    @assert_passes()
    def test_if_false_keeps_merged_assignment_type(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            if False:
                x = 1
            else:
                x = "x"
            assert_type(x, Literal[1, "x"])

    @assert_passes()
    def test_ifexp_true_keeps_merged_assignment_type(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            x = 1 if True else "x"  # E: value_always_true
            assert_type(x, Literal[1, "x"])

    @assert_passes()
    def test_ifexp_false_keeps_merged_assignment_type(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            x = 1 if False else "x"
            assert_type(x, Literal[1, "x"])

    @assert_passes(run_in_both_module_modes=True)
    def test_annotated_class_attribute_read_uses_annotation_type(self):
        from typing_extensions import assert_type

        class Visitor:
            should_check_environ_for_files: bool = True

            @classmethod
            def capybara(cls) -> None:
                assert_type(cls.should_check_environ_for_files, bool)
                if cls.should_check_environ_for_files:
                    pass

    @assert_passes()
    def test_for_else_with_always_present_iterable_keeps_body_assignment(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for x in (1,):
                y = x
            else:
                assert_type(y, Literal[1])
            assert_type(y, Literal[1])

    @assert_passes()
    def test_for_else_defines_name_for_maybe_empty_iterable(self):
        from typing_extensions import assert_type

        def capybara(xs: list[int]) -> None:
            for x in xs:
                y = x
            else:
                y = 0
            assert_type(y, int)

    @assert_passes()
    def test_while_else_merges_body_and_else_assignments(self):
        from typing_extensions import Literal, assert_type

        def capybara(flag: bool) -> None:
            while flag:
                y = 1
                flag = False
            else:
                y = 2
            assert_type(y, Literal[1, 2])

    @assert_passes()
    def test_except_tuple_binding_type(self):
        from typing_extensions import assert_type

        def capybara(flag: bool) -> None:
            try:
                if flag:
                    raise ValueError()
                raise TypeError()
            except (ValueError, TypeError) as exc:
                assert_type(exc, ValueError | TypeError)

    @assert_passes()
    def test_except_nested_tuple_binding_type(self):
        from typing_extensions import assert_type

        def capybara() -> None:
            try:
                raise TypeError()
            except (KeyError, (ValueError, TypeError)) as exc:
                assert_type(exc, KeyError | ValueError | TypeError)

    @assert_passes()
    def test_except_conditional_handler_expression(self):
        from typing_extensions import assert_type

        def capybara(flag: bool) -> None:
            try:
                raise FileNotFoundError()
            except FileNotFoundError if flag else FileExistsError as exc:
                assert_type(exc, FileNotFoundError | FileExistsError)

    @assert_passes()
    def test_except_baseexception_subclass_binding_type(self):
        from typing_extensions import assert_type

        def capybara() -> None:
            try:
                raise GeneratorExit()
            except GeneratorExit as exc:
                assert_type(exc, GeneratorExit)

    @assert_passes()
    def test_except_rejects_non_exception_handler(self):
        from typing_extensions import assert_type

        class NotExc:
            pass

        def capybara() -> None:
            try:
                raise ValueError()
            except NotExc as exc:  # E: bad_except_handler
                assert_type(exc, BaseException)

    @assert_passes()
    def test_except_dynamic_handler_type_falls_back_to_baseexception(self):
        from typing import Type

        from typing_extensions import assert_type

        def capybara(exc_type: Type[BaseException]) -> None:
            try:
                raise ValueError()
            except exc_type as exc:
                assert_type(exc, BaseException)

    @assert_passes()
    def test_except_unknown_handler_type_falls_back_to_baseexception(self):
        from typing_extensions import assert_type

        def capybara(exc_type: object) -> None:
            try:
                raise ValueError()
            except exc_type as exc:
                assert_type(exc, BaseException)

    @assert_passes()
    def test_try_else_refines_success_path_and_merges_result(self):
        from typing_extensions import Literal, assert_type

        def capybara(flag: bool) -> None:
            try:
                if flag:
                    raise ValueError()
                x = 1
            except ValueError:
                x = 2
            else:
                assert_type(x, Literal[1])
                x = 3
            assert_type(x, Literal[2, 3])

    @assert_passes()
    def test_try_else_does_not_see_except_only_bindings(self):
        def capybara() -> None:
            try:
                pass
            except ValueError:
                x = 1  # E: unused_variable
            else:
                print(x)  # E: undefined_name

    @assert_passes()
    def test_try_finally_keeps_preinitialized_assignment(self):
        from typing_extensions import Literal, assert_type

        def capybara(flag: bool) -> None:
            x = 0
            try:
                if flag:
                    x = 1
                else:
                    raise ValueError()
            except ValueError:
                x = 2
            finally:
                assert_type(x, Literal[0, 1, 2])
            assert_type(x, Literal[1, 2])

    @assert_passes()
    def test_try_finally_can_see_possibly_undefined_name(self):
        def capybara(flag: bool) -> None:
            try:
                if flag:
                    x = 1
            finally:
                print(x)  # E: possibly_undefined_name


class TestForLoops(TestNameCheckVisitorBase):
    @assert_passes()
    def test_tuple_destructuring_target(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for a, b in [(1, "x")]:
                assert_type(a, Literal[1])
                assert_type(b, Literal["x"])

    @assert_passes()
    def test_tuple_destructuring_target_after_loop(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for a, b in [(1, "x")]:
                pass
            assert_type(a, Literal[1])
            assert_type(b, Literal["x"])

    @assert_passes()
    def test_list_destructuring_target(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for [a, b] in [(1, "x")]:
                assert_type(a, Literal[1])
                assert_type(b, Literal["x"])

    @assert_passes()
    def test_nested_destructuring_target(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for a, (b, c) in [(1, ("x", True))]:
                assert_type(a, Literal[1])
                assert_type(b, Literal["x"])
                assert_type(c, Literal[True])

    @assert_passes()
    def test_continue_narrows_remaining_loop_path(self):
        from typing_extensions import Literal, assert_type

        def capybara() -> None:
            for x in (1, "x"):
                if x == 1:
                    continue
                assert_type(x, Literal["x"])


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
                assert_type(e, int)

    @assert_passes()
    def test_union_context_manager_assignment(self) -> None:
        class IntCM:
            def __enter__(self) -> int:
                return 1

            def __exit__(self, typ, value, tb) -> None:
                pass

        class StrCM:
            def __enter__(self) -> str:
                return "x"

            def __exit__(self, typ, value, tb) -> None:
                pass

        def capybara(flag: bool) -> None:
            cm: IntCM | StrCM = IntCM() if flag else StrCM()
            with cm as value:
                assert_type(value, int | str)

    @assert_passes()
    def test_tuple_destructuring_in_with_target(self) -> None:
        class PairCM:
            def __enter__(self) -> tuple[int, str]:
                return (1, "x")

            def __exit__(self, typ, value, tb) -> None:
                pass

        def capybara() -> None:
            with PairCM() as (a, b):
                assert_type(a, int)
                assert_type(b, str)

    @assert_passes()
    def test_union_context_manager_with_invalid_member(self) -> None:
        class GoodCM:
            def __enter__(self) -> int:
                return 1

            def __exit__(self, typ, value, tb) -> None:
                pass

        class BadCM:
            pass

        def capybara(flag: bool) -> None:
            cm: GoodCM | BadCM = GoodCM() if flag else BadCM()
            with cm as value:  # E: invalid_context_manager
                print(value)

    @assert_passes()
    def test_assert_error_block(self) -> None:
        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                f("x")

            with assert_error():  # E: inference_failure
                f(1)

    @assert_passes()
    def test_assert_error_nested_control_flow(self) -> None:
        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara(flag: bool) -> None:
            with assert_error():
                if flag:
                    f("x")
                else:
                    f("y")

    @assert_passes()
    def test_assert_error_nested_with_block(self) -> None:
        import contextlib

        from pycroscope.extensions import assert_error

        def f(x: int) -> None:
            pass

        def capybara() -> None:
            with assert_error():
                with contextlib.nullcontext(None) as value:
                    assert_type(value, None)
                    f("x")

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
                assert_type(e, int)

    @assert_passes()
    def test_union_async_context_manager_assignment(self) -> None:
        class IntCM:
            async def __aenter__(self) -> int:
                return 1

            async def __aexit__(self, typ, value, tb) -> None:
                pass

        class StrCM:
            async def __aenter__(self) -> str:
                return "x"

            async def __aexit__(self, typ, value, tb) -> None:
                pass

        async def capybara(flag: bool) -> None:
            cm: IntCM | StrCM = IntCM() if flag else StrCM()
            async with cm as value:
                assert_type(value, int | str)

    @assert_passes()
    def test_tuple_destructuring_in_async_with_target(self) -> None:
        class PairCM:
            async def __aenter__(self) -> tuple[int, str]:
                return (1, "x")

            async def __aexit__(self, typ, value, tb) -> None:
                pass

        async def capybara() -> None:
            async with PairCM() as (a, b):
                assert_type(a, int)
                assert_type(b, str)

    @assert_passes()
    def test_union_async_context_manager_with_invalid_member(self) -> None:
        class GoodCM:
            async def __aenter__(self) -> int:
                return 1

            async def __aexit__(self, typ, value, tb) -> None:
                pass

        class BadCM:
            pass

        async def capybara(flag: bool) -> None:
            cm: GoodCM | BadCM = GoodCM() if flag else BadCM()
            async with cm as value:  # E: invalid_context_manager
                print(value)


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


class TestFallbackValueDispatch(TestNameCheckVisitorBase):
    @assert_passes()
    def test_enum_ignore_in_conditional_class_body(self):
        from enum import Enum

        def make_enum(flag: bool) -> type[Enum]:
            class E(Enum):
                _value_: int
                _ignore_ = "bad keep" if flag else ["bad", "other"]
                bad = 1
                keep = 1

            return E

        make_enum(True)

    @assert_passes(run_in_both_module_modes=True)
    def test_conditional_slots_value_is_respected(self):
        from random import random

        class Slotted:
            __slots__ = ("x",) if random() else ["x"]
            x: int

            def mutate(self) -> None:
                self.y = 1  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_inherited_slots_are_respected(self):
        class Base:
            __slots__ = ("x",)
            x: int

        class Child(Base):
            __slots__ = ("y",)
            y: int

            def mutate(self) -> None:
                self.x = 1
                self.y = 2
                self.z = 3  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_conditional_typevar_identity_in_generic_bases(self):
        from random import random
        from typing import Generic, TypeVar

        T = TypeVar("T")

        if random():
            maybe_t = T
        else:
            maybe_t = T

        class Bad(Generic[maybe_t, maybe_t]):  # E: invalid_base
            pass


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
    def test_classvar_instance_override_mismatch(self):
        from typing import ClassVar

        class Base:
            x: int
            y: ClassVar[int] = 1

        class Child(Base):
            x: ClassVar[int]  # E: incompatible_override
            y: int  # E: incompatible_override

    @assert_passes()
    def test_unannotated_protocol_classvar_override(self):
        from typing import ClassVar, Protocol

        class Proto(Protocol):
            z: ClassVar[int]

        class ProtoImpl(Proto):
            z = 0

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
                assert_type(x, int)
            assert_type(x, int | None)

            if (y := opt()) is not None:
                assert_type(y, int)
            assert_type(y, int | None)

    @assert_passes()
    def test_and(self):
        from typing import Optional, Set

        def opt() -> Optional[int]:
            return None

        def capybara(cond):
            if (x := opt()) and cond:
                assert_type(x, int)
            assert_type(x, int | None)

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
        from typing_extensions import Literal

        def capybara(cond):
            (x := 2) if cond else (x := 1)
            assert_type(x, Literal[1, 2])

    @assert_passes()
    def test_comprehension_scope(self):
        from typing import List, Optional

        def capybara(elts: List[Optional[int]]) -> None:
            if any((x := i) is not None for i in elts):
                assert_type(x, int | None)
                print(i)  # E: undefined_name


class TestUnion(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union_to_union(self):
        from typing import Optional

        def capybara(x: Optional[str], y: Optional[str]) -> Optional[str]:
            assert_type(x, str | None)
            assert_type(y, str | None)
            return x or y


class TestContextManagerWithSuppression(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sync(self):
        import contextlib
        from collections.abc import Generator
        from types import TracebackType
        from typing import ContextManager, Optional, Type

        from typing_extensions import Literal

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
        def empty_contextlib_manager() -> Generator[None]:
            yield

        def use_suppress_exception():
            a = 2
            with SuppressException():
                a = 3
            assert_type(a, Literal[2, 3])

        def use_suppress_exception_multi_assignment():
            a = 2
            with SuppressException():
                a = 3
                a = 4
            assert_type(a, Literal[2, 3, 4])

        def use_empty_context():
            a = 2  # E: unused_assignment
            with EmptyContext():
                a = 3  # E: unused_assignment
                a = 4
            assert_type(a, Literal[4])

        def use_context_manager():
            a = 2  # E: unused_assignment
            with empty_context_manager():
                a = 3
            assert_type(a, Literal[3])

        def use_builtin_function():
            a = 2  # E: unused_assignment
            with open("test_file.txt"):
                a = 3
            assert_type(a, Literal[3])

        def use_contextlib_manager():
            a = 2  # E: unused_assignment
            with empty_contextlib_manager():
                a = 3
            assert_type(a, Literal[3])

        def use_optional_bool_return(x: int | str) -> None:
            if isinstance(x, int):
                with MaybeSuppressException():
                    raise ValueError
            assert_type(x, str)

        def use_nested_contexts():
            b = 2
            with SuppressException(), EmptyContext() as b:
                assert_type(b, None)
            assert_type(b, Literal[2] | None)

            c = 2  # E: unused_assignment
            with EmptyContext() as c, SuppressException():
                assert_type(c, None)
            assert_type(c, None)

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

        from typing_extensions import Literal

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
            assert_type(a, Literal[2, 3])

        async def use_async_empty_context():
            a = 2  # E: unused_assignment
            async with AsyncEmptyContext():
                a = 3
            assert_type(a, Literal[3])

        async def use_async_context_manager():
            a = 2  # E: unused_assignment
            async with async_empty_context_manager():
                a = 3
            assert_type(a, Literal[3])

        async def use_async_nested_contexts():
            b = 2
            async with AsyncSuppressException(), AsyncEmptyContext() as b:
                assert_type(b, None)
            assert_type(b, Literal[2] | None)

            c = 2  # E: unused_assignment
            async with AsyncEmptyContext() as c, AsyncSuppressException():
                assert_type(c, None)
            assert_type(c, None)

    def test_async_contextlib_manager(self):
        import contextlib
        from typing import AsyncIterator

        from typing_extensions import Literal

        @contextlib.asynccontextmanager
        async def async_empty_contextlib_manager() -> AsyncIterator[None]:
            yield

        async def use_async_contextlib_manager():
            a = 2  # E: unused_assignment
            async with async_empty_contextlib_manager():
                a = 3
            assert_type(a, Literal[3])


class TestTryStar(TestNameCheckVisitorBase):
    @skip_before((3, 11))
    def test_exception_group_binding_type(self):
        self.assert_passes("""
            def capybara() -> None:
                try:
                    raise ExceptionGroup("boom", [ValueError()])
                except* ValueError as eg:
                    assert_type(eg, ExceptionGroup[ValueError])
                print(eg)
            """)

    @skip_before((3, 11))
    def test_exception_group_tuple_handler_type(self):
        self.assert_passes("""
            def capybara() -> None:
                try:
                    raise ExceptionGroup("boom", [ValueError(), TypeError()])
                except* (ValueError, TypeError) as eg:
                    assert_type(eg, ExceptionGroup[ValueError | TypeError])
                print(eg)
            """)

    @skip_before((3, 11))
    def test_base_exception_group_binding_type(self):
        self.assert_passes("""
            def capybara() -> None:
                try:
                    raise BaseExceptionGroup("boom", [KeyboardInterrupt()])
                except* KeyboardInterrupt as eg:
                    assert_type(eg, BaseExceptionGroup[KeyboardInterrupt])
                print(eg)
            """)

    @skip_before((3, 11))
    def test_except_star_rejects_exception_group_handler_type(self):
        self.assert_passes("""
            def capybara() -> None:
                try:
                    raise ExceptionGroup("boom", [ValueError()])
                except* ExceptionGroup as eg:  # E: bad_except_handler
                    print(eg)
            """)


class TestProtocolInstantiation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_protocol_instantiation_is_rejected(self):
        from typing import Protocol

        class Proto(Protocol):
            def meth(self) -> int: ...

        def capybara() -> None:
            Proto()  # E: incompatible_call


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
