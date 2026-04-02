# static analysis: ignore
import sys

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before, skip_if


class TestRecursion(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Dict, List, Union

        JSON = Union[Dict[str, "JSON"], List["JSON"], int, str, float, bool, None]

        def f(x: JSON):
            pass

        def capybara():
            f([])
            f([1, 2, 3])
            f([[{1}]])  # E: incompatible_argument

    @assert_passes()
    def test_simple(self):
        from typing import Union

        Alias = Union[list["Alias"], int]

        x: Alias = 1

        def f(y: Alias):
            pass

        def capybara():
            f(x)
            f([x])
            f([1, 2, 3])
            f([[{1}]])  # E: incompatible_argument

    @assert_passes()
    def test_recursive_generic_implicit_alias_specialization(self):
        from typing import TypeVar

        T1 = TypeVar("T1", str, int)
        T2 = TypeVar("T2")

        GenericTypeAlias1 = list["GenericTypeAlias1[T1]" | T1]
        GenericTypeAlias2 = list["GenericTypeAlias2[T1, T2]" | T1 | T2]
        SpecializedTypeAlias1 = GenericTypeAlias1[str]

        g2: GenericTypeAlias1[str] = ["hi", "bye", [""], [["hi"]]]
        g1: SpecializedTypeAlias1 = ["hi", ["hi", "hi"]]
        g3: GenericTypeAlias1[str] = ["hi", [2.4]]  # E: incompatible_assignment
        g6: GenericTypeAlias2[str, int] = [  # E: incompatible_assignment
            [3, ["hi", 3, [3.4]]],
            "hi",
        ]

    @assert_passes()
    def test_explicit_typealias_cycle_detection(self):
        from typing import TypeAlias, Union

        RecursiveContainer: TypeAlias = Union[list["RecursiveContainer"], int]
        _ok: RecursiveContainer = [1, [2]]

        RecursiveUnion: TypeAlias = Union[  # E: invalid_type_alias
            "RecursiveUnion", int
        ]
        MutualReference1: TypeAlias = Union[  # E: invalid_type_alias
            "MutualReference2", int
        ]
        MutualReference2: TypeAlias = Union[  # E: invalid_type_alias
            "MutualReference1", str
        ]

    @assert_passes()
    def test_implicit_union_alias_runtime_call(self):
        ListAlias = list
        ListOrSetAlias = list | set

        x1: list[str] = ListAlias()
        x2 = ListAlias[int]()

        def bad_calls():
            x3 = ListOrSetAlias()  # E: incompatible_call
            return x3

    @assert_passes()
    def test_generic_implicit_alias_attribute_access(self):
        from typing import TypeVar

        T = TypeVar("T")
        SequenceAlias = list[T] | tuple[T, ...]

        def f(seq: SequenceAlias[int]) -> int:
            return seq.count(1)

    @assert_passes()
    def test_non_generic_implicit_alias_isinstance_runtime_classinfo(self):
        AstType = type[int] | tuple[type[int], ...]

        def includes(typ: AstType) -> bool:
            return isinstance(1, typ)

    @assert_passes()
    def test_implicit_alias_rejects_self_in_class_scope(self):
        from typing_extensions import Self

        class C:
            Alias = tuple[Self]  # E: invalid_self_usage

    @assert_passes()
    def test_implicit_alias_ignores_renamed_any(self):
        from typing import Any as TypingAny

        Alias = TypingAny
        x: Alias = 1

        print(x)

    @assert_passes()
    def test_implicit_alias_ignores_qualifier_expression(self):
        from typing import Final

        Alias = Final[int]
        x: Alias = 1

        print(x)

    @skip_if(sys.version_info >= (3, 14))
    @assert_passes(allow_import_failures=True)
    def test_implicit_alias_union_of_named_runtime_union_before_314(self):
        Base = list | set
        Alias = Base | tuple

        def f(x: Alias[int]) -> None:  # E: invalid_annotation
            print(x)

    @skip_if(sys.version_info < (3, 14))
    @assert_passes()
    def test_implicit_alias_union_of_named_runtime_union_314_plus(self):
        Base = list | set
        Alias = Base | tuple

        def f(x: Alias[int]) -> None:
            print(x)

    @assert_passes()
    def test_implicit_alias_union_of_named_generic_alias(self):
        from typing import TypeVar

        T = TypeVar("T")
        Base = list[T]
        Alias = Base | set[T]

        def f(x: Alias[int]) -> None:
            print(x)

    @assert_passes()
    def test_implicit_alias_union_of_typevar_and_runtime_type(self):
        from typing import TypeVar

        T = TypeVar("T")
        Alias = T | int

        def f(x: Alias[str]) -> None:
            print(x)

    @assert_passes()
    def test_implicit_alias_runtime_typeform_union(self):
        Alias = type[int] | tuple[type[int], ...]

        def f(x: Alias) -> bool:
            return isinstance(1, x)

    @assert_passes()
    def test_implicit_alias_ignores_paramspec_components(self):
        from typing import ParamSpec

        P = ParamSpec("P")
        ArgsAlias = P.args
        KwargsAlias = P.kwargs

        print(ArgsAlias, KwargsAlias)

    @assert_passes()
    def test_implicit_alias_from_typevar_name(self):
        from typing import TypeVar

        T = TypeVar("T")
        Alias = T

        print(Alias)

    @assert_passes()
    def test_implicit_alias_from_paramspec_name(self):
        from typing import ParamSpec

        P = ParamSpec("P")
        Alias = P

        print(Alias)

    @skip_before((3, 11))
    def test_implicit_alias_from_typevartuple_unpack(self):
        self.assert_passes("""
            from typing import TypeVarTuple

            Ts = TypeVarTuple("Ts")
            Alias = tuple[*Ts]

            print(Alias)
        """)

    @skip_before((3, 11))
    @assert_passes()
    def test_implicit_alias_of_alias_keeps_type_params_in_importable_mode(self):
        from typing import TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Box = list[T]
        Alias = Box

        def f(x: Alias[int]) -> None:
            assert_type(x, list[int])

    @skip_before((3, 11))
    @assert_passes()
    def test_implicit_alias_of_explicit_typealias_keeps_type_params(self):
        from typing import TypeAlias, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Box: TypeAlias = list[T]
        Alias = Box

        def f(x: Alias[int]) -> None:
            assert_type(x, list[int])

    @assert_passes()
    def test_implicit_alias_of_conditional_runtime_generic_keeps_type_params(self):
        from random import random
        from typing import TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Box = list[T] if random() else list[T]
        Alias = Box

        def f(x: Alias[int]) -> None:
            assert_type(x, list[int])

    @assert_passes()
    def test_implicit_alias_from_conditional_generic_value(self):
        from random import random
        from typing import TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Box = list[T] if random() else list[T]
        Alias = Box

        def f(x: Alias[int]) -> None:
            y = x
            assert_type(y, list[int])

    @assert_passes(allow_import_failures=True)
    def test_implicit_alias_infers_type_params_from_string_forward_refs(self):
        from typing import TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Alias = list["T"]

        def f(x: Alias[int]) -> None:
            assert_type(x, list[int])

    @skip_before((3, 11))
    def test_typevartuple_alias_empty_specialization(self):
        self.assert_passes("""
            from typing import Generic, TypeVarTuple

            from typing_extensions import assert_type

            Ts = TypeVarTuple("Ts")

            class Array(Generic[*Ts]):
                ...

            IntTuple = tuple[int, *Ts]
            NamedArray = tuple[str, Array[*Ts]]

            def capybara(a: IntTuple[()], b: NamedArray[()]) -> None:
                assert_type(a, tuple[int])
                assert_type(b, tuple[str, Array[()]])
        """)

    @skip_before((3, 11))
    def test_typevartuple_alias_unpack_specialization(self):
        self.assert_passes("""
            from typing import TypeVar, TypeVarTuple

            from typing_extensions import assert_type

            Ts = TypeVarTuple("Ts")
            T1 = TypeVar("T1")
            T2 = TypeVar("T2")

            TA7 = tuple[*Ts, T1, T2]

            def func7(a: TA7[*Ts, T1, T2]) -> tuple[tuple[*Ts], T1, T2]:
                raise NotImplementedError

            def has_expected_types(
                a: TA7[str, bool], b: TA7[str, bool, float], c: TA7[str, bool, float, int]
            ) -> None:
                assert_type(func7(a), tuple[tuple[()], str, bool])
                assert_type(func7(b), tuple[tuple[str], bool, float])
                assert_type(func7(c), tuple[tuple[str, bool], float, int])

            TA9 = tuple[*Ts, T1]
            TA10 = TA9[*tuple[int, ...]]

            def uses_unpacked_aliases(
                a: TA10, b: TA9[*tuple[int, ...], str], c: TA9[*tuple[int, ...], str]
            ) -> None:
                assert_type(a, tuple[*tuple[int, ...], int])
                assert_type(b, tuple[*tuple[int, ...], str])
                assert_type(c, tuple[*tuple[int, ...], str])
        """)

    @skip_before((3, 11))
    def test_typevartuple_alias_unpack_specialization_in_static_fallback(self):
        self.assert_passes(
            """
            from typing import Generic, TypeVar, TypeVarTuple

            from typing_extensions import assert_type

            Ts = TypeVarTuple("Ts")
            T1 = TypeVar("T1")
            T2 = TypeVar("T2")
            T3 = TypeVar("T3")

            class Array(Generic[*Ts]):
                pass

            NamedArray = tuple[str, Array[*Ts]]

            def check_empty_specialization(
                int_tuple: tuple[int], named_array: NamedArray[()]
            ) -> None:
                assert_type(int_tuple, tuple[int])
                assert_type(named_array, tuple[str, Array[()]])

            TA7 = tuple[*Ts, T1, T2]

            def func7(a: TA7[*Ts, T1, T2]) -> tuple[tuple[*Ts], T1, T2]:
                raise NotImplementedError

            TA8 = tuple[T1, *Ts, T2, T3]

            def func8(a: TA8[T1, *Ts, T2, T3]) -> tuple[tuple[*Ts], T1, T2, T3]:
                raise NotImplementedError

            def has_expected_types(
                a: TA7[str, bool],
                b: TA7[str, bool, float],
                c: TA7[str, bool, float, int],
                d: TA8[str, bool, float],
                e: TA8[str, bool, float, int],
            ) -> None:
                assert_type(func7(a), tuple[tuple[()], str, bool])
                assert_type(func7(b), tuple[tuple[str], bool, float])
                assert_type(func7(c), tuple[tuple[str, bool], float, int])
                assert_type(func8(d), tuple[tuple[()], str, bool, float])
                assert_type(func8(e), tuple[tuple[bool], str, float, int])
            """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 11))
    def test_typevartuple_alias_empty_specialization_through_call(self):
        self.assert_passes(
            """
            from typing import Generic, TypeAlias, TypeVarTuple

            from typing_extensions import assert_type

            Ts = TypeVarTuple("Ts")

            class Array(Generic[*Ts]):
                def shape(self) -> tuple[*Ts]:
                    raise NotImplementedError

            NamedArray: TypeAlias = tuple[str, Array[*Ts]]

            def unwrap(value: NamedArray[()]) -> Array[()]:
                return value[1]

            def capybara(value: NamedArray[()]) -> None:
                assert_type(unwrap(value), Array[()])
                assert_type(unwrap(value).shape(), tuple[()])
            """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 11))
    def test_typevartuple_direct_method_empty_specialization(self):
        self.assert_passes(
            """
            from typing import Generic, TypeVarTuple

            from typing_extensions import assert_type

            Ts = TypeVarTuple("Ts")

            class Array(Generic[*Ts]):
                def shape(self) -> tuple[*Ts]:
                    raise NotImplementedError

            def capybara(value: Array[()]) -> None:
                assert_type(value.shape(), tuple[()])
            """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 11))
    def test_typevartuple_implicit_alias_validation(self):
        self.assert_passes(
            """
            from typing import TypeVar, TypeVarTuple

            Ts = TypeVarTuple("Ts")
            Ts2 = TypeVarTuple("Ts2")
            T1 = TypeVar("T1")
            T2 = TypeVar("T2")

            # Multiple unbounded unpacks in tuple aliases are invalid.
            TA5 = tuple[T1, *Ts, T2, *Ts]  # E: invalid_annotation
            TA6 = tuple[T1, *Ts, T2, *tuple[int, ...]]  # E: invalid_annotation

            # A non-variadic parameter cannot be specialized with *Ts2.
            TA11 = tuple[T1, *Ts]
            TA12 = TA11[*Ts2]  # E: invalid_annotation
        """,
            allow_import_failures=True,
        )

    @assert_passes()
    def test_runtime_bitor_assignments_are_not_implicit_aliases(self):
        import re

        first = 1
        second = 2
        combined = first | second
        flags = re.VERBOSE | re.DOTALL

        print(combined, flags)


class TestTypeAliasType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing_extensions import TypeAliasType, assert_type

        MyType = TypeAliasType("MyType", int)

        def f(x: MyType):
            assert_type(x, MyType)
            assert_type(x + 1, int)

        def capybara(i: int, s: str):
            f(i)
            f(s)  # E: incompatible_argument

    @assert_passes()
    def test_typing_extensions_generic(self):
        from typing import List, Set, TypeVar, Union

        from typing_extensions import TypeAliasType, assert_type

        T = TypeVar("T")
        MyType = TypeAliasType("MyType", Union[List[T], Set[T]], type_params=(T,))

        def f(x: MyType[int]):
            assert_type(x, MyType[int])
            assert_type(list(x), List[int])

        def capybara(i: int, s: str):
            f([i])
            f([s])  # E: incompatible_argument

    @assert_passes()
    def test_typing_extensions_generic_subscripted_annotation(self):
        from typing import List, TypeVar

        from typing_extensions import TypeAliasType, assert_type

        T = TypeVar("T")
        MyType = TypeAliasType("MyType", List[T], type_params=(T,))

        def f(x: MyType[int]) -> None:
            assert_type(x, MyType[int])

        def capybara() -> None:
            f([1])

    @assert_passes()
    def test_typing_extensions_runtime_attributes(self):
        from typing import Literal, TypeVar

        from typing_extensions import TypeAliasType, assert_type

        T = TypeVar("T")
        Alias = TypeAliasType("Alias", list[T], type_params=(T,))

        assert_type(Alias.__name__, Literal["Alias"])
        print(Alias.__module__)
        print(Alias.__value__)
        print(Alias.__type_params__)

    @assert_passes()
    def test_typing_extensions_paramspec_list_arg(self):
        from typing import Callable, ParamSpec

        from typing_extensions import TypeAliasType

        P = ParamSpec("P")
        Alias = TypeAliasType("Alias", Callable[P, int], type_params=(P,))

        def f(x: Alias[[int, str]]) -> None:
            pass

    def test_typealiastype_bound_error_message_uses_concise_type_param_str(self):
        errors = self._run_str(
            """
            from typing import TypeVar

            from typing_extensions import TypeAliasType

            S = TypeVar("S", bound=str)
            Alias = TypeAliasType("Alias", list[S], type_params=(S,))

            bad: Alias[int]
            """,
            fail_after_first=False,
        )
        assert len(errors) == 1
        assert (
            "Type argument int is not compatible with ~S: str" in errors[0]["message"]
        )

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_variadic_specialization(self):
        self.assert_passes("""
            from typing import Callable, ParamSpec, TypeAliasType, TypeVar, TypeVarTuple

            TStr = TypeVar("TStr", bound=str)
            P = ParamSpec("P")
            Ts = TypeVarTuple("Ts")

            Alias = TypeAliasType(
                "Alias", Callable[P, TStr] | tuple[*Ts], type_params=(TStr, P, Ts)
            )

            ok1: Alias[str, ..., int, str]
            ok2: Alias[str, [int, str], int]
            bad: Alias[int, ...]  # E: invalid_specialization
        """)

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_conformance_specializations(self):
        self.assert_passes("""
            from typing import Callable, TypeAliasType, TypeVar, TypeVarTuple

            from typing_extensions import ParamSpec

            S = TypeVar("S")
            TStr = TypeVar("TStr", bound=str)
            P = ParamSpec("P")
            Ts = TypeVarTuple("Ts")

            GoodAlias = TypeAliasType(
                "GoodAlias",
                Callable[P, TStr] | list[S] | list["GoodAlias[S, TStr, P]"] | tuple[*Ts],
                type_params=(S, TStr, P, Ts),
            )

            x1: GoodAlias[str, str, ..., int, str]
            x2: GoodAlias[int, str, ..., int, str]
            x3: GoodAlias[int, str, [int, str], *tuple[int, str, int]]
            x4: GoodAlias[int, int, ...]  # E: invalid_specialization
        """)

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_scope_checks(self):
        self.assert_passes("""
            from typing import Generic, TypeAliasType, TypeVar

            S = TypeVar("S")
            T = TypeVar("T")
            my_tuple = (S, T)

            Bad1 = TypeAliasType("Bad1", list[S], type_params=(T,))  # E: invalid_type_alias
            Bad2 = TypeAliasType("Bad2", list[S])  # E: invalid_type_alias
            Bad3 = TypeAliasType("Bad3", int, type_params=my_tuple)  # E: invalid_type_alias

            class C(Generic[T]):
                Good = TypeAliasType("Good", list[T])
        """)

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_cycles(self):
        self.assert_passes(
            """
            from typing import TypeAliasType, TypeVar

            T = TypeVar("T")

            Bad1 = TypeAliasType("Bad1", "Bad1")  # E: invalid_type_alias
            Bad2 = TypeAliasType("Bad2", T | "Bad2[str]", type_params=(T,))  # E: invalid_type_alias
            Bad3 = TypeAliasType("Bad3", "Bad4")  # E: invalid_type_alias
            Bad4 = TypeAliasType("Bad4", Bad3)  # E: invalid_type_alias
            Bad5 = TypeAliasType("Bad5", list[Bad5])  # E: invalid_type_alias

            Good = TypeAliasType("Good", T | "list[Good[T]]", type_params=(T,))
        """,
            allow_import_failures=True,
        )

    @skip_before((3, 12))
    @assert_passes()
    def test_312_runtime_typealiastype_keyword_value_and_private_class_alias(self):
        from typing import TypeAliasType

        from typing_extensions import assert_type

        Alias = TypeAliasType("Alias", value=type[int])

        def takes_alias(cls: Alias) -> None:
            assert_type(cls, type[int])

        class Box:
            __Value = int

            def get(self, x: __Value) -> __Value:
                return x

        takes_alias(int)
        takes_alias(str)  # E: incompatible_argument
        assert_type(Box().get(1), int)

    @skip_before((3, 12))
    def test_312(self):
        self.assert_passes("""
            from typing_extensions import assert_type
            type MyType = int

            def f(x: MyType):
                assert_type(x, MyType)
                assert_type(x + 1, int)

            def capybara(i: int, s: str):
                f(i)
                f(s)  # E: incompatible_argument
        """)

    @skip_before((3, 12))
    def test_312_generic(self):
        self.assert_passes("""
            from typing_extensions import assert_type
            type MyType[T] = list[T] | set[T]

            def f(x: MyType[int]):
                assert_type(x, MyType[int])
                assert_type(list(x), list[int])

            def capybara(i: int, s: str):
                f([i])
                f([s])  # E: incompatible_argument
        """)

    @skip_before((3, 12))
    def test_312_generic_paramspec(self):
        self.assert_passes("""
            from typing import Callable

            type Alias[**P] = Callable[P, int]
        """)

    @skip_before((3, 12))
    def test_312_generic_paramspec_in_static_fallback(self):
        self.assert_passes(
            """
            from typing import Callable

            type Alias[**P] = Callable[P, int]
            """,
            allow_import_failures=True,
            force_runtime_module_load_failure=True,
        )

    @skip_before((3, 12))
    def test_312_annotated_runtime_alias(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            type Alias = tuple[int, str]

            def capybara(x: Alias) -> None:
                a, b = x
                assert_type(a, int)
                assert_type(b, str)
            """,
            annotate=True,
        )

    @skip_before((3, 12))
    def test_312_local_alias(self):
        self.assert_passes("""
            def capybara():
                type MyType = int  # E: invalid_type_alias
        """)

    @skip_before((3, 12))
    def test_312_type_alias_cannot_use_self_in_static_fallback(self):
        self.assert_passes(
            """
            from typing_extensions import Self

            class C:
                type Alias = Self
            """,
            allow_import_failures=True,
            force_runtime_module_load_failure=True,
        )

    def test_typealiastype_cannot_use_self(self):
        self.assert_passes("""
            from typing_extensions import Self, TypeAliasType

            class C:
                Alias = TypeAliasType("Alias", Self)  # E: invalid_self_usage
        """)

    @assert_passes(allow_import_failures=True)
    def test_typealiastype_missing_value_in_static_fallback(self):
        from typing_extensions import TypeAliasType

        Alias = TypeAliasType("Alias")  # E: incompatible_call

    @skip_before((3, 12))
    def test_typealiastype_type_params_in_class_scope(self):
        self.assert_passes(
            """
            from typing_extensions import TypeAliasType

            class C[T]:
                Alias = TypeAliasType("Alias", list[T], type_params=(T,))
            """,
            allow_import_failures=True,
        )

    @assert_passes(allow_import_failures=True)
    def test_typealiastype_type_params_reject_non_type_params(self):
        from typing_extensions import TypeAliasType, TypeVar

        T = TypeVar("T")
        Good = TypeAliasType("Good", list[T], type_params=(T,))
        Bad = TypeAliasType("Bad", int, type_params=(1,))  # E: invalid_type_alias

        print(Good, Bad)

    @skip_before((3, 12))
    def test_312_literal(self):
        self.assert_passes("""
            from typing import assert_type, Literal

            type MyType = Literal[1, 2, 3]

            def capybara(x: MyType):
                assert_type(x + 1, Literal[2, 3, 4])

            def pacarana(x: MyType):
                capybara(x)
        """)

    @skip_before((3, 12))
    def test_312_alias_object_semantics(self):
        self.assert_passes(
            """
            type Alias = int

            Alias.bit_count  # E: undefined_attribute
            Alias()  # E: not_callable
            print(Alias.__value__)
            print(Alias.__type_params__)

            def ok(x: Alias):
                print(x.bit_count())

            class C(Alias):  # E: invalid_base
                pass

            def f(x: object):
                if isinstance(x, Alias):  # E: incompatible_argument
                    pass
        """,
            allow_import_failures=True,
        )

    def test_implicit_alias_base_specialization(self):
        self.assert_passes(
            """
            from typing import Generic, TypeVar

            T = TypeVar("T")
            S = TypeVar("S")

            class Box(Generic[T]):
                pass

            Alias = Box[T]

            class Child(Alias[int]):
                pass

            class Nested(Alias[Alias[S]]):
                pass
        """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 12))
    def test_312_reject_old_typevars(self):
        self.assert_passes("""
            from typing import TypeVar

            V = TypeVar("V")
            type TA1[K] = dict[K, V]  # E: invalid_type_alias

            T1 = TypeVar("T1")
            type TA2 = list[T1]  # E: invalid_type_alias
        """)

    @skip_before((3, 12))
    def test_312_alias_bounds_and_circularity(self):
        self.assert_passes("""
            from typing import Callable

            type RecursiveAlias[T] = T | list[RecursiveAlias[T]]
            good: RecursiveAlias[int]

            type RecursiveWithBounds[S: int, T: str, **P] = (
                Callable[P, T] | list[S] | list[RecursiveWithBounds[S, T, P]]
            )
            bad_s: RecursiveWithBounds[str, str, ...]  # E: invalid_specialization
            bad_t: RecursiveWithBounds[int, int, ...]  # E: invalid_specialization

            type Circular1 = Circular1  # E: invalid_type_alias
            type Circular2[T] = T | Circular2[str]  # E: invalid_type_alias
            type Circular3 = Circular4  # E: invalid_type_alias
            type Circular4 = Circular3  # E: invalid_type_alias
        """)

    @skip_before((3, 12))
    def test_312_alias_redeclaration(self):
        self.assert_passes("""
            type Alias = int
            type Alias = int  # E: already_declared
        """)

    @skip_before((3, 12))
    def test_312_type_parameter_expr_fallbacks(self):
        self.assert_passes(
            """
            import collections.abc

            class UsesAttributeBound[T: collections.abc.Sequence[int]]:
                ...

            class UsesNonNameBound[T: int | str]:
                ...

            class UsesForwardRef[T: Later[int]]:
                ...

            class Later[S]:
                ...
            """,
            allow_import_failures=True,
            force_runtime_module_load_failure=True,
        )

    @skip_before((3, 12))
    def test_312_iteration(self):
        self.assert_passes("""
            from typing import assert_type, Literal

            type MyType = tuple[int, str, float]

            def capybara(t: MyType):
                x, y, z = t
                assert_type(x, int)
                assert_type(y, str)
                assert_type(z, float)

            def pacarana(x: MyType):
                capybara(x)
        """)
