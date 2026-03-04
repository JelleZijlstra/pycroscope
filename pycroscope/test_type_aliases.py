# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before


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

        g2: GenericTypeAlias1[str] = ["hi", "bye", [""], [["hi"]]]
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

        RecursiveUnion: TypeAlias = Union[  # E: invalid_annotation
            "RecursiveUnion", int
        ]
        MutualReference1: TypeAlias = Union[  # E: invalid_annotation
            "MutualReference2", int
        ]
        MutualReference2: TypeAlias = Union[  # E: invalid_annotation
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
    def test_typing_extensions_paramspec_list_arg(self):
        from typing import Callable, ParamSpec

        from typing_extensions import TypeAliasType

        P = ParamSpec("P")
        Alias = TypeAliasType("Alias", Callable[P, int], type_params=(P,))

        def f(x: Alias[[int, str]]) -> None:
            pass

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
            bad: Alias[int, ...]  # E: invalid_annotation
        """)

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_scope_checks(self):
        self.assert_passes("""
            from typing import Generic, TypeAliasType, TypeVar

            S = TypeVar("S")
            T = TypeVar("T")
            my_tuple = (S, T)

            Bad1 = TypeAliasType("Bad1", list[S], type_params=(T,))  # E: invalid_annotation
            Bad2 = TypeAliasType("Bad2", list[S])  # E: invalid_annotation
            Bad3 = TypeAliasType("Bad3", int, type_params=my_tuple)  # E: invalid_annotation

            class C(Generic[T]):
                Good = TypeAliasType("Good", list[T])
        """)

    @skip_before((3, 12))
    def test_312_runtime_typealiastype_cycles(self):
        self.assert_passes(
            """
            from typing import TypeAliasType, TypeVar

            T = TypeVar("T")

            Bad1 = TypeAliasType("Bad1", "Bad1")  # E: invalid_annotation
            Bad2 = TypeAliasType("Bad2", T | "Bad2[str]", type_params=(T,))  # E: invalid_annotation
            Bad3 = TypeAliasType("Bad3", "Bad4")  # E: invalid_annotation
            Bad4 = TypeAliasType("Bad4", Bad3)  # E: invalid_annotation
            Bad5 = TypeAliasType("Bad5", list[Bad5])  # E: invalid_annotation

            Good = TypeAliasType("Good", T | "list[Good[T]]", type_params=(T,))
        """,
            allow_import_failures=True,
        )

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
    def test_312_local_alias(self):
        self.assert_passes("""
            def capybara():
                type MyType = int  # E: invalid_annotation
        """)

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

    @skip_before((3, 12))
    def test_312_reject_old_typevars(self):
        self.assert_passes("""
            from typing import TypeVar

            V = TypeVar("V")
            type TA1[K] = dict[K, V]  # E: invalid_annotation

            T1 = TypeVar("T1")
            type TA2 = list[T1]  # E: invalid_annotation
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
            bad_s: RecursiveWithBounds[str, str, ...]  # E: invalid_annotation
            bad_t: RecursiveWithBounds[int, int, ...]  # E: invalid_annotation

            type Circular1 = Circular1  # E: invalid_annotation
            type Circular2[T] = T | Circular2[str]  # E: invalid_annotation
            type Circular3 = Circular4  # E: invalid_annotation
            type Circular4 = Circular3  # E: invalid_annotation
        """)

    @skip_before((3, 12))
    def test_312_alias_redeclaration(self):
        self.assert_passes("""
            type Alias = int
            type Alias = int  # E: invalid_annotation
        """)

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
