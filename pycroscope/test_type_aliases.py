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
    def test_312(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type
            type MyType = int

            def f(x: MyType):
                assert_type(x, MyType)
                assert_type(x + 1, int)

            def capybara(i: int, s: str):
                f(i)
                f(s)  # E: incompatible_argument
        """
        )

    @skip_before((3, 12))
    def test_312_generic(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type
            type MyType[T] = list[T] | set[T]

            def f(x: MyType[int]):
                assert_type(x, MyType[int])
                assert_type(list(x), list[int])

            def capybara(i: int, s: str):
                f([i])
                f([s])  # E: incompatible_argument
        """
        )

    @skip_before((3, 12))
    def test_312_generic_paramspec(self):
        self.assert_passes(
            """
            from typing import Callable

            type Alias[**P] = Callable[P, int]
        """
        )

    @skip_before((3, 12))
    def test_312_local_alias(self):
        self.assert_passes(
            """
            def capybara():
                type MyType = int  # E: invalid_annotation
        """
        )

    @skip_before((3, 12))
    def test_312_literal(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            type MyType = Literal[1, 2, 3]

            def capybara(x: MyType):
                assert_type(x + 1, Literal[2, 3, 4])

            def pacarana(x: MyType):
                capybara(x)
        """
        )

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
        self.assert_passes(
            """
            from typing import TypeVar

            V = TypeVar("V")
            type TA1[K] = dict[K, V]  # E: invalid_annotation

            T1 = TypeVar("T1")
            type TA2 = list[T1]  # E: invalid_annotation
        """
        )

    @skip_before((3, 12))
    def test_312_alias_bounds_and_circularity(self):
        self.assert_passes(
            """
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
        """
        )

    @skip_before((3, 12))
    def test_312_alias_redeclaration(self):
        self.assert_passes(
            """
            type Alias = int
            type Alias = int  # E: invalid_annotation
        """
        )

    @skip_before((3, 12))
    def test_312_iteration(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            type MyType = tuple[int, str, float]

            def capybara(t: MyType):
                x, y, z = t
                assert_type(x, int)
                assert_type(y, str)
                assert_type(z, float)

            def pacarana(x: MyType):
                capybara(x)
        """
        )
