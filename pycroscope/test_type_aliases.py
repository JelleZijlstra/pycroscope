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
        Alias = list["Alias"] | int

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
    def test_312_local_alias(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            def capybara():
                type MyType = int
                def f(x: MyType):
                    assert_type(x, MyType)
                    assert_type(x + 1, int)

                f(1)
                f("x")  # E: incompatible_argument
        """
        )

    @skip_before((3, 12))
    def test_312_literal(self):
        self.assert_passes(
            """
            from typing import assert_type, Literal

            type MyType = Literal[1, 2, 3]

            def capybara(x: MyType):
                assert_type(x + 1, int)

            def pacarana(x: MyType):
                capybara(x)
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
