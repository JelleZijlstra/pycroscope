# static analysis: ignore
import pytest

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestStub(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def capybara():
            print("keep")
            from _pycroscope_tests.deprecated import DeprecatedCapybara  # E: deprecated

            print("these imports")
            from _pycroscope_tests.deprecated import (
                deprecated_function,  # E: deprecated
            )

            print("separate")
            from _pycroscope_tests.deprecated import deprecated_overload

            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")

            deprecated_function(1)
            print(deprecated_function)
            DeprecatedCapybara()
            print(DeprecatedCapybara)

    @assert_passes()
    def test_multiline_import(self):
        def capybara():
            from _pycroscope_tests.deprecated import (
                DeprecatedCapybara,  # E: deprecated
                deprecated_function,  # E: deprecated
                deprecated_overload,
            )

            return [deprecated_function, deprecated_overload, DeprecatedCapybara]


class TestRuntime(TestNameCheckVisitorBase):
    @assert_passes()
    def test_overload(self):
        from pycroscope.extensions import deprecated, overload

        @overload
        @deprecated("int support is deprecated")
        def deprecated_overload(x: int) -> int: ...

        @overload
        def deprecated_overload(x: str) -> str: ...

        def deprecated_overload(x):
            return x

        def capybara():
            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")

    @assert_passes()
    def test_typing_extensions_deprecated_overload(self):
        from typing import overload

        from typing_extensions import deprecated

        @overload
        @deprecated("int support is deprecated")
        def deprecated_overload(x: int) -> int: ...

        @overload
        def deprecated_overload(x: str) -> str: ...

        def deprecated_overload(x: int | str):
            return x

        def capybara():
            deprecated_overload(1)  # E: deprecated
            deprecated_overload("x")
            deprecated_overload(1.0)  # E: incompatible_argument

    @assert_passes()
    def test_function(self):
        from pycroscope.extensions import deprecated

        @deprecated("no functioning capybaras")
        def deprecated_function(x: int) -> int:
            return x

        def capybara():
            print(deprecated_function)  # E: deprecated
            deprecated_function(1)  # E: deprecated

    @assert_passes()
    def test_method(self):
        from pycroscope.extensions import deprecated

        class Cls:
            @deprecated("no methodical capybaras")
            def deprecated_method(self, x: int) -> int:
                return x

        def capybara():
            Cls().deprecated_method(1)  # E: deprecated
            print(Cls.deprecated_method)  # E: deprecated

    @assert_passes()
    def test_class(self):
        from pycroscope.extensions import deprecated

        @deprecated("no classy capybaras")
        class DeprecatedClass:
            pass

        def capybara():
            print(DeprecatedClass)  # E: deprecated
            return DeprecatedClass()  # E: deprecated

    @assert_passes()
    def test_union_call_target(self):
        from pycroscope.extensions import deprecated

        @deprecated("old")
        def f() -> None:
            pass

        def g() -> None:
            pass

        def capybara(flag: bool) -> None:
            fn = f if flag else g  # E: deprecated
            fn()  # E: deprecated

    @assert_passes()
    def test_dunder_call(self):
        from pycroscope.extensions import deprecated

        class Invocable:
            @deprecated("Deprecated")
            def __call__(self) -> None:
                pass

        def capybara() -> None:
            Invocable()()  # E: deprecated

    @assert_passes()
    def test_intersection_dunder_call(self):
        from pycroscope.extensions import Intersection, deprecated

        class DeprecatedInvocable:
            @deprecated("Deprecated")
            def __call__(self) -> None:
                pass

        class PlainInvocable:
            def __call__(self) -> None:
                pass

        def capybara(value: Intersection[DeprecatedInvocable, PlainInvocable]) -> None:
            value()  # E: deprecated  # E: not_callable

    @assert_passes()
    def test_inherited_property(self):
        from pycroscope.extensions import deprecated

        class Base:
            @property
            @deprecated("getter")
            def greasy(self) -> int:
                return 1

            @property
            def shape(self) -> str:
                return "cube"

            @shape.setter
            @deprecated("setter")
            def shape(self, value: str) -> None:
                pass

        class Child(Base):
            pass

        def capybara(child: Child) -> None:
            child.greasy  # E: deprecated
            child.shape = "sphere"  # E: deprecated

    @assert_passes()
    def test_inherited_dunder_call(self):
        from pycroscope.extensions import deprecated

        class Base:
            @deprecated("Deprecated")
            def __call__(self) -> None:
                pass

        class Child(Base):
            pass

        def capybara() -> None:
            Child()()  # E: deprecated

    @assert_passes(run_in_both_module_modes=True)
    @pytest.mark.filterwarnings("ignore:.*:DeprecationWarning")
    def test_unimportable_module_deprecations(self):
        from typing import Protocol

        from typing_extensions import Self, deprecated, override

        class Spam:
            @deprecated("There is enough spam in the world")
            def __add__(self, other: object) -> Self:
                return self

            @property
            @deprecated("All spam will be equally greasy")
            def greasy(self) -> float:
                return 0.0

            @property
            def shape(self) -> str:
                return "cube"

            @shape.setter
            @deprecated("Shapes are becoming immutable")
            def shape(self, value: str) -> None:
                pass

        class Invocable:
            @deprecated("Deprecated")
            def __call__(self) -> None:
                pass

        @deprecated("Deprecated")
        def lorem() -> None:
            pass

        class SupportsFoo1(Protocol):
            @deprecated("Deprecated")
            def foo(self) -> None: ...

        class FooConcrete1(SupportsFoo1):
            @override
            def foo(self) -> None:
                pass

        spam = Spam()
        _ = spam + 1  # E: deprecated
        spam += 1  # E: deprecated
        spam.greasy  # E: deprecated
        spam.shape += "cube"  # E: deprecated

        invocable = Invocable()
        invocable()  # E: deprecated
        lorem()  # E: deprecated

        def foo_it(f: SupportsFoo1) -> None:
            f.foo()  # E: deprecated
