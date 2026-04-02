# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestConstructors(TestNameCheckVisitorBase):
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

    @assert_passes()
    def test_runtime_constructor_prefers_init_when_new_returns_self(self):
        from typing_extensions import Self

        class C:
            def __new__(cls) -> Self:
                return super().__new__(cls)

            def __init__(self, x: int) -> None:
                pass

        def f() -> None:
            # TODO: `C(1)` also fails at runtime, so ideally we would reject it too.
            # More ambitiously, we could report that `C` itself is not constructible.
            C()  # E: incompatible_call
            C(1)

    @assert_passes()
    def test_runtime_constructor_with_paramless_init_is_rejected(self):
        class C:
            def __init__() -> None:  # E: method_first_arg
                pass

        def f() -> None:
            C()  # E: incompatible_call

    @assert_passes()
    def test_runtime_constructor_with_paramless_new_is_rejected(self):
        class C:
            def __new__():
                return object.__new__(C)

        def f() -> None:
            C()  # E: incompatible_call

    @assert_passes()
    def test_constructor_subscript_ignores_invalid_runtime_signature_metadata(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class C(Generic[T]):
            def __new__(cls):
                return object.__new__(cls)

        C.__new__.__signature__ = 1

        assert_type(C[int](), C[int])

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
