# static analysis: ignore
from .error_code import ErrorCode
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before, skip_if_not_installed
from .value import (
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    TypedValue,
    assert_is_value,
    class_owner_from_key,
)

_ASYNC_GENERATOR_CONTEXT_MANAGER_NONE = GenericValue(
    class_owner_from_key("contextlib._AsyncGeneratorContextManager"), [KnownValue(None)]
)


class TestNestedFunction(TestNameCheckVisitorBase):
    @assert_passes()
    def test_inference(self):
        def capybara():
            def nested():
                pass

            class NestedClass(object):
                pass

            assert_type(nested(), None)
            nested(1)  # E: incompatible_call
            assert_type(NestedClass(), NestedClass)

    @assert_passes()
    def test_usage_in_nested_scope():
        def capybara(cond, x):
            if cond:

                def nested(y):
                    pass

                ys = [nested(y) for y in x]

                class Nested(object):
                    xs = ys

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq(self):
        from asynq import asynq
        from typing_extensions import Literal

        @asynq()
        def capybara():
            @asynq()
            def nested() -> Literal[3]:
                return 3

            assert_type(nested(), Literal[3])
            val = yield nested.asynq()
            assert_type(val, Literal[3])

    @assert_passes()
    def test_async_def(self):
        from pycroscope.value import make_coro_type

        def capybara():
            async def nested() -> int:
                return 1

            assert_is_value(nested(), make_coro_type(TypedValue(int)))

    @assert_passes()
    def test_bad_decorator(self):
        def decorator(fn):
            return fn

        def capybara():
            @decorator
            def nested():
                pass

            assert_is_value(nested, AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_attribute_set(self):
        from typing_extensions import Literal

        def capybara():
            def inner():
                pass

            inner.punare = 3
            assert_type(inner.punare, Literal[3])

    @assert_passes()
    def test_nested_in_method(self):
        class Capybara:
            def method(self):
                def nested(arg) -> int:
                    assert_is_value(arg, AnyValue(AnySource.unannotated))
                    # Make sure we don't think this is an access to Capybara.numerator
                    print(arg.numerator)
                    return 1

                assert_type(nested(1), int)


class TestFunctionDefinitions(TestNameCheckVisitorBase):
    @assert_passes()
    def test_keyword_only(self):
        from typing import Any

        from typing_extensions import Literal

        def capybara(a, *, b, c=3):
            assert_is_value(a, AnyValue(AnySource.unannotated))
            assert_is_value(b, AnyValue(AnySource.unannotated))
            assert_type(c, Any | Literal[3])
            capybara(1, b=2)

            fn = lambda a, *, b: None
            fn(a, b=b)

        def failing_capybara(a, *, b):
            capybara(1, 2)  # E: incompatible_call

    @assert_passes(settings={ErrorCode.missing_parameter_annotation: True})
    def test_pos_only(self):
        from typing import Optional

        def f(a: int, /) -> None:
            assert_type(a, int)

        def g(a: Optional[str] = None, /, b: int = 1) -> None:
            assert_type(a, str | None)
            assert_type(b, int)

        def h(a, b: int = 1, /, c: int = 2) -> None:  # E: missing_parameter_annotation
            assert_is_value(a, AnyValue(AnySource.unannotated))
            assert_type(b, int)

        def capybara() -> None:
            f(1)
            f("x")  # E: incompatible_argument
            f(a=1)  # E: incompatible_call
            g(a=1)  # E: incompatible_call
            g(b=1)
            g(None, b=1)
            h(1, 1, c=2)
            h(1)
            h(1, b=1)  # E: incompatible_call

    @assert_passes()
    def test_historical_positional_only(self):
        def f1(__x: int, __y__: int = 0) -> None:
            pass

        def f2(x: int, __y: int) -> None:  # E: invalid_positional_only
            pass

        class A:
            def m1(self, __x: int, __y__: int = 0) -> None:
                pass

            def m2(self, x: int, __y: int) -> None:  # E: invalid_positional_only
                pass

        def f4(x: int, /, __y: int) -> None:
            pass

        def capybara() -> None:
            f1(3, __y__=1)
            f1(__x=3)  # E: incompatible_call
            a = A()
            a.m1(3, __y__=1)
            a.m1(__x=3)  # E: incompatible_call
            f4(3, __y=4)

    @assert_passes()
    def test_lambda(self):
        from typing import Callable

        from typing_extensions import Literal

        def capybara():
            fun = lambda: 1
            x: Callable[[], int] = fun
            y: Callable[[], str] = fun  # E: incompatible_assignment
            print(x, y)
            fun(1)  # E: incompatible_call
            assert_type(fun(), Literal[1])

            fun2 = lambda a: a
            fun2()  # E: incompatible_call
            assert_type(fun2(1), Literal[1])

            fun3 = lambda c=3: c
            assert_is_value(
                fun3(), KnownValue(3) | AnyValue(AnySource.generic_argument)
            )
            assert_type(fun3(2), Literal[2, 3])

            fun4 = lambda a, b, c: a if c else b
            assert_type(fun4(1, 2, 3), Literal[1, 2])

    @assert_passes()
    def test_lambda_default_uses_narrowed_value(self):
        import ast

        from typing_extensions import assert_type

        def accept(node: ast.AST) -> str:
            return ast.dump(node)

        def capybara(value_node: ast.AST | None) -> None:
            if value_node is None:
                return
            evaluator = lambda value_node=value_node: accept(value_node)
            assert_type(evaluator(), str)
            evaluator2 = lambda: accept(value_node)
            assert_type(evaluator2(), str)


class TestDecorators(TestNameCheckVisitorBase):
    @assert_passes()
    def test_applied(self) -> None:
        def bad_deco(x: int) -> str:
            return "x"

        @bad_deco  # E: incompatible_argument
        def capybara():
            pass

    @assert_passes()
    def test_asynccontextmanager(self):
        from collections.abc import AsyncGenerator
        from contextlib import asynccontextmanager

        from pycroscope.test_functions import _ASYNC_GENERATOR_CONTEXT_MANAGER_NONE

        @asynccontextmanager
        async def make_cm() -> AsyncGenerator[None]:
            yield

        async def use_cm():
            assert_is_value(make_cm(), _ASYNC_GENERATOR_CONTEXT_MANAGER_NONE)
            async with make_cm() as value:
                assert_type(value, None)

    @assert_passes()
    def test_contextmanager_generator_variants(self):
        import collections.abc
        from contextlib import contextmanager
        from typing import Generator

        @contextmanager
        def typing_generator() -> Generator[int, None, None]:
            yield 1

        @contextmanager
        def collections_generator_one() -> collections.abc.Generator[int]:
            yield 2

        @contextmanager
        def collections_generator_three() -> collections.abc.Generator[int, None, None]:
            yield 3

        def capybara():
            with typing_generator() as a:
                assert_type(a, int)
            with collections_generator_one() as b:
                assert_type(b, int)
            with collections_generator_three() as c:
                assert_type(c, int)

    @assert_passes()
    def test_nullcontext(self):
        import collections.abc
        import contextlib
        import types
        from contextlib import AbstractContextManager

        def capybara(flag: bool) -> None:
            direct: AbstractContextManager[None] = contextlib.nullcontext()
            with direct as direct_value:
                assert_type(direct_value, None)

            with contextlib.nullcontext(None) as explicit_value:
                assert_type(explicit_value, None)

            none_value: types.NoneType = None
            assert_type(none_value, None)
            assert_type(None, types.NoneType)

            if flag:
                ctx: AbstractContextManager[None] = contextlib.nullcontext()
            else:

                @contextlib.contextmanager
                def make_cm() -> collections.abc.Generator[None]:
                    yield

                ctx = make_cm()

            with ctx:
                pass

            with contextlib.nullcontext():
                pass


class TestAsyncGenerator(TestNameCheckVisitorBase):
    @assert_passes()
    def test_not_a_generator(self):
        from pycroscope.value import make_coro_type

        async def capybara() -> None:
            async def agen():
                yield 1

            def gen():
                yield 2

            print(agen, gen, lambda: (yield 3))

        def caller() -> None:
            x = capybara()
            assert_is_value(x, make_coro_type(KnownValue(None)))

    @assert_passes()
    def test_is_a_generator(self):
        import collections.abc
        from typing import AsyncIterator

        async def capybara() -> AsyncIterator[int]:
            yield 1

        def caller() -> None:
            x = capybara()
            assert_is_value(
                x, GenericValue(collections.abc.AsyncIterator, [TypedValue(int)])
            )


class TestGenericFunctions(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_generic(self):
        self.assert_passes("""
            def func[T](x: T) -> T:
                return x

            def capybara(i: int):
                assert_type(func(i), int)
        """)

    @skip_before((3, 12))
    def test_generic_with_bound(self):
        self.assert_passes("""
            def func[T: int](x: T) -> T:
                return x

            def capybara(i: int, s: str, b: bool):
                assert_type(func(i), int)
                assert_type(func(b), bool)
                func(s)  # E: incompatible_argument
        """)

    @skip_before((3, 12))
    def test_generic_with_paramspec(self):
        self.assert_passes("""
            from typing import Callable

            def decorator[T, **P, R](
                x: type[T],
            ) -> Callable[[Callable[P, R]], Callable[P, R]]:
                def apply(func: Callable[P, R]) -> Callable[P, R]:
                    return func

                return apply
        """)
