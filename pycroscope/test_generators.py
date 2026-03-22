# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_if_not_installed
from .value import AnySource, AnyValue, assert_is_value


class TestGenerator(TestNameCheckVisitorBase):
    @assert_passes()
    def test_generator_return(self):
        from typing import Generator

        def gen(cond) -> Generator[int, str, bytes]:
            x = yield 1
            assert_type(x, str)
            yield "x"  # E: incompatible_yield
            if cond:
                return b"hello"
            else:
                return "hello"  # E: incompatible_return_value

        def capybara() -> Generator[int, int, int]:
            x = yield from gen(True)  # E: incompatible_yield
            assert_type(x, bytes)

            return 3

    @assert_passes()
    def test_iterable_return(self):
        from typing import Iterable

        def gen(cond) -> Iterable[int]:
            x = yield 1
            assert_type(x, None)

            yield "x"  # E: incompatible_yield

            if cond:
                return
            else:
                return 3  # E: incompatible_return_value

        def caller() -> Iterable[int]:
            x = yield from gen(True)
            assert_is_value(x, AnyValue(AnySource.generic_argument))

    @assert_passes()
    def test_yield_from_custom_awaitable(self):
        from typing import Awaitable, Generator

        class CustomAwaitable(Awaitable[int]):
            def __await__(self) -> Generator[None, None, int]:
                if False:
                    yield None
                return 42

        def capybara() -> Generator[None, None, int]:
            x = yield from CustomAwaitable()
            assert_type(x, int)
            return x

    @assert_passes()
    def test_yield_from_incompatible_yield_type(self):
        from typing import Generator

        def inner() -> Generator[str, None, int]:
            yield "capybara"
            return 42

        def outer() -> Generator[int, None, int]:
            x = yield from inner()  # E: incompatible_yield
            assert_type(x, int)
            return x

    @assert_passes()
    def test_await_dunder_await_only(self):
        from typing import Generator

        class CustomAwaitable:
            def __await__(self) -> Generator[None, None, int]:
                if False:
                    yield None
                return 42

        async def capybara() -> int:
            x = await CustomAwaitable()
            assert_type(x, int)
            return x

    @assert_passes()
    def test_await_generic_awaitable(self):
        from typing import Awaitable

        async def capybara(task: Awaitable[int]) -> int:
            x = await task
            assert_type(x, int)
            return x


class TestGeneratorReturn(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sync(self):
        from typing import Generator, Iterable, Protocol

        def gen() -> int:  # E: generator_return
            yield 1

        def caller() -> int:  # E: generator_return
            x = yield from [1, 2]
            print(x)

        def gen2() -> Iterable[int]:
            yield 1

        # E: missing_return
        def gen_missing_return(cond: bool) -> Generator[int, None, bytes]:
            if cond:
                return b"capybara"
            yield 1

        def caller2() -> Generator[int, None, None]:
            x = yield from [1, 2]
            print(x)

        class IntIterator(Protocol):
            def __next__(self) -> int: ...

        def gen_protocol_return() -> IntIterator:
            yield 1

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq(self):
        from asynq import ConstFuture, asynq

        @asynq()
        def asynq_gen() -> int:
            x = yield ConstFuture(1)
            return x

    @assert_passes()
    def test_async(self):
        from typing import AsyncGenerator, AsyncIterable, Awaitable, Protocol

        async def gen() -> int:  # E: generator_return
            yield 1

        async def gen2() -> AsyncIterable[int]:
            yield 1

        async def gen3() -> AsyncGenerator[int, None]:
            yield 1

        class AsyncIntIterator(Protocol):
            def __anext__(self) -> Awaitable[int]: ...

        async def gen_protocol_return() -> AsyncIntIterator:
            yield 1


class TestAsynqYieldHandling(TestNameCheckVisitorBase):
    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_const_future_and_none(self):
        from typing import Any

        from asynq import ConstFuture, FutureBase, asynq
        from typing_extensions import Literal

        @asynq()
        def capybara(condition):
            yield FutureBase()
            val = yield ConstFuture(3)
            assert_type(val, Literal[3])
            val2 = yield None
            assert_type(val2, None)

            if condition:
                task = ConstFuture(4)
            else:
                task = capybara.asynq(True)
            val3 = yield task
            assert_type(val3, Any | Literal[4])

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_unwrap_collections(self):
        from typing import Sequence

        from asynq import asynq
        from typing_extensions import Literal

        @asynq()
        def async_fn(n: int) -> str:
            return "async_fn"

        @asynq()
        def square(n: int) -> int:
            return int(n * n)

        class Capybara:
            @asynq()
            def async_method(self) -> str:
                return "capybara"

        @asynq()
        def capybara(ints: Sequence[Literal[0, 1, 2]]) -> int:
            val1 = yield async_fn.asynq(1)
            assert_type(val1, str)
            val2 = yield square.asynq(3)
            assert_type(val2, int)

            val3, val4 = yield async_fn.asynq(1), async_fn.asynq(2)
            assert_type(val3, str)
            assert_type(val4, str)

            val5 = yield Capybara().async_method.asynq()
            assert_type(val5, str)

            vals1 = yield [square.asynq(1), square.asynq(2), square.asynq(3)]
            assert_is_value(
                vals1,
                make_simple_sequence(
                    list, [TypedValue(int), TypedValue(int), TypedValue(int)]
                ),
            )

            vals2 = yield [square.asynq(i) for i in ints]
            for val in vals2:
                assert_type(val, int)

            vals3 = yield {1: square.asynq(1)}
            assert_is_value(
                vals3,
                DictIncompleteValue(dict, [KVPair(KnownValue(1), TypedValue(int))]),
            )

            vals4 = yield {i: square.asynq(i) for i in ints}
            assert_is_value(
                vals4,
                DictIncompleteValue(
                    dict,
                    [
                        KVPair(
                            KnownValue(0) | KnownValue(1) | KnownValue(2),
                            TypedValue(int),
                            is_many=True,
                        )
                    ],
                ),
            )
            return len(vals1) + len(vals2) + len(vals3) + len(vals4)

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_bad_async_yield(self):
        from asynq import asynq

        @asynq()
        def capybara():
            yield 1  # E: bad_async_yield

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_yield_without_value(self):
        from asynq import asynq

        @asynq()
        def capybara():
            yield  # E: yield_without_value

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_task_needs_yield(self):
        from asynq import asynq

        from pycroscope.asynq_tests import async_fn

        @asynq()
        def capybara(oid):
            async_fn.asynq(oid)  # E: task_needs_yield

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_task_needs_yield_constfuture(self):
        from asynq import ConstFuture, asynq

        @asynq()
        def capybara():  # E: task_needs_yield
            return ConstFuture(3)

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_task_needs_yield_async_return(self):
        from asynq import asynq

        @asynq()
        def async_fn():
            pass

        @asynq()
        def capybara():  # E: task_needs_yield
            return async_fn.asynq()

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq_return_type_inference(self):
        from asynq import AsyncTask, ConstFuture, FutureBase, async_proxy, asynq
        from typing_extensions import Literal

        def returns_3():
            return 3

        @asynq(pure=True)
        def pure_async_fn():
            return 4

        @asynq()
        def async_fn():
            return 3

        class WithAProperty:
            @property
            def this_is_one(self):
                return str(5)

        @async_proxy(pure=True)
        def pure_async_proxy(oid):
            return ConstFuture(oid)

        @async_proxy()
        def impure_async_proxy():
            return ConstFuture(3)

        def capybara(oid):
            assert_type(returns_3(), Literal[3])
            assert_is_value(
                pure_async_fn(), AsyncTaskIncompleteValue(AsyncTask, KnownValue(4))
            )
            assert_type(async_fn(), Literal[3])
            assert_is_value(
                async_fn.asynq(), AsyncTaskIncompleteValue(AsyncTask, KnownValue(3))
            )
            assert_type(WithAProperty().this_is_one, str)
            assert_is_value(pure_async_proxy(oid), AnyValue(AnySource.unannotated))
            assert_is_value(impure_async_proxy(), AnyValue(AnySource.unannotated))
            assert_is_value(
                impure_async_proxy.asynq(),
                AsyncTaskIncompleteValue(FutureBase, AnyValue(AnySource.unannotated)),
            )

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq_missing_return(self):
        from asynq import asynq

        @asynq()
        def capybara() -> int:  # E: missing_return
            yield capybara.asynq()

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq_missing_branch(self):
        from asynq import asynq

        @asynq()
        def capybara(cond: bool) -> int:  # E: missing_return
            if cond:
                return 3
            yield capybara.asynq(False)


class TestAsyncCoverageInFocusedRun(TestNameCheckVisitorBase):
    @assert_passes()
    def test_await_type_inference(self):
        from typing_extensions import Literal

        async def capybara(x):
            assert_is_value(x, AnyValue(AnySource.unannotated))
            return "hydrochoerus"

        async def kerodon(x):
            task = capybara(x)
            val = await task
            assert_type(val, Literal["hydrochoerus"])

    @assert_passes()
    def test_bad_await_operand(self):
        async def capybara():
            await None  # E: unsupported_operation

    @assert_passes()
    def test_exotic_awaitable(self):
        from typing import Awaitable, Iterable, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class Aww(Iterable[T], Awaitable[U]):
            pass

        async def capybara(aw: Aww[int, str]) -> None:
            assert_type(await aw, str)

    @assert_passes()
    def test_async_comprehension(self):
        from typing_extensions import Self

        class ANext:
            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> int:
                return 42

        class AIter:
            def __aiter__(self) -> ANext:
                return ANext()

        async def capybara():
            x = [y async for y in AIter()]
            assert_is_value(x, SequenceValue(list, [(True, TypedValue(int))]))

    @assert_passes()
    def test_bad_async_comprehension(self):
        async def capybara():
            return [x async for x in []]  # E: unsupported_operation

    @assert_passes()
    def test_missing_await_in_async_def(self):
        async def f():
            return 42

        async def capybara():
            f()  # E: missing_await

    @assert_passes()
    def test_missing_await_in_sync_def(self):
        async def f():
            return 42

        def capybara():
            f()  # E: missing_await

    @assert_passes()
    def test_missing_await_external(self):
        import asyncio

        async def capybara():
            asyncio.sleep(1)  # E: missing_await

    @assert_passes()
    def test_async_for_not_async_iterable(self):
        async def capybara() -> None:
            async for i in [1, 2, 3]:  # E: unsupported_operation  # E: unused_variable
                pass
