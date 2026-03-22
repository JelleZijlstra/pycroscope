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
