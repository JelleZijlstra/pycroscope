# static analysis: ignore
from .extensions import is_keyword, is_of_type, is_positional, is_provided, show_error
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import AnySource, AnyValue, assert_is_value


class TestTypeEvaluation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_is_provided(self):
        from typing import Union

        from pycroscope.extensions import evaluated

        @evaluated
        def simple_evaluated(x: int, y: str = ""):
            if is_provided(y):
                return int
            else:
                return str

        def simple_evaluated(*args: object) -> Union[int, str]:
            if len(args) >= 2:
                return 1
            else:
                return "x"

        def capybara(args, kwargs):
            assert_type(simple_evaluated(1), str)
            assert_type(simple_evaluated(1, "1"), int)
            assert_type(simple_evaluated(*args), str)
            assert_type(simple_evaluated(**kwargs), str)
            assert_type(simple_evaluated(1, y="1"), int)
            assert_type(simple_evaluated(1, **{"y": "1"}), int)

    @assert_passes()
    def test_is_of_type(self):
        from typing import Union

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def is_of_type_evaluated(x: int):
            if is_of_type(x, Literal[1]):
                return str
            else:
                return int

        def is_of_type_evaluated(x: int) -> Union[int, str]:
            if x == 1:
                return ""
            else:
                return 0

        def capybara(unannotated):
            assert_type(is_of_type_evaluated(1), str)
            assert_type(is_of_type_evaluated(2), int)
            assert_type(is_of_type_evaluated(unannotated), int)
            assert_type(is_of_type_evaluated(2 if unannotated else 1), int | str)

    @assert_passes()
    def test_not(self):
        from typing import Union

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def not_evaluated(x: int):
            if not is_of_type(x, Literal[1]):
                return str
            else:
                return int

        def not_evaluated(x: int) -> Union[int, str]:
            if x != 1:
                return ""
            else:
                return 0

        def capybara(unannotated):
            assert_type(not_evaluated(1), int)
            assert_type(not_evaluated(2), str)
            assert_type(not_evaluated(unannotated), str)
            assert_type(not_evaluated(2 if unannotated else 1), int | str)

    @assert_passes()
    def test_compare(self):
        from typing import Union

        from pycroscope.extensions import evaluated

        @evaluated
        def compare_evaluated(x: object):
            if x is None:
                return str
            elif x == 1:
                return float
            else:
                return int

        def compare_evaluated(x: object) -> Union[int, str, float]:
            raise NotImplementedError

        def capybara(unannotated):
            assert_type(compare_evaluated(None), str)
            assert_type(compare_evaluated(1), float | int)
            assert_type(compare_evaluated("x"), int)
            assert_type(
                compare_evaluated(None if unannotated else 1), str | float | int
            )

    @assert_passes()
    def test_error(self):
        from typing import Any

        from pycroscope.extensions import evaluated

        @evaluated
        def nonempty_please(x: str) -> int:
            if x == "":
                show_error("Non-empty string expected", argument=x)
                return Any
            else:
                return int

        def nonempty_please(x: str) -> int:
            assert x
            return len(x)

        def capybara():
            nonempty_please("")  # E: incompatible_call
            assert_type(nonempty_please("x"), int)

    @assert_passes()
    def test_restrict_kind(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def restrict_kind(x: str, y: int):
            if is_keyword(x):
                show_error("x must be positional", argument=x)
            if is_positional(y):
                show_error("y must be keyword", argument=y)
            return int

        def restrict_kind(*args, **kwargs):
            return 0

        def capybara(stuff):
            restrict_kind("x", y=1)
            restrict_kind(x="x", y=1)  # E: incompatible_call
            restrict_kind("x", 1)  # E: incompatible_call
            restrict_kind(*stuff, **stuff)
            restrict_kind(**stuff)  # E: incompatible_call
            restrict_kind(*stuff)  # E: incompatible_call

    @assert_passes()
    def test_pass(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def only_one(a: int):
            if a == 1:
                pass
            else:
                show_error("a must be 1", argument=a)
            return str

        def only_one(a: int) -> str:
            raise NotImplementedError

        def capybara():
            assert_type(only_one(1), str)
            assert_type(only_one(2), str)  # E: incompatible_call

    @assert_passes()
    def test_enum(self):
        import enum

        from pycroscope.extensions import evaluated

        class Color(enum.Enum):
            magenta = 1
            cyan = 2

        @evaluated
        def want_enum(color: Color):
            if color is Color.magenta:
                return str
            elif color is Color.cyan:
                return int
            else:
                return bool

        def want_enum(color: Color):
            raise NotImplementedError

        def capybara(c: Color):
            assert_type(want_enum(Color.magenta), str)
            assert_type(want_enum(Color.cyan), int)
            assert_type(want_enum(c), bool)

    @assert_passes()
    def test_platform(self):
        import sys

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def where_am_i():
            if sys.platform == "darwin":
                return Literal["On a Mac"]
            else:
                return Literal["Somewhere else"]

        def where_am_i():
            raise NotImplementedError

        def capybara():
            if sys.platform == "darwin":
                assert_type(where_am_i(), Literal["On a Mac"])
            else:
                assert_type(where_am_i(), Literal["Somewhere else"])

    @assert_passes()
    def test_version(self):
        import sys

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def is_self_available():
            if sys.version_info >= (3, 11):
                return Literal[True]
            return Literal[False]

        def is_self_available():
            return sys.version_info >= (3, 11)

        def capybara():
            if sys.version_info >= (3, 11):
                assert_type(is_self_available(), Literal[True])
            else:
                assert_type(is_self_available(), Literal[False])

    @assert_passes()
    def test_nested_ifs(self):
        from typing_extensions import Literal

        from pycroscope.extensions import evaluated, is_of_type

        @evaluated
        def is_int(i: int):
            if is_of_type(i, Literal[1, 2]):
                if i == 1:
                    return Literal[1]
                elif i == 2:
                    return Literal[2]
            return Literal[3]

        def capybara():
            assert_type(is_int(1), Literal[1])

    @assert_passes()
    def test_not_equals(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def want_one(x: int = 1, y: bool = True):
            if x != 1:
                show_error("want one", argument=x)
            if y is not True:
                show_error("want one", argument=y)
            return None

        def want_one(x: int = 1, y: bool = True) -> None:
            pass

        def capybara():
            want_one(2)  # E: incompatible_call
            want_one(y=False)  # E: incompatible_call

    @assert_passes()
    def test_reveal_type(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def has_default(x: int = 1):
            reveal_type(x)
            return None

        def has_default(x: int = 1) -> None:
            pass

        def capybara(i: int):
            has_default()  # E: incompatible_call
            has_default(i)  # E: incompatible_call

    @assert_passes()
    def test_return(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def maybe_use_header(x: bool) -> int:
            if x is True:
                return str

        def capybara(x: bool):
            assert_type(maybe_use_header(True), str)
            assert_type(maybe_use_header(x), int)

    @assert_passes()
    def test_generic(self):
        from typing import TypeVar

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        T1 = TypeVar("T1")

        @evaluated
        def identity(x: T1):
            return T1

        @evaluated
        def identity2(x: T1) -> T1:
            pass

        def no_evaluated(x: T1) -> T1:
            return x

        def capybara(unannotated):
            assert_type(identity(1), Literal[1])
            assert_is_value(identity(unannotated), AnyValue(AnySource.unannotated))
            assert_type(identity2(1), Literal[1])
            assert_is_value(identity2(unannotated), AnyValue(AnySource.unannotated))
            assert_type(no_evaluated(1), Literal[1])
            assert_is_value(no_evaluated(unannotated), AnyValue(AnySource.unannotated))


class TestBoolOp(TestNameCheckVisitorBase):
    @assert_passes()
    def test_and(self):
        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def use_and(a: int, b: str):
            if a == 1 and b == "x":
                return str
            return int

        def use_and(a: int, b: str) -> object:
            raise NotImplementedError

        def capybara(
            a: int, b: str, maybe_a: Literal[1, 2], maybe_b: Literal["x", "y"]
        ) -> None:
            assert_type(use_and(1, "x"), str)
            assert_type(use_and(a, b), int)
            assert_type(use_and(maybe_a, maybe_b), str | int)

    @assert_passes()
    def test_or(self):
        from typing_extensions import Literal

        from pycroscope.extensions import evaluated

        @evaluated
        def use_or(b: str):
            if b == "x" or b == "y":
                return str
            return int

        def use_or(b: str) -> object:
            raise NotImplementedError

        def capybara(
            b: str, x_or_y: Literal["x", "y"], x_or_z: Literal["x", "z"]
        ) -> None:
            assert_type(use_or("x"), str)
            assert_type(use_or("y"), str)
            assert_type(use_or(b), int)
            assert_type(use_or(x_or_y), str)
            assert_type(use_or(x_or_z), str | int)

    @assert_passes()
    def test_literal_or(self):
        from typing import Union

        from pycroscope.extensions import evaluated

        @evaluated
        def is_one(i: int):
            if i == 1 or i == -1:
                show_error("bad argument", argument=i)
                return int
            return str

        def is_one(i: int) -> Union[int, str]:
            raise NotImplementedError

        def capybara():
            val = is_one(-1)  # E: incompatible_call
            assert_type(val, int)
            assert_type(is_one(2), str)


class TestValidation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_bad(self):
        from pycroscope.extensions import evaluated

        @evaluated
        def bad_evaluator(a: int):
            if is_of_type(a, Literal[1]):  # E: undefined_name
                print("hello")  # E: bad_evaluator
            if is_of_type():  # E: bad_evaluator
                return  # E: bad_evaluator
            if is_of_type(b, int):  # E: bad_evaluator
                return None
            if is_of_type(a, int, exclude_any=None):  # E: bad_evaluator
                return None
            if is_of_type(a, int, exclude_any=bool(a)):  # E: bad_evaluator
                return None
            if is_of_type(a, int, bad_kwarg=True):  # E: bad_evaluator
                return None
            if not_a_function():  # E: bad_evaluator
                return None
            if ~is_provided(a):  # E: bad_evaluator
                return None
            if a == 1 == a:  # E: bad_evaluator
                return None
            if a > 1:  # E: bad_evaluator
                return None
            if a == len("x"):  # E: bad_evaluator
                return None

            if is_provided("x"):  # E: bad_evaluator
                return None

            if is_provided(b):  # E: bad_evaluator
                show_error()  # E: bad_evaluator
                show_error(1)  # E: bad_evaluator
                show_error("message", argument=b)  # E: bad_evaluator
                show_error("message", arg=a)  # E: bad_evaluator
                show_error("message", argument=a)

            if (is_provided,)[0](a):  # E: bad_evaluator
                return None
            return None

        def bad_evaluator(a: int) -> None:
            pass


class TestExamples(TestNameCheckVisitorBase):
    @assert_passes()
    def test_open(self):
        from io import (
            BufferedRandom,
            BufferedReader,
            BufferedWriter,
            FileIO,
            TextIOWrapper,
        )
        from typing import IO, Any, BinaryIO, Callable, Optional, Union

        from typing_extensions import Literal

        from pycroscope.extensions import evaluated, is_of_type

        _OpenFile = Union[str, bytes, int]
        _Opener = Callable[[str, int], int]

        # These are simplified
        OpenTextModeUpdating = Literal["r+", "w+", "a+", "x+"]
        OpenTextModeWriting = Literal["w", "wt", "tw", "a", "at", "ta", "x", "xt", "tx"]
        OpenTextModeReading = Literal["r", "rt", "tr"]
        OpenTextMode = Union[
            OpenTextModeUpdating, OpenTextModeWriting, OpenTextModeReading
        ]
        OpenBinaryModeUpdating = Literal["rb+", "wb+", "ab+", "xb+"]
        OpenBinaryModeWriting = Literal["wb", "bw", "ab", "ba", "xb", "bx"]
        OpenBinaryModeReading = Literal["rb", "br"]
        OpenBinaryMode = Union[
            OpenBinaryModeUpdating, OpenBinaryModeReading, OpenBinaryModeWriting
        ]

        @evaluated
        def open2(
            file: _OpenFile,
            mode: str = "r",
            buffering: int = -1,
            encoding: Optional[str] = None,
            errors: Optional[str] = None,
            newline: Optional[str] = None,
            closefd: bool = False,
            opener: Optional[_Opener] = None,
        ) -> IO[Any]:
            if is_of_type(mode, OpenTextMode):
                return TextIOWrapper
            elif is_of_type(mode, OpenBinaryMode):
                if encoding is not None:
                    show_error(
                        "'encoding' argument may not be provided in binary moode",
                        argument=encoding,
                    )
                if errors is not None:
                    show_error(
                        "'errors' argument may not be provided in binary moode",
                        argument=errors,
                    )
                if newline is not None:
                    show_error(
                        "'newline' argument may not be provided in binary moode",
                        argument=newline,
                    )
                if buffering == 0:
                    return FileIO
                elif buffering == -1 or buffering == 1:
                    if is_of_type(mode, OpenBinaryModeUpdating):
                        return BufferedRandom
                    elif is_of_type(mode, OpenBinaryModeWriting):
                        return BufferedWriter
                    elif is_of_type(mode, OpenBinaryModeReading):
                        return BufferedReader

                # Buffering cannot be determined: fall back to BinaryIO
                return BinaryIO
            # Fallback if mode is not specified
            return IO[Any]

        def capybara():
            assert_type(open2("x", "r"), TextIOWrapper)
            open2("x", "rb", encoding="utf-8")  # E: incompatible_call
            assert_type(open2("x", "rb", buffering=0), FileIO)
            assert_type(open2("x", "rb+"), BufferedRandom)
            assert_type(open2("x", "rb"), BufferedReader)
            assert_type(open2("x", "rb", buffering=1), BufferedReader)

    @assert_passes()
    def test_safe_upcast(self):
        from typing import Any, Type, TypeVar

        from pycroscope.extensions import evaluated, is_of_type, show_error

        T1 = TypeVar("T1")

        @evaluated
        def safe_upcast(typ: Type[T1], value: object):
            if is_of_type(value, T1):
                return T1
            show_error("unsafe cast")
            return Any

        def capybara():
            assert_type(safe_upcast(object, 1), object)
            assert_type(safe_upcast(int, 1), int)
            safe_upcast(str, 1)  # E: incompatible_call

    @assert_passes()
    def test_safe_contains(self):
        from typing import Collection, List, TypeVar

        from pycroscope.extensions import evaluated, is_of_type, show_error

        T1 = TypeVar("T1")
        T2 = TypeVar("T2")

        @evaluated
        def safe_contains(elt: T1, container: Collection[T2]) -> bool:
            if not is_of_type(elt, T2) and not is_of_type(container, Collection[T1]):
                show_error("Element cannot be a member of container")

        def capybara(lst: List[int], o: object):
            safe_contains(True, ["x"])  # E: incompatible_call
            safe_contains("x", lst)  # E: incompatible_call
            safe_contains(True, lst)
            safe_contains(o, lst)
