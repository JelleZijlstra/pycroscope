# static analysis: ignore

from .annotations import has_invalid_paramspec_usage
from .error_code import ErrorCode
from .signature import OverloadedSignature, Signature, SigParameter
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before, skip_if_not_installed
from .tests import make_simple_sequence
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    SequenceValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarParam,
    TypeVarValue,
    assert_is_value,
)


class TestAnnotations(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union(self):
        from typing import Dict, List, Match, Optional, Pattern, Set, Union

        def capybara() -> Union[int, str]:
            return 0

        def kerodon() -> Optional[int]:
            return None

        def complex() -> Union[List[str], Set[int], Dict[bytes, List[str]], int]:
            return []

        def union_in_subscript() -> List[Union[str, int]]:
            return []

        def check() -> None:
            assert_type(capybara(), int | str)
            assert_type(kerodon(), int | None)
            assert_type(complex(), list[str] | set[int] | dict[bytes, list[str]] | int)
            assert_type(union_in_subscript(), list[str | int])

        def rgx(m: Match[str], p: Pattern[bytes]) -> None:
            assert_type(p, Pattern[bytes])
            assert_type(m, Match[str])

    @assert_passes()
    def test_union_as_an_annotation(self):
        from types import UnionType

        def capybara(x: UnionType) -> None:
            pass

        def caller() -> None:
            capybara(1)  # E: incompatible_argument
            capybara(int | str)

    @assert_passes()
    def test_generic(self):
        from typing import Any, List

        def capybara(
            x: List[int], y: List, z: List[Any]  # E: missing_generic_parameters
        ) -> None:
            assert_type(x, list[int])
            assert_type(y, list)
            assert_type(z, list[Any])

    @assert_passes()
    def test_supports_int(self):
        from typing import SupportsInt

        def capybara(z: SupportsInt) -> None:
            assert_type(z, SupportsInt)

        def mara():
            capybara(1.0)
            capybara([])  # E: incompatible_argument

    @assert_passes()
    def test_supports_int_accepted(self):
        from typing import SupportsInt

        def capybara(z: SupportsInt) -> None:
            print(z)  # just test that this doesn't get rejected

    @assert_passes()
    def test_self_type(self):
        class Capybara:
            def f(self: int) -> None:
                assert_type(self, int)

            def g(self) -> None:
                assert_type(self, Capybara)

    @assert_passes()
    def test_newtype(self):
        from typing import NewType, Tuple

        X = NewType("X", int)
        Y = NewType("Y", Tuple[str, ...])
        Z = NewType("Z", tuple[str, ...])
        A = NewType("A", tuple[int, str])
        B = NewType("B", list[int])

        def capybara(x: X, y: Y, z: Z, a: A, b: B) -> None:
            assert_is_value(x, NewTypeValue("X", TypedValue(int), X))
            assert_is_value(
                y, NewTypeValue("Y", GenericValue(tuple, [TypedValue(str)]), Y)
            )
            assert_is_value(
                z, NewTypeValue("Z", GenericValue(tuple, [TypedValue(str)]), Z)
            )
            assert_is_value(
                a,
                NewTypeValue(
                    "A",
                    SequenceValue(
                        tuple, [(False, TypedValue(int)), (False, TypedValue(str))]
                    ),
                    A,
                ),
            )
            assert_is_value(
                b, NewTypeValue("B", GenericValue(list, [TypedValue(int)]), B)
            )

    @assert_passes()
    def test_literal(self):
        from typing_extensions import Literal

        def capybara(x: Literal[True], y: Literal[True, False]) -> None:
            assert_type(x, Literal[True])
            assert_type(y, Literal[True, False])

    @assert_passes()
    def test_literal_in_union(self):
        from typing import Union

        try:
            from typing import Literal
        except ImportError:
            from typing_extensions import Literal

        def capybara(x: Union[int, Literal["epoch"], None]) -> None:
            assert_type(x, int | Literal["epoch"] | None)

    @assert_passes()
    def test_contextmanager(self):
        from collections.abc import Generator
        from contextlib import contextmanager

        @contextmanager
        def capybara() -> Generator[int]:
            yield 3

        def kerodon():
            assert_is_value(
                capybara(),
                GenericValue("contextlib._GeneratorContextManager", [TypedValue(int)]),
            )

            with capybara() as e:
                assert_type(e, int)

            assert_is_value(
                post_capybara(),
                GenericValue("contextlib._GeneratorContextManager", [TypedValue(int)]),
            )

            with post_capybara() as e:
                assert_type(e, int)

        @contextmanager
        def post_capybara() -> Generator[int]:
            yield 3

    @assert_passes()
    def test_contextmanager_class(self):
        import sys
        from typing import ContextManager

        def f() -> ContextManager[int]:
            raise NotImplementedError

        if sys.version_info >= (3, 13):
            expected_args = [TypedValue(int), TypedValue(bool) | KnownValue(None)]
        else:
            expected_args = [TypedValue(int)]

        def capybara():
            assert_is_value(
                f(), GenericValue("contextlib.AbstractContextManager", expected_args)
            )
            with f() as x:
                assert_type(x, int)

    @assert_passes()
    def test_none_annotations(self):
        def mara() -> None:
            pass

        class Capybara:
            def __init__(self) -> None:
                pass

        def check() -> None:
            # Make sure we don't infer None if __init__ is annotated
            # as returning None.
            assert_type(Capybara(), Capybara)
            assert_type(mara(), None)

    @assert_passes()
    def test_annotations_function(self):
        def caviidae() -> None:
            x = int

            # tests that annotations in a nested functions are not evaluated in a context where
            # they don't exist
            def capybara(a: x, *b: x, c: x, d: x = 3, **kwargs: x):
                pass

            capybara(1, c=1)
            capybara(1, c="x")  # E: incompatible_argument

    @assert_passes()
    def annotations_class(self):
        class Caviidae:
            class Capybara:
                pass

            def eat(self, x: Capybara):
                assert_type(self, Caviidae)

            @staticmethod
            def static(x: "Caviidae"):
                assert_type(x, Caviidae)

    @assert_passes()
    def test_incompatible_annotations(self):
        def capybara(x: int) -> None:
            pass

        def kerodon():
            capybara("not an int")  # E: incompatible_argument

    @assert_passes()
    def test_incompatible_return_value(self):
        def capybara() -> int:
            return "not an int"  # E: incompatible_return_value

    @assert_passes()
    def test_incompatible_return_value_none(self):
        def capybara(x: bool) -> int:
            if not x:
                return  # E: incompatible_return_value
            return 42

    @assert_passes()
    def test_generator(self):
        from typing import Generator

        def capybara(x: bool) -> Generator[int, None, None]:
            if not x:
                return
            yield 42

    @assert_passes()
    def test_incompatible_return_value_pass(self):
        def f() -> int:  # E: missing_return
            pass

    @assert_passes()
    def test_allow_pass_in_abstractmethod(self):
        from abc import abstractmethod

        class X:
            @abstractmethod
            def f(self) -> int:
                pass

    @assert_passes()
    def test_no_return_none(self):
        def f() -> None:
            # TODO this should really be unannotated. The Any comes from _visit_function_body
            # but I'm not sure why we use that one.
            assert_is_value(g(), AnyValue(AnySource.inference))
            return g()  # E: incompatible_return_value

        def g():
            pass

    @assert_passes()
    def test_incompatible_default(self):
        def capybara(x: int = None) -> None:  # E: incompatible_default
            pass

    @assert_passes()
    def test_property(self):
        class Capybara:
            def __init__(self, x):
                self.x = x

            @property
            def f(self) -> int:
                return self.x

            def get_g(self) -> int:
                return self.x * 2

            g = property(get_g)

        def user(c: Capybara) -> None:
            assert_type(c.f, int)
            assert_type(c.get_g(), int)
            assert_type(c.g, int)

    @assert_passes()
    def test_annotations_override_return(self):
        from typing import Any

        from typing_extensions import Literal

        def f() -> Any:
            return 0

        def g():
            return 0

        def capybara():
            assert_is_value(f(), AnyValue(AnySource.explicit))
            assert_type(g(), Literal[0])

    @assert_passes()
    def test_cached_classmethod(self):
        # just test that this doesn't crash
        from functools import lru_cache

        class Capybara:
            @classmethod
            @lru_cache()
            def f(cls) -> int:
                return 3

    @assert_passes()
    def test_annassign(self):
        def capybara(y):
            x: int = y
            assert_is_value(y, AnyValue(AnySource.unannotated))
            assert_type(x, int)

    @assert_passes()
    def test_incompatible_annassign(self):
        def capybara(y: str):
            x: int = y  # E: incompatible_assignment
            print(x)

    @assert_passes()
    def test_typing_tuples(self):
        from typing import Tuple, Union

        def capybara(
            x: Tuple[int, ...],
            y: Tuple[int],
            z: Tuple[str, int],
            omega: Union[Tuple[str, int], None],
            empty: Tuple[()],
        ) -> None:
            assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(y, make_simple_sequence(tuple, [TypedValue(int)]))
            t_str_int = make_simple_sequence(tuple, [TypedValue(str), TypedValue(int)])
            assert_is_value(z, t_str_int)
            assert_type(omega, tuple[str, int] | None)
            assert_is_value(empty, SequenceValue(tuple, []))

    @assert_passes()
    def test_stringified_tuples(self):
        from typing import Tuple, Union

        def capybara(
            x: "Tuple[int, ...]",
            y: "Tuple[int]",
            z: "Tuple[str, int]",
            omega: "Union[Tuple[str, int], None]",
            empty: "Tuple[()]",
        ) -> None:
            assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(y, make_simple_sequence(tuple, [TypedValue(int)]))
            t_str_int = make_simple_sequence(tuple, [TypedValue(str), TypedValue(int)])
            assert_is_value(z, t_str_int)
            assert_type(omega, tuple[str, int] | None)
            assert_is_value(empty, SequenceValue(tuple, []))

    @assert_passes()
    def test_builtin_tuples(self):
        from collections.abc import Iterable
        from typing import Union

        def returner() -> Iterable[tuple[str, int]]:
            yield ("a", 1)

        def capybara(
            x: tuple[int, ...],
            y: tuple[int],
            z: tuple[str, int],
            omega: Union[tuple[str, int], None],
            empty: tuple[()],
            kappa: Iterable[tuple[str, int]],
        ) -> None:
            assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
            assert_is_value(y, make_simple_sequence(tuple, [TypedValue(int)]))
            t_str_int = make_simple_sequence(tuple, [TypedValue(str), TypedValue(int)])
            assert_is_value(z, t_str_int)
            assert_type(omega, tuple[str, int] | None)
            assert_is_value(empty, SequenceValue(tuple, []))
            for t in kappa:
                assert_is_value(t, t_str_int)
            for elt in returner():
                assert_is_value(elt, t_str_int)

    def test_builtin_tuples_string(self):
        self.assert_passes("""
            from __future__ import annotations
            from collections.abc import Iterable
            from typing import Union

            def returner() -> Iterable[tuple[str, int]]:
                yield ("a", 1)

            def capybara(
                x: tuple[int, ...],
                y: tuple[int],
                z: tuple[str, int],
                omega: Union[tuple[str, int], None],
                empty: tuple[()],
                kappa: Iterable[tuple[str, int]],
            ) -> None:
                assert_is_value(x, GenericValue(tuple, [TypedValue(int)]))
                assert_is_value(y, make_simple_sequence(tuple, [TypedValue(int)]))
                t_str_int = make_simple_sequence(tuple, [TypedValue(str), TypedValue(int)])
                assert_is_value(z, t_str_int)
                assert_type(omega, tuple[str, int] | None)
                assert_is_value(empty, SequenceValue(tuple, []))
                for t in kappa:
                    assert_is_value(t, t_str_int)
                for elt in returner():
                    assert_is_value(elt, t_str_int)
            """)

    @assert_passes()
    def test_invalid_annotation(self):
        def not_an_annotation(x: 1):  # E: invalid_annotation
            pass

        def invalid_type_argument(x: type[1]):  # E: invalid_annotation
            pass

        def forward_ref_undefined(x: "NoSuchType"):  # E: undefined_name
            pass

        def forward_ref_bad_attribute(
            x: "collections.defalutdict",  # E: undefined_name
        ):
            pass

        def test_typed_value_annotation() -> dict():  # E: invalid_annotation
            return {}

    def test_forward_ref_invalid_listcomp(self):
        from .annotations import Context, type_from_runtime

        val = type_from_runtime("[int for i in range(1)]", ctx=Context())
        assert isinstance(val, AnyValue)

    def test_forward_ref_invalid_unusual_expressions(self):
        from .annotations import Context, type_from_runtime

        for annotation in (
            "[int for i in range(1)]",
            "{int for i in range(1)}",
            "{i: int for i in range(1)}",
            "(int for i in range(1))",
            "int if True else str",
            "int or str",
            "(lambda: int)()",
            'f"int"',
        ):
            val = type_from_runtime(annotation, ctx=Context())
            assert isinstance(val, AnyValue), annotation

    @skip_before((3, 11))
    def test_type_from_value_preserves_empty_typevartuple_specialization(self):
        from typing import Generic, TypeVarTuple

        from .annotations import type_from_value

        Ts = TypeVarTuple("Ts")
        namespace = {"Generic": Generic, "Ts": Ts}
        exec("class Array(Generic[*Ts]):\n    pass", namespace)
        Array = namespace["Array"]

        assert type_from_value(KnownValue(Array[()])) == GenericValue(
            Array, [SequenceValue(tuple, [])]
        )

    @skip_before((3, 12))
    def test_type_from_runtime_preserves_runtime_paramspec_specialization(self):
        from .annotations import Context, type_from_runtime

        namespace: dict[str, object] = {}
        exec("class Callback[**P]:\n    pass", namespace)
        Callback = namespace["Callback"]

        assert type_from_runtime(Callback[[int, str]], ctx=Context()) == GenericValue(
            Callback,
            [
                SequenceValue(
                    tuple, [(False, TypedValue(int)), (False, TypedValue(str))]
                )
            ],
        )

    @assert_passes()
    def test_runtime_initvar_requires_single_argument(self):
        from dataclasses import InitVar, dataclass

        BadInitVar = InitVar[int, str]

        @dataclass
        class C:
            x: BadInitVar  # E: invalid_annotation

    @assert_passes()
    def test_forward_ref_errors_keep_annotation_lineno(self):
        def invalid_annotations(
            p1: "[int, str]",  # E: invalid_annotation
            p2: "(lambda : int)()",  # E: invalid_annotation
            p3: "1",  # E: invalid_annotation
        ) -> None:
            pass

    @assert_passes()
    def test_multiline_string_forward_ref(self):
        from typing_extensions import assert_type

        def f(x: """
                int |
                str |
                list[int]
            """) -> None:
            assert_type(x, int | str | list[int])

        f(1)
        f("x")
        f([1])
        f(1.0)  # E: incompatible_argument

    @assert_passes()
    def test_forward_ref_optional(self):
        import typing
        from typing import Optional

        def capybara(x: "X", y: "Optional[X]", z: "typing.Optional[X]"):
            assert_type(x, X)
            assert_type(y, X | None)
            assert_type(z, X | None)

        class X:
            pass

    @assert_passes()
    def test_forward_ref_list(self):
        from typing import List

        def capybara(x: "List[int]") -> "List[str]":
            assert_type(x, list[int])
            assert_type(capybara(x), list[str])
            return []

    @assert_passes()
    def test_forward_ref_incompatible(self):
        def f() -> "int":
            return ""  # E: incompatible_return_value

    @assert_passes()
    def test_pattern(self):
        from typing import Pattern

        def capybara(x: Pattern[str]):
            assert_type(x, Pattern[str])

    def test_future_annotations(self):
        self.assert_passes("""
            from __future__ import annotations
            from typing import List

            def f(x: int, y: List[str]):
                assert_type(x, int)
                assert_type(y, list[str])
            """)

    @assert_passes()
    def test_final(self):
        from typing_extensions import Final, Literal

        x: Final = 3

        class Mara:
            x: Final[str] = "x"

        def capybara():
            y: Final = 4
            assert_type(x, Literal[3])
            assert_type(y, Literal[4])

            z: Final[int] = 4
            assert_type(z, int)

            assert_type(Mara().x, str)

    @assert_passes()
    def test_type(self):
        from typing import Type

        def capybara(x: Type[str], y: "Type[int]"):
            assert_type(x, type[str])
            assert_type(y, type[int])

    @assert_passes()
    def test_lowercase_type(self):
        def capybara(x: type[str], y: "type[int]"):
            assert_type(x, type[str])
            assert_type(y, type[int])

    @assert_passes()
    def test_type_none(self):
        def capybara(x: type[None]):
            pass

        capybara(type(None))
        capybara(None.__class__)
        capybara(None)  # E: incompatible_argument

    @assert_passes()
    def test_generic_alias(self):
        from queue import Queue

        class I: ...

        class X:
            def __init__(self):
                self.q: Queue[I] = Queue()

        def f(x: Queue[I]) -> None:
            assert_type(x, Queue[I])

        def capybara(x: list[int], y: tuple[int, str], z: tuple[int, ...]) -> None:
            assert_type(x, list[int])
            assert_type(y, tuple[int, str])
            assert_type(z, tuple[int, ...])

    def test_pep604(self):
        self.assert_passes("""
            from __future__ import annotations

            def capybara(x: int | None, y: int | str) -> None:
                assert_type(x, int | None)
                assert_type(y, int | str)

            def caller():
                capybara(1, 2)
                capybara(None, "x")
            """)

    @assert_passes()
    def test_pep604_runtime(self):
        def capybara(x: int | None, y: int | str) -> None:
            assert_type(x, int | None)
            assert_type(y, int | str)

        def caller():
            capybara(1, 2)
            capybara(None, "x")

    @assert_passes()
    def test_stringified_ops(self):
        from typing_extensions import Literal

        def capybara(x: "int | str", y: "Literal[-1]"):
            assert_type(x, int | str)
            assert_type(y, Literal[-1])

    @assert_passes()
    def test_double_subscript(self):
        from typing import List, Set, TypeVar, Union

        T = TypeVar("T")

        # A bit weird but we hit this kind of case with generic
        # aliases in typeshed.
        def capybara(x: "Union[List[T], Set[T]][int]"):
            assert_type(x, list[int] | set[int])

    @assert_passes()
    def test_initvar(self):
        from dataclasses import InitVar, dataclass

        @dataclass
        class Capybara:
            x: InitVar[str]

        def f():
            Capybara(x=3)  # E: incompatible_argument

    @assert_passes()
    def test_classvar(self):
        from typing import ClassVar

        class Capybara:
            x: ClassVar[str]
            Alias: ClassVar = int

            y: Alias

        def caller(c: Capybara):
            assert_type(c.x, str)
            assert_type(c.y, int)
            assert_is_value(c.Alias, KnownValue(int))


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing import Iterable, Optional

        from typing_extensions import Annotated, assert_type

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
            in_optional: Optional[Annotated[int, 1]],
            in_iterable: Iterable[Annotated[int, 1]],
        ) -> None:
            assert_type(x, int)
            assert_type(y, int)
            assert_type(quoted, int)
            assert_type(nested, int)
            assert_type(nested_quoted, int)
            assert_type(in_optional, int | None)
            assert_type(in_iterable, Iterable[int])

    @assert_passes()
    def test_typing(self):
        from typing import Annotated, Iterable, Optional

        obj = object()

        def capybara(
            x: Annotated[int, "stuff"],
            y: Annotated[int, obj],
            quoted: "Annotated[int, int, str]",
            nested: Annotated[Annotated[int, 1], 2],
            nested_quoted: "Annotated[Annotated[int, 1], 2]",
            in_optional: Optional[Annotated[int, 1]],
            in_iterable: Iterable[Annotated[int, 1]],
        ) -> None:
            assert_type(x, int)
            assert_type(y, int)
            assert_type(quoted, int)
            assert_type(nested, int)
            assert_type(nested_quoted, int)
            assert_type(in_optional, int | None)
            assert_type(in_iterable, Iterable[int])

    @assert_passes()
    def test_genericalias_nested_class(self):
        from typing_extensions import assert_type

        def capybara():
            class Base:
                pass

            class Child:
                attrs: list[Base]

            assert_type(Child().attrs, list[Base])

    @assert_passes()
    def test_annotated_requires_metadata_and_is_not_callable(self):
        from typing import Annotated, TypeAlias

        Alias: TypeAlias = Annotated[int, "meta"]
        _bad: "Annotated[int]"  # E: invalid_annotation

        def capybara() -> None:
            Annotated()  # E: invalid_annotation
            Annotated[int, "meta"]()  # E: invalid_annotation
            Alias(1)  # E: invalid_annotation


class TestCallable(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Callable, Sequence, TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def capybara(
            x: Callable[..., int],
            y: Callable[[int], str],
            id_func: Callable[[T], T],
            takes_seq: Callable[[Sequence[T]], T],
            two_args: Callable[[int, str], bytes],
        ):
            assert_type(x(), int)
            assert_type(x(arg=3), int)
            assert_type(y(1), str)
            assert_type(id_func(1), Literal[1])
            assert_type(takes_seq([int("1")]), int)
            assert_type(two_args(1, "x"), bytes)

    @assert_passes()
    def test_stringified(self):
        from typing import Callable, Sequence, TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def capybara(
            x: "Callable[..., int]",
            y: "Callable[[int], str]",
            id_func: "Callable[[T], T]",
            takes_seq: "Callable[[Sequence[T]], T]",
            two_args: "Callable[[int, str], bytes]",
        ):
            assert_type(x(), int)
            assert_type(x(arg=3), int)
            assert_type(y(1), str)
            assert_type(id_func(1), Literal[1])
            assert_type(takes_seq([int("1")]), int)
            assert_type(two_args(1, "x"), bytes)

    @assert_passes()
    def test_invalid_callable_annotations(self):
        from typing import Callable

        def capybara() -> None:
            _v1: Callable[int]  # E: invalid_annotation
            _v2: Callable[int, int]  # E: invalid_annotation
            _v3: Callable[[], [int]]  # E: invalid_annotation
            _v4: Callable[int, int, int]  # E: invalid_annotation
            _v5: Callable[[...], int]  # E: invalid_annotation

    @assert_passes()
    def test_runtime_callable_requires_two_arguments(self):
        from typing import Callable

        class FakeMalformedCallable:
            __origin__ = Callable
            __args__ = (int,)

        BadCallable = FakeMalformedCallable()

        def f(x: BadCallable) -> None:  # E: invalid_annotation
            pass

    @assert_passes()
    def test_abc_callable(self):
        from collections.abc import Callable, Sequence
        from typing import TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def capybara(
            x: Callable[..., int],
            y: Callable[[int], str],
            id_func: Callable[[T], T],
            takes_seq: Callable[[Sequence[T]], T],
            two_args: Callable[[int, str], bytes],
        ):
            assert_type(x(), int)
            assert_type(x(arg=3), int)
            assert_type(y(1), str)
            assert_type(id_func(1), Literal[1])
            assert_type(takes_seq([int("1")]), int)
            assert_type(two_args(1, "x"), bytes)

    @assert_passes()
    def test_known_value(self):
        from typing import Any

        from typing_extensions import Literal

        class Capybara:
            def method(self, x: int) -> int:
                return 42

        def f(x: int) -> int:
            return 0

        def g(func: Literal[f]) -> None:
            pass

        def h(x: object) -> bool:
            return True

        def decorator(func: Any) -> Any:
            return func

        def capybara() -> None:
            def nested(x: int) -> int:
                return 2

            @decorator
            def decorated(x: int) -> int:
                return 2

            g(f)
            g(h)
            g(Capybara().method)
            g(nested)
            g(decorated)

    @assert_passes()
    def test_wrong_callable(self):
        from typing import Callable

        def takes_callable(x: Callable[[int], str]) -> None:
            pass

        def wrong_callable(x: str) -> int:
            return 0

        def capybara() -> None:
            takes_callable(wrong_callable)  # E: incompatible_argument

    @assert_passes()
    def test_invalid_literal_check_disabled_by_default(self):
        from enum import Enum

        from typing_extensions import Literal

        class Animal(Enum):
            CAT = 1
            helper = lambda x: str(x)

            @property
            def species(self) -> str:
                return "mammal"

        def accepts(
            member: Literal[Animal.CAT],
            helper: Literal[Animal.helper],
            species: Literal[Animal.species],
            obj_type: Literal[object],
            pi: Literal[3.14],
        ) -> None:
            pass

        def capybara() -> None:
            accepts(Animal.CAT, Animal.helper, Animal.species, object, 3.14)

    @assert_passes(settings={ErrorCode.invalid_literal: True})
    def test_invalid_literal_check_enabled(self):
        from enum import Enum

        from typing_extensions import Literal

        class Animal(Enum):
            CAT = 1
            helper = lambda x: str(x)

            @property
            def species(self) -> str:
                return "mammal"

        def accepts(
            member: Literal[Animal.CAT],
            helper: Literal[Animal.helper],  # E: invalid_literal
            species: Literal[Animal.species],  # E: invalid_literal
            obj_type: Literal[object],  # E: invalid_literal
            pi: Literal[3.14],  # E: invalid_literal
        ) -> None:
            pass

    @assert_passes()
    def test_known_value_error(self):
        from typing_extensions import Literal

        def f(x: int) -> int:
            return 0

        def g(func: Literal[f]) -> None:
            pass

        def h(x: bool) -> bool:
            return True

        def capybara() -> None:
            g(h)  # E: incompatible_argument

    @assert_passes()
    def test_asynq_callable_incompatible(self):
        from typing import Callable

        from pycroscope.extensions import AsynqCallable

        def f(x: AsynqCallable[[], int]) -> None:
            pass

        def capybara(func: Callable[[], int]) -> None:
            f(func)  # E: incompatible_argument

    @assert_passes()
    def test_asynq_callable_incompatible_literal(self):
        from pycroscope.extensions import AsynqCallable

        def f(x: AsynqCallable[[], int]) -> None:
            pass

        def func() -> int:
            return 0

        def capybara() -> None:
            f(func)  # E: incompatible_argument

    @assert_passes()
    def test_overlapping_annotation(self):
        from pycroscope.extensions import Overlapping

        def f(x: Overlapping[int]) -> None:
            pass

        def bare(x: Overlapping) -> None:  # E: invalid_annotation
            pass

        def capybara(i: int, s: str) -> None:
            f(i)
            f(s)  # E: incompatible_argument
            bare(i)

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_asynq_callable(self):
        from typing import Optional

        from asynq import asynq

        from pycroscope.extensions import AsynqCallable
        from pycroscope.signature import ELLIPSIS_PARAM, Signature

        @asynq()
        def func_example(x: int) -> str:
            return ""

        sig = Signature.make([ELLIPSIS_PARAM], is_asynq=True)

        @asynq()
        def bare_asynq_callable(fn: AsynqCallable) -> None:
            assert_is_value(fn, CallableValue(sig))
            yield fn.asynq()
            yield fn.asynq("some", "arguments")

        @asynq()
        def caller(
            func: AsynqCallable[[int], str],
            func2: Optional[AsynqCallable[[int], str]] = None,
        ) -> None:
            assert_type(func(1), str)
            val = yield func.asynq(1)
            assert_type(val, str)
            yield caller.asynq(func_example)
            if func2 is not None:
                yield func2.asynq(1)
            yield bare_asynq_callable.asynq(func_example)

    @skip_if_not_installed("asynq")
    @assert_passes(settings={ErrorCode.impure_async_call: False})
    def test_amap(self):
        from typing import Iterable, List, TypeVar

        from asynq import asynq

        from pycroscope.extensions import AsynqCallable

        T = TypeVar("T")
        U = TypeVar("U")

        @asynq()
        def amap(function: AsynqCallable[[T], U], sequence: Iterable[T]) -> List[U]:
            return (yield [function.asynq(elt) for elt in sequence])

        @asynq()
        def mapper(x: int) -> str:
            return ""

        @asynq()
        def caller():
            assert_type(amap(mapper, [1]), list[str])
            assert_type((yield amap.asynq(mapper, [1])), list[str])

    @assert_passes()
    def test_bare_callable(self):
        import collections.abc
        import typing

        def want_typing(c: typing.Callable) -> None:
            pass

        def want_abc(c: collections.abc.Callable) -> None:
            pass

        class MyCallable:
            def __call__(self) -> None:
                pass

        def capybara(m: MyCallable) -> None:
            want_typing(m)
            want_abc(m)
            want_typing(1)  # E: incompatible_argument
            want_abc(1)  # E: incompatible_argument

            from _pycroscope_tests.callable import StubCallable

            def inner(x: StubCallable) -> None:
                want_typing(x)
                want_abc(x)


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_bound(self):
        from typing import TypeVar

        from typing_extensions import Literal

        IntT = TypeVar("IntT", bound=int)
        expected = TypeVarValue(TypeVarParam(IntT, bound=TypedValue(int)))

        def f(x: IntT) -> IntT:
            assert_is_value(x, expected)
            print(x + 1)
            return x

        def capybara():
            assert_type(f(1), Literal[1])
            assert_type(f(True), Literal[True])
            x = f("")  # E: incompatible_argument
            assert_is_value(x, AnyValue(AnySource.error))

    @assert_passes()
    def test_bound_string_forward_ref(self):
        from typing import TypeVar

        from typing_extensions import Literal

        T = TypeVar("T", bound="ForwardRef | str")

        class ForwardRef:
            pass

        def f(x: T) -> T:
            return x

        def capybara() -> None:
            assert_type(f("x"), Literal["x"])

    @assert_passes()
    def test_bound_cannot_contain_typevars(self):
        from typing import TypeVar

        T = TypeVar("T")
        TypeVar("Bad", bound=list[T])  # E: invalid_annotation

    @assert_passes()
    def test_constraint(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", bytes, str)

        def whatever(x: Union[str, bytes]):
            pass

        def f(x: AnyStr) -> AnyStr:
            print(x.title())
            whatever(x)
            return x

        def capybara(s: str, b: bytes, sb: Union[str, bytes], unannotated):
            assert_type(f("x"), str)
            assert_type(f(b"x"), bytes)
            assert_type(f(s), str)
            assert_type(f(b), bytes)
            result = f(sb)
            assert_type(result, str | bytes)
            f(3)  # E: incompatible_argument
            assert_is_value(f(unannotated), AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_constraint_in_typeshed(self):
        import re

        def capybara():
            assert_type(re.escape("x"), str)

    @assert_passes()
    def test_callable_compatibility(self):
        from typing import Callable, Iterable, TypeVar, Union

        from typing_extensions import Literal, Protocol

        AnyStr = TypeVar("AnyStr", bytes, str)
        IntT = TypeVar("IntT", bound=int)

        class SupportsInt(Protocol):
            def __int__(self) -> int:
                raise NotImplementedError

        SupportsIntT = TypeVar("SupportsIntT", bound=SupportsInt)

        def find_int(objs: Iterable[SupportsIntT]) -> SupportsIntT:
            for obj in objs:
                if obj.__int__() == obj:
                    return obj
            raise ValueError

        def wants_float_func(f: Callable[[Iterable[float]], float]) -> float:
            return f([1.0, 2.0])

        def wants_int_func(f: Callable[[Iterable[int]], int]) -> int:
            return f([1, 2])

        def want_anystr_func(
            f: Callable[[AnyStr], AnyStr], s: Union[str, bytes]
        ) -> str:
            if isinstance(s, str):
                assert_type(f(s), str)
            else:
                assert_type(f(s), bytes)
            return ""

        def want_bounded_func(f: Callable[[IntT], IntT], i: int) -> None:
            assert_type(f(True), Literal[True])
            assert_type(f(i), int)

        def want_str_func(f: Callable[[str], str]):
            assert_type(f("x"), str)

        def anystr_func(s: AnyStr) -> AnyStr:
            return s

        def int_func(i: IntT) -> IntT:
            return i

        def capybara():
            want_anystr_func(anystr_func, "x")
            want_anystr_func(int_func, "x")  # E: incompatible_argument
            want_bounded_func(int_func, 0)
            want_bounded_func(anystr_func, 1)  # E: incompatible_argument
            want_str_func(anystr_func)
            want_str_func(int_func)  # E: incompatible_argument
            assert_is_value(find_int([1.0, 2.0]), KnownValue(1.0) | KnownValue(2.0))
            # TODO this should work with wants_float_func but it doesn't.
            # Some bug with binding TypeVars to a union?
            wants_int_func(find_int)
            wants_float_func(int_func)  # E: incompatible_argument

    @assert_passes()
    def test_getitem(self):
        from typing import Any, Dict, Iterable, TypeVar

        T = TypeVar("T", bound=Dict[str, Any])

        def _fetch_credentials(api: T, credential_names: Iterable[str]) -> T:
            api_with_credentials = api.copy()
            for name in credential_names:
                api_with_credentials[name] = str(api[name])
            return api_with_credentials


class TestParameterTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing_extensions import Annotated

        from pycroscope.extensions import ParameterTypeGuard

        def is_int(x: object) -> Annotated[bool, ParameterTypeGuard["x", int]]:
            return isinstance(x, int)

        def capybara(x: object) -> None:
            assert_type(x, object)
            if is_int(x):
                assert_type(x, int)

    @assert_passes()
    def test_generic(self):
        import collections.abc
        from typing import Iterable, Type, TypeVar, Union

        from typing_extensions import Annotated

        from pycroscope.extensions import ParameterTypeGuard

        T = TypeVar("T")

        def all_of_type(
            elts: Iterable[object], typ: Type[T]
        ) -> Annotated[bool, ParameterTypeGuard["elts", Iterable[T]]]:
            return all(isinstance(elt, typ) for elt in elts)

        def capybara(elts: Iterable[Union[int, str]]) -> None:
            assert_is_value(
                elts,
                GenericValue(
                    collections.abc.Iterable,
                    [MultiValuedValue([TypedValue(int), TypedValue(str)])],
                ),
            )
            if all_of_type(elts, int):
                assert_is_value(
                    elts, GenericValue(collections.abc.Iterable, [TypedValue(int)])
                )

    @assert_passes()
    def test_self(self):
        from typing import Union

        from typing_extensions import Annotated

        from pycroscope.extensions import ParameterTypeGuard

        class A:
            def is_b(self) -> Annotated[bool, ParameterTypeGuard["self", "B"]]:
                return isinstance(self, B)

        class B(A):
            pass

        class C(A):
            pass

        def capybara(obj: A) -> None:
            assert_type(obj, A)
            if obj.is_b():
                assert_type(obj, B)
            else:
                assert_type(obj, A)

        def narrow_union(union: Union[B, C]) -> None:
            assert_type(union, B | C)
            if union.is_b():
                assert_type(union, B)


class TestNoReturnGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing_extensions import Annotated

        from pycroscope.extensions import NoReturnGuard

        def assert_is_int(x: object) -> Annotated[None, NoReturnGuard["x", int]]:
            assert isinstance(x, int)

        def capybara(x):
            assert_is_int(x)
            assert_type(x, int)


class TestTypeGuard(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extesions(self):
        from typing import Union

        from typing_extensions import TypeGuard, assert_type

        def is_int(x: Union[int, str]) -> TypeGuard[int]:
            return x == 42

        def is_quoted_int(x: Union[int, str]) -> "TypeGuard[int]":
            return x == 42

        def capybara(x: Union[int, str]):
            if is_int(x):
                assert_type(x, int)
            else:
                assert_type(x, int | str)

        def pacarana(x: Union[int, str]):
            if is_quoted_int(x):
                assert_type(x, int)
            else:
                assert_type(x, int | str)

    @assert_passes()
    def test(self):
        from typing import Union

        from typing_extensions import assert_type

        from pycroscope.extensions import TypeGuard

        def is_int(x: Union[int, str]) -> TypeGuard[int]:
            return x == 42

        def capybara(x: Union[int, str]):
            if is_int(x):
                assert_type(x, int)
            else:
                assert_type(x, int | str)

    @assert_passes()
    def test_method(self) -> None:
        from typing import Union

        from typing_extensions import assert_type

        from pycroscope.extensions import TypeGuard

        class Cls:
            def is_int(self, x: Union[int, str]) -> TypeGuard[int]:
                return x == 43

        def capybara(x: Union[int, str]):
            cls = Cls()
            if cls.is_int(x):
                assert_type(x, int)
                assert_type(cls, Cls)
            else:
                assert_type(x, int | str)
                assert_type(cls, Cls)

    @assert_passes()
    def test_staticmethod(self) -> None:
        from typing import Union

        from typing_extensions import TypeGuard, assert_type

        class Cls:
            @staticmethod
            def is_int(x: Union[int, str]) -> TypeGuard[int]:
                return x == 43

        def capybara(x: Union[int, str]):
            if Cls().is_int(x):
                assert_type(x, int)
            else:
                assert_type(x, int | str)
            assert_type(x, int | str)
            if Cls.is_int(x):
                assert_type(x, int)

    @assert_passes()
    def test_callable_compatibility(self) -> None:
        from typing import Callable

        from typing_extensions import TypeGuard, TypeIs

        def is_int(x: object) -> TypeGuard[int]:
            return isinstance(x, int)

        def is_str(x: object) -> TypeGuard[str]:
            return isinstance(x, str)

        def is_typeis_int(x: object) -> TypeIs[int]:
            return isinstance(x, int)

        def needs_object_guard(f: Callable[[object], TypeGuard[object]]) -> None:
            pass

        def needs_int_guard(f: Callable[[object], TypeGuard[int]]) -> None:
            pass

        def capybara() -> None:
            needs_object_guard(is_int)
            needs_int_guard(is_int)
            needs_int_guard(is_str)  # E: incompatible_argument
            needs_int_guard(is_typeis_int)  # E: incompatible_argument


class TestCustomCheck(TestNameCheckVisitorBase):
    @assert_passes()
    def test_literal_only(self) -> None:
        from typing_extensions import Annotated

        from pycroscope.extensions import LiteralOnly

        def capybara(x: Annotated[str, LiteralOnly()]) -> str:
            return x

        def caller(x: str) -> None:
            capybara("x")
            capybara(x)  # E: incompatible_argument
            capybara(str(1))  # E: incompatible_argument
            capybara("x" if x else "y")
            capybara("x" if x else x)  # E: incompatible_argument

    @assert_passes()
    def test_reverse_direction(self):
        from typing import Any

        from typing_extensions import Annotated

        from pycroscope.extensions import CustomCheck
        from pycroscope.value import (
            CanAssign,
            CanAssignContext,
            CanAssignError,
            Value,
            flatten_values,
        )

        class DontAssignToAny(CustomCheck):
            def can_be_assigned(self, value: Value, ctx: CanAssignContext) -> CanAssign:
                for val in flatten_values(value, unwrap_annotated=True):
                    if isinstance(val, AnyValue):
                        return CanAssignError("Assignment to Any disallowed")
                return {}

        def want_any(x: Any) -> None:
            pass

        def capybara(arg: Annotated[str, DontAssignToAny()]) -> None:
            want_any(arg)  # E: incompatible_argument
            print(len(arg))

    @assert_passes()
    def test_no_any(self) -> None:
        from typing import List

        from typing_extensions import Annotated

        from pycroscope.extensions import NoAny

        def shallow(x: Annotated[List[int], NoAny()]) -> None:
            pass

        def deep(x: Annotated[List[int], NoAny(deep=True)]) -> None:
            pass

        def none_at_all(
            x: Annotated[List[int], NoAny(deep=True, allowed_sources=frozenset())],
        ) -> None:
            pass

        def capybara(unannotated) -> None:
            shallow(unannotated)  # E: incompatible_argument
            shallow([1])
            shallow([int(unannotated)])
            shallow([unannotated])
            deep(unannotated)  # E: incompatible_argument
            deep([1])
            deep([int(unannotated)])
            deep([unannotated])  # E: incompatible_argument
            none_at_all(unannotated)  # E: incompatible_argument
            none_at_all([1])
            none_at_all([int(unannotated)])
            none_at_all([unannotated])  # E: incompatible_argument

            lst = []
            for x in lst:
                assert_is_value(x, MultiValuedValue([]))
                shallow(x)
                deep(x)
                none_at_all(x)

    @assert_passes()
    def test_not_none(self) -> None:
        from dataclasses import dataclass
        from typing import Any, Optional

        from typing_extensions import Annotated

        from pycroscope.extensions import CustomCheck
        from pycroscope.value import (
            CanAssign,
            CanAssignContext,
            CanAssignError,
            KnownValue,
            Value,
            flatten_values,
        )

        @dataclass(frozen=True)
        class IsNot(CustomCheck):
            obj: object

            def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
                for subval in flatten_values(value):
                    if isinstance(subval, KnownValue):
                        if subval.val is self.obj:
                            return CanAssignError(f"Value may not be {self.obj!r}")
                return {}

        def capybara(x: Annotated[Any, IsNot(None)]) -> None:
            pass

        def caller(x: Optional[str]) -> None:
            capybara("x")
            capybara(None)  # E: incompatible_argument
            capybara(x)  # E: incompatible_argument
            capybara("x" if x else None)  # E: incompatible_argument

    @assert_passes()
    def test_greater_than(self) -> None:
        from dataclasses import dataclass
        from typing import Iterable, TypeVar, Union

        from typing_extensions import Annotated, TypeGuard

        from pycroscope.extensions import CustomCheck
        from pycroscope.value import (
            CanAssign,
            CanAssignContext,
            CanAssignError,
            KnownValue,
            TypeVarMap,
            TypeVarValue,
            Value,
            flatten_values,
        )

        @dataclass(frozen=True)
        class GreaterThan(CustomCheck):
            value: Union[int, TypeVar]

            def _can_assign_inner(self, value: Value) -> CanAssign:
                if isinstance(value, KnownValue):
                    if not isinstance(value.val, int):
                        return CanAssignError(f"Value {value.val!r} is not an int")
                    if value.val <= self.value:
                        return CanAssignError(
                            f"Value {value.val!r} is not greater than {self.value}"
                        )
                    return {}
                elif isinstance(value, AnyValue):
                    return {}
                else:
                    return CanAssignError(f"Size of {value} is not known")

            def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
                if isinstance(self.value, TypeVar):
                    return {}
                for subval in flatten_values(value, unwrap_annotated=False):
                    if isinstance(subval, AnnotatedValue):
                        can_assign = self._can_assign_inner(subval.value)
                        if not isinstance(can_assign, CanAssignError):
                            return can_assign
                        gts = list(subval.get_custom_check_of_type(GreaterThan))
                        if not gts:
                            return CanAssignError(f"Size of {value} is not known")
                        if not any(
                            check.value >= self.value
                            for check in gts
                            if isinstance(check.value, int)
                        ):
                            return CanAssignError(f"{subval} is too small")
                    else:
                        can_assign = self._can_assign_inner(subval)
                        if isinstance(can_assign, CanAssignError):
                            return can_assign
                return {}

            def walk_values(self) -> Iterable[Value]:
                if isinstance(self.value, TypeVar):
                    yield TypeVarValue(TypeVarParam(self.value))

            def substitute_typevars(self, typevars: TypeVarMap) -> "GreaterThan":
                if isinstance(self.value, TypeVar) and self.value in typevars:
                    value = typevars[self.value]
                    if isinstance(value, KnownValue) and isinstance(value.val, int):
                        return GreaterThan(value.val)
                return self

        def capybara(x: Annotated[int, GreaterThan(2)]) -> None:
            pass

        IntT = TypeVar("IntT", bound=int)

        def is_greater_than(
            x: int, limit: IntT
        ) -> TypeGuard[Annotated[int, GreaterThan(IntT)]]:
            return x > limit

        def caller(x: int) -> None:
            capybara(x)  # E: incompatible_argument
            if is_greater_than(x, 2):
                capybara(x)  # ok
            capybara(3)  # ok
            capybara(2)  # E: incompatible_argument


class TestExternalType(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self) -> None:
        import os
        from typing import Union

        from typing_extensions import Annotated

        from pycroscope.extensions import ExternalType

        def capybara(
            x: ExternalType["builtins.str"],
            y: ExternalType["os.stat_result"],
            z: Annotated[ExternalType["builtins.str"], 1] = "z",
            omega: Union[
                ExternalType["builtins.str"], ExternalType["builtins.int"]
            ] = 1,
        ) -> None:
            assert_type(x, str)
            assert_type(y, os.stat_result)
            assert_type(z, Annotated[str, 1])
            assert_type(omega, str | int)

        def user():
            sr = os.stat_result((1,) * 10)
            capybara("x", 1)  # E: incompatible_argument
            capybara(1, sr)  # E: incompatible_argument
            capybara("x", sr)


class TestRequired(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typing_extensions(self):
        from typing_extensions import NotRequired, Required, TypedDict

        class RNR(TypedDict):
            a: int
            b: Required[str]
            c: NotRequired[bytes]

        def take_rnr(td: RNR) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

        class NotTotal(TypedDict, total=False):
            a: int
            b: Required[str]
            c: NotRequired[bytes]

        def take_not_total(td: NotTotal) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int), required=False),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[bytes]"

        def take_stringify(td: Stringify) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

    @assert_passes()
    def test_typeddict_from_call(self):
        from typing import Any, Optional

        from typing_extensions import NotRequired, Required, TypedDict

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[bytes]"

        def make_td() -> Any:
            return Stringify

        def return_optional() -> Optional[Stringify]:
            return None

        def return_call() -> Optional[make_td()]:
            return None

        def capybara() -> None:
            assert_is_value(
                return_optional(),
                KnownValue(None)
                | TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )
            assert_is_value(
                return_call(),
                KnownValue(None)
                | TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

    @assert_passes()
    def test_typing(self):
        from typing import TypedDict

        from typing_extensions import NotRequired, Required

        class RNR(TypedDict):
            a: int
            b: Required[str]
            c: NotRequired[bytes]

        def take_rnr(td: RNR) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

        class NotTotal(TypedDict, total=False):
            a: int
            b: Required[str]
            c: NotRequired[bytes]

        def take_not_total(td: NotTotal) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int), required=False),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

        class Stringify(TypedDict):
            a: "int"
            b: "Required[str]"
            c: "NotRequired[bytes]"

        def take_stringify(td: Stringify) -> None:
            assert_is_value(
                td,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                        "c": TypedDictEntry(TypedValue(bytes), required=False),
                    }
                ),
            )

    @assert_passes()
    def test_unsupported_location(self):
        from typing_extensions import NotRequired, Required

        def f(x: Required[int]) -> None:  # E: invalid_annotation
            pass

        def g() -> Required[int]:  # E: invalid_annotation
            return 3

        class Capybara:
            x: Required[int]  # E: invalid_annotation
            y: NotRequired[int]  # E: invalid_annotation

    def test_readonly_attribute_annotation_contexts(self):
        self.assert_passes("""
            from __future__ import annotations

            from typing import Annotated, ClassVar

            from typing_extensions import Final, ReadOnly

            class Valid:
                a: ReadOnly[int]
                b: ClassVar[ReadOnly[int]]
                c: ReadOnly[ClassVar[int]]
                d: Annotated[ReadOnly[int], "meta"]
                e: ReadOnly[Annotated[int, "meta"]]

                def __init__(self, value: int):
                    self.f: ReadOnly[int] = value
                    self.f = value + 1

            not_ok: ReadOnly[int] = 1  # E: invalid_annotation

            def bad_param(x: ReadOnly[int]) -> None:  # E: invalid_annotation
                y: ReadOnly[int] = x  # E: invalid_annotation
                print(y)

            class Invalid:
                a: Final[ReadOnly[int]] = 1  # E: invalid_annotation
                b: ReadOnly[Final[int]] = 1  # E: invalid_annotation

            print(Valid, Invalid, bad_param, not_ok)
        """)

    @assert_passes()
    def test_invalid_qualifiers_in_typeddict(self):
        from typing import TYPE_CHECKING, ClassVar, TypedDict

        from typing_extensions import NotRequired, ReadOnly, Required

        if TYPE_CHECKING:

            class TD(TypedDict):
                a: Required[Required[int]]  # E: invalid_annotation
                b: Required[NotRequired[int]]  # E: invalid_annotation
                c: NotRequired[Required[int]]  # E: invalid_annotation
                d: NotRequired[NotRequired[int]]  # E: invalid_annotation
                e: ReadOnly[ReadOnly[int]]  # E: invalid_annotation
                f: ClassVar[int]  # E: invalid_annotation

    @assert_passes()
    def test_invalid_qualifier_arity(self):
        from dataclasses import InitVar
        from typing import TYPE_CHECKING, ClassVar, TypedDict

        from typing_extensions import Final, NotRequired, ReadOnly, Required

        if TYPE_CHECKING:

            class TD(TypedDict):
                req: Required[int, str]  # E: invalid_annotation
                notreq: NotRequired[int, str]  # E: invalid_annotation
                readonly: ReadOnly[int, str]  # E: invalid_annotation

            class C:
                x: ClassVar[int, str] = 1  # E: invalid_annotation
                y: Final[int, str] = 1  # E: invalid_annotation
                z: InitVar[int, str]  # E: invalid_annotation

    @assert_passes()
    def test_duplicate_final_non_typeddict(self):
        from typing import TYPE_CHECKING, Final

        if TYPE_CHECKING:
            x: Final[Final[int]] = 1  # E: invalid_annotation

    @assert_passes()
    def test_dataclass_final_fields(self):
        from dataclasses import dataclass
        from typing import TYPE_CHECKING, ClassVar

        from typing_extensions import Final

        if TYPE_CHECKING:

            @dataclass
            class D:
                final_no_default: Final[int]
                final_with_default: Final[str] = "foo"
                final_classvar: ClassVar[Final[int]] = 4

            D.final_no_default = 10  # E: incompatible_assignment
            D.final_with_default = "baz"  # E: incompatible_assignment
            D.final_classvar = 10  # E: incompatible_assignment

    @skip_before((3, 13))
    @assert_passes()
    def test_dataclass_final_assert_type_not_runtime_polluted(self):
        from dataclasses import dataclass
        from typing import ClassVar, Final, assert_type

        @dataclass
        class D:
            final_no_default: Final[int]
            final_with_default: Final[str] = "foo"
            final_classvar: ClassVar[Final[int]] = 4

        d = D(final_no_default=1, final_with_default="bar")
        assert_type(d.final_no_default, int)
        assert_type(d.final_with_default, str)

        D.final_classvar = 10  # E: incompatible_assignment
        d.final_no_default = 10  # E: incompatible_assignment
        d.final_with_default = "baz"  # E: incompatible_assignment
        D.final_no_default = 10  # E: incompatible_assignment
        D.final_with_default = "baz"  # E: incompatible_assignment

    @assert_passes()
    def test_non_dataclass_still_rejects_classvar_final(self):
        from typing import TYPE_CHECKING, ClassVar

        from typing_extensions import Final

        if TYPE_CHECKING:

            class C:
                x: ClassVar[Final[int]] = 1  # E: invalid_annotation
                y: Final[ClassVar[int]] = 1  # E: invalid_annotation

    @assert_passes()
    def test_classvar_invalid_locations_and_type_params(self):
        from typing import (
            Any,
            Callable,
            ClassVar,
            Generic,
            ParamSpec,
            TypeAlias,
            TypeVar,
            cast,
        )

        T = TypeVar("T")
        P = ParamSpec("P")

        class C(Generic[T, P]):
            bad_t: ClassVar[T] = cast(Any, 0)
            bad_nested: ClassVar[list[T]] = cast(Any, 0)
            bad_paramspec: ClassVar[Callable[P, Any]] = cast(Any, 0)

            def method(self) -> None:
                local: ClassVar[int] = 0  # E: invalid_annotation
                print(local)
                self.attr: ClassVar[int] = 0  # E: invalid_annotation

        outside: ClassVar[int] = 0  # E: invalid_annotation
        Alias: TypeAlias = ClassVar[str]  # E: invalid_annotation

    @assert_passes(settings={ErrorCode.classvar_type_parameters: True})
    def test_classvar_type_parameters_check_enabled(self):
        from typing import Any, Callable, ClassVar, Generic, ParamSpec, TypeVar, cast

        T = TypeVar("T")
        P = ParamSpec("P")

        class C(Generic[T, P]):
            bad_t: ClassVar[T] = cast(Any, 0)  # E: classvar_type_parameters
            bad_nested: ClassVar[list[T]] = cast(Any, 0)  # E: classvar_type_parameters
            # E: classvar_type_parameters
            bad_paramspec: ClassVar[Callable[P, Any]] = cast(Any, 0)


class TestParamSpec(TestNameCheckVisitorBase):
    @assert_passes()
    def test_invalid_annotation_contexts(self):
        from typing_extensions import ParamSpec

        P = ParamSpec("P")

        _x: P  # E: invalid_annotation

        def f(arg: P) -> None:  # E: invalid_annotation
            pass

    @assert_passes()
    def test_additional_invalid_annotation_contexts(self):
        from typing import Callable

        from typing_extensions import ParamSpec

        P = ParamSpec("P")

        _x: list[P]  # E: invalid_annotation
        _y: Callable[[int, str], P]  # E: invalid_annotation

    @assert_passes()
    def test_basic(self):
        from typing import Callable, List, TypeVar

        from typing_extensions import ParamSpec

        P = ParamSpec("P")
        T = TypeVar("T")

        def wrapped(a: int) -> str:
            return str(a)

        def wrapper(c: Callable[P, T]) -> Callable[P, List[T]]:
            raise NotImplementedError

        def quoted_wrapper(c: "Callable[P, T]") -> "Callable[P, List[T]]":
            raise NotImplementedError

        def capybara() -> None:
            assert_type(wrapped(1), str)

            refined = wrapper(wrapped)
            assert_type(refined(1), list[str])
            refined("too", "many", "arguments")  # E: incompatible_call

            quoted_refined = quoted_wrapper(wrapped)
            assert_type(quoted_refined(1), list[str])
            quoted_refined("too", "many", "arguments")  # E: incompatible_call

    @assert_passes()
    def test_concatenate(self):
        from typing import Callable, List, TypeVar

        from typing_extensions import Concatenate, ParamSpec

        P = ParamSpec("P")
        T = TypeVar("T")

        def wrapped(a: int) -> str:
            return str(a)

        def wrapper(c: Callable[P, T]) -> Callable[Concatenate[str, P], List[T]]:
            raise NotImplementedError

        def quoted_wrapper(
            c: "Callable[P, T]",
        ) -> "Callable[Concatenate[str, P], List[T]]":
            raise NotImplementedError

        def capybara() -> None:
            assert_type(wrapped(1), str)

            refined = wrapper(wrapped)
            assert_type(refined("x", 1), list[str])
            refined(1)  # E: incompatible_call

            quoted_refined = quoted_wrapper(wrapped)
            assert_type(quoted_refined("x", 1), list[str])
            quoted_refined(1)  # E: incompatible_call

    @assert_passes()
    def test_match_any(self):
        from typing import Callable, List, TypeVar

        from typing_extensions import ParamSpec

        P = ParamSpec("P")
        T = TypeVar("T")

        def wrapper(c: Callable[P, T]) -> Callable[P, List[T]]:
            raise NotImplementedError

        def capybara(unannotated):
            refined = wrapper(unannotated)
            assert_is_value(
                refined(), GenericValue(list, [AnyValue(AnySource.generic_argument)])
            )

    @assert_passes()
    def test_paramspec_args_kwargs(self):
        from typing import Callable, TypeVar

        from typing_extensions import Concatenate, ParamSpec

        P = ParamSpec("P")
        R = TypeVar("R")

        class Request:
            pass

        def with_request(f: Callable[Concatenate[Request, P], R]) -> Callable[P, R]:
            def inner(*args: P.args, **kwargs: P.kwargs) -> R:
                return f(Request(), *args, **kwargs)

            return inner

        def takes_int_str(request: Request, x: int, y: str) -> int:
            return x + 7

        def capybara():
            func = with_request(takes_int_str)
            func(1, "A")
            func(1, 2)  # E: incompatible_argument

    @assert_passes()
    def test_compatibility(self):
        from typing import Callable, TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def want_callable(c: Callable[[int], T]) -> T:
            return c(1)

        def string_func(s: str) -> str:
            return s

        def capybara(unannotated, other):
            assert_type(want_callable(lambda _: 3), Literal[3])
            assert_is_value(
                want_callable(unannotated), AnyValue(AnySource.generic_argument)
            )
            # generates an AnnotatedValue(MultiValuedValue(...))
            assert_is_value(
                want_callable(unannotated or other),
                AnyValue(AnySource.generic_argument),
            )

            assert_is_value(
                want_callable(string_func),  # E: incompatible_argument
                AnyValue(AnySource.error),
            )
            want_callable(1)  # E: incompatible_argument
            want_callable(int(unannotated))  # E: incompatible_argument

    @assert_passes()
    def test_args_kwargs(self):
        from typing import Callable, TypeVar

        from typing_extensions import Concatenate, ParamSpec

        P = ParamSpec("P")
        R = TypeVar("R")

        class Request:
            pass

        def with_request(f: Callable[Concatenate[Request, P], R]) -> Callable[P, R]:
            def inner(*args: P.args, **kwargs: P.kwargs) -> R:
                return f(Request(), *args, **kwargs)

            return inner

        def takes_int_str(request: Request, x: int, y: str) -> int:
            return x + 7

        def capybara():
            func = with_request(takes_int_str)
            func(1, "A")
            func(1, 2)  # E: incompatible_argument

    @assert_passes()
    def test_apply(self):
        from typing import Callable, TypeVar

        from typing_extensions import ParamSpec, assert_type

        P = ParamSpec("P")
        T = TypeVar("T")

        def apply(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        def sample(x: int) -> str:
            return str(x)

        def capybara() -> None:
            assert_type(apply(sample, 1), str)
            apply(sample, "x")  # E: incompatible_call

    @assert_passes()
    def test_apply_bound_method(self):
        from typing import Callable, TypeVar

        from typing_extensions import ParamSpec, assert_type

        P = ParamSpec("P")
        T = TypeVar("T")

        def apply(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        class X:
            def sample(self, x: int) -> str:
                return str(x)

        def capybara(x: X) -> None:
            assert_type(apply(x.sample, 1), str)
            apply(x.sample, "x")  # E: incompatible_call

    @assert_passes()
    def test_apply_any(self):
        from typing import Any, Callable, TypeVar

        from typing_extensions import ParamSpec, assert_type

        P = ParamSpec("P")
        T = TypeVar("T")

        def apply(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        def capybara(x, args, kwargs) -> None:
            assert_type(apply(x, 1), Any)
            assert_type(apply(x.foo, 1), Any)
            assert_type(apply(x), Any)
            assert_type(apply(x, y=3), Any)
            assert_type(apply(x, *args, **kwargs), Any)

    @assert_passes()
    def test_param_spec_errors(self):
        from typing import Callable, Concatenate, TypeVar

        from typing_extensions import ParamSpec

        P = ParamSpec("P")
        T = TypeVar("T")

        _bad_stored_args: P.args  # E: invalid_annotation
        _bad_stored_kwargs: P.kwargs  # E: invalid_annotation

        def bad_component_mixup(
            *args: P.kwargs,  # E: invalid_annotation
            **kwargs: P.args,  # E: invalid_annotation
        ) -> None:
            pass

        def bad_only_args(*args: P.args) -> None:  # E: invalid_annotation
            pass

        def bad_only_kwargs(**kwargs: P.kwargs) -> None:  # E: invalid_annotation
            pass

        def bad_non_paramspec_kwargs(
            *args: P.args, **kwargs: object  # E: invalid_annotation
        ) -> None:
            pass

        def bad_intervening_param(
            *args: P.args,  # E: invalid_annotation
            x: int,
            **kwargs: P.kwargs,  # E: invalid_annotation
        ) -> None:
            pass

        def apply(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> None:
            func(*args)  # E: incompatible_call
            func(*args, *args, **kwargs)  # E: incompatible_call  # E: incompatible_call
            func(**kwargs)  # E: incompatible_call
            func(*args, **kwargs, **kwargs)  # E: incompatible_call
            func(*args, **kwargs)
            func(1, *args, **kwargs)  # E: incompatible_call

        def apply_with_prefix(
            func: Callable[Concatenate[int, P], T], *args: P.args, **kwargs: P.kwargs
        ) -> None:
            func(1, *args, **kwargs)
            func(*args, 1, **kwargs)  # E: incompatible_call

        def outer(func: Callable[P, None]) -> Callable[P, None]:
            def with_head(x: int, *args: P.args, **kwargs: P.kwargs) -> None:
                func(*args, **kwargs)

            def forward(*args: P.args, **kwargs: P.kwargs) -> None:
                with_head(1, *args, **kwargs)
                with_head(x=1, *args, **kwargs)  # E: incompatible_call

            return forward

    def test_invalid_paramspec_usage_on_overloaded_signature(self):
        overloaded = OverloadedSignature(
            [
                Signature.make(
                    [SigParameter("x", annotation=TypedValue(int))],
                    return_annotation=TypedValue(int),
                ),
                Signature.make(
                    [SigParameter("x", annotation=TypedValue(str))],
                    return_annotation=TypedValue(str),
                ),
            ]
        )
        value = CallableValue(overloaded)
        assert not has_invalid_paramspec_usage(value, can_assign_ctx=None)


class TestTypeAlias(TestNameCheckVisitorBase):
    @assert_passes()
    def test_runtime(self):
        from typing_extensions import TypeAlias

        X: TypeAlias = int
        Y = X
        Z: "TypeAlias" = int

        def capybara(x: X, y: Y, x_quoted: "X", y_quoted: "Y", z: Z) -> None:
            assert_type(x, int)
            assert_type(y, int)
            assert_type(x_quoted, int)
            assert_type(y_quoted, int)
            assert_type(z, int)

    @assert_passes()
    def test_bare_paramspec_is_invalid(self):
        from typing import Callable

        from typing_extensions import ParamSpec, TypeAlias

        P = ParamSpec("P")
        Bad: TypeAlias = P  # E: invalid_annotation
        Good: TypeAlias = Callable[P, int]

    @assert_passes()
    def test_unspecialized_typevar_alias_defaults_to_any(self):
        from typing import Any, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")
        Alias = list[T]

        def capybara(x: Alias) -> None:
            assert_type(x, list[Any])

    @assert_passes()
    def test_generic_alias_constructor_call_preserves_type_args(self):
        from typing_extensions import assert_type

        ListAlias = list
        x = ListAlias[int]()
        assert_type(x, list[int])

    @assert_passes()
    def test_generic_alias_object_constructor_call_preserves_type_args(self):
        from typing_extensions import assert_type

        GenericCtor = list[int]
        x = GenericCtor()
        assert_type(x, list[int])

    @assert_passes()
    def test_explicit_type_alias_generics_and_paramspec(self):
        from typing import Callable, Concatenate, ParamSpec, TypeAlias, TypeVar

        from typing_extensions import assert_type

        P = ParamSpec("P")
        R = TypeVar("R")
        T = TypeVar("T")

        ListAlias: TypeAlias = list
        GenericAlias: TypeAlias = list[T]
        CallableAlias: TypeAlias = Callable[P, None]
        ConcatenateAlias: TypeAlias = Callable[Concatenate[int, P], R]

        def capybara(value: CallableAlias) -> None:
            assert_type(value, Callable[..., None])

        try:
            _bad1: GenericAlias[int, int]  # E: invalid_annotation
            _bad2: ConcatenateAlias[int, int]  # E: invalid_annotation
            _bad3: ListAlias[int]  # E: invalid_annotation
        except TypeError:
            pass

    @assert_passes()
    def test_explicit_type_alias_paramspec_list_form_specialization(self):
        from typing import Callable, Concatenate, ParamSpec, TypeAlias, TypeVar

        from typing_extensions import assert_type

        P = ParamSpec("P")
        R = TypeVar("R")

        Alias: TypeAlias = Callable[Concatenate[int, P], R]

        def capybara(x: Alias[[str, str], None]) -> None:
            assert_type(x, Callable[[int, str, str], None])

    @assert_passes()
    def test_unspecialized_two_param_alias_defaults_type_params_to_any(self):
        from typing import Any, TypeVar

        from typing_extensions import assert_type

        K = TypeVar("K")
        V = TypeVar("V")
        Alias = dict[K, V]

        def capybara(x: Alias) -> None:
            assert_type(x, dict[Any, Any])


class TestUnpack(TestNameCheckVisitorBase):
    @assert_passes()
    def test_in_tuple(self):
        from typing import Tuple

        from typing_extensions import Unpack

        def capybara(
            x: Tuple[int, Unpack[Tuple[str, ...]]],
            y: "Tuple[int, Unpack[Tuple[str, ...]]]",
        ):
            assert_is_value(
                x,
                SequenceValue(
                    tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
                ),
            )
            assert_is_value(
                y,
                SequenceValue(
                    tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
                ),
            )

    @skip_before((3, 11))
    def test_native_unpack(self):
        self.assert_passes("""
            obj: tuple[int, *tuple[str, ...]] = (1, "x", "y")

            def capybara(
                x: tuple[int, *tuple[str, ...]], y: "tuple[int, *tuple[str, ...]]"
            ):
                assert_is_value(
                    x,
                    SequenceValue(
                        tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
                    ),
                )
                assert_is_value(
                    y,
                    SequenceValue(
                        tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
                    ),
                )
                assert_is_value(
                    obj,
                    SequenceValue(
                        tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
                    ),
                )
            """)

    def test_invalid_tuple_ellipsis_forms(self):
        self.assert_passes("""
            t1: tuple[int, int, ...]  # E: invalid_annotation
            t2: tuple[...]  # E: invalid_annotation
            t3: tuple[..., int]  # E: invalid_annotation
            t4: tuple[int, ..., int]  # E: invalid_annotation
            """)

    @skip_before((3, 11))
    def test_invalid_tuple_ellipsis_forms_with_unpack(self):
        self.assert_passes("""
            t1: tuple[*tuple[str], ...]  # E: invalid_annotation
            t2: tuple[*tuple[str, ...], ...]  # E: invalid_annotation
            """)

    @skip_before((3, 11))
    def test_only_one_unbounded_unpack_in_tuple(self):
        self.assert_passes("""
            from typing import TypeVarTuple, Unpack

            Ts = TypeVarTuple("Ts")

            t1: tuple[*tuple[str], *tuple[str]]
            t2: tuple[*tuple[str, *tuple[str, ...]]]
            t3: tuple[*tuple[str, ...], *tuple[int, ...]]  # E: invalid_annotation
            t6: tuple[Unpack[tuple[str]], Unpack[tuple[str]]]
            t7: tuple[Unpack[tuple[str, ...]], Unpack[tuple[int, ...]]]  # E: invalid_annotation

            def capybara() -> None:
                t4: tuple[*tuple[str], *Ts]
                t5: tuple[*tuple[str, ...], *Ts]  # E: invalid_annotation
            """)

    @skip_before((3, 11))
    def test_typevartuple_must_be_unpacked(self):
        self.assert_passes("""
            from typing import TypeVarTuple

            Ts = TypeVarTuple("Ts")

            bad_tuple: tuple[Ts]  # E: invalid_annotation

            def bad(*args: Ts) -> None:  # E: invalid_annotation
                ...
            """)

    @skip_before((3, 11))
    def test_typevartuple_unpack_in_generic_argument(self):
        self.assert_passes("""
            from typing import Generic, TypeVarTuple

            Shape = TypeVarTuple("Shape")

            class Array(Generic[*Shape]):
                ...

            def multiply(x: Array[*Shape], y: Array[*Shape]) -> Array[*Shape]:
                raise NotImplementedError
            """)

    @assert_passes(allow_import_failures=True)
    def test_unresolved_tuple_member_preserves_ellipsis(self):
        from typing import Any

        from missing_module import SomeType  # type: ignore
        from typing_extensions import assert_type

        def capybara(
            x: tuple[SomeType, ...], y: tuple[Any, ...], z: tuple[SomeType]
        ) -> None:
            assert_type(x, tuple[Any, ...])
            assert_type(y, tuple[Any, ...])
            assert_type(z, tuple[Any])


class TestMissinGenericParameters(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing import Dict, List, Set

        def capybara(
            x: list,  # E: missing_generic_parameters
            y: List,  # E: missing_generic_parameters
            z: List[int],
            a: Set[list],  # E: missing_generic_parameters
            b: Dict[str, list],  # E: missing_generic_parameters
        ) -> set:  # E: missing_generic_parameters
            return {1}

    @assert_passes()
    def test_with_pep_585(self):
        def capybara(
            a: set[list],  # E: missing_generic_parameters
            b: dict[str, list],  # E: missing_generic_parameters
        ) -> None:
            pass

    @assert_passes()
    def test_union_or(self):
        def capybara(
            x: list | int,  # E: missing_generic_parameters
            y: str | list,  # E: missing_generic_parameters
            z: float | bool | set,  # E: missing_generic_parameters
        ) -> None:
            pass


class TestIfTypeChecking(TestNameCheckVisitorBase):
    @assert_passes()
    def test_typevar(self):
        from typing import TYPE_CHECKING, TypeVar

        from typing_extensions import Literal, assert_type

        if TYPE_CHECKING:
            T = TypeVar("T")

            def capybara(x: T) -> T:
                return x

            assert_type(capybara(1), Literal[1])

    @assert_passes()
    def test_namedtuple(self):
        from collections import namedtuple
        from typing import TYPE_CHECKING, Any, NamedTuple

        from typing_extensions import assert_type

        if TYPE_CHECKING:
            TypedCapybara = NamedTuple("TypedCapybara", [("x", int), ("y", str)])
            UntypedCapybara = namedtuple("UntypedCapybara", ["x", "y"])

        def capybara(t: "TypedCapybara", u: "UntypedCapybara") -> None:
            assert_type(t.x, int)
            assert_type(t.y, str)

            assert_type(u.x, Any)
            assert_type(u.y, Any)


class TestFloatInt(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        from typing_extensions import assert_type

        def capybara(x: float):
            assert_is_value(x, TypedValue(float) | TypedValue(int))
            assert_type(x, float)

            if isinstance(x, float):
                # can't express this type for assert_type()
                assert_is_value(x, TypedValue(float))
            else:
                assert_is_value(x, TypedValue(int))
                assert_type(x, int)

    @assert_passes()
    def test_complex(self):
        from typing_extensions import assert_type

        def capybara(x: complex):
            assert_is_value(
                x, TypedValue(complex) | TypedValue(float) | TypedValue(int)
            )
            assert_type(x, complex)

            if isinstance(x, float):
                # can't express this type for assert_type()
                assert_is_value(x, TypedValue(float))
            else:
                assert_is_value(x, TypedValue(int) | TypedValue(complex))

    @assert_passes()
    def test_cast(self):
        from typing import cast

        from typing_extensions import assert_type

        def capybara(x):
            f = cast(float, x)
            assert_is_value(f, TypedValue(float) | TypedValue(int))
            assert_type(f, float)

    @assert_passes()
    def test_float_subclass(self):
        from typing_extensions import assert_type

        class MyFloat(float):
            pass

        def capybara(x: MyFloat):
            assert_type(x, MyFloat)

        def caller():
            capybara(MyFloat(1.0))
            capybara(1.0)  # E: incompatible_argument

    @assert_passes()
    def test_complex_subclass(self):
        from typing_extensions import assert_type

        class MyComplex(complex):
            pass

        def capybara(x: MyComplex):
            assert_type(x, MyComplex)

        def caller():
            capybara(MyComplex(1.0))
            capybara(1.0j)  # E: incompatible_argument


class TestProtocol(TestNameCheckVisitorBase):
    @assert_passes()
    def test_conditional_annotations(self):
        from types import SimpleNamespace
        from typing import Protocol

        class ProtocolA(Protocol):
            a: int

            if len("x") > 0:
                b: str
            else:
                c: float

        obj = SimpleNamespace(a=1, b="x")

        def capybara(x: ProtocolA) -> None:
            pass

        def caller() -> None:
            capybara(obj)
