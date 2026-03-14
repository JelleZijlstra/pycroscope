# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_if_not_installed
from .value import (
    AnySource,
    AnyValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    assert_is_value,
)


class TestExtraKeys(TestNameCheckVisitorBase):
    @assert_passes()
    def test_signature(self):
        from typing_extensions import TypedDict

        from pycroscope.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        def capybara() -> None:
            x = TD(a="a", b=1)
            assert_is_value(
                x,
                TypedDictValue(
                    {"a": TypedDictEntry(TypedValue(str))}, extra_keys=TypedValue(int)
                ),
            )

            TD(a="a", b="b")  # E: incompatible_argument

    @assert_passes()
    def test_methods(self):
        from typing import Union

        from typing_extensions import Literal, TypedDict

        from pycroscope.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        class NormalTD(TypedDict):
            a: str

        def getitem(td: TD, ntd: NormalTD) -> None:
            td["b"] = 3
            ntd["b"] = 3  # E: invalid_typeddict_key

        def setitem(td: TD) -> None:
            assert_type(td["b"], int)

        def get(td: TD) -> None:
            assert_type(td.get("b", "x"), Union[int, Literal["x"]])

        def pop(td: TD) -> None:
            assert_type(td.pop("b"), int)

        def setdefault(td: TD) -> None:
            assert_type(td.setdefault("b", "x"), Union[int, Literal["x"]])

    @assert_passes()
    def test_kwargs_annotation(self):
        from typing_extensions import TypedDict, Unpack

        from pycroscope.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        def caller(**kwargs: Unpack[TD]) -> None:
            assert_type(kwargs["b"], int)

        def capybara() -> None:
            caller(a="x", b=1)
            caller(a="x", b="y")  # E: incompatible_argument

    @assert_passes()
    def test_compatibility(self):
        from typing import Any, Dict

        from typing_extensions import ReadOnly, TypedDict

        from pycroscope.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        @has_extra_keys(bool)
        class TD2(TypedDict):
            a: str

        @has_extra_keys(bytes)
        class TD3(TypedDict):
            a: str

        @has_extra_keys(ReadOnly[int])
        class TD4(TypedDict):
            a: str

        def want_td(td: TD) -> None:
            pass

        def want_td4(td: TD4) -> None:
            pass

        def capybara(td: TD, td2: TD2, td3: TD3, anydict: Dict[str, Any]) -> None:
            want_td(td)
            want_td(td2)  # E: incompatible_argument
            want_td(td3)  # E: incompatible_argument
            want_td(anydict)  # E: incompatible_argument

            want_td4(td)
            want_td4(td2)
            want_td4(td3)  # E: incompatible_argument
            want_td4(anydict)  # E: incompatible_argument

    @assert_passes()
    def test_iteration(self):
        from typing import Union

        from typing_extensions import Literal, TypedDict

        from pycroscope.extensions import has_extra_keys

        @has_extra_keys(int)
        class TD(TypedDict):
            a: str

        class TD2(TypedDict):
            a: str

        def capybara(td: TD, td2: TD2) -> None:
            for k, v in td.items():
                assert_type(k, str)
                assert_type(v, Union[int, str])
            for k in td:
                assert_type(k, Union[str, Literal["a"]])

            for k, v in td2.items():
                assert_type(k, str)
                assert_type(v, str)
            for k in td2:
                assert_type(k, Union[str, Literal["a"]])

    @assert_passes()
    def test_explicit_items_compatible_with_extra_items(self):
        from typing_extensions import NotRequired, ReadOnly, TypedDict

        class MovieBase2(TypedDict, extra_items=int | None):
            name: str

        class MovieDetails(TypedDict, extra_items=int | None):
            name: str
            year: NotRequired[int]

        class MovieWithYear2(TypedDict, extra_items=int | None):
            name: str
            year: int | None

        class MovieSI(TypedDict, extra_items=ReadOnly[str | int]):
            name: str

        class MovieDetails5(TypedDict, extra_items=int):
            name: str
            actors: list[str]

        details2: MovieDetails = {"name": "Kill Bill Vol. 1", "year": 2003}
        movie2: MovieBase2 = details2  # E: incompatible_assignment

        details3: MovieWithYear2 = {"name": "Kill Bill Vol. 1", "year": 2003}
        movie3: MovieBase2 = details3  # E: incompatible_assignment

        details5: MovieDetails5 = {
            "name": "Kill Bill Vol. 2",
            "actors": ["Uma Thurman"],
        }
        movie5: MovieSI = details5  # E: incompatible_assignment
        print(movie2, movie3, movie5)

    @assert_passes()
    def test_closed_constructor(self):
        from typing_extensions import TypedDict

        class ClosedMovie(TypedDict, closed=True):
            name: str

        ClosedMovie(name="No Country for Old Men")
        # E: incompatible_argument
        ClosedMovie(name="No Country for Old Men", year=2007)


class TestTypedDict(TestNameCheckVisitorBase):
    @assert_passes()
    def test_constructor(self):
        from typing_extensions import NotRequired, TypedDict

        class Capybara(TypedDict):
            x: int
            y: str

        class MaybeCapybara(TypedDict):
            x: int
            y: NotRequired[str]

        def capybara():
            cap = Capybara(x=1, y="2")
            assert_is_value(
                cap,
                TypedDictValue(
                    {
                        "x": TypedDictEntry(TypedValue(int)),
                        "y": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            Capybara(x=1)  # E: incompatible_call

            maybe_cap = MaybeCapybara(x=1)
            assert_is_value(
                maybe_cap,
                TypedDictValue(
                    {
                        "x": TypedDictEntry(TypedValue(int)),
                        "y": TypedDictEntry(TypedValue(str), required=False),
                    }
                ),
            )

    @assert_passes()
    def test_function_scope_constructor_and_isinstance(self):
        from typing import TypedDict

        def run(movie: dict[str, object]) -> None:
            class Movie(TypedDict):
                name: str
                year: int

            m = Movie(name="Blade Runner", year=1982)
            assert_is_value(
                m,
                TypedDictValue(
                    {
                        "name": TypedDictEntry(TypedValue(str)),
                        "year": TypedDictEntry(TypedValue(int)),
                    }
                ),
            )

            if isinstance(movie, Movie):  # E: incompatible_argument
                pass

    @assert_passes()
    def test_function_scope_class_assignable_to_type(self):
        from typing import TypedDict

        def run() -> None:
            class Movie(TypedDict):
                name: str

            cls: type = Movie
            assert_type(cls, type)

    @assert_passes()
    def test_typevar_bound_disallows_typeddict(self):
        from typing import TypeVar

        from typing_extensions import TypedDict

        TypeVar("T", bound=TypedDict)  # E: invalid_annotation

    @assert_passes()
    def test_typevar_bound_allows_typeddict_class(self):
        from typing import TypeVar

        from typing_extensions import TypedDict

        class Movie(TypedDict):
            name: str

        TypeVar("T", bound=Movie)

        def run() -> None:
            class InnerMovie(TypedDict):
                name: str

            TypeVar("InnerT", bound=InnerMovie)

    @assert_passes()
    def test_unknown_key(self):
        from typing_extensions import TypedDict

        class Capybara(TypedDict):
            x: int

        def user(c: Capybara):
            assert_type(c["x"], int)
            c["y"]  # E: invalid_typeddict_key

    @assert_passes()
    def test_basic(self):
        from typing_extensions import TypedDict as TETypedDict

        T2 = TETypedDict("T2", {"a": int, "b": str})

        def capybara(y: T2):
            assert_is_value(
                y,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            assert_type(y["a"], int)

    @assert_passes()
    def test_functional_syntax_validation(self):
        from typing_extensions import TypedDict

        T = TypedDict("T", {"a": int})
        my_dict = {"a": int}
        Bad1 = TypedDict("Bad1", my_dict)  # E: incompatible_call
        Bad2 = TypedDict("Bad2", {1: int})  # E: incompatible_call
        Bad3 = TypedDict("WrongName", {"a": int})  # E: incompatible_call

        def capybara(x: T) -> None:
            assert_type(x["a"], int)
            print(Bad1, Bad2, Bad3)

    @assert_passes(run_in_both_module_modes=True)
    def test_class_syntax_method_validation(self):
        from typing import Generic, TypeVar

        from typing_extensions import TypedDict

        class Movie(TypedDict):
            director: "Person"

        class Person(TypedDict):
            name: str

        class BadTypedDict(TypedDict):
            name: str

            def method(self):  # E: invalid_annotation
                pass

        T = TypeVar("T")

        class GenericTypedDict(TypedDict, Generic[T]):
            value: T

        def capybara(movie: Movie) -> str:
            return movie["director"]["name"]

    @assert_passes(run_in_both_module_modes=True)
    def test_stdlib_typeddict_class_syntax_and_newtype_base(self):
        from typing import Generic, NewType, TypedDict, TypeVar

        class Movie(TypedDict):
            director: "Person"

        class Person(TypedDict):
            name: str

        T = TypeVar("T")

        class GenericMovie(TypedDict, Generic[T]):
            value: T

        BadMovieId = NewType("BadMovieId", Movie)  # E: incompatible_call

        def capybara(movie: Movie, generic_movie: GenericMovie[int]) -> str:
            print(BadMovieId)
            print(generic_movie)
            return movie["director"]["name"]

    @assert_passes()
    def test_functional_syntax_keyword_fields(self):
        from typing_extensions import TypedDict

        try:
            Movie = TypedDict("Movie", name=str, year=int)
        except TypeError:
            Movie = TypedDict("Movie", {"name": str, "year": int})
        ok: Movie = {"name": "Blade Runner", "year": 1982}
        bad: Movie = {"name": "Blade Runner", "year": ""}  # E: incompatible_assignment
        print(ok, bad)

    @assert_passes()
    def test_functional_syntax_qualifiers(self):
        from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

        Band = TypedDict("Band", {"name": str, "members": ReadOnly[list[str]]})
        RecursiveMovie = TypedDict(
            "RecursiveMovie",
            {"title": Required[str], "predecessor": NotRequired["RecursiveMovie"]},
        )

        band: Band = {"name": "blur", "members": []}
        band["members"] = ["Damon Albarn"]  # E: readonly_typeddict
        movie: RecursiveMovie = {
            "title": "Beethoven 3",
            "predecessor": {"title": "Beethoven 2"},
        }
        print(movie)

    @skip_if_not_installed("mypy_extensions")
    @assert_passes()
    def test_mypy_extensions(self):
        from mypy_extensions import TypedDict as METypedDict

        T = METypedDict("T", {"a": int, "b": str})

        def capybara(x: T):
            assert_is_value(
                x,
                TypedDictValue(
                    {
                        "a": TypedDictEntry(TypedValue(int)),
                        "b": TypedDictEntry(TypedValue(str)),
                    }
                ),
            )
            assert_type(x["a"], int)

    @assert_passes()
    def test_unknown_key_unresolved(self):
        from typing_extensions import TypedDict

        T = TypedDict("T", {"a": int, "b": str})

        def capybara(x: T):
            val = x["not a key"]  # E: invalid_typeddict_key
            assert_is_value(val, AnyValue(AnySource.error))

    @assert_passes()
    def test_invalid_key(self):
        from typing_extensions import TypedDict

        T = TypedDict("T", {"a": int, "b": str})

        def capybara(x: T):
            x[0]  # E: invalid_typeddict_key

    @assert_passes()
    def test_total(self):
        from typing_extensions import TypedDict

        class TD(TypedDict, total=False):
            a: int
            b: str

        class TD2(TD):
            c: float

        def f(td: TD) -> None:
            pass

        def g(td2: TD2) -> None:
            pass

        def caller() -> None:
            f({})
            f({"a": 1})
            f({"a": 1, "b": "c"})
            f({"a": "a"})  # E: incompatible_argument
            g({"c": 1.0})
            g({})  # E: incompatible_argument

    @assert_passes()
    def test_disallow_non_typeddict_base(self):
        from typing import Generic, TypeVar

        from typing_extensions import TypedDict

        T = TypeVar("T")

        def capybara() -> None:
            class NonTypedDict:
                pass

            class BadTypedDict(TypedDict, NonTypedDict):  # E: invalid_base
                x: int

            class BadGenericTypedDict(TypedDict, Generic):  # E: invalid_base
                x: int

            class GenericTypedDict(TypedDict, Generic[T]):
                x: T

    @assert_passes()
    def test_typeddict_assignment_key_checks(self):
        from typing_extensions import TypedDict

        class Movie(TypedDict):
            name: str
            year: int

        def capybara(variable_key: str) -> None:
            movie: Movie
            movie = {"name": "Blade Runner"}  # E: incompatible_assignment
            print(movie)
            movie = {  # E: incompatible_assignment
                "name": "Blade Runner",
                "year": 1982.1,
            }
            print(movie)
            movie = {"name": "", "year": 1900, "other": 2}  # E: incompatible_assignment
            print(movie)
            movie = {variable_key: "", "year": 1900}  # E: incompatible_assignment
            print(movie)

    @assert_passes()
    def test_typeddict_clear_variants(self):
        from typing_extensions import NotRequired, ReadOnly, TypedDict

        class NonClosed(TypedDict):
            optional: NotRequired[int]

        class ClosedRequired(TypedDict, closed=True):
            required: int

        class ClosedOptional(TypedDict, closed=True):
            optional: NotRequired[int]

        class ClosedReadonly(TypedDict, closed=True):
            optional: ReadOnly[NotRequired[int]]

        class ClosedReadonlyExtra(TypedDict, extra_items=ReadOnly[int]):
            optional: NotRequired[int]

        class ClosedMutableExtra(TypedDict, extra_items=int):
            optional: NotRequired[int]

        def capybara(
            non_closed: NonClosed,
            closed_required: ClosedRequired,
            closed_optional: ClosedOptional,
            closed_readonly: ClosedReadonly,
            closed_readonly_extra: ClosedReadonlyExtra,
            closed_mutable_extra: ClosedMutableExtra,
        ) -> None:
            non_closed.clear()  # E: incompatible_call
            closed_required.clear()  # E: incompatible_call
            closed_optional.clear()
            closed_readonly.clear()  # E: incompatible_call
            closed_readonly_extra.clear()  # E: incompatible_call
            closed_mutable_extra.clear()

    @assert_passes()
    def test_typeddict_popitem_variants(self):
        from typing_extensions import (
            Never,
            NotRequired,
            ReadOnly,
            TypedDict,
            assert_type,
        )

        class NonClosed(TypedDict):
            optional: NotRequired[int]

        class ClosedRequired(TypedDict, closed=True):
            required: int

        class ClosedOptional(TypedDict, closed=True):
            optional: NotRequired[int]

        class ClosedReadonly(TypedDict, closed=True):
            optional: ReadOnly[NotRequired[int]]

        class ClosedReadonlyExtra(TypedDict, extra_items=ReadOnly[int]):
            optional: NotRequired[int]

        class ClosedMutableExtra(TypedDict, extra_items=int):
            optional: NotRequired[int]

        class ClosedEmpty(TypedDict, closed=True):
            pass

        def capybara(
            non_closed: NonClosed,
            closed_required: ClosedRequired,
            closed_optional: ClosedOptional,
            closed_readonly: ClosedReadonly,
            closed_readonly_extra: ClosedReadonlyExtra,
            closed_mutable_extra: ClosedMutableExtra,
            closed_empty: ClosedEmpty,
        ) -> None:
            # E: incompatible_call
            assert_type(non_closed.popitem()[1], object)
            closed_required.popitem()  # E: incompatible_call
            closed_optional.popitem()
            closed_readonly.popitem()  # E: incompatible_call
            closed_readonly_extra.popitem()  # E: incompatible_call
            closed_mutable_extra.popitem()
            assert_type(closed_empty.popitem()[1], Never)


class TestReadOnly(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing import Any, Dict

        from typing_extensions import NotRequired, ReadOnly, TypedDict

        class TD(TypedDict):
            a: ReadOnly[NotRequired[int]]
            b: ReadOnly[str]

        def capybara(td: TD, anydict: Dict[str, Any]) -> None:
            td["a"] = 1  # E: readonly_typeddict
            td["b"] = "a"  # E: readonly_typeddict
            td.update(anydict)  # E: invalid_typeddict_key
            td.setdefault("a", 1)  # E: readonly_typeddict
            td.setdefault("b", "a")  # E: readonly_typeddict
            td.pop("a")  # E: readonly_typeddict
            td.pop("b")  # E: incompatible_argument
            del td["a"]  # E: readonly_typeddict
            del td["b"]  # E: readonly_typeddict

    @assert_passes()
    def test_compatibility(self):
        from typing import Any, Dict

        from typing_extensions import ReadOnly, TypedDict

        class TD(TypedDict):
            a: int

        class TD2(TypedDict):
            a: ReadOnly[int]

        class TD3(TypedDict):
            a: bool

        class TD4(TypedDict):
            a: str

        def want_td(td: TD) -> None:
            pass

        def want_td2(td: TD2) -> None:
            pass

        def capybara(
            td: TD, td2: TD2, td3: TD3, td4: TD4, anydict: Dict[str, Any]
        ) -> None:
            want_td(td)
            want_td(td2)  # E: incompatible_argument
            want_td(td3)  # E: incompatible_argument
            want_td(td4)  # E: incompatible_argument
            want_td(anydict)  # E: incompatible_argument

            want_td2(td)
            want_td2(td2)
            want_td2(td3)
            want_td2(td4)  # E: incompatible_argument
            want_td2(anydict)  # E: incompatible_argument

    @assert_passes(run_in_both_module_modes=True)
    def test_inheritance_validation_at_module_scope(self):
        from typing_extensions import NotRequired, ReadOnly, Required, TypedDict

        class F1(TypedDict):
            a: Required[int]
            b: ReadOnly[NotRequired[int]]
            c: ReadOnly[Required[int]]

        class F3(F1):
            a: ReadOnly[int]  # E: invalid_annotation

        class F4(F1):
            a: NotRequired[int]  # E: invalid_annotation

        class F5(F1):
            b: ReadOnly[Required[int]]

        class F6(F1):
            c: ReadOnly[NotRequired[int]]  # E: invalid_annotation

        class TD_A1(TypedDict):
            x: int
            y: ReadOnly[int]

        class TD_A2(TypedDict):
            x: float
            y: ReadOnly[float]

        class TD_A(TD_A1, TD_A2): ...  # E: invalid_base

        class TD_B1(TypedDict):
            x: ReadOnly[NotRequired[int]]
            y: ReadOnly[Required[int]]

        class TD_B2(TypedDict):
            x: ReadOnly[Required[int]]
            y: ReadOnly[NotRequired[int]]

        class TD_B(TD_B1, TD_B2): ...  # E: invalid_base

    @assert_passes()
    def test_update_ignores_uninhabitable_optional_key(self):
        from typing_extensions import Never, NotRequired, ReadOnly, TypedDict

        class A(TypedDict):
            x: ReadOnly[int]
            y: int

        class B(TypedDict):
            x: NotRequired[Never]
            y: ReadOnly[int]

        def capybara(a: A, b: B) -> None:
            a.update(a)  # E: readonly_typeddict
            a.update(b)

    @assert_passes()
    def test_annotated_plus_qualifier(self):
        from typing_extensions import Annotated, ReadOnly, TypedDict

        class TD(TypedDict):
            a: Annotated[ReadOnly[int], ""]

        td: TD = {"a": 1}  # OK
        td2: TD = {"a": "x"}  # E: incompatible_assignment

        def capybara(td: TD) -> None:
            assert_type(td["a"], int)
            td["a"] = 1  # E: readonly_typeddict

    @assert_passes()
    def test_annotated_plus_multiple_qualifiers(self):
        from typing_extensions import (
            Annotated,
            NotRequired,
            ReadOnly,
            Required,
            TypedDict,
        )

        class Movie2(TypedDict):
            title: Required[ReadOnly[str]]  # OK
            year: Annotated[NotRequired[ReadOnly[int]], ""]  # OK

        m2: Movie2 = {"title": "", "year": 1991}
        m2["title"] = ""  # E: readonly_typeddict
        m2["year"] = 1992  # E: readonly_typeddict


class TestClosed(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing import Any, Dict

        from typing_extensions import NotRequired, TypedDict

        class Closed(TypedDict, closed=True):
            a: NotRequired[int]
            b: str

        class Open(TypedDict):
            a: NotRequired[int]
            b: str

        def want_closed(td: Closed) -> None:
            pass

        def want_open(td: Open) -> None:
            pass

        def capybara(closed: Closed, open: Open, anydict: Dict[str, Any]) -> None:
            closed["a"] = 1
            closed["b"] = "a"
            closed["a"] = "x"  # E: incompatible_argument

            open["a"] = 1
            open["b"] = "a"
            open["a"] = "x"  # E: incompatible_argument

            closed.update(anydict)  # E: invalid_typeddict_key
            open.update(anydict)  # E: invalid_typeddict_key

            x: Closed = {"a": 1, "b": "a", "c": "x"}  # E: incompatible_assignment
            y: Open = {"a": 1, "b": "a"}
            print(x, y)

            want_closed(closed)
            want_closed(open)  # E: incompatible_argument

            want_open(open)
            want_open(closed)
