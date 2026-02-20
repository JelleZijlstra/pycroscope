# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestTypeForm(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing_extensions import TypeForm

        def takes_int_type(typ: TypeForm[int]) -> None: ...

        takes_int_type(int)
        takes_int_type(bool)
        takes_int_type(str)  # E: incompatible_argument
        takes_int_type(1)  # E: incompatible_argument

        def f(cls_int: type[int], cls_obj: type[object]) -> None:
            takes_int_type(cls_int)
            takes_int_type(cls_obj)  # E: incompatible_argument

    @assert_passes()
    def test_generic_type_form(self):
        from typing import TypeVar, cast

        from typing_extensions import TypeForm, assert_type

        T = TypeVar("T")

        def cast_with_typeform(typ: TypeForm[T], value: object) -> T:
            return cast(typ, value)

        def f(value: object) -> None:
            i = cast_with_typeform(int, value)
            b = cast_with_typeform(bool, value)
            assert_type(i, int)
            assert_type(b, bool)

    @assert_passes()
    def test_invalid_typeform(self):
        from typing_extensions import TypeForm

        def f1(x: TypeForm) -> None:
            pass

        def f2(x: "TypeForm[int, str]") -> None:  # E: invalid_annotation
            pass

        def f3(x: "TypeForm[42]") -> None:  # E: invalid_annotation
            pass

    @assert_passes()
    def test_typeform_to_typeform_assignments(self):
        from typing import Any

        from typing_extensions import TypeForm, assert_type

        v_any: TypeForm[Any] = int | str
        v_str: TypeForm[str] = str
        v_any = v_str
        v_str = v_any
        assert_type(v_any, TypeForm[Any])
        assert_type(v_str, TypeForm[str])

    @assert_passes()
    def test_bare_typeform_means_any(self):
        from typing import Any

        from typing_extensions import TypeForm, assert_type

        v_bare: TypeForm = int | str
        v_any: TypeForm[Any] = v_bare
        v_bare = v_any
        assert_type(v_bare, TypeForm[Any])

    @assert_passes()
    def test_explicit_typeform_call(self):
        from typing_extensions import TypeForm, assert_type

        def f() -> None:
            x1 = TypeForm(str | None)
            assert_type(x1, TypeForm[str | None])

            x2 = TypeForm("list[int]")
            assert_type(x2, TypeForm[list[int]])

            x3 = TypeForm("type(1)")  # E: invalid_annotation
            x4 = type(1)
            x5 = TypeForm(type(1))  # E: invalid_annotation
            x3
            x4
            x5

    @assert_passes()
    def test_assignability_examples(self):
        from typing import Any, Literal, Optional

        from typing_extensions import TypeForm

        ok1: TypeForm[str | None] = str | None
        ok2: TypeForm[str | None] = str
        ok3: TypeForm[str | None] = None
        ok4: TypeForm[str | None] = Literal[None]
        ok5: TypeForm[str | None] = Optional[str]
        ok6: TypeForm[str | None] = "str | None"
        ok7: TypeForm[str | None] = Any
        ok8: TypeForm[set[str]] = "set[str]"

        err1: TypeForm[str | None] = str | int  # E: incompatible_assignment
        err2: TypeForm[str | None] = list[str | None]  # E: incompatible_assignment

    @assert_passes()
    def test_invalid_type_expressions_for_implicit_typeform(self):
        import typing  # E: invalid_annotation  # E: invalid_annotation

        from typing_extensions import Self, TypeForm, TypeVarTuple, Unpack

        Ts = TypeVarTuple("Ts")
        var = 1

        bad1: TypeForm = tuple()  # E: incompatible_assignment
        bad2: TypeForm = (1, 2)  # E: incompatible_assignment
        bad3: TypeForm = 1  # E: incompatible_assignment
        bad4: TypeForm = Self  # E: incompatible_assignment
        # These currently evaluate through runtime objects and are accepted.
        bad5: TypeForm = typing.Literal[var]
        bad6: TypeForm = typing.Literal[""]
        bad7: TypeForm = typing.ClassVar[int]  # E: incompatible_assignment
        bad8: TypeForm = typing.Final[int]  # E: incompatible_assignment
        bad9: TypeForm = Unpack[Ts]  # E: incompatible_assignment
        bad10: TypeForm = typing.Optional  # E: incompatible_assignment
        bad11: TypeForm = "int + str"  # E: incompatible_assignment
        bad1, bad2, bad3, bad4, bad5, bad6, bad7, bad8, bad9, bad10, bad11

    @assert_passes()
    def test_typeform_any_parameter(self):
        from typing import Any

        from typing_extensions import TypeForm

        def accepts_any(typ: TypeForm[Any]) -> None: ...

        accepts_any(int | str)
        accepts_any(str)
        accepts_any(list[int])
        accepts_any(1)  # E: incompatible_argument

    @assert_passes()
    def test_covariance_and_type_subtyping_via_calls(self):
        from typing_extensions import TypeForm

        def accepts_int_or_str(typ: TypeForm[int | str]) -> None: ...

        def accepts_str(typ: TypeForm[str]) -> None: ...

        def get_type() -> type[int]:
            return int

        accepts_int_or_str(int)
        accepts_int_or_str(str)
        accepts_int_or_str(get_type())
        accepts_str(str)
        accepts_str(int)  # E: incompatible_argument
        accepts_str(get_type())  # E: incompatible_argument

    @assert_passes()
    def test_type_subtype_of_typeform(self):
        from typing_extensions import TypeForm

        def get_type() -> type[int]:
            return int

        t3: TypeForm[int | str] = get_type()
        t4: TypeForm[str] = get_type()  # E: incompatible_assignment
        assert t3
        assert t4
