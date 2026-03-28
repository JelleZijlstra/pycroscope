# static analysis: ignore
import collections.abc
from unittest import mock

from .annotated_types import MinLen
from .error_code import ErrorCode
from .input_sig import ELLIPSIS, InputSigValue
from .stacked_scopes import Composite
from .suggested_type import prepare_type, should_suggest_type
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import (
    NO_RETURN_VALUE,
    AnySource,
    AnyValue,
    GenericValue,
    IntersectionValue,
    KnownValue,
    PredicateValue,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedValue,
    UnboundMethodValue,
)


class TestSuggestedType(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.suggested_return_type: True})
    def test_return(self):
        def capybara():  # E: suggested_return_type
            return 1

        def kerodon(cond):  # E: suggested_return_type
            if cond:
                return 1
            else:
                return 2

    @assert_passes(settings={ErrorCode.suggested_parameter_type: True})
    def test_parameter(self):
        def capybara(a):  # E: suggested_parameter_type
            pass

        def annotated(b: int):
            pass

        class Mammalia:
            # should not suggest a type for this
            def method(self):
                pass

        def kerodon(unannotated):
            capybara(1)
            annotated(2)

            m = Mammalia()
            m.method()
            Mammalia.method(unannotated)


class A:
    pass


class B(A):
    pass


class C(A):
    pass


def test_prepare_type() -> None:
    assert prepare_type(KnownValue(int) | KnownValue(str)) == TypedValue(type)
    assert prepare_type(KnownValue(C) | KnownValue(B)) == SubclassValue(TypedValue(A))
    assert prepare_type(KnownValue(int)) == SubclassValue(TypedValue(int))

    assert prepare_type(SubclassValue(TypedValue(B)) | KnownValue(C)) == SubclassValue(
        TypedValue(A)
    )
    assert prepare_type(SubclassValue(TypedValue(B)) | KnownValue(B)) == SubclassValue(
        TypedValue(B)
    )
    assert prepare_type(KnownValue(None) | TypedValue(str)) == KnownValue(
        None
    ) | TypedValue(str)
    assert prepare_type(KnownValue(True) | KnownValue(False)) == TypedValue(bool)


def test_prepare_type_converts_input_sig_to_any() -> None:
    assert prepare_type(InputSigValue(ELLIPSIS)) == AnyValue(AnySource.inference)
    assert prepare_type(
        GenericValue(
            collections.abc.Callable, [InputSigValue(ELLIPSIS), TypedValue(int)]
        )
    ) == GenericValue(
        collections.abc.Callable, [AnyValue(AnySource.inference), TypedValue(int)]
    )


def test_prepare_type_intersection() -> None:
    assert prepare_type(IntersectionValue((KnownValue(True), TypedValue(bool)))) == (
        IntersectionValue((TypedValue(bool), TypedValue(bool)))
    )


def test_prepare_type_intersection_uses_intersect_multi() -> None:
    with mock.patch(
        "pycroscope.relations.intersect_multi", return_value=TypedValue(bool)
    ) as mock_intersect:
        ctx = object()
        result = prepare_type(
            IntersectionValue((KnownValue(True), TypedValue(bool))), ctx=ctx
        )
    assert result == TypedValue(bool)
    mock_intersect.assert_called_once_with([TypedValue(bool), TypedValue(bool)], ctx)


def test_should_suggest_type_intersection() -> None:
    assert not should_suggest_type(
        IntersectionValue((AnyValue(AnySource.inference), TypedValue(int)))
    )


def test_should_suggest_type_union() -> None:
    assert not should_suggest_type(NO_RETURN_VALUE)
    assert not should_suggest_type(TypedValue(int) | AnyValue(AnySource.inference))
    assert should_suggest_type(TypedValue(int) | KnownValue(1))


def test_should_not_suggest_non_annotation_values() -> None:
    assert not should_suggest_type(
        SyntheticClassObjectValue("Cls", TypedValue("mod.Cls"))
    )
    assert not should_suggest_type(SyntheticModuleValue(("mod",)))
    assert not should_suggest_type(UnboundMethodValue("f", Composite(TypedValue(int))))
    assert not should_suggest_type(PredicateValue(MinLen(1)))
