from typing import TypeVar

from .predicates import is_universally_assignable
from .value import (
    AnySource,
    AnyValue,
    IntersectionValue,
    TypedValue,
    TypeVarParam,
    TypeVarValue,
)


def test_is_universally_assignable_intersection() -> None:
    typevar = TypeVarValue(TypeVarParam(TypeVar("T_pred")))
    value = IntersectionValue((typevar, AnyValue(AnySource.unannotated)))
    assert is_universally_assignable(value, TypedValue(int))


def test_is_universally_assignable_non_universal() -> None:
    assert not is_universally_assignable(TypedValue(int), TypedValue(int))
