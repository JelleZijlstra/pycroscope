"""

Support for annotations from the annotated_types library.

"""

import enum
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any

from pycroscope.value import CanAssign, CanAssignContext, Value, flatten_values

from .extensions import CustomCheck, PredicateCheck
from .predicates import MaxLen, MinLen
from .value import (
    AnnotatedValue,
    AnyValue,
    CanAssignError,
    CustomCheckExtension,
    KnownValue,
    Predicate,
    PredicateValue,
    unannotate,
)

try:
    import annotated_types
except ImportError:  # pragma: no cover

    def get_annotated_types_extension(
        obj: object,
    ) -> Iterable[CustomCheckExtension | Value]:
        return []

else:

    def get_annotated_types_extension(
        obj: object,
    ) -> Iterable[CustomCheckExtension | Value]:
        if isinstance(obj, annotated_types.GroupedMetadata):
            for value in obj:
                if not isinstance(value, annotated_types.BaseMetadata):
                    continue
                maybe_ext = _get_single_annotated_types_extension(value)
                if maybe_ext is not None:
                    if isinstance(maybe_ext, Predicate):
                        yield PredicateValue(maybe_ext)
                    else:
                        yield CustomCheckExtension(maybe_ext)
        elif isinstance(obj, annotated_types.BaseMetadata):
            maybe_ext = _get_single_annotated_types_extension(obj)
            if maybe_ext is not None:
                if isinstance(maybe_ext, Predicate):
                    yield PredicateValue(maybe_ext)
                else:
                    yield CustomCheckExtension(maybe_ext)

    def _get_single_annotated_types_extension(
        obj: annotated_types.BaseMetadata,
    ) -> CustomCheck | Predicate | None:
        if isinstance(obj, annotated_types.Gt):
            return Gt(obj.gt)
        elif isinstance(obj, annotated_types.Ge):
            return Ge(obj.ge)
        elif isinstance(obj, annotated_types.Lt):
            return Lt(obj.lt)
        elif isinstance(obj, annotated_types.Le):
            return Le(obj.le)
        elif isinstance(obj, annotated_types.MultipleOf):
            return MultipleOf(obj.multiple_of)
        elif isinstance(obj, annotated_types.MinLen):
            return MinLen(obj.min_length)
        elif isinstance(obj, annotated_types.MaxLen):
            return MaxLen(obj.max_length)
        elif isinstance(obj, annotated_types.Timezone):
            return Timezone(obj.tz)
        elif isinstance(obj, annotated_types.Predicate):
            return AnnotatedPredicate(obj.func)
        else:
            # In case annotated_types adds more kinds of checks in the future
            return None


@dataclass(frozen=True)
class AnnotatedTypesCheck(PredicateCheck):
    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        for subval in flatten_values(value):
            original_subval = subval
            if isinstance(subval, AnnotatedValue):
                if any(
                    ext == self or self.is_compatible_metadata(ext)
                    for ext in subval.get_custom_check_of_type(AnnotatedTypesCheck)
                ):
                    continue
            subval = unannotate(subval)
            if isinstance(subval, AnyValue):
                continue
            if isinstance(subval, KnownValue):
                try:
                    result = self.predicate(subval.val)
                except Exception as e:
                    return CanAssignError(
                        f"Failed to check {subval.val} against predicate {self}",
                        children=[CanAssignError(repr(e))],
                    )
                else:
                    if not result:
                        return CanAssignError(
                            f"Value {subval.val} does not match predicate {self}"
                        )
            else:
                can_assign = self.can_assign_non_literal(original_subval)
                if isinstance(can_assign, CanAssignError):
                    return can_assign
        return {}

    def predicate(self, value: Any) -> bool:
        raise NotImplementedError  # pragma: no cover

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        """Override this to allow metadata that is not exactly the same as the
        one in this object to match. For example, Gt(5) should accept Gt(4), as that
        is a strictly weaker condition.
        """
        return False  # pragma: no cover

    def can_assign_non_literal(self, value: Value) -> CanAssign:
        """Override this to allow some non-Literal values. For example, the
        MinLen extensions can allow sequences with sufficiently known lengths."""
        return CanAssignError(
            f"Cannot determine whether {value} fulfills predicate {self}"
        )


@dataclass(frozen=True)
class EnumName(AnnotatedTypesCheck):
    enum_cls: type[enum.Enum]

    def predicate(self, value: str) -> bool:
        try:
            self.enum_cls[value]
        except KeyError:
            return False
        else:
            return True


@dataclass(frozen=True)
class Gt(AnnotatedTypesCheck):
    value: Any

    def predicate(self, value: Any) -> bool:
        return value > self.value

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if isinstance(metadata, Gt):
            return metadata.value >= self.value
        elif isinstance(metadata, Ge):
            return metadata.value > self.value
        else:
            return False


@dataclass(frozen=True)
class Ge(AnnotatedTypesCheck):
    value: Any

    def predicate(self, value: Any) -> bool:
        return value >= self.value

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if isinstance(metadata, Gt):
            return metadata.value >= self.value
        elif isinstance(metadata, Ge):
            return metadata.value >= self.value
        else:
            return False


@dataclass(frozen=True)
class Lt(AnnotatedTypesCheck):
    value: Any

    def predicate(self, value: Any) -> bool:
        return value < self.value

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if isinstance(metadata, Lt):
            return metadata.value <= self.value
        elif isinstance(metadata, Le):
            return metadata.value < self.value
        else:
            return False


@dataclass(frozen=True)
class Le(AnnotatedTypesCheck):
    value: Any

    def predicate(self, value: Any) -> bool:
        return value <= self.value

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if isinstance(metadata, Lt):
            return metadata.value <= self.value
        elif isinstance(metadata, Le):
            return metadata.value <= self.value
        else:
            return False


@dataclass(frozen=True)
class MultipleOf(AnnotatedTypesCheck):
    value: Any

    def predicate(self, value: Any) -> bool:
        return value % self.value == 0

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if isinstance(metadata, MultipleOf):
            # If we want a MultipleOf(5), but we're passed a MultipleOf(10), that's ok
            return metadata.value % self.value == 0
        else:
            return False


@dataclass(frozen=True)
class Timezone(AnnotatedTypesCheck):
    value: str | timezone | tzinfo | type(...) | None

    def predicate(self, value: Any) -> bool:
        if not isinstance(value, datetime):
            return False
        if self.value is None:
            return value.tzinfo is None
        elif self.value is ...:
            return value.tzinfo is not None
        elif isinstance(self.value, timezone):
            return value.tzinfo == self.value
        else:
            return False

    def is_compatible_metadata(self, metadata: PredicateCheck) -> bool:
        if (
            self.value is ...
            and isinstance(metadata, Timezone)
            and metadata.value is not None
        ):
            return True
        return False


@dataclass(frozen=True)
class AnnotatedPredicate(AnnotatedTypesCheck):
    predicate_callable: Callable[[Any], bool]

    def predicate(self, value: Any) -> bool:
        return self.predicate_callable(value)
