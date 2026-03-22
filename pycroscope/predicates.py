"""

Reusable predicates.

"""

import enum
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import comb

from typing_extensions import assert_never

from pycroscope.value import TypeVarMap

from .boolability import Boolability, get_boolability
from .relations import Relation, has_relation, intersect_values, is_subtype
from .safe import safe_issubclass
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnySource,
    AnyValue,
    CanAssignContext,
    CanAssignError,
    DictIncompleteValue,
    GenericValue,
    GradualType,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    Predicate,
    PredicateValue,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    freshen_typevars_for_inference,
    is_overlapping,
    replace_fallback,
    unannotate,
    unite_values,
)


def is_universally_assignable(value: Value, target_value: Value) -> bool:
    if isinstance(value, TypeVarValue):
        return True
    value = replace_fallback(value)
    if value is NO_RETURN_VALUE or isinstance(value, AnyValue):
        return True
    elif value == TypedValue(type) and isinstance(target_value, SubclassValue):
        return True
    elif isinstance(value, MultiValuedValue):
        return all(
            is_universally_assignable(subval, target_value) for subval in value.vals
        )
    elif isinstance(value, IntersectionValue):
        return all(
            is_universally_assignable(subval, target_value) for subval in value.vals
        )
    return False


@dataclass
class IsAssignablePredicate:
    """Predicate that filters out values that are not assignable to pattern_value.

    This only works reliably for simple pattern_values, such as TypedValue.

    """

    pattern_value: Value
    ctx: CanAssignContext
    positive_only: bool

    def __call__(self, value: Value, positive: bool) -> Value | None:
        pattern_value = freshen_typevars_for_inference(self.pattern_value)
        compatible = is_overlapping(pattern_value, value, self.ctx)
        if positive:
            if not compatible:
                return None
            if pattern_value.is_assignable(value, self.ctx):
                if is_universally_assignable(value, unannotate(pattern_value)):
                    return pattern_value
                return value
            else:
                return pattern_value
        elif not self.positive_only:
            if pattern_value.is_assignable(
                value, self.ctx
            ) and not is_universally_assignable(value, unannotate(pattern_value)):
                return None
        return value


_OPERATOR = {
    (True, True): operator.is_,
    (False, True): operator.is_not,
    (True, False): operator.eq,
    (False, False): operator.ne,
}


@dataclass
class EqualsPredicate:
    """Predicate that filters out values that are not equal to pattern_val."""

    pattern_val: object
    ctx: CanAssignContext
    use_is: bool = False

    def __call__(self, value: Value, positive: bool) -> Value | None:
        inner_value = unannotate(value)
        if isinstance(inner_value, KnownValue):
            op = _OPERATOR[(positive, self.use_is)]
            try:
                result = op(inner_value.val, self.pattern_val)
            except Exception:
                pass
            else:
                if not result:
                    return None
        elif positive:
            known_self = KnownValue(self.pattern_val)
            if value.is_assignable(known_self, self.ctx):
                return known_self
            else:
                return None
        else:
            pattern_type = type(self.pattern_val)
            if pattern_type is bool:
                simplified = unannotate(value)
                if isinstance(simplified, TypedValue) and simplified.typ is bool:
                    return KnownValue(not self.pattern_val)
            elif safe_issubclass(pattern_type, enum.Enum) and not safe_issubclass(
                pattern_type, enum.Flag
            ):
                simplified = unannotate(value)
                if isinstance(simplified, TypedValue) and simplified.typ is type(
                    self.pattern_val
                ):
                    return unite_values(
                        *[
                            KnownValue(val)
                            for val in pattern_type
                            if val is not self.pattern_val
                        ]
                    )
        return value


@dataclass
class InPredicate:
    """Predicate that filters out values that are not in pattern_vals."""

    pattern_vals: Sequence[object]
    pattern_type: type
    ctx: CanAssignContext

    def __call__(self, value: Value, positive: bool) -> Value | None:
        inner_value = unannotate(value)
        if isinstance(inner_value, KnownValue):
            try:
                if positive:
                    result = inner_value.val in self.pattern_vals
                else:
                    result = inner_value.val not in self.pattern_vals
            except Exception:
                pass
            else:
                if not result:
                    return None
        elif positive:
            acceptable_values = [
                KnownValue(pattern_val)
                for pattern_val in self.pattern_vals
                if value.is_assignable(KnownValue(pattern_val), self.ctx)
            ]
            if acceptable_values:
                return unite_values(*acceptable_values)
            else:
                return None
        else:
            if safe_issubclass(self.pattern_type, enum.Enum) and not safe_issubclass(
                self.pattern_type, enum.Flag
            ):
                simplified = unannotate(value)
                if (
                    isinstance(simplified, TypedValue)
                    and simplified.typ is self.pattern_type
                ):
                    return unite_values(
                        *[
                            KnownValue(val)
                            for val in self.pattern_type
                            if val not in self.pattern_vals
                        ]
                    )
        return value


@dataclass(frozen=True)
class HasAttr(Predicate):
    attr: str
    value: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Predicate:
        return HasAttr(self.attr, self.value.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield from self.value.walk_values()

    def __str__(self) -> str:
        return f"x.{self.attr}: {self.value}"

    def has_relation_simple_type(
        self, other: SimpleType, relation: Relation, ctx: CanAssignContext
    ) -> bool:
        match other:
            case AnyValue():
                if relation is Relation.SUBTYPE:
                    return False
                else:
                    return True
            case TypeFormValue():
                return False
            case PredicateValue():
                if (
                    isinstance(other.predicate, HasAttr)
                    and self.attr == other.predicate.attr
                ):
                    can_assign = has_relation(
                        self.value, other.predicate.value, relation, ctx
                    )
                    return not isinstance(can_assign, CanAssignError)
                else:
                    return False
            case _:
                attr_value = ctx.get_attribute_from_value(other, self.attr)
                if attr_value is UNINITIALIZED_VALUE:
                    return False
                can_assign = has_relation(self.value, attr_value, relation, ctx)
                return not isinstance(can_assign, CanAssignError)

    def intersect_with_simple_type(
        self, other: SimpleType, ctx: CanAssignContext
    ) -> Value | None:
        match other:
            case AnyValue() | TypeFormValue():
                return None
            case PredicateValue():
                if (
                    isinstance(other.predicate, HasAttr)
                    and self.attr == other.predicate.attr
                ):
                    intersected_value = intersect_values(
                        self.value, other.predicate.value, ctx
                    )
                    if intersected_value is NO_RETURN_VALUE:
                        return NO_RETURN_VALUE
                    else:
                        return PredicateValue(HasAttr(self.attr, intersected_value))
                else:
                    return None
            case (
                SyntheticClassObjectValue()
                | SyntheticModuleValue()
                | KnownValue()
                | UnboundMethodValue()
            ):
                return self._intersect_with_value(other, ctx, is_final=True)
            case TypedValue():
                tobj = ctx.make_type_object(other.typ)
                return self._intersect_with_value(other, ctx, is_final=tobj.is_final)
            case SubclassValue():
                return self._intersect_with_value(other, ctx, is_final=False)
            case _:
                assert_never(other)

    def _intersect_with_value(
        self, other: Value, ctx: CanAssignContext, *, is_final: bool
    ) -> Value | None:
        attr_value = ctx.get_attribute_from_value(other, self.attr)
        if attr_value is UNINITIALIZED_VALUE:
            return NO_RETURN_VALUE if is_final else None
        intersected_value = intersect_values(self.value, attr_value, ctx)
        if intersected_value is NO_RETURN_VALUE:
            return NO_RETURN_VALUE
        elif is_subtype(self.value, attr_value, ctx):
            # e.g. if we have HasAttr("x", object) intersecting with something with x: int,
            # we can drop the HasAttr
            return other
        else:
            return None


@dataclass(frozen=True)
class MaxLen(Predicate):
    """Predicate that the value has a length no greater than the specified maximum."""

    max_len: int

    def intersect_with_simple_type(
        self, other: SimpleType, ctx: CanAssignContext
    ) -> Value | None:
        return _intersect_min_or_max_len(other, min_len=None, max_len=self.max_len)

    def has_relation_simple_type(
        self, other: SimpleType, relation: Relation, ctx: CanAssignContext
    ) -> bool:
        return _has_relation_min_or_max_len(
            other, relation, min_len=None, max_len=self.max_len
        )

    def __str__(self) -> str:
        return f"len(x) <= {self.max_len}"


@dataclass(frozen=True)
class MinLen(Predicate):
    """Predicate that the value has a length no less than the specified minimum."""

    min_len: int

    def intersect_with_simple_type(
        self, other: SimpleType, ctx: CanAssignContext
    ) -> Value | None:
        return _intersect_min_or_max_len(other, min_len=self.min_len, max_len=None)

    def has_relation_simple_type(
        self, other: SimpleType, relation: Relation, ctx: CanAssignContext
    ) -> bool:
        return _has_relation_min_or_max_len(
            other, relation, min_len=self.min_len, max_len=None
        )

    def __str__(self) -> str:
        return f"len(x) >= {self.min_len}"


def _has_relation_min_or_max_len(
    value: SimpleType, relation: Relation, *, min_len: int | None, max_len: int | None
) -> bool:
    match value:
        case AnyValue():
            if relation is Relation.SUBTYPE:
                return False
            else:
                return True
        case SyntheticClassObjectValue() | SubclassValue():
            # TODO: check if it has __len__ on the metaclass; return Never if it doesn't
            return False
        case TypeFormValue() | SyntheticModuleValue() | UnboundMethodValue():
            return False
        case PredicateValue():
            match value.predicate:
                case MaxLen(max_len2) if max_len is not None:
                    return max_len2 <= max_len
                case MinLen(min_len2) if min_len is not None:
                    return min_len2 >= min_len
                case _:
                    return False
        case KnownValue():
            try:
                length = len(value.val)
            except Exception:
                return False
            if min_len is not None and length < min_len:
                return False
            if max_len is not None and length > max_len:
                return False
            return True
        case TypedValue():
            value_min, value_max = _get_len_bounds(value)
            if min_len is not None and value_min is not None:
                return value_min >= min_len
            if max_len is not None and value_max is not None:
                return value_max <= max_len
            return False
        case _:
            assert_never(value)


def _intersect_min_or_max_len(
    value: SimpleType, *, min_len: int | None, max_len: int | None
) -> Value | None:
    match value:
        case AnyValue() | TypeFormValue():
            return None
        case SyntheticClassObjectValue() | SubclassValue():
            # TODO: check if it has __len__ on the metaclass; return Never if it doesn't
            return None
        case SyntheticModuleValue() | UnboundMethodValue():
            return NO_RETURN_VALUE
        case PredicateValue():
            match value.predicate:
                case MaxLen(max_len2) if max_len is not None:
                    return PredicateValue(MaxLen(min(max_len, max_len2)))
                case MinLen(min_len2) if min_len is not None:
                    return PredicateValue(MinLen(max(min_len, min_len2)))
                case MaxLen(max_len2) if min_len is not None and max_len2 < min_len:
                    return NO_RETURN_VALUE
                case MinLen(min_len2) if max_len is not None and min_len2 > max_len:
                    return NO_RETURN_VALUE
                case _:
                    return None
        case KnownValue():
            try:
                length = len(value.val)
            except Exception:
                return NO_RETURN_VALUE
            if min_len is not None and length < min_len:
                return NO_RETURN_VALUE
            if max_len is not None and length > max_len:
                return NO_RETURN_VALUE
            return value
        case TypedValue():
            value_min, value_max = _get_len_bounds(value)
            if min_len is not None and value_max is not None and value_max < min_len:
                return NO_RETURN_VALUE
            if max_len is not None and value_min is not None and value_min > max_len:
                return NO_RETURN_VALUE
            if isinstance(value, SequenceValue):
                if max_len is not None:
                    expanded = _expand_tuple_members_to_max_len(value.members, max_len)
                    if expanded is None:
                        return None
                    return unite_values(
                        *(
                            SequenceValue(value.typ, members=members)
                            for members in expanded
                        )
                    )
                if min_len is not None:
                    expanded = _expand_tuple_members_to_min_len(value.members, min_len)
                    if expanded is None:
                        return None
                    return unite_values(
                        *(
                            SequenceValue(value.typ, members=members)
                            for members in expanded
                        )
                    )
            elif value.typ is tuple:
                if isinstance(value, GenericValue):
                    type_arg = value.args[0]
                else:
                    type_arg = AnyValue(AnySource.generic_argument)
                if max_len is not None:
                    expanded = _expand_tuple_members_to_max_len(
                        ((True, type_arg),), max_len
                    )
                    if expanded is None:
                        return None
                    return unite_values(
                        *(
                            SequenceValue(value.typ, members=members)
                            for members in expanded
                        )
                    )
                if min_len is not None:
                    expanded = _expand_tuple_members_to_min_len(
                        ((True, type_arg),), min_len
                    )
                    if expanded is None:
                        return None
                    return unite_values(
                        *(
                            SequenceValue(value.typ, members=members)
                            for members in expanded
                        )
                    )
            return None
        case _:
            assert_never(value)


def _get_len_bounds(value: TypedValue) -> tuple[int | None, int | None]:
    if isinstance(value, SequenceValue):
        min_len = 0
        max_len = 0
        for is_many, _ in value.members:
            if is_many:
                max_len = None
            else:
                min_len += 1
                if max_len is not None:
                    max_len += 1
        return min_len, max_len
    if isinstance(value, GenericValue) and value.typ is tuple and len(value.args) == 1:
        return 0, None
    if isinstance(value, DictIncompleteValue):
        min_len = 0
        max_len = 0
        has_unbounded_tail = False
        for pair in value.kv_pairs:
            if pair.is_many:
                has_unbounded_tail = True
            else:
                if pair.is_required:
                    min_len += 1
                max_len += 1
        if has_unbounded_tail:
            return min_len, None
        return min_len, max_len
    if isinstance(value, TypedDictValue):
        min_len = sum(entry.required for entry in value.items.values())
        if value.extra_keys is not NO_RETURN_VALUE:
            return min_len, None
        return min_len, len(value.items)
    return None, None


_MAX_EXACT_TUPLE_LENGTH = 64
_MAX_EXACT_TUPLE_EXPANSIONS = 128


def _iter_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _iter_compositions(total - first, parts - 1):
            yield (first, *rest)


def _expand_tuple_members_to_exact_len(
    members: tuple[tuple[bool, Value], ...], target_len: int
) -> list[tuple[tuple[bool, Value], ...]] | None:
    if target_len < 0:
        return []
    if target_len > _MAX_EXACT_TUPLE_LENGTH:
        return None
    fixed_count = sum(not is_many for is_many, _ in members)
    many_values = [value for is_many, value in members if is_many]
    if target_len < fixed_count:
        return []
    if not many_values:
        if target_len == fixed_count:
            return [members]
        return []
    extra = target_len - fixed_count
    if len(many_values) > 1:
        expansions = comb(extra + len(many_values) - 1, len(many_values) - 1)
        if expansions > _MAX_EXACT_TUPLE_EXPANSIONS:
            return None
    result = []
    for counts in _iter_compositions(extra, len(many_values)):
        many_index = 0
        expanded = []
        for is_many, member in members:
            if not is_many:
                expanded.append((False, member))
                continue
            expanded.extend((False, member) for _ in range(counts[many_index]))
            many_index += 1
        result.append(tuple(expanded))
    return result


def _expand_tuple_members_to_min_len(
    members: tuple[tuple[bool, Value], ...], target_len: int
) -> list[tuple[tuple[bool, Value], ...]] | None:
    if target_len < 0:
        return [members]
    if target_len > _MAX_EXACT_TUPLE_LENGTH:
        return None
    fixed_count = sum(not is_many for is_many, _ in members)
    many_values = [value for is_many, value in members if is_many]
    if target_len <= fixed_count:
        return [members]
    if not many_values:
        return []
    needed = target_len - fixed_count
    if len(many_values) > 1:
        expansions = comb(needed + len(many_values) - 1, len(many_values) - 1)
        if expansions > _MAX_EXACT_TUPLE_EXPANSIONS:
            return None
    result = []
    for counts in _iter_compositions(needed, len(many_values)):
        many_index = 0
        expanded = []
        for is_many, member in members:
            if not is_many:
                expanded.append((False, member))
                continue
            expanded.extend((False, member) for _ in range(counts[many_index]))
            expanded.append((True, member))
            many_index += 1
        result.append(tuple(expanded))
    return result


def _expand_tuple_members_to_max_len(
    members: tuple[tuple[bool, Value], ...], target_len: int
) -> list[tuple[tuple[bool, Value], ...]] | None:
    if target_len < 0:
        return []
    if target_len > _MAX_EXACT_TUPLE_LENGTH:
        return None
    fixed_count = sum(not is_many for is_many, _ in members)
    if fixed_count > target_len:
        return []

    expanded = []
    for current_len in range(fixed_count, target_len + 1):
        exact = _expand_tuple_members_to_exact_len(members, current_len)
        if exact is None:
            return None
        expanded.extend(exact)
        if len(expanded) > _MAX_EXACT_TUPLE_EXPANSIONS:
            return None

    deduped = []
    for option in expanded:
        if option not in deduped:
            deduped.append(option)
    return deduped


@dataclass(frozen=True)
class Truthy(Predicate):
    def has_relation(
        self, other: GradualType, relation: Relation, ctx: CanAssignContext
    ) -> bool:
        boolability = get_boolability(other)
        return boolability in (
            Boolability.value_always_true,
            Boolability.type_always_true,
        )

    def intersect_with(self, other: GradualType, ctx: CanAssignContext) -> Value | None:
        match get_boolability(other):
            case Boolability.value_always_true | Boolability.type_always_true:
                return other
            case Boolability.value_always_false:
                return NO_RETURN_VALUE
            case (
                Boolability.value_always_false_mutable
                | Boolability.value_always_true_mutable
                | Boolability.boolable
            ):
                return None


TRUTHY = Truthy()


@dataclass(frozen=True)
class Falsy(Predicate):
    def has_relation(
        self, other: GradualType, relation: Relation, ctx: CanAssignContext
    ) -> bool:
        boolability = get_boolability(other)
        return boolability is Boolability.value_always_false

    def intersect_with(self, other: GradualType, ctx: CanAssignContext) -> Value | None:
        match get_boolability(other):
            case Boolability.value_always_false:
                return other
            case Boolability.value_always_true | Boolability.type_always_true:
                return NO_RETURN_VALUE
            case (
                Boolability.value_always_true_mutable
                | Boolability.value_always_false_mutable
                | Boolability.boolable
            ):
                return None


FALSY = Falsy()
