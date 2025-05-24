"""

Relations between types.

See the typing spec:
https://typing.python.org/en/latest/spec/concepts.html#summary-of-type-relations

"""

import collections.abc
import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from types import FunctionType
from typing import Optional, Protocol, Union

from typing_extensions import Literal, TypeAlias, assert_never

import pycroscope
from pycroscope.find_unused import used
from pycroscope.safe import safe_equals, safe_isinstance
from pycroscope.typevar import resolve_bounds_map
from pycroscope.value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    BoundsMap,
    CallableValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    DictIncompleteValue,
    Extension,
    GenericValue,
    IterableValue,
    KnownValue,
    LowerBound,
    MultiValuedValue,
    NewTypeValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    SequenceValue,
    SubclassValue,
    SyntheticModuleValue,
    T,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeVarMap,
    TypeVarValue,
    UnboundMethodValue,
    UpperBound,
    Value,
    VariableNameValue,
    flatten_values,
    intersect_bounds_maps,
    stringify_object,
    typify_literal,
    unify_bounds_maps,
)


class Relation(enum.Enum):
    SUBTYPE = 1
    ASSIGNABLE = 2
    CONSISTENT = 3
    EQUIVALENT = 4

    @property
    def description(self) -> str:
        if self is Relation.SUBTYPE:
            return "a subtype of"
        elif self is Relation.ASSIGNABLE:
            return "assignable to"
        elif self is Relation.CONSISTENT:
            return "consistent with"
        elif self is Relation.EQUIVALENT:
            return "equivalent to"
        else:
            assert_never(self)


# Subclasses of Value that represent real types in the type system.
GradualType: TypeAlias = Union[
    AnyValue,
    TypeAliasValue,
    KnownValue,
    SyntheticModuleValue,
    UnboundMethodValue,
    NewTypeValue,
    TypedValue,
    SubclassValue,
    MultiValuedValue,
    TypeVarValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    AnnotatedValue,
]

GRADUAL_TYPE = GradualType.__args__


class NotAGradualType(Exception):
    """Raised when a value is not a gradual type."""


@used
def is_equivalent(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    """Return whether ``left`` and ``right`` are equivalent types."""
    result = has_relation(left, right, Relation.EQUIVALENT, ctx)
    return not isinstance(result, CanAssignError)


@used
def is_equivalent_with_reason(
    left: Value, right: Value, ctx: CanAssignContext
) -> CanAssign:
    """Return whether ``left`` and ``right`` are equivalent types."""
    return has_relation(left, right, Relation.EQUIVALENT, ctx)


@used
def is_consistent(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    """Return whether ``left`` and ``right`` are consistent types."""
    result = has_relation(left, right, Relation.CONSISTENT, ctx)
    return not isinstance(result, CanAssignError)


@used
def is_consistent_with_reason(
    left: Value, right: Value, ctx: CanAssignContext
) -> CanAssign:
    """Return whether ``left`` and ``right`` are consistent types."""
    return has_relation(left, right, Relation.CONSISTENT, ctx)


@used
def is_assignable(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    """Return whether ``right`` is assignable to ``left``."""
    result = has_relation(left, right, Relation.ASSIGNABLE, ctx)
    return not isinstance(result, CanAssignError)


@used
def is_assignable_with_reason(
    left: Value, right: Value, ctx: CanAssignContext
) -> CanAssign:
    """Return whether ``right`` is assignable to ``left``."""
    return has_relation(left, right, Relation.ASSIGNABLE, ctx)


@used
def is_subtype(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    """Return whether ``right`` is a subtype of ``left``."""
    result = has_relation(left, right, Relation.SUBTYPE, ctx)
    if isinstance(result, CanAssignError):
        return False
    return True


@used
def is_subtype_with_reason(
    left: Value, right: Value, ctx: CanAssignContext
) -> CanAssign:
    """Return whether ``right`` is a subtype of ``left``."""
    return has_relation(left, right, Relation.SUBTYPE, ctx)


def has_relation(
    left: Value, right: Value, relation: Relation, ctx: CanAssignContext
) -> CanAssign:
    left = _gradualize(left)
    right = _gradualize(right)
    if relation is Relation.EQUIVALENT:
        # A is equivalent to B if A is a subtype of B and B is a subtype of A.
        result1 = _has_relation(left, right, Relation.SUBTYPE, ctx)
        result2 = _has_relation(right, left, Relation.SUBTYPE, ctx)
        if isinstance(result1, CanAssignError) or isinstance(result2, CanAssignError):
            children = [
                elt for elt in (result1, result2) if isinstance(elt, CanAssignError)
            ]
            return CanAssignError(
                f"{left} is not {relation.description} {right}", children=children
            )
        return unify_bounds_maps([result1, result2])
    elif relation is Relation.CONSISTENT:
        # A is consistent with B if A is assignable to B and B is assignable to A.
        result1 = _has_relation(left, right, Relation.ASSIGNABLE, ctx)
        result2 = _has_relation(right, left, Relation.ASSIGNABLE, ctx)
        if isinstance(result1, CanAssignError) or isinstance(result2, CanAssignError):
            children = [
                elt for elt in (result1, result2) if isinstance(elt, CanAssignError)
            ]
            return CanAssignError(
                f"{left} is not {relation.description} {right}", children=children
            )
        return unify_bounds_maps([result1, result2])
    else:
        return _has_relation(left, right, relation, ctx)


def _gradualize(value: Value) -> GradualType:
    if not isinstance(value, GRADUAL_TYPE):
        raise NotAGradualType(f"Encountered non-type {value}")
    return value


def _has_relation(
    left: GradualType,
    right: GradualType,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    # TypeVarValue
    if isinstance(left, TypeVarValue):
        if left == right:
            return {}
        if isinstance(right, TypeVarValue):
            bounds = [*left.get_inherent_bounds(), *right.get_inherent_bounds()]
        else:
            bounds = [LowerBound(left.typevar, right), *left.get_inherent_bounds()]
        return left.make_bounds_map(bounds, right, ctx)
    if isinstance(right, TypeVarValue) and not isinstance(left, MultiValuedValue):
        bounds = [UpperBound(right.typevar, left), *right.get_inherent_bounds()]
        return right.make_bounds_map(bounds, left, ctx)

    # TypeAliasValue
    if isinstance(left, TypeAliasValue):
        if isinstance(right, TypeAliasValue):
            if left.alias is right.alias or ctx.can_aliases_assume_compatibility(
                left, right
            ):
                return {}
            with ctx.aliases_assume_compatibility(left, right):
                left_inner = _gradualize(left.get_value())
                return _has_relation(left_inner, right, relation, ctx)
        left_inner = _gradualize(left.get_value())
        return _has_relation(left_inner, right, relation, ctx)
    if isinstance(right, TypeAliasValue):
        right_inner = _gradualize(right.get_value())
        return _has_relation(left, right_inner, relation, ctx)

    # AnnotatedValue
    if isinstance(left, AnnotatedValue):
        left_inner = _gradualize(left.value)
        can_assign = _has_relation(left_inner, right, relation, ctx)
        if isinstance(can_assign, CanAssignError):
            return can_assign
        bounds_maps = [can_assign]
        for ext in left.get_metadata_of_type(Extension):
            custom_can_assign = ext.can_assign(right, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            bounds_maps.append(custom_can_assign)
        return unify_bounds_maps(bounds_maps)
    if isinstance(right, AnnotatedValue) and not isinstance(left, MultiValuedValue):
        right_inner = _gradualize(right.value)
        can_assign = _has_relation(left, right_inner, relation, ctx)
        if isinstance(can_assign, CanAssignError):
            return can_assign
        bounds_maps = [can_assign]
        for ext in right.get_metadata_of_type(Extension):
            custom_can_assign = ext.can_be_assigned(left, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            bounds_maps.append(custom_can_assign)
        return unify_bounds_maps(bounds_maps)

    # Never (special case of MultiValuedValue)
    if right is NO_RETURN_VALUE:
        return {}
    if left is NO_RETURN_VALUE:
        return CanAssignError(f"{right} is not {relation.description} {left}")

    # MultiValuedValue
    if isinstance(left, MultiValuedValue):
        if isinstance(right, MultiValuedValue):
            bounds_maps = []
            for val in right.vals:
                val = _gradualize(val)
                can_assign = _has_relation(left, val, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    # Adding an additional layer here isn't helpful
                    return can_assign
                bounds_maps.append(can_assign)
            return unify_bounds_maps(bounds_maps)
        else:
            # right is a subtype if it's a subtype of any of the members
            bounds_maps = []
            errors = []
            for val in left.vals:
                val = _gradualize(val)
                can_assign = _has_relation(val, right, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    errors.append(can_assign)
                else:
                    bounds_maps.append(can_assign)
            if not bounds_maps:
                return CanAssignError(
                    f"{right} is not {relation.description} {left}", children=errors
                )
            return intersect_bounds_maps(bounds_maps)
    if isinstance(right, MultiValuedValue):
        # right is a subtype if all the members are subtypes of left
        bounds_maps = []
        for val in right.vals:
            val = _gradualize(val)
            can_assign = _has_relation(left, val, relation, ctx)
            if isinstance(can_assign, CanAssignError):
                # Adding an additional layer here isn't helpful
                return can_assign
            bounds_maps.append(can_assign)
        return unify_bounds_maps(bounds_maps)
    assert not isinstance(right, (TypeVarValue, AnnotatedValue))

    # AnyValue
    if isinstance(left, AnyValue):
        if (
            isinstance(left, VariableNameValue)
            and isinstance(right, VariableNameValue)
            and left != right
        ):
            return CanAssignError(f"Types {left} and {right} are different")
        if isinstance(right, AnyValue):
            # Any is a subtype etc. of itself
            return {}
        else:
            if relation is Relation.SUBTYPE:
                return CanAssignError("No type is a subtype of Any")
            elif relation is Relation.ASSIGNABLE:
                return {}  # everything is assignable to Any
            else:
                assert_never(relation)
    if isinstance(right, AnyValue):
        if relation is Relation.SUBTYPE:
            return CanAssignError("Any is not a subtype of anything")
        elif relation is Relation.ASSIGNABLE:
            return {}  # Any is assignable to everything
        else:
            assert_never(relation)

    # SyntheticModuleValue
    if isinstance(left, SyntheticModuleValue):
        if isinstance(right, SyntheticModuleValue):
            if left.module_path == right.module_path:
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, SyntheticModuleValue):
        return CanAssignError(f"{right} is not {relation.description} {left}")

    # ParamSpecArgs and Kwargs
    if isinstance(left, ParamSpecArgsValue):
        if (
            isinstance(right, ParamSpecArgsValue)
            and left.param_spec is right.param_spec
        ):
            # TODO: This isn't quite right, the "same" ParamSpec may refer to a different scope.
            return {}
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, ParamSpecArgsValue):
        return has_relation(left, right.get_fallback_value(), relation, ctx)
    if isinstance(left, ParamSpecKwargsValue):
        if (
            isinstance(right, ParamSpecKwargsValue)
            and left.param_spec is right.param_spec
        ):
            return {}
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, ParamSpecKwargsValue):
        return has_relation(left, right.get_fallback_value(), relation, ctx)

    # NewTypeValue
    if isinstance(left, NewTypeValue):
        if isinstance(right, NewTypeValue):
            if left.newtype is right.newtype:
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, NewTypeValue):
        right_inner = _gradualize(right.value)
        return _has_relation(left, right_inner, relation, ctx)

    # UnboundMethodValue
    if isinstance(left, UnboundMethodValue):
        if isinstance(right, UnboundMethodValue) and left == right:
            return {}
        sig = left.get_signature(ctx)
        if sig is not None:
            return _has_relation(CallableValue(sig), right, relation, ctx)
        return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, UnboundMethodValue):
        sig = right.get_signature(ctx)
        if sig is None:
            return CanAssignError(f"{right} is not {relation.description} {left}")
        return _has_relation(left, CallableValue(sig), relation, ctx)

    if left is HashableProtoValue:
        # Protocol doesn't deal well with type.__hash__ at the moment, so to make
        # sure types are recognized as hashable, we use this custom object.
        if isinstance(right, SubclassValue):
            return {}
        elif isinstance(right, TypedValue) and right.typ is type:
            return {}
        # And that means we also get to use this more direct check for KnownValue
        elif isinstance(right, KnownValue):
            try:
                hash(right.val)
            except Exception as e:
                return CanAssignError(
                    f"{right.val!r} is not hashable", children=[CanAssignError(repr(e))]
                )
            else:
                return {}

    # SubclassValue
    if isinstance(left, SubclassValue):
        if isinstance(right, SubclassValue):
            return _has_relation(left.typ, right.typ, relation, ctx)
        elif isinstance(right, KnownValue):
            if not safe_isinstance(right.val, type):
                return CanAssignError(f"{right} is not a type")
            elif isinstance(left.typ, TypeVarValue):
                return {
                    left.typ.typevar: [
                        LowerBound(left.typ.typevar, TypedValue(right.val))
                    ]
                }
            elif isinstance(left.typ, TypedValue):
                left_tobj = left.typ.get_type_object(ctx)
                return left_tobj.can_assign(left, TypedValue(right.val), ctx)
            else:
                assert_never(left.typ)
        elif isinstance(right, TypedValue):
            # metaclass
            right_tobj = right.get_type_object(ctx)
            if not right_tobj.is_assignable_to_type(type):
                return CanAssignError(f"{right} is not a type")
            if isinstance(left.typ, TypeVarValue):
                return {left.typ.typevar: [LowerBound(left.typ.typevar, right)]}
            elif isinstance(left.typ, TypedValue):
                if right_tobj.is_metatype_of(left.typ.get_type_object(ctx)):
                    return {}
                return CanAssignError(f"{right} is not {relation.description} {left}")
            else:
                assert_never(left.typ)
        else:
            assert_never(right)
    if isinstance(right, SubclassValue):
        if isinstance(left, KnownValue):
            return CanAssignError(f"{right} is not {relation.description} {left}")
        elif isinstance(left, TypedValue):
            left_tobj = left.get_type_object(ctx)
            if isinstance(right.typ, TypedValue):
                return left_tobj.can_assign(left, right, ctx)
            elif isinstance(right.typ, TypeVarValue):
                return {right.typ.typevar: [UpperBound(right.typ.typevar, left)]}
            else:
                assert_never(right.typ)
        else:
            assert_never(left)

    # Special case for thrift enums
    if isinstance(left, TypedValue):
        left_tobj = left.get_type_object(ctx)
        if left_tobj.is_thrift_enum:
            return _has_relation_thrift_enum(left, right, relation, ctx)

    # KnownValue
    if isinstance(left, KnownValue):
        # Make Literal[function] equivalent to a Callable type
        if isinstance(left.val, FunctionType):
            signature = ctx.get_signature(left.val)
            if signature is not None:
                return _has_relation(CallableValue(signature), right, relation, ctx)
        if isinstance(right, KnownValue):
            if left.val is right.val:
                return {}
            elif safe_equals(left.val, right.val) and type(left.val) is type(right.val):
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        elif isinstance(right, TypedValue):
            return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            assert_never(right)

    if isinstance(left, CallableValue):
        signature = ctx.signature_from_value(right)
        if isinstance(signature, pycroscope.signature.BoundMethodSignature):
            signature = signature.get_signature(ctx=ctx)
        if signature is None:
            return CanAssignError(f"{right} is not a callable type")
        return pycroscope.signature.signatures_have_relation(
            left.signature, signature, relation, ctx
        )

    if isinstance(right, KnownValue):
        right = typify_literal(right)

    # TypedValue
    if isinstance(left, SequenceValue):
        if isinstance(right, SequenceValue):
            return _has_relation_sequence(left, right, relation, ctx)
        elif relation is Relation.SUBTYPE:
            return CanAssignError(f"{right} is not {relation.description} {left}")
        elif relation is Relation.ASSIGNABLE:
            if (
                isinstance(right, TypedValue)
                and left.typ is right.typ
                and (
                    type(right) is TypedValue
                    or (
                        isinstance(right, GenericValue)
                        and all(
                            is_equivalent(arg, AnyValue(AnySource.inference), ctx)
                            for arg in right.args
                        )
                    )
                )
            ):
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            assert_never(relation)
    if isinstance(left, TypedDictValue):
        if isinstance(right, TypedDictValue):
            return _has_relation_typeddict(left, right, relation, ctx)
        elif isinstance(right, DictIncompleteValue):
            return _has_relation_typeddict_dict(left, right, relation, ctx)
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")

    if isinstance(left, GenericValue):
        if isinstance(right, TypedValue) and not isinstance(right.typ, super):
            generic_args = right.get_generic_args_for_type(left.typ, ctx)
            # If we don't think it's a generic base, try super;
            # runtime isinstance() may disagree.
            if generic_args is not None and len(left.args) == len(generic_args):
                bounds_maps = []
                for i, (my_arg, their_arg) in enumerate(zip(left.args, generic_args)):
                    can_assign = has_relation(my_arg, their_arg, relation, ctx)
                    if isinstance(can_assign, CanAssignError):
                        return _maybe_specify_error_for_generic(
                            i, left, right, can_assign, relation, ctx
                        )
                    bounds_maps.append(can_assign)
                if not bounds_maps:
                    return CanAssignError(
                        f"{right} is not {relation.description} to {left}"
                    )
                return unify_bounds_maps(bounds_maps)

    if isinstance(left, TypedValue):
        left_tobj = left.get_type_object(ctx)
        if isinstance(right, TypedValue):
            if left.literal_only and not right.literal_only:
                return CanAssignError(f"{right} is not a literal")
            return left_tobj.can_assign(left, right, ctx)
        elif isinstance(right, KnownValue):
            can_assign = left_tobj.can_assign(left, right, ctx)
            if isinstance(can_assign, CanAssignError):
                if left_tobj.is_instance(right.val):
                    return {}
            return can_assign
        else:
            assert_never(right)

    assert_never(left)


def _has_relation_thrift_enum(
    left: TypedValue,
    right: Union[TypedValue, KnownValue],
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    if isinstance(right, KnownValue):
        if not isinstance(right.val, int):
            return CanAssignError(f"{right} is not an int")
        assert hasattr(left.typ, "_VALUES_TO_NAMES"), f"{left} is not a Thrift enum"
        if right.val in left.typ._VALUES_TO_NAMES:
            return {}
        return CanAssignError(f"{right} is not {relation.description} {left}")
    elif isinstance(right, TypedValue):
        tobj = right.get_type_object(ctx)
        if tobj.is_assignable_to_type(int):
            return {}
        return left.get_type_object(ctx).can_assign(left, right, ctx)
    else:
        assert_never(right)


def _maybe_specify_error_for_generic(
    i: int,
    left: GenericValue,
    right: Value,
    error: CanAssignError,
    relation: Relation,
    ctx: CanAssignContext,
) -> CanAssignError:
    expected = left.get_arg(i)
    if isinstance(right, DictIncompleteValue) and left.typ in {
        dict,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
    }:
        if i == 0:
            for pair in reversed(right.kv_pairs):
                can_assign = has_relation(expected, pair.key, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(
                        f"In key of key-value pair {pair}", [can_assign]
                    )
        elif i == 1:
            for pair in reversed(right.kv_pairs):
                can_assign = has_relation(expected, pair.value, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(
                        f"In value of key-value pair {pair}", [can_assign]
                    )
    elif isinstance(right, TypedDictValue) and left.typ in {
        dict,
        collections.abc.Mapping,
        collections.abc.MutableMapping,
    }:
        if i == 0:
            for key in right.items:
                can_assign = has_relation(expected, KnownValue(key), relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(f"In TypedDict key {key!r}", [can_assign])
        elif i == 1:
            for key, entry in right.items.items():
                can_assign = has_relation(expected, entry.typ, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(f"In TypedDict key {key!r}", [can_assign])
    elif isinstance(right, SequenceValue) and left.typ in {
        list,
        set,
        tuple,
        collections.abc.Iterable,
        collections.abc.Sequence,
        collections.abc.MutableSequence,
        collections.abc.Container,
        collections.abc.Collection,
    }:
        for i, (_, key) in enumerate(right.members):
            can_assign = has_relation(expected, key, relation, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(f"In element {i}", [can_assign])

    return CanAssignError(f"In generic argument {i} to {left}", [error])


@dataclass(frozen=True)
class _LazySequenceValue(Value):
    seq: SequenceValue
    start_idx: int = 0
    end_idx: int = 0  # i.e., include the entire thing
    prefix: Optional[Value] = None
    suffix: Optional[Value] = None

    def __len__(self) -> int:
        size = len(self.seq.members) - self.start_idx + self.end_idx
        if self.prefix is not None:
            size += 1
        if self.suffix is not None:
            size += 1
        return size

    def __getitem__(self, idx: int) -> tuple[bool, Value]:
        if idx < 0:
            idx += len(self)
        if self.prefix is not None:
            if idx == 0:
                return False, self.prefix
            idx -= 1
        idx += self.start_idx
        inner_len = len(self.seq.members) + self.end_idx
        if idx < inner_len:
            return self.seq.members[idx]
        if idx == inner_len and self.suffix is not None:
            return False, self.suffix
        raise IndexError(idx)

    def add_prefix(self, prefix: Value) -> "_LazySequenceValue":
        if self.prefix is not None:
            raise ValueError("Prefix already set")
        return replace(self, prefix=prefix)

    def add_suffix(self, suffix: Value) -> "_LazySequenceValue":
        if self.suffix is not None:
            raise ValueError("Suffix already set")
        return replace(self, suffix=suffix)

    def slice_left(self) -> "_LazySequenceValue":
        if self.prefix is not None:
            return replace(self, prefix=None)
        return replace(self, start_idx=self.start_idx + 1)

    def slice_right(self) -> "_LazySequenceValue":
        if self.suffix is not None:
            return replace(self, suffix=None)
        return replace(self, end_idx=self.end_idx - 1)

    def get_fallback_value(self) -> SequenceValue:
        members = []
        if self.prefix is not None:
            members.append(self.prefix)
        members.extend(self.seq.members[self.start_idx : -self.end_idx])
        if self.suffix is not None:
            members.append(self.suffix)
        return SequenceValue(self.seq.typ, members)

    def decompose_left(self) -> Iterable["_LazySequenceValue"]:
        if self[0][0]:
            # If the first element is tuple[T, ...], we can decompose it into...
            # ... the case where it is empty
            yield self.slice_left()
            # ... the case where it contains a single element, followed by a new tuple[T, ...]
            yield self.add_prefix(self[0][1])
        else:
            # If the first element is Single, we can't decompose.
            yield self

    def decompose_right(self) -> Iterable["_LazySequenceValue"]:
        if self[-1][0]:
            # If the last element is tuple[T, ...], we can decompose it into...
            # ... the case where it is empty
            yield self.slice_right()
            # ... the case where it contains a single element, followed by a new tuple[T, ...]
            yield self.add_suffix(self[-1][1])
        else:
            # If the last element is Single, we can't decompose.
            yield self

    def __str__(self) -> str:
        return str(self.get_fallback_value())

    def __iter__(self) -> Iterator[tuple[bool, Value]]:
        if self.prefix is not None:
            yield False, self.prefix
        for i in range(self.start_idx, len(self.seq.members) - self.end_idx):
            yield self.seq.members[i]
        if self.suffix is not None:
            yield False, self.suffix


def _has_relation_lazy_sequence(
    a: _LazySequenceValue,
    b: _LazySequenceValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    len_a = len(a)
    len_b = len(b)

    # Is either empty?
    if len_a == 0:
        if len_b == 0:
            return {}
        else:
            return CanAssignError(
                f"Non-empty sequence {b} is not {relation.description} empty sequence {a}"
            )
    if len_b == 0:
        if all(is_many for is_many, _ in a):
            # If all elements are Many, we can still assign empty sequence to it
            return {}
        else:
            return CanAssignError(
                f"Empty sequence {b} is not {relation.description} non-empty sequence {a}"
            )

    # Do both have a Single element on the left?
    if a[0][0] is False and b[0][0] is False:
        # If so, check whether they're compatible
        can_assign = _has_relation(
            _gradualize(a[0][1]), _gradualize(b[0][1]), relation, ctx
        )
        if isinstance(can_assign, CanAssignError):
            if a.start_idx == b.start_idx:
                text = f"position {a.start_idx}"
            else:
                text = f"positions {a.start_idx} and {b.start_idx}"
            return CanAssignError(
                f"Elements at {text} are not compatible", [can_assign]
            )
        return _has_relation_lazy_sequence(
            a.slice_left(), b.slice_left(), relation, ctx
        )

    # Do both have a Single element on the right?
    if a[-1][0] is False and b[-1][0] is False:
        # If so, check whether they're compatible
        can_assign = _has_relation(
            _gradualize(a[-1][1]), _gradualize(b[-1][1]), relation, ctx
        )
        if isinstance(can_assign, CanAssignError):
            a_end = len(a.seq.members) + a.end_idx - 1
            b_end = len(b.seq.members) + b.end_idx - 1
            if a_end == b_end:
                text = f"position {a_end}"
            else:
                text = f"positions {a_end} and {b_end}"
            return CanAssignError(
                f"Elements at {text} are not compatible", [can_assign]
            )
        return _has_relation_lazy_sequence(
            a.slice_right(), b.slice_right(), relation, ctx
        )

    # Do both have a Many on the left and also on the right?
    if a[0][0] is True and b[0][0] is True and a[-1][0] is True and b[-1][0] is True:
        can_assign = _has_relation(
            _gradualize(a[0][1]), _gradualize(b[0][1]), relation, ctx
        )
        if isinstance(can_assign, CanAssignError):
            # If the leftmost Many in a is not compatible, assume it's empty
            # and continue.
            return _has_relation_lazy_sequence(a.slice_left(), b, relation, ctx)
        else:
            # If the leftmost Many is compatible, we can succeed in three ways:
            # 1. Consume A's leftmost and continue. (Example: A = (*object, *int), B = (*object,))
            can_assign1 = _has_relation_lazy_sequence(
                a.slice_left(), b.slice_left(), relation, ctx
            )
            if not isinstance(can_assign1, CanAssignError):
                return can_assign1
            # 2. Consume B's leftmost and continue. (Example: A = (*object,), B = (*object, *int))
            can_assign2 = _has_relation_lazy_sequence(a.slice_left(), b, relation, ctx)
            if not isinstance(can_assign2, CanAssignError):
                return can_assign2
            # 3. Consume both leftmost and continue.
            # (Example: A = (*object, int, *int), B = (*object, int, *int))
            can_assign3 = _has_relation_lazy_sequence(a, b.slice_left(), relation, ctx)
            if not isinstance(can_assign3, CanAssignError):
                return can_assign3
            return CanAssignError(
                f"{b} is not {relation.description} {a}",
                children=[can_assign1, can_assign2, can_assign3],
            )

    # Now there is at least one Many-Single match on either the left or right end.
    # Decompose both sides at once; if we only decompose one at a time, we'll miss
    # some matches.
    if a[0][0]:
        a_decomposed = a.decompose_left()
    else:
        a_decomposed = [a]
    if b[0][0]:
        b_decomposed = b.decompose_left()
    else:
        b_decomposed = [b]

    if a[-1][0]:
        a_decomposed = [a_dd for a_d in a_decomposed for a_dd in a_d.decompose_right()]
    if b[-1][0]:
        b_decomposed = [b_dd for b_d in b_decomposed for b_dd in b_d.decompose_right()]

    return _has_relation_lazy_seq_multi(a_decomposed, b_decomposed, relation, ctx)


def _has_relation_lazy_seq_multi(
    a_iter: Iterable[_LazySequenceValue],
    b_iter: Iterable[_LazySequenceValue],
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    bounds_maps = []
    for b in b_iter:
        errors = []
        inner_bounds_maps = []
        for a in a_iter:
            can_assign = _has_relation_lazy_sequence(a, b, relation, ctx)
            if isinstance(can_assign, CanAssignError):
                errors.append(can_assign)
            else:
                inner_bounds_maps.append(can_assign)
        if not inner_bounds_maps:
            return CanAssignError(
                f"{b} is not {relation.description} any of {' | '.join(map(str, a_iter))}",
                children=errors,
            )
        bounds_maps.append(intersect_bounds_maps(inner_bounds_maps))
    return unify_bounds_maps(bounds_maps)


# I think the right algorithm is "double-ended deunioning".
# Notation: Single elements (written as "int") are standard tuple elements.
# Many elements (written as "*int") are unpacked tuple elements.
# 1. Remove matching Single elements on both ends. If their types are compatible, great,
#    if not, return an error.
# 2. If on either end you have a Many-Single match, deunion both ends simultaneously
#    Deunion means to decompose a Many into either an empty tuple or (Single, Many)
#    (if on the left end) or (Many, Single) (if on the right end).
#    Is (*int, int) assignable to (int, *int)? -> check whether (int) | (int, *int, int)
#    is assignable to (int) | (int, *int, int)
# 3. If both sides are a single Many, just compare them
# 4. Else, we must have multiple Manys on one side, which means we're out of spec territory.
#    Check the relation between the leftmost elements in A and B.
# 5. If they're not compatible, remove the leftmost element from A and continue.
# 6. If they are compatible, then try three variants:
#    a. Remove leftmost from A and remove leftmost from B.
#    b. Remove leftmost from A and leave B.
#    c. Leave A and remove leftmost from B.
#    If any of these succeeds, we succeed.


def _has_relation_sequence(
    left: SequenceValue,
    right: SequenceValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    can_assign = left.get_type_object(ctx).can_assign(left, right, ctx)
    if isinstance(can_assign, CanAssignError):
        return CanAssignError(
            f"{stringify_object(right.typ)} is not {relation.description}"
            f" {stringify_object(left.typ)}"
        )

    if not left.members:
        if not right.members:
            return can_assign
        else:
            return CanAssignError(
                f"Non-empty {stringify_object(right.typ)} is not"
                f" {relation.description} empty {stringify_object(left.typ)}"
            )

    inner_can_assign = _has_relation_sequence_inner(
        can_assign, left, right, relation, ctx
    )
    if isinstance(inner_can_assign, CanAssignError):
        return CanAssignError(
            f"{right} is not {relation.description} {left}", [inner_can_assign]
        )
    return inner_can_assign


def _compare_single_element(
    left: SequenceValue,
    right: SequenceValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
    my_idx: int,
    their_idx: int,
    *,
    their_min_idx: int = 0,
    their_max_idx: Optional[int] = None,
) -> CanAssign:
    if (
        their_idx >= len(right.members)
        or their_idx < their_min_idx
        or (their_max_idx is not None and their_idx > their_max_idx)
    ):
        return CanAssignError(f"{right} does not contain as many members as {left}")
    their_is_many, their_member = right.members[their_idx]
    if their_is_many:
        return CanAssignError(
            f"Member {their_idx} is a single element, but an unpacked type is"
            " provided"
        )
    _, my_member = left.members[my_idx]
    my_gradualized = _gradualize(my_member)
    their_gradualized = _gradualize(their_member)
    can_assign = _has_relation(my_gradualized, their_gradualized, relation, ctx)
    if isinstance(can_assign, CanAssignError):
        return CanAssignError(
            f"Types for member {my_idx} are incompatible", [can_assign]
        )
    else:
        return can_assign


def _has_relation_sequence_inner(
    bounds_map: BoundsMap,
    left: SequenceValue,
    right: SequenceValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    my_len = len(left.members)
    their_len = len(right.members)
    bounds_maps = [bounds_map]

    # First, look at simple values on the left end of left
    their_idx = my_idx = 0
    for i in range(my_len):
        my_is_many, my_member = left.members[i]
        if my_is_many:
            break
        my_idx += 1
        can_assign = _compare_single_element(left, right, relation, ctx, i, their_idx)
        if isinstance(can_assign, CanAssignError):
            return can_assign
        bounds_maps.append(can_assign)
        their_idx += 1
    if my_idx == my_len:
        # We consumed all of left. Is there anything left in right?
        if their_idx < their_len:
            remaining = their_len - their_idx
            return CanAssignError(f"{right} has {remaining} extra members")
        return unify_bounds_maps(bounds_maps)

    # Now we consume more from the right
    my_end_idx = my_len - 1
    their_end_idx = their_len - 1
    for i in range(my_len - 1, my_idx, -1):
        my_is_many, my_member = left.members[i]
        if my_is_many:
            break
        my_end_idx -= 1
        can_assign = _compare_single_element(
            left, right, relation, ctx, i, their_end_idx, their_min_idx=their_idx
        )
        if isinstance(can_assign, CanAssignError):
            return can_assign
        bounds_maps.append(can_assign)
        their_end_idx -= 1

    # Now look at what's left, and match them up greedily. Possibly we could do
    # something more precise with backtracking, but the spec only requires support
    # for one unpacking per tuple type.
    my_remaining = left.members[my_idx : my_end_idx + 1]
    for my_remaining_idx, (my_is_many, my_member) in enumerate(
        my_remaining, start=my_idx + 1
    ):
        if my_is_many:
            my_gradualized = _gradualize(my_member)
            while their_idx <= their_end_idx:
                _, their_member = right.members[their_idx]
                can_assign = _has_relation(
                    my_gradualized, _gradualize(their_member), relation, ctx
                )
                if isinstance(can_assign, CanAssignError):
                    break
                else:
                    bounds_maps.append(can_assign)
                    their_idx += 1
        else:
            can_assign = _compare_single_element(
                left,
                right,
                relation,
                ctx,
                my_remaining_idx,
                their_idx,
                their_max_idx=their_end_idx,
            )
            if isinstance(can_assign, CanAssignError):
                return can_assign
            bounds_maps.append(can_assign)
            their_idx += 1

    # Now is there anything left in right?
    if their_idx <= their_end_idx:
        remaining = their_end_idx - their_idx + 1
        return CanAssignError(f"{right} has {remaining} extra members")
    return unify_bounds_maps(bounds_maps)


def _map_relation(relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE]) -> Relation:
    if relation is Relation.SUBTYPE:
        return Relation.EQUIVALENT
    elif relation is Relation.ASSIGNABLE:
        return Relation.CONSISTENT
    else:
        assert_never(relation)


def _has_relation_typeddict(
    left: TypedDictValue,
    right: TypedDictValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    bounds_maps = []
    for key, entry in left.items.items():
        if key not in right.items:
            if entry.required:
                return CanAssignError(f"Required key {key} is missing in {right}")
            if not entry.readonly:
                # "other" may be a subclass of its TypedDict type that sets a different key
                return CanAssignError(f"Mutable key {key} is missing in {right}")
            extra_keys_type = _gradualize(right.extra_keys or TypedValue(object))
            can_assign = _has_relation(
                _gradualize(entry.typ), extra_keys_type, relation, ctx
            )
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"Type for key {key} is not {relation.description} extra keys type"
                    f" {extra_keys_type}",
                    children=[can_assign],
                )
        else:
            their_entry = right.items[key]
            if entry.required and not their_entry.required:
                return CanAssignError(f"Required key {key} is non-required in {right}")
            if not entry.required and not entry.readonly and their_entry.required:
                # This means we may del the key, but the other TypedDict does not
                # allow it
                return CanAssignError(f"Mutable key {key} is required in {right}")
            if not entry.readonly and their_entry.readonly:
                return CanAssignError(f"Mutable key {key} is readonly in {right}")

            if entry.readonly:
                relation_to_use = relation
            else:
                relation_to_use = _map_relation(relation)

            can_assign = has_relation(entry.typ, their_entry.typ, relation_to_use, ctx)
            if isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"Types for key {key} are incompatible", children=[can_assign]
                )
            bounds_maps.append(can_assign)

    if not left.extra_keys_readonly and right.extra_keys_readonly:
        return CanAssignError(f"Extra keys are readonly in {right}")
    if left.extra_keys is not None:
        if left.extra_keys_readonly:
            relation_to_use = relation
        else:
            relation_to_use = _map_relation(relation)
        their_extra_keys = right.extra_keys or TypedValue(object)
        can_assign = has_relation(
            left.extra_keys, their_extra_keys, relation_to_use, ctx
        )
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                "Types for extra keys are incompatible", children=[can_assign]
            )
        bounds_maps.append(can_assign)
    return unify_bounds_maps(bounds_maps)


def _has_relation_typeddict_dict(
    left: TypedDictValue,
    right: DictIncompleteValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    bounds_maps = []
    for key, entry in left.items.items():
        their_value = right.get_value(KnownValue(key), ctx)
        if their_value is UNINITIALIZED_VALUE:
            if entry.required:
                return CanAssignError(f"Key {key} is missing in {right}")
            else:
                continue
        can_assign = has_relation(entry.typ, their_value, relation, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                f"Types for key {key} are incompatible", children=[can_assign]
            )
        bounds_maps.append(can_assign)
    for pair in right.kv_pairs:
        for key_type in flatten_values(pair.key, unwrap_annotated=True):
            if isinstance(key_type, KnownValue):
                if not isinstance(key_type.val, str):
                    return CanAssignError(f"Key {pair.key} is not a string")
                if key_type.val not in left.items:
                    if left.extra_keys is NO_RETURN_VALUE:
                        return CanAssignError(
                            f"Key {key_type.val!r} is not allowed in closed"
                            f" TypedDict {left}"
                        )
                    elif left.extra_keys is not None:
                        can_assign = has_relation(
                            left.extra_keys, pair.value, relation, ctx
                        )
                        if isinstance(can_assign, CanAssignError):
                            return CanAssignError(
                                f"Type for extra key {pair.key} is" " incompatible",
                                children=[can_assign],
                            )
                        bounds_maps.append(can_assign)
            else:
                can_assign = has_relation(TypedValue(str), key_type, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    return CanAssignError(
                        f"Type for key {pair.key} is not a string",
                        children=[can_assign],
                    )
                if left.extra_keys is NO_RETURN_VALUE:
                    return CanAssignError(
                        f"Key {pair.key} is not allowed in closed TypedDict" f" {left}"
                    )
                elif left.extra_keys is not None:
                    can_assign = has_relation(
                        left.extra_keys, pair.value, relation, ctx
                    )
                    if isinstance(can_assign, CanAssignError):
                        return CanAssignError(
                            f"Type for extra key {pair.key} is incompatible",
                            children=[can_assign],
                        )
                    bounds_maps.append(can_assign)
    return unify_bounds_maps(bounds_maps)


def get_tv_map(
    left: Value, right: Value, relation: Relation, ctx: CanAssignContext
) -> Union[TypeVarMap, CanAssignError]:
    bounds_map = has_relation(left, right, relation, ctx)
    if isinstance(bounds_map, CanAssignError):
        return bounds_map
    tv_map, errors = resolve_bounds_map(bounds_map, ctx)
    if errors:
        return CanAssignError(children=list(errors))
    return tv_map


def is_iterable(
    value: Value, relation: Relation, ctx: CanAssignContext
) -> Union[CanAssignError, Value]:
    """Check whether a value is iterable."""
    tv_map = get_tv_map(IterableValue, value, relation, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get(T, AnyValue(AnySource.generic_argument))


class HashableProto(Protocol):
    def __hash__(self) -> int:
        raise NotImplementedError


HashableProtoValue = TypedValue(HashableProto)


def check_hashability(value: Value, ctx: CanAssignContext) -> Optional[CanAssignError]:
    """Check whether a value is hashable.

    Return None if it is hashable, otherwise a CanAssignError.

    """
    can_assign = is_assignable_with_reason(HashableProtoValue, value, ctx)
    HashableProtoValue.can_assign(value, ctx)
    if isinstance(can_assign, CanAssignError):
        return can_assign
    return None


def can_assign_and_used_any(
    param_typ: Value, var_value: Value, ctx: CanAssignContext
) -> tuple[CanAssign, bool]:
    subtype_result = is_subtype_with_reason(param_typ, var_value, ctx)
    if not isinstance(subtype_result, CanAssignError):
        return subtype_result, False
    assignability_result = is_assignable_with_reason(param_typ, var_value, ctx)
    return assignability_result, True
