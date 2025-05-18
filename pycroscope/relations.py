"""

Relations between types.

See the typing spec:
https://typing.python.org/en/latest/spec/concepts.html#summary-of-type-relations

"""

import collections.abc
import enum
from types import FunctionType
from typing import Union

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
    flatten_values,
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

    # TypeAliasValue
    if isinstance(left, TypeAliasValue):
        left_inner = _gradualize(left.get_value())
        return _has_relation(left_inner, right, relation, ctx)
    if isinstance(right, TypeAliasValue):
        right_inner = _gradualize(right.get_value())
        return _has_relation(left, right_inner, relation, ctx)

    # AnnotatedValue
    if isinstance(left, AnnotatedValue):
        left_inner = _gradualize(left.value)
        can_assign = _has_relation(left_inner, right, relation, ctx)
        if (
            isinstance(can_assign, CanAssignError)
            or relation is not Relation.ASSIGNABLE
        ):
            return can_assign
        bounds_maps = [can_assign]
        for ext in left.get_metadata_of_type(Extension):
            custom_can_assign = ext.can_assign(right, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            bounds_maps.append(custom_can_assign)
        return unify_bounds_maps(bounds_maps)
    if isinstance(right, AnnotatedValue):
        right_inner = _gradualize(right.value)
        can_assign = _has_relation(left, right_inner, relation, ctx)
        if (
            isinstance(can_assign, CanAssignError)
            or relation is not Relation.ASSIGNABLE
        ):
            return can_assign
        bounds_maps = [can_assign]
        for ext in right.get_metadata_of_type(Extension):
            custom_can_assign = ext.can_be_assigned(left, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            bounds_maps.append(custom_can_assign)
        return unify_bounds_maps(bounds_maps)

    # AnyValue
    if isinstance(left, AnyValue):
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
            return unify_bounds_maps(bounds_maps)
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

    # TypeVarValue
    if isinstance(left, TypeVarValue):
        if isinstance(right, TypeVarValue):
            if left.typevar is right.typevar:
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            bounds = [LowerBound(left.typevar, right), *left.get_inherent_bounds()]
            return left.make_bounds_map(bounds, right, ctx)
    if isinstance(right, TypeVarValue):
        bounds = [UpperBound(right.typevar, left), *right.get_inherent_bounds()]
        return right.make_bounds_map(bounds, left, ctx)

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
        return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(left, ParamSpecKwargsValue):
        if (
            isinstance(right, ParamSpecKwargsValue)
            and left.param_spec is right.param_spec
        ):
            return {}
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, ParamSpecKwargsValue):
        return CanAssignError(f"{right} is not {relation.description} {left}")

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
        if isinstance(right, UnboundMethodValue):
            if left == right:
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, UnboundMethodValue):
        sig = right.get_signature(ctx)
        if sig is None:
            return CanAssignError(f"{right} is not {relation.description} {left}")
        return _has_relation(left, CallableValue(sig), relation, ctx)

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
    if isinstance(right, KnownValue):
        right = typify_literal(right)
        if isinstance(right, KnownValue):
            left_tobj = left.get_type_object(ctx)
            can_assign = left_tobj.can_assign(left, right, ctx)
            if isinstance(can_assign, CanAssignError):
                if left_tobj.is_instance(right.val):
                    return {}
            return can_assign

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

    if isinstance(left, CallableValue):
        signature = ctx.signature_from_value(right)
        if isinstance(signature, pycroscope.signature.BoundMethodSignature):
            signature = signature.get_signature(ctx=ctx)
        if signature is None:
            return CanAssignError(f"{right} is not a callable type")
        return pycroscope.signature.signatures_have_relation(
            left.signature, signature, relation, ctx
        )
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
        else:
            assert_never(right)
        raise NotImplementedError

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
    my_len = len(left.members)
    their_len = len(right.members)
    if my_len != their_len:
        type_str = stringify_object(right.typ)
        return CanAssignError(
            f"{type_str} of length {their_len} is not {relation.description} {type_str} of"
            f" length {my_len}"
        )
    if my_len == 0:
        return {}  # they're both empty
    bounds_maps = [can_assign]
    for i, ((my_is_many, my_member), (their_is_many, their_member)) in enumerate(
        zip(left.members, right.members)
    ):
        my_member = _gradualize(my_member)
        their_member = _gradualize(their_member)
        if my_is_many != their_is_many:
            if my_is_many:
                return CanAssignError(
                    f"Member {i} is an unpacked type, but a single element is"
                    " provided"
                )
            else:
                return CanAssignError(
                    f"Member {i} is a single element, but an unpacked type is"
                    " provided"
                )
        can_assign = _has_relation(my_member, their_member, relation, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                f"Types for member {i} are incompatible", [can_assign]
            )
        bounds_maps.append(can_assign)
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

            can_assign = has_relation(their_entry.typ, entry.typ, relation_to_use, ctx)
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
