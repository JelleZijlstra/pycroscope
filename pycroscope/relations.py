"""

Relations between types.

See the typing spec:
https://typing.python.org/en/latest/spec/concepts.html#summary-of-type-relations

"""

import collections.abc
import enum
import struct
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from math import comb
from types import FunctionType
from typing import Literal, Protocol

from typing_extensions import assert_never

import pycroscope
from pycroscope.analysis_lib import Sentinel
from pycroscope.annotated_types import MaxLen, MinLen
from pycroscope.extensions import PredicateCheck
from pycroscope.find_unused import used
from pycroscope.safe import (
    is_instance_of_typing_name,
    safe_equals,
    safe_isinstance,
    safe_issubclass,
)
from pycroscope.typevar import resolve_bounds_map
from pycroscope.value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    BasicType,
    CallableValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    DictIncompleteValue,
    Extension,
    GenericValue,
    GradualType,
    IntersectionValue,
    IterableValue,
    KnownValue,
    LowerBound,
    MultiValuedValue,
    NewTypeValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    PredicateValue,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticModuleValue,
    T,
    TypeAliasValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeVarMap,
    TypeVarValue,
    UnboundMethodValue,
    UpperBound,
    Value,
    VariableNameValue,
    flatten_values,
    gradualize,
    intersect_bounds_maps,
    stringify_object,
    typify_literal,
    unify_bounds_maps,
    unite_values,
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
    left = gradualize(left)
    right = gradualize(right)
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


def _has_relation(
    left: GradualType,
    right: GradualType,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
    *,
    original_left: GradualType | None = None,
    original_right: GradualType | None = None,
) -> CanAssign:
    if original_right is None:
        original_right = right
    if original_left is None:
        original_left = left

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
                left_inner = gradualize(left.get_value())
                return _has_relation(left_inner, right, relation, ctx)
        left_inner = gradualize(left.get_value())
        return _has_relation(left_inner, right, relation, ctx)
    if isinstance(right, TypeAliasValue):
        right_inner = gradualize(right.get_value())
        return _has_relation(left, right_inner, relation, ctx)

    # AnnotatedValue
    if isinstance(left, AnnotatedValue):
        left_inner = gradualize(left.value)
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
    if isinstance(right, AnnotatedValue):
        right_inner = gradualize(right.value)
        can_assign = _has_relation(
            left, right_inner, relation, ctx, original_right=right
        )
        if isinstance(can_assign, CanAssignError):
            return can_assign
        bounds_maps = [can_assign]
        for ext in right.get_metadata_of_type(Extension):
            custom_can_assign = ext.can_be_assigned(left, ctx)
            if isinstance(custom_can_assign, CanAssignError):
                return custom_can_assign
            bounds_maps.append(custom_can_assign)
        return unify_bounds_maps(bounds_maps)

    # PredicateValue
    if isinstance(left, PredicateValue):
        if isinstance(right, PredicateValue):
            if _predicate_implies(right.predicate, left.predicate):
                return {}
            return CanAssignError(f"{right} is not {relation.description} {left}")
        if _value_guarantees_predicate(right, left.predicate, ctx):
            return {}
        return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, PredicateValue):
        if left == TypedValue(object):
            return {}
        return CanAssignError(f"{right} is not {relation.description} {left}")

    # TypeFormValue
    if isinstance(left, TypeFormValue):
        return left.can_assign(right, ctx)
    if isinstance(right, TypeFormValue):
        right_inner = gradualize(right.get_fallback_value())
        return _has_relation(left, right_inner, relation, ctx, original_right=right)

    # NewTypeValue
    if isinstance(left, NewTypeValue):
        if isinstance(right, NewTypeValue):
            if left.newtype is right.newtype:
                return {}
            else:
                return CanAssignError(f"{right} is not {relation.description} {left}")
        elif isinstance(right, AnyValue):
            pass
        else:
            return CanAssignError(f"{right} is not {relation.description} {left}")
    if isinstance(right, NewTypeValue):
        right_inner = gradualize(right.value)
        return _has_relation(left, right_inner, relation, ctx, original_right=right)

    # IntersectionValue
    if isinstance(left, IntersectionValue):
        # Try to simplify first
        left = intersect_multi(left.vals, ctx)
        if not isinstance(left, IntersectionValue):
            return _has_relation(left, original_right, relation, ctx)
        if isinstance(right, IntersectionValue):
            right = intersect_multi(right.vals, ctx)
        # Must be a subtype of all the members
        bounds_maps = []
        errors = []
        for val in left.vals:
            can_assign = _has_relation(gradualize(val), original_right, relation, ctx)
            if isinstance(can_assign, CanAssignError):
                errors.append(can_assign)
            else:
                bounds_maps.append(can_assign)
        if errors:
            return CanAssignError(
                f"{right} is not {relation.description} {left}", children=errors
            )
        return intersect_bounds_maps(bounds_maps)
    if isinstance(right, IntersectionValue):
        right = intersect_multi(right.vals, ctx)
        if not isinstance(right, IntersectionValue):
            return _has_relation(original_left, right, relation, ctx)
        if isinstance(left, MultiValuedValue):
            # For expected unions, first test each union arm against the whole
            # intersection. Decomposing the right side too early can lose
            # correlations (e.g. Any & Predicate[...] should match a union arm
            # that expects the same intersection).
            bounds_maps = []
            errors = []
            for val in left.vals:
                can_assign = _has_relation(gradualize(val), right, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    errors.append(can_assign)
                else:
                    bounds_maps.append(can_assign)
            if not bounds_maps:
                return CanAssignError(
                    f"{right} is not {relation.description} {left}", children=errors
                )
            return intersect_bounds_maps(bounds_maps)
        # At least one member must be a subtype
        bounds_maps = []
        errors = []
        for val in right.vals:
            can_assign = _has_relation(original_left, gradualize(val), relation, ctx)
            if isinstance(can_assign, CanAssignError):
                errors.append(can_assign)
            else:
                bounds_maps.append(can_assign)
        if not bounds_maps:
            return CanAssignError(
                f"{right} is not {relation.description} {left}", children=errors
            )
        return unify_bounds_maps(bounds_maps)

    # Never (special case of MultiValuedValue)
    if right is NO_RETURN_VALUE:
        return {}
    if left is NO_RETURN_VALUE:
        return CanAssignError(f"{right} is not {relation.description} {left}")

    # MultiValuedValue
    if isinstance(left, MultiValuedValue):
        if isinstance(right, MultiValuedValue):
            return _has_relation_union(left, right.vals, relation, ctx)
        else:
            # right is a subtype if it's a subtype of any of the members
            bounds_maps = []
            errors = []
            for val in left.vals:
                val = gradualize(val)
                can_assign = _has_relation(val, original_right, relation, ctx)
                if isinstance(can_assign, CanAssignError):
                    errors.append(can_assign)
                else:
                    bounds_maps.append(can_assign)
            if not bounds_maps:
                # Try decomposing the right
                decomposed = right.decompose()
                if decomposed is not None:
                    can_assign = _has_relation_union(left, decomposed, relation, ctx)
                    if not isinstance(can_assign, CanAssignError):
                        return can_assign
                    # Fall back to the original error
                return CanAssignError(
                    f"{right} is not {relation.description} {left}", children=errors
                )
            return intersect_bounds_maps(bounds_maps)
    if isinstance(right, MultiValuedValue):
        # right is a subtype if all the members are subtypes of left
        bounds_maps = []
        for val in right.vals:
            val = gradualize(val)
            can_assign = _has_relation(original_left, val, relation, ctx)
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
    assert not isinstance(left, NewTypeValue)

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
                    if isinstance(my_arg, TypeVarValue) and is_instance_of_typing_name(
                        my_arg.typevar, "ParamSpec"
                    ):
                        my_arg = pycroscope.input_sig.wrap_type_param(my_arg.typevar)
                    if isinstance(
                        their_arg, TypeVarValue
                    ) and is_instance_of_typing_name(their_arg.typevar, "ParamSpec"):
                        their_arg = pycroscope.input_sig.wrap_type_param(
                            their_arg.typevar
                        )
                    left_is_input_sig = isinstance(
                        my_arg, pycroscope.input_sig.InputSigValue
                    )
                    right_is_input_sig = isinstance(
                        their_arg, pycroscope.input_sig.InputSigValue
                    )
                    if left_is_input_sig and right_is_input_sig:
                        can_assign = pycroscope.input_sig.input_sigs_have_relation(
                            my_arg.input_sig, their_arg.input_sig, relation, ctx
                        )
                    else:
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


def _has_relation_union(
    left: GradualType,
    right_vals: Iterable[Value],
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    bounds_maps = []
    for val in right_vals:
        val = gradualize(val)
        can_assign = _has_relation(left, val, relation, ctx)
        if isinstance(can_assign, CanAssignError):
            # Adding an additional layer here isn't helpful
            return can_assign
        bounds_maps.append(can_assign)
    return unify_bounds_maps(bounds_maps)


def _has_relation_thrift_enum(
    left: TypedValue,
    right: TypedValue | KnownValue,
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
    prefix: Value | None = None
    suffix: Value | None = None

    def __post_init__(self) -> None:
        assert len(self) >= 0

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
        if idx < 0:
            raise IndexError(idx)
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
            members.append((False, self.prefix))
        members.extend(self.seq.members[self.start_idx : -self.end_idx])
        if self.suffix is not None:
            members.append((False, self.suffix))
        return SequenceValue(self.seq.typ, list(self))

    def decompose_left(self) -> Iterable["_LazySequenceValue"]:
        if len(self) > 0 and self[0][0]:
            # If the first element is tuple[T, ...], we can decompose it into...
            # ... the case where it is empty
            yield self.slice_left()
            # ... the case where it contains a single element, followed by a new tuple[T, ...]
            yield self.add_prefix(self[0][1])
        else:
            # If the first element is Single, we can't decompose.
            yield self

    def decompose_right(self) -> Iterable["_LazySequenceValue"]:
        if len(self) > 0 and self[-1][0]:
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
        for i in range(self.start_idx, len(self.seq.members) + self.end_idx):
            yield self.seq.members[i]
        if self.suffix is not None:
            yield False, self.suffix


def _has_relation_lazy_sequence(
    a: _LazySequenceValue,
    b: _LazySequenceValue,
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    """Check the relation between two sequences A and B.

    Sequences can contain either single values (Single, represented in examples as "int")
    or an unknown number of values (Many, represented as "*int").

    The algorithm is:

    - If a is empty:
        - If b is empty: success.
        - Else: error.
    - If b is empty:
        - If all elements in a are Many: success.
        - Else: error.
    - If both a and b have a Single element on the left:
        - Check whether they're compatible.
        - If so, continue checking the rest of the sequences.
        - If not: error.
    - If both a and b have a Single element on the right:
        - Check whether they're compatible.
        - If so, continue checking the rest of the sequences.
        - If not: error.
    - If all elements at the left and right ends are Many:
        - Check whether the leftmost elements are compatible.
        - If not, remove the leftmost element from A and continue.
        - If so, we can succeed in three ways:
            - Consume A's leftmost and continue.
              Example: A = (*object, *int), B = (*int,)
            - Consume B's leftmost and continue.
              Example: A = (*object,), B = (*object, *int)
            - Consume both leftmost and continue.
              Example: A = (*object, int, *int), B = (*object, int, *int)
    - In all cases where one of the two has a Many on one end and the other has a
    - Single on the same end, decompose the Many elements *T into two options:
      either they are empty, or they contain (T, *T) (on the left)
      or (*T, T) (on the right). Then compare the resulting unions.
      Examples:
        - (*int) :> (int, *int, int) turns into:
            - () | (int,) | (int, *int) :> (int, *int, int)
            - (int, *int) :> (int, *int, int)
            - (*int) :> (*int, int)
            - Succeeds by the step above.
        - (*int, int) -> (int, *int) turns into:
            - (int,) | (int, *int, int) -> (int,) | (int, *int, int)
            - Succeeds as the unions contain the same elements.
      This last step is incorrect. It fails to recognize this:
      (*int, int, str, int, *int) == (int, *int, str, *int, int)
    """
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
                f"Empty sequence {b!r} is not {relation.description} non-empty sequence {a}"
            )

    # Do both have a Single element on the left?
    if a[0][0] is False and b[0][0] is False:
        # If so, check whether they're compatible
        can_assign = _has_relation(
            gradualize(a[0][1]), gradualize(b[0][1]), relation, ctx
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
            gradualize(a[-1][1]), gradualize(b[-1][1]), relation, ctx
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
            gradualize(a[0][1]), gradualize(b[0][1]), relation, ctx
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
    if a[0][0] and not b[0][0]:
        a_decomposed = a.decompose_left()
    else:
        a_decomposed = [a]
    if b[0][0] and not a[0][0]:
        b_decomposed = b.decompose_left()
    else:
        b_decomposed = [b]

    if a[-1][0] and not b[-1][0]:
        a_decomposed = [a_dd for a_d in a_decomposed for a_dd in a_d.decompose_right()]
    if b[-1][0] and not a[-1][0]:
        b_decomposed = [b_dd for b_d in b_decomposed for b_dd in b_d.decompose_right()]

    return _has_relation_lazy_seq_multi(a_decomposed, b_decomposed, relation, ctx)


def _has_relation_lazy_seq_multi(
    a_iter: Iterable[_LazySequenceValue],
    b_iter: Iterable[_LazySequenceValue],
    relation: Literal[Relation.SUBTYPE, Relation.ASSIGNABLE],
    ctx: CanAssignContext,
) -> CanAssign:
    bounds_maps = []
    a_iter = list(a_iter)
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

    return _has_relation_lazy_sequence(
        _LazySequenceValue(left), _LazySequenceValue(right), relation, ctx
    )


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
            extra_keys_type = gradualize(right.extra_keys or TypedValue(object))
            can_assign = _has_relation(
                gradualize(entry.typ), extra_keys_type, relation, ctx
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
) -> TypeVarMap | CanAssignError:
    bounds_map = has_relation(left, right, relation, ctx)
    if isinstance(bounds_map, CanAssignError):
        return bounds_map
    tv_map, errors = resolve_bounds_map(bounds_map, ctx)
    if errors:
        return CanAssignError(children=list(errors))
    return tv_map


def is_iterable(
    value: Value, relation: Relation, ctx: CanAssignContext
) -> CanAssignError | Value:
    """Check whether a value is iterable."""
    tv_map = get_tv_map(IterableValue, value, relation, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get(T, AnyValue(AnySource.generic_argument))


class HashableProto(Protocol):
    def __hash__(self) -> int:
        raise NotImplementedError


HashableProtoValue = TypedValue(HashableProto)


def check_hashability(value: Value, ctx: CanAssignContext) -> CanAssignError | None:
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


Irreducible = Sentinel("Irreducible")
TypeOrIrreducible = GradualType | Literal[Irreducible]

_MAX_EXACT_TUPLE_LENGTH = 64
_MAX_EXACT_TUPLE_EXPANSIONS = 128


def _iter_compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _iter_compositions(total - first, parts - 1):
            yield (first, *rest)


def _tuple_members_from_value(value: Value) -> tuple[tuple[bool, Value], ...] | None:
    if isinstance(value, SequenceValue) and value.typ is tuple:
        return value.members
    if isinstance(value, GenericValue) and value.typ is tuple and len(value.args) == 1:
        return ((True, value.args[0]),)
    return None


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


def _narrow_tuple_to_exact_len(value: Value, target_len: int) -> Value | None:
    members = _tuple_members_from_value(value)
    if members is None:
        return None
    expanded = _expand_tuple_members_to_exact_len(members, target_len)
    if expanded is None:
        return None
    if not expanded:
        return NO_RETURN_VALUE
    return unite_values(*[SequenceValue(tuple, members) for members in expanded])


def _narrow_tuple_to_min_len(value: Value, target_len: int) -> Value | None:
    members = _tuple_members_from_value(value)
    if members is None:
        return None
    expanded = _expand_tuple_members_to_min_len(members, target_len)
    if expanded is None:
        return None
    if not expanded:
        return NO_RETURN_VALUE
    return unite_values(*[SequenceValue(tuple, members) for members in expanded])


def _narrow_tuple_to_max_len(value: Value, target_len: int) -> Value | None:
    members = _tuple_members_from_value(value)
    if members is None:
        return None
    expanded = _expand_tuple_members_to_max_len(members, target_len)
    if expanded is None:
        return None
    if not expanded:
        return NO_RETURN_VALUE
    return unite_values(*[SequenceValue(tuple, members) for members in expanded])


def _get_len_bounds(value: Value) -> tuple[int | None, int | None]:
    if isinstance(value, KnownValue):
        try:
            length = len(value.val)
        except Exception:
            return None, None
        return length, length
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
        min_len = sum(pair.is_required and not pair.is_many for pair in value.kv_pairs)
        max_len = 0
        for pair in value.kv_pairs:
            if pair.is_many:
                max_len = None
                break
            max_len += 1
        return min_len, max_len
    if isinstance(value, TypedDictValue):
        min_len = sum(entry.required for entry in value.items.values())
        if value.extra_keys is not NO_RETURN_VALUE:
            return min_len, None
        return min_len, len(value.items)
    if isinstance(value, TypedValue) and value.typ in {
        str,
        bytes,
        bytearray,
        tuple,
        list,
        set,
        frozenset,
        dict,
    }:
        return 0, None
    return None, None


def _predicate_implies(stronger: PredicateCheck, weaker: PredicateCheck) -> bool:
    if stronger == weaker:
        return True
    return weaker.is_compatible_metadata(stronger)


def _intersect_len_predicate(
    predicate: MinLen | MaxLen, value: SimpleType
) -> TypeOrIrreducible:
    if not isinstance(predicate.value, int):
        return Irreducible

    min_len, max_len = _get_len_bounds(value)
    if isinstance(value, KnownValue) and min_len is None and max_len is None:
        return NO_RETURN_VALUE
    if isinstance(predicate, MinLen):
        target = predicate.value
        tuple_narrowed = _narrow_tuple_to_min_len(value, target)
        if tuple_narrowed is NO_RETURN_VALUE:
            return NO_RETURN_VALUE
        if tuple_narrowed is not None:
            return tuple_narrowed
        if max_len is not None and max_len < target:
            return NO_RETURN_VALUE
        if min_len is not None and min_len >= target:
            return value
        return Irreducible
    else:
        target = predicate.value
        tuple_narrowed = _narrow_tuple_to_max_len(value, target)
        if tuple_narrowed is NO_RETURN_VALUE:
            return NO_RETURN_VALUE
        if tuple_narrowed is not None:
            return tuple_narrowed
        if min_len is not None and min_len > target:
            return NO_RETURN_VALUE
        if max_len is not None and max_len <= target:
            return value
        return Irreducible


def _intersect_predicate_predicate(
    left: PredicateValue, right: PredicateValue
) -> TypeOrIrreducible:
    left_pred = left.predicate
    right_pred = right.predicate
    if isinstance(left_pred, MinLen) and isinstance(right_pred, MinLen):
        if isinstance(left_pred.value, int) and isinstance(right_pred.value, int):
            return PredicateValue(MinLen(max(left_pred.value, right_pred.value)))
    if isinstance(left_pred, MaxLen) and isinstance(right_pred, MaxLen):
        if isinstance(left_pred.value, int) and isinstance(right_pred.value, int):
            return PredicateValue(MaxLen(min(left_pred.value, right_pred.value)))
    if isinstance(left_pred, MinLen) and isinstance(right_pred, MaxLen):
        if (
            isinstance(left_pred.value, int)
            and isinstance(right_pred.value, int)
            and left_pred.value > right_pred.value
        ):
            return NO_RETURN_VALUE
    if isinstance(left_pred, MaxLen) and isinstance(right_pred, MinLen):
        if (
            isinstance(left_pred.value, int)
            and isinstance(right_pred.value, int)
            and right_pred.value > left_pred.value
        ):
            return NO_RETURN_VALUE
    if _predicate_implies(left_pred, right_pred):
        return left
    if _predicate_implies(right_pred, left_pred):
        return right
    return Irreducible


def _intersect_predicate(
    left: PredicateValue, right: SimpleType, ctx: CanAssignContext
) -> TypeOrIrreducible:
    if isinstance(right, PredicateValue):
        return _intersect_predicate_predicate(left, right)
    if isinstance(left.predicate, (MinLen, MaxLen)):
        return _intersect_len_predicate(left.predicate, right)
    if isinstance(right, KnownValue):
        can_assign = left.predicate.can_assign(right, ctx)
        if isinstance(can_assign, CanAssignError):
            return NO_RETURN_VALUE
        return right
    return Irreducible


def _value_guarantees_predicate(
    value: GradualType, predicate: PredicateCheck, ctx: CanAssignContext
) -> bool:
    if value is NO_RETURN_VALUE:
        return True
    if isinstance(value, AnyValue):
        return True
    if isinstance(value, MultiValuedValue):
        return all(
            _value_guarantees_predicate(gradualize(val), predicate, ctx)
            for val in value.vals
        )
    if isinstance(value, IntersectionValue):
        return any(
            _value_guarantees_predicate(gradualize(val), predicate, ctx)
            for val in value.vals
        )
    if isinstance(value, PredicateValue):
        return _predicate_implies(value.predicate, predicate)
    if isinstance(predicate, MinLen) and isinstance(predicate.value, int):
        min_len, _ = _get_len_bounds(value)
        return min_len is not None and min_len >= predicate.value
    if isinstance(predicate, MaxLen) and isinstance(predicate.value, int):
        _, max_len = _get_len_bounds(value)
        return max_len is not None and max_len <= predicate.value
    if isinstance(value, KnownValue):
        try:
            return not isinstance(predicate.can_assign(value, ctx), CanAssignError)
        except Exception:
            return False
    return False


def intersect_multi(values: Sequence[Value], ctx: CanAssignContext) -> GradualType:
    """Intersect multiple values."""
    if not values:
        return TypedValue(object)

    result = values[0]
    for value in values[1:]:
        result = intersect_values(result, value, ctx)
    return gradualize(result)


def intersect_values(left: Value, right: Value, ctx: CanAssignContext) -> GradualType:
    value = _intersect_values_inner(left, right, ctx)
    if value is Irreducible:
        return IntersectionValue((left, right))
    return value


def _intersect_values_inner(
    left: Value, right: Value, ctx: CanAssignContext
) -> TypeOrIrreducible:
    left = gradualize(left)
    right = gradualize(right)

    if (result := _simple_intersection(left, right, ctx)) is not None:
        return result

    if isinstance(left, TypeAliasValue):
        return _intersect_alias(left, right, ctx)
    if isinstance(right, TypeAliasValue):
        return _intersect_alias(right, left, ctx)

    if isinstance(
        left,
        (
            AnnotatedValue,
            NewTypeValue,
            ParamSpecArgsValue,
            ParamSpecKwargsValue,
            TypeVarValue,
        ),
    ):
        return _intersect_wrapper(left, right, ctx)
    if isinstance(
        right,
        (
            AnnotatedValue,
            NewTypeValue,
            ParamSpecArgsValue,
            ParamSpecKwargsValue,
            TypeVarValue,
        ),
    ):
        return _intersect_wrapper(right, left, ctx)

    return _intersect_basic_types(left, right, ctx)


def _intersect_wrapper(
    left: (
        NewTypeValue
        | AnnotatedValue
        | ParamSpecArgsValue
        | ParamSpecKwargsValue
        | TypeVarValue
    ),
    right: GradualType,
    ctx: CanAssignContext,
) -> TypeOrIrreducible:
    left_inner = left.get_fallback_value()
    result = _intersect_values_inner(left_inner, right, ctx)
    if result is NO_RETURN_VALUE:
        return NO_RETURN_VALUE
    if result is Irreducible:
        return Irreducible
    # Example: NT = NewType("NT", Literal[1, 2, 3])
    # NT & Literal[2, 3, 4] = NT & Literal[2, 3]
    return IntersectionValue((left_inner, result))


def _intersect_alias(
    left: TypeAliasValue, right: GradualType, ctx: CanAssignContext
) -> TypeOrIrreducible:
    left_inner = left.get_value()
    if isinstance(right, TypeAliasValue):
        if ctx.can_aliases_assume_compatibility(left, right):
            # Treat it as irreducible
            # TODO: Is this right?
            return Irreducible
        with ctx.aliases_assume_compatibility(left, right):
            right_inner = right.get_value()
            return _intersect_values_inner(left_inner, right_inner, ctx)
    else:
        return _intersect_values_inner(left_inner, right, ctx)


def _intersect_basic_types(
    left: BasicType, right: BasicType, ctx: CanAssignContext
) -> TypeOrIrreducible:
    if isinstance(left, MultiValuedValue):
        return _intersect_union(left, right, ctx)
    if isinstance(right, MultiValuedValue):
        return _intersect_union(right, left, ctx)

    if isinstance(left, IntersectionValue):
        return _intersect_intersection(left, right, ctx)
    if isinstance(right, IntersectionValue):
        return _intersect_intersection(right, left, ctx)

    return _intersect_simple_types(left, right, ctx)


def _simple_intersection(
    left: GradualType, right: GradualType, ctx: CanAssignContext
) -> TypeOrIrreducible | None:
    # Anything & Never = Never
    if left is NO_RETURN_VALUE or right is NO_RETURN_VALUE:
        return NO_RETURN_VALUE

    # Anything & object = Anything
    if left == TypedValue(object):
        return right
    if right == TypedValue(object):
        return left

    # Intersections with Any don't simplify
    if isinstance(left, AnyValue):
        return Irreducible
    if isinstance(right, AnyValue):
        return Irreducible

    # If one is a subtype of the other, the narrower type prevails
    if is_subtype(left, right, ctx):
        return right
    if is_subtype(right, left, ctx):
        return left
    return None


def _intersect_simple_types(
    left: SimpleType, right: SimpleType, ctx: CanAssignContext
) -> TypeOrIrreducible:
    if isinstance(left, PredicateValue):
        return _intersect_predicate(left, right, ctx)
    if isinstance(right, PredicateValue):
        return _intersect_predicate(right, left, ctx)
    if isinstance(left, TypeFormValue) or isinstance(right, TypeFormValue):
        # Most TypeForm intersections don't have a useful reduction rule.
        # We keep them as intersections unless a prior generic simplification
        # (for example X & object, X & Never, or obvious subtype relations)
        # already applied in _simple_intersection().
        return Irreducible
    if isinstance(left, (KnownValue, SyntheticModuleValue, UnboundMethodValue)):
        return _intersect_singular_type(left, right, ctx)
    if isinstance(right, (KnownValue, SyntheticModuleValue, UnboundMethodValue)):
        return _intersect_singular_type(right, left, ctx)
    if isinstance(left, AnyValue) or isinstance(right, AnyValue):
        return Irreducible
    if isinstance(left, SubclassValue):
        if isinstance(right, SubclassValue):
            return _intersect_subclass_values(left, right, ctx)
        else:
            return _intersect_subclass_typed(left, right, ctx)
    if isinstance(right, SubclassValue):
        return _intersect_subclass_typed(right, left, ctx)
    return _intersect_typed(left, right, ctx)


def _intersect_typed(
    left: TypedValue, right: TypedValue, ctx: CanAssignContext
) -> TypeOrIrreducible:
    if isinstance(left, TypedDictValue) and isinstance(right, TypedDictValue):
        return _intersect_typeddict(left, right, ctx)

    # If either type is final, the intersection reduces to Never unless the final class
    # is a subclass of the other class. In that case, we treat the intersection as irreducible.
    # Note that we would have already reduced intersections where the *types* are subtypes;
    # the irreducible case can happen in certain situations involving generics.
    left_tobj = left.get_type_object(ctx)
    right_tobj = right.get_type_object(ctx)
    if left_tobj.is_final and not left_tobj.is_assignable_to_type_object(right_tobj):
        return NO_RETURN_VALUE
    if right_tobj.is_final and not right_tobj.is_assignable_to_type_object(left_tobj):
        return NO_RETURN_VALUE

    # If both types are nominal and are real (non-synthetic) types, we can mirror CPython's logic
    # to check whether a child class can exist at runtime.
    if (
        not left_tobj.is_protocol
        and not right_tobj.is_protocol
        and isinstance(left_tobj.typ, type)
        and isinstance(right_tobj.typ, type)
        and not _can_nominal_types_intersect(left_tobj.typ, right_tobj.typ)
    ):
        return NO_RETURN_VALUE

    # TODO: Consider more options for reducing the intersection:
    # - Certain cases involving incompatible metaclasses can return to Never
    # - Possibly some cases with attributes of incompatible types (possibly debatable)
    # - Cases involving SequenceValue, DictIncompleteValue, and other concrete subclasses of
    #   TypedValue
    return Irreducible


def _can_nominal_types_intersect(t1: type[object], t2: type[object]) -> bool:
    sb1 = _solid_base(t1)
    sb2 = _solid_base(t2)
    return safe_issubclass(sb1, sb2) or safe_issubclass(sb2, sb1)


SIZEOF_PYOBJECT = struct.calcsize("P")


def _shape_differs(t1: type[object], t2: type[object]) -> bool:
    """Check whether two types differ in shape.

    Mirrors the shape_differs() function in typeobject.c in CPython."""
    if sys.version_info >= (3, 12):
        return (
            t1.__basicsize__ != t2.__basicsize__ or t1.__itemsize__ != t2.__itemsize__
        )
    else:
        # CPython had more complicated logic before 3.12:
        # https://github.com/python/cpython/blob/f3c6f882cddc8dc30320d2e73edf019e201394fc/Objects/typeobject.c#L2224
        # We attempt to mirror it here well enough to support the most common cases.
        if t1.__itemsize__ or t2.__itemsize__:
            return (
                t1.__basicsize__ != t2.__basicsize__
                or t1.__itemsize__ != t2.__itemsize__
            )
        t_size = t1.__basicsize__
        if (
            # TODO: better understanding of attributes of type objects
            not t2.__weakrefoffset__  # static analysis: ignore[value_always_true]
            and t1.__weakrefoffset__ + SIZEOF_PYOBJECT == t_size
        ):
            t_size -= SIZEOF_PYOBJECT
        if (
            not t2.__dictoffset__  # static analysis: ignore[value_always_true]
            and t1.__dictoffset__ + SIZEOF_PYOBJECT == t_size
        ):
            t_size -= SIZEOF_PYOBJECT
        if (
            not t2.__weakrefoffset__  # static analysis: ignore[value_always_true]
            and t2.__weakrefoffset__ == t_size
        ):
            t_size -= SIZEOF_PYOBJECT
        return t_size != t2.__basicsize__


def _solid_base(typ: type[object]) -> type[object]:
    """Return the "solid base" of a type, mirroring the logic in CPython's typeobject.c.

    A "solid base" is a base that determines the shape of the type."""
    if typ is object:
        return object
    base = typ.__base__
    assert base is not None, f"Type {typ} has no base"
    if _shape_differs(typ, base):
        return typ
    return _solid_base(base)


def _intersect_maybe_mutable(
    left: Value,
    left_readonly: bool,
    right: Value,
    right_readonly: bool,
    ctx: CanAssignContext,
) -> tuple[Value, bool]:
    if left_readonly:
        if right_readonly:
            return intersect_values(left, right, ctx), True
        else:
            if is_assignable(left, right, ctx):
                return intersect_values(left, right, ctx), False
            else:
                return NO_RETURN_VALUE, False
    else:
        if right_readonly:
            if is_assignable(right, left, ctx):
                return intersect_values(left, right, ctx), False
            else:
                return NO_RETURN_VALUE, False
        else:
            if is_consistent(left, right, ctx):
                return intersect_values(left, right, ctx), False
            else:
                return NO_RETURN_VALUE, False


def _intersect_entry_with_extra(
    entry: TypedDictEntry, right: TypedDictValue, ctx: CanAssignContext
) -> TypedDictEntry | None:
    value, readonly = _intersect_maybe_mutable(
        entry.typ,
        entry.readonly,
        right.extra_keys or TypedValue(object),
        right.extra_keys_readonly,
        ctx,
    )
    if entry.required and value is NO_RETURN_VALUE:
        return None
    return TypedDictEntry(typ=value, required=entry.required, readonly=readonly)


def _intersect_typeddict(
    left: TypedDictValue, right: TypedDictValue, ctx: CanAssignContext
) -> TypeOrIrreducible:
    left_extra = left.extra_keys or TypedValue(object)
    right_extra = right.extra_keys or TypedValue(object)
    if left.extra_keys is None and right.extra_keys is None:
        extra_keys = None
        extra_readonly = False
    else:
        extra_keys, extra_readonly = _intersect_maybe_mutable(
            left_extra,
            left.extra_keys_readonly,
            right_extra,
            right.extra_keys_readonly,
            ctx,
        )

    items = {}
    for key, entry in left.items.items():
        if key in right.items:
            their_entry = right.items[key]
            required = entry.required or their_entry.required

            result, readonly = _intersect_maybe_mutable(
                entry.typ, entry.readonly, their_entry.typ, their_entry.readonly, ctx
            )
            if required and result is NO_RETURN_VALUE:
                return NO_RETURN_VALUE
            items[key] = TypedDictEntry(
                typ=result, required=required, readonly=readonly
            )
        else:
            new_entry = _intersect_entry_with_extra(entry, right, ctx)
            if new_entry is None:
                return NO_RETURN_VALUE
            items[key] = new_entry
    for key, entry in right.items.items():
        if key not in items:
            new_entry = _intersect_entry_with_extra(entry, left, ctx)
            if new_entry is None:
                return NO_RETURN_VALUE
            items[key] = new_entry

    return TypedDictValue(
        items=items, extra_keys=extra_keys, extra_keys_readonly=extra_readonly
    )


def _intersect_subclass_typed(
    left: SubclassValue, right: TypedValue, ctx: CanAssignContext
) -> TypeOrIrreducible:
    if is_assignable(right, TypedValue(type), ctx):
        return Irreducible
    else:
        return NO_RETURN_VALUE


def _intersect_subclass_values(
    left: SubclassValue, right: SubclassValue, ctx: CanAssignContext
) -> TypeOrIrreducible:
    result = _intersect_values_inner(left.typ, right.typ, ctx)
    if result is NO_RETURN_VALUE:
        return NO_RETURN_VALUE
    elif result is Irreducible:
        return Irreducible
    elif isinstance(result, (TypedValue, TypeVarValue)):
        return SubclassValue(result)
    elif isinstance(result, IntersectionValue):
        vals = []
        for subval in result.vals:
            if isinstance(subval, (TypedValue, TypeVarValue)):
                vals.append(SubclassValue(subval))
            else:
                return Irreducible
        return IntersectionValue(tuple(vals))
    else:
        return Irreducible


def _intersect_singular_type(
    left: KnownValue | SyntheticModuleValue | UnboundMethodValue,
    right: GradualType,
    ctx: CanAssignContext,
) -> TypeOrIrreducible:
    if is_assignable(right, left, ctx):
        # Now it is irreducible (e.g. tuple[Any, ...] & Literal[(1, 2)])
        return Irreducible
    else:
        # Otherwise it must be an empty type.
        return NO_RETURN_VALUE


def _intersect_union(
    left: MultiValuedValue, right: GradualType, ctx: CanAssignContext
) -> GradualType:
    vals = [intersect_values(subval, right, ctx) for subval in left.vals]
    return gradualize(unite_values(*vals))


def _intersect_intersection(
    left: IntersectionValue, right: GradualType, ctx: CanAssignContext
) -> GradualType:
    intersections = [
        _intersect_values_inner(subval, right, ctx) for subval in left.vals
    ]
    results = []
    should_add_right = False
    for subval, new_subval in zip(left.vals, intersections):
        if new_subval is Irreducible:
            results.append(subval)
            should_add_right = True
        elif isinstance(new_subval, IntersectionValue):
            for subsubval in new_subval.vals:
                results.append(gradualize(subsubval))
        else:
            results.append(new_subval)
    if should_add_right:
        results.append(right)
    return IntersectionValue(tuple(dict.fromkeys(results)))
