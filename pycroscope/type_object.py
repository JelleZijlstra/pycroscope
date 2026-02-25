"""

An object that represents a type.

"""

import collections.abc
import inspect
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass, field
from typing import cast
from unittest import mock

from pycroscope.signature import (
    BoundMethodSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
)

from .safe import safe_getattr, safe_in, safe_isinstance, safe_issubclass
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    BoundsMap,
    CallableValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    HasAttrExtension,
    KnownValue,
    NotAGradualType,
    SelfT,
    SubclassValue,
    TypedValue,
    Value,
    replace_fallback,
    stringify_object,
    unify_bounds_maps,
)


def get_mro(typ: type | super) -> Sequence[type]:
    if isinstance(typ, super):
        typ_for_mro = typ.__thisclass__
    else:
        typ_for_mro = typ
    try:
        return inspect.getmro(typ_for_mro)
    except AttributeError:
        # It's not actually a class.
        return []


@dataclass
class TypeObject:
    typ: type | super | str
    base_classes: set[type | str] = field(default_factory=set)
    is_final: bool = False
    is_protocol: bool = False
    protocol_members: set[str] = field(default_factory=set)
    is_thrift_enum: bool = field(init=False)
    is_universally_assignable: bool = field(init=False)
    artificial_bases: set[type] = field(default_factory=set, init=False)
    _protocol_positive_cache: dict[Value, BoundsMap] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.typ, str):
            # Synthetic type
            self.is_universally_assignable = False
            self.is_thrift_enum = False
            return
        if isinstance(self.typ, super):
            self.is_universally_assignable = False
        else:
            assert isinstance(self.typ, type), repr(self.typ)
            self.is_universally_assignable = issubclass(self.typ, mock.NonCallableMock)
            if safe_getattr(self.typ, "__final__", False):
                self.is_final = True
        self.is_thrift_enum = hasattr(self.typ, "_VALUES_TO_NAMES")
        self.base_classes |= set(get_mro(self.typ))
        if self.is_thrift_enum:
            self.artificial_bases.add(int)
        self.base_classes |= self.artificial_bases

    def is_assignable_to_type(self, typ: type) -> bool:
        for base in self.base_classes:
            if isinstance(base, str):
                continue
            else:
                if safe_issubclass(base, typ):
                    return True
        return self.is_universally_assignable

    def is_assignable_to_type_object(self, other: "TypeObject") -> bool:
        if isinstance(other.typ, super):
            return False
        if isinstance(other.typ, str):
            return (
                self.is_universally_assignable
                # TODO actually check protocols
                or other.is_protocol
                or other.typ in self.base_classes
            )
        return self.is_assignable_to_type(other.typ)

    def can_assign(
        self,
        self_val: Value,
        other_val: KnownValue | TypedValue | SubclassValue | AnnotatedValue,
        ctx: CanAssignContext,
    ) -> CanAssign:
        other_basic = replace_fallback(other_val)
        if not isinstance(other_basic, (KnownValue, TypedValue, SubclassValue)):
            return CanAssignError(f"Cannot assign {other_val} to {self}")
        other = other_basic.get_type_object(ctx)
        if other.is_universally_assignable:
            return {}
        if isinstance(self.typ, super):
            if isinstance(other.typ, super):
                return {}
            return CanAssignError(f"Cannot assign to super object {self}")
        if not self.is_protocol:
            if self.typ is object:
                return {}
            if other.is_protocol:
                if self._is_callable_protocol_assignment_target(other):
                    return self._can_assign_callable_protocol(self_val, other_val, ctx)
                return CanAssignError(
                    f"Cannot assign protocol {other_val} to non-protocol {self}"
                )
            if isinstance(self.typ, str):
                if safe_in(self.typ, other.base_classes):
                    return {}
                return CanAssignError(f"Cannot assign {other_val} to {self}")
            else:
                for base in other.base_classes:
                    if base is self.typ:
                        return {}
                    if isinstance(base, type) and safe_issubclass(base, self.typ):
                        return {}
                return CanAssignError(f"Cannot assign {other_val} to {self}")
        else:
            if isinstance(other.typ, super):
                return CanAssignError(
                    f"Cannot assign super object {other_val} to protocol {self}"
                )
            use_cache = not isinstance(other_val, AnnotatedValue)
            if use_cache:
                bounds_map = self._protocol_positive_cache.get(other_basic)
                if bounds_map is not None:
                    return bounds_map
            # This is a guard against infinite recursion if the Protocol is recursive
            if ctx.can_assume_compatibility(self, other):
                return {}
            with ctx.assume_compatibility(self, other):
                result = self._is_compatible_with_protocol(self_val, other_val, ctx)
                if isinstance(result, CanAssignError) and other.artificial_bases:
                    for base in other.artificial_bases:
                        subresult = self._is_compatible_with_protocol(
                            self_val, TypedValue(base), ctx
                        )
                        if not isinstance(subresult, CanAssignError):
                            result = subresult
                            break
            if use_cache and not isinstance(result, CanAssignError):
                self._protocol_positive_cache[other_basic] = result
            return result

    def _is_callable_protocol_assignment_target(self, other: "TypeObject") -> bool:
        return (
            self.typ is collections.abc.Callable
            and "__call__" in other.protocol_members
        )

    def _can_assign_callable_protocol(
        self,
        self_val: Value,
        other_val: KnownValue | TypedValue | SubclassValue | AnnotatedValue,
        ctx: CanAssignContext,
    ) -> CanAssign:
        expected_sig = self._as_concrete_signature(
            ctx.signature_from_value(self_val), ctx
        )
        actual_sig = self._as_concrete_signature(
            ctx.signature_from_value(other_val), ctx
        )
        if expected_sig is None or actual_sig is None:
            return CanAssignError(
                f"Cannot assign protocol {other_val} to non-protocol {self}"
            )
        return expected_sig.can_assign(actual_sig, ctx)

    @staticmethod
    def _as_concrete_signature(
        sig: Signature | BoundMethodSignature | OverloadedSignature | None,
        ctx: CanAssignContext,
    ) -> Signature | OverloadedSignature | None:
        if isinstance(sig, BoundMethodSignature):
            return sig.get_signature(ctx=ctx)
        return sig

    def _is_compatible_with_protocol(
        self, self_val: Value, other_val: Value, ctx: CanAssignContext
    ) -> CanAssign:
        from .relations import Relation, has_relation

        other_basic = replace_fallback(other_val)
        if isinstance(other_basic, (KnownValue, TypedValue, SubclassValue)):
            other_type_obj = other_basic.get_type_object(ctx)
        else:
            other_type_obj = None

        bounds_maps = []
        for member in self.protocol_members:
            expected = ctx.get_attribute_from_value(
                self_val, member, prefer_typeshed=True
            )
            if expected is UNINITIALIZED_VALUE:
                # In static fallback mode, synthetic protocol members may not have
                # a retrievable attribute type. Keep enforcing member presence.
                expected = AnyValue(AnySource.inference)
            expected = expected.substitute_typevars({SelfT: other_val})
            if _is_synthetic_type_name(self.typ, ctx):
                expected = _maybe_bind_dunder_protocol_member(
                    expected, member, self_val, ctx
                )
            # For __call__, we check compatibility with the other object itself.
            if member == "__call__":
                actual = other_val
            # Hack to allow types to be hashable. This avoids a bug where type objects
            # don't match the Hashable protocol if they define a __hash__ method themselves:
            # we compare against the __hash__ instance method, but compared to the protocol
            # it has an extra parameter (self).
            # It's a little unclear to me how this is supposed to work on protocols in
            # general: should they match against the type or the instance? PEP 544 suggests
            # that we should perhaps have a special case for matching against class objects
            # and modules, but that feels odd.
            # A better solution probably first requires a rewrite of the attribute fetching
            # system to make it more robust.
            elif member == "__hash__" and _should_use_permissive_dunder_hash(other_val):
                actual = AnyValue(AnySource.inference)
            else:
                actual = ctx.get_attribute_from_value(other_val, member)
                actual = _maybe_bind_dunder_protocol_member(
                    actual, member, other_val, ctx
                )
                if (
                    actual is UNINITIALIZED_VALUE
                    and other_type_obj is not None
                    and other_type_obj.is_protocol
                    and member in other_type_obj.protocol_members
                ):
                    actual = AnyValue(AnySource.inference)
            if actual is UNINITIALIZED_VALUE:
                can_assign = CanAssignError(f"{other_val} has no attribute {member!r}")
            else:
                can_assign = has_relation(expected, actual, Relation.ASSIGNABLE, ctx)
                if isinstance(can_assign, CanAssignError):
                    can_assign = CanAssignError(
                        f"Value of protocol member {member!r} conflicts", [can_assign]
                    )

            if isinstance(can_assign, CanAssignError):
                return can_assign
            bounds_maps.append(can_assign)
        return unify_bounds_maps(bounds_maps)

    def overrides_eq(self, self_val: Value, ctx: CanAssignContext) -> bool:
        if self.typ is type(None):
            return False
        member = ctx.get_attribute_from_value(self_val, "__eq__")
        sig = ctx.signature_from_value(member)
        if isinstance(sig, BoundMethodSignature):
            sig = sig.signature
        if isinstance(sig, OverloadedSignature):
            return True
        elif isinstance(sig, Signature):
            if len(sig.parameters) != 2:
                return True
            param = list(sig.parameters.values())[1]
            if param.kind in (
                ParameterKind.POSITIONAL_ONLY,
                ParameterKind.POSITIONAL_OR_KEYWORD,
            ) and param.annotation == TypedValue(object):
                return False
        return True

    def is_instance(self, obj: object) -> bool:
        """Whether obj is an instance of this type."""
        return safe_isinstance(obj, self.typ)

    def is_exactly(self, types: Container[type]) -> bool:
        return self.typ in types

    def can_be_unbound_method(self) -> bool:
        return self.is_exactly({cast(type, Callable), collections.abc.Callable, object})

    def is_metatype_of(self, other: "TypeObject") -> bool:
        if isinstance(self.typ, type) and isinstance(other.typ, type):
            return issubclass(self.typ, type) and safe_isinstance(other.typ, self.typ)
        else:
            # TODO handle this for synthetic types (if necessary)
            return False

    def has_attribute(self, attr: str, ctx: CanAssignContext) -> bool:
        """Whether this type definitely has this attribute."""
        if self.is_protocol:
            return attr in self.protocol_members
        # We don't use ctx.get_attribute because that may have false positives.
        for base in self.base_classes:
            try:
                present = attr in base.__dict__
            except Exception:
                present = False
            if present:
                return True
        return False

    def __str__(self) -> str:
        base = stringify_object(self.typ)
        if self.is_protocol:
            protocol_members = self._get_protocol_members_for_display()
            return (
                f"{base} (Protocol with members"
                f" {', '.join(map(repr, protocol_members))})"
            )
        return base

    def _get_protocol_members_for_display(self) -> list[str]:
        if not isinstance(self.typ, type):
            return sorted(self.protocol_members)

        members: list[str] = []
        seen = set()
        for base in get_mro(self.typ):
            for attr in base.__dict__:
                if attr in self.protocol_members and attr not in seen:
                    members.append(attr)
                    seen.add(attr)

            annotations = safe_getattr(base, "__annotations__", None)
            if not isinstance(annotations, dict):
                continue
            for attr in annotations:
                if attr in self.protocol_members and attr not in seen:
                    members.append(attr)
                    seen.add(attr)

        if len(seen) == len(self.protocol_members):
            return members
        return [*members, *sorted(self.protocol_members - seen)]


def _maybe_bind_dunder_protocol_member(
    value: Value, member: str, self_value: Value, ctx: CanAssignContext
) -> Value:
    if not (member.startswith("__") and member.endswith("__")):
        return value
    if value is UNINITIALIZED_VALUE:
        return value
    try:
        unwrapped = replace_fallback(value)
    except NotAGradualType:
        return value
    if isinstance(unwrapped, AnnotatedValue):
        extension_value: Value | None = None
        for extension in unwrapped.get_metadata_of_type(HasAttrExtension):
            if extension.attribute_name == KnownValue(member):
                extension_value = extension.attribute_type
                break
        if extension_value is None:
            return value
        value = extension_value
        try:
            unwrapped = replace_fallback(value)
        except NotAGradualType:
            return value
    if not isinstance(unwrapped, CallableValue):
        return value
    signature = unwrapped.signature
    if isinstance(signature, BoundMethodSignature):
        return value
    if isinstance(signature, (Signature, OverloadedSignature)):
        try:
            self_annotation_value = replace_fallback(self_value)
        except NotAGradualType:
            self_annotation_value = self_value
        bound = signature.bind_self(
            self_value=self_value, self_annotation_value=self_annotation_value, ctx=ctx
        )
        if bound is not None:
            return CallableValue(bound, unwrapped.typ)
    return value


def _should_use_permissive_dunder_hash(val: Value) -> bool:
    val = replace_fallback(val)
    if isinstance(val, SubclassValue):
        return True
    elif isinstance(val, KnownValue) and safe_isinstance(val.val, type):
        return True
    return False


def _is_synthetic_type_name(typ: object, ctx: CanAssignContext) -> bool:
    if not isinstance(typ, str):
        return False
    if typ.startswith("_typeshed."):
        return False
    synthetic_protocol_members = getattr(ctx, "synthetic_protocol_members", None)
    if (
        isinstance(synthetic_protocol_members, dict)
        and typ in synthetic_protocol_members
    ):
        return True
    synthetic_type_bases = getattr(ctx, "synthetic_type_bases", None)
    if isinstance(synthetic_type_bases, dict) and typ in synthetic_type_bases:
        return True
    synthetic_class_attributes = getattr(ctx, "synthetic_class_attributes", None)
    if (
        isinstance(synthetic_class_attributes, dict)
        and typ in synthetic_class_attributes
    ):
        return True
    return typ.startswith("<") or "/" in typ
