"""

An object that represents a type.

"""

import collections.abc
import inspect
from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import get_args, get_origin
from unittest import mock

from typing_extensions import assert_never

from pycroscope.relations import (
    infer_positional_generic_typevar_map,
    translate_generic_typevar_map,
)
from pycroscope.signature import (
    BoundMethodSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    mark_ellipsis_style_any_tail_parameters,
)

from .safe import (
    is_namedtuple_class,
    is_typing_name,
    safe_getattr,
    safe_in,
    safe_isinstance,
    safe_issubclass,
)
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
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    PredicateValue,
    SelfT,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedValue,
    TypeFormValue,
    TypeVarLike,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    flatten_values,
    freshen_typevars_for_inference,
    get_tv_map,
    replace_fallback,
    stringify_object,
    unify_bounds_maps,
    unite_values,
)

SYNTHETIC_PROPERTY_GETTER_PREFIX = "%property_getter:"
SYNTHETIC_PROPERTY_SETTER_PREFIX = "%property_setter:"
SYNTHETIC_READONLY_ATTRIBUTES_KEY = "%readonly_attributes"


def _as_concrete_signature(
    sig: Signature | BoundMethodSignature | OverloadedSignature | None,
    ctx: CanAssignContext,
) -> Signature | OverloadedSignature | None:
    if isinstance(sig, BoundMethodSignature):
        return sig.get_signature(ctx=ctx)
    return sig


@dataclass(frozen=True)
class _MemberDescriptor:
    value: Value
    is_classvar: bool
    is_method: bool
    is_property: bool
    property_has_setter: bool
    is_writable: bool
    is_readonly: bool


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
    _protocol_positive_cache: dict[tuple[Value, Value], BoundsMap] = field(
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
            use_cache = not isinstance(self_val, AnnotatedValue) and not isinstance(
                other_val, AnnotatedValue
            )
            cache_key: tuple[Value, Value] | None = None
            if use_cache:
                cache_key = (self_val, other_val)
                bounds_map = self._protocol_positive_cache.get(cache_key)
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
            if (
                use_cache
                and cache_key is not None
                and not isinstance(result, CanAssignError)
            ):
                self._protocol_positive_cache[cache_key] = result
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
        expected_sig = _as_concrete_signature(ctx.signature_from_value(self_val), ctx)
        actual_sig = _as_concrete_signature(ctx.signature_from_value(other_val), ctx)
        if expected_sig is None or actual_sig is None:
            return CanAssignError(
                f"Cannot assign protocol {other_val} to non-protocol {self}"
            )
        return expected_sig.can_assign(actual_sig, ctx)

    def _is_compatible_with_protocol(
        self, self_val: Value, other_val: Value, ctx: CanAssignContext
    ) -> CanAssign:
        from .relations import Relation, has_relation

        other_basic = replace_fallback(other_val)
        if isinstance(other_basic, (KnownValue, TypedValue, SubclassValue)):
            other_type_obj = other_basic.get_type_object(ctx)
        else:
            other_type_obj = None
        other_type_key = _receiver_key_from_value(other_val)

        bounds_maps = []
        protocol_type_key: type | str | None
        if isinstance(self.typ, (type, str)):
            protocol_type_key = self.typ
        else:
            protocol_type_key = None
        if len(self.protocol_members) > 1:
            protocol_self_typevar_map = _collect_protocol_self_typevar_map(
                protocol_type_key, self.protocol_members, other_val, ctx
            )
            actual_self_typevar_map = _collect_protocol_self_typevar_map(
                other_type_key, self.protocol_members, other_val, ctx
            )
        else:
            protocol_self_typevar_map = {}
            actual_self_typevar_map = {}
        apply_synthetic_member_rules = (
            isinstance(protocol_type_key, str)
            and _get_synthetic_class_for_key(protocol_type_key, ctx) is not None
        )
        class_object_check = _is_definitely_class_object_value(other_val)
        for member in self.protocol_members:
            is_dunder_member = member.startswith("__") and member.endswith("__")
            use_descriptor_rules = apply_synthetic_member_rules or (
                class_object_check and not is_dunder_member
            )
            # For __call__, we check compatibility with the other object itself.
            if member == "__call__":
                expected = UNINITIALIZED_VALUE
                if isinstance(self.typ, str):
                    checker_ctx = safe_getattr(ctx, "checker", ctx)
                    get_synthetic_class = safe_getattr(
                        checker_ctx, "get_synthetic_class", None
                    )
                    if callable(get_synthetic_class):
                        synthetic_class = get_synthetic_class(self.typ)
                        if synthetic_class is not None:
                            expected = ctx.get_attribute_from_value(
                                synthetic_class, member
                            )
                if expected is UNINITIALIZED_VALUE:
                    expected_signature = _as_concrete_signature(
                        ctx.signature_from_value(self_val), ctx
                    )
                    if expected_signature is None:
                        expected = AnyValue(AnySource.inference)
                    else:
                        if _should_mark_protocol_call_tail(self_val):
                            expected_signature = _mark_protocol_call_signature_tail(
                                expected_signature
                            )
                        expected = CallableValue(expected_signature)
                expected = _bind_protocol_call_expected(
                    expected, other_val, ctx, protocol_self_value=self_val
                )
                expected = expected.substitute_typevars({SelfT: other_val})
                if other_type_obj is not None and other_type_obj.is_protocol:
                    actual = _get_protocol_call_member_value(
                        other_type_obj.typ, other_val, ctx
                    )
                    if actual is UNINITIALIZED_VALUE:
                        actual = other_val
                else:
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
                expected = ctx.get_attribute_from_value(
                    self_val, member, prefer_typeshed=True
                )
                if expected is UNINITIALIZED_VALUE:
                    # In static fallback mode, synthetic protocol members may not have
                    # a retrievable attribute type. Keep enforcing member presence.
                    expected = AnyValue(AnySource.inference)
                expected = expected.substitute_typevars({SelfT: other_val})
                actual = AnyValue(AnySource.inference)
            else:
                expected = ctx.get_attribute_from_value(
                    self_val, member, prefer_typeshed=True
                )
                if expected is UNINITIALIZED_VALUE:
                    # In static fallback mode, synthetic protocol members may not have
                    # a retrievable attribute type. Keep enforcing member presence.
                    expected = AnyValue(AnySource.inference)
                if protocol_self_typevar_map:
                    expected = expected.substitute_typevars(protocol_self_typevar_map)
                expected = expected.substitute_typevars({SelfT: other_val})
                actual = ctx.get_attribute_from_value(other_val, member)
                if (
                    class_object_check
                    and other_type_key is not None
                    and _should_refine_class_object_member_lookup(
                        actual, other_type_key, member, ctx
                    )
                ):
                    class_owner = _class_object_value_for_key(other_type_key, ctx)
                    if class_owner is not None:
                        refined_actual = ctx.get_attribute_from_value(
                            class_owner, member
                        )
                        if refined_actual is not UNINITIALIZED_VALUE:
                            actual = refined_actual
                if actual_self_typevar_map and actual is not UNINITIALIZED_VALUE:
                    actual = actual.substitute_typevars(actual_self_typevar_map)
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
                if _is_callable_member_value(expected, ctx):
                    expected = _normalize_protocol_member_value_for_relation(
                        expected, ctx, receiver=self_val
                    )
                if _is_callable_member_value(actual, ctx):
                    actual = _normalize_protocol_member_value_for_relation(
                        actual, ctx, receiver=other_val
                    )
                if not use_descriptor_rules:
                    can_assign = has_relation(
                        expected, actual, Relation.ASSIGNABLE, ctx
                    )
                else:
                    expected_desc = _describe_member_for_type(
                        protocol_type_key,
                        self_val,
                        member,
                        expected,
                        ctx,
                        class_object_access=False,
                    )
                    actual_desc = _describe_member_for_type(
                        other_type_key,
                        other_val,
                        member,
                        actual,
                        ctx,
                        class_object_access=class_object_check,
                    )
                    if class_object_check and expected_desc.is_classvar:
                        can_assign = CanAssignError(
                            f"Protocol member {member!r} is a ClassVar"
                        )
                    elif (
                        class_object_check
                        and not expected_desc.is_classvar
                        and not expected_desc.is_method
                        and not expected_desc.is_property
                        and not _is_callable_member_value(expected_desc.value, ctx)
                        and other_type_key is not None
                        and not actual_desc.is_classvar
                        and not _is_member_from_metaclass(other_type_key, member, ctx)
                    ):
                        can_assign = CanAssignError(
                            f"Protocol member {member!r} is an instance attribute"
                        )
                    elif (
                        not class_object_check
                        and not is_dunder_member
                        and expected_desc.is_classvar != actual_desc.is_classvar
                    ):
                        can_assign = CanAssignError(
                            f"ClassVar status of protocol member {member!r} conflicts"
                        )
                    else:
                        expected_for_relation = expected_desc.value
                        actual_for_relation = actual_desc.value
                        expected_is_writable_data_member = (
                            not expected_desc.is_method
                            and not expected_desc.is_property
                            and not expected_desc.is_classvar
                            and not expected_desc.is_readonly
                            and not _is_callable_member_value(expected_desc.value, ctx)
                        )
                        expected_requires_writable = (
                            expected_desc.is_property
                            and expected_desc.property_has_setter
                        ) or expected_is_writable_data_member
                        if class_object_check:
                            expected_requires_writable = False
                        if expected_requires_writable and not actual_desc.is_writable:
                            can_assign = CanAssignError(
                                f"Protocol member {member!r} is not writable"
                            )
                        else:
                            expected_for_relation = freshen_typevars_for_inference(
                                expected_for_relation
                            )
                            if _is_callable_member_value(expected_for_relation, ctx):
                                expected_for_relation = (
                                    _normalize_protocol_member_value_for_relation(
                                        expected_for_relation, ctx, receiver=self_val
                                    )
                                )
                            if _is_callable_member_value(actual_for_relation, ctx):
                                actual_for_relation = (
                                    _normalize_protocol_member_value_for_relation(
                                        actual_for_relation, ctx, receiver=other_val
                                    )
                                )
                            can_assign = has_relation(
                                expected_for_relation,
                                actual_for_relation,
                                Relation.ASSIGNABLE,
                                ctx,
                            )
                            if (
                                not isinstance(can_assign, CanAssignError)
                                and expected_requires_writable
                            ):
                                reverse = has_relation(
                                    actual_for_relation,
                                    expected_for_relation,
                                    Relation.ASSIGNABLE,
                                    ctx,
                                )
                                if isinstance(reverse, CanAssignError):
                                    can_assign = reverse
                if isinstance(can_assign, CanAssignError):
                    can_assign = CanAssignError(
                        f"Value of protocol member {member!r} conflicts", [can_assign]
                    )

            if isinstance(can_assign, CanAssignError):
                return can_assign
            bounds_maps.append(can_assign)
        bounds_map = unify_bounds_maps(bounds_maps)
        # Protocol members can introduce shared type-variable constraints; reject
        # matches where those constraints cannot be solved consistently.
        from .typevar import resolve_bounds_map

        _, errors = resolve_bounds_map(bounds_map, ctx)
        if errors:
            return CanAssignError(
                "Conflicting type constraints for protocol members", list(errors)
            )
        return bounds_map

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
        if not isinstance(self.typ, type):
            return False
        return self.typ in types

    def can_be_unbound_method(self) -> bool:
        return self.is_exactly({Callable, collections.abc.Callable, object})

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


def _describe_member_for_type(
    class_key: type | str | None,
    owner_value: Value,
    member: str,
    resolved_value: Value,
    ctx: CanAssignContext,
    *,
    class_object_access: bool = False,
) -> _MemberDescriptor:
    is_classvar = class_key is not None and _is_member_classvar(
        class_key, member, ctx, seen=set()
    )
    is_property_member = class_key is not None and _is_member_property(
        class_key, member, ctx, seen=set()
    )
    if class_object_access:
        property_getter, property_has_setter = None, False
    else:
        property_getter, property_has_setter = _get_property_member_value(
            class_key, owner_value, member, resolved_value, ctx
        )
    is_property = property_getter is not None
    if class_object_access and is_property_member:
        is_property = False
    is_method = (
        not is_property
        and class_key is not None
        and _is_member_method(class_key, member, ctx, seen=set())
    )
    if class_object_access and is_property_member:
        is_method = False
    value = property_getter if property_getter is not None else resolved_value
    if (
        class_object_access
        and is_property_member
        and not _is_property_marker_value(value)
    ):
        value = TypedValue(property)
    is_readonly = class_key is not None and is_readonly_annotated_member(
        class_key, member, ctx
    )
    if is_classvar or is_method:
        is_writable = False
    elif is_property:
        is_writable = property_has_setter
    else:
        is_writable = class_key is None or not _is_readonly_instance_member(
            class_key, member, ctx
        )
    return _MemberDescriptor(
        value=value,
        is_classvar=is_classvar,
        is_method=is_method,
        is_property=is_property,
        property_has_setter=property_has_setter,
        is_writable=is_writable,
        is_readonly=is_readonly,
    )


def _readonly_names_from_mapping(attributes: Mapping[str, Value]) -> set[str]:
    raw = attributes.get(SYNTHETIC_READONLY_ATTRIBUTES_KEY)
    if isinstance(raw, KnownValue) and isinstance(
        raw.val, (set, frozenset, tuple, list)
    ):
        return {item for item in raw.val if isinstance(item, str)}
    return set()


def _runtime_annotation_has_readonly(annotation: object) -> bool:
    origin = get_origin(annotation)
    if is_typing_name(annotation, "ReadOnly") or is_typing_name(origin, "ReadOnly"):
        return True
    if isinstance(annotation, str):
        return "ReadOnly" in annotation
    if origin is not None:
        args = get_args(annotation)
        if args and (
            is_typing_name(origin, "Annotated")
            or is_typing_name(origin, "ClassVar")
            or is_typing_name(origin, "Final")
        ):
            return _runtime_annotation_has_readonly(args[0])
    return False


def is_readonly_annotated_member(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    return _is_readonly_annotated_member(class_key, member, ctx, seen=set())


def _is_readonly_annotated_member(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> bool:
    if class_key in seen:
        return False
    seen.add(class_key)

    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        if member in _readonly_names_from_mapping(synthetic.class_attributes):
            return True
        annotations = synthetic.class_attributes.get("__annotations__")
        if (
            isinstance(annotations, KnownValue)
            and isinstance(annotations.val, Mapping)
            and member in annotations.val
            and _runtime_annotation_has_readonly(annotations.val[member])
        ):
            return True

    if isinstance(class_key, type):
        annotations = safe_getattr(class_key, "__annotations__", None)
        if (
            isinstance(annotations, Mapping)
            and member in annotations
            and _runtime_annotation_has_readonly(annotations[member])
        ):
            return True

    return any(
        _is_readonly_annotated_member(base, member, ctx, seen=seen)
        for base in _iter_base_keys(class_key, ctx)
    )


def _class_key_from_value(value: Value) -> type | str | None:
    keys = list(dict.fromkeys(_iter_class_keys_from_value(value)))
    if len(keys) == 1:
        return keys[0]
    return None


def _iter_class_keys_from_value(value: Value) -> list[type | str]:
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        keys: list[type | str] = []
        for subval in value.vals:
            keys.extend(_iter_class_keys_from_value(subval))
        return keys
    if isinstance(value, IntersectionValue):
        keys = []
        for member in value.vals:
            keys.extend(_iter_class_keys_from_value(member))
        return keys
    return _iter_class_keys_from_simple_value(value)


def _iter_class_keys_from_simple_value(value: SimpleType) -> list[type | str]:
    if isinstance(value, SyntheticClassObjectValue):
        return _typed_class_key(value.class_type)
    if isinstance(value, SubclassValue):
        return _iter_class_keys_from_value(value.typ)
    if isinstance(value, TypedValue):
        return _typed_class_key(value)
    if isinstance(value, KnownValue) and isinstance(value.val, type):
        return [value.val]
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypeFormValue,
            PredicateValue,
            KnownValue,
        ),
    ):
        return []
    assert_never(value)


def _typed_class_key(value: Value) -> list[type | str]:
    if isinstance(value, TypedValue) and isinstance(value.typ, (type, str)):
        return [value.typ]
    return []


def _checker_ctx(ctx: CanAssignContext) -> object:
    return safe_getattr(ctx, "checker", ctx)


def _get_synthetic_class_for_key(
    class_key: type | str, ctx: CanAssignContext
) -> SyntheticClassObjectValue | None:
    get_synthetic_class = safe_getattr(ctx, "get_synthetic_class", None)
    if callable(get_synthetic_class):
        synthetic = get_synthetic_class(class_key)
        if synthetic is not None:
            return synthetic
    checker = _checker_ctx(ctx)
    get_synthetic_class = safe_getattr(checker, "get_synthetic_class", None)
    if callable(get_synthetic_class):
        return get_synthetic_class(class_key)
    return None


def _iter_base_keys(class_key: type | str, ctx: CanAssignContext) -> list[type | str]:
    bases: list[type | str] = []
    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        for base in synthetic.base_classes:
            for flattened in flatten_values(
                replace_fallback(base), unwrap_annotated=True
            ):
                flattened_key = _class_key_from_value(flattened)
                if flattened_key is not None:
                    bases.append(flattened_key)
    if isinstance(class_key, type):
        bases.extend(
            base
            for base in safe_getattr(class_key, "__bases__", ())
            if isinstance(base, type)
        )
    else:
        checker = _checker_ctx(ctx)
        get_generic_bases = safe_getattr(checker, "get_generic_bases", None)
        if callable(get_generic_bases):
            try:
                generic_bases = get_generic_bases(class_key)
            except Exception:
                generic_bases = {}
            for base in generic_bases:
                if isinstance(base, (type, str)) and base != class_key:
                    bases.append(base)
    return bases


def _is_member_classvar(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> bool:
    if class_key in seen:
        return False
    seen.add(class_key)

    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        classvars = synthetic.class_attributes.get("%classvars")
        if isinstance(classvars, KnownValue) and isinstance(
            classvars.val, (set, frozenset, tuple, list)
        ):
            if member in {name for name in classvars.val if isinstance(name, str)}:
                return True

    if isinstance(class_key, type):
        annotations = safe_getattr(class_key, "__annotations__", None)
        if isinstance(annotations, dict):
            annotation = annotations.get(member)
            if annotation is not None:
                origin = get_origin(annotation)
                if safe_getattr(origin, "__name__", None) == "ClassVar":
                    return True
                if safe_getattr(annotation, "__name__", None) == "ClassVar":
                    return True
                if isinstance(annotation, str) and "ClassVar" in annotation:
                    return True

    return any(
        _is_member_classvar(base, member, ctx, seen=seen)
        for base in _iter_base_keys(class_key, ctx)
    )


def _is_member_defined_on_class_key(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> bool:
    if class_key in seen:
        return False
    seen.add(class_key)

    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        if member in synthetic.class_attributes:
            return True
        annotations = synthetic.class_attributes.get("__annotations__")
        if (
            isinstance(annotations, KnownValue)
            and isinstance(annotations.val, dict)
            and member in annotations.val
        ):
            return True

    if isinstance(class_key, type):
        base_dict = safe_getattr(class_key, "__dict__", None)
        if isinstance(base_dict, Mapping) and member in base_dict:
            return True
        annotations = safe_getattr(class_key, "__annotations__", None)
        if isinstance(annotations, Mapping) and member in annotations:
            return True

    return any(
        _is_member_defined_on_class_key(base, member, ctx, seen=seen)
        for base in _iter_base_keys(class_key, ctx)
    )


def _is_member_method(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> bool:
    if class_key in seen:
        return False
    seen.add(class_key)

    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None and member in synthetic.method_attributes:
        return True

    if isinstance(class_key, type):
        descriptor = inspect.getattr_static(class_key, member, UNINITIALIZED_VALUE)
        if descriptor is not UNINITIALIZED_VALUE:
            if isinstance(descriptor, property):
                return False
            if isinstance(descriptor, (staticmethod, classmethod)):
                return True
            if inspect.isfunction(descriptor) or inspect.ismethoddescriptor(descriptor):
                return True
            return False

    return any(
        _is_member_method(base, member, ctx, seen=seen)
        for base in _iter_base_keys(class_key, ctx)
    )


def _is_member_property(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> bool:
    if class_key in seen:
        return False
    seen.add(class_key)

    property_getter = _get_synthetic_property_metadata(
        class_key, member, SYNTHETIC_PROPERTY_GETTER_PREFIX, ctx, seen=set()
    )
    if property_getter is not None:
        return True

    synthetic_member_value = _get_synthetic_member_value(
        class_key, member, ctx, seen=set()
    )
    if synthetic_member_value is not None and _is_property_marker_value(
        synthetic_member_value
    ):
        return True

    if isinstance(class_key, type):
        descriptor = inspect.getattr_static(class_key, member, UNINITIALIZED_VALUE)
        if isinstance(descriptor, property):
            return True

    return any(
        _is_member_property(base, member, ctx, seen=seen)
        for base in _iter_base_keys(class_key, ctx)
    )


def _is_property_marker_value(value: Value) -> bool:
    value = replace_fallback(value)
    return (
        isinstance(value, KnownValue)
        and isinstance(value.val, property)
        or isinstance(value, (TypedValue, GenericValue))
        and value.typ is property
    )


def _get_property_member_value(
    class_key: type | str | None,
    owner_value: Value,
    member: str,
    resolved_value: Value,
    ctx: CanAssignContext,
) -> tuple[Value | None, bool]:
    if class_key is None:
        return None, False

    property_getter = _get_synthetic_property_metadata(
        class_key, member, SYNTHETIC_PROPERTY_GETTER_PREFIX, ctx, seen=set()
    )
    if property_getter is not None:
        property_setter = _get_synthetic_property_metadata(
            class_key, member, SYNTHETIC_PROPERTY_SETTER_PREFIX, ctx, seen=set()
        )
        return property_getter, property_setter is not None

    synthetic_member_value = _get_synthetic_member_value(
        class_key, member, ctx, seen=set()
    )
    if synthetic_member_value is not None and _is_property_marker_value(
        synthetic_member_value
    ):
        if resolved_value is not UNINITIALIZED_VALUE and not _is_property_marker_value(
            resolved_value
        ):
            return resolved_value, False
        return AnyValue(AnySource.inference), False

    if isinstance(class_key, type):
        descriptor = inspect.getattr_static(class_key, member, UNINITIALIZED_VALUE)
        if isinstance(descriptor, property):
            getter = resolved_value
            if getter is UNINITIALIZED_VALUE or _is_property_marker_value(getter):
                getter = ctx.get_attribute_from_value(owner_value, member)
            if getter is UNINITIALIZED_VALUE or _is_property_marker_value(getter):
                getter = AnyValue(AnySource.inference)
            return getter, descriptor.fset is not None

    return None, False


def _get_synthetic_property_metadata(
    class_key: type | str,
    member: str,
    prefix: str,
    ctx: CanAssignContext,
    *,
    seen: set[type | str],
) -> Value | None:
    if class_key in seen:
        return None
    seen.add(class_key)
    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is None:
        return None
    key = f"{prefix}{member}"
    if key in synthetic.class_attributes:
        return synthetic.class_attributes[key]
    for base in _iter_base_keys(class_key, ctx):
        found = _get_synthetic_property_metadata(base, member, prefix, ctx, seen=seen)
        if found is not None:
            return found
    return None


def _get_synthetic_member_value(
    class_key: type | str, member: str, ctx: CanAssignContext, *, seen: set[type | str]
) -> Value | None:
    if class_key in seen:
        return None
    seen.add(class_key)
    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is None:
        return None
    if member in synthetic.class_attributes:
        return synthetic.class_attributes[member]
    for base in _iter_base_keys(class_key, ctx):
        found = _get_synthetic_member_value(base, member, ctx, seen=seen)
        if found is not None:
            return found
    return None


def _is_readonly_instance_member(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    if isinstance(class_key, type):
        if is_namedtuple_class(class_key):
            fields = safe_getattr(class_key, "_fields", None)
            if isinstance(fields, tuple) and member in fields:
                return True
        dataclass_params = safe_getattr(class_key, "__dataclass_params__", None)
        if safe_getattr(dataclass_params, "frozen", None) is True:
            return True
    if is_readonly_annotated_member(class_key, member, ctx):
        return True

    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        namedtuple_marker = synthetic.class_attributes.get("%namedtuple")
        fields = synthetic.class_attributes.get("%instance_only_annotations")
        if isinstance(namedtuple_marker, KnownValue) and namedtuple_marker.val is True:
            if (
                isinstance(fields, KnownValue)
                and isinstance(fields.val, (set, frozenset, tuple, list))
                and member in fields.val
            ):
                return True
            if (
                member in synthetic.class_attributes
                and not member.startswith("%")
                and member not in synthetic.method_attributes
            ):
                return True
    if synthetic is not None and synthetic.dataclass_frozen is True:
        return True
    return False


def _is_callable_member_value(value: Value, ctx: CanAssignContext) -> bool:
    if isinstance(value, CallableValue):
        return True
    if isinstance(value, KnownValue) and callable(value.val):
        return True
    signature = ctx.signature_from_value(value)
    if isinstance(signature, (Signature, OverloadedSignature, BoundMethodSignature)):
        return True
    value = replace_fallback(value)
    if isinstance(value, CallableValue):
        return True
    if isinstance(value, KnownValue) and callable(value.val):
        return True
    signature = ctx.signature_from_value(value)
    return isinstance(signature, (Signature, OverloadedSignature, BoundMethodSignature))


def _normalize_protocol_member_value_for_relation(
    value: Value, ctx: CanAssignContext, *, receiver: Value | None = None
) -> Value:
    signature = ctx.signature_from_value(value)
    if isinstance(signature, BoundMethodSignature):
        bound = signature.get_signature(ctx=ctx)
        if bound is None:
            bound = signature.signature.bind_self(
                self_value=replace_fallback(signature.self_composite.value), ctx=ctx
            )
        if bound is not None:
            return CallableValue(bound)
        return CallableValue(signature.signature)
    if (
        receiver is not None
        and isinstance(signature, (Signature, OverloadedSignature))
        and _callable_value_missing_receiver(value, ctx)
    ):
        bound = signature.bind_self(self_value=replace_fallback(receiver), ctx=ctx)
        if bound is not None:
            return CallableValue(bound)
    return value


def _metaclass_key_for_class(
    class_key: type | str, ctx: CanAssignContext
) -> type | str | None:
    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        metaclass = synthetic.class_attributes.get("%metaclass")
        if isinstance(metaclass, Value):
            metaclass_key = _class_key_from_value(metaclass)
            if metaclass_key is not None:
                return metaclass_key
    if isinstance(class_key, type):
        return type(class_key)
    return None


def _is_member_from_metaclass(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    metaclass_key = _metaclass_key_for_class(class_key, ctx)
    if metaclass_key is None:
        return False
    return _is_member_defined_on_class_key(metaclass_key, member, ctx, seen=set())


def _should_refine_class_object_member_lookup(
    actual: Value, class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    if _is_member_from_metaclass(class_key, member, ctx):
        return False
    if actual is UNINITIALIZED_VALUE:
        return True
    unwrapped = replace_fallback(actual)
    if isinstance(unwrapped, AnyValue):
        return True
    if not _is_member_method(class_key, member, ctx, seen=set()):
        return False
    return _callable_value_missing_receiver(unwrapped, ctx)


def _callable_value_missing_receiver(value: Value, ctx: CanAssignContext) -> bool:
    signature = ctx.signature_from_value(value)
    if isinstance(signature, BoundMethodSignature):
        return True
    if isinstance(signature, Signature):
        return signature.bind_self(ctx=ctx) is None
    if isinstance(signature, OverloadedSignature):
        if not signature.signatures:
            return True
        return any(subsig.bind_self(ctx=ctx) is None for subsig in signature.signatures)
    return False


def _signature_has_receiver_parameter(
    signature: Signature,
    self_value: Value,
    *,
    ctx: CanAssignContext,
    member: str,
    allow_any_annotation: bool = False,
) -> bool:
    first_parameter = next(iter(signature.parameters.values()), None)
    if first_parameter is None or first_parameter.kind not in (
        ParameterKind.POSITIONAL_ONLY,
        ParameterKind.POSITIONAL_OR_KEYWORD,
    ):
        return False
    annotation = replace_fallback(first_parameter.annotation)
    if allow_any_annotation and isinstance(annotation, AnyValue):
        return True
    if isinstance(annotation, AnyValue):
        return False
    if any(
        isinstance(subval, TypeVarValue) and subval.typevar_param.typevar is SelfT
        for subval in annotation.walk_values()
    ):
        return True
    annotation_key = _receiver_key_from_value(annotation)
    self_key = _receiver_key_from_value(self_value)
    if annotation_key is not None and self_key is not None:
        if annotation_key == self_key or stringify_object(
            annotation_key
        ) == stringify_object(self_key):
            return True
        if isinstance(annotation_key, str) and isinstance(self_key, type):
            return annotation_key.rsplit(".", maxsplit=1)[-1] == self_key.__name__
        if isinstance(annotation_key, type) and isinstance(self_key, str):
            return self_key.rsplit(".", maxsplit=1)[-1] == annotation_key.__name__
        if not isinstance(annotation_key, str) and not isinstance(self_key, str):
            return False
        return not isinstance(
            get_tv_map(first_parameter.annotation, self_value, ctx), CanAssignError
        )
    return not isinstance(
        get_tv_map(first_parameter.annotation, self_value, ctx), CanAssignError
    )


def _receiver_key_from_value(value: Value) -> type | str | None:
    key = _class_key_from_value(value)
    if key is not None:
        return key
    narrowed = replace_fallback(value)
    if isinstance(narrowed, KnownValue) and not isinstance(narrowed.val, type):
        return type(narrowed.val)
    return None


def _class_object_value_for_key(
    class_key: type | str, ctx: CanAssignContext
) -> Value | None:
    synthetic = _get_synthetic_class_for_key(class_key, ctx)
    if synthetic is not None:
        return synthetic
    if isinstance(class_key, type):
        return KnownValue(class_key)
    return None


def _bind_protocol_call_expected(
    value: Value,
    self_value: Value,
    ctx: CanAssignContext,
    *,
    protocol_self_value: Value | None = None,
) -> Value:
    unwrapped = replace_fallback(value)
    if not isinstance(unwrapped, CallableValue):
        return value
    signature = unwrapped.signature
    if isinstance(signature, BoundMethodSignature):
        return value
    receiver_reference = (
        self_value if protocol_self_value is None else protocol_self_value
    )
    allow_any_annotation = protocol_self_value is not None
    if isinstance(signature, (Signature, OverloadedSignature)):
        if isinstance(signature, Signature):
            has_receiver_parameter = _signature_has_receiver_parameter(
                signature,
                receiver_reference,
                ctx=ctx,
                member="__call__",
                allow_any_annotation=allow_any_annotation,
            )
        else:
            has_receiver_parameter = all(
                _signature_has_receiver_parameter(
                    sig,
                    receiver_reference,
                    ctx=ctx,
                    member="__call__",
                    allow_any_annotation=allow_any_annotation,
                )
                for sig in signature.signatures
            )
        if not has_receiver_parameter:
            return value
        bound = signature.bind_self(self_value=self_value, ctx=ctx)
        if bound is not None:
            return CallableValue(bound, unwrapped.typ)
    return value


def _get_protocol_call_member_value(
    protocol_typ: type | super | str, self_value: Value, ctx: CanAssignContext
) -> Value:
    call_member = UNINITIALIZED_VALUE
    if isinstance(protocol_typ, super):
        return call_member
    if isinstance(protocol_typ, str):
        checker_ctx = safe_getattr(ctx, "checker", ctx)
        get_synthetic_class = safe_getattr(checker_ctx, "get_synthetic_class", None)
        if callable(get_synthetic_class):
            synthetic_class = get_synthetic_class(protocol_typ)
            if synthetic_class is not None:
                call_member = ctx.get_attribute_from_value(synthetic_class, "__call__")
    if call_member is UNINITIALIZED_VALUE:
        call_member = ctx.get_attribute_from_value(
            self_value, "__call__", prefer_typeshed=True
        )
    if call_member is UNINITIALIZED_VALUE:
        return call_member
    return _bind_protocol_call_expected(
        call_member, self_value, ctx, protocol_self_value=self_value
    )


def _should_mark_protocol_call_tail(value: Value) -> bool:
    return not isinstance(replace_fallback(value), GenericValue)


def _mark_protocol_call_signature_tail(
    signature: Signature | OverloadedSignature,
) -> Signature | OverloadedSignature:
    if isinstance(signature, Signature):
        marked_params = mark_ellipsis_style_any_tail_parameters(
            list(signature.parameters.values())
        )
        return replace(
            signature, parameters={param.name: param for param in marked_params}
        )
    marked_signatures = []
    for sig in signature.signatures:
        marked_params = mark_ellipsis_style_any_tail_parameters(
            list(sig.parameters.values())
        )
        marked_signatures.append(
            replace(sig, parameters={param.name: param for param in marked_params})
        )
    return OverloadedSignature(marked_signatures)


def _collect_protocol_self_typevar_map(
    protocol_key: type | str | None,
    protocol_members: set[str],
    receiver_value: Value,
    ctx: CanAssignContext,
) -> dict[TypeVarLike, Value]:
    """Collect typevar substitutions implied by receiver annotations.

    This propagates `self: T` constraints across protocol members.
    """
    synthetic_class = None
    if isinstance(protocol_key, str):
        synthetic_class = _get_synthetic_class_for_key(protocol_key, ctx)
        if synthetic_class is None:
            return {}
    elif not isinstance(protocol_key, type):
        return {}

    tv_map: dict[TypeVarLike, Value] = {}
    for member in protocol_members:
        receiver_for_match = receiver_value
        if synthetic_class is not None:
            raw_attr = synthetic_class.class_attributes.get(member, UNINITIALIZED_VALUE)
            if raw_attr is UNINITIALIZED_VALUE:
                continue
            raw_attr = replace_fallback(raw_attr)
            if isinstance(raw_attr, GenericValue) and raw_attr.typ is classmethod:
                if not raw_attr.args:
                    continue
                self_annotation = SubclassValue.make(raw_attr.args[0])
                receiver_key = _class_key_from_value(receiver_value)
                if receiver_key is not None:
                    class_object = _class_object_value_for_key(receiver_key, ctx)
                else:
                    class_object = None
                if class_object is not None:
                    receiver_for_match = class_object
                else:
                    receiver_for_match = SubclassValue.make(
                        receiver_value.get_type_value()
                    )
            elif isinstance(raw_attr, GenericValue) and raw_attr.typ is staticmethod:
                continue
            else:
                signature = _as_concrete_signature(
                    ctx.signature_from_value(raw_attr), ctx
                )
                if signature is None:
                    continue
                signatures = (
                    signature.signatures
                    if isinstance(signature, OverloadedSignature)
                    else [signature]
                )
                self_annotation = None
                for concrete in signatures:
                    parameters = list(concrete.parameters.values())
                    if parameters:
                        self_annotation = parameters[0].annotation
                        break
                if self_annotation is None:
                    continue
        else:
            descriptor = inspect.getattr_static(
                protocol_key, member, UNINITIALIZED_VALUE
            )
            if descriptor is UNINITIALIZED_VALUE:
                continue
            if isinstance(descriptor, property):
                callable_obj = descriptor.fget
            elif isinstance(descriptor, staticmethod):
                callable_obj = None
            elif isinstance(descriptor, classmethod):
                callable_obj = descriptor.__func__
                receiver_key = _class_key_from_value(receiver_value)
                if receiver_key is not None:
                    class_object = _class_object_value_for_key(receiver_key, ctx)
                else:
                    class_object = None
                if class_object is not None:
                    receiver_for_match = class_object
                else:
                    receiver_for_match = SubclassValue.make(
                        receiver_value.get_type_value()
                    )
            elif inspect.isfunction(descriptor):
                callable_obj = descriptor
            else:
                callable_obj = None
            if callable_obj is None:
                continue
            signature = _as_concrete_signature(
                ctx.signature_from_value(KnownValue(callable_obj)), ctx
            )
            if signature is None:
                continue
            signatures = (
                signature.signatures
                if isinstance(signature, OverloadedSignature)
                else [signature]
            )
            for concrete in signatures:
                parameters = list(concrete.parameters.values())
                if not parameters:
                    continue
                self_annotation = parameters[0].annotation
                if not any(
                    isinstance(subvalue, TypeVarValue)
                    for subvalue in self_annotation.walk_values()
                ):
                    continue
                inferred = get_tv_map(self_annotation, receiver_for_match, ctx)
                if isinstance(inferred, CanAssignError):
                    continue
                translated = translate_generic_typevar_map(
                    self_annotation, inferred, ctx
                )
                if not translated:
                    translated = infer_positional_generic_typevar_map(
                        self_annotation, receiver_for_match, ctx
                    )
                for typevar, value in {**inferred, **translated}.items():
                    existing = tv_map.get(typevar)
                    if existing is None:
                        tv_map[typevar] = value
                    elif existing != value:
                        tv_map[typevar] = unite_values(existing, value)
            continue
        if not any(
            isinstance(subvalue, TypeVarValue)
            for subvalue in self_annotation.walk_values()
        ):
            continue
        inferred = get_tv_map(self_annotation, receiver_for_match, ctx)
        if isinstance(inferred, CanAssignError):
            continue
        translated = translate_generic_typevar_map(self_annotation, inferred, ctx)
        if not translated:
            translated = infer_positional_generic_typevar_map(
                self_annotation, receiver_for_match, ctx
            )
        for typevar, value in {**inferred, **translated}.items():
            existing = tv_map.get(typevar)
            if existing is None:
                tv_map[typevar] = value
            elif existing != value:
                tv_map[typevar] = unite_values(existing, value)
    return tv_map


def _should_use_permissive_dunder_hash(val: Value) -> bool:
    return _is_definitely_class_object_value(val)


def _is_definitely_class_object_value(value: Value) -> bool:
    """Return whether the value definitely represents a class object.

    For unions, all members must be class objects.
    For intersections, any class-object member is enough because the intersection
    value must satisfy all member constraints at once.
    """
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        return bool(value.vals) and all(
            _is_definitely_class_object_value(subval) for subval in value.vals
        )
    if isinstance(value, IntersectionValue):
        return bool(value.vals) and any(
            _is_definitely_class_object_value(subval) for subval in value.vals
        )
    if isinstance(value, KnownValue):
        return safe_isinstance(value.val, type)
    if isinstance(value, (SubclassValue, SyntheticClassObjectValue)):
        return True
    if isinstance(value, TypedValue):
        if isinstance(value.typ, type):
            return safe_issubclass(value.typ, type)
        if isinstance(value.typ, str):
            return value.typ in {"type", "builtins.type"}
        return False
    return False
