"""

An object that represents a type.

"""

import collections.abc
import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal
from unittest import mock

from typing_extensions import assert_never

if sys.version_info >= (3, 14):
    pass
else:
    pass

if TYPE_CHECKING:
    from .relations import Relation

from pycroscope.input_sig import (
    FullSignature,
    InputSigValue,
    coerce_paramspec_specialization_to_input_sig,
)
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
    is_instance_of_typing_name,
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
    ClassSymbol,
    DataclassFieldInfo,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    PredicateValue,
    PropertyInfo,
    SelfT,
    SelfTVV,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedValue,
    TypeFormValue,
    TypeParam,
    TypeVarLike,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    freshen_typevars_for_inference,
    get_tv_map,
    match_typevar_arguments,
    replace_fallback,
    stringify_object,
    tuple_members_from_value,
    unify_bounds_maps,
    unite_values,
)

EXCLUDED_PROTOCOL_MEMBERS: set[str] = {
    "__abstractmethods__",
    "__annotate__",
    "__annotate_func__",
    "__annotations__",
    "__annotations_cache__",
    "__dict__",
    "__doc__",
    "__init__",
    "__new__",
    "__module__",
    "__parameters__",
    "__slots__",
    "__subclasshook__",
    "__weakref__",
    "_abc_impl",
    "_abc_cache",
    "_is_protocol",
    "__next_in_mro__",
    "_abc_generic_negative_cache_version",
    "__orig_bases__",
    "__args__",
    "_abc_registry",
    "__extra__",
    "_abc_generic_negative_cache",
    "__origin__",
    "__tree_hash__",
    "_gorg",
    "_is_runtime_protocol",
    "__protocol_attrs__",
    "__callable_proto_members_only__",
    "__non_callable_proto_members__",
    "__static_attributes__",
    "__firstlineno__",
}


def runtime_type_generic_alias(typ: type) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


def class_keys_match(left: type | str, right: type | str) -> bool:
    if left == right:
        return True
    if isinstance(left, type) and isinstance(right, str):
        return runtime_type_generic_alias(left) == right
    if isinstance(left, str) and isinstance(right, type):
        return left == runtime_type_generic_alias(right)
    return False


def _as_concrete_signature(
    sig: Signature | BoundMethodSignature | OverloadedSignature | None,
    ctx: CanAssignContext,
) -> Signature | OverloadedSignature | None:
    if isinstance(sig, BoundMethodSignature):
        return sig.get_signature(ctx=ctx)
    return sig


@dataclass(frozen=True)
class TypeObjectAttribute:
    value: Value
    symbol: ClassSymbol
    owner: "TypeObject"
    is_property: bool
    property_has_setter: bool


def get_mro(typ: type) -> Sequence[type]:
    try:
        return inspect.getmro(typ)
    except AttributeError:
        # It's not actually a class.
        return []


MroValue = TypedValue | AnyValue


@dataclass(frozen=True)
class DataclassFieldRecord:
    field_name: str
    field_info: DataclassFieldInfo


@dataclass(kw_only=True)
class TypeObject:
    typ: type | str
    mro: tuple[MroValue, ...]
    """Types that we consider the type to inherit from for purposes of subtyping, but that
    are not actual bases."""
    base_classes: set[type | str] = field(default_factory=set)
    declared_type_params: tuple[TypeParam, ...] = field(
        default_factory=tuple, repr=False
    )
    is_final: bool = False
    is_protocol: bool = False
    protocol_members: set[str] = field(default_factory=set)
    declared_symbols: dict[str, ClassSymbol] = field(default_factory=dict, repr=False)
    dataclass_fields: tuple[DataclassFieldRecord, ...] = field(
        default_factory=tuple, repr=False
    )
    virtual_bases: list[MroValue] = field(default_factory=list)
    is_thrift_enum: bool = field(init=False)
    is_universally_assignable: bool = field(init=False)
    _protocol_positive_cache: dict[tuple[Value, Value], BoundsMap] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.typ, str):
            # Synthetic type
            self.is_universally_assignable = False
            self.is_thrift_enum = False
            return
        assert isinstance(self.typ, type), repr(self.typ)
        self.is_universally_assignable = issubclass(self.typ, mock.NonCallableMock)
        if safe_getattr(self.typ, "__final__", False):
            self.is_final = True
        self.is_thrift_enum = hasattr(self.typ, "_VALUES_TO_NAMES")
        self.base_classes |= set(get_mro(self.typ))
        if self.is_thrift_enum:
            self.base_classes.add(int)
            self.virtual_bases.append(TypedValue(int))

    def get_declared_symbol(self, name: str) -> ClassSymbol | None:
        return self.declared_symbols.get(name)

    def get_declared_symbols(self) -> dict[str, ClassSymbol]:
        return self.declared_symbols

    def get_attribute(
        self,
        name: str,
        ctx: CanAssignContext,
        *,
        on_class: bool,
        receiver_value: TypedValue | None = None,
    ) -> TypeObjectAttribute | None:
        match = self._get_declared_symbol_with_owner(name, ctx)
        if match is None:
            return None
        owner, declared_symbol = match
        symbol = _specialize_symbol_for_owner(
            self, owner, declared_symbol, ctx, receiver_value=receiver_value
        )
        value = _get_attribute_value_from_symbol(
            symbol, ctx, on_class=on_class, receiver_value=receiver_value
        )
        if (
            symbol.property_info is not None
            and on_class
            and (value is UNINITIALIZED_VALUE or not _is_property_marker_value(value))
        ):
            value = TypedValue(property)
        return TypeObjectAttribute(
            value=value,
            symbol=symbol,
            owner=owner,
            is_property=symbol.property_info is not None and not on_class,
            property_has_setter=(
                symbol.property_info.setter_type is not None
                if symbol.property_info is not None
                else False
            ),
        )

    def get_declared_symbol_from_mro(
        self, name: str, ctx: CanAssignContext
    ) -> ClassSymbol | None:
        match = self._get_declared_symbol_with_owner(name, ctx)
        return None if match is None else match[1]

    def _get_declared_symbol_with_owner(
        self, name: str, ctx: CanAssignContext
    ) -> tuple["TypeObject", ClassSymbol] | None:
        symbol = self.declared_symbols.get(name)
        if symbol is not None:
            return self, symbol
        for mro_value in self.mro:
            if isinstance(mro_value, AnyValue):
                return None
            type_obj = mro_value.get_type_object(ctx)
            symbol = type_obj.declared_symbols.get(name)
            if symbol is not None:
                return type_obj, symbol
        return None

    def is_assignable_to_type(self, typ: type) -> bool:
        for base in self.base_classes:
            if isinstance(base, str):
                continue
            else:
                if safe_issubclass(base, typ):
                    return True
        return self.is_universally_assignable

    def is_assignable_to_type_object(self, other: "TypeObject") -> bool:
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
        *,
        relation: "Literal[Relation.SUBTYPE, Relation.ASSIGNABLE] | None" = None,
    ) -> CanAssign:
        from .relations import Relation, _compare_tuple_sequences

        if relation is None:
            relation = Relation.ASSIGNABLE
        other_basic = replace_fallback(other_val)
        if not isinstance(other_basic, (KnownValue, TypedValue, SubclassValue)):
            return CanAssignError(f"Cannot assign {other_val} to {self}")
        if self.typ is tuple:
            self_tuple_members = tuple_members_from_value(self_val, ctx)
            other_tuple_members = tuple_members_from_value(other_val, ctx)
            if self_tuple_members is not None and other_tuple_members is not None:
                return _compare_tuple_sequences(
                    SequenceValue(tuple, self_tuple_members),
                    SequenceValue(tuple, other_tuple_members),
                    relation,
                    ctx,
                )
        other = other_basic.get_type_object(ctx)
        if other.is_universally_assignable:
            return {}
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
                if isinstance(result, CanAssignError) and other.virtual_bases:
                    for base in other.virtual_bases:
                        assert isinstance(base, TypedValue), (self, base)
                        subresult = self._is_compatible_with_protocol(
                            self_val, base, ctx
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
            if (
                not isinstance(result, CanAssignError)
                and isinstance(self.typ, type)
                and isinstance(other.typ, type)
                and not other.is_protocol
                and not isinstance(other_basic, SubclassValue)
            ):
                ctx.record_protocol_implementation(self.typ, other.typ)
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
        assert isinstance(
            other_basic, (KnownValue, TypedValue, SubclassValue)
        ), other_basic
        other_type_obj = other_basic.get_type_object(ctx)
        other_type_key = _receiver_key_from_value(other_val)

        bounds_maps = []
        if len(self.protocol_members) > 1:
            protocol_self_typevar_map = _collect_protocol_self_typevar_map(
                self, self.protocol_members, other_val, ctx
            )
            actual_self_typevar_map = _collect_protocol_self_typevar_map(
                other_type_obj, self.protocol_members, other_val, ctx
            )
        else:
            protocol_self_typevar_map = {}
            actual_self_typevar_map = {}
        apply_synthetic_member_rules = (
            isinstance(self.typ, str)
            and _get_synthetic_class_for_key(self.typ, ctx) is not None
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
                    symbol = self.get_declared_symbol_from_mro(member, ctx)
                    if symbol is not None:
                        expected = symbol.get_effective_type()
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
                    actual = _get_protocol_call_member_initializer(
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
                if not class_object_check:
                    specialized_expected, _ = _specialize_declared_property_value(
                        self, member, expected, ctx
                    )
                    if specialized_expected is not None:
                        expected = specialized_expected
                if protocol_self_typevar_map:
                    expected = expected.substitute_typevars(protocol_self_typevar_map)
                expected = expected.substitute_typevars({SelfT: other_val})
                actual = ctx.get_attribute_from_value(other_val, member)
                if (
                    not class_object_check
                    and actual is not UNINITIALIZED_VALUE
                    and other_type_obj is not None
                ):
                    specialized_actual, _ = _specialize_declared_property_value(
                        other_type_obj, member, actual, ctx
                    )
                    if specialized_actual is not None:
                        actual = specialized_actual
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
                    expected = _normalize_protocol_initializer_for_relation(
                        expected, ctx, receiver=self_val
                    )
                if _is_callable_member_value(actual, ctx):
                    actual = _normalize_protocol_initializer_for_relation(
                        actual, ctx, receiver=other_val
                    )
                if not use_descriptor_rules:
                    can_assign = has_relation(
                        expected, actual, Relation.ASSIGNABLE, ctx
                    )
                else:
                    actual_member_tobj = (
                        ctx.make_type_object(other_type_key)
                        if class_object_check and other_type_key is not None
                        else other_type_obj
                    )
                    expected_access = _resolve_member_access(
                        self,
                        member=member,
                        resolved_value=expected,
                        ctx=ctx,
                        class_object_access=False,
                    )
                    actual_access = _resolve_member_access(
                        actual_member_tobj,
                        member=member,
                        resolved_value=actual,
                        ctx=ctx,
                        class_object_access=class_object_check,
                    )
                    if class_object_check and expected_access.symbol.is_classvar:
                        can_assign = CanAssignError(
                            f"Protocol member {member!r} is a ClassVar"
                        )
                    elif (
                        class_object_check
                        and not expected_access.symbol.is_classvar
                        and not expected_access.symbol.is_method
                        and not expected_access.is_property
                        and not _is_callable_member_value(expected_access.value, ctx)
                        and other_type_key is not None
                        and not actual_access.symbol.is_classvar
                        and not _is_member_from_metaclass(other_type_key, member, ctx)
                    ):
                        can_assign = CanAssignError(
                            f"Protocol member {member!r} is an instance attribute"
                        )
                    elif (
                        not class_object_check
                        and not is_dunder_member
                        and expected_access.symbol.is_classvar
                        != actual_access.symbol.is_classvar
                    ):
                        can_assign = CanAssignError(
                            f"ClassVar status of protocol member {member!r} conflicts"
                        )
                    else:
                        expected_for_relation = expected_access.value
                        actual_for_relation = actual_access.value
                        expected_is_writable_data_member = (
                            not expected_access.symbol.is_method
                            and not expected_access.is_property
                            and not expected_access.symbol.is_classvar
                            and not expected_access.symbol.is_readonly
                            and not _is_callable_member_value(
                                expected_access.value, ctx
                            )
                        )
                        expected_requires_writable = (
                            expected_access.is_property
                            and expected_access.property_has_setter
                        ) or expected_is_writable_data_member
                        if class_object_check:
                            expected_requires_writable = False
                        if expected_requires_writable and not _is_writable_member(
                            actual_access, tobj=other_type_obj, ctx=ctx
                        ):
                            can_assign = CanAssignError(
                                f"Protocol member {member!r} is not writable"
                            )
                        else:
                            expected_for_relation = freshen_typevars_for_inference(
                                expected_for_relation
                            )
                            if _is_callable_member_value(expected_for_relation, ctx):
                                expected_for_relation = (
                                    _normalize_protocol_initializer_for_relation(
                                        expected_for_relation, ctx, receiver=self_val
                                    )
                                )
                            if _is_callable_member_value(actual_for_relation, ctx):
                                actual_for_relation = (
                                    _normalize_protocol_initializer_for_relation(
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


def _prefer_existing_symbol_type(existing: Value, new: Value) -> bool:
    return isinstance(new, AnyValue) and new.source is AnySource.inference


def _merge_symbol_initializer(
    existing: Value | None, new: Value | None
) -> Value | None:
    if new is None:
        return existing
    if existing is None:
        return new
    if _prefer_existing_symbol_type(existing, new):
        return existing
    return new


def _merge_property_info(
    existing: PropertyInfo | None, new: PropertyInfo | None
) -> PropertyInfo | None:
    if new is None:
        return existing
    if existing is None:
        return new
    return PropertyInfo(
        (
            existing.getter_type
            if _prefer_existing_symbol_type(existing.getter_type, new.getter_type)
            else new.getter_type
        ),
        setter_type=_merge_symbol_initializer(existing.setter_type, new.setter_type),
        getter_deprecation=new.getter_deprecation or existing.getter_deprecation,
        setter_deprecation=new.setter_deprecation or existing.setter_deprecation,
    )


def merge_declared_symbol(
    existing: ClassSymbol | None, new: ClassSymbol
) -> ClassSymbol:
    if existing is None:
        return new
    return ClassSymbol(
        annotation=_merge_symbol_initializer(existing.annotation, new.annotation),
        qualifiers=existing.qualifiers | new.qualifiers,
        is_instance_only=existing.is_instance_only or new.is_instance_only,
        is_method=existing.is_method or new.is_method,
        is_classmethod=existing.is_classmethod or new.is_classmethod,
        is_staticmethod=existing.is_staticmethod or new.is_staticmethod,
        returns_self_on_class_access=(
            existing.returns_self_on_class_access or new.returns_self_on_class_access
        ),
        property_info=_merge_property_info(existing.property_info, new.property_info),
        initializer=_merge_symbol_initializer(existing.initializer, new.initializer),
        dataclass_field=(
            new.dataclass_field
            if new.dataclass_field is not None
            else existing.dataclass_field
        ),
    )


def _is_writable_member(
    resolved_access: TypeObjectAttribute, tobj: TypeObject, ctx: CanAssignContext
) -> bool:
    if resolved_access.symbol.is_classvar or resolved_access.symbol.is_method:
        return False
    if resolved_access.is_property:
        return resolved_access.property_has_setter
    return not resolved_access.symbol.is_readonly and not _is_frozen_dataclass(
        tobj, ctx
    )


def _resolve_member_access(
    tobj: TypeObject,
    *,
    member: str,
    resolved_value: Value,
    ctx: CanAssignContext,
    class_object_access: bool,
) -> TypeObjectAttribute:
    access = tobj.get_attribute(member, ctx, on_class=class_object_access)
    if access is None:
        return TypeObjectAttribute(
            value=resolved_value,
            symbol=ClassSymbol(initializer=resolved_value),
            owner=tobj,
            is_property=False,
            property_has_setter=False,
        )
    symbol = access.symbol
    owner = access.owner
    if class_object_access:
        property_getter, property_has_setter = None, False
    elif symbol.property_info is not None:
        property_has_setter = access.property_has_setter
        if resolved_value is UNINITIALIZED_VALUE or _is_property_marker_value(
            resolved_value
        ):
            property_getter = access.value
        else:
            property_getter = _substitute_symbol_value(
                resolved_value,
                _get_symbol_owner_substitutions_from_type_objects(tobj, owner, ctx),
            )
    else:
        property_getter, property_has_setter = None, False
    is_property = property_getter is not None
    if class_object_access and symbol.is_property:
        is_property = False
    if (
        not class_object_access
        and not symbol.is_classvar
        and not is_property
        and not symbol.is_method
        and symbol.annotation is not None
    ):
        value = symbol.annotation
    else:
        value = property_getter if property_getter is not None else resolved_value
    if (
        class_object_access
        and symbol.is_property
        and not _is_property_marker_value(value)
    ):
        value = TypedValue(property)
    return TypeObjectAttribute(
        value=value,
        symbol=symbol,
        owner=owner,
        is_property=is_property,
        property_has_setter=property_has_setter,
    )


def is_readonly_annotated_member(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    match = lookup_declared_symbol_with_owner(class_key, member, ctx)
    return match is not None and match[1].is_readonly


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
    if isinstance(value, TypedValue):
        return [value.typ]
    return []


def _checker_ctx(ctx: object) -> object:
    return safe_getattr(ctx, "checker", ctx)


def normalize_synthetic_descriptor_attribute(
    value: Value,
    *,
    is_self_returning_classmethod: bool = False,
    unknown_descriptor_means_any: bool = True,
) -> Value:
    if isinstance(value, GenericValue) and value.typ is staticmethod:
        if not value.args:
            if unknown_descriptor_means_any:
                return AnyValue(AnySource.inference)
            return value
        wrapped = next(iter(value.args))
        if isinstance(wrapped, InputSigValue):
            if isinstance(wrapped.input_sig, FullSignature):
                return_annotation = (
                    value.args[1]
                    if len(value.args) > 1
                    else wrapped.input_sig.sig.return_value
                )
                return CallableValue(
                    replace(wrapped.input_sig.sig, return_value=return_annotation)
                )
            return AnyValue(AnySource.inference)
        return wrapped
    if isinstance(value, GenericValue) and value.typ is classmethod:
        if not value.args:
            if unknown_descriptor_means_any:
                return AnyValue(AnySource.inference)
            return value
        wrapped = value.args[1] if len(value.args) >= 2 else value.args[0]
        if isinstance(wrapped, InputSigValue):
            if isinstance(wrapped.input_sig, FullSignature):
                return_annotation = (
                    value.args[2]
                    if len(value.args) > 2
                    else wrapped.input_sig.sig.return_value
                )
                if (
                    is_self_returning_classmethod
                    and isinstance(return_annotation, AnyValue)
                    and return_annotation.source is AnySource.generic_argument
                ):
                    return_annotation = SelfTVV
                return CallableValue(
                    replace(wrapped.input_sig.sig, return_value=return_annotation)
                )
            return AnyValue(AnySource.inference)
        if isinstance(wrapped, CallableValue):
            return_annotation = (
                value.args[2] if len(value.args) > 2 else wrapped.signature.return_value
            )
            if (
                is_self_returning_classmethod
                and isinstance(return_annotation, AnyValue)
                and return_annotation.source is AnySource.generic_argument
            ):
                return_annotation = SelfTVV
            return CallableValue(
                replace(wrapped.signature, return_value=return_annotation)
            )
        return wrapped
    if isinstance(value, KnownValue) and isinstance(value.val, staticmethod):
        return KnownValue(value.val.__func__)
    if isinstance(value, KnownValue) and isinstance(value.val, classmethod):
        return KnownValue(value.val.__func__)
    return value


def _class_key_and_generic_args_from_type_value(
    receiver_value: TypedValue,
) -> tuple[type | str, Sequence[Value]]:
    generic_args = (
        receiver_value.args if isinstance(receiver_value, GenericValue) else ()
    )
    return receiver_value.typ, generic_args


def _typevar_map_from_generic_args(
    type_params: Sequence[TypeParam], generic_args: Sequence[Value]
) -> dict[TypeVarLike, Value]:
    if not type_params:
        return {}
    substitutions: dict[TypeVarLike, Value] = {}
    matched = match_typevar_arguments(type_params, generic_args)
    if matched is None:
        return substitutions
    for typevar, value in matched:
        if is_instance_of_typing_name(typevar, "ParamSpec"):
            value = coerce_paramspec_specialization_to_input_sig(value)
        substitutions[typevar] = value.substitute_typevars(substitutions)
    return substitutions


def _typevar_map_from_type_value(
    receiver_value: TypedValue, type_params: Sequence[TypeParam]
) -> dict[TypeVarLike, Value]:
    _, generic_args = _class_key_and_generic_args_from_type_value(receiver_value)
    return _typevar_map_from_generic_args(type_params, generic_args)


def _specialize_symbol_for_owner(
    receiver_tobj: TypeObject,
    owner_tobj: TypeObject,
    symbol: ClassSymbol,
    ctx: CanAssignContext,
    *,
    receiver_value: TypedValue | None = None,
) -> ClassSymbol:
    substitutions = _get_symbol_owner_substitutions_from_type_objects(
        receiver_tobj, owner_tobj, ctx, receiver_value=receiver_value
    )
    return replace(
        symbol,
        annotation=(
            _substitute_symbol_value(symbol.annotation, substitutions)
            if symbol.annotation is not None
            else None
        ),
        property_info=(
            symbol.property_info.substitute_typevars(substitutions)
            if symbol.property_info is not None
            else None
        ),
        initializer=(
            _substitute_symbol_value(symbol.initializer, substitutions)
            if symbol.initializer is not None
            else None
        ),
    )


def _bind_attribute_signature(
    value: Value, *, receiver_value: Value, ctx: CanAssignContext
) -> Value:
    signature = ctx.signature_from_value(value)
    if isinstance(signature, BoundMethodSignature):
        bound = signature.get_signature(ctx=ctx)
        if bound is not None:
            return CallableValue(bound)
        return value
    if isinstance(signature, (Signature, OverloadedSignature)):
        bound = signature.bind_self(self_value=receiver_value, ctx=ctx)
        if bound is not None:
            return CallableValue(bound)
    return value


def _specialize_self_returning_classmethod(
    raw_attr: Value,
    normalized_attr: Value,
    *,
    receiver_value: TypedValue | None,
    ctx: CanAssignContext,
) -> Value:
    if receiver_value is None or not isinstance(normalized_attr, CallableValue):
        return normalized_attr
    raw_attr = replace_fallback(raw_attr)
    if not (
        isinstance(raw_attr, GenericValue)
        and raw_attr.typ is classmethod
        and raw_attr.args
    ):
        return normalized_attr
    receiver_for_self: TypedValue | TypeVarValue
    if isinstance(receiver_value, GenericValue):
        receiver_for_self = TypedValue(receiver_value.typ)
    else:
        receiver_for_self = receiver_value
    inferred = get_tv_map(raw_attr.args[0], SubclassValue(receiver_for_self), ctx)
    if isinstance(inferred, CanAssignError):
        return normalized_attr
    inferred = {**inferred, SelfT: receiver_for_self}
    return CallableValue(normalized_attr.signature.substitute_typevars(inferred))


def _classmethod_receiver_value_from_type_value(
    receiver_value: TypedValue,
) -> SubclassValue:
    class_key, _ = _class_key_and_generic_args_from_type_value(receiver_value)
    return SubclassValue(TypedValue(class_key))


def _get_attribute_value_from_symbol(
    symbol: ClassSymbol,
    ctx: CanAssignContext,
    *,
    on_class: bool,
    receiver_value: TypedValue | None,
) -> Value:
    if symbol.property_info is not None:
        if on_class:
            return UNINITIALIZED_VALUE
        return symbol.property_info.getter_type
    declared_value = symbol.get_effective_type()
    raw_value = symbol.initializer if symbol.initializer is not None else declared_value
    raw_value = normalize_synthetic_descriptor_attribute(
        raw_value,
        is_self_returning_classmethod=symbol.returns_self_on_class_access,
        unknown_descriptor_means_any=False,
    )
    if symbol.is_classmethod:
        raw_value = _specialize_self_returning_classmethod(
            symbol.initializer if symbol.initializer is not None else raw_value,
            raw_value,
            receiver_value=receiver_value,
            ctx=ctx,
        )
        if receiver_value is None:
            return raw_value
        return _bind_attribute_signature(
            raw_value,
            receiver_value=_classmethod_receiver_value_from_type_value(receiver_value),
            ctx=ctx,
        )
    if on_class:
        if not symbol.is_method:
            return declared_value
        return raw_value
    if not symbol.is_classvar and not symbol.is_method and symbol.initializer is None:
        return declared_value
    if symbol.is_method and not symbol.is_staticmethod and not symbol.is_classmethod:
        if receiver_value is None:
            return raw_value
        return _bind_attribute_signature(
            raw_value, receiver_value=receiver_value, ctx=ctx
        )
    if (
        not symbol.is_classvar
        and not symbol.is_method
        and symbol.initializer is not None
        and symbol.annotation is not None
    ):
        return declared_value
    return raw_value


def _substitute_symbol_value(
    value: Value, substitutions: dict[TypeVarLike, Value]
) -> Value:
    if not substitutions:
        return value
    return value.substitute_typevars(substitutions)


def _mro_generic_args(value: MroValue) -> Sequence[Value]:
    if isinstance(value, SequenceValue) and value.typ is tuple:
        return (value,)
    if isinstance(value, GenericValue):
        return value.args
    return ()


def _get_symbol_owner_substitutions_from_type_objects(
    receiver_tobj: TypeObject,
    owner_tobj: TypeObject,
    ctx: CanAssignContext,
    *,
    receiver_value: TypedValue | None = None,
) -> dict[TypeVarLike, Value]:
    receiver_substitutions: dict[TypeVarLike, Value] = {}
    if receiver_value is not None:
        receiver_substitutions = _typevar_map_from_type_value(
            receiver_value, receiver_tobj.declared_type_params
        )
    if owner_tobj is receiver_tobj:
        return receiver_substitutions
    owner_value = next(
        (
            mro_value
            for mro_value in receiver_tobj.mro
            if isinstance(mro_value, TypedValue)
            and mro_value.get_type_object(ctx) is owner_tobj
        ),
        None,
    )
    if owner_value is None:
        return {}
    if receiver_substitutions:
        owner_value = owner_value.substitute_typevars(receiver_substitutions)
    owner_substitutions = _typevar_map_from_generic_args(
        owner_tobj.declared_type_params, _mro_generic_args(owner_value)
    )
    if not owner_substitutions:
        return {}
    if not receiver_substitutions:
        return owner_substitutions
    return {**receiver_substitutions, **owner_substitutions}


def _make_type_object_for_key(class_key: type | str, ctx: object) -> TypeObject | None:
    make_type_object = safe_getattr(ctx, "make_type_object", None)
    if callable(make_type_object):
        return make_type_object(class_key)
    checker = _checker_ctx(ctx)
    make_type_object = safe_getattr(checker, "make_type_object", None)
    if callable(make_type_object):
        return make_type_object(class_key)
    return None


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


def lookup_declared_symbol_with_owner(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> tuple[type | str, ClassSymbol] | None:
    type_object = _make_type_object_for_key(class_key, ctx)
    if type_object is None:
        return None
    match = type_object._get_declared_symbol_with_owner(member, ctx)
    if match is None:
        return None
    owner_tobj, symbol = match
    return owner_tobj.typ, symbol


def _is_member_defined_on_class_key(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    match = lookup_declared_symbol_with_owner(class_key, member, ctx)
    return match is not None and not match[1].is_initvar


def _is_member_method(
    class_key: type | str, member: str, ctx: CanAssignContext
) -> bool:
    match = lookup_declared_symbol_with_owner(class_key, member, ctx)
    return match is not None and match[1].is_method


def _is_property_marker_value(value: Value) -> bool:
    value = replace_fallback(value)
    return (
        isinstance(value, KnownValue)
        and isinstance(value.val, property)
        or isinstance(value, TypedValue)
        and value.typ is property
    )


def _specialize_declared_property_value(
    receiver_tobj: TypeObject | None, member: str, value: Value, ctx: CanAssignContext
) -> tuple[Value | None, bool]:
    if receiver_tobj is None:
        return None, False
    match = receiver_tobj._get_declared_symbol_with_owner(member, ctx)
    if match is None:
        return None, False
    owner_tobj, symbol = match
    if symbol.property_info is None:
        return None, False
    substitutions = _get_symbol_owner_substitutions_from_type_objects(
        receiver_tobj, owner_tobj, ctx
    )
    if value is not UNINITIALIZED_VALUE and not _is_property_marker_value(value):
        return (
            _substitute_symbol_value(value, substitutions),
            symbol.property_info.setter_type is not None,
        )
    return (
        _substitute_symbol_value(symbol.property_info.getter_type, substitutions),
        symbol.property_info.setter_type is not None,
    )


def _is_frozen_dataclass(tobj: TypeObject, ctx: CanAssignContext) -> bool:
    if isinstance(tobj.typ, type):
        dataclass_params = safe_getattr(tobj.typ, "__dataclass_params__", None)
        if safe_getattr(dataclass_params, "frozen", None) is True:
            return True
    elif isinstance(tobj.typ, str):
        synthetic = _get_synthetic_class_for_key(tobj.typ, ctx)
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


def _normalize_protocol_initializer_for_relation(
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
        metaclass = synthetic.metaclass
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
    return _is_member_defined_on_class_key(metaclass_key, member, ctx)


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
    if not _is_member_method(class_key, member, ctx):
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


def _get_protocol_call_member_initializer(
    protocol_typ: type | str, self_value: Value, ctx: CanAssignContext
) -> Value:
    call_member = UNINITIALIZED_VALUE
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
    tobj: TypeObject,
    protocol_members: set[str],
    receiver_value: Value,
    ctx: CanAssignContext,
) -> dict[TypeVarLike, Value]:
    """Collect typevar substitutions implied by receiver annotations.

    This propagates `self: T` constraints across protocol members.
    """
    tv_map: dict[TypeVarLike, Value] = {}
    for member in protocol_members:
        receiver_for_match = receiver_value
        if isinstance(tobj.typ, str):
            symbol = tobj.declared_symbols.get(member)
            if symbol is None:
                raw_attr = UNINITIALIZED_VALUE
            elif symbol.is_property:
                raw_attr = symbol.get_effective_type()
            elif symbol.initializer is not None:
                raw_attr = symbol.initializer
            else:
                raw_attr = UNINITIALIZED_VALUE
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
            descriptor = inspect.getattr_static(tobj.typ, member, UNINITIALIZED_VALUE)
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
