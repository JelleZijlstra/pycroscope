"""

Code for retrieving the value of attributes.

"""

import enum
import inspect
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, get_origin

import typing_extensions
from typing_extensions import assert_never

from pycroscope.predicates import HasAttr

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotations
else:
    from inspect import get_annotations  # pragma: no cover

from . import dataclass as dataclass_helpers
from .annotated_types import EnumName
from .annotations import (
    _RuntimeAnnotationsContext,
    annotation_expr_from_annotations,
    type_from_runtime,
    type_from_value,
)
from .input_sig import coerce_paramspec_specialization_to_input_sig
from .options import Options, PyObjectSequenceOption
from .relations import Relation, has_relation, subtract_values
from .safe import (
    is_async_fn,
    is_bound_classmethod,
    is_instance_of_typing_name,
    is_typing_name,
    safe_isinstance,
    safe_issubclass,
)
from .signature import (
    BoundMethodSignature,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
)
from .stacked_scopes import Composite
from .type_object import (
    AttributePolicy,
    MroValue,
    TypeObject,
    TypeObjectAttribute,
    _class_key_from_value,
    _get_attribute_value_from_symbol,
    _get_cached_property_return_type,
    _is_property_marker_value,
    _specialize_symbol_for_owner,
    class_keys_match,
    normalize_synthetic_descriptor_attribute,
)
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    ClassSymbol,
    CustomCheckExtension,
    GenericBases,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    Qualifier,
    SelfT,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SuperValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    _iter_typevar_map_items,
    _typevar_map_from_varlike_pairs,
    annotate_value,
    receiver_to_self_type,
    replace_fallback,
    set_self,
    shield_nested_self_typevars,
    stringify_object,
    unite_values,
)

# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)
NoneType = type(None)
_ENUM_INSTANCE_DESCRIPTOR_TYPES = tuple(
    descriptor_type
    for descriptor_type in (
        getattr(enum, "property", None),
        types.DynamicClassAttribute,
    )
    if descriptor_type is not None
)
if sys.version_info >= (3, 12):
    import typing as _typing

    RuntimeTypeAliasType = _typing.TypeAliasType | typing_extensions.TypeAliasType
else:
    RuntimeTypeAliasType = typing_extensions.TypeAliasType  # pragma: no cover


@dataclass
class AttrContext:
    root_composite: Composite
    # Optional approximation of root_composite.value used only for attribute lookup.
    # This lets us fall back to a bound/constraint view for dispatch while still
    # retaining the original root value for self-binding and metadata checks.
    lookup_root_value: Value | None
    attr: str
    options: Options = field(repr=False)
    skip_mro: bool
    skip_unwrap: bool
    prefer_typeshed: bool

    @property
    def root_value(self) -> Value:
        """The original value of the attribute receiver expression."""
        return self.root_composite.value

    def get_self_value(self) -> Value:
        return self.root_value

    def record_usage(self, obj: Any, val: Value) -> None:
        pass

    def record_attr_read(self, obj: Any) -> None:
        pass

    def get_property_type_from_argspec(self, obj: property) -> Value:
        raise NotImplementedError

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        raise NotImplementedError

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        raise NotImplementedError

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> tuple[Value, object]:
        raise NotImplementedError

    def should_ignore_none_attributes(self) -> bool:
        raise NotImplementedError

    def get_signature(self, obj: object) -> MaybeSignature:
        raise NotImplementedError

    def signature_from_value(self, value: Value) -> MaybeSignature:
        raise NotImplementedError

    def get_can_assign_context(self) -> CanAssignContext:
        raise NotImplementedError

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value]
    ) -> GenericBases:
        raise NotImplementedError

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        raise NotImplementedError

    def get_bound_self_type(self) -> Value | None:
        raise NotImplementedError

    def clone_for_attribute_lookup(
        self, root_composite: Composite, attr: str
    ) -> "AttrContext":
        return replace(
            self,
            root_composite=root_composite,
            attr=attr,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=False,
        )

    def get_type_object_attribute_policy(
        self, *, on_class: bool, receiver_value: Value | None
    ) -> AttributePolicy:
        return AttributePolicy(on_class=on_class, receiver_value=receiver_value)


def _get_type_object_attribute(
    type_object: TypeObject,
    attr_name: str,
    ctx: AttrContext,
    *,
    on_class: bool,
    receiver_value: Value | None,
) -> TypeObjectAttribute | None:
    return type_object.get_attribute(
        attr_name,
        ctx.get_type_object_attribute_policy(
            on_class=on_class, receiver_value=receiver_value
        ),
    )


def get_attribute(ctx: AttrContext) -> Value:
    if (
        isinstance(ctx.root_value, TypeAliasValue)
        and ctx.root_value.uses_type_alias_object_semantics
    ):
        return _get_attribute_from_type_alias(ctx.root_value, ctx)
    lookup_root_value = (
        ctx.root_value if ctx.lookup_root_value is None else ctx.lookup_root_value
    )
    if (
        isinstance(lookup_root_value, TypeVarValue)
        and lookup_root_value.typevar_param.bound is not None
    ):
        class_key = _class_key_from_value(lookup_root_value.typevar_param.bound)
        if class_key is not None:
            can_assign_ctx = ctx.get_can_assign_context()
            type_object = can_assign_ctx.make_type_object(class_key)
            symbol = type_object.get_declared_symbol_from_mro(ctx.attr, can_assign_ctx)
            if symbol is not None:
                uses_self = (
                    symbol.annotation is not None
                    and _contains_self_typevar(symbol.annotation)
                ) or (
                    symbol.initializer is not None
                    and _contains_self_typevar(symbol.initializer)
                )
                if uses_self or symbol.is_classmethod:
                    attribute = _get_type_object_attribute(
                        type_object,
                        ctx.attr,
                        ctx,
                        on_class=False,
                        receiver_value=lookup_root_value,
                    )
                    if attribute is not None:
                        return attribute.value
    if (
        isinstance(ctx.root_value, SubclassValue)
        and isinstance(ctx.root_value.typ, TypeVarValue)
        and ctx.root_value.typ.typevar_param.bound is not None
    ):
        class_key = _class_key_from_value(ctx.root_value.typ.typevar_param.bound)
        if class_key is not None:
            symbol = (
                ctx.get_can_assign_context()
                .make_type_object(class_key)
                .get_declared_symbol_from_mro(ctx.attr, ctx.get_can_assign_context())
            )
            if (
                symbol is not None
                and not symbol.is_method
                and symbol.property_info is None
                and symbol.annotation is not None
                and _contains_self_typevar(symbol.annotation)
            ):
                return set_self(symbol.annotation, ctx.root_value.typ)

    original_lookup_root_value = lookup_root_value
    lookup_root_value = _maybe_specialize_class_partial_root(lookup_root_value, ctx)
    if lookup_root_value != original_lookup_root_value:
        ctx = replace(ctx, lookup_root_value=lookup_root_value)
    super_value = _extract_super_value(lookup_root_value)
    if super_value is not None:
        attribute_value = _get_attribute_from_super_value(super_value, ctx)
        if (
            (
                isinstance(attribute_value, AnyValue)
                or attribute_value is UNINITIALIZED_VALUE
            )
            and isinstance(ctx.root_value, PredicateValue)
            and isinstance(ctx.root_value.predicate, HasAttr)
            and ctx.root_value.predicate.attr == ctx.attr
        ):
            return ctx.root_value.predicate.value
        return attribute_value
    root_value = replace_fallback(lookup_root_value)
    if isinstance(root_value, KnownValue) and is_typing_name(
        type(root_value.val), "TypeAliasType"
    ):
        return _get_attribute_from_runtime_type_alias(root_value.val, ctx)
    attribute_value: Value = UNINITIALIZED_VALUE
    if isinstance(root_value, KnownValue):
        attribute_value = _get_attribute_from_known(root_value.val, ctx)
    elif isinstance(root_value, TypedValue):
        if (
            isinstance(root_value, CallableValue)
            and ctx.attr == "asynq"
            and root_value.signature.is_asynq
        ):
            return root_value.get_asynq_value()
        if isinstance(root_value, SequenceValue):
            exact_namedtuple_member = _get_namedtuple_member_from_sequence_value(
                root_value, ctx
            )
            if exact_namedtuple_member is not None:
                return exact_namedtuple_member
        if isinstance(root_value, GenericValue):
            args = root_value.args
        else:
            args = ()
        if isinstance(root_value.typ, str):
            attribute_value = _get_attribute_from_synthetic_typed_value(root_value, ctx)
        else:
            attribute_value = _get_attribute_from_typed(root_value.typ, args, ctx)
    elif isinstance(root_value, SubclassValue):
        synthetic_name: str | None = None
        if isinstance(root_value.typ, TypedValue):
            if isinstance(root_value.typ.typ, str):
                synthetic_name = root_value.typ.typ
            else:
                attribute_value = _get_attribute_from_subclass(
                    root_value.typ.typ, root_value.typ, ctx
                )
        elif isinstance(root_value.typ, TypeVarValue):
            if root_value.typ.typevar_param.bound is not None:
                bound = replace_fallback(root_value.typ.typevar_param.bound)
                if isinstance(bound, TypedValue) and isinstance(bound.typ, str):
                    synthetic_name = bound.typ
        else:
            assert_never(root_value.typ)
        if synthetic_name is not None:
            can_assign_ctx = ctx.get_can_assign_context()
            type_object = can_assign_ctx.make_type_object(synthetic_name)
            attribute = _get_type_object_attribute(
                type_object, ctx.attr, ctx, on_class=True, receiver_value=root_value.typ
            )
            if attribute is not None and _should_use_resolved_class_attribute(
                attribute
            ):
                self_value: Value = root_value.typ
                bound_self_type = _get_bound_self_type_from_ctx(ctx)
                if bound_self_type is not None and (
                    attribute.symbol.returns_self_on_class_access
                    or _contains_self_typevar(attribute.value)
                ):
                    self_value = bound_self_type
                attribute_value = _rebind_resolved_lookup_value(
                    attribute.value,
                    lookup_receiver=root_value.typ,
                    self_value=self_value,
                )
                return attribute_value
            synthetic_class = ctx.get_synthetic_class(synthetic_name)
            if synthetic_class is not None:
                attribute_value = _get_attribute_from_synthetic_class_inner(
                    synthetic_name, synthetic_class, ctx, seen={id(synthetic_class)}
                )
                if attribute_value is UNINITIALIZED_VALUE:
                    tobj = ctx.get_can_assign_context().make_type_object(synthetic_name)
                    if tobj.has_any_base():
                        attribute_value = AnyValue(AnySource.from_another)
                else:
                    self_value = root_value.typ
                    bound_self_type = _get_bound_self_type_from_ctx(ctx)
                    symbol = (
                        ctx.get_can_assign_context()
                        .make_type_object(synthetic_name)
                        .get_declared_symbol_from_mro(
                            ctx.attr, ctx.get_can_assign_context()
                        )
                    )
                    if (
                        bound_self_type is not None
                        and symbol is not None
                        and not symbol.is_method
                        and symbol.annotation is not None
                        and _contains_self_typevar(symbol.annotation)
                    ):
                        attribute_value = symbol.annotation
                        self_value = bound_self_type
                    elif bound_self_type is not None and (
                        _is_synthetic_self_classmethod_attribute(
                            synthetic_class, ctx.attr, ctx
                        )
                        or _contains_self_typevar(attribute_value)
                    ):
                        self_value = bound_self_type
                    attribute_value = set_self(attribute_value, self_value)
            else:
                attribute_value = AnyValue(AnySource.inference)
    elif isinstance(root_value, UnboundMethodValue):
        attribute_value = _get_attribute_from_unbound(root_value, ctx)
    elif isinstance(root_value, AnyValue):
        attribute_value = AnyValue(AnySource.from_another)
    elif isinstance(root_value, MultiValuedValue):
        raise TypeError("caller should unwrap MultiValuedValue")
    elif isinstance(root_value, IntersectionValue):
        raise TypeError("caller should unwrap IntersectionValue")
    elif isinstance(root_value, SyntheticModuleValue):
        module = ".".join(root_value.module_path)
        attribute_value = ctx.resolve_name_from_typeshed(module, ctx.attr)
    elif isinstance(root_value, TypeFormValue):
        attribute_value = _get_attribute_from_typed(object, (), ctx)
    elif isinstance(root_value, PredicateValue):
        if isinstance(root_value.predicate, HasAttr):
            if root_value.predicate.attr == ctx.attr:
                attribute_value = root_value.predicate.value
            else:
                attribute_value = UNINITIALIZED_VALUE
        else:
            attribute_value = _get_attribute_from_typed(object, (), ctx)
    elif isinstance(root_value, SyntheticClassObjectValue):
        if isinstance(root_value.class_type, TypedDictValue):
            attribute_value = _get_attribute_from_subclass(dict, root_value, ctx)
        elif isinstance(root_value.class_type.typ, str):
            attribute_value = _get_attribute_from_synthetic_class(
                root_value.class_type.typ, root_value, ctx
            )
        else:
            attribute_value = _get_attribute_from_synthetic_class(
                stringify_object(root_value.class_type.typ),
                root_value,
                ctx,
                runtime_type=root_value.class_type.typ,
            )
    else:
        assert_never(root_value)
    if (
        (
            isinstance(attribute_value, AnyValue)
            or attribute_value is UNINITIALIZED_VALUE
        )
        and isinstance(ctx.root_value, PredicateValue)
        and isinstance(ctx.root_value.predicate, HasAttr)
        and ctx.root_value.predicate.attr == ctx.attr
    ):
        return ctx.root_value.predicate.value
    return _maybe_restore_type_self_annotation(ctx, attribute_value)


def _maybe_restore_type_self_annotation(ctx: AttrContext, value: Value) -> Value:
    if _contains_self_typevar(value):
        return value
    root_value = ctx.root_value
    if isinstance(root_value, AnnotatedValue):
        root_value = root_value.value
    self_type: TypeVarValue | None = None
    if isinstance(root_value, SubclassValue) and isinstance(
        root_value.typ, TypeVarValue
    ):
        self_type = root_value.typ
    else:
        bound_self_type = _get_bound_self_type_from_ctx(ctx)
        if isinstance(bound_self_type, TypeVarValue):
            bound_self_key = _class_key_from_value(bound_self_type)
            root_key = _class_key_from_value(root_value)
            if (
                bound_self_key is not None
                and root_key is not None
                and class_keys_match(bound_self_key, root_key)
            ):
                self_type = bound_self_type
    if self_type is None:
        return value
    bound = self_type.typevar_param.bound
    if bound is None:
        return value
    class_key = _class_key_from_value(bound)
    if class_key is None:
        return value
    symbol = (
        ctx.get_can_assign_context()
        .make_type_object(class_key)
        .get_declared_symbol_from_mro(ctx.attr, ctx.get_can_assign_context())
    )
    if (
        symbol is None
        or symbol.is_method
        or symbol.property_info is not None
        or symbol.annotation is None
        or not _contains_self_typevar(symbol.annotation)
    ):
        return value
    return set_self(symbol.annotation, self_type)


def _maybe_specialize_class_partial_root(root_value: Value, ctx: AttrContext) -> Value:
    if not (
        isinstance(root_value, PartialValue)
        and root_value.operation is PartialValueOperation.SUBSCRIPT
    ):
        return root_value
    root = replace_fallback(root_value.root)
    can_assign_ctx = ctx.get_can_assign_context()
    class_key: type | str | None = None
    if isinstance(root, SyntheticClassObjectValue) and isinstance(
        root.class_type, TypedValue
    ):
        class_key = root.class_type.typ
    elif isinstance(root, KnownValue) and isinstance(root.val, type):
        class_key = root.val
    if class_key is None:
        return root_value
    specialized_args = tuple(
        _specialized_class_partial_member_to_type(member, can_assign_ctx)
        for member in root_value.members
    )
    if isinstance(root, SyntheticClassObjectValue) and isinstance(
        root.class_type, TypedValue
    ):
        return SyntheticClassObjectValue(
            root.name, GenericValue(root.class_type.typ, specialized_args)
        )
    assert isinstance(root, KnownValue) and isinstance(root.val, type)
    synthetic_class = ctx.get_synthetic_class(root.val)
    if synthetic_class is not None and isinstance(
        synthetic_class.class_type, TypedValue
    ):
        return SyntheticClassObjectValue(
            synthetic_class.name,
            GenericValue(synthetic_class.class_type.typ, specialized_args),
        )
    generic_bases = ctx.get_generic_bases(root.val, specialized_args)
    typevars = generic_bases.get(root.val)
    if typevars is None:
        return root_value
    return KnownValueWithTypeVars(root.val, typevars)


def _specialized_class_partial_member_to_type(
    member: Value, ctx: CanAssignContext
) -> Value:
    converted = type_from_value(member, visitor=ctx, suppress_errors=True)
    if converted != AnyValue(AnySource.error):
        return converted
    return member


def _get_namedtuple_member_from_sequence_value(
    root_value: SequenceValue, ctx: AttrContext
) -> Value | None:
    if not isinstance(root_value.typ, str):
        return None
    type_object = ctx.get_can_assign_context().make_type_object(root_value.typ)
    if not type_object.is_direct_namedtuple():
        return None
    fields = type_object.get_namedtuple_fields()
    for i, namedtuple_field in enumerate(fields):
        if namedtuple_field.name != ctx.attr or i >= len(root_value.members):
            continue
        is_many, member = root_value.members[i]
        if is_many:
            return None
        return member
    return None


def _extract_super_value(value: Value) -> SuperValue | None:
    if isinstance(value, SuperValue):
        return value
    if isinstance(value, AnnotatedValue):
        return _extract_super_value(value.value)
    return None


def _super_receiver_type_value(value: Value) -> tuple[TypedValue | None, bool]:
    value = replace_fallback(value)
    if isinstance(value, TypedValue):
        return value, False
    if isinstance(value, SubclassValue) and isinstance(value.typ, TypedValue):
        return value.typ, True
    if isinstance(value, KnownValue):
        if isinstance(value.val, type):
            return TypedValue(value.val), True
        return TypedValue(type(value.val)), False
    return None, False


def _super_thisclass_key(value: Value) -> type | str | None:
    value = replace_fallback(value)
    if isinstance(value, KnownValue) and isinstance(value.val, type):
        return value.val
    if isinstance(value, TypedValue):
        return value.typ
    return None


def _super_mro_values(
    receiver_value: TypedValue, ctx: CanAssignContext
) -> Sequence[MroValue]:
    # TODO: switch to just using the MRO; that currently doesn't work because it gets set too late
    if isinstance(receiver_value.typ, type):
        return [TypedValue(base) for base in receiver_value.typ.__mro__]
    return [
        entry.get_mro_value() for entry in receiver_value.get_type_object(ctx).get_mro()
    ]


# TODO: in principle this should be doable with TypeObject.get_attribute if we add a flag
# that says to skip MRO elements up to a certain point.
def _get_attribute_from_super_value(super_value: SuperValue, ctx: AttrContext) -> Value:
    if super_value.selfobj is None:
        return AnyValue(AnySource.inference)
    receiver_value, is_class_access = _super_receiver_type_value(super_value.selfobj)
    thisclass_key = _super_thisclass_key(super_value.thisclass)
    can_assign_ctx = ctx.get_can_assign_context()
    if receiver_value is None or thisclass_key is None:
        return AnyValue(AnySource.inference)

    receiver_tobj = receiver_value.get_type_object(can_assign_ctx)
    saw_thisclass = False
    for mro_value in _super_mro_values(receiver_value, can_assign_ctx):
        if isinstance(mro_value, AnyValue):
            continue
        owner_key = _class_key_from_value(mro_value)
        if owner_key is None:
            continue
        if not saw_thisclass:
            if class_keys_match(owner_key, thisclass_key):
                saw_thisclass = True
            continue
        owner_tobj = mro_value.get_type_object(can_assign_ctx)
        symbol = owner_tobj.get_declared_symbol(ctx.attr)
        if symbol is not None:
            symbol = _specialize_symbol_for_owner(
                receiver_tobj,
                owner_tobj,
                symbol,
                can_assign_ctx,
                receiver_value=receiver_value,
            )
            result = _get_attribute_value_from_symbol(
                symbol,
                can_assign_ctx,
                on_class=is_class_access and not symbol.is_method,
                receiver_value=receiver_value,
            )
            if (
                is_class_access
                and symbol.property_info is not None
                and (
                    result is UNINITIALIZED_VALUE
                    or not _is_property_marker_value(result)
                )
            ):
                result = TypedValue(property)
            result = set_self(result, receiver_value)
            ctx.record_usage(super, result)
            return result
    return UNINITIALIZED_VALUE


def _get_attribute_from_type_alias(value: TypeAliasValue, ctx: AttrContext) -> Value:
    type_params = tuple(
        param.typevar_param.typevar if isinstance(param, TypeVarValue) else param
        for param in value.alias.get_type_params()
    )
    if ctx.attr == "__value__":
        return value.get_value()
    if ctx.attr == "__type_params__":
        return KnownValue(type_params)
    if ctx.attr == "__name__":
        return KnownValue(value.name)
    if ctx.attr == "__module__":
        return KnownValue(value.module)
    return UNINITIALIZED_VALUE


def _get_attribute_from_runtime_type_alias(
    value: RuntimeTypeAliasType, ctx: AttrContext
) -> Value:
    if ctx.attr == "__value__":
        return KnownValue(value.__value__)
    if ctx.attr == "__type_params__":
        return KnownValue(tuple(value.__type_params__))
    if ctx.attr == "__name__":
        return KnownValue(value.__name__)
    if ctx.attr == "__module__":
        return KnownValue(value.__module__)
    return UNINITIALIZED_VALUE


def may_have_dynamic_attributes(typ: type) -> bool:
    """These types have typeshed stubs, but instances may have other attributes."""
    if typ is type or typ is super or typ is types.FunctionType:
        return True
    return False


def _get_attribute_from_subclass(
    typ: type, self_value: Value, ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    bound_self_type = _get_bound_self_type_from_ctx(ctx)
    bound_self_key = (
        _class_key_from_value(bound_self_type) if bound_self_type is not None else None
    )
    if (
        bound_self_type is not None
        and bound_self_key is not None
        and class_keys_match(typ, bound_self_key)
    ):
        self_value = bound_self_type
    if isinstance(self_value, SubclassValue) and isinstance(
        self_value.typ, TypeVarValue
    ):
        can_assign_ctx = ctx.get_can_assign_context()
        type_object = can_assign_ctx.make_type_object(typ)
        symbol = type_object.get_declared_symbol_from_mro(ctx.attr, can_assign_ctx)
        if symbol is not None:
            uses_self = (
                symbol.annotation is not None
                and _contains_self_typevar(symbol.annotation)
            ) or (
                symbol.initializer is not None
                and _contains_self_typevar(symbol.initializer)
            )
            if uses_self or symbol.is_classmethod:
                attribute = _get_type_object_attribute(
                    type_object, ctx.attr, ctx, on_class=True, receiver_value=self_value
                )
                if attribute is not None:
                    ctx.record_usage(typ, attribute.value)
                    return attribute.value

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(type(typ))
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    elif ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(TypedValue(object))])
    elif ctx.attr in {"__name__", "__qualname__", "__module__"}:
        # type[T] represents an arbitrary subclass of T, so class identity
        # attributes should be widened from base-class literals.
        return TypedValue(str)
    elif ctx.attr == "__doc__":
        return unite_values(TypedValue(str), KnownValue(None))
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(typ),
        ctx.attr,
        ctx,
        on_class=True,
        receiver_value=self_value,
    )
    if attribute is not None and _should_use_resolved_class_attribute(attribute):
        ctx.record_usage(typ, attribute.value)
        return attribute.value
    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=True)
    if result is UNINITIALIZED_VALUE:
        synthetic_attr = _get_runtime_attribute_from_synthetic_class(
            typ, (), ctx, on_class=True
        )
        if synthetic_attr is not UNINITIALIZED_VALUE:
            return synthetic_attr
        tobj = ctx.get_can_assign_context().make_type_object(typ)
        if tobj.has_any_base():
            return AnyValue(AnySource.from_another)
    if should_unwrap:
        result = _unwrap_value_from_subclass(result, ctx)
    if isinstance(self_value, GenericValue):
        result = _substitute_typevars(typ, self_value.args, result, provider, ctx)
    result = set_self(result, self_value)
    ctx.record_usage(typ, result)
    return result


_TCAA = Callable[[object], bool]


class TreatClassAttributeAsAny(PyObjectSequenceOption[_TCAA]):
    """Allows treating certain class attributes as Any.

    Instances of this option are callables that take an object found among
    a class's attributes and return True if the attribute should instead
    be treated as Any.

    """

    default_value: ClassVar[Sequence[_TCAA]] = [
        lambda cls_val: cls_val is None or cls_val is NotImplemented
    ]
    name = "treat_class_attribute_as_any"

    @classmethod
    def should_treat_as_any(cls, val: object, options: Options) -> bool:
        option_value = options.get_value_for(cls)
        return any(func(val) for func in option_value)


_CAT = Callable[[object], tuple[Value, Value] | None]


class ClassAttributeTransformer(PyObjectSequenceOption[_CAT]):
    """Transform certain class attributes.

    Instances of this option are callables that take an object found among
    a class's attributes and return either None (if the value should not
    be transformed) or a pair of a get and set type. To disallow setting
    the value, return :data:`pycroscope.value.NO_RETURN_VALUE`.

    If multiple transformers match an object, the first one is used.

    """

    default_value: ClassVar[Sequence[_CAT]] = []
    name = "class_attribute_transformers"

    @classmethod
    def transform_attribute_types(
        cls, val: object, options: Options
    ) -> tuple[Value, Value] | None:
        option_value = options.get_value_for(cls)
        for transformer in option_value:
            result = transformer(val)
            if result is not None:
                return result
        return None

    @classmethod
    def transform_attribute(cls, val: object, options: Options) -> Value | None:
        transformed = cls.transform_attribute_types(val, options)
        if transformed is not None:
            return transformed[0]
        return None


def _unwrap_value_from_subclass(result: Value, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue) or ctx.skip_unwrap:
        return result
    cls_val = result.val
    if (
        isinstance(
            cls_val,
            (
                types.FunctionType,
                types.MethodType,
                MethodDescriptorType,
                SlotWrapperType,
                classmethod,
                staticmethod,
            ),
        )
        or (
            # non-static method
            _static_hasattr(cls_val, "decorator")
            and _static_hasattr(cls_val, "instance")
            and not isinstance(cls_val.instance, type)
        )
        or is_async_fn(cls_val)
    ):
        # static or class method
        return KnownValue(cls_val)
    elif _static_hasattr(cls_val, "__get__"):
        return AnyValue(AnySource.inference)  # can't figure out what this will return
    elif TreatClassAttributeAsAny.should_treat_as_any(cls_val, ctx.options):
        return AnyValue(AnySource.error)
    else:
        transformed = ClassAttributeTransformer.transform_attribute(
            cls_val, ctx.options
        )
        if transformed is not None:
            return transformed
        return KnownValue(cls_val)


def _get_attribute_from_synthetic_typed_value(
    root_value: TypedValue, ctx: AttrContext
) -> Value:
    """Resolve a synthetic instance attribute via ``TypeObject.get_attribute()``."""
    if not isinstance(root_value.typ, str):
        return UNINITIALIZED_VALUE
    can_assign_ctx = ctx.get_can_assign_context()
    type_object = can_assign_ctx.make_type_object(root_value.typ)
    attribute = _get_type_object_attribute(
        type_object, ctx.attr, ctx, on_class=False, receiver_value=root_value
    )
    if attribute is None:
        return UNINITIALIZED_VALUE
    return attribute.value


def _get_typed_instance_lookup_receiver(ctx: AttrContext) -> Value | None:
    lookup_root = ctx.lookup_root_value
    if lookup_root is not None:
        lookup_root = replace_fallback(lookup_root)
        if isinstance(lookup_root, (TypedValue, GenericValue)):
            return lookup_root
    root_value = replace_fallback(ctx.root_value)
    if isinstance(root_value, (TypedValue, GenericValue)):
        return root_value
    return None


def _get_instance_lookup_receiver(ctx: AttrContext) -> Value | None:
    self_value = ctx.get_self_value()
    if _contains_self_typevar(self_value):
        return self_value
    return _get_typed_instance_lookup_receiver(ctx)


def _rebind_resolved_lookup_value(
    value: Value, *, lookup_receiver: Value, self_value: Value
) -> Value:
    """Upgrade a resolved attribute from a lookup approximation to the true self."""
    if _contains_self_typevar(value):
        return set_self(value, self_value)
    if receiver_to_self_type(lookup_receiver) == receiver_to_self_type(self_value):
        return value
    return set_self(value, self_value)


def _get_attribute_from_synthetic_class(
    fq_name: str, self_value: Value, ctx: AttrContext, runtime_type: type | None = None
) -> Value:
    # First check values that are special in Python.
    if ctx.attr == "__class__":
        return KnownValue(type)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    assert isinstance(self_value, SyntheticClassObjectValue)
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(fq_name),
        ctx.attr,
        ctx,
        on_class=True,
        receiver_value=self_value.class_type,
    )
    if attribute is not None and _should_use_resolved_class_attribute(attribute):
        return attribute.value
    result = _get_attribute_from_synthetic_class_inner(
        fq_name, self_value, ctx, seen={id(self_value)}, runtime_type=runtime_type
    )
    if result is UNINITIALIZED_VALUE:
        tobj = ctx.get_can_assign_context().make_type_object(fq_name)
        if tobj.has_any_base():
            return AnyValue(AnySource.from_another)
        return result
    return set_self(result, self_value.class_type)


def _get_attribute_from_synthetic_class_inner(
    fq_name: str,
    self_value: SyntheticClassObjectValue,
    ctx: AttrContext,
    *,
    seen: set[int],
    runtime_type: type | None = None,
) -> Value:
    direct = _get_direct_attribute_from_synthetic_class(self_value, ctx.attr, ctx)
    if direct is not UNINITIALIZED_VALUE:
        direct = dataclass_helpers.maybe_resolve_synthetic_descriptor_attribute(
            self_value,
            ctx.attr,
            direct,
            ctx,
            on_class=True,
            descriptor_get_type=_synthetic_descriptor_get_type,
        )
        return direct
    if _is_instance_only_enum_attr(self_value.class_type, ctx.attr):
        return UNINITIALIZED_VALUE

    tobj = ctx.get_can_assign_context().make_type_object(self_value.class_type.typ)
    if not tobj.has_stubs():
        for base in tobj.get_direct_bases():
            result = _get_attribute_from_synthetic_base(
                base, self_value, ctx, seen=seen
            )
            if result is not UNINITIALIZED_VALUE:
                return result

    result, _ = ctx.get_attribute_from_typeshed_recursively(fq_name, on_class=True)
    if result is not UNINITIALIZED_VALUE:
        return result

    if runtime_type is not None:
        return _get_attribute_from_subclass(runtime_type, self_value.class_type, ctx)
    return result


def _get_direct_attribute_from_synthetic_class(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> Value:
    if _is_synthetic_initvar_attribute(self_value, attr_name, ctx):
        return UNINITIALIZED_VALUE
    symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    if symbol is None:
        return UNINITIALIZED_VALUE
    if symbol.is_property:
        class_type = self_value.class_type
        if isinstance(class_type, TypedValue):
            can_assign_ctx = ctx.get_can_assign_context()
            attribute = _get_type_object_attribute(
                can_assign_ctx.make_type_object(class_type.typ),
                attr_name,
                ctx,
                on_class=True,
                receiver_value=class_type,
            )
            if attribute is not None:
                return attribute.value
        raw_value = symbol.initializer
    elif symbol.annotation is not None and not symbol.is_method:
        raw_value = symbol.annotation
    else:
        raw_value = symbol.initializer
    if raw_value is None:
        if symbol.is_classvar and not symbol.is_method:
            return AnyValue(AnySource.inference)
        return UNINITIALIZED_VALUE
    result = _normalize_synthetic_class_attribute(
        raw_value,
        is_self_returning_classmethod=_is_synthetic_self_classmethod_attribute(
            self_value, attr_name, ctx
        ),
    )
    if _should_deliteralize_synthetic_enum_attr(self_value, attr_name, ctx):
        return _deliteralize_value(result)
    return result


def _get_direct_attribute_from_synthetic_instance(
    self_value: SyntheticClassObjectValue,
    attr_name: str,
    ctx: AttrContext,
    *,
    receiver_value: Value | None = None,
) -> Value:
    class_type = self_value.class_type
    can_assign_ctx = ctx.get_can_assign_context()
    typed_receiver_value = (
        can_assign_ctx.make_type_object(class_type.typ)
        if isinstance(class_type.typ, str)
        else None
    )
    receiver_type_value = (
        replace_fallback(receiver_value) if receiver_value is not None else None
    )
    receiver_class_key = (
        _class_key_from_value(receiver_type_value)
        if receiver_type_value is not None
        else None
    )
    if receiver_class_key is not None:
        typed_receiver_value = can_assign_ctx.make_type_object(receiver_class_key)
    if receiver_value is not None and typed_receiver_value is not None:
        attribute = _get_type_object_attribute(
            typed_receiver_value,
            attr_name,
            ctx,
            on_class=False,
            receiver_value=receiver_value,
        )
        if attribute is not None and _should_use_resolved_instance_attribute(attribute):
            return attribute.value
    if isinstance(class_type, TypedValue):
        attribute = _get_type_object_attribute(
            can_assign_ctx.make_type_object(class_type.typ),
            attr_name,
            ctx,
            on_class=False,
            receiver_value=class_type,
        )
        if attribute is not None and _should_use_resolved_instance_attribute(attribute):
            return attribute.value
    symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    if (
        symbol is not None
        and not symbol.is_method
        and not symbol.is_classvar
        and not symbol.is_initvar
    ):
        return symbol.get_effective_type()
    return _get_direct_attribute_from_synthetic_class(self_value, attr_name, ctx)


def _should_use_resolved_instance_attribute(attribute: TypeObjectAttribute) -> bool:
    symbol = attribute.symbol
    if attribute.is_property or symbol.is_classmethod:
        return True
    if symbol.is_classvar or symbol.is_initvar:
        return False
    if not symbol.is_method:
        return True
    return attribute.value != attribute.declared_value


def _should_use_resolved_class_attribute(attribute: TypeObjectAttribute) -> bool:
    symbol = attribute.symbol
    return (
        attribute.is_metaclass_owner
        or attribute.is_property
        or (symbol.annotation is not None and not symbol.is_classmethod)
        or (
            symbol.is_classmethod
            and (
                symbol.returns_self_on_class_access
                or _contains_self_typevar(attribute.value)
            )
        )
    )


def _maybe_use_resolved_typed_instance_attribute(
    attribute: TypeObjectAttribute,
    *,
    resolved_value: Value,
    receiver_value: Value,
    self_value: Value,
    plain_typed_receiver: bool,
) -> Value | None:
    symbol = attribute.symbol
    raw_runtime_value = replace_fallback(attribute.raw_value)
    if attribute.is_property:
        if plain_typed_receiver:
            runtime_property = (
                raw_runtime_value.val
                if isinstance(raw_runtime_value, KnownValue)
                and isinstance(raw_runtime_value.val, property)
                else None
            )
            if runtime_property is None:
                return None
        return _rebind_resolved_lookup_value(
            resolved_value, lookup_receiver=receiver_value, self_value=self_value
        )
    if symbol.is_classmethod:
        return resolved_value
    if (
        symbol.is_instance_only
        and not symbol.is_classvar
        and not symbol.is_initvar
        and not symbol.is_method
    ):
        if plain_typed_receiver and _contains_typevar(attribute.value):
            return None
        return _rebind_resolved_lookup_value(
            resolved_value, lookup_receiver=receiver_value, self_value=self_value
        )
    if (
        not symbol.is_classvar
        and not symbol.is_initvar
        and attribute.value != attribute.declared_value
    ):
        if symbol.is_method:
            if _contains_self_typevar(receiver_value) or not isinstance(
                replace_fallback(receiver_value), TypedValue
            ):
                return resolved_value
        else:
            normalized_resolved_value = replace_fallback(resolved_value)
            if plain_typed_receiver and (
                isinstance(normalized_resolved_value, TypedValue)
                and isinstance(raw_runtime_value, KnownValue)
                and _static_hasattr(raw_runtime_value.val, "fn")
            ):
                return None
            return _rebind_resolved_lookup_value(
                resolved_value, lookup_receiver=receiver_value, self_value=self_value
            )
    return None


def _get_synthetic_declared_symbol(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> ClassSymbol | None:
    class_type = self_value.class_type
    if not isinstance(class_type, TypedValue):
        return None
    can_assign_ctx = ctx.get_can_assign_context()
    type_object = can_assign_ctx.make_type_object(class_type.typ)
    symbol = type_object.get_synthetic_declared_symbols().get(attr_name)
    if symbol is not None:
        return symbol
    mangled = _maybe_mangle_private_name(attr_name, self_value.name)
    if mangled is None:
        return None
    return type_object.get_synthetic_declared_symbols().get(mangled)


def _synthetic_descriptor_get_type(
    descriptor: Value, *, on_class: bool, instance_value: Value, ctx: AttrContext
) -> Value | None:
    cached_property_return = _get_cached_property_return_type(
        descriptor, ctx.get_can_assign_context()
    )
    if cached_property_return is not None:
        return cached_property_return
    match = _synthetic_descriptor_method_match(
        descriptor,
        "__get__",
        [
            KnownValue(None) if on_class else instance_value,
            AnyValue(AnySource.inference),
        ],
        ctx,
    )
    if match is None:
        return None
    get_signature, _ = match
    return_value = get_signature.return_value
    if not on_class:
        return_value = subtract_values(
            return_value,
            receiver_to_self_type(descriptor, ctx.get_can_assign_context()),
            ctx.get_can_assign_context(),
        )
    return return_value


def _synthetic_descriptor_method_match(
    descriptor: Value, method_name: str, args: Sequence[Value], ctx: AttrContext
) -> tuple[Signature, int] | None:
    signature = _synthetic_descriptor_method_signature_any(descriptor, method_name, ctx)
    if signature is None:
        return None
    selected = _select_matching_synthetic_signature(signature, args, ctx)
    if selected is not None:
        return selected, 0
    # Synthetic dunder methods are often exposed unbound; retry with the
    # descriptor value as an explicit first argument.
    selected = _select_matching_synthetic_signature(signature, [descriptor, *args], ctx)
    if selected is None:
        return None
    return selected, 1


def _synthetic_descriptor_method_signature_any(
    descriptor: Value, method_name: str, ctx: AttrContext
) -> Signature | OverloadedSignature | None:
    descriptor = replace_fallback(descriptor)
    if isinstance(descriptor, AnnotatedValue):
        return _synthetic_descriptor_method_signature_any(
            descriptor.value, method_name, ctx
        )
    if not isinstance(descriptor, (KnownValue, TypedValue, SyntheticClassObjectValue)):
        return None
    descriptor, restore_typevars = shield_nested_self_typevars(descriptor)
    method_ctx = ctx.clone_for_attribute_lookup(Composite(descriptor), method_name)
    method_value = get_attribute(method_ctx)
    if method_value is UNINITIALIZED_VALUE:
        return None
    signature = _signature_from_synthetic_attribute(method_value, method_ctx)
    if signature is None:
        return None
    if restore_typevars:
        signature = signature.substitute_typevars(restore_typevars)
    return signature


def _signature_from_synthetic_attribute(
    value: Value, ctx: AttrContext
) -> Signature | OverloadedSignature | None:
    signature = ctx.signature_from_value(value)
    if signature is None and isinstance(value, KnownValue):
        signature = ctx.get_signature(value.val)
    if isinstance(signature, BoundMethodSignature):
        signature = signature.get_signature(ctx=ctx.get_can_assign_context())
    if isinstance(signature, (Signature, OverloadedSignature)):
        return signature
    return None


def _select_matching_synthetic_signature(
    signature: Signature | OverloadedSignature, args: Sequence[Value], ctx: AttrContext
) -> Signature | None:
    if isinstance(signature, Signature):
        if _signature_accepts_args(signature, args, ctx):
            return signature
        return None
    for overload in signature.signatures:
        if _signature_accepts_args(overload, args, ctx):
            return overload
    return None


def _signature_accepts_args(
    signature: Signature, args: Sequence[Value], ctx: AttrContext
) -> bool:
    can_assign_ctx = ctx.get_can_assign_context()
    positional_params = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
    ]
    variadic_param = signature.get_param_of_kind(ParameterKind.VAR_POSITIONAL)
    if len(args) > len(positional_params) and variadic_param is None:
        return False
    for index, arg in enumerate(args):
        if index < len(positional_params):
            parameter = positional_params[index]
        elif variadic_param is not None:
            parameter = variadic_param
        else:
            return False
        can_assign = has_relation(
            parameter.annotation, arg, Relation.ASSIGNABLE, can_assign_ctx
        )
        if isinstance(can_assign, CanAssignError):
            return False
    for parameter in positional_params[len(args) :]:
        if parameter.default is None:
            return False
    return True


def _maybe_resolve_synthetic_property_attribute(
    value: Value, ctx: AttrContext
) -> Value:
    if value is UNINITIALIZED_VALUE:
        return value
    candidate = replace_fallback(value)
    if isinstance(candidate, AnnotatedValue):
        candidate = replace_fallback(candidate.value)
    if isinstance(candidate, KnownValue) and isinstance(candidate.val, property):
        return ctx.get_property_type_from_argspec(candidate.val)
    return value


def _is_synthetic_initvar_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> bool:
    symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    return symbol is not None and symbol.is_initvar


def _is_synthetic_self_classmethod_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> bool:
    class_type = self_value.class_type
    can_assign_ctx = ctx.get_can_assign_context()
    if isinstance(class_type, TypedValue):
        attribute = _get_type_object_attribute(
            can_assign_ctx.make_type_object(class_type.typ),
            attr_name,
            ctx,
            on_class=True,
            receiver_value=class_type,
        )
        symbol = None if attribute is None else attribute.symbol
    else:
        symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    return symbol is not None and symbol.returns_self_on_class_access


def _maybe_mangle_private_name(attr_name: str, class_name: str) -> str | None:
    if not attr_name.startswith("__") or attr_name.endswith("__"):
        return None
    return f"_{class_name}{attr_name}"


def _normalize_synthetic_class_attribute(
    value: Value, *, is_self_returning_classmethod: bool = False
) -> Value:
    return normalize_synthetic_descriptor_attribute(
        value,
        is_self_returning_classmethod=is_self_returning_classmethod,
        unknown_descriptor_means_any=False,
    )


def _is_instance_only_enum_attr(value: Value, attr_name: str) -> bool:
    class_type = replace_fallback(value)
    if not isinstance(class_type, TypedValue) or not isinstance(class_type.typ, type):
        return False
    if not safe_issubclass(class_type.typ, Enum):
        return False
    return isinstance(Enum.__dict__.get(attr_name), _ENUM_INSTANCE_DESCRIPTOR_TYPES)


def _should_deliteralize_synthetic_enum_attr(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> bool:
    class_type = self_value.class_type
    if not isinstance(class_type, TypedValue) or not isinstance(class_type.typ, type):
        return False
    if not safe_issubclass(class_type.typ, Enum):
        return False
    symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    if (
        symbol is not None
        and isinstance(symbol.initializer, KnownValue)
        and safe_isinstance(symbol.initializer.val, Enum)
    ):
        return False
    try:
        return attr_name not in class_type.typ.__members__
    except Exception:
        return False


def _deliteralize_value(value: Value) -> Value:
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        return unite_values(*[_deliteralize_value(subval) for subval in value.vals])
    if isinstance(value, IntersectionValue):
        return IntersectionValue(
            tuple(_deliteralize_value(subval) for subval in value.vals)
        )
    return _deliteralize_simple_value(value)


def _deliteralize_simple_value(value: SimpleType) -> Value:
    if isinstance(value, KnownValue):
        return TypedValue(type(value.val))
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return value
    assert_never(value)


def _get_attribute_from_synthetic_base(
    base: Value,
    self_value: SyntheticClassObjectValue,
    ctx: AttrContext,
    *,
    seen: set[int],
) -> Value:
    if (
        isinstance(base, PartialValue)
        and base.operation is PartialValueOperation.SUBSCRIPT
    ):
        runtime_value = replace_fallback(base.runtime_value)
        if isinstance(runtime_value, GenericValue):
            base = runtime_value
        else:
            root = replace_fallback(base.root)
            members = tuple(base.members)
            if isinstance(root, SyntheticClassObjectValue):
                class_type = root.class_type
                if isinstance(class_type, TypedValue):
                    base = GenericValue(class_type.typ, members)
            elif isinstance(root, KnownValue) and isinstance(root.val, type):
                base = GenericValue(root.val, members)
            elif isinstance(root, TypedValue):
                base = GenericValue(root.typ, members)

    base = replace_fallback(base)

    if isinstance(base, GenericValue):
        if isinstance(base.typ, str):
            synthetic_base = ctx.get_synthetic_class(base.typ)
            if synthetic_base is not None:
                base_id = id(synthetic_base)
                if base_id not in seen:
                    seen_with_base = {*seen, base_id}
                    result = _get_attribute_from_synthetic_class_inner(
                        base.typ, synthetic_base, ctx, seen=seen_with_base
                    )
                    if result is not UNINITIALIZED_VALUE:
                        return _substitute_typevars(
                            base.typ, base.args, result, base.typ, ctx
                        )
            result, provider = ctx.get_attribute_from_typeshed_recursively(
                base.typ, on_class=True
            )
            if result is not UNINITIALIZED_VALUE:
                return _substitute_typevars(base.typ, base.args, result, provider, ctx)
            return UNINITIALIZED_VALUE
        if isinstance(base.typ, type):
            result = _get_attribute_from_subclass(base.typ, self_value.class_type, ctx)
            if result is not UNINITIALIZED_VALUE:
                return _substitute_typevars(base.typ, base.args, result, base.typ, ctx)
            return result

    if isinstance(base, SyntheticClassObjectValue):
        base_id = id(base)
        if base_id in seen:
            return UNINITIALIZED_VALUE
        seen_with_base = {*seen, base_id}
        if isinstance(base.class_type, TypedDictValue):
            return _get_attribute_from_subclass(dict, self_value.class_type, ctx)
        if isinstance(base.class_type.typ, str):
            return _get_attribute_from_synthetic_class_inner(
                base.class_type.typ, base, ctx, seen=seen_with_base
            )
        return _get_attribute_from_subclass(
            base.class_type.typ, self_value.class_type, ctx
        )

    if isinstance(base, KnownValue):
        if isinstance(base.val, type):
            return _get_attribute_from_subclass(base.val, self_value.class_type, ctx)
        origin = get_origin(base.val)
        if isinstance(origin, type):
            return _get_attribute_from_subclass(origin, self_value.class_type, ctx)

    if isinstance(base, TypedValue):
        if isinstance(base.typ, str):
            synthetic_base = ctx.get_synthetic_class(base.typ)
            if synthetic_base is not None:
                base_id = id(synthetic_base)
                if base_id not in seen:
                    seen_with_base = {*seen, base_id}
                    result = _get_attribute_from_synthetic_class_inner(
                        base.typ, synthetic_base, ctx, seen=seen_with_base
                    )
                    if result is not UNINITIALIZED_VALUE:
                        return result
            result, _ = ctx.get_attribute_from_typeshed_recursively(
                base.typ, on_class=True
            )
            return result
        return _get_attribute_from_subclass(base.typ, self_value.class_type, ctx)

    if isinstance(base, AnyValue):
        return AnyValue(AnySource.from_another)

    return UNINITIALIZED_VALUE


def _contains_self_typevar(value: Value) -> bool:
    return any(
        isinstance(subval, TypeVarValue) and subval.typevar_param.typevar is SelfT
        for subval in value.walk_values()
    )


def _contains_typevar(value: Value) -> bool:
    return any(isinstance(subval, TypeVarValue) for subval in value.walk_values())


def _get_bound_self_type_from_ctx(ctx: AttrContext) -> Value | None:
    return ctx.get_bound_self_type()


def _is_synthetic_instance_method_attribute(
    typ: type, attr_name: str, ctx: AttrContext
) -> bool:
    symbol = (
        ctx.get_can_assign_context()
        .make_type_object(typ)
        .get_declared_symbol_from_mro(attr_name, ctx.get_can_assign_context())
    )
    return symbol is not None and symbol.is_method


def _get_attribute_from_typed(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    elif ctx.attr == "__doc__" and typ is type and generic_args:
        return unite_values(TypedValue(str), KnownValue(None))
    classvar_type = _get_classvar_attribute_type_from_runtime_annotations(typ, ctx)
    if classvar_type is not None:
        ctx.record_usage(typ, classvar_type)
        return set_self(classvar_type, ctx.get_self_value())
    elif ctx.attr in {"__name__", "__qualname__", "__module__"} and (
        typ in {types.FunctionType, types.BuiltinFunctionType}
    ):
        # These are writable instance attributes on function objects. Returning
        # class-level literals like Literal["function"] is too strict.
        return TypedValue(str)
    elif ctx.attr in {"__name__", "__qualname__", "__module__"} and getattr(
        typ, "_is_protocol", False
    ):
        # Protocol base classes expose class identity literals at runtime
        # (e.g. Literal["typing"]), but protocol members should use str.
        return TypedValue(str)
    elif ctx.attr == "__annotations__" and typ in {
        types.FunctionType,
        types.BuiltinFunctionType,
    }:
        return GenericValue(dict, [TypedValue(str), AnyValue(AnySource.explicit)])
    receiver_value = _get_instance_lookup_receiver(ctx)
    if receiver_value is not None:
        can_assign_ctx = ctx.get_can_assign_context()
        plain_typed_receiver = isinstance(replace_fallback(receiver_value), TypedValue)
        attribute = _get_type_object_attribute(
            can_assign_ctx.make_type_object(typ),
            ctx.attr,
            ctx,
            on_class=False,
            receiver_value=receiver_value,
        )
        if attribute is not None:
            resolved_value = _substitute_typevars(
                typ, generic_args, attribute.value, typ, ctx
            )
            resolved_instance = _maybe_use_resolved_typed_instance_attribute(
                attribute,
                resolved_value=resolved_value,
                receiver_value=receiver_value,
                self_value=ctx.get_self_value(),
                plain_typed_receiver=plain_typed_receiver,
            )
            if resolved_instance is not None:
                return resolved_instance
    synthetic_class = ctx.get_synthetic_class(typ)
    if synthetic_class is not None and _contains_self_typevar(ctx.get_self_value()):
        synthetic_result = _get_direct_attribute_from_synthetic_instance(
            synthetic_class, ctx.attr, ctx, receiver_value=receiver_value
        )
        if synthetic_result is not UNINITIALIZED_VALUE:
            synthetic_result = (
                dataclass_helpers.maybe_resolve_synthetic_descriptor_attribute(
                    synthetic_class,
                    ctx.attr,
                    synthetic_result,
                    ctx,
                    on_class=False,
                    descriptor_get_type=_synthetic_descriptor_get_type,
                )
            )
        if synthetic_result is not UNINITIALIZED_VALUE:
            synthetic_result = _maybe_resolve_synthetic_property_attribute(
                synthetic_result, ctx
            )
        if synthetic_result is not UNINITIALIZED_VALUE:
            is_bound_method = _is_synthetic_instance_method_attribute(
                typ, ctx.attr, ctx
            )
            synthetic_result = _substitute_typevars(
                typ, generic_args, synthetic_result, typ, ctx
            )
            if is_bound_method:
                return synthetic_result
            return set_self(synthetic_result, ctx.get_self_value())
    synthetic_attr = _get_runtime_attribute_from_synthetic_class(
        typ, generic_args, ctx, on_class=False
    )
    if synthetic_attr is not UNINITIALIZED_VALUE:
        return synthetic_attr
    if ctx.attr == "__hash__":
        synthetic_class = ctx.get_synthetic_class(typ)
        if synthetic_class is not None:
            synthetic_hash = _get_direct_attribute_from_synthetic_class(
                synthetic_class, "__hash__", ctx
            )
            if synthetic_hash is not UNINITIALIZED_VALUE:
                return set_self(synthetic_hash, ctx.get_self_value())
        # Preserve explicit __hash__ = None from runtime classes. The generic
        # class-attribute unwrapping path widens None to Any, which hides
        # unhashable types in assignability checks.
        try:
            mro = list(type.mro(typ))
        except Exception:
            mro = []
        for base_cls in mro:
            try:
                base_dict = base_cls.__dict__
            except Exception:
                continue
            if "__hash__" not in base_dict:
                continue
            if base_dict["__hash__"] is None:
                return set_self(KnownValue(None), ctx.get_self_value())
            break

    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=False)
    result = _substitute_typevars(typ, generic_args, result, provider, ctx)
    if should_unwrap:
        result = _unwrap_value_from_typed(result, typ, ctx)
    ctx.record_usage(typ, result)
    result = set_self(result, ctx.get_self_value())
    if ctx.attr in {"value", "_value_"} and safe_issubclass(typ, Enum):
        enum_value_type = (
            ctx.get_can_assign_context().make_type_object(typ).get_enum_value_type()
        )
        if enum_value_type is not None:
            return enum_value_type
    if ctx.attr == "name" and safe_issubclass(typ, Enum) and result == TypedValue(str):
        return annotate_value(result, [CustomCheckExtension(EnumName(typ))])
    return result


def _get_runtime_attribute_from_synthetic_class(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext, *, on_class: bool
) -> Value:
    if ctx.attr == "__slots__":
        try:
            return KnownValue(getattr(typ, ctx.attr))
        except Exception:
            pass
    if on_class and safe_issubclass(typ, Enum):
        try:
            runtime_enum_member = getattr(typ, ctx.attr)
        except Exception:
            pass
        else:
            if safe_isinstance(runtime_enum_member, Enum):
                return KnownValue(runtime_enum_member)
    synthetic_class = ctx.get_synthetic_class(typ)
    if synthetic_class is None:
        return UNINITIALIZED_VALUE
    type_object = ctx.get_can_assign_context().make_type_object(typ)
    symbol = _get_synthetic_declared_symbol(synthetic_class, ctx.attr, ctx)
    if type_object.get_direct_dataclass_info() is None:
        if _maybe_mangle_private_name(ctx.attr, synthetic_class.name) is None:
            if symbol is None:
                return UNINITIALIZED_VALUE
            if on_class and symbol.is_instance_only:
                return UNINITIALIZED_VALUE
            if (
                not on_class
                and not symbol.is_instance_only
                and (symbol.is_method or symbol.is_classvar)
            ):
                return UNINITIALIZED_VALUE

    if symbol is None or not symbol.is_method:
        if on_class:
            direct = _get_direct_attribute_from_synthetic_class(
                synthetic_class, ctx.attr, ctx
            )
        else:
            direct = _get_direct_attribute_from_synthetic_instance(
                synthetic_class,
                ctx.attr,
                ctx,
                receiver_value=_get_typed_instance_lookup_receiver(ctx),
            )
        if direct is not UNINITIALIZED_VALUE:
            direct = dataclass_helpers.maybe_resolve_synthetic_descriptor_attribute(
                synthetic_class,
                ctx.attr,
                direct,
                ctx,
                on_class=on_class,
                descriptor_value=(symbol.initializer if symbol is not None else None),
                descriptor_get_type=_synthetic_descriptor_get_type,
            )
        if direct is not UNINITIALIZED_VALUE:
            direct = _substitute_typevars(typ, generic_args, direct, typ, ctx)
            if on_class:
                direct = _unwrap_value_from_subclass(direct, ctx)
            else:
                direct = _unwrap_value_from_typed(direct, typ, ctx)
            return set_self(direct, ctx.get_self_value())
    return UNINITIALIZED_VALUE


def _substitute_typevars(
    typ: type | str,
    generic_args: Sequence[Value],
    result: Value,
    provider: object,
    ctx: AttrContext,
) -> Value:
    generic_bases = ctx.get_generic_bases(typ, generic_args)
    provider_key: type | str | None
    if isinstance(provider, (type, str)) and provider in generic_bases:
        provider_key = provider
    else:
        provider_key = None
    if provider_key is None and not isinstance(provider, str):
        origin = get_origin(provider)
        if isinstance(origin, (type, str)) and origin in generic_bases:
            provider_key = origin
    if provider_key is not None:
        provider_typevars = generic_bases[provider_key]
        substituted_typevars = _typevar_map_from_varlike_pairs(
            (
                typevar,
                (
                    coerce_paramspec_specialization_to_input_sig(value)
                    if is_instance_of_typing_name(typevar, "ParamSpec")
                    else value
                ),
            )
            for typevar, value in _iter_typevar_map_items(provider_typevars)
        )
        result = result.substitute_typevars(substituted_typevars)
    if generic_args and typ in generic_bases:
        tv_map = generic_bases[typ]
        if isinstance(result, KnownValueWithTypeVars):
            merged_typevars = result.typevars.merge(tv_map)
            result = KnownValueWithTypeVars(result.val, merged_typevars)
        else:
            result = result.substitute_typevars(tv_map)
    return result


def _unwrap_value_from_typed(result: Value, typ: type, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue) or ctx.skip_unwrap:
        return result
    typevars = result.typevars if isinstance(result, KnownValueWithTypeVars) else None
    cls_val = result.val
    if isinstance(cls_val, property):
        can_assign_ctx = ctx.get_can_assign_context()
        tobj = can_assign_ctx.make_type_object(typ)
        attr = tobj.get_attribute(
            ctx.attr,
            ctx.get_type_object_attribute_policy(
                on_class=False, receiver_value=ctx.lookup_root_value
            ),
        )
        if attr is not None:
            return attr.value
        return ctx.get_property_type_from_argspec(cls_val)
    elif is_bound_classmethod(cls_val):
        return result
    elif inspect.isfunction(cls_val):
        # either a staticmethod or an unbound method
        try:
            descriptor = inspect.getattr_static(typ, ctx.attr)
        except AttributeError:
            # probably a super call; assume unbound method
            if ctx.attr == "__new__":
                # __new__ is implicitly a staticmethod
                return result
            return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return result
        else:
            return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif isinstance(cls_val, (types.MethodType, MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif _static_hasattr(cls_val, "binder_cls") and _static_hasattr(cls_val, "fn"):
        # qcore/asynq-style decorators expose a binder type on the descriptor but
        # still behave like methods when accessed through instances.
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif (
        _static_hasattr(cls_val, "decorator")
        and _static_hasattr(cls_val, "instance")
        and not isinstance(cls_val.instance, type)
    ):
        # non-static method
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif is_async_fn(cls_val):
        # static or class method
        return result
    elif _static_hasattr(cls_val, "func_code"):
        # Cython function probably
        return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    transformed = ClassAttributeTransformer.transform_attribute(cls_val, ctx.options)
    if transformed is not None:
        return transformed
    if _static_hasattr(cls_val, "__get__"):
        typeshed_type = ctx.get_attribute_from_typeshed(typ, on_class=False)
        if typeshed_type is not UNINITIALIZED_VALUE:
            return typeshed_type
        return AnyValue(AnySource.inference)
    elif TreatClassAttributeAsAny.should_treat_as_any(cls_val, ctx.options):
        return AnyValue(AnySource.error)
    else:
        return result


_KAH = Callable[[object, str], Value | None]


def _default_transformer(obj: object, attr: str) -> Value | None:
    # Type alias to Any
    if obj is Any:
        return AnyValue(AnySource.explicit)

    # Avoid generating huge Union type with the actual value
    if obj is sys and attr == "modules":
        return GenericValue(dict, [TypedValue(str), TypedValue(types.ModuleType)])

    return None


class KnownAttributeHook(PyObjectSequenceOption[_KAH]):
    """Allows hooking into the inferred value for an attribute on a literal."""

    default_value: ClassVar[Sequence[_KAH]] = [_default_transformer]
    name = "known_attribute_hook"

    @classmethod
    def get_attribute(cls, obj: object, attr: str, options: Options) -> Value | None:
        option_value = options.get_value_for(cls)
        for transformer in option_value:
            result = transformer(obj, attr)
            if result is not None:
                return result
        return None


def _get_attribute_from_known(obj: object, ctx: AttrContext) -> Value:
    if safe_isinstance(obj, type):
        ctx.record_attr_read(obj)
    else:
        ctx.record_attr_read(type(obj))

    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
        if ctx.attr in {"__name__", "__qualname__", "__module__"}:
            return TypedValue(str)
        if ctx.attr == "__annotations__":
            return GenericValue(dict, [TypedValue(str), AnyValue(AnySource.explicit)])

    if (obj is None or obj is NoneType) and ctx.should_ignore_none_attributes():
        # This usually indicates some context is set to None
        # in the module and initialized later.
        return AnyValue(AnySource.error)

    hooked_value = KnownAttributeHook.get_attribute(obj, ctx.attr, ctx.options)
    if hooked_value is not None:
        return hooked_value

    if not safe_isinstance(obj, type):
        classvar_type = _get_classvar_attribute_type_from_runtime_annotations(
            type(obj), ctx
        )
        if classvar_type is not None:
            ctx.record_usage(type(obj), classvar_type)
            return classvar_type
        can_assign_ctx = ctx.get_can_assign_context()
        attribute = _get_type_object_attribute(
            can_assign_ctx.make_type_object(type(obj)),
            ctx.attr,
            ctx,
            on_class=False,
            receiver_value=KnownValue(obj),
        )
        if attribute is not None and (
            attribute.symbol.is_classmethod or _contains_self_typevar(attribute.value)
        ):
            ctx.record_usage(type(obj), attribute.value)
            return attribute.value

    if safe_isinstance(obj, type):
        if (
            (bound_self_type := _get_bound_self_type_from_ctx(ctx)) is not None
            and (bound_self_key := _class_key_from_value(bound_self_type)) is not None
            and class_keys_match(obj, bound_self_key)
        ):
            can_assign_ctx = ctx.get_can_assign_context()
            type_object = can_assign_ctx.make_type_object(obj)
            symbol = type_object.get_declared_symbol_from_mro(ctx.attr, can_assign_ctx)
            if (
                symbol is not None
                and not symbol.is_method
                and symbol.annotation is not None
                and _contains_self_typevar(symbol.annotation)
            ):
                result = set_self(symbol.annotation, bound_self_type)
                ctx.record_usage(obj, result)
                return result
            attribute = _get_type_object_attribute(
                type_object,
                ctx.attr,
                ctx,
                on_class=True,
                receiver_value=bound_self_type,
            )
            if attribute is not None and (
                attribute.symbol.returns_self_on_class_access
                or _contains_self_typevar(attribute.value)
            ):
                result = attribute.value
                ctx.record_usage(obj, result)
                return result

    synthetic_attr = UNINITIALIZED_VALUE
    if safe_isinstance(obj, type) and safe_issubclass(obj, Enum):
        synthetic_attr = _get_runtime_attribute_from_synthetic_class(
            obj, (), ctx, on_class=True
        )
        if synthetic_attr is not UNINITIALIZED_VALUE:
            return synthetic_attr
    result, _, _ = _get_attribute_from_mro(obj, ctx, on_class=True)
    if result is UNINITIALIZED_VALUE and safe_isinstance(obj, type):
        if synthetic_attr is UNINITIALIZED_VALUE:
            synthetic_attr = _get_runtime_attribute_from_synthetic_class(
                obj, (), ctx, on_class=True
            )
        if synthetic_attr is not UNINITIALIZED_VALUE:
            return synthetic_attr
        tobj = ctx.get_can_assign_context().make_type_object(obj)
        if tobj.has_any_base():
            result = AnyValue(AnySource.from_another)
    if isinstance(result, KnownValue) and (
        safe_isinstance(result.val, types.MethodType)
        or safe_isinstance(result.val, types.BuiltinFunctionType)
        and result.val.__self__ is obj
    ):
        result = set_self(result, ctx.get_self_value())
    elif safe_isinstance(obj, type):
        result = set_self(result, ctx.get_self_value())
    if isinstance(obj, (types.ModuleType, type)):
        ctx.record_usage(obj, result)
    else:
        ctx.record_usage(type(obj), result)
    return result


def _get_attribute_from_unbound(
    root_value: UnboundMethodValue, ctx: AttrContext
) -> Value:
    if root_value.secondary_attr_name is not None:
        return AnyValue(AnySource.inference)
    method = root_value.get_method()
    if method is None:
        return AnyValue(AnySource.inference)
    try:
        getattr(method, ctx.attr)
    except AttributeError:
        return UNINITIALIZED_VALUE
    result = UnboundMethodValue(
        root_value.attr_name, root_value.composite, secondary_attr_name=ctx.attr
    )
    ctx.record_usage(type(method), result)
    return result


def _get_triple_from_annotations(
    annotations: dict[str, object], typ: object, ctx: AttrContext
) -> tuple[Value, object, bool] | None:
    attr_expr = annotation_expr_from_annotations(
        annotations, ctx.attr, ctx=_RuntimeAnnotationsContext(typ)
    )
    if attr_expr is not None:
        attr_type, qualifiers = attr_expr.maybe_unqualify(
            {Qualifier.ClassVar, Qualifier.Final, Qualifier.InitVar}
        )
        if Qualifier.InitVar in qualifiers:
            return None
        if attr_type is not None:
            return (attr_type, typ, False)
    return None


def _get_classvar_attribute_type_from_runtime_annotations(
    typ: type[object], ctx: AttrContext
) -> Value | None:
    """Returns the declared type of a runtime ClassVar attribute, if available."""
    try:
        mro = list(type.mro(typ))
    except Exception:
        return None

    for base_cls in mro:
        if ctx.skip_mro and base_cls is not typ:
            continue

        try:
            if sys.version_info >= (3, 14):
                annotations = get_annotations(base_cls, format=Format.FORWARDREF)
            else:
                annotations = get_annotations(base_cls)  # pragma: no cover
        except Exception:
            continue

        attr_expr = annotation_expr_from_annotations(
            annotations, ctx.attr, ctx=_RuntimeAnnotationsContext(base_cls)
        )
        if attr_expr is None:
            continue

        attr_type, qualifiers = attr_expr.maybe_unqualify(
            {Qualifier.ClassVar, Qualifier.Final, Qualifier.InitVar}
        )
        if (
            Qualifier.ClassVar in qualifiers
            and Qualifier.InitVar not in qualifiers
            and attr_type is not None
        ):
            return attr_type

    return None


def _get_attribute_from_mro(
    typ: object, ctx: AttrContext, on_class: bool
) -> tuple[Value, object, bool]:
    # Then go through the MRO and find base classes that may define the attribute.
    if safe_isinstance(typ, type) and safe_issubclass(typ, Enum):
        # Special case, to avoid picking an attribute of Enum instances (e.g., name)
        # over an Enum member. Ideally we'd have a more principled way to support this
        # but I haven't thought of one.
        try:
            return KnownValue(getattr(typ, ctx.attr)), typ, True
        except Exception:
            pass
        if on_class and isinstance(
            Enum.__dict__.get(ctx.attr), _ENUM_INSTANCE_DESCRIPTOR_TYPES
        ):
            return UNINITIALIZED_VALUE, object, False
    elif safe_isinstance(typ, types.ModuleType):
        try:
            annotations = typ.__annotations__
        except Exception:
            pass
        else:
            triple = _get_triple_from_annotations(annotations, typ, ctx)
            if triple is not None:
                return triple

    if safe_isinstance(typ, type):
        try:
            mro = list(type.mro(typ))
        except Exception:
            # broken mro method
            pass
        else:
            for base_cls in mro:
                if ctx.skip_mro and base_cls is not typ:
                    continue

                typeshed_type = ctx.get_attribute_from_typeshed(
                    base_cls, on_class=on_class or ctx.skip_unwrap
                )
                if typeshed_type is not UNINITIALIZED_VALUE:
                    if ctx.prefer_typeshed:
                        return typeshed_type, base_cls, False
                    # If it's a callable, we'll probably do better
                    # getting the attribute from the type ourselves,
                    # because we may have our own implementation.
                    if not isinstance(typeshed_type, CallableValue):
                        return typeshed_type, base_cls, False

                try:
                    base_dict = base_cls.__dict__
                except Exception:
                    continue

                try:
                    # Make sure to use only __annotations__ that are actually on this
                    # class, not ones inherited from a base class.
                    # Starting in 3.10, __annotations__ is not inherited.
                    if sys.version_info >= (3, 14):
                        annotations = get_annotations(
                            base_cls, format=Format.FORWARDREF
                        )
                    else:
                        annotations = get_annotations(base_cls)  # pragma: no cover
                except Exception:
                    pass
                else:
                    triple = _get_triple_from_annotations(annotations, base_cls, ctx)
                    if triple is not None:
                        return triple

                try:
                    # Make sure we use only the object from this class, but do invoke
                    # the descriptor protocol with getattr.
                    base_dict[ctx.attr]
                except Exception:
                    pass
                else:
                    try:
                        val = KnownValue(getattr(typ, ctx.attr))
                    except Exception:
                        if (
                            ctx.attr == "__slots__"
                            and safe_isinstance(typ, type)
                            and ctx.get_synthetic_class(typ) is not None
                        ):
                            return UNINITIALIZED_VALUE, object, False
                        val = AnyValue(AnySource.inference)
                    return val, base_cls, True

                if typeshed_type is not UNINITIALIZED_VALUE:
                    return typeshed_type, base_cls, False

    attrs_type = get_attrs_attribute(typ, ctx)
    if attrs_type is not None:
        return attrs_type, typ, False

    if not ctx.skip_mro:
        # Even if we didn't find it any __dict__, maybe getattr() finds it directly.
        try:
            return KnownValue(getattr(typ, ctx.attr)), typ, True
        except AttributeError:
            pass
        except Exception:
            if (
                ctx.attr == "__slots__"
                and safe_isinstance(typ, type)
                and ctx.get_synthetic_class(typ) is not None
            ):
                return UNINITIALIZED_VALUE, object, False
            # It exists, but has a broken __getattr__ or something
            return AnyValue(AnySource.inference), typ, True

    return UNINITIALIZED_VALUE, object, False


def _static_hasattr(value: object, attr: str) -> bool:
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def get_attrs_attribute(typ: object, ctx: AttrContext) -> Value | None:
    try:
        if hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == ctx.attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(
                            attr_attr.type, ctx=_RuntimeAnnotationsContext(typ)
                        )
                    else:
                        return AnyValue(AnySource.unannotated)
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
