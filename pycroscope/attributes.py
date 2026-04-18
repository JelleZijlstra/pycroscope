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

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotations
else:
    from inspect import get_annotations  # pragma: no cover
from .annotated_types import EnumName
from .annotations import (
    RuntimeAnnotationsContext,
    annotation_expr_from_annotations,
    type_from_runtime,
    type_from_value,
)
from .input_sig import coerce_paramspec_specialization_to_input_sig
from .options import Options, PyObjectSequenceOption
from .predicates import HasAttr
from .relations import intersect_multi
from .safe import (
    is_async_fn,
    is_bound_classmethod,
    is_instance_of_typing_name,
    is_typing_name,
    safe_isinstance,
    safe_issubclass,
)
from .signature import MaybeSignature
from .stacked_scopes import Composite
from .type_object import (
    AttributePolicy,
    TypeObject,
    TypeObjectAttribute,
    _class_key_from_value,
)
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    ClassKey,
    ClassOwner,
    ClassSymbol,
    CustomCheckExtension,
    GenericBases,
    GenericValue,
    GradualType,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    NewTypeValue,
    OverlappingValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    PartialCallValue,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    Qualifier,
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
    TypeVarTupleBindingValue,
    TypeVarTupleValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    _iter_typevar_map_items,
    _typevar_map_from_varlike_pairs,
    annotate_value,
    gradualize,
    replace_fallback,
    set_self,
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
    # TODO: I don't understand why we need this
    lookup_root_value: Value | None
    attr: str
    options: Options = field(repr=False)
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

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        raise NotImplementedError

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
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
        self, typ: ClassKey, generic_args: Sequence[Value]
    ) -> GenericBases:
        raise NotImplementedError

    def get_synthetic_class(self, typ: ClassKey) -> SyntheticClassObjectValue | None:
        raise NotImplementedError

    def clone_for_attribute_lookup(
        self, root_composite: Composite, attr: str
    ) -> "AttrContext":
        return replace(
            self, root_composite=root_composite, attr=attr, prefer_typeshed=False
        )

    def get_type_object_attribute_policy(
        self, *, on_class: bool, receiver: Value
    ) -> AttributePolicy:
        return AttributePolicy(on_class=on_class, receiver=receiver)


def _get_type_object_attribute(
    type_object: TypeObject,
    attr_name: str,
    ctx: AttrContext,
    *,
    on_class: bool,
    receiver_value: Value,
) -> TypeObjectAttribute | None:
    return type_object.get_attribute(
        attr_name,
        ctx.get_type_object_attribute_policy(
            on_class=on_class, receiver=receiver_value
        ),
    )


def get_attribute(ctx: AttrContext) -> Value:
    value, _ = get_attribute_with_error(ctx)
    return value


def get_attribute_with_error(ctx: AttrContext) -> tuple[Value, CanAssignError | None]:
    """Get the value of an attribute.

    This is the main entry point for attribute retrieval. It handles all the
    special cases for different types of root values and falls back to
    TypeObject lookup for normal classes.

    May also return an error describing the issue if the attribute does not exist.

    """
    root_value = gradualize(ctx.root_value)
    return _get_attribute_from_value(root_value, ctx)


def _get_attribute_from_value(
    root_value: GradualType, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    match root_value:
        case AnyValue():
            return AnyValue(AnySource.from_another), None
        case SyntheticModuleValue(module_path=module_path):
            module = ".".join(module_path)
            attribute_value = ctx.resolve_name_from_typeshed(module, ctx.attr)
            if attribute_value is UNINITIALIZED_VALUE:
                return _get_attribute_from_value(TypedValue(types.ModuleType), ctx)
            return attribute_value, None
        case MultiValuedValue(vals=vals):
            if not vals:
                return AnyValue(AnySource.inference), None
            results = [_get_attribute_from_value(gradualize(val), ctx) for val in vals]
            values, errors = zip(*results)
            unified_value = unite_values(*values)
            if any(errors):
                error = CanAssignError(
                    f"Some members of union are missing attribute '{ctx.attr}'",
                    children=[err for err in errors if err is not None],
                )
            else:
                error = None
            return unified_value, error
        case IntersectionValue(vals=vals):
            if not vals:
                return AnyValue(AnySource.inference), None
            results = [_get_attribute_from_value(gradualize(val), ctx) for val in vals]
            filtered_results = [
                (value, error)
                for value, error in results
                if value is not UNINITIALIZED_VALUE
                and value is not NO_RETURN_VALUE
                and error is None
            ]
            if not filtered_results:
                error = CanAssignError(
                    f"All members of intersection are missing attribute '{ctx.attr}'",
                    children=[err for _, err in results if err is not None],
                )
                return UNINITIALIZED_VALUE, error
            values, errors = zip(*filtered_results)
            intersected_value = intersect_multi(values, ctx.get_can_assign_context())
            if any(errors):
                error = CanAssignError(
                    f"Error getting attribute '{ctx.attr}'",
                    children=[err for err in errors if err is not None],
                )
            else:
                error = None
            return intersected_value, error
        case PartialValue(
            operation=PartialValueOperation.SUBSCRIPT,
            root=root,
            members=members,
            runtime_value=TypedValue(types.GenericAlias),
        ):
            return _get_attribute_from_generic_alias(root, members, ctx)
        case (
            AnnotatedValue()
            | OverlappingValue()
            | TypeAliasValue()
            | ParamSpecArgsValue()
            | ParamSpecKwargsValue()
            | TypeAliasValue()
            | NewTypeValue()
            | PartialCallValue()
            | PartialValue()
        ):
            return _get_attribute_from_value(
                gradualize(root_value.get_fallback_value()), ctx
            )
        case SuperValue():
            return _get_attribute_from_super_value(root_value, ctx)
        case PredicateValue(predicate=HasAttr(attr=attr, value=val)) if (
            attr == ctx.attr
        ):
            return val, None
        case PredicateValue(predicate=HasAttr()):
            # TODO: This fixes some tests for now: test_hasattr_on_class_object_preserves_name.
            # A better solution would be to get better at understanding
            # attributes on TypedValue(object).
            return UNINITIALIZED_VALUE, None
        case PredicateValue() | TypeFormValue():
            return _get_attribute_from_value(TypedValue(object), ctx)
        case TypeVarValue():
            if (
                root_value.typevar_param.bound is not None
                or root_value.typevar_param.constraints
            ):
                return _get_attribute_from_value(
                    gradualize(root_value.get_fallback_value()), ctx
                )
            else:
                # TODO: _get_attribute_from_value(TypedValue(object), ctx)
                # This currently leads to false positives with iterating over enums.
                # Might need to wait for better TypeVar handling to fix this.
                return _get_attribute(root_value, ctx), None
        case TypeVarTupleBindingValue() | TypeVarTupleValue():
            # TODO: Not sure these should be part of GradualType at all
            return _get_attribute_from_value(TypedValue(object), ctx)

        # TODO
        case (
            KnownValue()
            | SyntheticClassObjectValue()
            | TypedValue()
            | SubclassValue()
            | UnboundMethodValue()
        ):
            return _get_attribute(root_value, ctx), None
        case _:
            assert_never(root_value)


# attr_exceptions and attr_blocked in CPython genericaliasobject.c
_GENERIC_ALIAS_ATTR_EXCEPTIONS = {
    "__class__",
    "__origin__",
    "__args__",
    "__unpacked__",
    "__parameters__",
    "__typing_unpacked_tuple_args__",
    "__mro_entries__",
    "__reduce_ex__",
    "__reduce__",
}

_GENERIC_ALIAS_ATTR_BLOCKED = {"__bases__", "__copy__", "__deepcopy__"}


def _get_attribute_from_generic_alias(
    root: Value, members: Sequence[Value], ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    if ctx.attr in _GENERIC_ALIAS_ATTR_BLOCKED:
        return UNINITIALIZED_VALUE, CanAssignError(
            f"GenericAlias has no attribute '{ctx.attr}'"
        )
    if ctx.attr in _GENERIC_ALIAS_ATTR_EXCEPTIONS:
        return _get_attribute_from_value(TypedValue(types.GenericAlias), ctx)
    return _get_attribute_from_value(gradualize(root), ctx)


# TODO: Remove this and replace with the switch in _get_attribute_from_value.
def _get_attribute(lookup_root_value: Value, ctx: AttrContext) -> Value:
    if (
        isinstance(ctx.root_value, PartialValue)
        and ctx.root_value.operation is PartialValueOperation.PEP_695_ALIAS
    ):
        assert isinstance(ctx.root_value.root, TypeAliasValue)
        attribute_value = _get_attribute_from_type_alias(ctx.root_value.root, ctx)
        if attribute_value is not UNINITIALIZED_VALUE:
            return attribute_value
        if ctx.lookup_root_value is None:
            lookup_root_value = ctx.root_value.runtime_value
    if (
        isinstance(lookup_root_value, TypeVarValue)
        and lookup_root_value.typevar_param.bound is not None
    ):
        class_key = _class_key_from_value(lookup_root_value.typevar_param.bound)
        if class_key is not None:
            can_assign_ctx = ctx.get_can_assign_context()
            type_object = can_assign_ctx.make_type_object(class_key)
            attribute = _get_type_object_attribute(
                type_object,
                ctx.attr,
                ctx,
                on_class=False,
                receiver_value=lookup_root_value,
            )
            if attribute is not None and (
                attribute.symbol.is_classmethod
                or _contains_self_typevar(attribute.value)
            ):
                return attribute.value
    if (
        isinstance(ctx.root_value, SubclassValue)
        and isinstance(ctx.root_value.typ, TypeVarValue)
        and ctx.root_value.typ.typevar_param.bound is not None
    ):
        class_key = _class_key_from_value(ctx.root_value.typ.typevar_param.bound)
        if class_key is not None:
            tobj = ctx.get_can_assign_context().make_type_object(class_key)
            attr = tobj.get_attribute(
                ctx.attr,
                ctx.get_type_object_attribute_policy(
                    on_class=True, receiver=ctx.root_value.typ
                ),
            )
            if attr is not None:
                return attr.value

    original_lookup_root_value = lookup_root_value
    lookup_root_value = _maybe_specialize_class_partial_root(lookup_root_value, ctx)
    if lookup_root_value != original_lookup_root_value:
        ctx = replace(ctx, lookup_root_value=lookup_root_value)
    super_value = _extract_super_value(lookup_root_value)
    if super_value is not None:
        attribute_value, _ = _get_attribute_from_super_value(super_value, ctx)
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
    attribute_value: Value = UNINITIALIZED_VALUE
    if isinstance(root_value, KnownValue):
        if is_typing_name(type(root_value.val), "TypeAliasType"):
            attribute_value = _get_attribute_from_runtime_type_alias(
                root_value.val, ctx
            )
            if attribute_value is not UNINITIALIZED_VALUE:
                return attribute_value
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
        if isinstance(root_value.typ, ClassOwner):
            attribute_value = _get_attribute_from_synthetic_typed_value(root_value, ctx)
        else:
            attribute_value = _get_attribute_from_typed(root_value.typ, args, ctx)
    elif isinstance(root_value, SubclassValue):
        synthetic_name: ClassKey | None = None
        if isinstance(root_value.typ, TypedValue):
            if isinstance(root_value.typ.typ, ClassOwner):
                synthetic_name = root_value.typ.typ
            else:
                attribute_value = _get_attribute_from_subclass(
                    root_value.typ.typ, root_value, ctx
                )
        elif isinstance(root_value.typ, TypeVarValue):
            if root_value.typ.typevar_param.bound is not None:
                bound = replace_fallback(root_value.typ.typevar_param.bound)
                if isinstance(bound, TypedValue) and isinstance(bound.typ, ClassOwner):
                    synthetic_name = bound.typ
        else:
            assert_never(root_value.typ)
        if synthetic_name is not None:
            can_assign_ctx = ctx.get_can_assign_context()
            tobj = can_assign_ctx.make_type_object(synthetic_name)
            attribute = _get_type_object_attribute(
                tobj, ctx.attr, ctx, on_class=True, receiver_value=root_value.typ
            )
            if attribute is None:
                return UNINITIALIZED_VALUE
            return attribute.value
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
        else:
            attribute_value = _get_attribute_from_synthetic_class(
                root_value.class_type.typ, ctx.root_composite.value, ctx
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
    return attribute_value


def _maybe_specialize_class_partial_root(root_value: Value, ctx: AttrContext) -> Value:
    if not (
        isinstance(root_value, PartialValue)
        and root_value.operation is PartialValueOperation.SUBSCRIPT
    ):
        return root_value
    root = replace_fallback(root_value.root)
    can_assign_ctx = ctx.get_can_assign_context()
    class_key: ClassKey | None = None
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
    if not isinstance(root_value.typ, ClassOwner):
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
        # TypeObject returns the declared field type; this path preserves the
        # exact value stored in a synthetic namedtuple SequenceValue.
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


def _super_thisclass_key(value: Value) -> ClassKey | None:
    value = replace_fallback(value)
    if isinstance(value, KnownValue) and isinstance(value.val, type):
        return value.val
    if isinstance(value, TypedValue):
        return value.typ
    return None


def _ca_error(value: Value, ctx: AttrContext) -> CanAssignError:
    return CanAssignError(f"{value} has no attribute '{ctx.attr}'")


def _get_attribute_from_super_value(
    super_value: SuperValue, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    if super_value.selfobj is None:
        return AnyValue(AnySource.inference), None
    receiver_value, is_class_access = _super_receiver_type_value(super_value.selfobj)
    thisclass_key = _super_thisclass_key(super_value.thisclass)
    if receiver_value is None or thisclass_key is None:
        return AnyValue(AnySource.inference), None

    receiver_tobj = receiver_value.get_type_object(ctx.get_can_assign_context())
    policy = AttributePolicy(
        receiver=receiver_value,
        on_class=is_class_access,
        anchor=thisclass_key,
        # TODO: Maybe shouldn't be necessary, but without this some methods get inferred wrong.
        prefer_symbolic=True,
    )
    attr = receiver_tobj.get_attribute(ctx.attr, policy)
    if attr is None:
        return UNINITIALIZED_VALUE, _ca_error(receiver_value, ctx)
    return attr.value, attr.error


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

    # TypeObject.get_attribute() is still less precise for these type[T]
    # metadata attributes.
    if ctx.attr == "__class__":
        return KnownValue(type(typ))
    if ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(TypedValue(object))])
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(typ),
        ctx.attr,
        ctx,
        on_class=True,
        receiver_value=self_value,
    )
    if attribute is None:
        return UNINITIALIZED_VALUE
    ctx.record_usage(typ, attribute.value)
    return attribute.value


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
    if not isinstance(result, KnownValue):
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
    can_assign_ctx = ctx.get_can_assign_context()
    type_object = can_assign_ctx.make_type_object(root_value.typ)
    attribute = _get_type_object_attribute(
        type_object, ctx.attr, ctx, on_class=False, receiver_value=root_value
    )
    if attribute is None:
        return UNINITIALIZED_VALUE
    return attribute.value


def _get_attribute_from_synthetic_class(
    class_key: ClassKey, self_value: Value, ctx: AttrContext
) -> Value:
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(class_key),
        ctx.attr,
        ctx,
        on_class=True,
        receiver_value=self_value,
    )
    if attribute is None:
        return UNINITIALIZED_VALUE
    return attribute.value


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
        # Synthetic namedtuple fields after import failure can still reach this
        # fallback when TypeObject cannot produce the specialized property value.
        raw_value = symbol.initializer
    elif symbol.annotation is not None and not symbol.is_method:
        raw_value = symbol.annotation
    else:
        raw_value = symbol.initializer
    if raw_value is None:
        if symbol.is_classvar and not symbol.is_method:
            return AnyValue(AnySource.inference)
        return UNINITIALIZED_VALUE
    result = raw_value
    if _should_deliteralize_synthetic_enum_attr(self_value, attr_name, ctx):
        return _deliteralize_value(result)
    return result


def _maybe_use_resolved_typed_instance_attribute(
    attribute: TypeObjectAttribute,
    *,
    resolved_value: Value,
    typ: type,
    ctx: AttrContext,
) -> Value | None:
    symbol = attribute.symbol
    if symbol.is_method and not symbol.is_classmethod:
        # TypeObject may return a symbolic callable here, but callers of
        # attributes.py still rely on UnboundMethodValue for receiver binding.
        legacy_method_value = _unwrap_value_from_typed(attribute.raw_value, typ, ctx)
        if isinstance(legacy_method_value, UnboundMethodValue):
            return legacy_method_value
    if attribute.is_property:
        if ctx.attr in {"name", "value", "_value_"} and safe_issubclass(typ, Enum):
            return None
        return resolved_value
    if symbol.is_classmethod:
        return resolved_value
    if (
        symbol.is_instance_only
        and not symbol.is_classvar
        and not symbol.is_initvar
        and not symbol.is_method
        and _contains_typevar(attribute.value)
        and ctx.get_can_assign_context().make_type_object(typ).is_namedtuple_like()
    ):
        # TypeObject currently leaves inherited synthetic namedtuple fields
        # under-specialized after import failure; the runtime fallback below
        # substitutes the base arguments correctly.
        return None
    return resolved_value


def _get_synthetic_declared_symbol(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> ClassSymbol | None:
    class_type = self_value.class_type
    can_assign_ctx = ctx.get_can_assign_context()
    type_object = can_assign_ctx.make_type_object(class_type.typ)
    symbol = type_object.get_synthetic_declared_symbols().get(attr_name)
    if symbol is not None:
        return symbol
    mangled = _maybe_mangle_private_name(attr_name, self_value.name)
    if mangled is None:
        return None
    return type_object.get_synthetic_declared_symbols().get(mangled)


def _is_synthetic_initvar_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> bool:
    symbol = _get_synthetic_declared_symbol(self_value, attr_name, ctx)
    return symbol is not None and symbol.is_initvar


def _maybe_mangle_private_name(attr_name: str, class_name: str) -> str | None:
    if not attr_name.startswith("__") or attr_name.endswith("__"):
        return None
    return f"_{class_name}{attr_name}"


def _should_deliteralize_synthetic_enum_attr(
    self_value: SyntheticClassObjectValue, attr_name: str, ctx: AttrContext
) -> bool:
    class_type = self_value.class_type
    if not isinstance(class_type.typ, type) or not safe_issubclass(
        class_type.typ, Enum
    ):
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
    elif isinstance(
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
    else:
        assert_never(value)


def _contains_self_typevar(value: Value) -> bool:
    if isinstance(value, KnownValueWithTypeVars) and any(
        type_param.is_self for type_param, _ in value.typevars.iter_typevars()
    ):
        return True
    return any(
        isinstance(subval, TypeVarValue) and subval.typevar_param.is_self
        for subval in value.walk_values()
    )


def _contains_typevar(value: Value) -> bool:
    return any(isinstance(subval, TypeVarValue) for subval in value.walk_values())


def _get_attribute_from_typed(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(typ),
        ctx.attr,
        ctx,
        on_class=False,
        receiver_value=ctx.root_composite.value,
    )
    # Adding "if attribute is None: return UNINITIALIZED_VALUE" here breaks two tests:
    # pycroscope/test_attributes.py::TestAttributes::test_attrs
    # (missing attrs support in type_object.py?)
    # TestImportFailureHandling::test_explicit_type_alias_uses_runtime_attribute_semantics
    # (some weirdness about how we represent type aliases?)
    if attribute is not None:
        resolved_instance = _maybe_use_resolved_typed_instance_attribute(
            attribute, resolved_value=attribute.value, typ=typ, ctx=ctx
        )
        if resolved_instance is not None:
            return resolved_instance
    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=False)
    result = _substitute_typevars(typ, generic_args, result, provider, ctx)
    if should_unwrap:
        result = _unwrap_value_from_typed(result, typ, ctx)
    ctx.record_usage(typ, result)
    assert safe_isinstance(provider, type), repr(provider)
    if ctx.attr in {"value", "_value_"} and safe_issubclass(typ, Enum):
        enum_value_type = (
            ctx.get_can_assign_context().make_type_object(typ).get_enum_value_type()
        )
        if enum_value_type is not None:
            return enum_value_type
    if ctx.attr == "name" and safe_issubclass(typ, Enum) and result == TypedValue(str):
        return annotate_value(result, [CustomCheckExtension(EnumName(typ))])
    return result


def _get_runtime_class_attribute_from_synthetic_class(
    typ: type, ctx: AttrContext
) -> Value:
    # Runtime Enum lookup preserves literal enum members for enum.member()
    # helpers and aliases that TypeObject currently widens to Any.
    if safe_issubclass(typ, Enum):
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
            if symbol is None or symbol.is_instance_only:
                return UNINITIALIZED_VALUE

    if symbol is None or not symbol.is_method:
        direct = _get_direct_attribute_from_synthetic_class(
            synthetic_class, ctx.attr, ctx
        )
        if direct is not UNINITIALIZED_VALUE:
            direct = _substitute_typevars(typ, (), direct, typ, ctx)
            direct = _unwrap_value_from_subclass(direct, ctx)
            return direct
    return UNINITIALIZED_VALUE


def _substitute_typevars(
    typ: ClassKey,
    generic_args: Sequence[Value],
    result: Value,
    provider: object,
    ctx: AttrContext,
) -> Value:
    generic_bases = ctx.get_generic_bases(typ, generic_args)
    provider_key: ClassKey | None = _normalize_class_key(provider)
    if provider_key not in generic_bases:
        provider_key = None
    if provider_key is None and not isinstance(provider, ClassOwner):
        origin = get_origin(provider)
        provider_key = _normalize_class_key(origin)
        if provider_key not in generic_bases:
            provider_key = None
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


def _normalize_class_key(value: object) -> ClassKey | None:
    if isinstance(value, (type, ClassOwner)):
        return value
    return None


def _unwrap_value_from_typed(result: Value, typ: type, ctx: AttrContext) -> Value:
    if not isinstance(result, KnownValue):
        return result
    typevars = result.typevars if isinstance(result, KnownValueWithTypeVars) else None
    cls_val = result.val
    if isinstance(cls_val, property):
        can_assign_ctx = ctx.get_can_assign_context()
        tobj = can_assign_ctx.make_type_object(typ)
        attr = tobj.get_attribute(
            ctx.attr,
            ctx.get_type_object_attribute_policy(
                on_class=False, receiver=ctx.lookup_root_value or ctx.root_value
            ),
        )
        if attr is not None:
            # The initial TypeObject result may be bypassed for enum and
            # namedtuple precision; runtime property unwrapping still needs
            # this resolved descriptor value.
            return attr.value
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
            return UnboundMethodValue(
                ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
            )
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return result
        else:
            return UnboundMethodValue(
                ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
            )
    elif isinstance(cls_val, (types.MethodType, MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
        return UnboundMethodValue(
            ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
        )
    elif _static_hasattr(cls_val, "binder_cls") and _static_hasattr(cls_val, "fn"):
        # qcore/asynq-style decorators expose a binder type on the descriptor but
        # still behave like methods when accessed through instances.
        return UnboundMethodValue(
            ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
        )
    elif (
        _static_hasattr(cls_val, "decorator")
        and _static_hasattr(cls_val, "instance")
        and not isinstance(cls_val.instance, type)
    ):
        # non-static method
        return UnboundMethodValue(
            ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
        )
    elif is_async_fn(cls_val):
        # static or class method
        return result
    elif _static_hasattr(cls_val, "func_code"):
        # Cython function probably
        return UnboundMethodValue(
            ctx.attr, ctx.root_composite, typevars=typevars, owner=typ
        )
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
        can_assign_ctx = ctx.get_can_assign_context()
        type_object = can_assign_ctx.make_type_object(obj)
        attribute = _get_type_object_attribute(
            type_object, ctx.attr, ctx, on_class=True, receiver_value=KnownValue(obj)
        )
        if attribute is not None and (
            attribute.symbol.returns_self_on_class_access
            or _contains_self_typevar(attribute.value)
            or (
                ctx.attr not in {"__doc__", "__name__", "__qualname__", "__module__"}
                and not safe_issubclass(obj, Enum)
                and attribute.symbol.annotation is not None
                and not attribute.symbol.is_method
            )
        ):
            result = attribute.value
            ctx.record_usage(obj, result)
            return result

    if safe_isinstance(obj, type) and safe_issubclass(obj, Enum):
        # Keep this before runtime MRO lookup: enum nonmembers need the synthetic
        # class view to deliteralize values such as enum.nonmember(2) to int.
        synthetic_attr = _get_runtime_class_attribute_from_synthetic_class(obj, ctx)
        if synthetic_attr is not UNINITIALIZED_VALUE:
            return synthetic_attr

    result, provider, _ = _get_attribute_from_mro(obj, ctx, on_class=True)
    if result is UNINITIALIZED_VALUE and safe_isinstance(obj, type):
        # TypeObject.get_attribute() above only returns selected class-object
        # attributes for Self-sensitive cases. Synthetic dataclass-transform
        # members such as __match_args__ still need this class-only fallback.
        synthetic_attr = _get_runtime_class_attribute_from_synthetic_class(obj, ctx)
        if synthetic_attr is not UNINITIALIZED_VALUE:
            return synthetic_attr
        tobj = ctx.get_can_assign_context().make_type_object(obj)
        if tobj.has_any_base():
            result = AnyValue(AnySource.from_another)
    if not safe_isinstance(provider, type):
        provider = type(provider)
    if safe_isinstance(obj, type) or (
        isinstance(result, KnownValue)
        and (
            safe_isinstance(result.val, types.MethodType)
            or safe_isinstance(result.val, types.BuiltinFunctionType)
            and result.val.__self__ is obj
        )
    ):
        # Runtime class-object lookup still produces values with unspecialized
        # Self for importable classes. TypeObject.get_attribute() handles many
        # Self-sensitive cases above, but not all runtime MRO fallbacks.
        result = set_self(result, ctx.get_self_value(), provider)
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
        root_value.attr_name,
        root_value.composite,
        secondary_attr_name=ctx.attr,
        owner=root_value.owner,
    )
    ctx.record_usage(type(method), result)
    return result


def _get_triple_from_annotations(
    annotations: dict[str, object], owner: object, ctx: AttrContext
) -> tuple[Value, object, bool] | None:
    attr_expr = annotation_expr_from_annotations(
        annotations,
        ctx.attr,
        ctx=RuntimeAnnotationsContext(
            owner=owner, self_key=owner if isinstance(owner, type) else None
        ),
    )
    if attr_expr is not None:
        attr_type, qualifiers = attr_expr.maybe_unqualify(
            {Qualifier.ClassVar, Qualifier.Final, Qualifier.InitVar}
        )
        if Qualifier.InitVar in qualifiers:
            return None
        if attr_type is not None:
            return (attr_type, owner, False)
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
            use_runtime_annotations = (
                ctx.get_can_assign_context().make_type_object(typ).is_namedtuple_like()
            )
            for base_cls in mro:
                typeshed_type = ctx.get_attribute_from_typeshed(
                    base_cls, on_class=on_class
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

                if use_runtime_annotations:
                    try:
                        # Make sure to use only __annotations__ that are actually on
                        # this class, not ones inherited from a base class.
                        if sys.version_info >= (3, 14):
                            annotations = get_annotations(
                                base_cls, format=Format.FORWARDREF
                            )
                        else:
                            annotations = get_annotations(base_cls)  # pragma: no cover
                    except Exception:
                        pass
                    else:
                        triple = _get_triple_from_annotations(
                            annotations, base_cls, ctx
                        )
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
        if safe_isinstance(typ, type) and hasattr(typ, "__attrs_attrs__"):
            for attr_attr in typ.__attrs_attrs__:
                if attr_attr.name == ctx.attr:
                    if attr_attr.type is not None:
                        return type_from_runtime(
                            attr_attr.type,
                            ctx=RuntimeAnnotationsContext(owner=typ, self_key=typ),
                        )
                    else:
                        return AnyValue(AnySource.unannotated)
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
