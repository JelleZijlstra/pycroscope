"""

Code for retrieving the value of attributes.

"""

import collections.abc
import enum
import sys
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import typing_extensions
from typing_extensions import assert_never

if sys.version_info >= (3, 14):
    pass
else:
    pass  # pragma: no cover
from .annotations import RuntimeAnnotationsContext, type_from_runtime
from .options import Options, PyObjectSequenceOption
from .predicates import HasAttr
from .relations import intersect_multi
from .safe import safe_getattr, safe_isinstance
from .signature import MaybeSignature
from .stacked_scopes import Composite
from .type_object import AttributePolicy, TypeObject, TypeObjectAttribute
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
    GenericBases,
    GenericValue,
    GradualType,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    OverlappingValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    PartialCallValue,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
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
    gradualize,
    replace_fallback,
    set_self,
    stringify_object,
    unite_values,
)

# these don't appear to be in the standard types module
SlotWrapperType = type(type.__init__)
MethodDescriptorType = type(list.append)
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
        return AttributePolicy(
            on_class=on_class, receiver=receiver, receiver_composite=self.root_composite
        )


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
        case KnownValue():
            return _get_attribute_from_known(root_value, ctx)
        case CallableValue() if ctx.attr == "asynq" and root_value.signature.is_asynq:
            return root_value.get_asynq_value(), None
        case TypedValue():
            return _get_attribute_from_typed(root_value, ctx)
        case UnboundMethodValue():
            return _get_attribute_from_unbound(root_value, ctx)
        case SubclassValue(typ=inner_typ):
            match inner_typ:
                case TypedValue():
                    return _get_attribute_from_subclass(
                        inner_typ.typ, ctx.root_value, ctx
                    )
                case TypeVarValue():
                    fallback_type = gradualize(
                        inner_typ.get_fallback_value().get_type_value(
                            ctx.get_can_assign_context()
                        )
                    )
                    return _get_attribute_from_value(fallback_type, ctx)
                case _:
                    assert_never(inner_typ)
        case SyntheticClassObjectValue(class_type=TypedDictValue()):
            return _get_attribute_from_subclass(dict, ctx.root_value, ctx)
        case SyntheticClassObjectValue(class_type=class_type):
            return _get_attribute_from_subclass(class_type.typ, ctx.root_value, ctx)
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
                return AnyValue(AnySource.inference), None
        case TypeVarTupleBindingValue() | TypeVarTupleValue():
            # TODO: Not sure these should be part of GradualType at all
            return _get_attribute_from_value(TypedValue(object), ctx)
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
    typ: ClassKey, self_value: Value, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    ctx.record_attr_read(typ)

    can_assign_ctx = ctx.get_can_assign_context()
    tobj = can_assign_ctx.make_type_object(typ)
    # TypeObject.get_attribute() is still less precise for these type[T]
    # metadata attributes.
    if ctx.attr == "__class__":
        match tobj.get_metaclass():
            case TypedValue() as metaclass:
                return SubclassValue(metaclass), None
            case AnyValue():
                return SubclassValue(TypedValue(type)), None
            case metaclass:
                assert_never(metaclass)
    if ctx.attr == "__bases__":
        return GenericValue(tuple, [SubclassValue(TypedValue(object))]), None
    attribute = _get_type_object_attribute(
        tobj, ctx.attr, ctx, on_class=True, receiver_value=self_value
    )
    if attribute is None:
        return UNINITIALIZED_VALUE, CanAssignError(
            f"Class {stringify_object(typ)} has no attribute '{ctx.attr}'"
        )
    ctx.record_usage(typ, attribute.value)
    return attribute.value, attribute.error


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


def _get_attribute_from_typed(
    value: TypedValue, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    ctx.record_attr_read(value.typ)
    can_assign_ctx = ctx.get_can_assign_context()
    attribute = _get_type_object_attribute(
        can_assign_ctx.make_type_object(value.typ),
        ctx.attr,
        ctx,
        on_class=False,
        receiver_value=ctx.root_composite.value,
    )
    if attribute is None:
        return UNINITIALIZED_VALUE, _ca_error(value, ctx)
    return attribute.value, attribute.error


def _normalize_class_key(value: object) -> ClassKey | None:
    if isinstance(value, (type, ClassOwner)):
        return value
    return None


_KAH = Callable[[object, str], Value | None]


def _default_transformer(obj: object, attr: str) -> Value | None:
    # Type alias to Any
    if obj is Any:
        return AnyValue(AnySource.explicit)

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


def _get_attribute_from_known(
    value: KnownValue, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    obj = value.val
    if safe_isinstance(obj, type):
        ctx.record_attr_read(obj)
    else:
        ctx.record_attr_read(type(obj))

    if obj is None and ctx.should_ignore_none_attributes():
        # This usually indicates some context is set to None
        # in the module and initialized later.
        return AnyValue(AnySource.error), None

    hooked_value = KnownAttributeHook.get_attribute(obj, ctx.attr, ctx.options)
    if hooked_value is not None:
        return hooked_value, None

    result, error = _get_attribute_from_known_inner(value, ctx)
    if isinstance(obj, (types.ModuleType, type)):
        ctx.record_usage(obj, result)
    else:
        ctx.record_usage(type(obj), result)
    return result, error


def _get_attribute_from_known_inner(
    value: KnownValue, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    obj = value.val

    # We have a few things we can return:
    # - The raw runtime value from getattr()
    # - For modules:
    #   - The runtime annotation
    #   - The annotation from stubs
    # - The value from TypeObject.get_attribute(), which can be better for methods.
    # We use some heuristics to prioritize between these.

    default = object()
    runtime_obj = safe_getattr(obj, ctx.attr, default)
    if runtime_obj is default:
        runtime_value = None
    else:
        runtime_value = KnownValue(runtime_obj)

    if safe_isinstance(obj, types.ModuleType):
        if obj is collections.abc and runtime_value is not None:
            # Prefer the runtime lookup for collections.abc because typeshed pretends
            # its values are imported from typing.
            return runtime_value, None
        attribute_value = ctx.resolve_name_from_typeshed(obj.__name__, ctx.attr)
        if attribute_value is not UNINITIALIZED_VALUE:
            return attribute_value, None

        try:
            mod_annos = obj.__annotations__
        except Exception:
            pass
        else:
            if ctx.attr in mod_annos:
                annotation = mod_annos[ctx.attr]
                annotation_value = type_from_runtime(
                    annotation,
                    ctx.get_can_assign_context(),
                    globals=obj.__dict__,
                    ctx=RuntimeAnnotationsContext(ctx, self_key=None),
                )
                return annotation_value, None

    if safe_isinstance(obj, type):
        tobj = ctx.get_can_assign_context().make_type_object(obj)
        on_class = True
    else:
        tobj = ctx.get_can_assign_context().make_type_object(type(obj))
        on_class = False
    policy = ctx.get_type_object_attribute_policy(on_class=on_class, receiver=value)
    type_object_attr = tobj.get_attribute(ctx.attr, policy)

    # Even if there's no runtime attribute, we believe the annotation if there is one.
    if runtime_value is None:
        if tobj.has_any_base():
            return AnyValue(AnySource.from_another), None
        if (
            type_object_attr is not None
            and type_object_attr.symbol.annotation is not None
        ):
            return type_object_attr.value, type_object_attr.error
        return UNINITIALIZED_VALUE, _ca_error(value, ctx)

    if (
        tobj.is_enum()
        and safe_isinstance(obj, type)
        and safe_isinstance(runtime_value.val, obj)
    ):
        return runtime_value, None

    if (
        type_object_attr is not None
        and type_object_attr.symbol.annotation is not None
        # If it's read-only, we may have a more precise runtime type.
        # But if it's also a ClassVar, believe the annotation
        and (
            (
                not type_object_attr.symbol.is_readonly
                and not type_object_attr.symbol.is_final
            )
            or type_object_attr.symbol.is_classvar
        )
    ):
        # If there's an annotation and the attribute is mutable, we believe the annotation
        return type_object_attr.value, type_object_attr.error

    if type_object_attr is not None and (
        safe_isinstance(obj, type)
        or (
            isinstance(runtime_value, KnownValue)
            and (
                safe_isinstance(
                    runtime_value.val, (types.MethodType, types.BuiltinFunctionType)
                )
                and runtime_value.val.__self__ is obj
            )
        )
    ):
        # Runtime class-object lookup still produces values with unspecialized
        # Self for importable classes. TypeObject.get_attribute() handles many
        # Self-sensitive cases above, but not all runtime MRO fallbacks.
        if safe_isinstance(obj, type):
            self_value = TypedValue(obj)
        else:
            self_value = ctx.get_self_value()
        runtime_value = set_self(runtime_value, self_value, type_object_attr.owner.typ)

    if not safe_isinstance(obj, (types.GenericAlias, typing._GenericAlias)):
        return runtime_value, None

    if type_object_attr is not None:
        return type_object_attr.value, type_object_attr.error

    return runtime_value, None


def _get_attribute_from_unbound(
    root_value: UnboundMethodValue, ctx: AttrContext
) -> tuple[Value, CanAssignError | None]:
    if root_value.secondary_attr_name is not None:
        return AnyValue(AnySource.inference), None
    method = root_value.get_method()
    if method is None:
        return AnyValue(AnySource.inference), None
    try:
        getattr(method, ctx.attr)
    except AttributeError:
        return UNINITIALIZED_VALUE, CanAssignError(
            f"{method} has no attribute '{ctx.attr}'"
        )
    result = UnboundMethodValue(
        root_value.attr_name,
        root_value.composite,
        secondary_attr_name=ctx.attr,
        owner=root_value.owner,
    )
    ctx.record_usage(type(method), result)
    return result, None


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
