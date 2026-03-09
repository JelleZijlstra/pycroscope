"""

Code for retrieving the value of attributes.

"""

import ast
import enum
import inspect
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, ClassVar, cast, get_origin

from typing_extensions import assert_never

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotations
else:
    from inspect import get_annotations

from .annotated_types import EnumName
from .annotations import Context, annotation_expr_from_annotations, type_from_runtime
from .input_sig import (
    FullSignature,
    InputSigValue,
    coerce_paramspec_specialization_to_input_sig,
)
from .options import Options, PyObjectSequenceOption
from .relations import Relation, has_relation
from .safe import (
    is_async_fn,
    is_bound_classmethod,
    is_instance_of_typing_name,
    is_typing_name,
    safe_getattr,
    safe_isinstance,
    safe_issubclass,
)
from .signature import (
    BoundMethodSignature,
    InvalidSignature,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
)
from .stacked_scopes import Composite
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    CustomCheckExtension,
    GenericBases,
    GenericValue,
    HasAttrExtension,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    Qualifier,
    SelfTVV,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticEnumMember,
    SyntheticModuleValue,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    annotate_value,
    has_any_base_value,
    replace_fallback,
    set_self,
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
_SYNTHETIC_PROPERTY_GETTER_PREFIX = "%property_getter:"


@dataclass
class AttrContext:
    root_composite: Composite
    attr: str
    options: Options = field(repr=False)
    skip_mro: bool
    skip_unwrap: bool
    prefer_typeshed: bool

    @property
    def root_value(self) -> Value:
        return self.root_composite.value

    def get_self_value(self) -> Value:
        return self.root_value

    def record_usage(self, obj: Any, val: Value) -> None:
        pass

    def record_attr_read(self, obj: Any) -> None:
        pass

    def get_property_type_from_argspec(self, obj: property) -> Value:
        return AnyValue(AnySource.inference)

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        return UNINITIALIZED_VALUE

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        return UNINITIALIZED_VALUE

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> tuple[Value, object]:
        return UNINITIALIZED_VALUE, None

    def should_ignore_none_attributes(self) -> bool:
        return False

    def get_signature(self, obj: object) -> MaybeSignature:
        return None

    def signature_from_value(self, value: Value) -> MaybeSignature:
        return None

    def get_can_assign_context(self) -> CanAssignContext | None:
        return None

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value]
    ) -> GenericBases:
        return {}

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        return None

    def bind_synthetic_instance_attribute(self, attr_name: str, value: Value) -> Value:
        return value

    def should_include_synthetic_methods(self) -> bool:
        return False

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


def get_attribute(ctx: AttrContext) -> Value:
    if (
        isinstance(ctx.root_value, TypeAliasValue)
        and ctx.root_value.uses_type_alias_object_semantics
    ):
        return _get_attribute_from_type_alias(ctx.root_value, ctx)

    root_value = replace_fallback(ctx.root_value)
    if isinstance(root_value, KnownValue) and is_typing_name(
        type(root_value.val), "TypeAliasType"
    ):
        return _get_attribute_from_runtime_type_alias(root_value.val, ctx)
    if isinstance(root_value, KnownValue):
        attribute_value = _get_attribute_from_known(root_value.val, ctx)
    elif isinstance(root_value, TypedValue):
        if (
            isinstance(root_value, CallableValue)
            and ctx.attr == "asynq"
            and root_value.signature.is_asynq
        ):
            return root_value.get_asynq_value()
        if isinstance(root_value, GenericValue):
            args = root_value.args
        else:
            args = ()
        if isinstance(root_value.typ, str):
            attribute_value = _get_attribute_from_synthetic_type(
                root_value.typ, args, ctx
            )
        else:
            attribute_value = _get_attribute_from_typed(root_value.typ, args, ctx)
    elif isinstance(root_value, SubclassValue):
        if isinstance(root_value.typ, TypedValue):
            if isinstance(root_value.typ.typ, str):
                attribute_value = AnyValue(AnySource.inference)
            else:
                attribute_value = _get_attribute_from_subclass(
                    root_value.typ.typ, root_value.typ, ctx
                )
        elif isinstance(root_value.typ, AnyValue):
            attribute_value = AnyValue(AnySource.from_another)
        else:
            attribute_value = _get_attribute_from_known(type, ctx)
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
        isinstance(attribute_value, AnyValue) or attribute_value is UNINITIALIZED_VALUE
    ) and isinstance(ctx.root_value, AnnotatedValue):
        for guard in ctx.root_value.get_metadata_of_type(HasAttrExtension):
            if guard.attribute_name == KnownValue(ctx.attr):
                return guard.attribute_type
    return attribute_value


def _get_attribute_from_type_alias(value: TypeAliasValue, ctx: AttrContext) -> Value:
    type_params = tuple(
        param.typevar if isinstance(param, TypeVarValue) else param
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


def _get_attribute_from_runtime_type_alias(value: object, ctx: AttrContext) -> Value:
    if ctx.attr == "__value__":
        return KnownValue(cast(Any, value).__value__)
    if ctx.attr == "__type_params__":
        return KnownValue(tuple(cast(Any, value).__type_params__))
    if ctx.attr == "__name__":
        return KnownValue(cast(Any, value).__name__)
    if ctx.attr == "__module__":
        return KnownValue(cast(Any, value).__module__)
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
    result, provider, should_unwrap = _get_attribute_from_mro(typ, ctx, on_class=True)
    if should_unwrap:
        result = _unwrap_value_from_subclass(result, ctx)
    if isinstance(self_value, GenericValue):
        result = _substitute_typevars(typ, self_value.args, result, provider, ctx)
    result = set_self(result, ctx.get_self_value())
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

    TODO: The set type is currently ignored.

    """

    default_value: ClassVar[Sequence[_CAT]] = []
    name = "class_attribute_transformers"

    @classmethod
    def transform_attribute(cls, val: object, options: Options) -> Value | None:
        option_value = options.get_value_for(cls)
        for transformer in option_value:
            result = transformer(val)
            if result is not None:
                return result[0]
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


def _get_attribute_from_synthetic_type(
    fq_name: str, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    # First check values that are special in Python
    if ctx.attr == "__class__":
        # TODO: a KnownValue for synthetic types?
        return AnyValue(AnySource.inference)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    synthetic_class = ctx.get_synthetic_class(fq_name)
    if synthetic_class is not None:
        result = _get_direct_attribute_from_synthetic_instance(
            synthetic_class, ctx.attr
        )
        if result is not UNINITIALIZED_VALUE:
            result = _maybe_resolve_synthetic_dataclass_descriptor_attribute(
                synthetic_class, ctx.attr, result, ctx, on_class=False
            )
        if result is UNINITIALIZED_VALUE:
            result = _get_synthetic_dataclass_attribute(
                synthetic_class, ctx, on_class=False
            )
        if result is not UNINITIALIZED_VALUE:
            result = _maybe_resolve_synthetic_property_attribute(result, ctx)
            if _is_synthetic_method_attribute(synthetic_class, ctx.attr) and (
                not ctx.should_include_synthetic_methods()
            ):
                result = UNINITIALIZED_VALUE
        if result is not UNINITIALIZED_VALUE:
            result = ctx.bind_synthetic_instance_attribute(ctx.attr, result)
            result = _substitute_typevars(fq_name, generic_args, result, fq_name, ctx)
            return set_self(result, ctx.get_self_value())
    result, provider = ctx.get_attribute_from_typeshed_recursively(
        fq_name, on_class=False
    )
    if result is UNINITIALIZED_VALUE:
        result, provider = _get_attribute_from_synthetic_type_bases(
            fq_name, generic_args, ctx
        )
    result = _substitute_typevars(fq_name, generic_args, result, provider, ctx)
    result = _maybe_resolve_synthetic_property_attribute(result, ctx)
    result = set_self(result, ctx.get_self_value())
    return result


def _get_attribute_from_synthetic_type_bases(
    fq_name: str, generic_args: Sequence[Value], ctx: AttrContext
) -> tuple[Value, object]:
    generic_bases = ctx.get_generic_bases(fq_name, generic_args)
    for base_typ in generic_bases:
        if base_typ == fq_name:
            continue
        if isinstance(base_typ, str):
            synthetic_class = ctx.get_synthetic_class(base_typ)
            if synthetic_class is not None:
                result = _get_direct_attribute_from_synthetic_instance(
                    synthetic_class, ctx.attr
                )
                if result is not UNINITIALIZED_VALUE:
                    if _is_synthetic_method_attribute(synthetic_class, ctx.attr) and (
                        not ctx.should_include_synthetic_methods()
                    ):
                        result = UNINITIALIZED_VALUE
                if result is not UNINITIALIZED_VALUE:
                    result = ctx.bind_synthetic_instance_attribute(ctx.attr, result)
                    return result, base_typ
            result, provider = ctx.get_attribute_from_typeshed_recursively(
                base_typ, on_class=False
            )
            if result is not UNINITIALIZED_VALUE:
                return result, provider
            continue

        result = ctx.get_attribute_from_typeshed(base_typ, on_class=False)
        if result is UNINITIALIZED_VALUE:
            result, _, should_unwrap = _get_attribute_from_mro(
                base_typ, ctx, on_class=False
            )
            if should_unwrap:
                result = _unwrap_value_from_typed(result, base_typ, ctx)
        if result is not UNINITIALIZED_VALUE:
            return result, base_typ

    return UNINITIALIZED_VALUE, object


def _get_attribute_from_synthetic_class(
    fq_name: str, self_value: Value, ctx: AttrContext, runtime_type: type | None = None
) -> Value:
    # First check values that are special in Python.
    if ctx.attr == "__class__":
        return KnownValue(type)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
    assert isinstance(self_value, SyntheticClassObjectValue)
    result = _get_attribute_from_synthetic_class_inner(
        fq_name, self_value, ctx, seen={id(self_value)}, runtime_type=runtime_type
    )
    if result is UNINITIALIZED_VALUE:
        if _synthetic_class_has_any_base(self_value):
            return AnyValue(AnySource.from_another)
        return result
    result = set_self(result, self_value.class_type)
    return result


def _get_attribute_from_synthetic_class_inner(
    fq_name: str,
    self_value: SyntheticClassObjectValue,
    ctx: AttrContext,
    *,
    seen: set[int],
    runtime_type: type | None = None,
) -> Value:
    direct = _get_direct_attribute_from_synthetic_class(self_value, ctx.attr)
    if direct is not UNINITIALIZED_VALUE:
        direct = _maybe_resolve_synthetic_dataclass_descriptor_attribute(
            self_value, ctx.attr, direct, ctx, on_class=True
        )
        return direct
    if _is_instance_only_enum_attr(self_value.class_type, ctx.attr):
        return UNINITIALIZED_VALUE
    dataclass_attr = _get_synthetic_dataclass_attribute(self_value, ctx, on_class=True)
    if dataclass_attr is not UNINITIALIZED_VALUE:
        return dataclass_attr

    for base in self_value.base_classes:
        result = _get_attribute_from_synthetic_base(base, self_value, ctx, seen=seen)
        if result is not UNINITIALIZED_VALUE:
            return result

    result, _ = ctx.get_attribute_from_typeshed_recursively(fq_name, on_class=True)
    if result is not UNINITIALIZED_VALUE:
        return result
    if runtime_type is not None:
        return _get_attribute_from_subclass(runtime_type, self_value.class_type, ctx)
    return result


def _get_direct_attribute_from_synthetic_class(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> Value:
    selected_name = _select_synthetic_attribute_name(self_value, attr_name)
    if _is_synthetic_initvar_attribute(self_value, selected_name):
        return UNINITIALIZED_VALUE
    if selected_name not in self_value.class_attributes:
        return UNINITIALIZED_VALUE
    result = _normalize_synthetic_class_attribute(
        self_value.class_attributes[selected_name],
        is_self_returning_classmethod=_is_synthetic_self_classmethod_attribute(
            self_value, selected_name
        ),
    )
    if _should_deliteralize_synthetic_enum_attr(self_value, selected_name):
        return _deliteralize_value(result)
    return result


def _get_direct_attribute_from_synthetic_instance(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> Value:
    selected_name = _select_synthetic_attribute_name(self_value, attr_name)
    getter_key = f"{_SYNTHETIC_PROPERTY_GETTER_PREFIX}{selected_name}"
    if getter_key in self_value.class_attributes:
        return self_value.class_attributes[getter_key]
    return _get_direct_attribute_from_synthetic_class(self_value, attr_name)


def _select_synthetic_attribute_name(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> str:
    selected_name = attr_name
    if selected_name not in self_value.class_attributes:
        mangled = _maybe_mangle_private_name(selected_name, self_value.name)
        if mangled is not None and mangled in self_value.class_attributes:
            selected_name = mangled
    return selected_name


def _is_synthetic_method_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> bool:
    selected_name = attr_name
    if selected_name not in self_value.method_attributes:
        mangled = _maybe_mangle_private_name(selected_name, self_value.name)
        if mangled is not None:
            selected_name = mangled
    if selected_name in self_value.method_attributes:
        return True
    return (
        self_value.is_dataclass
        and selected_name == "__init__"
        and _synthetic_dataclass_init_enabled(self_value)
    )


def _synthetic_dataclass_flag(
    self_value: SyntheticClassObjectValue, flag_name: str, *, default: bool
) -> bool:
    raw_value = self_value.class_attributes.get(flag_name)
    if isinstance(raw_value, KnownValue) and isinstance(raw_value.val, bool):
        return raw_value.val
    return default


def _synthetic_dataclass_init_enabled(self_value: SyntheticClassObjectValue) -> bool:
    return _synthetic_dataclass_flag(self_value, "%dataclass_init", default=True)


def _synthetic_dataclass_match_args_enabled(
    self_value: SyntheticClassObjectValue,
) -> bool:
    return _synthetic_dataclass_flag(self_value, "%dataclass_match_args", default=True)


def _iter_synthetic_dataclass_base_init_parameters(
    base: Value, ctx: AttrContext, *, seen: set[int]
) -> list[SigParameter]:
    base = replace_fallback(base)
    synthetic_base: SyntheticClassObjectValue | None = None
    if isinstance(base, SyntheticClassObjectValue):
        synthetic_base = base
    elif isinstance(base, GenericValue):
        if isinstance(base.typ, (type, str)):
            synthetic_base = ctx.get_synthetic_class(base.typ)
    elif isinstance(base, TypedValue):
        if isinstance(base.typ, (type, str)):
            synthetic_base = ctx.get_synthetic_class(base.typ)
    elif isinstance(base, KnownValue) and isinstance(base.val, type):
        synthetic_base = ctx.get_synthetic_class(base.val)
    if synthetic_base is None or not synthetic_base.is_dataclass:
        return []
    return _get_synthetic_dataclass_init_parameters(
        synthetic_base, ctx, include_inherited=True, seen=seen
    )


def _synthetic_dataclass_converter_input_types(
    synthetic_class: SyntheticClassObjectValue,
) -> dict[str, Value]:
    raw_converter_input_types = synthetic_class.class_attributes.get(
        "%dataclass_converter_input_types"
    )
    if not (
        isinstance(raw_converter_input_types, KnownValue)
        and isinstance(raw_converter_input_types.val, Mapping)
    ):
        return {}
    return {
        field_name: input_type
        for field_name, input_type in raw_converter_input_types.val.items()
        if isinstance(field_name, str) and isinstance(input_type, Value)
    }


def _get_synthetic_dataclass_init_parameters(
    synthetic_class: SyntheticClassObjectValue,
    ctx: AttrContext,
    *,
    include_inherited: bool,
    seen: set[int] | None,
) -> list[SigParameter]:
    if seen is None:
        seen = set()
    synthetic_id = id(synthetic_class)
    if synthetic_id in seen:
        return []
    seen.add(synthetic_id)

    params_by_field: dict[str, SigParameter] = {}
    field_order: list[str] = []
    if include_inherited:
        for base in synthetic_class.base_classes:
            for inherited in _iter_synthetic_dataclass_base_init_parameters(
                base, ctx, seen=seen
            ):
                if inherited.name not in params_by_field:
                    field_order.append(inherited.name)
                params_by_field[inherited.name] = inherited

    classvar_names: set[str] = set()
    classvars = synthetic_class.class_attributes.get("%classvars")
    if isinstance(classvars, KnownValue) and isinstance(
        classvars.val, (set, frozenset, tuple, list)
    ):
        classvar_names.update(name for name in classvars.val if isinstance(name, str))

    default_fields: set[str] = set()
    default_names = synthetic_class.class_attributes.get("%dataclass_default_fields")
    if isinstance(default_names, KnownValue) and isinstance(
        default_names.val, (set, frozenset, tuple, list)
    ):
        default_fields.update(
            name for name in default_names.val if isinstance(name, str)
        )

    init_false_fields: set[str] = set()
    init_false_names = synthetic_class.class_attributes.get(
        "%dataclass_init_false_fields"
    )
    if isinstance(init_false_names, KnownValue) and isinstance(
        init_false_names.val, (set, frozenset, tuple, list)
    ):
        init_false_fields.update(
            name for name in init_false_names.val if isinstance(name, str)
        )

    kw_only_fields: set[str] = set()
    kw_only_names = synthetic_class.class_attributes.get("%dataclass_kw_only_fields")
    if isinstance(kw_only_names, KnownValue) and isinstance(
        kw_only_names.val, (set, frozenset, tuple, list)
    ):
        kw_only_fields.update(
            name for name in kw_only_names.val if isinstance(name, str)
        )

    aliases: dict[str, str] = {}
    alias_names = synthetic_class.class_attributes.get("%dataclass_field_aliases")
    if isinstance(alias_names, KnownValue) and isinstance(alias_names.val, dict):
        aliases = {
            field_name: alias
            for field_name, alias in alias_names.val.items()
            if isinstance(field_name, str) and isinstance(alias, str)
        }
    converter_input_types = _synthetic_dataclass_converter_input_types(synthetic_class)

    ordered_fields: list[str] = []
    order = synthetic_class.class_attributes.get("%dataclass_field_order")
    if isinstance(order, KnownValue) and isinstance(order.val, (tuple, list)):
        ordered_fields = [name for name in order.val if isinstance(name, str)]
    field_names = (
        ordered_fields
        if ordered_fields
        else [
            name
            for name in synthetic_class.class_attributes
            if not name.startswith("%")
            and not (name.startswith("__") and name.endswith("__"))
        ]
    )

    for field_name in field_names:
        attr = synthetic_class.class_attributes.get(field_name, UNINITIALIZED_VALUE)
        excluded = (
            attr is UNINITIALIZED_VALUE
            or field_name in synthetic_class.method_attributes
            or field_name in classvar_names
            or field_name in init_false_fields
        )
        if excluded:
            params_by_field.pop(field_name, None)
            continue

        parameter_name = aliases.get(field_name, field_name)
        annotation = converter_input_types.get(field_name)
        if annotation is None:
            annotation = _synthetic_dataclass_parameter_annotation_for_field(attr, ctx)

        parameter = SigParameter(
            parameter_name,
            (
                ParameterKind.KEYWORD_ONLY
                if field_name in kw_only_fields
                else ParameterKind.POSITIONAL_OR_KEYWORD
            ),
            default=KnownValue(...) if field_name in default_fields else None,
            annotation=annotation,
        )
        if field_name not in field_order:
            field_order.append(field_name)
        params_by_field[field_name] = parameter

    params = [params_by_field[name] for name in field_order if name in params_by_field]
    positional_params = [
        parameter
        for parameter in params
        if parameter.kind is not ParameterKind.KEYWORD_ONLY
    ]
    kw_only_params = [
        parameter
        for parameter in params
        if parameter.kind is ParameterKind.KEYWORD_ONLY
    ]
    return [*positional_params, *kw_only_params]


def _synthetic_dataclass_init_callable_value(
    synthetic_class: SyntheticClassObjectValue, ctx: AttrContext, *, on_class: bool
) -> Value:
    parameters = _get_synthetic_dataclass_init_parameters(
        synthetic_class, ctx, include_inherited=True, seen=None
    )
    if on_class:
        parameters = [
            SigParameter(
                "self",
                ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation=AnyValue(AnySource.inference),
            ),
            *parameters,
        ]
    try:
        signature = Signature.make(parameters, KnownValue(None))
    except InvalidSignature:
        return AnyValue(AnySource.inference)
    return CallableValue(signature)


def _synthetic_dataclass_parameter_annotation_for_field(
    attr: Value, ctx: AttrContext
) -> Value:
    descriptor_set_type = _synthetic_descriptor_set_type(attr, ctx)
    if descriptor_set_type is not None:
        return descriptor_set_type
    if isinstance(attr, KnownValue):
        return TypedValue(type(attr.val))
    return attr


def _synthetic_descriptor_set_type(descriptor: Value, ctx: AttrContext) -> Value | None:
    signature = _synthetic_descriptor_method_signature_any(descriptor, "__set__", ctx)
    if signature is None:
        return None
    signatures = (
        [signature] if isinstance(signature, Signature) else signature.signatures
    )
    for set_signature in signatures:
        positional_params = [
            parameter
            for parameter in set_signature.parameters.values()
            if parameter.kind
            in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) >= 3:
            return positional_params[2].annotation
        if len(positional_params) >= 2:
            return positional_params[1].annotation
    return None


def _synthetic_descriptor_get_type(
    descriptor: Value, *, on_class: bool, instance_value: Value, ctx: AttrContext
) -> Value | None:
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
    return get_signature.return_value


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
    if not isinstance(
        descriptor, (KnownValue, TypedValue, GenericValue, SyntheticClassObjectValue)
    ):
        return None
    method_ctx = ctx.clone_for_attribute_lookup(Composite(descriptor), method_name)
    method_value = get_attribute(method_ctx)
    if method_value is UNINITIALIZED_VALUE:
        return None
    return _signature_from_synthetic_attribute(method_value, method_ctx)


def _signature_from_synthetic_attribute(
    value: Value, ctx: AttrContext
) -> Signature | OverloadedSignature | None:
    signature = ctx.signature_from_value(value)
    if signature is None and isinstance(value, KnownValue):
        signature = ctx.get_signature(value.val)
    can_assign_ctx = ctx.get_can_assign_context()
    if isinstance(signature, BoundMethodSignature):
        if can_assign_ctx is None:
            return None
        signature = signature.get_signature(ctx=can_assign_ctx)
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
    if can_assign_ctx is None:
        return False
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


def _maybe_resolve_synthetic_dataclass_descriptor_attribute(
    synthetic_class: SyntheticClassObjectValue,
    attr_name: str,
    value: Value,
    ctx: AttrContext,
    *,
    on_class: bool,
) -> Value:
    if (
        not synthetic_class.is_dataclass
        or attr_name in synthetic_class.method_attributes
        or (attr_name.startswith("__") and attr_name.endswith("__"))
    ):
        return value
    descriptor_get_type = _synthetic_descriptor_get_type(
        value, on_class=on_class, instance_value=ctx.get_self_value(), ctx=ctx
    )
    if descriptor_get_type is not None:
        return descriptor_get_type
    if not on_class and isinstance(value, AnnotatedValue):
        inner_value = replace_fallback(value.value)
        if isinstance(inner_value, KnownValue):
            return annotate_value(TypedValue(type(inner_value.val)), value.metadata)
    if not on_class and isinstance(value, KnownValue):
        return TypedValue(type(value.val))
    return value


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


def _get_synthetic_dataclass_attribute(
    synthetic_class: SyntheticClassObjectValue, ctx: AttrContext, *, on_class: bool
) -> Value:
    if not synthetic_class.is_dataclass:
        return UNINITIALIZED_VALUE
    if ctx.attr == "__init__":
        if not _synthetic_dataclass_init_enabled(synthetic_class):
            return UNINITIALIZED_VALUE
        return _synthetic_dataclass_init_callable_value(
            synthetic_class, ctx, on_class=on_class
        )
    if ctx.attr == "__match_args__":
        if not _synthetic_dataclass_match_args_enabled(synthetic_class):
            return UNINITIALIZED_VALUE
        init_parameters = _get_synthetic_dataclass_init_parameters(
            synthetic_class, ctx, include_inherited=True, seen=None
        )
        return KnownValue(
            tuple(
                param.name
                for param in init_parameters
                if param.kind is not ParameterKind.KEYWORD_ONLY
            )
        )
    if ctx.attr == "__dataclass_fields__":
        return GenericValue(dict, [TypedValue(str), AnyValue(AnySource.explicit)])
    return UNINITIALIZED_VALUE


def _is_synthetic_initvar_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> bool:
    initvar_fields = self_value.class_attributes.get("%dataclass_initvar_fields")
    if not isinstance(initvar_fields, KnownValue) or not isinstance(
        initvar_fields.val, (set, frozenset, tuple, list)
    ):
        return False
    return attr_name in initvar_fields.val


def _is_synthetic_self_classmethod_attribute(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> bool:
    self_classmethods = self_value.class_attributes.get("%self_classmethods")
    if not isinstance(self_classmethods, KnownValue) or not isinstance(
        self_classmethods.val, (set, frozenset, tuple, list)
    ):
        return False
    return attr_name in self_classmethods.val


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
        wrapped = value.args[1] if len(value.args) >= 2 else next(iter(value.args))

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
        return wrapped
    if isinstance(value, KnownValue) and isinstance(value.val, staticmethod):
        return KnownValue(value.val.__func__)
    if isinstance(value, KnownValue) and isinstance(value.val, classmethod):
        return KnownValue(value.val.__func__)
    return value


def _normalize_synthetic_class_attribute(
    value: Value, *, is_self_returning_classmethod: bool = False
) -> Value:
    return normalize_synthetic_descriptor_attribute(
        value,
        is_self_returning_classmethod=is_self_returning_classmethod,
        unknown_descriptor_means_any=False,
    )


def _maybe_mangle_private_name(attr_name: str, class_name: str) -> str | None:
    if not attr_name.startswith("__") or attr_name.endswith("__"):
        return None
    return f"_{class_name}{attr_name}"


def _is_instance_only_enum_attr(value: Value, attr_name: str) -> bool:
    class_type = replace_fallback(value)
    if not isinstance(class_type, TypedValue) or not isinstance(class_type.typ, type):
        return False
    if not safe_issubclass(class_type.typ, Enum):
        return False
    return isinstance(Enum.__dict__.get(attr_name), _ENUM_INSTANCE_DESCRIPTOR_TYPES)


def _should_deliteralize_synthetic_enum_attr(
    self_value: SyntheticClassObjectValue, attr_name: str
) -> bool:
    class_type = self_value.class_type
    if not isinstance(class_type, TypedValue) or not isinstance(class_type.typ, type):
        return False
    if not safe_issubclass(class_type.typ, Enum):
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
        if isinstance(value.val, SyntheticEnumMember):
            return value
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
                if isinstance(class_type, TypedValue) and isinstance(
                    class_type.typ, (type, str)
                ):
                    base = GenericValue(class_type.typ, members)
            elif isinstance(root, KnownValue) and isinstance(root.val, type):
                base = GenericValue(root.val, members)
            elif isinstance(root, TypedValue) and isinstance(root.typ, (type, str)):
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
            result, _ = ctx.get_attribute_from_typeshed_recursively(
                base.typ, on_class=True
            )
            return result
        return _get_attribute_from_subclass(base.typ, self_value.class_type, ctx)

    if isinstance(base, MultiValuedValue):
        for subval in base.vals:
            result = _get_attribute_from_synthetic_base(
                subval, self_value, ctx, seen=seen
            )
            if result is not UNINITIALIZED_VALUE:
                return result
        return UNINITIALIZED_VALUE

    if isinstance(base, AnyValue):
        return AnyValue(AnySource.from_another)

    return UNINITIALIZED_VALUE


def _synthetic_class_has_any_base(self_value: SyntheticClassObjectValue) -> bool:
    return any(has_any_base_value(base) for base in self_value.base_classes)


def _get_attribute_from_typed(
    typ: type, generic_args: Sequence[Value], ctx: AttrContext
) -> Value:
    ctx.record_attr_read(typ)

    # First check values that are special in Python
    if ctx.attr == "__class__":
        return KnownValue(typ)
    elif ctx.attr == "__dict__":
        return TypedValue(dict)
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
    elif ctx.attr == "__hash__":
        synthetic_class = ctx.get_synthetic_class(typ)
        if synthetic_class is not None:
            synthetic_hash = _get_direct_attribute_from_synthetic_class(
                synthetic_class, "__hash__"
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
        enum_value_type = _enum_member_value_type(typ)
        if enum_value_type is not None:
            return enum_value_type
    if ctx.attr == "name" and safe_issubclass(typ, Enum) and result == TypedValue(str):
        return annotate_value(result, [CustomCheckExtension(EnumName(typ))])
    return result


def _enum_member_value_type(typ: type[Enum]) -> Value | None:
    values: list[Value] = []
    try:
        members = list(typ)
    except Exception:
        return None
    for member in members:
        if isinstance(member, SyntheticEnumMember):
            values.append(KnownValue(member.value))
            continue
        try:
            values.append(KnownValue(member.value))
        except Exception:
            return None
    if not values:
        return None
    return unite_values(*values)


def _substitute_typevars(
    typ: type | str,
    generic_args: Sequence[Value],
    result: Value,
    provider: object,
    ctx: AttrContext,
) -> Value:
    if isinstance(typ, (type, str)):
        generic_bases = ctx.get_generic_bases(typ, generic_args)
    else:
        generic_bases = {}
    provider_key: type | str | None
    if isinstance(provider, (type, str)) and provider in generic_bases:
        provider_key = provider
    else:
        provider_key = None
    if provider_key is None and not isinstance(provider, str):
        origin = get_origin(provider)
        if isinstance(origin, (type, str)) and origin in generic_bases:
            provider_key = origin
        else:
            maybe_origin = safe_getattr(provider, "__origin__", None)
            if isinstance(maybe_origin, (type, str)) and maybe_origin in generic_bases:
                provider_key = maybe_origin
    if provider_key is None and isinstance(provider, str):
        for base_key in generic_bases:
            if isinstance(base_key, type):
                runtime_name = f"{base_key.__module__}.{base_key.__qualname__}"
                if provider == runtime_name:
                    provider_key = base_key
                    break
                if (
                    provider.startswith("typing.")
                    and base_key.__module__ == "collections.abc"
                    and provider.removeprefix("typing.") == base_key.__qualname__
                ):
                    provider_key = base_key
                    break
    if provider_key is not None:
        provider_typevars = generic_bases[provider_key]
        substituted_typevars = {
            typevar: (
                coerce_paramspec_specialization_to_input_sig(value)
                if is_instance_of_typing_name(typevar, "ParamSpec")
                else value
            )
            for typevar, value in provider_typevars.items()
        }
        result = result.substitute_typevars(substituted_typevars)
    if generic_args and typ in generic_bases:
        ordered_typevars = [
            val.typevar if isinstance(val, TypeVarValue) else None
            for val in generic_bases[typ].values()
        ]
        tv_map = {
            typevar: arg
            for typevar, arg in zip(ordered_typevars, generic_args)
            if typevar is not None
        }
        if isinstance(result, KnownValueWithTypeVars):
            merged_typevars = {**result.typevars, **tv_map}
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
        return ctx.get_property_type_from_argspec(cls_val)
    elif is_bound_classmethod(cls_val):
        return result
    elif inspect.isfunction(cls_val):
        # either a staticmethod or an unbound method
        try:
            descriptor = inspect.getattr_static(typ, ctx.attr)
        except AttributeError:
            # probably a super call; assume unbound method
            if ctx.attr != "__new__":
                return UnboundMethodValue(
                    ctx.attr, ctx.root_composite, typevars=typevars
                )
            else:
                # __new__ is implicitly a staticmethod
                return result
        if isinstance(descriptor, staticmethod) or ctx.attr == "__new__":
            return result
        else:
            return UnboundMethodValue(ctx.attr, ctx.root_composite, typevars=typevars)
    elif isinstance(cls_val, (types.MethodType, MethodDescriptorType, SlotWrapperType)):
        # built-in method; e.g. scope_lib.tests.SimpleDatabox.get
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

    result, _, _ = _get_attribute_from_mro(obj, ctx, on_class=True)
    if (
        isinstance(result, KnownValue)
        and (
            safe_isinstance(result.val, types.MethodType)
            or safe_isinstance(result.val, types.BuiltinFunctionType)
            and result.val.__self__ is obj
        )
        and isinstance(ctx.root_value, AnnotatedValue)
    ):
        result = set_self(result, ctx.get_self_value())
    elif safe_isinstance(obj, type):
        result = set_self(result, TypedValue(obj))
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


@dataclass
class AnnotationsContext(Context):
    attr_ctx: AttrContext
    cls: object

    def __post_init__(self) -> None:
        super().__init__()

    def get_name(self, node: ast.Name) -> Value:
        try:
            if isinstance(self.cls, types.ModuleType):
                globals = self.cls.__dict__
            else:
                globals = sys.modules[self.cls.__module__].__dict__
        except Exception:
            return AnyValue(AnySource.error)
        else:
            return self.get_name_from_globals(node.id, globals)

    def get_signature(self, callable: object) -> MaybeSignature:
        return self.attr_ctx.get_signature(callable)


def _get_triple_from_annotations(
    annotations: dict[str, object], typ: object, ctx: AttrContext
) -> tuple[Value, object, bool] | None:
    attr_expr = annotation_expr_from_annotations(
        annotations, ctx.attr, ctx=AnnotationsContext(ctx, typ)
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
                        annotations = get_annotations(base_cls)
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
                            attr_attr.type, ctx=AnnotationsContext(ctx, typ)
                        )
                    else:
                        return AnyValue(AnySource.unannotated)
    except Exception:
        # Guard against silly objects throwing exceptions on hasattr()
        # or similar shenanigans.
        pass
    return None
