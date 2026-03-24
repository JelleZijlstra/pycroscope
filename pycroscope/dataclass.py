"""Dataclass-specific helpers."""

from collections.abc import Callable

import pycroscope

from .signature import (
    ELLIPSIS_PARAM,
    InvalidSignature,
    ParameterKind,
    Signature,
    SigParameter,
)
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    ClassSymbol,
    DataclassInfo,
    GenericValue,
    KnownValue,
    Qualifier,
    SyntheticClassObjectValue,
    TypedValue,
    Value,
    annotate_value,
    replace_fallback,
)


def synthesize_dataclass_hash_attribute(
    semantics: DataclassInfo | None,
) -> Value | None:
    if semantics is None:
        return None
    if semantics.unsafe_hash is True:
        return AnyValue(AnySource.inference)
    if semantics.eq is False:
        return AnyValue(AnySource.inference)
    if (
        semantics.eq is True
        and semantics.frozen is False
        and semantics.unsafe_hash is False
    ):
        return KnownValue(None)
    if semantics.eq is True and semantics.frozen is True:
        return AnyValue(AnySource.inference)
    return None


def synthesize_dataclass_fields_attribute() -> Value:
    return GenericValue(dict, [TypedValue(str), AnyValue(AnySource.explicit)])


def dataclass_init_enabled(value: SyntheticClassObjectValue) -> bool:
    if value.dataclass_info is None:
        return True
    return value.dataclass_info.init


def dataclass_match_args_enabled(value: SyntheticClassObjectValue) -> bool:
    if value.dataclass_info is None:
        return True
    return value.dataclass_info.match_args


def get_synthetic_constructor_signature(
    value: SyntheticClassObjectValue,
    instance_type: Value,
    *,
    get_field_parameters: Callable[[SyntheticClassObjectValue], list[SigParameter]],
) -> Signature | None:
    params = get_field_parameters(value)
    if not params and value.dataclass_info is None:
        return None
    try:
        return Signature.make(params, instance_type)
    except InvalidSignature:
        return Signature.make([ELLIPSIS_PARAM], instance_type)


def get_synthetic_init_value(
    value: SyntheticClassObjectValue,
    *,
    get_field_parameters: Callable[[SyntheticClassObjectValue], list[SigParameter]],
) -> Value | None:
    if not dataclass_init_enabled(value):
        return None
    params = get_field_parameters(value)
    try:
        signature = Signature.make(
            [
                SigParameter(
                    "self",
                    ParameterKind.POSITIONAL_OR_KEYWORD,
                    annotation=AnyValue(AnySource.inference),
                ),
                *params,
            ],
            KnownValue(None),
        )
    except InvalidSignature:
        return AnyValue(AnySource.inference)
    return CallableValue(signature)


def get_synthetic_match_args_value(
    value: SyntheticClassObjectValue,
    *,
    get_field_parameters: Callable[[SyntheticClassObjectValue], list[SigParameter]],
) -> Value | None:
    if not dataclass_match_args_enabled(value):
        return None
    params = get_field_parameters(value)
    return KnownValue(
        tuple(
            param.name
            for param in params
            if param.kind is not ParameterKind.KEYWORD_ONLY
        )
    )


def _get_local_synthetic_initializer(
    type_object: "pycroscope.type_object.TypeObject", name: str
) -> Value | None:
    # Dataclass-generated members need to consult only the local synthetic overlay.
    # Looking at the merged TypeObject view would treat runtime members (such as a
    # runtime __slots__) as if pycroscope had already synthesized them, which skips
    # adding the overlay entries that the rest of the checker currently relies on.
    symbol = type_object.get_synthetic_declared_symbols().get(name)
    if symbol is None:
        return None
    return symbol.initializer


def apply_synthetic_attributes(
    synthetic_class: SyntheticClassObjectValue,
    semantics: DataclassInfo | None,
    *,
    type_object: "pycroscope.type_object.TypeObject",
    get_slot_names: Callable[[SyntheticClassObjectValue], tuple[str, ...] | None],
    get_field_parameters: Callable[[SyntheticClassObjectValue], list[SigParameter]],
) -> None:
    if semantics is None:
        return
    type_object.set_dataclass_info(semantics)

    if (
        semantics.slots is True
        and _get_local_synthetic_initializer(type_object, "__slots__") is None
    ):
        slot_names = get_slot_names(synthetic_class)
        if slot_names is not None:
            slot_value = KnownValue(slot_names)
            type_object.add_declared_symbol(
                "__slots__", ClassSymbol(initializer=slot_value)
            )

    if _get_local_synthetic_initializer(type_object, "__dataclass_fields__") is None:
        dataclass_fields_value = synthesize_dataclass_fields_attribute()
        type_object.add_declared_symbol(
            "__dataclass_fields__",
            ClassSymbol(
                annotation=dataclass_fields_value,
                qualifiers=frozenset({Qualifier.ClassVar}),
                initializer=dataclass_fields_value,
            ),
        )

    if _get_local_synthetic_initializer(type_object, "__init__") is None:
        init_value = get_synthetic_init_value(
            synthetic_class, get_field_parameters=get_field_parameters
        )
        if init_value is not None:
            type_object.add_declared_symbol(
                "__init__", ClassSymbol(is_method=True, initializer=init_value)
            )

    if _get_local_synthetic_initializer(type_object, "__match_args__") is None:
        match_args_value = get_synthetic_match_args_value(
            synthetic_class, get_field_parameters=get_field_parameters
        )
        if match_args_value is not None:
            type_object.add_declared_symbol(
                "__match_args__", ClassSymbol(initializer=match_args_value)
            )

    if _get_local_synthetic_initializer(type_object, "__hash__") is None:
        hash_value = synthesize_dataclass_hash_attribute(semantics)
        if hash_value is not None:
            type_object.add_declared_symbol(
                "__hash__", ClassSymbol(is_method=True, initializer=hash_value)
            )


def maybe_resolve_synthetic_descriptor_attribute(
    synthetic_class: SyntheticClassObjectValue,
    attr_name: str,
    value: Value,
    ctx: "pycroscope.attributes.AttrContext",
    *,
    on_class: bool,
    descriptor_get_type: Callable[..., Value | None],
) -> Value:
    if (
        not synthetic_class.is_dataclass
        or (
            (
                symbol := ctx.get_can_assign_context()
                .make_type_object(synthetic_class.class_type.typ)
                .get_declared_symbol(attr_name)
            )
            is not None
            and symbol.is_method
        )
        or (attr_name.startswith("__") and attr_name.endswith("__"))
    ):
        return value
    resolved = descriptor_get_type(
        value, on_class=on_class, instance_value=ctx.get_self_value(), ctx=ctx
    )
    if resolved is not None:
        return resolved
    if not on_class and isinstance(value, AnnotatedValue):
        inner_value = replace_fallback(value.value)
        if isinstance(inner_value, KnownValue):
            return annotate_value(TypedValue(type(inner_value.val)), value.metadata)
    if not on_class and isinstance(value, KnownValue):
        return TypedValue(type(value.val))
    return value
