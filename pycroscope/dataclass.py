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
    AnySource,
    AnyValue,
    CallableValue,
    ClassKey,
    ClassSymbol,
    DataclassInfo,
    GenericValue,
    KnownValue,
    Qualifier,
    TypedValue,
    Value,
    get_self_param,
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


def get_synthetic_constructor_signature(
    type_object: "pycroscope.type_object.TypeObject",
    instance_type: Value,
    *,
    get_field_parameters: Callable[[ClassKey], list[SigParameter]],
) -> Signature | None:
    params = get_field_parameters(type_object.typ)
    if not params and type_object.get_direct_dataclass_info() is None:
        return None
    try:
        return Signature.make(params, instance_type)
    except InvalidSignature:
        return Signature.make([ELLIPSIS_PARAM], instance_type)


def get_synthetic_init_value(
    type_object: "pycroscope.type_object.TypeObject",
    *,
    get_field_parameters: Callable[[ClassKey], list[SigParameter]],
) -> Value | None:
    dataclass_info = type_object.get_direct_dataclass_info()
    if dataclass_info is not None and not dataclass_info.init:
        return None
    params = get_field_parameters(type_object.typ)
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
            self_param=get_self_param(type_object.typ),
        )
    except InvalidSignature:
        return AnyValue(AnySource.inference)
    return CallableValue(signature)


def get_synthetic_match_args_value(
    type_object: "pycroscope.type_object.TypeObject",
    *,
    get_field_parameters: Callable[[ClassKey], list[SigParameter]],
) -> Value | None:
    dataclass_info = type_object.get_direct_dataclass_info()
    if dataclass_info is not None and not dataclass_info.match_args:
        return None
    params = get_field_parameters(type_object.typ)
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
    semantics: DataclassInfo | None,
    *,
    type_object: "pycroscope.type_object.TypeObject",
    get_slot_names: Callable[[ClassKey], tuple[str, ...] | None],
    get_field_parameters: Callable[[ClassKey], list[SigParameter]],
) -> None:
    if semantics is None:
        return
    type_object.set_dataclass_info(semantics)

    if (
        semantics.slots is True
        and _get_local_synthetic_initializer(type_object, "__slots__") is None
    ):
        slot_names = get_slot_names(type_object.typ)
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
            type_object, get_field_parameters=get_field_parameters
        )
        if init_value is not None:
            type_object.add_declared_symbol(
                "__init__", ClassSymbol(is_method=True, initializer=init_value)
            )

    if _get_local_synthetic_initializer(type_object, "__match_args__") is None:
        match_args_value = get_synthetic_match_args_value(
            type_object, get_field_parameters=get_field_parameters
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

    if semantics.eq and _get_local_synthetic_initializer(type_object, "__eq__") is None:
        eq_value = CallableValue(
            Signature.make(
                [
                    SigParameter("self", ParameterKind.POSITIONAL_OR_KEYWORD),
                    SigParameter("other", ParameterKind.POSITIONAL_OR_KEYWORD),
                ],
                TypedValue(bool),
            )
        )
        type_object.add_declared_symbol(
            "__eq__", ClassSymbol(is_method=True, initializer=eq_value)
        )

    # TODO: should also synthesize order comparison methods if order=True
