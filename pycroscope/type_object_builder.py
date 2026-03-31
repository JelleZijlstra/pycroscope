"""Helpers for computing raw ``TypeObject`` ancestry and symbol metadata."""

import inspect
import sys
from collections.abc import Iterator, Mapping
from dataclasses import MISSING, replace
from typing import get_args, get_origin

from typing_extensions import assert_never

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotations

from .annotations import (
    _RuntimeAnnotationsContext,
    annotation_expr_from_runtime,
    make_type_param,
    type_from_runtime,
)
from .arg_spec import ArgSpecCache
from .checker import Checker
from .safe import is_namedtuple_class, safe_getattr
from .type_object import DataclassFieldRecord
from .value import (
    AnySource,
    AnyValue,
    ClassSymbol,
    DataclassFieldInfo,
    FunctionDecorator,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    PropertyInfo,
    Qualifier,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedValue,
    TypeFormValue,
    TypeParam,
    TypeVarMap,
    UnboundMethodValue,
    Value,
    get_namedtuple_field_annotation,
    match_typevar_arguments,
    replace_fallback,
    type_param_to_value,
)


def _get_runtime_dataclass_fields(typ: type) -> tuple[DataclassFieldRecord, ...]:
    dataclass_fields = safe_getattr(typ, "__dataclass_fields__", None)
    if not isinstance(dataclass_fields, Mapping):
        return ()
    records: list[DataclassFieldRecord] = []
    for name, field in dataclass_fields.items():
        if not isinstance(name, str):
            continue
        records.append(
            DataclassFieldRecord(
                field_name=name, field_info=_runtime_dataclass_field_info(field)
            )
        )
    return tuple(records)


def _runtime_dataclass_field_info(field: object) -> DataclassFieldInfo:
    has_default = (
        safe_getattr(field, "default", MISSING) is not MISSING
        or safe_getattr(field, "default_factory", MISSING) is not MISSING
    )
    init = safe_getattr(field, "init", True)
    if not isinstance(init, bool):
        init = True
    kw_only = safe_getattr(field, "kw_only", False)
    if not isinstance(kw_only, bool):
        kw_only = False
    return DataclassFieldInfo(has_default=has_default, init=init, kw_only=kw_only)


def _iter_base_type_values(
    value: Value,
    arg_spec_cache: ArgSpecCache,
    seen_known_bases: frozenset[int] = frozenset(),
) -> Iterator[TypedValue | AnyValue]:
    if isinstance(value, PartialValue):
        if value.operation is not PartialValueOperation.SUBSCRIPT:
            return
        root: Value = value.root
        if isinstance(root, SyntheticClassObjectValue):
            root = root.class_type
        elif isinstance(root, KnownValue):
            root = arg_spec_cache._type_from_base(root.val)
        root = replace_fallback(root)
        members = tuple(
            (
                arg_spec_cache._type_from_base(member.val)
                if arg_spec_cache is not None and isinstance(member, KnownValue)
                else member
            )
            for member in value.members
        )
        if isinstance(root, SequenceValue) and root.typ is tuple:
            yield SequenceValue(tuple, [(False, member) for member in members])
            return
        if isinstance(root, (TypedValue, GenericValue)):
            yield GenericValue(root.typ, members)
        return
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        for subval in value.vals:
            yield from _iter_base_type_values(subval, arg_spec_cache, seen_known_bases)
        return
    if isinstance(value, IntersectionValue):
        for subval in value.vals:
            yield from _iter_base_type_values(subval, arg_spec_cache, seen_known_bases)
        return
    yield from _iter_base_type_values_from_simple(
        value, arg_spec_cache, seen_known_bases
    )


def _iter_base_type_values_from_simple(
    value: SimpleType, arg_spec_cache: ArgSpecCache, seen_known_bases: frozenset[int]
) -> Iterator[TypedValue | AnyValue]:
    if isinstance(value, KnownValue):
        base_id = id(value.val)
        if base_id in seen_known_bases:
            return
        yield from _iter_base_type_values(
            arg_spec_cache._type_from_base(value.val),
            arg_spec_cache,
            seen_known_bases | {base_id},
        )
        return
    if isinstance(value, SyntheticClassObjectValue):
        yield from _iter_base_type_values(
            value.class_type, arg_spec_cache, seen_known_bases
        )
        return
    if isinstance(value, SequenceValue) and value.typ is tuple:
        yield value
        return
    if isinstance(value, GenericValue) and value.typ is tuple:
        yield SequenceValue(tuple, [(True, value.args[0])])
        return
    if isinstance(value, TypedValue):
        yield value
        return
    if isinstance(value, SubclassValue):
        if isinstance(value.typ, TypedValue):
            yield value.typ
        return
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        if isinstance(value, AnyValue):
            yield value
        return
    assert_never(value)


def _default_type_argument_for_param(
    type_param: TypeParam, substitutions: TypeVarMap, checker: "Checker"
) -> Value:
    if type_param.default is not None:
        default = type_param.default
        if isinstance(default, KnownValue):
            default = type_from_runtime(
                default.val, ctx=checker.arg_spec_cache.default_context
            )
        return default.substitute_typevars(substitutions)
    return type_param_to_value(type_param)


def _add_runtime_declared_symbols(typ: type, symbols: dict[str, ClassSymbol]) -> None:
    class_dict = safe_getattr(typ, "__dict__", None)
    namedtuple_fields = frozenset(_runtime_namedtuple_field_names(typ))
    runtime_dataclass_fields = {
        record.field_name: record.field_info
        for record in _get_runtime_dataclass_fields(typ)
    }
    for name in namedtuple_fields:
        raw_value = (
            _runtime_member_value(class_dict[name], typ)
            if isinstance(class_dict, Mapping) and name in class_dict
            else None
        )
        property_value = raw_value if isinstance(raw_value, KnownValue) else None
        annotation = _value_from_runtime_annotation(
            get_namedtuple_field_annotation(typ, name), typ
        )
        symbols[name] = ClassSymbol(
            annotation=annotation,
            qualifiers=frozenset({Qualifier.ReadOnly}),
            is_instance_only=True,
            property_info=(
                _runtime_property_info(property_value.val, typ)
                if property_value is not None
                and isinstance(property_value.val, property)
                else None
            ),
            initializer=raw_value,
        )
    try:
        if sys.version_info >= (3, 14):
            annotations = get_annotations(typ, format=Format.FORWARDREF)
        else:
            annotations = inspect.get_annotations(typ)  # pragma: no cover
    except Exception:
        annotations = safe_getattr(typ, "__annotations__", None)
    if isinstance(annotations, Mapping):
        for name, annotation in annotations.items():
            if not isinstance(name, str):
                continue
            if name in namedtuple_fields:
                continue
            symbol = _symbol_from_runtime_annotation(annotation, typ)
            is_instance_only = (
                not symbol.is_classvar
                and not symbol.is_initvar
                and (
                    name in runtime_dataclass_fields
                    or not isinstance(class_dict, Mapping)
                    or name not in class_dict
                )
            )
            symbols[name] = replace(
                symbol,
                is_instance_only=is_instance_only,
                dataclass_field=runtime_dataclass_fields.get(name),
            )
    if isinstance(class_dict, Mapping):
        for name, raw_value in class_dict.items():
            if not isinstance(name, str):
                continue
            if name in namedtuple_fields:
                continue
            existing = symbols.get(name)
            deprecation_message = _runtime_deprecation_message(raw_value)
            if (
                existing is not None
                and not existing.is_classvar
                and not existing.is_initvar
            ):
                is_property = existing.property_info is not None
                function_decorators = set(existing.function_decorators)
                is_method = existing.is_method
            else:
                is_property = isinstance(raw_value, property)
                function_decorators = set()
                maybe_func = raw_value
                is_staticmethod = isinstance(raw_value, staticmethod)
                is_classmethod = isinstance(raw_value, classmethod)
                if is_staticmethod:
                    function_decorators.add(FunctionDecorator.staticmethod)
                    maybe_func = raw_value.__func__
                if is_classmethod:
                    function_decorators.add(FunctionDecorator.classmethod)
                    maybe_func = raw_value.__func__
                is_method = _is_runtime_method_member(
                    raw_value,
                    is_property=is_property,
                    is_staticmethod=is_staticmethod,
                    is_classmethod=is_classmethod,
                )
                if safe_getattr(maybe_func, "__final__", False):
                    function_decorators.add(FunctionDecorator.final)
                if safe_getattr(maybe_func, "__isabstractmethod__", False):
                    function_decorators.add(FunctionDecorator.abstractmethod)
                if safe_getattr(maybe_func, "__override__", False):
                    function_decorators.add(FunctionDecorator.override)
            qualifiers = set(existing.qualifiers if existing is not None else ())
            if _is_runtime_member_final(raw_value):
                qualifiers.add(Qualifier.Final)
            symbols[name] = ClassSymbol(
                annotation=existing.annotation if existing is not None else None,
                qualifiers=frozenset(qualifiers),
                is_instance_only=(
                    existing.is_instance_only if existing is not None else False
                ),
                is_method=is_method,
                deprecation_message=deprecation_message,
                function_decorators=frozenset(function_decorators),
                property_info=(
                    _runtime_property_info(raw_value, typ) if is_property else None
                ),
                initializer=_runtime_member_value(raw_value, typ),
                dataclass_field=(
                    existing.dataclass_field if existing is not None else None
                ),
            )


def _runtime_member_value(raw_value: object, owner: type) -> Value:
    value = KnownValue(raw_value)
    orig_class = safe_getattr(raw_value, "__orig_class__", None)
    origin = get_origin(orig_class)
    args = get_args(orig_class)
    if not isinstance(origin, type) or not args:
        return value
    runtime_params = safe_getattr(origin, "__parameters__", ())
    if not isinstance(runtime_params, tuple) or not runtime_params:
        return value

    ctx = _RuntimeAnnotationsContext(owner)
    with ctx.suppress_errors():
        type_params = tuple(make_type_param(param, ctx) for param in runtime_params)
        type_arguments = tuple(type_from_runtime(arg, ctx=ctx) for arg in args)
    matched = match_typevar_arguments(type_params, type_arguments)
    if matched is None:
        return value

    typevars = TypeVarMap()
    params_by_typevar = {type_param.typevar: type_param for type_param in type_params}
    for runtime_typevar, arg_value in matched:
        typevars = typevars.with_value(params_by_typevar[runtime_typevar], arg_value)
    return KnownValueWithTypeVars(raw_value, typevars)


def _is_runtime_method_member(
    raw_value: object, *, is_property: bool, is_staticmethod: bool, is_classmethod: bool
) -> bool:
    if not is_property and (is_staticmethod or is_classmethod):
        return True
    if inspect.isfunction(raw_value):
        return True
    has_objclass = safe_getattr(raw_value, "__objclass__", None) is not None
    if inspect.ismethoddescriptor(raw_value) and (
        has_objclass or safe_getattr(raw_value, "func_code", None) is not None
    ):
        return True
    return False


def _is_runtime_member_final(raw_value: object) -> bool:
    if safe_getattr(raw_value, "__final__", False):
        return True
    if isinstance(raw_value, property):
        return any(
            safe_getattr(accessor, "__final__", False)
            for accessor in (raw_value.fget, raw_value.fset, raw_value.fdel)
            if accessor is not None
        )
    if isinstance(raw_value, (classmethod, staticmethod)):
        return safe_getattr(
            safe_getattr(raw_value, "__func__", None), "__final__", False
        )
    return False


def _runtime_property_info(raw_value: property, owner: type) -> PropertyInfo:
    getter_type: Value
    if raw_value.fget is None:
        getter_type = AnyValue(AnySource.inference)
    else:
        getter_annotations = safe_getattr(raw_value.fget, "__annotations__", None)
        if isinstance(getter_annotations, Mapping) and "return" in getter_annotations:
            getter_type = _value_from_runtime_annotation(
                getter_annotations["return"], owner
            )
        else:
            getter_type = AnyValue(AnySource.unannotated)

    setter_type: Value | None = None
    if raw_value.fset is not None:
        try:
            parameters = list(inspect.signature(raw_value.fset).parameters.values())
        except Exception:
            setter_type = AnyValue(AnySource.inference)
        else:
            value_param = parameters[1] if len(parameters) >= 2 else None
            if value_param is None or value_param.annotation is inspect.Parameter.empty:
                setter_type = AnyValue(AnySource.unannotated)
            else:
                setter_type = _value_from_runtime_annotation(
                    value_param.annotation, owner
                )

    return PropertyInfo(
        getter_type=getter_type,
        setter_type=setter_type,
        getter_deprecation=_accessor_deprecation_message(raw_value.fget),
        setter_deprecation=_accessor_deprecation_message(raw_value.fset),
    )


def _runtime_deprecation_message(raw_value: object) -> str | None:
    if isinstance(raw_value, property):
        return None
    if isinstance(raw_value, (classmethod, staticmethod)):
        return _accessor_deprecation_message(raw_value.__func__)
    return _accessor_deprecation_message(raw_value)


def _accessor_deprecation_message(accessor: object | None) -> str | None:
    deprecated = safe_getattr(accessor, "__deprecated__", None)
    return deprecated if isinstance(deprecated, str) else None


_CLASS_SYMBOL_ALLOWED_QUALIFIERS = frozenset(
    {Qualifier.ClassVar, Qualifier.Final, Qualifier.ReadOnly, Qualifier.InitVar}
)


def _symbol_from_runtime_annotation(annotation: object, owner: type) -> ClassSymbol:
    ctx = _RuntimeAnnotationsContext(owner)
    with ctx.suppress_errors():
        expr = annotation_expr_from_runtime(annotation, ctx=ctx)
        typ, qualifiers = expr.maybe_unqualify(_CLASS_SYMBOL_ALLOWED_QUALIFIERS)
    return ClassSymbol(
        annotation=(
            typ if typ is not None else AnyValue(AnySource.incomplete_annotation)
        ),
        qualifiers=frozenset(qualifiers),
    )


def _value_from_runtime_annotation(annotation: object, owner: type) -> Value:
    ctx = _RuntimeAnnotationsContext(owner)
    with ctx.suppress_errors():
        expr = annotation_expr_from_runtime(annotation, ctx=ctx)
        typ, _ = expr.maybe_unqualify(set())
    return typ if typ is not None else AnyValue(AnySource.incomplete_annotation)


def _runtime_namedtuple_field_names(typ: type) -> tuple[str, ...]:
    if not is_namedtuple_class(typ):
        return ()
    fields_obj = safe_getattr(typ, "_fields", None)
    if not isinstance(fields_obj, tuple):
        annos = safe_getattr(typ, "__annotations__", None)
        if isinstance(annos, Mapping):
            return tuple(name for name in annos if isinstance(name, str))
        return ()
    return tuple(name for name in fields_obj if isinstance(name, str))
