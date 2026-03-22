"""Helpers for computing raw ``TypeObject`` ancestry and symbol metadata."""

import inspect
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING, replace

from typing_extensions import assert_never

if sys.version_info >= (3, 14):
    from annotationlib import Format, get_annotations

from pycroscope.checker import Checker

from .annotations import (
    _RuntimeAnnotationsContext,
    annotation_expr_from_runtime,
    type_from_runtime,
)
from .arg_spec import ArgSpecCache, GenericBases
from .input_sig import coerce_paramspec_specialization_to_input_sig
from .safe import is_namedtuple_class, safe_getattr
from .type_object import (
    DataclassFieldRecord,
    MroValue,
    _class_key_from_value,
    class_keys_match,
    merge_declared_symbol,
)
from .value import (
    AnySource,
    AnyValue,
    ClassSymbol,
    DataclassFieldInfo,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    ParamSpecParam,
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
    TypeVarLike,
    UnboundMethodValue,
    Value,
    get_namedtuple_field_annotation,
    replace_fallback,
    type_param_to_value,
)

_SyntheticGenericBases = dict[type | str, dict[TypeVarLike, Value]]


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


def _get_synthetic_dataclass_fields(
    checker: Checker, synthetic_class: SyntheticClassObjectValue
) -> tuple[DataclassFieldRecord, ...]:
    if not synthetic_class.is_dataclass:
        return ()

    ordered: list[str] = []
    records_by_name: dict[str, DataclassFieldRecord] = {}
    for base in synthetic_class.base_classes:
        for base_value in _iter_base_type_values(base, checker._arg_spec_cache):
            base_type_object = checker.make_type_object(base_value.typ)
            for record in base_type_object.get_dataclass_fields():
                if record.field_name not in records_by_name:
                    ordered.append(record.field_name)
                records_by_name[record.field_name] = record

    local_fields = synthetic_class.dataclass_field_order
    type_object = checker.make_type_object(synthetic_class.class_type.typ)
    declared_symbols = type_object.get_declared_symbols()
    if not local_fields:
        local_fields = tuple(
            name
            for name, symbol in declared_symbols.items()
            if symbol.dataclass_field is not None
        )
    class_type = synthetic_class.class_type
    if not isinstance(class_type, TypedValue):
        return tuple(records_by_name[name] for name in ordered)
    for field_name in local_fields:
        symbol = type_object.get_declared_symbol(field_name)
        if symbol is None:
            continue
        record = DataclassFieldRecord(
            field_name=field_name,
            field_info=(
                symbol.dataclass_field
                if symbol.dataclass_field is not None
                else DataclassFieldInfo()
            ),
        )
        if field_name not in records_by_name:
            ordered.append(field_name)
        records_by_name[field_name] = record
    return tuple(records_by_name[name] for name in ordered)


def _get_direct_mro_bases(
    checker: Checker, typ: type | str
) -> tuple[list[type | str], tuple[MroValue, ...]]:
    direct_base_keys = _get_direct_mro_base_keys(checker, typ)
    if not direct_base_keys:
        return direct_base_keys, ()
    generic_bases = _get_generic_bases_for_class_definition(checker, typ)
    tuple_base = checker._namedtuple_tuple_base(typ)
    direct_base_values = tuple(
        _specialize_mro_base_value(
            checker, base_key, generic_bases=generic_bases, tuple_base=tuple_base
        )
        for base_key in direct_base_keys
    )
    return direct_base_keys, direct_base_values


def _get_direct_mro_base_keys(checker: Checker, typ: type | str) -> list[type | str]:
    direct_bases: list[type | str] = []
    synthetic_class = checker.get_synthetic_class(typ)
    if synthetic_class is not None:
        for base in synthetic_class.base_classes:
            for base_value in _iter_base_type_values(base, checker.arg_spec_cache):
                _append_unique_class_key(direct_bases, base_value.typ)
    if not direct_bases and isinstance(typ, type):
        for base in safe_getattr(typ, "__bases__", ()):
            if isinstance(base, type):
                _append_unique_class_key(direct_bases, base)
    elif synthetic_class is None:
        stub_bases = checker.ts_finder.get_bases_for_value(TypedValue(typ)) or []
        for base in stub_bases:
            for base_value in _iter_base_type_values(base, checker.arg_spec_cache):
                _append_unique_class_key(direct_bases, base_value.typ)
    if not direct_bases and typ is not object and typ != "builtins.object":
        direct_bases.append(object)
    return direct_bases


def _specialize_mro_base_value(
    checker: Checker,
    base_key: type | str,
    *,
    generic_bases: GenericBases,
    tuple_base: SequenceValue | None = None,
) -> MroValue:
    if tuple_base is not None and class_keys_match(base_key, tuple):
        return tuple_base
    type_params = tuple(checker.get_type_parameters(base_key))
    if not type_params:
        return TypedValue(base_key)
    tv_map = generic_bases.get(base_key, {})
    substitutions: dict[TypeVarLike, Value] = {}
    args: list[Value] = []
    for type_param in type_params:
        arg = tv_map.get(type_param.typevar)
        if arg is None:
            arg = _default_type_argument_for_param(type_param, substitutions, checker)
        else:
            arg = arg.substitute_typevars(substitutions)
        if isinstance(type_param, ParamSpecParam):
            arg = coerce_paramspec_specialization_to_input_sig(arg)
        substitutions[type_param.typevar] = arg
        args.append(arg)
    if (
        base_key is tuple
        and len(args) == 1
        and isinstance(args[0], SequenceValue)
        and args[0].typ is tuple
    ):
        return args[0]
    return GenericValue(base_key, args)


def _self_mro_value(
    typ: type | str,
    *,
    declared_type_params: Sequence[TypeParam],
    direct_base_values: Sequence[Value],
    tuple_base: SequenceValue | None,
) -> TypedValue:
    if tuple_base is not None:
        return tuple_base
    if declared_type_params:
        return GenericValue(
            typ,
            [type_param_to_value(type_param) for type_param in declared_type_params],
        )
    if len(direct_base_values) == 1 and isinstance(
        direct_base_values[0], (GenericValue, SequenceValue)
    ):
        return direct_base_values[0]
    return TypedValue(typ)


def _mro_substitution_map_for_base(
    checker: Checker, base_value: Value, type_params: Sequence[TypeParam]
) -> dict[TypeVarLike, Value]:
    if not type_params:
        return {}
    if isinstance(base_value, SequenceValue) and base_value.typ is tuple:
        generic_args: Sequence[Value] = (base_value,)
    elif isinstance(base_value, GenericValue):
        generic_args = base_value.args
    else:
        generic_args = ()
    specialized_args = checker.arg_spec_cache._specialize_generic_type_params(
        type_params, generic_args
    )
    substitutions: dict[TypeVarLike, Value] = {}
    for type_param, arg in zip(type_params, specialized_args):
        if isinstance(type_param, ParamSpecParam):
            arg = coerce_paramspec_specialization_to_input_sig(arg)
        substitutions[type_param.typevar] = arg.substitute_typevars(substitutions)
    return substitutions


def _specialize_mro_tail_for_base(
    checker: Checker,
    base_value: MroValue,
    type_params: Sequence[TypeParam],
    tail: Sequence[MroValue],
) -> tuple[MroValue, ...]:
    substitutions = _mro_substitution_map_for_base(checker, base_value, type_params)
    if not substitutions:
        return tuple(tail)
    return tuple(value.substitute_typevars(substitutions) for value in tail)


def _merge_mro_value_sequences(
    checker: Checker, sequences: Sequence[Sequence[MroValue]]
) -> tuple[MroValue, ...]:
    pending = [list(sequence) for sequence in sequences if sequence]
    result: list[MroValue] = []
    while pending:
        candidate: MroValue | None = None
        candidate_key: type | str | None = None
        for sequence in pending:
            head = sequence[0]
            head_key = _class_key_from_value(head)
            if head_key is None:
                continue
            if any(
                any(
                    tail_key is not None and class_keys_match(head_key, tail_key)
                    for tail_key in (_class_key_from_value(tail) for tail in other[1:])
                )
                for other in pending
            ):
                continue
            candidate = head
            candidate_key = head_key
            break
        if candidate is None:
            candidate = pending[0][0]
            candidate_key = _class_key_from_value(candidate)
        result.append(candidate)
        new_pending: list[list[MroValue]] = []
        for sequence in pending:
            if sequence and candidate_key is not None:
                head_key = _class_key_from_value(sequence[0])
                if head_key is not None and class_keys_match(head_key, candidate_key):
                    sequence = sequence[1:]
            elif sequence and sequence[0] == candidate:
                sequence = sequence[1:]
            if sequence:
                new_pending.append(sequence)
        pending = new_pending
    return tuple(result)


def _get_generic_bases_for_class_definition(
    checker: Checker, typ: type | str
) -> GenericBases:
    synthetic_bases = checker._get_synthetic_generic_bases(typ)
    if synthetic_bases is None:
        declared_type_params = tuple(checker.get_type_parameters(typ))
    else:
        declared_type_params = checker._get_synthetic_declared_type_params(typ)
    generic_bases = checker.arg_spec_cache.get_generic_bases(
        typ, [type_param_to_value(type_param) for type_param in declared_type_params]
    )
    merged: _SyntheticGenericBases = {
        base: dict(tv_map) for base, tv_map in generic_bases.items()
    }
    if synthetic_bases is None:
        checker._augment_namedtuple_generic_bases(typ, merged, {})
        return merged

    substitution_map = {
        type_param.typevar: type_param_to_value(type_param)
        for type_param in declared_type_params
    }
    if declared_type_params:
        merged.setdefault(typ, {})
        direct_base_map: dict[TypeVarLike, Value] = merged[typ]
        for type_param in declared_type_params:
            direct_base_map[type_param.typevar] = substitution_map[type_param.typevar]
    for base, tv_map in synthetic_bases.items():
        substituted_tv_map = {
            tv: value.substitute_typevars(substitution_map)
            for tv, value in tv_map.items()
        }
        merged.setdefault(base, {}).update(substituted_tv_map)
    checker._augment_namedtuple_generic_bases(typ, merged, substitution_map)
    return merged


def _append_unique_class_key(keys: list[type | str], key: type | str) -> None:
    if any(class_keys_match(existing, key) for existing in keys):
        return
    keys.append(key)


def _iter_base_type_values(
    value: Value,
    arg_spec_cache: ArgSpecCache | None,
    seen_known_bases: frozenset[int] = frozenset(),
) -> Iterator[TypedValue]:
    if isinstance(value, PartialValue):
        if value.operation is not PartialValueOperation.SUBSCRIPT:
            return
        root: Value = value.root
        if isinstance(root, SyntheticClassObjectValue):
            root = root.class_type
        elif isinstance(root, KnownValue):
            if arg_spec_cache is not None:
                root = arg_spec_cache._type_from_base(root.val)
            elif isinstance(root.val, type):
                root = TypedValue(root.val)
            else:
                return
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
    value: SimpleType,
    arg_spec_cache: ArgSpecCache | None,
    seen_known_bases: frozenset[int],
) -> Iterator[TypedValue]:
    if isinstance(value, KnownValue):
        if arg_spec_cache is not None:
            base_id = id(value.val)
            if base_id in seen_known_bases:
                return
            yield from _iter_base_type_values(
                arg_spec_cache._type_from_base(value.val),
                arg_spec_cache,
                seen_known_bases | {base_id},
            )
        elif isinstance(value.val, type):
            yield TypedValue(value.val)
        return
    if isinstance(value, SyntheticClassObjectValue):
        yield from _iter_base_type_values(
            value.class_type, arg_spec_cache, seen_known_bases
        )
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
        return
    assert_never(value)


def _default_type_argument_for_param(
    type_param: TypeParam, substitutions: dict[TypeVarLike, Value], checker: "Checker"
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
            KnownValue(class_dict[name])
            if isinstance(class_dict, Mapping) and name in class_dict
            else None
        )
        annotation = _value_from_runtime_annotation(
            get_namedtuple_field_annotation(typ, name), typ
        )
        symbols[name] = ClassSymbol(
            annotation=annotation,
            qualifiers=frozenset({Qualifier.ReadOnly}),
            is_instance_only=True,
            property_info=(
                _runtime_property_info(raw_value.val, typ)
                if raw_value is not None and isinstance(raw_value.val, property)
                else None
            ),
            initializer=raw_value,
        )
    try:
        if sys.version_info >= (3, 14):
            annotations = get_annotations(typ, format=Format.FORWARDREF)
        else:
            annotations = inspect.get_annotations(typ)
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
            if (
                existing is not None
                and not existing.is_classvar
                and not existing.is_initvar
            ):
                is_property = existing.property_info is not None
                is_staticmethod = existing.is_staticmethod
                is_classmethod = existing.is_classmethod
                is_method = existing.is_method
            else:
                is_property = isinstance(raw_value, property)
                is_staticmethod = isinstance(raw_value, staticmethod)
                is_classmethod = isinstance(raw_value, classmethod)
                is_method = (
                    (not is_property and (is_staticmethod or is_classmethod))
                    or inspect.isfunction(raw_value)
                    or inspect.ismethoddescriptor(raw_value)
                )
            symbols[name] = ClassSymbol(
                annotation=existing.annotation if existing is not None else None,
                qualifiers=existing.qualifiers if existing is not None else frozenset(),
                is_instance_only=(
                    existing.is_instance_only if existing is not None else False
                ),
                is_method=is_method,
                is_classmethod=is_classmethod,
                is_staticmethod=is_staticmethod,
                property_info=(
                    _runtime_property_info(raw_value, typ) if is_property else None
                ),
                initializer=KnownValue(raw_value),
                dataclass_field=(
                    existing.dataclass_field if existing is not None else None
                ),
            )


def _add_synthetic_declared_symbols(
    declared_symbols: Mapping[str, ClassSymbol], symbols: dict[str, ClassSymbol]
) -> None:
    for name, symbol in declared_symbols.items():
        symbols[name] = merge_declared_symbol(symbols.get(name), symbol)


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

    return PropertyInfo(getter_type=getter_type, setter_type=setter_type)


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
        return ()
    return tuple(name for name in fields_obj if isinstance(name, str))
