"""

An object that represents a type.

"""

import ast
import collections.abc
import enum
import functools
import inspect
import sys
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, replace
from types import FunctionType
from typing import TYPE_CHECKING, Literal, get_origin
from unittest import mock

from typing_extensions import assert_never

import pycroscope

if TYPE_CHECKING:
    from .relations import Relation

from .annotations import make_type_param, type_from_runtime
from .input_sig import AnySig, FullSignature, InputSigValue
from .options import PyObjectSequenceOption
from .relations import (
    infer_positional_generic_typevar_map,
    intersect_values,
    subtract_values,
    translate_generic_typevar_map,
)
from .safe import (
    is_direct_namedtuple_class,
    is_instance_of_typing_name,
    is_namedtuple_class,
    is_typing_name,
    not_none,
    safe_getattr,
    safe_isinstance,
    safe_issubclass,
)
from .signature import (
    BoundMethodSignature,
    CallContext,
    Impl,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    mark_ellipsis_style_any_tail_parameters,
)
from .value import (
    NO_RETURN_VALUE,
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
    DataclassInfo,
    DataclassTransformInfo,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    ParamSpecParam,
    PredicateValue,
    PropertyInfo,
    Qualifier,
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
    TypeVarMap,
    TypeVarParam,
    TypeVarTupleBindingValue,
    TypeVarTupleParam,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    _has_nested_self_typevar,
    _iter_typevar_map_items,
    _typevar_map_from_varlike_pairs,
    _with_typevar_map_value,
    default_value_for_type_param,
    freshen_typevars_for_inference,
    get_single_typevartuple_param,
    get_tv_map,
    match_typevar_arguments,
    receiver_to_self_type,
    replace_fallback,
    replace_known_sequence_value,
    set_self,
    shield_nested_self_typevars,
    stringify_object,
    substitute_typevartuple_binding,
    tuple_members_from_value,
    type_param_to_value,
    typevartuple_binding_to_generic_args,
    typevartuple_binding_to_tuple_value,
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

_BaseProvider = Callable[[type], set[type]]


class AdditionalBaseProviders(PyObjectSequenceOption[_BaseProvider]):
    """Sets functions that provide additional (virtual) base classes for a class.
    These are used for the purpose of type checking.

    For example, if the following is configured to be used as a base provider:

        def provider(typ: type) -> Set[type]:
            if typ is B:
                return {A}
            return set()

    Then to the type checker `B` is a subclass of `A`.

    """

    name = "additional_base_providers"


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
class ResolvedAttribute:
    value: Value
    # Legacy compatibility view for callers that still expect the old
    # declared/raw split from TypeObjectAttribute.
    declared_value: Value
    raw_value: Value
    # Legacy compatibility symbol view. Prefer runtime_symbol and
    # typeshed_symbol when source provenance matters.
    symbol: ClassSymbol
    runtime_symbol: ClassSymbol | None
    typeshed_symbol: ClassSymbol | None
    owner: "TypeObject"
    is_property: bool
    property_has_setter: bool
    is_metaclass_owner: bool

    @property
    def override_value(self) -> Value:
        if self.symbol.is_property:
            return self.raw_value
        if self.symbol.is_method:
            return normalize_synthetic_descriptor_attribute(
                self.raw_value,
                is_self_returning_classmethod=self.symbol.returns_self_on_class_access,
            )
        if self.symbol.annotation is not None:
            return self.declared_value
        return self.raw_value


TypeObjectAttribute = ResolvedAttribute


@dataclass(frozen=True, kw_only=True)
class AttributePolicy:
    on_class: bool = False
    is_special_lookup: bool = False
    receiver_value: Value | None = None
    visitor: "pycroscope.name_check_visitor.NameCheckVisitor | None" = None
    node: ast.AST | None = None


@dataclass(frozen=True)
class SelectedAttribute:
    mro_index: int
    owner: "TypeObject"
    symbol: ClassSymbol
    is_metaclass_owner: bool


@dataclass(frozen=True)
class SpecializedAttribute:
    selected: SelectedAttribute
    annotation: Value | None
    initializer: Value | None
    property_info: PropertyInfo | None

    def as_symbol(self) -> ClassSymbol:
        return replace(
            self.selected.symbol,
            annotation=self.annotation,
            initializer=self.initializer,
            property_info=self.property_info,
        )


@dataclass(frozen=True)
class MergedAttribute:
    """Attribute view derived from merging runtime and typeshed data.

    This is the input to descriptor resolution after runtime/typeshed
    candidates have been selected and specialized independently.
    """

    owner: "TypeObject"
    runtime_symbol: ClassSymbol | None
    typeshed_symbol: ClassSymbol | None
    annotation: Value | None
    initializer: Value | None
    property_info: PropertyInfo | None
    is_method: bool
    is_classmethod: bool
    is_staticmethod: bool
    is_classvar: bool
    is_initvar: bool
    returns_self_on_class_access: bool
    is_metaclass_owner: bool


def get_mro(typ: type) -> Sequence[type]:
    try:
        return inspect.getmro(typ)
    except AttributeError:
        # It's not actually a class.
        return []


def _extract_runtime_protocol_members(typ: type) -> set[str]:
    if (
        typ is object
        or is_typing_name(typ, "Generic")
        or is_typing_name(typ, "Protocol")
    ):
        return set()
    members: set[str] = set(typ.__dict__) - EXCLUDED_PROTOCOL_MEMBERS
    # Starting in 3.10 __annotations__ always exists on types.
    if sys.version_info >= (3, 10) or hasattr(typ, "__annotations__"):
        members |= set(typ.__annotations__)
    return members


def _get_additional_bases(
    checker: "pycroscope.checker.Checker", typ: type
) -> set[type | str]:
    bases: set[type | str] = set()
    for provider in checker.options.get_value_for(AdditionalBaseProviders):
        bases |= provider(typ)
    return bases


MroValue = TypedValue | AnyValue


def direct_bases_from_values(
    base_values: Sequence[Value], checker: "pycroscope.checker.Checker"
) -> tuple[MroValue, ...]:
    import pycroscope.type_object_builder as type_object_builder

    direct_bases = [
        converted
        for base in base_values
        for converted in type_object_builder._iter_base_type_values(
            base, checker.arg_spec_cache
        )
    ]
    return tuple(_replace_invalid_bases(direct_bases or [TypedValue(object)]))


def _iter_base_type_objects(
    base_value: Value, checker: "pycroscope.checker.Checker"
) -> Iterator["TypeObject"]:
    import pycroscope.type_object_builder as type_object_builder

    for converted in type_object_builder._iter_base_type_values(
        base_value, checker.arg_spec_cache
    ):
        if isinstance(converted, AnyValue):
            continue
        yield converted.get_type_object(checker)


@dataclass(frozen=True)
class MroEntry:
    tobj: "TypeObject | None"  # None only for is_any=True entries
    tv_map: TypeVarMap
    value: MroValue | None
    """Value to store directly. Should be set only if it's Any or a SequenceValue."""
    is_virtual: bool
    """A virtual base is a base that is not part of the MRO of this class at runtime,
    but that is included for typing purposes.

    For example, list inherits directly from object at runtime, but in typeshed it is
    declared as inheriting from MutableSequence. Therefore, MutableSequence (and its bases)
    should be marked as virtual bases.
    """

    @property
    def is_any(self) -> bool:
        return isinstance(self.value, AnyValue)

    def __repr__(self) -> str:
        if self.is_any:
            return "Any"
        assert self.tobj is not None
        result = repr(self.tobj.typ)
        if self.is_virtual:
            result = f"~{result}"
        if self.tv_map:
            args_str = ", ".join(
                f"{tv.__name__}={value}"
                for tv, value in _iter_typevar_map_items(self.tv_map)
            )
            result += f"[{args_str}]"
        return result

    def substitute_typevars(self, typevars: TypeVarMap) -> "MroEntry":
        if self.is_any:
            return self
        assert self.tobj is not None
        substituted_tv_map = TypeVarMap()
        for param in self.tobj.get_declared_type_params():
            value = self.tv_map.get_value(param)
            if value is None:
                continue
            substituted_tv_map = substituted_tv_map.with_value(
                param, _substitute_type_param_value(param, value, typevars)
            )
        if self.value is None:
            new_value = None
        else:
            new_value = self.value.substitute_typevars(typevars)
        return replace(self, tv_map=substituted_tv_map, value=new_value)

    def get_mro_value(self) -> MroValue:
        if self.value is not None:
            return self.value
        assert self.tobj is not None
        if self.tv_map:
            return GenericValue(
                self.tobj.typ, _generic_args_from_tv_map(self.tobj, self.tv_map)
            )
        return TypedValue(self.tobj.typ)


ANY_MRO_ENTRY = MroEntry(
    None, TypeVarMap(), value=AnyValue(AnySource.inference), is_virtual=False
)


def _generic_args_from_tv_map(tobj: "TypeObject", tv_map: TypeVarMap) -> list[Value]:
    args: list[Value] = []
    for param in tobj.get_declared_type_params():
        arg = tv_map.get_value(param)
        assert arg is not None
        if isinstance(param, TypeVarTupleParam):
            binding = tv_map.get_typevartuple(param)
            assert binding is not None
            if tobj.typ is tuple:
                args.append(typevartuple_binding_to_tuple_value(binding))
            else:
                args.extend(typevartuple_binding_to_generic_args(binding))
            continue
        args.append(arg)
    return args


def _substitute_type_param_value(
    type_param: TypeParam | None, value: Value, typevars: TypeVarMap
) -> Value:
    if not isinstance(type_param, TypeVarTupleParam):
        return value.substitute_typevars(typevars)
    if isinstance(value, TypeVarTupleBindingValue):
        return TypeVarTupleBindingValue(
            substitute_typevartuple_binding(value.binding, typevars)
        )
    normalized_value = replace_known_sequence_value(value)
    if isinstance(normalized_value, SequenceValue) and normalized_value.typ is tuple:
        return TypeVarTupleBindingValue(
            substitute_typevartuple_binding(normalized_value.members, typevars)
        )
    return value.substitute_typevars(typevars)


@dataclass(frozen=True)
class DataclassFieldRecord:
    field_name: str
    field_info: DataclassFieldInfo


@dataclass(frozen=True)
class NamedTupleField:
    name: str
    typ: Value
    default: Value | None


class TypeObject:
    """Represents one logical type, with lazy and incrementally populated metadata."""

    typ: type | str
    _checker: "pycroscope.checker.Checker"
    _direct_bases: tuple[MroValue, ...] | None
    _mro: Sequence[MroEntry] | None
    _declared_type_params: tuple[TypeParam, ...] | None
    _is_final: bool | None
    _is_disjoint_base: bool | None
    _directly_disjoint_base: bool | None
    _is_protocol: bool | None
    _protocol_members: set[str] | None
    _synthetic_declared_symbols: dict[str, ClassSymbol] | None
    _declared_symbols: MutableMapping[str, ClassSymbol] | None
    _dataclass_fields: tuple[DataclassFieldRecord, ...] | None
    _direct_dataclass_info: DataclassInfo | None
    _direct_dataclass_transform_info: DataclassTransformInfo | None
    _virtual_bases: list[MroValue] | None
    _virtual_symbols: dict[str, ClassSymbol] | None
    _is_thrift_enum: bool | None
    _is_direct_namedtuple: bool | None
    _namedtuple_fields: Sequence[NamedTupleField] | None
    _computing_namedtuple_data: bool
    _resolving_synthetic_namedtuple_class: bool
    _is_universally_assignable: bool | None
    _protocol_positive_cache: dict[tuple[Value, Value], BoundsMap]
    _has_stubs: bool | None
    _metaclass: MroValue | None

    def __init__(self, checker: "pycroscope.checker.Checker", typ: type | str) -> None:
        self.typ = typ
        self._checker = checker
        self._synthetic_declared_symbols = None
        self._declared_symbols = None
        self._protocol_positive_cache = {}
        self._direct_bases = None
        self._mro = None
        self._declared_type_params = None
        self._is_final = None
        self._is_disjoint_base = None
        self._directly_disjoint_base = None
        self._is_protocol = None
        self._protocol_members = None
        self._dataclass_fields = None
        self._direct_dataclass_info = None
        self._direct_dataclass_transform_info = None
        self._virtual_bases = None
        self._virtual_symbols = None
        self._is_thrift_enum = None
        self._is_direct_namedtuple = None
        self._namedtuple_fields = None
        self._computing_namedtuple_data = False
        self._resolving_synthetic_namedtuple_class = False
        self._is_universally_assignable = None
        self._has_stubs = None
        self._metaclass = None

    def has_stubs(self) -> bool:
        if self._has_stubs is None:
            self._has_stubs = self._checker.ts_finder.has_stubs(self.typ)
        return self._has_stubs

    def has_any_base(self) -> bool:
        return any(entry.is_any for entry in self.get_mro())

    def _compute_declared_symbols(self) -> dict[str, ClassSymbol]:
        import pycroscope.type_object_builder as type_object_builder

        direct_symbols: dict[str, ClassSymbol] = {}
        if isinstance(self.typ, type):
            type_object_builder._add_runtime_declared_symbols(self.typ, direct_symbols)
        if self._virtual_symbols is not None:
            _add_synthetic_declared_symbols(self._virtual_symbols, direct_symbols)
        synthetic_symbols: Mapping[str, ClassSymbol] = (
            self.get_synthetic_declared_symbols()
        )
        _add_synthetic_declared_symbols(synthetic_symbols, direct_symbols)
        return direct_symbols

    def _compute_declared_type_params(self) -> tuple[TypeParam, ...]:
        # First try stubs
        ts_bases = self._checker.ts_finder.get_bases(self.typ)
        if ts_bases is not None:
            return tuple(_compute_type_params_from_bases(ts_bases))

        # If it's a runtime class, try that
        if isinstance(self.typ, type):
            return tuple(_compute_type_params_from_runtime(self.typ, self._checker))

        return tuple(_compute_type_params_from_bases(self.get_direct_bases()))

    def _compute_direct_bases(self) -> tuple[MroValue, ...]:
        if self.typ is object:
            return ()
        ts_bases = self._checker.ts_finder.get_bases(self.typ)
        if ts_bases is not None:
            if not ts_bases:
                return (TypedValue(object),)
            return tuple(_replace_invalid_bases(ts_bases))

        if isinstance(self.typ, type):
            return tuple(_extract_runtime_direct_bases(self.typ, self._checker))

        # TODO: raise error if direct bases are not initialized for synthetic types
        return (TypedValue(object),)

    def _compute_mro(self) -> list[MroEntry]:
        direct_bases = self.get_direct_bases()
        virtual_bases = self.get_virtual_bases()

        params = self.get_declared_type_params()
        if params:
            parent_tv_map = TypeVarMap()
            for param in params:
                parent_tv_map = parent_tv_map.with_value(
                    param, type_param_to_value(param)
                )
            self_args = [type_param_to_value(param) for param in params]
            self_value = GenericValue(self.typ, self_args)
        else:
            parent_tv_map = TypeVarMap()
            self_value = TypedValue(self.typ)
        self_mro_entry = _get_mro_entry(self_value, parent_tv_map, self._checker)
        child_mros = [
            _get_mro_from_mro_value(base, parent_tv_map, self._checker)
            for base in direct_bases
        ] + [
            _get_mro_from_mro_value(base, parent_tv_map, self._checker, virtual=True)
            for base in virtual_bases
        ]
        if isinstance(self.typ, str):
            child_mros = [
                _mark_direct_base_non_virtual(child_mro) for child_mro in child_mros
            ]
        result = _linearize_mros(self_mro_entry, child_mros)
        if isinstance(result, str):
            return [
                self_mro_entry,
                ANY_MRO_ENTRY,
                _make_object_mro_entry(self._checker),
            ]
        if isinstance(self.typ, type):
            runtime_mro = tuple(get_mro(self.typ))
            result = [
                _mark_entry_virtual_by_runtime_mro(entry, runtime_mro)
                for entry in result
            ]
        return result

    def _compute_is_final(self) -> bool:
        if isinstance(self.typ, str):
            return self._checker.ts_finder.is_final(self.typ)
        return self._checker.ts_finder.is_final(self.typ) or safe_getattr(
            self.typ, "__final__", False
        )

    def _compute_is_disjoint_base(self) -> bool:
        if isinstance(self.typ, type):
            class_dict = safe_getattr(self.typ, "__dict__", None)
            return isinstance(class_dict, Mapping) and bool(
                class_dict.get("__disjoint_base__", False)
            )
        return bool(self._directly_disjoint_base)

    def _compute_is_protocol(self) -> bool:
        if isinstance(self.typ, str):
            return any(
                (base_key := _class_key_from_value(base_value)) is not None
                and is_typing_name(base_key, "Protocol")
                for base_value in self.get_direct_bases()
            )
        return self._checker.ts_finder.is_protocol(self.typ) or (
            is_instance_of_typing_name(self.typ, "_ProtocolMeta")
            and safe_getattr(self.typ, "_is_protocol", False)
        )

    def _compute_protocol_members(self) -> set[str]:
        if not self.is_protocol():
            return set()
        return (
            self._get_protocol_members_contributed_by_self()
            | self._get_protocol_members_contributed_by_protocol_bases()
        )

    def _compute_dataclass_fields(self) -> tuple[DataclassFieldRecord, ...]:
        import pycroscope.type_object_builder as type_object_builder

        if self.get_direct_dataclass_info() is not None:
            ordered_names: list[str] = []
            records_by_name: dict[str, DataclassFieldRecord] = {}
            for base_value in self.get_direct_bases():
                if not isinstance(base_value, TypedValue):
                    continue
                for record in self._checker.make_type_object(
                    base_value.typ
                ).get_dataclass_fields():
                    if record.field_name not in records_by_name:
                        ordered_names.append(record.field_name)
                    records_by_name[record.field_name] = record
            for record in self.get_direct_dataclass_fields():
                if record.field_name not in records_by_name:
                    ordered_names.append(record.field_name)
                records_by_name[record.field_name] = record
            return tuple(records_by_name[name] for name in ordered_names)
        if isinstance(self.typ, str):
            return ()
        return type_object_builder._get_runtime_dataclass_fields(self.typ)

    def _get_synthetic_namedtuple_class(self) -> SyntheticClassObjectValue | None:
        if self._resolving_synthetic_namedtuple_class:
            return None
        self._resolving_synthetic_namedtuple_class = True
        try:
            synthetic_class = self._checker.get_synthetic_class(self.typ)
            if synthetic_class is None:
                return None
            if self._is_direct_namedtuple:
                return synthetic_class
            for base_value in self.get_direct_bases():
                if (
                    isinstance(base_value, TypedValue)
                    and self._checker.make_type_object(
                        base_value.typ
                    ).is_namedtuple_like()
                ):
                    return synthetic_class
            return None
        finally:
            self._resolving_synthetic_namedtuple_class = False

    def _namedtuple_fields_from_runtime(self, typ: type) -> tuple[NamedTupleField, ...]:
        fields: list[NamedTupleField] = []
        defaults = safe_getattr(typ, "_field_defaults", None)
        annos = safe_getattr(typ, "__annotations__", None)
        fields_obj = safe_getattr(typ, "_fields", None)
        if not isinstance(fields_obj, tuple):
            fields_obj = tuple(annos) if isinstance(annos, Mapping) else ()
        for field_name in fields_obj:
            if not isinstance(field_name, str):
                continue
            field_type = type_from_runtime(
                annos.get(field_name, object) if annos else object,
                visitor=self._checker,
                suppress_errors=True,
            )
            if field_type == TypedValue(object) and not annos:
                field_type = AnyValue(AnySource.unannotated)
            default = (
                KnownValue(defaults[field_name])
                if isinstance(defaults, Mapping) and field_name in defaults
                else None
            )
            fields.append(NamedTupleField(field_name, field_type, default))
        return tuple(fields)

    def _compute_synthetic_namedtuple_fields(
        self,
        synthetic_class: SyntheticClassObjectValue,
        *,
        include_inherited_fields: bool = True,
    ) -> tuple[NamedTupleField, ...]:
        inherited_fields = (
            tuple(self._iter_synthetic_namedtuple_base_fields(synthetic_class))
            if include_inherited_fields
            else ()
        )
        prefer_declared_type = not inherited_fields
        field_names = [field.name for field in inherited_fields]
        for field_name in self._get_synthetic_namedtuple_local_field_names(
            synthetic_class, has_inherited_fields=bool(inherited_fields)
        ):
            if field_name not in field_names:
                field_names.append(field_name)
        inherited_by_name = {field.name: field for field in inherited_fields}
        return tuple(
            inherited_by_name.get(
                field_name,
                NamedTupleField(
                    field_name,
                    self._get_synthetic_namedtuple_field_value(
                        synthetic_class,
                        field_name,
                        prefer_declared_type=prefer_declared_type,
                    )
                    or AnyValue(AnySource.inference),
                    None,
                ),
            )
            for field_name in field_names
        )

    def _iter_synthetic_namedtuple_base_fields(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> Iterator[NamedTupleField]:
        seen: set[str] = set()
        for base_value in self.get_direct_bases():
            if not isinstance(base_value, TypedValue):
                continue
            if class_keys_match(base_value.typ, self.typ):
                continue
            base_tobj = self._checker.make_type_object(base_value.typ)
            if not base_tobj.is_namedtuple_like():
                continue
            substitutions = (
                base_tobj.get_substitutions(base_value.args)
                if isinstance(base_value, GenericValue)
                else TypeVarMap()
            )
            for field in base_tobj.get_namedtuple_fields():
                if substitutions:
                    field = replace(
                        field,
                        typ=field.typ.substitute_typevars(substitutions),
                        default=(
                            field.default.substitute_typevars(substitutions)
                            if field.default is not None
                            else None
                        ),
                    )
                if field.name in seen:
                    continue
                seen.add(field.name)
                yield field

    def _get_synthetic_namedtuple_local_field_names(
        self, synthetic_class: SyntheticClassObjectValue, *, has_inherited_fields: bool
    ) -> tuple[str, ...]:
        if has_inherited_fields:
            return ()
        declared_symbols = self.get_synthetic_declared_symbols()
        allowed_names = {
            name
            for name, symbol in declared_symbols.items()
            if symbol.is_instance_only
            and not symbol.is_classvar
            and not symbol.is_initvar
        }
        ordered = [
            name
            for name, symbol in declared_symbols.items()
            if not symbol.is_method and (not allowed_names or name in allowed_names)
        ]
        for name in allowed_names:
            symbol = declared_symbols.get(name)
            if symbol is None or symbol.is_method or name in ordered:
                continue
            ordered.append(name)
        return tuple(ordered)

    def _get_synthetic_namedtuple_field_value(
        self,
        synthetic_class: SyntheticClassObjectValue,
        field_name: str,
        *,
        prefer_declared_type: bool,
    ) -> Value | None:
        symbol = self.get_synthetic_declared_symbols().get(field_name)
        if symbol is not None:
            if (
                prefer_declared_type
                and not symbol.is_classvar
                and not symbol.is_initvar
                and not symbol.is_method
            ):
                return symbol.get_declared_type()
            if symbol.initializer is not None and not (
                prefer_declared_type and not symbol.is_method
            ):
                return symbol.initializer
        for field in self._iter_synthetic_namedtuple_base_fields(synthetic_class):
            if field.name == field_name:
                return field.typ
        return None

    def _compute_is_thrift_enum(self) -> bool:
        return isinstance(self.typ, type) and hasattr(self.typ, "_VALUES_TO_NAMES")

    def _compute_virtual_bases(self) -> list[MroValue]:
        present_keys = {self.typ} | {
            val.typ for val in self.get_direct_bases() if isinstance(val, TypedValue)
        }
        return [
            TypedValue(key)
            for key in self._iter_candidate_virtual_bases()
            if key not in present_keys
        ]

    def _iter_candidate_virtual_bases(self) -> Iterator[type | str]:
        if isinstance(self.typ, type):
            yield from _get_additional_bases(self._checker, self.typ)
        if self._compute_is_thrift_enum():
            yield int

    def _compute_is_universally_assignable(self) -> bool:
        return isinstance(self.typ, type) and issubclass(self.typ, mock.NonCallableMock)

    def is_direct_namedtuple(self) -> bool:
        if self._is_direct_namedtuple is None:
            if self._computing_namedtuple_data:
                is_direct, _ = self._compute_namedtuple_data(
                    include_inherited_fields=False
                )
                return is_direct
            self._computing_namedtuple_data = True
            try:
                self._is_direct_namedtuple, self._namedtuple_fields = (
                    self._compute_namedtuple_data()
                )
            finally:
                self._computing_namedtuple_data = False
        return self._is_direct_namedtuple

    def get_namedtuple_fields(self) -> Sequence[NamedTupleField]:
        if self._namedtuple_fields is None:
            if self._computing_namedtuple_data:
                _, fields = self._compute_namedtuple_data(
                    include_inherited_fields=False
                )
                return fields
            self._computing_namedtuple_data = True
            try:
                self._is_direct_namedtuple, self._namedtuple_fields = (
                    self._compute_namedtuple_data()
                )
            finally:
                self._computing_namedtuple_data = False
        if (
            self._is_direct_namedtuple
            and self._namedtuple_fields
            and "__new__" not in self.get_synthetic_declared_symbols()
        ):
            _add_namedtuple_dunder_new_symbol(self, self._namedtuple_fields)
        return self._namedtuple_fields

    def set_namedtuple_fields(
        self, fields: Sequence[NamedTupleField], *, constructor_impl: Impl | None = None
    ) -> None:
        self._is_direct_namedtuple = True
        self._namedtuple_fields = tuple(fields)
        self._specialize_namedtuple_direct_bases()
        _add_namedtuple_dunder_new_symbol(
            self, self._namedtuple_fields, constructor_impl=constructor_impl
        )

    def set_is_direct_namedtuple(self, is_direct_namedtuple: bool) -> None:
        self._is_direct_namedtuple = is_direct_namedtuple
        if not is_direct_namedtuple and self._namedtuple_fields is None:
            self._invalidate_synthetic_state()

    def _specialize_namedtuple_direct_bases(self) -> None:
        if not self._namedtuple_fields or self._direct_bases is None:
            return
        tuple_base = SequenceValue(
            tuple, [(False, field.typ) for field in self._namedtuple_fields]
        )
        direct_bases: list[MroValue] = []
        changed = False
        for base_value in self._direct_bases:
            if (
                isinstance(base_value, TypedValue)
                and not isinstance(base_value, SequenceValue)
                and base_value.typ is tuple
            ):
                direct_bases.append(tuple_base)
                changed = True
            else:
                direct_bases.append(base_value)
        if changed:
            self._direct_bases = tuple(direct_bases)
            self._update_loaded_synthetic_fields()
            self._protocol_positive_cache.clear()

    def _sync_declared_symbols(self) -> None:
        if self._declared_symbols is None:
            return
        if isinstance(self.typ, type) or self._virtual_symbols is not None:
            self._declared_symbols = self._compute_declared_symbols()
        else:
            self._declared_symbols = self.get_synthetic_declared_symbols()

    def _invalidate_synthetic_state(self) -> None:
        self._checker.arg_spec_cache.known_argspecs.pop(self.typ, None)
        self._checker.arg_spec_cache.generic_bases_cache.pop(self.typ, None)
        self._checker.arg_spec_cache.type_params_cache.pop(self.typ, None)
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def replace_virtual_symbols(self, symbols: Mapping[str, ClassSymbol]) -> None:
        self._virtual_symbols = dict(symbols)
        self._sync_declared_symbols()
        self._invalidate_synthetic_state()

    def set_runtime_namedtuple(self, runtime_class_value: KnownValue) -> None:
        import pycroscope.type_object_builder as type_object_builder

        runtime_class = runtime_class_value.val
        assert isinstance(runtime_class, type), runtime_class
        namedtuple_fields = self._namedtuple_fields_from_runtime(runtime_class)
        self.set_direct_bases((TypedValue(tuple),))
        declared_symbols: dict[str, ClassSymbol] = {}
        type_object_builder._add_runtime_declared_symbols(
            runtime_class, declared_symbols
        )
        self.replace_virtual_symbols(declared_symbols)
        if namedtuple_fields:
            self.set_namedtuple_fields(
                namedtuple_fields,
                constructor_impl=_make_namedtuple_constructor_impl(
                    self.typ, namedtuple_fields
                ),
            )

    def _compute_namedtuple_data(
        self, *, include_inherited_fields: bool = True
    ) -> tuple[bool, Sequence[NamedTupleField]]:
        if isinstance(self.typ, type) and is_direct_namedtuple_class(self.typ):
            return True, self._namedtuple_fields_from_runtime(self.typ)
        synthetic_class = self._get_synthetic_namedtuple_class()
        if synthetic_class is not None:
            return (
                False,
                self._compute_synthetic_namedtuple_fields(
                    synthetic_class, include_inherited_fields=include_inherited_fields
                ),
            )
        if isinstance(self.typ, type) and is_namedtuple_class(self.typ):
            return False, self._namedtuple_fields_from_runtime(self.typ)
        return False, ()

    def _update_loaded_synthetic_fields(self) -> None:
        if self._mro is not None:
            self._mro = self._compute_mro()
        if self._is_final is not None:
            self._is_final = self._compute_is_final()
        if self._is_disjoint_base is not None:
            self._is_disjoint_base = self._compute_is_disjoint_base()
        if self._is_protocol is not None:
            self._is_protocol = self._compute_is_protocol()
        if self._protocol_members is not None:
            self._protocol_members = self._compute_protocol_members()
        if self._dataclass_fields is not None:
            self._dataclass_fields = self._compute_dataclass_fields()
        if self._is_thrift_enum is not None:
            self._is_thrift_enum = self._compute_is_thrift_enum()
        if self._is_universally_assignable is not None:
            self._is_universally_assignable = self._compute_is_universally_assignable()

    def _ensure_synthetic_class(self) -> SyntheticClassObjectValue:
        synthetic_class = self._checker.make_synthetic_class(self.typ)
        return synthetic_class

    def get_synthetic_declared_symbols(self) -> MutableMapping[str, ClassSymbol]:
        if self._synthetic_declared_symbols is None:
            self._synthetic_declared_symbols = {}
        return self._synthetic_declared_symbols

    def set_direct_bases(self, base_values: Sequence[MroValue]) -> None:
        self._direct_bases = tuple(base_values)
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def set_declared_type_params(self, type_params: Sequence[TypeParam]) -> None:
        self._declared_type_params = tuple(type_params)
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def clear_declared_type_params(self) -> None:
        self._declared_type_params = None
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def set_dataclass_info(self, dataclass_info: DataclassInfo | None) -> None:
        self._ensure_synthetic_class()
        self._direct_dataclass_info = dataclass_info
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def set_is_disjoint_base(self, is_disjoint_base: bool) -> None:
        self._ensure_synthetic_class()
        self._directly_disjoint_base = is_disjoint_base
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def set_dataclass_transform_info(
        self, dataclass_transform_info: DataclassTransformInfo | None
    ) -> None:
        self._ensure_synthetic_class()
        self._direct_dataclass_transform_info = dataclass_transform_info
        self._update_loaded_synthetic_fields()
        self._protocol_positive_cache.clear()

    def clear_declared_symbols(self) -> None:
        self._ensure_synthetic_class()
        self.get_synthetic_declared_symbols().clear()
        self._sync_declared_symbols()
        self._invalidate_synthetic_state()

    def set_declared_symbol(self, name: str, symbol: ClassSymbol) -> None:
        self._ensure_synthetic_class()
        synthetic_symbols = self.get_synthetic_declared_symbols()
        synthetic_symbols[name] = symbol
        self._sync_declared_symbols()
        self._invalidate_synthetic_state()

    def add_declared_symbol(self, name: str, symbol: ClassSymbol) -> None:
        self._ensure_synthetic_class()
        synthetic_symbols = self.get_synthetic_declared_symbols()
        synthetic_symbols[name] = merge_declared_symbol(
            synthetic_symbols.get(name), symbol
        )
        self._sync_declared_symbols()
        self._invalidate_synthetic_state()

    # TODO: We need to do something more precise here. We currently assume the metaclass
    # is `type` unless an explicit metaclass was set, but in fact we need to inherit the
    # metaclass from bases.
    def get_metaclass(self) -> MroValue:
        if self._metaclass is None:
            self._metaclass = self._compute_metaclass()
        return self._metaclass

    def _compute_metaclass(self) -> MroValue:
        if isinstance(self.typ, type):
            return TypedValue(type(self.typ))
        return TypedValue(type)  # placeholder

    def set_metaclass(self, metaclass: MroValue) -> None:
        self._metaclass = metaclass

    def get_direct_bases(self) -> tuple[MroValue, ...]:
        if self._direct_bases is None:
            self._direct_bases = self._compute_direct_bases()
        return self._direct_bases

    def get_direct_base_type_objects(self) -> Iterable["TypeObject"]:
        for base in self.get_direct_bases():
            if isinstance(base, TypedValue):
                yield base.get_type_object(self._checker)

    def get_mro(self) -> Sequence[MroEntry]:
        if self._mro is None:
            self._mro = self._compute_mro()
        return self._mro

    def is_in_mro(self, typ: type | str) -> bool:
        return any(
            entry.tobj is not None and entry.tobj.typ == typ for entry in self.get_mro()
        )

    def _get_protocol_members_contributed_by_self(self) -> set[str]:
        if isinstance(self.typ, str) or self._checker.ts_finder.is_protocol(self.typ):
            members = {
                attr
                for attr in self._checker.ts_finder.get_all_attributes(self.typ)
                if attr != "__slots__"
            }
            return members | self._get_protocol_members_from_overlay()
        return (
            _extract_runtime_protocol_members(self.typ)
            | self._get_protocol_members_from_overlay()
        )

    def _get_protocol_members_contributed_by_protocol_bases(self) -> set[str]:
        members = set()
        for base_tobj in self.get_direct_base_type_objects():
            if base_tobj.is_protocol():
                members |= base_tobj.get_protocol_members()
        return members

    def _get_protocol_members_from_overlay(self) -> set[str]:
        if self._checker.get_synthetic_class(self.typ) is None:
            return set()
        declared_symbols = self.get_declared_symbols()
        runtime_names: set[str] | None = None
        if isinstance(self.typ, type):
            runtime_names = _extract_runtime_protocol_members(self.typ)
        return {
            member
            for member in declared_symbols
            if runtime_names is None or member in runtime_names
            if member not in EXCLUDED_PROTOCOL_MEMBERS and member != "__slots__"
        }

    def get_declared_type_params(self) -> tuple[TypeParam, ...]:
        if self._declared_type_params is None:
            self._declared_type_params = self._compute_declared_type_params()
        return self._declared_type_params

    def get_substitutions(self, args: Sequence[Value]) -> TypeVarMap:
        params = self.get_declared_type_params()
        if not params:
            return TypeVarMap()
        return _match_up_generic_params(params, args)

    def get_substitutions_for_base(
        self, base: type | str, args: Sequence[Value]
    ) -> TypeVarMap | None:
        for mro_entry in self.get_mro():
            if mro_entry.tobj is None:
                continue
            mro_value = mro_entry.get_mro_value()
            if not isinstance(mro_value, TypedValue):
                continue
            if mro_value.typ != base:
                continue
            mro_value = mro_value.substitute_typevars(self.get_substitutions(args))
            if not isinstance(mro_value, GenericValue):
                return mro_entry.tobj.get_substitutions(())
            return mro_entry.tobj.get_substitutions(mro_value.args)
        return None

    def get_generic_args_for_base(
        self, base: type | str, args: Sequence[Value]
    ) -> list[Value] | None:
        substitutions = self.get_substitutions_for_base(base, args)
        if substitutions is None:
            return None
        return _generic_args_from_tv_map(
            self._checker.make_type_object(base), substitutions
        )

    def is_final(self) -> bool:
        if self._is_final is None:
            self._is_final = self._compute_is_final()
        return self._is_final

    def is_disjoint_base(self) -> bool:
        if self._is_disjoint_base is None:
            self._is_disjoint_base = self._compute_is_disjoint_base()
        return self._is_disjoint_base

    def is_protocol(self) -> bool:
        if self._is_protocol is None:
            self._is_protocol = self._compute_is_protocol()
        return self._is_protocol

    def get_protocol_members(self) -> set[str]:
        if self._protocol_members is None:
            self._protocol_members = self._compute_protocol_members()
        return self._protocol_members

    def get_dataclass_fields(self) -> tuple[DataclassFieldRecord, ...]:
        """Return dataclass fields for this class and all dataclass bases in MRO order."""
        if self._dataclass_fields is None:
            self._dataclass_fields = self._compute_dataclass_fields()
        return self._dataclass_fields

    def get_direct_dataclass_info(self) -> DataclassInfo | None:
        return self._direct_dataclass_info

    def get_direct_dataclass_transform_info(self) -> DataclassTransformInfo | None:
        return self._direct_dataclass_transform_info

    def is_dataclass(self) -> bool:
        for entry in self.get_mro():
            if entry.tobj is None:
                continue
            if entry.tobj.get_direct_dataclass_info() is not None:
                return True
            if isinstance(entry.tobj.typ, type):
                dataclass_params = safe_getattr(
                    entry.tobj.typ, "__dataclass_params__", None
                )
                if dataclass_params is not None:
                    return True
        return False

    def get_dataclass_frozen_status(self) -> tuple[bool, bool | None]:
        for entry in self.get_mro():
            if entry.tobj is None:
                continue
            if (dataclass_info := entry.tobj.get_direct_dataclass_info()) is not None:
                return True, dataclass_info.frozen
            if isinstance(entry.tobj.typ, type):
                dataclass_params = safe_getattr(
                    entry.tobj.typ, "__dataclass_params__", None
                )
                if dataclass_params is None:
                    continue
                frozen = safe_getattr(dataclass_params, "frozen", None)
                return True, frozen if isinstance(frozen, bool) else None
        return False, None

    def get_dataclass_order_status(self) -> tuple[bool, bool | None]:
        for entry in self.get_mro():
            if entry.tobj is None:
                continue
            if (dataclass_info := entry.tobj.get_direct_dataclass_info()) is not None:
                return True, dataclass_info.order
            if isinstance(entry.tobj.typ, type):
                dataclass_params = safe_getattr(
                    entry.tobj.typ, "__dataclass_params__", None
                )
                if dataclass_params is None:
                    continue
                order = safe_getattr(dataclass_params, "order", None)
                return True, order if isinstance(order, bool) else None
        return False, None

    def get_direct_dataclass_fields(self) -> tuple[DataclassFieldRecord, ...]:
        """Return declaration-order dataclass fields defined directly on this class."""
        if self.get_direct_dataclass_info() is None:
            return ()
        declared_symbols = self.get_synthetic_declared_symbols()
        records: list[DataclassFieldRecord] = []
        for field_name, symbol in declared_symbols.items():
            if symbol.dataclass_field is None:
                continue
            records.append(
                DataclassFieldRecord(
                    field_name=field_name, field_info=symbol.dataclass_field
                )
            )
        return tuple(records)

    def is_namedtuple_like(self) -> bool:
        return self._get_synthetic_namedtuple_class() is not None or (
            isinstance(self.typ, type) and is_namedtuple_class(self.typ)
        )

    def get_namedtuple_field(self, field_name: str) -> NamedTupleField | None:
        for field in self.get_namedtuple_fields():
            if field.name == field_name:
                return field
        return None

    def get_virtual_bases(self) -> list[MroValue]:
        if self._virtual_bases is None:
            self._virtual_bases = self._compute_virtual_bases()
        return self._virtual_bases

    def is_thrift_enum(self) -> bool:
        if self._is_thrift_enum is None:
            self._is_thrift_enum = self._compute_is_thrift_enum()
        return self._is_thrift_enum

    def get_enum_value_type(self) -> Value | None:
        if not self.is_assignable_to_type(enum.Enum):
            return None

        values: list[Value] = []
        if isinstance(self.typ, type):
            seen: set[int] = set()
            try:
                members = self.typ.__members__.values()
            except Exception:
                members = ()
            for member in members:
                if not isinstance(member, enum.Enum):
                    continue
                member_id = id(member)
                if member_id in seen:
                    continue
                seen.add(member_id)
                values.append(KnownValue(member.value))
        if values:
            return unite_values(*values)

        symbol = self.get_declared_symbol("_value_")
        if symbol is None:
            return None
        return symbol.annotation

    def is_universally_assignable(self) -> bool:
        if self._is_universally_assignable is None:
            self._is_universally_assignable = self._compute_is_universally_assignable()
        return self._is_universally_assignable

    def get_declared_symbol(self, name: str) -> ClassSymbol | None:
        return self.get_declared_symbols().get(name)

    def get_declared_symbols(self) -> MutableMapping[str, ClassSymbol]:
        if self._declared_symbols is None:
            if isinstance(self.typ, str) and self._virtual_symbols is None:
                self._declared_symbols = self.get_synthetic_declared_symbols()
            else:
                self._declared_symbols = self._compute_declared_symbols()
        assert self._declared_symbols is not None
        return self._declared_symbols

    def get_attribute(
        self, name: str, policy: AttributePolicy
    ) -> TypeObjectAttribute | None:
        """Look up an attribute in three phases.

        1. Select runtime/typeshed symbols from the relevant MROs.
        2. Specialize the selected symbols for the receiver and owner.
        3. Apply descriptor semantics to the merged result.
        """
        ctx = self._checker
        on_class = policy.on_class
        receiver_value = policy.receiver_value
        typed_receiver_value = _receiver_type_value(receiver_value, ctx)
        # TODO: Revisit whether __class__ and __dict__ should remain bespoke
        # get_attribute() special cases instead of falling out of general lookup.
        if not on_class and name == "__class__":
            return TypeObjectAttribute(
                value=AnyValue(AnySource.inference),
                declared_value=AnyValue(AnySource.inference),
                raw_value=AnyValue(AnySource.inference),
                symbol=ClassSymbol(initializer=AnyValue(AnySource.inference)),
                runtime_symbol=ClassSymbol(initializer=AnyValue(AnySource.inference)),
                typeshed_symbol=None,
                owner=self,
                is_property=False,
                property_has_setter=False,
                is_metaclass_owner=False,
            )
        if not on_class and name == "__dict__":
            return TypeObjectAttribute(
                value=TypedValue(dict),
                declared_value=TypedValue(dict),
                raw_value=TypedValue(dict),
                symbol=ClassSymbol(initializer=TypedValue(dict)),
                runtime_symbol=ClassSymbol(initializer=TypedValue(dict)),
                typeshed_symbol=None,
                owner=self,
                is_property=False,
                property_has_setter=False,
                is_metaclass_owner=False,
            )
        # Phase 1: select runtime/typeshed candidates from the declared MRO and,
        # for class access, the metaclass MRO.
        declared_selected = self._select_declared_attribute(name)
        metaclass_selected = (
            self._select_metaclass_attribute(name) if on_class else None
        )

        # Phase 2: specialize the selected candidates for the receiver and the
        # owner they were found on, then choose the lookup winner according to
        # Python's class-vs-metaclass precedence rules.
        merged_declared = self._specialize_selected_attribute_pair(
            ctx, receiver_value=typed_receiver_value, selected=declared_selected
        )
        merged_metaclass = self._specialize_selected_attribute_pair(
            ctx, receiver_value=typed_receiver_value, selected=metaclass_selected
        )
        merged_attribute = self._choose_merged_attribute(
            merged_declared, merged_metaclass, ctx, policy=policy
        )
        if merged_attribute is None:
            if isinstance(self.typ, str) and self.has_any_base():
                unknown_value = AnyValue(AnySource.from_another)
                return TypeObjectAttribute(
                    value=unknown_value,
                    declared_value=unknown_value,
                    raw_value=unknown_value,
                    symbol=ClassSymbol(initializer=unknown_value),
                    runtime_symbol=ClassSymbol(initializer=unknown_value),
                    typeshed_symbol=None,
                    owner=self,
                    is_property=False,
                    property_has_setter=False,
                    is_metaclass_owner=False,
                )
            return None
        if merged_attribute.is_initvar and not on_class:
            return None
        # Phase 3: apply descriptor semantics to the merged attribute that
        # survived lookup precedence.
        return _resolve_merged_attribute_access(
            self, merged_attribute, ctx, policy=policy
        )

    def _choose_merged_attribute(
        self,
        declared_attribute: MergedAttribute | None,
        metaclass_attribute: MergedAttribute | None,
        ctx: CanAssignContext,
        *,
        policy: AttributePolicy,
    ) -> MergedAttribute | None:
        """Apply Python lookup precedence after generic specialization."""
        if (
            policy.on_class
            and metaclass_attribute is not None
            and (
                policy.is_special_lookup
                or _is_data_descriptor(metaclass_attribute, ctx)
            )
        ):
            return metaclass_attribute
        if declared_attribute is not None:
            return declared_attribute
        return metaclass_attribute if policy.on_class else None

    def _select_declared_attribute(
        self, name: str
    ) -> tuple[SelectedAttribute | None, SelectedAttribute | None]:
        """Select runtime/typeshed class-MRO symbols before specialization."""
        runtime_attr = self.get_selected_attribute(
            name, is_metaclass_owner=False, use_typeshed=False
        )
        typeshed_attr = self.get_selected_attribute(
            name, is_metaclass_owner=False, use_typeshed=True
        )
        return runtime_attr, typeshed_attr

    def _select_metaclass_attribute(
        self, name: str
    ) -> tuple[SelectedAttribute | None, SelectedAttribute | None]:
        """Select runtime/typeshed metaclass-MRO symbols before specialization."""
        metaclass = self.get_metaclass()
        match metaclass:
            case TypedValue():
                metaclass_tobj = metaclass.get_type_object(self._checker)
            case AnyValue():
                return None, None
            case _:
                assert_never(metaclass)
        runtime_attr = metaclass_tobj.get_selected_attribute(
            name, is_metaclass_owner=True, use_typeshed=False
        )
        typeshed_attr = metaclass_tobj.get_selected_attribute(
            name, is_metaclass_owner=True, use_typeshed=True
        )
        return runtime_attr, typeshed_attr

    def _specialize_selected_attribute_pair(
        self,
        ctx: CanAssignContext,
        *,
        receiver_value: TypedValue | TypeVarValue | GenericValue | None,
        selected: tuple[SelectedAttribute | None, SelectedAttribute | None] | None,
    ) -> MergedAttribute | None:
        """Convert selected runtime/typeshed symbols into one merged view."""
        if selected is None:
            return None
        runtime_selected, typeshed_selected = selected
        runtime_attribute = (
            _specialize_selected_attribute(
                self, runtime_selected, ctx, receiver_value=receiver_value
            )
            if runtime_selected is not None
            else None
        )
        typeshed_attribute = (
            _specialize_selected_attribute(
                self, typeshed_selected, ctx, receiver_value=receiver_value
            )
            if typeshed_selected is not None
            else None
        )
        return _make_merged_attribute(runtime_attribute, typeshed_attribute)

    def get_declared_symbol_from_mro(
        self, name: str, ctx: CanAssignContext
    ) -> ClassSymbol | None:
        match = self.get_declared_symbol_with_owner(name, ctx)
        return None if match is None else match[1]

    def get_declared_symbol_with_owner(
        self, name: str, ctx: CanAssignContext
    ) -> tuple["TypeObject", ClassSymbol] | None:
        selected = self._select_declared_attribute(name)
        if selected is None:
            return None
        runtime_selected, typeshed_selected = selected
        symbol = _merge_runtime_and_typeshed_symbol(
            runtime_selected.symbol if runtime_selected is not None else None,
            typeshed_selected.symbol if typeshed_selected is not None else None,
        )
        if symbol is None:
            return None
        if runtime_selected is not None:
            return runtime_selected.owner, symbol
        elif typeshed_selected is not None:
            return typeshed_selected.owner, symbol
        else:
            assert (
                False
            ), "At least one of runtime_selected or typeshed_selected must be non-None"

    def get_selected_attribute(
        self, name: str, *, is_metaclass_owner: bool, use_typeshed: bool
    ) -> SelectedAttribute | None:
        for i, entry in enumerate(self.get_mro()):
            if entry.tobj is None:
                continue
            if use_typeshed:
                symbol = self._checker.ts_finder.get_direct_symbol(entry.tobj.typ, name)
            else:
                symbol = entry.tobj.get_declared_symbols().get(name)
            if symbol is None:
                continue
            return SelectedAttribute(
                mro_index=i,
                owner=entry.tobj,
                symbol=symbol,
                is_metaclass_owner=is_metaclass_owner,
            )
        return None

    def is_assignable_to_type(self, typ: type | str) -> bool:
        return self.is_universally_assignable() or any(
            entry.tobj is not None
            and (
                entry.tobj.typ == typ
                or (
                    safe_isinstance(entry.tobj.typ, type)
                    and safe_isinstance(typ, type)
                    and safe_issubclass(entry.tobj.typ, typ)
                )
            )
            for entry in self.get_mro()
        )

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
        if class_keys_match(self.typ, other.typ):
            return {}
        if other.is_universally_assignable():
            return {}
        if not self.is_protocol():
            if self.typ is object:
                return {}
            if other.is_protocol():
                if self._is_callable_protocol_assignment_target(other):
                    return self._can_assign_callable_protocol(self_val, other_val, ctx)
                return CanAssignError(
                    f"Cannot assign protocol {other_val} to non-protocol {self}"
                )
            if other.is_assignable_to_type(self.typ):
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
                if (
                    isinstance(result, CanAssignError)
                    and other.is_thrift_enum()
                    and other.get_virtual_bases()
                ):
                    for base in other.get_virtual_bases():
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
                and not other.is_protocol()
                and not isinstance(other_basic, SubclassValue)
            ):
                ctx.record_protocol_implementation(self.typ, other.typ)
            return result

    def _is_callable_protocol_assignment_target(self, other: "TypeObject") -> bool:
        return (
            self.typ is collections.abc.Callable
            and "__call__" in other.get_protocol_members()
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
        protocol_members = self.get_protocol_members()
        if len(protocol_members) > 1:
            protocol_self_typevar_map = _collect_protocol_self_typevar_map(
                self, protocol_members, other_val, ctx
            )
            actual_self_typevar_map = _collect_protocol_self_typevar_map(
                other_type_obj, protocol_members, other_val, ctx
            )
        else:
            protocol_self_typevar_map = TypeVarMap()
            actual_self_typevar_map = TypeVarMap()
        apply_synthetic_member_rules = (
            isinstance(self.typ, str)
            and _get_synthetic_class_for_key(self.typ, ctx) is not None
        )
        class_object_check = _is_definitely_class_object_value(other_val)
        for member in protocol_members:
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
                    expected,
                    other_val,
                    ctx,
                    member=member,
                    protocol_self_value=self_val,
                )
                expected = _substitute_receiver_self_typevar(expected, other_val)
                if other_type_obj is not None and other_type_obj.is_protocol():
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
                expected = _substitute_receiver_self_typevar(expected, other_val)
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
                if _protocol_member_is_method(self, member, ctx):
                    expected = _bind_protocol_call_expected(
                        expected,
                        other_val,
                        ctx,
                        member=member,
                        protocol_self_value=self_val,
                    )
                expected = _substitute_receiver_self_typevar(expected, other_val)
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
                    and other_type_obj.is_protocol()
                    and member in other_type_obj.get_protocol_members()
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
        if self.is_protocol():
            return attr in self.get_protocol_members()
        match = self.get_declared_symbol_with_owner(attr, ctx)
        if match is None:
            return False
        _, symbol = match
        return not symbol.is_instance_only and symbol.initializer is not None

    def __str__(self) -> str:
        base = stringify_object(self.typ)
        if self.is_protocol():
            protocol_members = self._get_protocol_members_for_display()
            return (
                f"{base} (Protocol with members"
                f" {', '.join(map(repr, protocol_members))})"
            )
        return base

    def _get_protocol_members_for_display(self) -> list[str]:
        protocol_members = self.get_protocol_members()
        if not isinstance(self.typ, type):
            return sorted(protocol_members)

        members: list[str] = []
        seen = set()
        for base in get_mro(self.typ):
            for attr in base.__dict__:
                if attr in protocol_members and attr not in seen:
                    members.append(attr)
                    seen.add(attr)

            annotations = safe_getattr(base, "__annotations__", None)
            if not isinstance(annotations, dict):
                continue
            for attr in annotations:
                if attr in protocol_members and attr not in seen:
                    members.append(attr)
                    seen.add(attr)

        if len(seen) == len(protocol_members):
            return members
        return [*members, *sorted(protocol_members - seen)]


def _prefer_existing_symbol_type(existing: Value, new: Value) -> bool:
    return isinstance(new, AnyValue) and new.source is AnySource.inference


def _merge_symbol_type_information(
    runtime_value: Value | None, typeshed_value: Value | None
) -> Value | None:
    if runtime_value is None:
        return typeshed_value
    if typeshed_value is None:
        return runtime_value
    if isinstance(typeshed_value, AnyValue) and not isinstance(runtime_value, AnyValue):
        return runtime_value
    return typeshed_value


def _merge_runtime_and_typeshed_property_info(
    runtime_info: PropertyInfo | None,
    typeshed_attribute: SpecializedAttribute | ClassSymbol | None,
) -> PropertyInfo | None:
    if typeshed_attribute is None:
        return runtime_info
    if runtime_info is None:
        return typeshed_attribute.property_info
    if typeshed_attribute.property_info is not None:
        return PropertyInfo(
            fget=_merge_runtime_and_typeshed_symbol(
                runtime_info.fget, typeshed_attribute.property_info.fget
            ),
            fset=_merge_runtime_and_typeshed_symbol(
                runtime_info.fset, typeshed_attribute.property_info.fset
            ),
            fdel=_merge_runtime_and_typeshed_symbol(
                runtime_info.fdel, typeshed_attribute.property_info.fdel
            ),
        )
    if typeshed_attribute.annotation is None:
        return runtime_info
    # Else, we try to use typeshed's annotation to get more precise getter/setter types
    # This helps with cases like type.__doc__
    fget = CallableValue(
        Signature.make(
            [SigParameter(name="self", kind=ParameterKind.POSITIONAL_ONLY)],
            typeshed_attribute.annotation,
        )
    )
    fget_symbol = ClassSymbol(initializer=fget, is_method=True)
    if Qualifier.ReadOnly in (
        typeshed_attribute.selected.symbol.qualifiers
        if isinstance(typeshed_attribute, SpecializedAttribute)
        else typeshed_attribute.qualifiers
    ):
        fset_symbol = None
    else:
        fset = CallableValue(
            Signature.make(
                [
                    SigParameter(name="self", kind=ParameterKind.POSITIONAL_ONLY),
                    SigParameter(
                        name="value",
                        kind=ParameterKind.POSITIONAL_ONLY,
                        annotation=typeshed_attribute.annotation,
                    ),
                ],
                KnownValue(None),
            )
        )
        fset_symbol = ClassSymbol(initializer=fset, is_method=True)
    return PropertyInfo(fget=fget_symbol, fset=fset_symbol, fdel=runtime_info.fdel)


def _merge_runtime_and_typeshed_symbol(
    runtime_symbol: ClassSymbol | None, typeshed_symbol: ClassSymbol | None
) -> ClassSymbol | None:
    if runtime_symbol is None:
        return typeshed_symbol
    if typeshed_symbol is None:
        return runtime_symbol
    property_info = _merge_runtime_and_typeshed_property_info(
        runtime_symbol.property_info, typeshed_symbol
    )
    return ClassSymbol(
        annotation=_merge_symbol_type_information(
            runtime_symbol.annotation, typeshed_symbol.annotation
        ),
        qualifiers=runtime_symbol.qualifiers | typeshed_symbol.qualifiers,
        function_decorators=runtime_symbol.function_decorators
        | typeshed_symbol.function_decorators,
        deprecation_message=runtime_symbol.deprecation_message
        or typeshed_symbol.deprecation_message,
        is_instance_only=(
            runtime_symbol.is_instance_only or typeshed_symbol.is_instance_only
        ),
        is_method=runtime_symbol.is_method,
        returns_self_on_class_access=(
            runtime_symbol.returns_self_on_class_access
            or (
                typeshed_symbol.returns_self_on_class_access
                if runtime_symbol.is_method
                else False
            )
        ),
        property_info=property_info,
        initializer=_merge_symbol_type_information(
            runtime_symbol.initializer, typeshed_symbol.initializer
        ),
        dataclass_field=(
            runtime_symbol.dataclass_field
            if runtime_symbol.dataclass_field is not None
            else typeshed_symbol.dataclass_field
        ),
    )


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
        fget=(
            existing.fget
            if new.fget is None
            else merge_declared_symbol(existing.fget, new.fget)
        ),
        fset=(
            existing.fset
            if new.fset is None
            else merge_declared_symbol(existing.fset, new.fset)
        ),
        fdel=(
            existing.fdel
            if new.fdel is None
            else merge_declared_symbol(existing.fdel, new.fdel)
        ),
    )


def merge_declared_symbol(
    existing: ClassSymbol | None, new: ClassSymbol
) -> ClassSymbol:
    if existing is None:
        return new
    return ClassSymbol(
        annotation=_merge_symbol_initializer(existing.annotation, new.annotation),
        qualifiers=existing.qualifiers | new.qualifiers,
        function_decorators=existing.function_decorators | new.function_decorators,
        deprecation_message=existing.deprecation_message or new.deprecation_message,
        is_instance_only=existing.is_instance_only or new.is_instance_only,
        is_method=existing.is_method or new.is_method,
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
        if not _attribute_blocks_writes(resolved_access, ctx):
            return True
        return resolved_access.property_has_setter
    return not resolved_access.symbol.is_readonly and not _is_frozen_dataclass(tobj)


def _attribute_blocks_writes(
    access: TypeObjectAttribute, ctx: CanAssignContext
) -> bool:
    if not access.is_property:
        return False
    if access.symbol.property_info is not None:
        return True
    return _descriptor_has_method(access.raw_value, "__set__", ctx) or (
        _descriptor_has_method(access.raw_value, "__delete__", ctx)
    )


def _resolve_member_access(
    tobj: TypeObject,
    *,
    member: str,
    resolved_value: Value,
    ctx: CanAssignContext,
    class_object_access: bool,
) -> TypeObjectAttribute:
    access = tobj.get_attribute(member, AttributePolicy(on_class=class_object_access))
    if access is None:
        return TypeObjectAttribute(
            value=resolved_value,
            declared_value=resolved_value,
            raw_value=resolved_value,
            symbol=ClassSymbol(initializer=resolved_value),
            runtime_symbol=ClassSymbol(initializer=resolved_value),
            typeshed_symbol=None,
            owner=tobj,
            is_property=False,
            property_has_setter=False,
            is_metaclass_owner=False,
        )
    symbol = access.symbol
    owner = access.owner
    metaclass_owner = access.is_metaclass_owner
    if class_object_access and not metaclass_owner:
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
    if class_object_access and symbol.is_property and not metaclass_owner:
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
        and not metaclass_owner
        and not _is_property_marker_value(value)
    ):
        value = TypedValue(property)
    return TypeObjectAttribute(
        value=value,
        declared_value=access.declared_value,
        raw_value=access.raw_value,
        symbol=symbol,
        runtime_symbol=access.runtime_symbol,
        typeshed_symbol=access.typeshed_symbol,
        owner=owner,
        is_property=is_property,
        property_has_setter=property_has_setter,
        is_metaclass_owner=access.is_metaclass_owner,
    )


def _class_key_from_value(value: Value) -> type | str | None:
    # This helper is intentionally a little broader than the abstraction we
    # ultimately want: many callers use it for "what class does this value point
    # at?" regardless of whether the value is itself a class object/subclass or
    # an instance of that class. That keeps older call sites simple, but it also
    # means the helper conflates class-like and instance-like values. Over time
    # we should tighten callers so class-object queries and instance-type queries
    # go through more specific helpers.
    keys = list(dict.fromkeys(_iter_class_keys_from_value(value)))
    if len(keys) == 1:
        return keys[0]
    return None


def _iter_class_keys_from_value(value: Value) -> list[type | str]:
    if isinstance(value, AnnotatedValue):
        return _iter_class_keys_from_value(value.value)
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
    elif isinstance(value, SubclassValue):
        return _iter_class_keys_from_value(value.typ)
    elif isinstance(value, TypedValue):
        return _typed_class_key(value)
    elif isinstance(value, KnownValue):
        origin = get_origin(value.val)
        if isinstance(origin, type):
            return [origin]
        if isinstance(value.val, type):
            return [value.val]
        return []
    elif isinstance(
        value,
        (
            AnyValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return []
    else:
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


def _get_cached_property_return_type(
    descriptor: Value, ctx: CanAssignContext
) -> Value | None:
    descriptor = replace_fallback(descriptor)
    if isinstance(descriptor, AnnotatedValue):
        descriptor = replace_fallback(descriptor.value)
    if not (
        isinstance(descriptor, KnownValue)
        and isinstance(descriptor.val, functools.cached_property)
    ):
        return None
    func = safe_getattr(descriptor.val, "func", None)
    if func is None:
        return None
    signature = ctx.get_signature(func)
    if isinstance(signature, Signature):
        return signature.return_value
    if isinstance(signature, OverloadedSignature):
        return unite_values(*(sig.return_value for sig in signature.signatures))
    return None


def _class_key_and_generic_args_from_type_value(
    receiver_value: TypedValue | TypeVarValue | GenericValue,
) -> tuple[type | str, Sequence[Value]]:
    if isinstance(receiver_value, TypeVarValue):
        assert receiver_value.typevar_param.bound is not None
        class_key = _class_key_from_value(receiver_value.typevar_param.bound)
        assert class_key is not None
        return class_key, ()
    generic_args = (
        receiver_value.args if isinstance(receiver_value, GenericValue) else ()
    )
    return receiver_value.typ, generic_args


def _receiver_type_value(
    receiver_value: Value | None, ctx: CanAssignContext
) -> TypedValue | TypeVarValue | GenericValue | None:
    if receiver_value is None:
        return None
    if isinstance(receiver_value, (TypeVarValue, GenericValue)):
        return receiver_value
    if isinstance(receiver_value, SubclassValue):
        return receiver_value.typ
    resolved = replace_fallback(receiver_value)
    if isinstance(resolved, KnownValueWithTypeVars) and not isinstance(
        resolved.val, type
    ):
        runtime_type = type(resolved.val)
        type_params = ctx.get_type_parameters(runtime_type)
        if not type_params:
            return TypedValue(runtime_type)
        return GenericValue(
            runtime_type,
            [
                resolved.typevars.get_value(
                    type_param, default_value_for_type_param(type_param)
                )
                for type_param in type_params
            ],
        )
    if isinstance(resolved, TypedValue):
        return resolved
    if isinstance(resolved, TypeVarValue):
        return resolved
    if isinstance(resolved, GenericValue):
        return resolved
    if isinstance(resolved, KnownValueWithTypeVars) and isinstance(resolved.val, type):
        type_params = ctx.get_type_parameters(resolved.val)
        if not type_params:
            return TypedValue(resolved.val)
        return GenericValue(
            resolved.val,
            [
                resolved.typevars.get_value(
                    type_param, default_value_for_type_param(type_param)
                )
                for type_param in type_params
            ],
        )
    if isinstance(resolved, KnownValue):
        if isinstance(resolved.val, type):
            return TypedValue(resolved.val)
        return TypedValue(type(resolved.val))
    return None


def _typevar_map_from_generic_args(
    type_params: Sequence[TypeParam], generic_args: Sequence[Value]
) -> TypeVarMap:
    if not type_params:
        return TypeVarMap()
    substitutions = TypeVarMap()
    matched = match_typevar_arguments(type_params, generic_args)
    if matched is None:
        return substitutions
    for typevar, value in matched:
        substitutions = _with_typevar_map_value(
            substitutions, typevar, value.substitute_typevars(substitutions)
        )
    return substitutions


def _typevar_map_from_type_value(
    receiver_value: TypedValue | TypeVarValue, type_params: Sequence[TypeParam]
) -> TypeVarMap:
    _, generic_args = _class_key_and_generic_args_from_type_value(receiver_value)
    return _typevar_map_from_generic_args(type_params, generic_args)


def _specialize_symbol_for_owner(
    receiver_tobj: TypeObject,
    owner_tobj: TypeObject,
    symbol: ClassSymbol,
    ctx: CanAssignContext,
    *,
    receiver_value: TypedValue | TypeVarValue | None = None,
) -> ClassSymbol:
    substitutions = _get_symbol_owner_substitutions_from_type_objects(
        receiver_tobj, owner_tobj, ctx, receiver_value=receiver_value
    )
    return symbol.substitute_typevars(substitutions)


def _specialize_selected_attribute(
    receiver_tobj: TypeObject,
    selected: SelectedAttribute,
    ctx: CanAssignContext,
    *,
    receiver_value: TypedValue | TypeVarValue | None = None,
) -> SpecializedAttribute:
    specialized_symbol = _specialize_symbol_for_owner(
        receiver_tobj,
        selected.owner,
        selected.symbol,
        ctx,
        receiver_value=receiver_value,
    )
    return SpecializedAttribute(
        selected=selected,
        annotation=specialized_symbol.annotation,
        initializer=specialized_symbol.initializer,
        property_info=specialized_symbol.property_info,
    )


def _bind_attribute_signature(
    value: Value,
    *,
    receiver_value: Value,
    ctx: CanAssignContext,
    self_annotation_value: Value | None = None,
) -> Value:
    if self_annotation_value is None:
        self_annotation_value = receiver_value
    signature = ctx.signature_from_value(value)
    if isinstance(signature, BoundMethodSignature):
        shielded_signature, restore_typevars = _shield_nested_self_in_signature(
            signature.signature
        )
        signature = replace(signature, signature=shielded_signature)
        bound = signature.get_signature(
            ctx=ctx, self_annotation_value=self_annotation_value, preserve_impl=True
        )
        if bound is None and self_annotation_value == receiver_value:
            bound = signature.get_signature(ctx=ctx, preserve_impl=True)
        if bound is not None:
            result: Value = CallableValue(bound)
            if restore_typevars:
                result = result.substitute_typevars(restore_typevars)
            return result
        return value
    if isinstance(signature, (Signature, OverloadedSignature)):
        if self_annotation_value is None:
            self_annotation_value = receiver_value
        signature, restore_typevars = _shield_nested_self_in_signature(signature)
        bound = signature.bind_self(
            self_value=receiver_value,
            self_annotation_value=self_annotation_value,
            preserve_impl=True,
            ctx=ctx,
        )
        if bound is None and self_annotation_value == receiver_value:
            bound = signature.bind_self(
                self_value=receiver_value, preserve_impl=True, ctx=ctx
            )
        if bound is not None:
            result = CallableValue(bound)
            if restore_typevars:
                result = result.substitute_typevars(restore_typevars)
            return result
    return value


def _shield_nested_self_in_signature(
    signature: Signature | OverloadedSignature,
) -> tuple[Signature | OverloadedSignature, TypeVarMap]:
    if isinstance(signature, OverloadedSignature):
        restore_typevars = TypeVarMap()
        shielded_signatures = []
        for inner_sig in signature.signatures:
            shielded_sig, inner_restore = _shield_nested_self_in_signature(inner_sig)
            assert isinstance(shielded_sig, Signature)
            shielded_signatures.append(shielded_sig)
            restore_typevars = restore_typevars.merge(inner_restore)
        return OverloadedSignature(shielded_signatures), restore_typevars

    restore_typevars = TypeVarMap()
    parameters = {}
    for name, parameter in signature.parameters.items():
        annotation = parameter.annotation
        inner_restore = TypeVarMap()
        if _has_nested_self_typevar(annotation):
            annotation, inner_restore = shield_nested_self_typevars(annotation)
        parameters[name] = replace(parameter, annotation=annotation)
        restore_typevars = restore_typevars.merge(inner_restore)
    return_value = signature.return_value
    return_restore = TypeVarMap()
    if _has_nested_self_typevar(return_value):
        return_value, return_restore = shield_nested_self_typevars(return_value)
    restore_typevars = restore_typevars.merge(return_restore)
    return replace(signature, parameters=parameters, return_value=return_value), (
        restore_typevars
    )


def _value_contains_self_typevar(value: Value) -> bool:
    return any(
        isinstance(subval, TypeVarValue) and subval.typevar_param.typevar is SelfT
        for subval in value.walk_values()
    )


def _specialize_self_returning_classmethod(
    raw_attr: Value,
    normalized_attr: Value,
    *,
    receiver_value: Value | None,
    ctx: CanAssignContext,
) -> Value:
    if receiver_value is None or not isinstance(normalized_attr, CallableValue):
        return normalized_attr
    receiver_for_self: TypedValue | TypeVarValue | GenericValue
    match receiver_value:
        case KnownValueWithTypeVars() if isinstance(receiver_value.val, type):
            type_params = ctx.get_type_parameters(receiver_value.val)
            if type_params:
                receiver_for_self = GenericValue(
                    receiver_value.val,
                    [
                        receiver_value.typevars.get_value(
                            type_param, default_value_for_type_param(type_param)
                        )
                        for type_param in type_params
                    ],
                )
            else:
                receiver_for_self = TypedValue(receiver_value.val)
        case SubclassValue(
            typ=TypeVarValue() | GenericValue() | TypedValue() as subclass_type
        ):
            receiver_for_self = subclass_type
        case TypeVarValue():
            receiver_for_self = receiver_value
        case GenericValue():
            receiver_for_self = receiver_value
        case TypedValue():
            receiver_for_self = receiver_value
        case _:
            return normalized_attr
    substitutions = TypeVarMap()
    raw_attr = replace_fallback(raw_attr)
    if (
        isinstance(raw_attr, GenericValue)
        and raw_attr.typ is classmethod
        and raw_attr.args
    ):
        inferred = get_tv_map(raw_attr.args[0], SubclassValue(receiver_for_self), ctx)
        if not isinstance(inferred, CanAssignError):
            substitutions = inferred
    substitutions = substitutions.with_typevar(TypeVarParam(SelfT), receiver_for_self)
    signature = normalized_attr.signature.substitute_typevars(substitutions)
    return CallableValue(
        _rewrite_self_returning_classmethod_signature(signature, receiver_for_self)
    )


def _rewrite_self_returning_classmethod_signature(
    signature: Signature | OverloadedSignature, receiver_for_self: Value
) -> Signature | OverloadedSignature:
    receiver_key = _class_key_from_value(receiver_for_self)

    def rewrite_return(return_value: Value) -> Value:
        root = replace_fallback(return_value)
        if receiver_key is None:
            return return_value
        if (
            isinstance(root, KnownValueWithTypeVars)
            and isinstance(root.val, type)
            and class_keys_match(root.val, receiver_key)
        ):
            return receiver_for_self
        if (
            isinstance(root, SubclassValue)
            and isinstance(root.typ, TypeVarValue)
            and root.typ.typevar_param.typevar is SelfT
        ):
            return SubclassValue.make(receiver_for_self)
        if (
            isinstance(root, SubclassValue)
            and (subclass_key := _class_key_from_value(root.typ)) is not None
            and class_keys_match(subclass_key, receiver_key)
        ):
            return SubclassValue.make(receiver_for_self)
        return return_value

    if isinstance(signature, Signature):
        return replace(signature, return_value=rewrite_return(signature.return_value))
    return OverloadedSignature(
        tuple(
            replace(sig, return_value=rewrite_return(sig.return_value))
            for sig in signature.signatures
        )
    )


def _classmethod_receiver_value_from_type_value(
    receiver_value: TypedValue | TypeVarValue | GenericValue,
) -> SubclassValue:
    return SubclassValue(receiver_value)


def _is_informative_runtime_attribute(attribute: SpecializedAttribute) -> bool:
    if isinstance(attribute.selected.owner.typ, str):
        # for synthetic classes, assume all symbols were explicitly created and are useful
        return True
    if attribute.selected.symbol.annotation is not None:
        return True  # it was annotated
    if isinstance(attribute.selected.symbol.initializer, KnownValue):
        val = attribute.selected.symbol.initializer.val
        if isinstance(val, (staticmethod, classmethod)):
            val = val.__func__
        if isinstance(val, FunctionType) and safe_getattr(val, "__annotations__", None):
            return True  # it has annotations, so it is likely useful

    return False


def _make_merged_attribute(
    runtime_attribute: SpecializedAttribute | None,
    typeshed_attribute: SpecializedAttribute | None,
) -> MergedAttribute | None:
    if runtime_attribute is None and typeshed_attribute is None:
        return None

    # We ignore the typeshed attribute if it's later in the MRO
    # and we have a useful runtime attribute
    if (
        runtime_attribute is not None
        and typeshed_attribute is not None
        and typeshed_attribute.selected.mro_index > runtime_attribute.selected.mro_index
        and _is_informative_runtime_attribute(runtime_attribute)
    ):
        typeshed_attribute = None
    runtime_symbol = (
        runtime_attribute.as_symbol() if runtime_attribute is not None else None
    )
    typeshed_symbol = (
        typeshed_attribute.as_symbol() if typeshed_attribute is not None else None
    )
    owner = (
        runtime_attribute.selected.owner
        if runtime_attribute is not None
        else not_none(typeshed_attribute).selected.owner
    )
    is_metaclass_owner = (
        runtime_attribute.selected.is_metaclass_owner
        if runtime_attribute is not None
        else not_none(typeshed_attribute).selected.is_metaclass_owner
    )
    if runtime_attribute is not None and typeshed_attribute is not None:
        assert (
            runtime_attribute.selected.is_metaclass_owner
            == typeshed_attribute.selected.is_metaclass_owner
        )
    property_info = _merge_runtime_and_typeshed_property_info(
        runtime_attribute.property_info if runtime_attribute is not None else None,
        typeshed_attribute,
    )
    is_method = (
        runtime_symbol.is_method
        if runtime_symbol is not None
        else not_none(typeshed_symbol).is_method
    )
    is_classmethod = (
        runtime_symbol.is_classmethod if runtime_symbol is not None else False
    ) or (typeshed_symbol.is_classmethod if typeshed_symbol is not None else False)
    is_staticmethod = (
        runtime_symbol.is_staticmethod if runtime_symbol is not None else False
    ) or (typeshed_symbol.is_staticmethod if typeshed_symbol is not None else False)
    is_classvar = (
        runtime_symbol.is_classvar if runtime_symbol is not None else False
    ) or (typeshed_symbol.is_classvar if typeshed_symbol is not None else False)
    is_initvar = (
        runtime_symbol.is_initvar if runtime_symbol is not None else False
    ) or (typeshed_symbol.is_initvar if typeshed_symbol is not None else False)
    returns_self_on_class_access = (
        runtime_symbol.returns_self_on_class_access
        if runtime_symbol is not None and runtime_symbol.returns_self_on_class_access
        else (
            typeshed_symbol.returns_self_on_class_access
            if (runtime_symbol is None or runtime_symbol.is_method)
            and typeshed_symbol is not None
            else False
        )
    )
    annotation = _merge_symbol_type_information(
        runtime_attribute.annotation if runtime_attribute is not None else None,
        typeshed_attribute.annotation if typeshed_attribute is not None else None,
    )
    initializer = _merge_symbol_type_information(
        runtime_attribute.initializer if runtime_attribute is not None else None,
        typeshed_attribute.initializer if typeshed_attribute is not None else None,
    )
    return MergedAttribute(
        owner=owner,
        runtime_symbol=runtime_symbol,
        typeshed_symbol=typeshed_symbol,
        annotation=annotation,
        initializer=initializer,
        property_info=property_info,
        is_method=is_method,
        is_classmethod=is_classmethod,
        is_staticmethod=is_staticmethod,
        is_classvar=is_classvar,
        is_initvar=is_initvar,
        returns_self_on_class_access=returns_self_on_class_access,
        is_metaclass_owner=is_metaclass_owner,
    )


def _merged_attribute_symbol(merged_attribute: MergedAttribute) -> ClassSymbol:
    symbol = _merge_runtime_and_typeshed_symbol(
        merged_attribute.runtime_symbol, merged_attribute.typeshed_symbol
    )
    assert symbol is not None
    return symbol


def _merged_attribute_effective_value(merged_attribute: MergedAttribute) -> Value:
    if merged_attribute.annotation is not None:
        return merged_attribute.annotation
    if merged_attribute.initializer is not None:
        return merged_attribute.initializer
    return AnyValue(AnySource.inference)


def _merged_attribute_lookup_value(merged_attribute: MergedAttribute) -> Value:
    if (
        merged_attribute.runtime_symbol is not None
        and merged_attribute.runtime_symbol.initializer is not None
    ):
        return merged_attribute.runtime_symbol.initializer
    if merged_attribute.initializer is not None:
        return merged_attribute.initializer
    return _merged_attribute_effective_value(merged_attribute)


def _make_resolved_attribute(
    merged_attribute: MergedAttribute,
    *,
    value: Value,
    is_property: bool,
    property_has_setter: bool,
) -> ResolvedAttribute:
    return ResolvedAttribute(
        value=value,
        declared_value=_merged_attribute_effective_value(merged_attribute),
        raw_value=_merged_attribute_lookup_value(merged_attribute),
        symbol=_merged_attribute_symbol(merged_attribute),
        runtime_symbol=merged_attribute.runtime_symbol,
        typeshed_symbol=merged_attribute.typeshed_symbol,
        owner=merged_attribute.owner,
        is_property=is_property,
        property_has_setter=property_has_setter,
        is_metaclass_owner=merged_attribute.is_metaclass_owner,
    )


def _resolve_merged_attribute_access(
    receiver_tobj: TypeObject,
    merged_attribute: MergedAttribute,
    ctx: CanAssignContext,
    *,
    policy: AttributePolicy,
) -> ResolvedAttribute:
    """Resolve a merged member to the final value seen by this access."""
    resolved_descriptor = _resolve_descriptor_access(
        receiver_tobj, merged_attribute, ctx, policy=policy
    )
    if resolved_descriptor is None:
        resolved_descriptor = _make_resolved_attribute(
            merged_attribute,
            value=_get_nondescriptor_value(
                merged_attribute,
                on_class=policy.on_class,
                receiver_value=policy.receiver_value,
                ctx=ctx,
            ),
            is_property=False,
            property_has_setter=False,
        )
    return resolved_descriptor


def _resolve_descriptor_access(
    receiver_tobj: TypeObject,
    merged_attribute: MergedAttribute,
    ctx: CanAssignContext,
    *,
    policy: AttributePolicy,
) -> ResolvedAttribute | None:
    """Apply descriptor behavior, including any receiver binding, to a member."""
    on_class = policy.on_class
    receiver_value = policy.receiver_value
    effective_value = _merged_attribute_effective_value(merged_attribute)
    lookup_value = _merged_attribute_lookup_value(merged_attribute)
    descriptor_like_instance_access = (
        not on_class or merged_attribute.is_metaclass_owner
    )
    if merged_attribute.property_info is not None:
        property_has_setter = merged_attribute.property_info.fset is not None
        if descriptor_like_instance_access:
            if receiver_value is None:
                receiver_value = TypedValue(receiver_tobj.typ)
            receiver_arg = receiver_value
            typed_receiver_value = _receiver_type_value(receiver_value, ctx)
            if (
                on_class
                and merged_attribute.is_metaclass_owner
                and typed_receiver_value is not None
            ):
                receiver_arg = _classmethod_receiver_value_from_type_value(
                    typed_receiver_value
                )
            if merged_attribute.property_info.fget is None:
                return None
            fget = merged_attribute.property_info.fget.initializer
            assert fget is not None
            if policy.visitor is None:
                # Note this path means errors don't get shown!
                value = ctx.get_call_result(fget, (receiver_arg,))
            else:
                value = policy.visitor.get_call_result(
                    fget, (receiver_arg,), node=policy.node
                )

            return _make_resolved_attribute(
                merged_attribute,
                value=value,
                is_property=True,
                property_has_setter=property_has_setter,
            )
        return _make_resolved_attribute(
            merged_attribute,
            value=TypedValue(property),
            is_property=False,
            property_has_setter=property_has_setter,
        )

    lookup_descriptor_value = normalize_synthetic_descriptor_attribute(
        lookup_value,
        is_self_returning_classmethod=merged_attribute.returns_self_on_class_access,
        unknown_descriptor_means_any=False,
    )
    typed_descriptor_value = _get_typed_descriptor_value(
        merged_attribute, lookup_descriptor_value, ctx
    )
    if merged_attribute.is_classmethod:
        typed_descriptor_value = _specialize_self_returning_classmethod(
            lookup_value, typed_descriptor_value, receiver_value=receiver_value, ctx=ctx
        )
        typed_receiver_value = _receiver_type_value(receiver_value, ctx)
        if typed_receiver_value is not None and not _is_prebound_synthetic_classmethod(
            lookup_value
        ):
            typed_descriptor_value = _bind_attribute_signature(
                typed_descriptor_value,
                receiver_value=typed_receiver_value,
                self_annotation_value=_classmethod_receiver_value_from_type_value(
                    typed_receiver_value
                ),
                ctx=ctx,
            )
            if not on_class:
                typed_descriptor_value = set_self(
                    typed_descriptor_value, typed_receiver_value
                )
            if merged_attribute.returns_self_on_class_access and isinstance(
                typed_descriptor_value, CallableValue
            ):
                typed_descriptor_value = CallableValue(
                    _rewrite_self_returning_classmethod_signature(
                        typed_descriptor_value.signature, typed_receiver_value
                    )
                )
        if receiver_value is not None and _value_contains_self_typevar(
            typed_descriptor_value
        ):
            typed_descriptor_value = set_self(typed_descriptor_value, receiver_value)
        return _make_resolved_attribute(
            merged_attribute,
            value=typed_descriptor_value,
            is_property=False,
            property_has_setter=False,
        )
    if merged_attribute.is_staticmethod:
        return _make_resolved_attribute(
            merged_attribute,
            value=typed_descriptor_value,
            is_property=False,
            property_has_setter=False,
        )
    if (
        merged_attribute.is_method
        and not merged_attribute.is_staticmethod
        and _is_callable_member_value(typed_descriptor_value, ctx)
    ):
        if descriptor_like_instance_access and receiver_value is not None:
            typed_descriptor_value = _bind_attribute_signature(
                typed_descriptor_value, receiver_value=receiver_value, ctx=ctx
            )
            if _value_contains_self_typevar(typed_descriptor_value):
                typed_descriptor_value = set_self(
                    typed_descriptor_value, receiver_value
                )
        return _make_resolved_attribute(
            merged_attribute,
            value=typed_descriptor_value,
            is_property=False,
            property_has_setter=False,
        )
    if (
        merged_attribute.annotation is not None
        and not merged_attribute.is_method
        and not merged_attribute.is_classvar
        and not (
            _descriptor_has_method(lookup_value, "__get__", ctx)
            or _descriptor_has_method(lookup_value, "__set__", ctx)
            or _descriptor_has_method(lookup_value, "__delete__", ctx)
        )
    ):
        return None
    raw_has_get = _descriptor_has_method(lookup_value, "__get__", ctx)
    raw_has_set = _descriptor_has_method(lookup_value, "__set__", ctx)
    raw_has_delete = _descriptor_has_method(lookup_value, "__delete__", ctx)
    if (
        merged_attribute.annotation is not None
        and not merged_attribute.is_method
        and not merged_attribute.is_classvar
        and not raw_has_get
        and not raw_has_set
        and not raw_has_delete
    ):
        return None
    descriptor_value = lookup_value
    if _descriptor_has_method(effective_value, "__get__", ctx):
        descriptor_value = effective_value
    descriptor_get_value = _get_descriptor_get_value(
        descriptor_value,
        receiver_tobj,
        ctx,
        on_class=on_class,
        receiver_value=receiver_value,
        is_metaclass_owner=merged_attribute.is_metaclass_owner,
    )
    if descriptor_get_value is None:
        if (
            descriptor_like_instance_access
            and merged_attribute.is_metaclass_owner
            and merged_attribute.annotation is not None
            and raw_has_get
        ):
            return _make_resolved_attribute(
                merged_attribute,
                value=effective_value,
                is_property=True,
                property_has_setter=raw_has_set,
            )
        return None
    if (
        descriptor_like_instance_access
        and isinstance(descriptor_get_value, AnyValue)
        and merged_attribute.annotation is not None
    ):
        descriptor_get_value = effective_value
    if descriptor_like_instance_access and receiver_value is not None:
        descriptor_get_value = set_self(descriptor_get_value, receiver_value)
    return _make_resolved_attribute(
        merged_attribute,
        value=descriptor_get_value,
        is_property=descriptor_like_instance_access,
        property_has_setter=raw_has_set,
    )


def _is_prebound_synthetic_classmethod(value: Value) -> bool:
    value = replace_fallback(value)
    return isinstance(value, GenericValue) and value.typ is classmethod


def _get_nondescriptor_value(
    merged_attribute: MergedAttribute,
    *,
    on_class: bool,
    receiver_value: Value | None,
    ctx: CanAssignContext,
) -> Value:
    """Return the final fallback value when no descriptor behavior applies."""
    effective_value = _merged_attribute_effective_value(merged_attribute)
    lookup_value = normalize_synthetic_descriptor_attribute(
        _merged_attribute_lookup_value(merged_attribute),
        is_self_returning_classmethod=merged_attribute.returns_self_on_class_access,
        unknown_descriptor_means_any=False,
    )
    if not on_class or merged_attribute.is_metaclass_owner:
        if merged_attribute.is_classvar and merged_attribute.annotation is not None:
            value = effective_value
        elif (
            not merged_attribute.is_classvar
            and not merged_attribute.is_method
            and merged_attribute.initializer is None
        ):
            value = effective_value
        elif (
            not merged_attribute.is_classvar
            and not merged_attribute.is_method
            and merged_attribute.initializer is not None
            and merged_attribute.annotation is not None
        ):
            value = effective_value
        else:
            value = lookup_value
        if receiver_value is not None:
            value = set_self(value, receiver_value)
        return value
    if not merged_attribute.is_method:
        return effective_value
    return lookup_value


def _get_typed_descriptor_value(
    merged_attribute: MergedAttribute, lookup_value: Value, ctx: CanAssignContext
) -> Value:
    initializer = merged_attribute.initializer
    if initializer is None:
        return lookup_value
    typed_value = normalize_synthetic_descriptor_attribute(
        initializer,
        is_self_returning_classmethod=merged_attribute.returns_self_on_class_access,
        unknown_descriptor_means_any=False,
    )
    if _is_callable_member_value(typed_value, ctx):
        return typed_value
    return lookup_value


def _get_attribute_value_from_symbol(
    symbol: ClassSymbol,
    ctx: CanAssignContext,
    *,
    on_class: bool,
    receiver_value: TypedValue | None,
) -> Value:
    """Compatibility wrapper for callers that still resolve from bare symbols."""
    merged_attribute = MergedAttribute(
        owner=(
            receiver_value.get_type_object(ctx)
            if receiver_value is not None
            else ctx.make_type_object(object)
        ),
        runtime_symbol=symbol,
        typeshed_symbol=None,
        annotation=symbol.annotation,
        initializer=symbol.initializer,
        property_info=symbol.property_info,
        is_method=symbol.is_method,
        is_classmethod=symbol.is_classmethod,
        is_staticmethod=symbol.is_staticmethod,
        is_classvar=symbol.is_classvar,
        is_initvar=symbol.is_initvar,
        returns_self_on_class_access=symbol.returns_self_on_class_access,
        is_metaclass_owner=False,
    )
    value = _resolve_merged_attribute_access(
        merged_attribute.owner,
        merged_attribute,
        ctx,
        policy=AttributePolicy(on_class=on_class, receiver_value=receiver_value),
    ).value
    return value


def _is_data_descriptor(
    merged_attribute: MergedAttribute, ctx: CanAssignContext
) -> bool:
    """Whether the merged member should win descriptor precedence."""
    if merged_attribute.property_info is not None:
        return True
    if merged_attribute.is_method:
        return False
    return _descriptor_has_method(
        _merged_attribute_lookup_value(merged_attribute), "__set__", ctx
    ) or _descriptor_has_method(
        _merged_attribute_lookup_value(merged_attribute), "__delete__", ctx
    )


def _get_descriptor_get_value(
    descriptor: Value,
    receiver_tobj: TypeObject,
    ctx: CanAssignContext,
    *,
    on_class: bool,
    receiver_value: Value | None,
    is_metaclass_owner: bool,
) -> Value | None:
    """Return the value produced by a descriptor's ``__get__`` call."""
    cached_property_return = _get_cached_property_return_type(descriptor, ctx)
    if cached_property_return is not None:
        return cached_property_return
    descriptor_receiver: Value
    if on_class and not is_metaclass_owner:
        descriptor_receiver = KnownValue(None)
    elif receiver_value is not None:
        descriptor_receiver = receiver_value
    else:
        descriptor_receiver = TypedValue(receiver_tobj.typ)
    match = _descriptor_method_match(
        descriptor, "__get__", [descriptor_receiver, AnyValue(AnySource.inference)], ctx
    )
    if match is None:
        return None
    get_signature, _ = match
    return_value = get_signature.return_value
    self_type = receiver_to_self_type(descriptor, ctx)
    if on_class and not is_metaclass_owner:
        return_without_self = subtract_values(return_value, self_type, ctx)
        if return_without_self != return_value:
            narrowed_return = intersect_values(return_value, self_type, ctx)
            if narrowed_return is not NO_RETURN_VALUE:
                return_value = narrowed_return
    else:
        return_value = subtract_values(return_value, self_type, ctx)
    return return_value


def _runtime_descriptor_owner(descriptor: KnownValue | TypedValue) -> type | None:
    if isinstance(descriptor, KnownValue) and not isinstance(descriptor.val, type):
        return type(descriptor.val)
    if isinstance(descriptor, TypedValue) and isinstance(descriptor.typ, type):
        return descriptor.typ
    return None


def _runtime_type_declares_method(typ: type, method_name: str) -> bool:
    return any(method_name in cls.__dict__ for cls in typ.__mro__)


def _descriptor_has_method(
    descriptor: Value, method_name: str, ctx: CanAssignContext
) -> bool:
    """Whether the descriptor provides a retrievable dunder method."""
    descriptor = replace_fallback(descriptor)
    if isinstance(descriptor, AnnotatedValue):
        return _descriptor_has_method(descriptor.value, method_name, ctx)
    if isinstance(descriptor, KnownValue):
        runtime_owner = _runtime_descriptor_owner(descriptor)
        if runtime_owner is not None and not _runtime_type_declares_method(
            runtime_owner, method_name
        ):
            return False
        if (
            descriptor.get_type_object(ctx).get_declared_symbol_from_mro(
                method_name, ctx
            )
            is None
        ):
            return False
        return (
            _descriptor_method_signature_any(descriptor, method_name, ctx) is not None
        )
    if isinstance(descriptor, TypedValue):
        runtime_owner = _runtime_descriptor_owner(descriptor)
        if runtime_owner is not None and not _runtime_type_declares_method(
            runtime_owner, method_name
        ):
            return False
        if (
            descriptor.get_type_object(ctx).get_declared_symbol_from_mro(
                method_name, ctx
            )
            is None
        ):
            return False
        return (
            _descriptor_method_signature_any(descriptor, method_name, ctx) is not None
        )
    if isinstance(descriptor, SyntheticClassObjectValue):
        if (
            ctx.make_type_object(
                descriptor.class_type.typ
            ).get_declared_symbol_from_mro(method_name, ctx)
            is None
        ):
            return False
        return (
            _descriptor_method_signature_any(descriptor, method_name, ctx) is not None
        )
    return False


def _descriptor_method_match(
    descriptor: Value, method_name: str, args: Sequence[Value], ctx: CanAssignContext
) -> tuple[Signature, int] | None:
    """Pick the descriptor overload that accepts the simulated descriptor call."""
    signature = _descriptor_method_signature_any(descriptor, method_name, ctx)
    if signature is None:
        return None
    selected = _select_matching_descriptor_signature(signature, args, ctx)
    if selected is not None:
        return selected, 0
    selected = _select_matching_descriptor_signature(
        signature, [descriptor, *args], ctx
    )
    if selected is None:
        return None
    return selected, 1


def _descriptor_method_signature_any(
    descriptor: Value, method_name: str, ctx: CanAssignContext
) -> Signature | OverloadedSignature | None:
    """Retrieve the descriptor dunder signature, if it can be modeled."""
    descriptor = replace_fallback(descriptor)
    if isinstance(descriptor, AnnotatedValue):
        return _descriptor_method_signature_any(descriptor.value, method_name, ctx)
    if not isinstance(descriptor, (KnownValue, TypedValue, SyntheticClassObjectValue)):
        return None
    descriptor, restore_typevars = shield_nested_self_typevars(descriptor)
    method_value = UNINITIALIZED_VALUE
    direct_signature: Signature | OverloadedSignature | None = None
    if isinstance(descriptor, SyntheticClassObjectValue):
        if isinstance(descriptor.class_type, TypedValue):
            descriptor_tobj = ctx.make_type_object(descriptor.class_type.typ)
            if descriptor_tobj.get_declared_symbol_from_mro(method_name, ctx) is None:
                return None
            attribute = descriptor_tobj.get_attribute(
                method_name, AttributePolicy(receiver_value=descriptor)
            )
            if attribute is not None:
                method_value = attribute.value
    else:
        assert isinstance(descriptor, (KnownValue, TypedValue))
        runtime_owner = _runtime_descriptor_owner(descriptor)
        if runtime_owner is not None and not _runtime_type_declares_method(
            runtime_owner, method_name
        ):
            return None
        descriptor_tobj = descriptor.get_type_object(ctx)
        if descriptor_tobj.get_declared_symbol_from_mro(method_name, ctx) is None:
            return None
        if isinstance(descriptor, TypedValue):
            merged_attribute = descriptor_tobj._specialize_selected_attribute_pair(
                ctx,
                receiver_value=descriptor,
                selected=descriptor_tobj._select_declared_attribute(method_name),
            )
            if merged_attribute is not None:
                direct_signature = _signature_from_descriptor_attribute(
                    _merged_attribute_lookup_value(merged_attribute), ctx
                )
                if direct_signature is not None:
                    direct_signature = direct_signature.bind_self(
                        self_value=descriptor, ctx=ctx
                    )
        attribute = descriptor_tobj.get_attribute(
            method_name, AttributePolicy(receiver_value=descriptor)
        )
        if attribute is not None:
            method_value = attribute.value
    if direct_signature is not None:
        if restore_typevars:
            direct_signature = direct_signature.substitute_typevars(restore_typevars)
        return direct_signature
    if method_value is UNINITIALIZED_VALUE:
        return None
    signature = _signature_from_descriptor_attribute(method_value, ctx)
    if signature is None:
        return None
    if restore_typevars:
        signature = signature.substitute_typevars(restore_typevars)
    return signature


def _signature_from_descriptor_attribute(
    value: Value, ctx: CanAssignContext
) -> Signature | OverloadedSignature | None:
    """Extract a concrete callable signature from a descriptor method value."""
    signature = ctx.signature_from_value(value)
    if isinstance(signature, BoundMethodSignature):
        signature = signature.get_signature(ctx=ctx)
    if isinstance(signature, (Signature, OverloadedSignature)):
        return signature
    return None


def _select_matching_descriptor_signature(
    signature: Signature | OverloadedSignature,
    args: Sequence[Value],
    ctx: CanAssignContext,
) -> Signature | None:
    """Choose the overload compatible with the simulated descriptor call."""
    if isinstance(signature, Signature):
        if _descriptor_signature_accepts_args(signature, args, ctx):
            return signature
        return None
    for overload in signature.signatures:
        if _descriptor_signature_accepts_args(overload, args, ctx):
            return overload
    return None


def _descriptor_signature_accepts_args(
    signature: Signature, args: Sequence[Value], ctx: CanAssignContext
) -> bool:
    """Cheap yes/no call check for simulated descriptor invocations.

    Descriptor matching is especially sensitive to symbolic ``Self`` values.
    Use direct parameter/argument assignability here so overload selection does
    not trigger the broader call inference machinery.
    """
    from .relations import Relation, has_relation

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
        if isinstance(
            has_relation(parameter.annotation, arg, Relation.ASSIGNABLE, ctx),
            CanAssignError,
        ):
            return False
    for parameter in positional_params[len(args) :]:
        if parameter.default is None:
            return False
    return True


def _substitute_symbol_value(value: Value, substitutions: TypeVarMap) -> Value:
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
    receiver_value: TypedValue | TypeVarValue | None = None,
) -> TypeVarMap:
    receiver_substitutions = TypeVarMap()
    if receiver_value is not None:
        receiver_substitutions = _typevar_map_from_type_value(
            receiver_value, receiver_tobj.get_declared_type_params()
        )
    if owner_tobj is receiver_tobj:
        return receiver_substitutions
    owner_value = next(
        (
            entry.get_mro_value()
            for entry in receiver_tobj.get_mro()
            if entry.tobj is owner_tobj
        ),
        None,
    )
    if owner_value is None:
        return TypeVarMap()
    if receiver_substitutions:
        owner_value = owner_value.substitute_typevars(receiver_substitutions)
    owner_substitutions = _typevar_map_from_generic_args(
        owner_tobj.get_declared_type_params(), _mro_generic_args(owner_value)
    )
    if not owner_substitutions:
        return TypeVarMap()
    if not receiver_substitutions:
        return owner_substitutions
    return receiver_substitutions.merge(owner_substitutions)


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
    match = type_object.get_declared_symbol_with_owner(member, ctx)
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


def _is_frozen_dataclass(tobj: TypeObject) -> bool:
    _, frozen = tobj.get_dataclass_frozen_status()
    return frozen is True


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
    tobj = ctx.make_type_object(class_key)
    metaclass = tobj.get_metaclass()
    match metaclass:
        case TypedValue():
            return metaclass.typ
        case AnyValue():
            return None
        case _:
            assert_never(metaclass)


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
    member: str,
    protocol_self_value: Value | None = None,
) -> Value:
    unwrapped = replace_fallback(value)
    if not isinstance(unwrapped, CallableValue):
        return value
    signature = unwrapped.signature
    if member == "__call__":
        if isinstance(signature, BoundMethodSignature):
            return value
        receiver_reference = (
            self_value if protocol_self_value is None else protocol_self_value
        )
        allow_any_annotation = protocol_self_value is not None
    else:
        if isinstance(signature, BoundMethodSignature):
            signature = signature.signature
        receiver_reference = (
            self_value if protocol_self_value is None else protocol_self_value
        )
        allow_any_annotation = False
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
        call_member, self_value, ctx, member="__call__", protocol_self_value=self_value
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
) -> TypeVarMap:
    """Collect typevar substitutions implied by receiver annotations.

    This propagates `self: T` constraints across protocol members.
    """
    tv_map = TypeVarMap()
    for member in sorted(protocol_members):
        match = tobj.get_declared_symbol_with_owner(member, ctx)
        if match is None:
            continue
        _, symbol = match
        collected = _get_protocol_receiver_annotation(
            tobj, symbol, receiver_value=receiver_value, ctx=ctx
        )
        if collected is None:
            continue
        self_annotation, receiver_for_match = collected
        tv_map = _merge_protocol_receiver_typevars(
            tv_map, self_annotation, receiver_for_match, ctx
        )
    return tv_map


def _protocol_member_is_method(
    tobj: TypeObject, member: str, ctx: CanAssignContext
) -> bool:
    match = tobj.get_declared_symbol_with_owner(member, ctx)
    return match is not None and match[1].is_method


def _substitute_receiver_self_typevar(value: Value, receiver_value: Value) -> Value:
    """Substitute only receiver-bound ``Self`` occurrences.

    Nested ``Self`` values may belong to surrounding type arguments such as
    ``Iterable[Self]`` and should not be rebound to the protocol receiver.
    """
    shielded, restore_typevars = shield_nested_self_typevars(value)
    substituted = shielded.substitute_typevars(
        TypeVarMap(typevars={SelfT: receiver_value})
    )
    if restore_typevars:
        substituted = substituted.substitute_typevars(restore_typevars)
    return substituted


def _get_protocol_receiver_annotation(
    owner_tobj: TypeObject,
    symbol: ClassSymbol,
    *,
    receiver_value: Value,
    ctx: CanAssignContext,
) -> tuple[Value, Value] | None:
    if symbol.is_staticmethod or symbol.initializer is None:
        return None
    if symbol.is_classmethod:
        receiver_for_match = _protocol_classmethod_receiver_value(receiver_value, ctx)
        raw_attr = replace_fallback(symbol.initializer)
        if isinstance(raw_attr, GenericValue) and raw_attr.typ is classmethod:
            if not raw_attr.args:
                return None
            return SubclassValue.make(raw_attr.args[0]), receiver_for_match
    else:
        receiver_for_match = receiver_value
    callable_obj = _get_protocol_member_callable(symbol)
    if callable_obj is None:
        return None
    signature = _as_concrete_signature(ctx.signature_from_value(callable_obj), ctx)
    if signature is None:
        return None
    self_annotation = _get_first_parameter_annotation(signature)
    if self_annotation is None:
        return None
    if (
        isinstance(self_annotation, AnyValue)
        and self_annotation.source is AnySource.unannotated
    ):
        self_annotation = _default_protocol_receiver_annotation(owner_tobj, symbol)
    return self_annotation, receiver_for_match


def _default_protocol_receiver_annotation(
    owner_tobj: TypeObject, symbol: ClassSymbol
) -> Value:
    params = owner_tobj.get_declared_type_params()
    if params:
        owner_value: Value = GenericValue(
            owner_tobj.typ, [type_param_to_value(param) for param in params]
        )
    else:
        owner_value = TypedValue(owner_tobj.typ)
    if symbol.is_classmethod:
        return SubclassValue.make(owner_value)
    return owner_value


def _get_protocol_member_callable(symbol: ClassSymbol) -> Value | None:
    initializer = symbol.initializer
    if initializer is None:
        return None
    if symbol.is_property:
        initializer = replace_fallback(initializer)
        if (
            isinstance(initializer, KnownValue)
            and isinstance(initializer.val, property)
            and initializer.val.fget is not None
        ):
            return KnownValue(initializer.val.fget)
        return None
    return normalize_synthetic_descriptor_attribute(
        initializer,
        is_self_returning_classmethod=symbol.returns_self_on_class_access,
        unknown_descriptor_means_any=False,
    )


def _get_first_parameter_annotation(
    signature: Signature | OverloadedSignature,
) -> Value | None:
    signatures = (
        signature.signatures
        if isinstance(signature, OverloadedSignature)
        else [signature]
    )
    for concrete in signatures:
        parameters = list(concrete.parameters.values())
        if parameters:
            return parameters[0].annotation
    return None


def _protocol_classmethod_receiver_value(
    receiver_value: Value, ctx: CanAssignContext
) -> Value:
    if not _is_definitely_class_object_value(receiver_value):
        receiver_type = _receiver_type_value(receiver_value, ctx)
        if receiver_type is not None:
            return SubclassValue.make(receiver_type)
        return SubclassValue.make(receiver_value.get_type_value())
    receiver_key = _class_key_from_value(receiver_value)
    if receiver_key is not None:
        class_object = _class_object_value_for_key(receiver_key, ctx)
        if class_object is not None:
            return class_object
    return SubclassValue.make(receiver_value.get_type_value())


def _merge_protocol_receiver_typevars(
    tv_map: TypeVarMap,
    self_annotation: Value,
    receiver_for_match: Value,
    ctx: CanAssignContext,
) -> TypeVarMap:
    if not any(
        isinstance(subvalue, TypeVarValue) for subvalue in self_annotation.walk_values()
    ):
        return tv_map
    inferred = get_tv_map(self_annotation, receiver_for_match, ctx)
    if isinstance(inferred, CanAssignError):
        return tv_map
    translated = translate_generic_typevar_map(self_annotation, inferred, ctx)
    if not translated:
        translated = infer_positional_generic_typevar_map(
            self_annotation, receiver_for_match, ctx
        )
    return _merge_protocol_receiver_typevar_maps(tv_map, inferred.merge(translated))


def _is_placeholder_typevartuple_binding(binding: Sequence[tuple[bool, Value]]) -> bool:
    return (
        len(binding) == 1
        and binding[0][0]
        and isinstance(binding[0][1], AnyValue)
        and binding[0][1].source is AnySource.generic_argument
    )


def _merge_protocol_receiver_typevar_maps(
    existing: TypeVarMap, new: TypeVarMap
) -> TypeVarMap:
    merged = existing
    for typevar, value in new.iter_typevars():
        type_param = TypeVarParam(typevar)
        existing_value = merged.get_typevar(type_param)
        if existing_value is None:
            merged = merged.with_typevar(type_param, value)
        elif existing_value != value:
            merged = merged.with_typevar(
                type_param, unite_values(existing_value, value)
            )
    for paramspec, input_sig in new.iter_paramspecs():
        type_param = ParamSpecParam(paramspec)
        existing_sig = merged.get_paramspec(type_param)
        if existing_sig is None or (
            isinstance(existing_sig, AnySig) and not isinstance(input_sig, AnySig)
        ):
            merged = merged.with_paramspec(type_param, input_sig)
    for typevartuple, binding in new.iter_typevartuples():
        type_param = TypeVarTupleParam(typevartuple)
        existing_binding = merged.get_typevartuple(type_param)
        if existing_binding is None or (
            _is_placeholder_typevartuple_binding(existing_binding)
            and not _is_placeholder_typevartuple_binding(binding)
        ):
            merged = merged.with_typevartuple(type_param, binding)
    return merged


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
        elif isinstance(value.typ, str):
            return value.typ in {"type", "builtins.type"}
        else:
            assert_never(value.typ)
    return False


def _compute_type_params_from_bases(bases: Sequence[Value]) -> Iterable[TypeParam]:
    # If any base is Protocol or Generic, it determines the type parameter order
    for base in bases:
        if isinstance(base, GenericValue) and (
            is_typing_name(base.typ, "Protocol") or is_typing_name(base.typ, "Generic")
        ):
            for arg in base.args:
                type_param = _extract_type_param(arg)
                if type_param is not None:
                    yield type_param
            return

    # Else we have to walk the bases
    type_params = []
    for base in bases:
        type_params.extend(_compute_type_params_from_base(base))
    yield from dict.fromkeys(type_params)  # deduplicate while preserving order


def _compute_type_params_from_base(base: Value) -> Iterable[TypeParam]:
    if isinstance(base, SequenceValue):
        for _, member in base.members:
            type_param = _extract_type_param(member)
            if type_param is not None:
                yield type_param
            else:
                yield from _compute_type_params_from_base(member)
    elif isinstance(base, GenericValue):
        for arg in base.args:
            type_param = _extract_type_param(arg)
            if type_param is not None:
                yield type_param
            else:
                yield from _compute_type_params_from_base(arg)


def _extract_type_param(value: Value) -> TypeParam | None:
    match value:
        case TypeVarValue(typevar_param=typevar_param):
            return typevar_param
        case InputSigValue(input_sig=ParamSpecParam() as param):
            return param
        case _:
            return get_single_typevartuple_param(value)


def _compute_type_params_from_runtime(
    typ: type, checker: "pycroscope.checker.Checker"
) -> list[TypeParam]:
    runtime_type_params = safe_getattr(typ, "__type_params__", ())
    if not runtime_type_params:
        runtime_type_params = safe_getattr(typ, "__parameters__", ())
    try:
        runtime_type_params_iter = iter(runtime_type_params)
    except TypeError:
        return []
    type_params: list[TypeParam] = []
    for type_param in runtime_type_params_iter:
        try:
            assert checker._arg_spec_cache is not None
            type_params.append(
                make_type_param(type_param, ctx=checker._arg_spec_cache.default_context)
            )
        except TypeError:
            continue
    return type_params


def _replace_invalid_bases(bases: Sequence[Value]) -> Iterable[MroValue]:
    for base in bases:
        if isinstance(base, (TypedValue, AnyValue)):
            yield base
        elif isinstance(base, KnownValue) and is_typing_name(base.val, "Any"):
            yield AnyValue(AnySource.explicit)
        else:
            yield AnyValue(AnySource.inference)


def _extract_runtime_direct_bases(
    typ: type, checker: "pycroscope.checker.Checker"
) -> Iterable[MroValue]:
    try:
        class_dict = typ.__dict__
        raw_bases = iter(class_dict["__orig_bases__"])
    except Exception:
        try:
            raw_bases = iter(typ.__bases__)
        except Exception:
            return []
    bases = [
        type_from_runtime(base, visitor=checker, suppress_errors=True)
        for base in raw_bases
    ]
    if is_namedtuple_class(typ):
        bases = [_replace_tuple(base, typ, checker) for base in bases]
    return _replace_invalid_bases(bases)


def _replace_tuple(
    base: Value, typ: type, checker: "pycroscope.checker.Checker"
) -> Value:
    if base == TypedValue(tuple):
        if typ.__annotations__:
            field_types = tuple(typ.__annotations__.values())
            return SequenceValue(
                tuple,
                [
                    (False, type_from_runtime(field_type, visitor=checker))
                    for field_type in field_types
                ],
            )
        else:
            return SequenceValue(
                tuple, [(False, AnyValue(AnySource.unannotated)) for _ in typ._fields]
            )
    return base


def _get_mro_entry(
    mro_value: MroValue,
    parent_tv_map: TypeVarMap,
    checker: "pycroscope.checker.Checker",
) -> MroEntry:
    match mro_value:
        case TypedValue():
            tv_map = _get_tv_map_for_mro(mro_value, parent_tv_map, checker)
            tobj = mro_value.get_type_object(checker)
            if isinstance(mro_value, SequenceValue):
                value = mro_value
            else:
                value = None
            return MroEntry(tobj=tobj, tv_map=tv_map, value=value, is_virtual=False)
        case AnyValue():
            return ANY_MRO_ENTRY
        case _:
            assert_never(mro_value)


def _mark_direct_base_non_virtual(mro: list[MroEntry]) -> list[MroEntry]:
    if not mro or mro[0].is_any:
        return mro
    return [replace(mro[0], is_virtual=False), *mro[1:]]


def _mark_entry_virtual_by_runtime_mro(
    entry: MroEntry, runtime_mro: Sequence[type]
) -> MroEntry:
    if entry.is_any or entry.tobj is None:
        return entry
    is_in_runtime_mro = any(
        class_keys_match(entry.tobj.typ, runtime_base) for runtime_base in runtime_mro
    )
    return replace(entry, is_virtual=not is_in_runtime_mro)


def _get_tv_map_for_mro(
    mro_value: TypedValue,
    parent_tv_map: TypeVarMap,
    checker: "pycroscope.checker.Checker",
) -> TypeVarMap:
    tobj = mro_value.get_type_object(checker)
    params = tobj.get_declared_type_params()
    if params:
        if isinstance(mro_value, GenericValue):
            tv_map = _match_up_generic_params(params, mro_value.args)
            substituted = TypeVarMap()
            for param in params:
                value = tv_map.get_value(param)
                if value is None:
                    continue
                substituted = substituted.with_value(
                    param, value.substitute_typevars(parent_tv_map)
                )
            return substituted
        else:
            substitutions = TypeVarMap()
            for param in params:
                substitutions = substitutions.with_value(
                    param, AnyValue(AnySource.generic_argument)
                )
            return substitutions
    else:
        return TypeVarMap()


def _make_object_mro_entry(checker: "pycroscope.checker.Checker") -> MroEntry:
    tobj = checker.make_type_object(object)
    return MroEntry(tobj=tobj, tv_map=TypeVarMap(), value=None, is_virtual=False)


def _get_mro_from_mro_value(
    mro_value: MroValue,
    parent_tv_map: TypeVarMap,
    checker: "pycroscope.checker.Checker",
    *,
    virtual: bool = False,
) -> list[MroEntry]:
    match mro_value:
        case TypedValue():
            tobj = mro_value.get_type_object(checker)
            tv_map = _get_tv_map_for_mro(mro_value, parent_tv_map, checker)
            mro = tobj.get_mro()
            mro = [mro_entry.substitute_typevars(tv_map) for mro_entry in mro]
            if isinstance(mro_value, SequenceValue) and mro_value.typ is tuple:
                assert mro[0].tobj is not None and mro[0].tobj.typ is tuple, repr(
                    mro[0]
                )
                mro[0] = replace(mro[0], value=mro_value)
        case AnyValue():
            mro = [ANY_MRO_ENTRY, _make_object_mro_entry(checker)]
        case _:
            assert_never(mro_value)
    if virtual:
        mro = [replace(entry, is_virtual=True) for entry in mro]
    return mro


def _match_up_generic_params(
    type_params: Sequence[TypeParam], generic_args: Sequence[Value]
) -> TypeVarMap:
    """Match generic arguments to type parameters,
    returning a mapping of type variables to values."""
    seq = match_typevar_arguments(type_params, generic_args)
    if seq is None:
        substitutions = TypeVarMap()
        for param in type_params:
            substitutions = substitutions.with_value(
                param, default_value_for_type_param(param)
            )
        return substitutions
    return _typevar_map_from_varlike_pairs(seq)


def _entries_match(a: MroEntry, b: MroEntry) -> bool:
    if a.is_any:
        return b.is_any
    return a.tobj is b.tobj


def _linearize_mros(
    head: MroEntry, tail_mros: Sequence[list[MroEntry]]
) -> list[MroEntry] | str:
    """Linearize MROs using the C3 algorithm.

    Returns either an MRO or an error message if the MRO is invalid.
    """
    mro = [head]
    tails = [list(tail) for tail in tail_mros if tail]
    while tails:
        candidate = None
        for i, tail in enumerate(tails):
            candidate = tail[0]
            if not any(
                _entries_match(candidate, other_tail[j])
                for j, other_tail in enumerate(tails)
                if i != j
                for j in range(1, len(other_tail))
            ):
                break
        else:
            assert candidate is not None
            # No valid candidate found, MRO is invalid
            return (
                f"Cannot create consistent MRO because {candidate.tobj} "
                f"appears multiple times in conflicting positions"
            )
        for other_tail in tails:
            for entry in other_tail:
                if _entries_match(candidate, entry) and not entry.is_virtual:
                    candidate = replace(candidate, is_virtual=False)
                    break
            if not candidate.is_virtual:
                break
        mro.append(candidate)
        for tail in tails:
            if tail and _entries_match(tail[0], candidate):
                tail.pop(0)
        tails = [tail for tail in tails if tail]
    return mro


def _add_namedtuple_dunder_new_symbol(
    tobj: TypeObject,
    fields: Sequence[NamedTupleField],
    *,
    constructor_impl: Impl | None = None,
) -> ClassSymbol:
    parameters = {
        "__self": SigParameter(name="__self", kind=ParameterKind.POSITIONAL_ONLY)
    }
    parameters.update(
        {
            field.name: SigParameter(
                name=field.name,
                kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                annotation=field.typ,
                default=field.default,
            )
            for field in fields
        }
    )
    signature = Signature(
        parameters=parameters, return_value=SelfTVV, impl=constructor_impl
    )
    symbol = ClassSymbol(
        annotation=None,
        initializer=CallableValue(signature),
        is_method=True,
        property_info=None,
    )
    tobj.add_declared_symbol("__new__", symbol)
    return symbol


def _make_namedtuple_constructor_impl(
    typ: type | str, fields: Sequence[NamedTupleField]
) -> Impl:
    def infer_namedtuple_return(ctx: CallContext) -> Value:
        return SequenceValue(
            typ, tuple((False, ctx.vars[field.name]) for field in fields)
        )

    return infer_namedtuple_return


def _add_synthetic_declared_symbols(
    declared_symbols: Mapping[str, ClassSymbol], symbols: dict[str, ClassSymbol]
) -> None:
    for name, symbol in declared_symbols.items():
        symbols[name] = merge_declared_symbol(symbols.get(name), symbol)
