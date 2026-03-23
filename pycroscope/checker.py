"""

The checker maintains global state that is preserved across different modules.

"""

import ast
import collections.abc
import enum
import inspect
import types
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TypeVar, cast

import pycroscope

from . import dataclass as dataclass_helpers
from .annotations import type_from_runtime, type_from_value
from .arg_spec import ArgSpecCache, GenericBases
from .attributes import AttrContext, get_attribute
from .extensions import get_overloads as get_runtime_overloads
from .input_sig import InputSigValue, coerce_paramspec_specialization_to_input_sig
from .node_visitor import Failure
from .options import Options
from .reexport import ImplicitReexportTracker
from .safe import safe_getattr, safe_isinstance, safe_issubclass
from .shared_options import VariableNameValues
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    BoundMethodSignature,
    ConcreteSignature,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    _promote_constructor_type_arg,
    make_bound_method,
)
from .stacked_scopes import Composite
from .suggested_type import CallableTracker
from .type_object import (
    EXCLUDED_PROTOCOL_MEMBERS,
    DataclassFieldRecord,
    TypeObject,
    class_keys_match,
    direct_bases_from_values,
    normalize_synthetic_descriptor_attribute,
    runtime_type_generic_alias,
)
from .typeshed import TypeshedFinder
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    ClassSymbol,
    DataclassFieldInfo,
    GenericValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    ParamSpecParam,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    SelfT,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypeAlias,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeParam,
    TypeVarLike,
    TypeVarMap,
    TypeVarParam,
    TypeVarTupleValue,
    TypeVarType,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    flatten_values,
    get_inherited_synthetic_member_initializer,
    get_synthetic_member_initializer,
    get_tv_map,
    is_union,
    iter_type_params_in_value,
    replace_fallback,
    set_self,
    type_param_to_value,
    unite_values,
)

_SyntheticGenericBases = dict[type | str, dict[TypeVarLike, Value]]


@dataclass(frozen=True)
class _DataclassFieldEntry:
    parameter: SigParameter
    is_initvar: bool


def _replace_signature_return(
    signature: MaybeSignature, return_value: Value
) -> MaybeSignature:
    if isinstance(signature, (Signature, OverloadedSignature)):
        return signature.replace_return_value(return_value)
    if isinstance(signature, BoundMethodSignature):
        return BoundMethodSignature(
            signature.signature.replace_return_value(return_value),
            signature.self_composite,
            return_override=signature.return_override,
        )
    return signature


def _infer_type_params_from_signature(signature: MaybeSignature) -> list[TypeParam]:
    seen: set[object] = set()
    inferred: list[TypeParam] = []

    def _record(value: Value) -> None:
        for type_param in iter_type_params_in_value(value):
            if type_param.typevar in seen:
                continue
            seen.add(type_param.typevar)
            inferred.append(type_param)

    def _walk(sig: MaybeSignature) -> None:
        if isinstance(sig, OverloadedSignature):
            for sub_sig in sig.signatures:
                _walk(sub_sig)
            return
        if not isinstance(sig, Signature):
            return
        for parameter in sig.parameters.values():
            _record(parameter.annotation)
        _record(sig.return_value)

    _walk(signature)
    return inferred


def _apply_type_parameter_defaults(
    type_params: Sequence[TypeParam], checker: "Checker"
) -> list[Value]:
    specialized: list[Value] = []
    substitutions: dict[TypeVarLike, Value] = {}
    for type_param in type_params:
        value = pycroscope.type_object_builder._default_type_argument_for_param(
            type_param, substitutions, checker
        )
        substitutions[type_param.typevar] = value
        specialized.append(value)
    return specialized


def _replace_signature_returns(
    signature: MaybeSignature, return_signature: MaybeSignature
) -> MaybeSignature:
    if isinstance(signature, Signature) and isinstance(return_signature, Signature):
        return signature.replace_return_value(return_signature.return_value)
    if isinstance(signature, OverloadedSignature) and isinstance(
        return_signature, OverloadedSignature
    ):
        if len(signature.signatures) == len(return_signature.signatures):
            return OverloadedSignature(
                [
                    sig.replace_return_value(ret_sig.return_value)
                    for sig, ret_sig in zip(
                        signature.signatures, return_signature.signatures
                    )
                ]
            )
    if isinstance(return_signature, (Signature, OverloadedSignature)):
        return _replace_signature_return(signature, return_signature.return_value)
    return signature


def _iter_signature_variants(signature: MaybeSignature | None) -> Iterable[Signature]:
    if signature is None:
        return
    if isinstance(signature, BoundMethodSignature):
        yield from _iter_signature_variants(signature.signature)
    elif isinstance(signature, Signature):
        yield signature
    else:
        for sub_sig in signature.signatures:
            yield from _iter_signature_variants(sub_sig)


def _map_maybe_signature(
    signature: MaybeSignature, transform: Callable[[Signature], Signature]
) -> MaybeSignature:
    if isinstance(signature, Signature):
        return transform(signature)
    if isinstance(signature, OverloadedSignature):
        mapped_signatures: list[Signature] = []
        for sub_sig in signature.signatures:
            mapped = _map_maybe_signature(sub_sig, transform)
            assert isinstance(mapped, Signature)
            mapped_signatures.append(mapped)
        return OverloadedSignature(mapped_signatures)
    if isinstance(signature, BoundMethodSignature):
        inner = _map_maybe_signature(signature.signature, transform)
        assert isinstance(inner, ConcreteSignature)
        return BoundMethodSignature(
            inner, signature.self_composite, return_override=signature.return_override
        )
    return signature


def _signature_allows_runtime_call(signature: MaybeSignature) -> bool:
    return any(sig.allow_call for sig in _iter_signature_variants(signature))


def _set_signature_allow_call(
    signature: MaybeSignature,
    *,
    allow_call: bool,
    callable_object: object | None = None,
) -> MaybeSignature:
    if not allow_call:
        return signature

    def transform(sig: Signature) -> Signature:
        return dataclass_replace(
            sig,
            allow_call=True,
            callable=sig.callable if callable_object is None else callable_object,
        )

    return _map_maybe_signature(signature, transform)


def _signature_has_return_annotation(signature: ConcreteSignature) -> bool:
    if isinstance(signature, Signature):
        return signature.has_return_annotation
    return all(sig.has_return_annotation for sig in signature.signatures)


def _signature_has_impl(signature: MaybeSignature | None) -> bool:
    return any(
        sig.impl is not None or sig.evaluator is not None
        for sig in _iter_signature_variants(signature)
    )


def _signature_uses_custom_constructor(
    signature: MaybeSignature | None, cls: type
) -> bool:
    return any(
        sig.callable is not None and sig.callable is not cls
        for sig in _iter_signature_variants(signature)
    )


def _signature_uses_metaclass_call(signature: MaybeSignature | None) -> bool:
    return any(
        getattr(sig.callable, "__name__", None) == "__call__"
        for sig in _iter_signature_variants(signature)
    )


def _signature_returns_typeddict(signature: MaybeSignature | None) -> bool:
    return any(
        isinstance(sig.return_value, TypedDictValue)
        for sig in _iter_signature_variants(signature)
    )


def _combine_signatures(signatures: Sequence[Signature]) -> ConcreteSignature | None:
    if not signatures:
        return None
    if len(signatures) == 1:
        return signatures[0]
    return OverloadedSignature(list(signatures))


def _make_incompatible_constructor_signature(instance_type: Value) -> Signature:
    return Signature.make(
        [
            SigParameter(
                "%self", kind=ParameterKind.POSITIONAL_ONLY, annotation=NO_RETURN_VALUE
            )
        ],
        instance_type,
    )


def _is_incompatible_constructor_signature(signature: ConcreteSignature) -> bool:
    signatures = (
        signature.signatures
        if isinstance(signature, OverloadedSignature)
        else [signature]
    )
    for sig in signatures:
        params = list(sig.parameters.values())
        if len(params) != 1:
            return False
        param = params[0]
        if (
            param.name != "%self"
            or param.kind is not ParameterKind.POSITIONAL_ONLY
            or param.annotation is not NO_RETURN_VALUE
        ):
            return False
    return True


def _extract_generic_args_from_self_annotation(
    annotation: Value, class_type: type | str
) -> tuple[Value, ...] | None:
    root = replace_fallback(annotation)
    if isinstance(root, SubclassValue):
        root = replace_fallback(root.typ)
    if isinstance(root, GenericValue) and class_keys_match(root.typ, class_type):
        return tuple(root.args)
    return None


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


def _synthetic_descriptor_set_type(descriptor: Value, ctx: AttrContext) -> Value | None:
    descriptor = replace_fallback(descriptor)
    if isinstance(descriptor, AnnotatedValue):
        return _synthetic_descriptor_set_type(descriptor.value, ctx)
    if not isinstance(descriptor, (KnownValue, TypedValue, SyntheticClassObjectValue)):
        return None
    method_ctx = ctx.clone_for_attribute_lookup(Composite(descriptor), "__set__")
    method_value = get_attribute(method_ctx)
    if method_value is UNINITIALIZED_VALUE:
        return None
    signature = _signature_from_synthetic_attribute(method_value, method_ctx)
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


@dataclass
class Checker:
    raw_options: InitVar[Options | None] = None
    options: Options = field(init=False)
    _arg_spec_cache: ArgSpecCache | None = field(default=None, init=False, repr=False)
    ts_finder: TypeshedFinder = field(init=False, repr=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False, repr=False)
    callable_tracker: CallableTracker = field(init=False, repr=False)
    type_object_cache: dict[type | str, TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )
    synthetic_classes: dict[type | str, SyntheticClassObjectValue] = field(
        default_factory=dict, init=False, repr=False
    )
    _relation_cache: dict[object, object] = field(
        default_factory=dict, init=False, repr=False
    )
    assumed_compatibilities: list[tuple[TypeObject, TypeObject]] = field(
        default_factory=list
    )
    alias_assumed_compatibilities: set[tuple[TypeAliasValue, TypeAliasValue]] = field(
        default_factory=set
    )
    vnv_map: dict[str, VariableNameValue] = field(default_factory=dict)
    type_alias_cache: dict[object, TypeAlias] = field(default_factory=dict)
    runtime_callable_self_annotation_cache: dict[object, bool] = field(
        default_factory=dict, init=False, repr=False
    )
    runtime_class_self_annotation_cache: dict[type, bool] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self, raw_options: Options | None) -> None:
        if raw_options is None:
            self.options = Options.from_option_list()
        else:
            self.options = raw_options
        self.ts_finder = TypeshedFinder.make(self, self.options)
        self._arg_spec_cache = ArgSpecCache(
            self.options,
            self.ts_finder,
            self,
            vnv_provider=self.maybe_get_variable_name_value,
        )
        self.reexport_tracker = ImplicitReexportTracker(self.options)
        self.callable_tracker = CallableTracker()

        for vnv in self.options.get_value_for(VariableNameValues):
            for variable in vnv.varnames:
                self.vnv_map[variable] = vnv

    def maybe_get_variable_name_value(self, varname: str) -> VariableNameValue | None:
        return VariableNameValue.from_varname(varname, self.vnv_map)

    @property
    def arg_spec_cache(self) -> ArgSpecCache:
        arg_spec_cache = self._arg_spec_cache
        assert arg_spec_cache is not None
        return arg_spec_cache

    def resolve_name(
        self,
        node: ast.Name,
        error_node: ast.AST | None = None,
        suppress_errors: bool = False,
    ) -> tuple[Value, object]:
        return AnyValue(AnySource.inference), node.id

    def get_type_alias_cache(self) -> dict[object, TypeAlias]:
        return self.type_alias_cache

    def perform_final_checks(self) -> list[Failure]:
        return self.callable_tracker.check(self)

    def _canonical_type_object_key(self, typ: type | str) -> type | str:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None and isinstance(
            synthetic_class.class_type, TypedValue
        ):
            class_key = synthetic_class.class_type.typ
            if isinstance(class_key, type):
                return class_key
        return typ

    def _get_cached_type_object(self, typ: type | str) -> TypeObject | None:
        canonical_key = typ
        try:
            canonical_key = self._canonical_type_object_key(typ)
        except Exception:
            pass
        cached = self.type_object_cache.get(canonical_key)
        if cached is None and typ != canonical_key:
            cached = self.type_object_cache.get(typ)
        if cached is None and isinstance(canonical_key, type):
            cached = self.type_object_cache.get(
                runtime_type_generic_alias(canonical_key)
            )
        if cached is None and isinstance(typ, type):
            cached = self.type_object_cache.get(runtime_type_generic_alias(typ))
        return cached

    def make_type_object(self, typ: type | str) -> TypeObject:
        try:
            canonical_key = self._canonical_type_object_key(typ)
        except Exception:
            return TypeObject(self, typ)
        cached = self._get_cached_type_object(canonical_key)
        needs_sync = False
        if cached is None:
            cached = TypeObject(self, canonical_key)
            self.type_object_cache[canonical_key] = cached
            self.type_object_cache[typ] = cached
            if isinstance(canonical_key, type):
                self.type_object_cache[runtime_type_generic_alias(canonical_key)] = (
                    cached
                )
            if isinstance(typ, type):
                self.type_object_cache[runtime_type_generic_alias(typ)] = cached
            needs_sync = True
        elif isinstance(canonical_key, type) and cached.typ is not canonical_key:
            cached.typ = canonical_key
            needs_sync = True
        if needs_sync:
            self._sync_synthetic_class_type_object(canonical_key, cached)
        self.type_object_cache[canonical_key] = cached
        self.type_object_cache[typ] = cached
        if isinstance(canonical_key, type):
            self.type_object_cache[runtime_type_generic_alias(canonical_key)] = cached
        if isinstance(typ, type):
            self.type_object_cache[runtime_type_generic_alias(typ)] = cached
        return cached

    def get_type_object_for_value(
        self, value: SimpleType, current_class: type | str | None
    ) -> tuple[TypeObject, bool]:
        """Return a tuple of the type object, and whether it is for the class or instance."""
        match value:
            case AnyValue() | TypeFormValue() | UnboundMethodValue() | PredicateValue():
                return self.make_type_object(object), False
            case SyntheticModuleValue():
                return self.make_type_object(types.ModuleType), False
            case KnownValue(val) if safe_isinstance(val, type):
                return self.make_type_object(val), True
            case KnownValue(val=val):
                return self.make_type_object(type(val)), False
            case SyntheticClassObjectValue(class_type=TypedValue(typ)):
                return self.make_type_object(typ), True
            case TypedValue(typ=typ):
                return self.make_type_object(typ), False
            case SubclassValue(TypedValue(typ)):
                return self.make_type_object(typ), True
            case SubclassValue(TypeVarValue() as tv):
                if tv.typevar_param.typevar is SelfT and current_class is not None:
                    return self.make_type_object(current_class), True
                # TODO: could be more precise
                return self.make_type_object(object), True
            case _:
                # TODO: should be assert_never(value) but our narrowing isn't good enough
                assert False

    def _sync_synthetic_class_type_object(
        self, typ: type | str, type_object: TypeObject
    ) -> None:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is None:
            return
        type_object.adopt_synthetic_class(synthetic_class)

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        generic_bases = self.arg_spec_cache.get_generic_bases(typ, generic_args)
        synthetic_bases = self._get_synthetic_generic_bases(typ)
        substitution_map: dict[TypeVarLike, Value] = {}
        if synthetic_bases is None:
            declared_type_params = tuple(self.get_type_parameters(typ))
        else:
            declared_type_params = self._get_synthetic_declared_type_params(typ)
        specialized_args = self.arg_spec_cache._specialize_generic_type_params(
            declared_type_params, generic_args
        )
        for type_param, concrete_arg in zip(declared_type_params, specialized_args):
            if isinstance(type_param, ParamSpecParam):
                concrete_arg = coerce_paramspec_specialization_to_input_sig(
                    concrete_arg
                )
            substitution_map[type_param.typevar] = concrete_arg

        merged: _SyntheticGenericBases = {
            base: dict(tv_map) for base, tv_map in generic_bases.items()
        }
        if synthetic_bases is None:
            self._augment_namedtuple_generic_bases(typ, merged, substitution_map)
            return merged

        if declared_type_params:
            if typ not in merged:
                merged[typ] = {}
            direct_base_map: dict[TypeVarLike, Value] = merged[typ]
            for type_param in declared_type_params:
                runtime_typevar = type_param.typevar
                if runtime_typevar in substitution_map:
                    direct_base_map[runtime_typevar] = substitution_map[runtime_typevar]
        for base, tv_map in synthetic_bases.items():
            substituted_tv_map = {
                tv: value.substitute_typevars(substitution_map)
                for tv, value in tv_map.items()
            }
            if base not in merged:
                merged[base] = {}
            base_map: dict[TypeVarLike, Value] = merged[base]
            base_map.update(substituted_tv_map)
        self._augment_namedtuple_generic_bases(typ, merged, substitution_map)
        return merged

    def get_type_parameters(self, typ: type | str) -> list[TypeParam]:
        declared_type_params = self._get_synthetic_declared_type_params(typ)
        if declared_type_params:
            return list(declared_type_params)
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None:
            inferred = self._infer_synthetic_type_params(synthetic_class)
            if inferred:
                return list(inferred)
        return self.arg_spec_cache.get_type_parameters(typ)

    def register_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> None:
        class_type = synthetic_class.class_type
        if not isinstance(class_type, TypedValue):
            return
        if isinstance(class_type, TypedDictValue):
            return
        typ = class_type.typ
        had_cached_type_object = any(
            key in self.type_object_cache
            for key in self._iter_generic_override_keys(typ)
        )
        if not had_cached_type_object and isinstance(typ, type):
            had_cached_type_object = (
                runtime_type_generic_alias(typ) in self.type_object_cache
            )
        for key in self._iter_generic_override_keys(typ):
            if key in self.synthetic_classes:
                assert self.synthetic_classes[key] is synthetic_class, (
                    f"Conflicting synthetic classes for key {key} "
                    f"(from {synthetic_class.class_type}):"
                    f" {self.synthetic_classes[key]} vs {synthetic_class}"
                )
            self.synthetic_classes[key] = synthetic_class
        if had_cached_type_object:
            cached = self._get_cached_type_object(typ)
            if cached is not None:
                self._sync_synthetic_class_type_object(typ, cached)

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        for key in self._iter_generic_override_keys(typ):
            synthetic_class = self.synthetic_classes.get(key)
            if synthetic_class is not None:
                return synthetic_class
        return None

    def make_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None:
            return synthetic_class
        if isinstance(typ, str):
            name = typ.rsplit(".", 1)[-1]
        else:
            name = typ.__name__
        synthetic_class = SyntheticClassObjectValue(name, TypedValue(typ))
        self.register_synthetic_class(synthetic_class)
        return synthetic_class

    def register_synthetic_type_bases(
        self,
        typ: type | str,
        base_values: Sequence[Value],
        *,
        declared_type_params: Sequence[TypeParam] = (),
    ) -> None:
        merged_generic_bases: _SyntheticGenericBases = {typ: {}}
        for base in base_values:
            for converted in pycroscope.type_object_builder._iter_base_type_values(
                base, self.arg_spec_cache
            ):
                if not isinstance(converted, TypedValue):
                    continue
                base_typ = converted.typ
                # Preserve direct synthetic bases even when we cannot infer a
                # richer generic mapping for them (common for local synthetic
                # classes with no typeshed entry).
                merged_generic_bases.setdefault(base_typ, {})
                if isinstance(converted, SequenceValue) and converted.typ is tuple:
                    generic_args = (converted,)
                else:
                    generic_args = (
                        converted.args if isinstance(converted, GenericValue) else ()
                    )
                for gb_typ, tv_map in self.get_generic_bases(
                    base_typ, generic_args
                ).items():
                    merged_generic_bases.setdefault(gb_typ, {}).update(tv_map)

        synthetic_class = self.make_synthetic_class(typ)
        self.make_type_object(typ).set_direct_bases(
            direct_bases_from_values(base_values, self)
        )
        merged_copy: _SyntheticGenericBases = {}
        for gb_typ, tv_map in merged_generic_bases.items():
            merged_copy[gb_typ] = dict(tv_map)
        synthetic_class.generic_bases.clear()
        synthetic_class.generic_bases.update(merged_copy)
        self.make_type_object(typ).set_declared_type_params(declared_type_params)

    def register_synthetic_protocol_members(
        self, typ: type | str, members: set[str]
    ) -> None:
        cleaned_members = {
            member
            for member in members
            if member not in EXCLUDED_PROTOCOL_MEMBERS and member != "__slots__"
        }
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is None:
            return
        type_object = self.make_type_object(typ)
        for member in cleaned_members:
            existing = type_object.get_declared_symbol(member)
            if existing is None:
                type_object.add_declared_symbol(
                    member, ClassSymbol(initializer=AnyValue(AnySource.inference))
                )

    def _iter_generic_override_keys(self, typ: type | str) -> Iterator[type | str]:
        yield typ
        if isinstance(typ, type):
            yield runtime_type_generic_alias(typ)

    def _get_synthetic_generic_bases(
        self, typ: type | str
    ) -> _SyntheticGenericBases | None:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is None:
            return None
        if synthetic_class.generic_bases:
            return {
                base_typ: dict(tv_map)
                for base_typ, tv_map in synthetic_class.generic_bases.items()
            }
        if self.make_type_object(typ).get_declared_type_params():
            return {}
        return None

    def _augment_namedtuple_generic_bases(
        self,
        typ: type | str,
        generic_bases: _SyntheticGenericBases,
        substitution_map: TypeVarMap,
    ) -> None:
        tuple_base = self._namedtuple_tuple_base(typ)
        if tuple_base is None:
            return
        if substitution_map:
            tuple_base = tuple_base.substitute_typevars(substitution_map)
            assert isinstance(tuple_base, SequenceValue)
        tuple_type_params = self.arg_spec_cache.get_type_parameters(tuple)
        if len(tuple_type_params) != 1:
            return
        generic_bases.setdefault(tuple, {})[tuple_type_params[0].typevar] = tuple_base

    def _namedtuple_tuple_base(self, typ: type | str) -> SequenceValue | None:
        type_object = self.make_type_object(typ)
        if not type_object.is_namedtuple_like():
            return None
        fields = tuple(type_object.get_namedtuple_fields())
        if not fields:
            return None
        return SequenceValue(tuple, [(False, field.typ) for field in fields])

    def _get_synthetic_declared_type_params(
        self, typ: type | str
    ) -> tuple[TypeParam, ...]:
        tobj = self.make_type_object(typ)
        return tobj.get_declared_type_params()

    def get_signature(
        self, obj: object, is_asynq: bool = False
    ) -> ConcreteSignature | None:
        sig = self.arg_spec_cache.get_argspec(obj, is_asynq=is_asynq)
        if isinstance(sig, Signature):
            return sig
        elif isinstance(sig, BoundMethodSignature):
            return sig.get_signature(ctx=self)
        elif isinstance(sig, OverloadedSignature):
            return sig
        return None

    def can_assume_compatibility(self, left: TypeObject, right: TypeObject) -> bool:
        return (left, right) in self.assumed_compatibilities

    @contextmanager
    def assume_compatibility(
        self, left: TypeObject, right: TypeObject
    ) -> Generator[None]:
        """Context manager that notes that left and right can be assumed to be compatible."""
        pair = (left, right)
        self.assumed_compatibilities.append(pair)
        try:
            yield
        finally:
            new_pair = self.assumed_compatibilities.pop()
            assert pair == new_pair

    def can_aliases_assume_compatibility(
        self, left: TypeAliasValue, right: TypeAliasValue
    ) -> bool:
        return (left, right) in self.alias_assumed_compatibilities

    @contextmanager
    def aliases_assume_compatibility(
        self, left: TypeAliasValue, right: TypeAliasValue
    ) -> Generator[None]:
        pair = (left, right)
        self.alias_assumed_compatibilities.add(pair)
        try:
            yield
        finally:
            self.alias_assumed_compatibilities.discard(pair)

    def get_relation_cache(self) -> dict[object, object] | None:
        return self._relation_cache

    def has_active_relation_assumptions(self) -> bool:
        return bool(self.assumed_compatibilities or self.alias_assumed_compatibilities)

    def display_value(self, value: Value) -> str:
        message = f"'{value!s}'"
        if isinstance(value, KnownValue):
            sig = self.arg_spec_cache.get_argspec(value.val)
        elif isinstance(value, UnboundMethodValue):
            sig = value.get_signature(self)
        elif isinstance(value, SyntheticClassObjectValue):
            sig = self.signature_from_value(value)
        else:
            sig = None
        if sig is not None:
            message += f", signature is {sig!s}"
        return message

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""
        pass

    def record_protocol_implementation(
        self, protocol: type[object], implementing_class: type[object]
    ) -> None:
        """Record that implementing_class was shown assignable to protocol."""
        pass

    def _get_runtime_overloaded_method_signature(
        self, typ: type, attr: str
    ) -> OverloadedSignature | None:
        fq_name = f"{typ.__module__}.{typ.__qualname__}.{attr}"
        overloads = get_runtime_overloads(fq_name)
        if not overloads:
            return None
        signatures: list[Signature] = []
        for overload in overloads:
            sig = self.arg_spec_cache.get_argspec(overload)
            if isinstance(sig, OverloadedSignature):
                return sig
            if isinstance(sig, Signature):
                signatures.append(sig)
        if not signatures:
            return None
        return OverloadedSignature(signatures)

    def _get_unbound_method_owner(self, value: UnboundMethodValue) -> type | None:
        root = replace_fallback(value.composite.value)
        if isinstance(root, TypedValue) and isinstance(root.typ, type):
            return root.typ
        if isinstance(root, KnownValue) and isinstance(root.val, type):
            return root.val
        return None

    def _as_concrete_signature(
        self, signature: MaybeSignature
    ) -> ConcreteSignature | None:
        if isinstance(signature, BoundMethodSignature):
            return signature.get_signature(ctx=self)
        if isinstance(signature, (Signature, OverloadedSignature)):
            return signature
        return None

    def _is_uninformative_constructor_signature(
        self, signature: ConcreteSignature
    ) -> bool:
        if signature is ANY_SIGNATURE:
            return True
        if isinstance(signature, OverloadedSignature):
            return all(
                self._is_uninformative_constructor_signature(sig)
                for sig in signature.signatures
            )
        if (
            len(signature.parameters) == 1
            and next(iter(signature.parameters.values())).kind is ParameterKind.ELLIPSIS
            and isinstance(signature.return_value, AnyValue)
        ):
            return True
        params = list(signature.parameters.values())
        if (
            len(params) == 2
            and params[0].kind is ParameterKind.VAR_POSITIONAL
            and params[1].kind is ParameterKind.VAR_KEYWORD
            and self._is_any_vararg_annotation(params[0].annotation)
            and self._is_any_kwarg_annotation(params[1].annotation)
            and isinstance(signature.return_value, AnyValue)
        ):
            return True
        return False

    def _is_any_vararg_annotation(self, annotation: Value) -> bool:
        if isinstance(annotation, AnyValue):
            return True
        if (
            isinstance(annotation, GenericValue)
            and annotation.typ is tuple
            and len(annotation.args) == 1
            and isinstance(annotation.args[0], AnyValue)
        ):
            return True
        return False

    def _is_any_kwarg_annotation(self, annotation: Value) -> bool:
        if isinstance(annotation, AnyValue):
            return True
        if (
            isinstance(annotation, GenericValue)
            and annotation.typ is dict
            and len(annotation.args) == 2
            and isinstance(annotation.args[0], TypedValue)
            and annotation.args[0].typ is str
            and isinstance(annotation.args[1], AnyValue)
        ):
            return True
        return False

    def _is_permissive_annotated_value(self, annotation: Value) -> bool:
        if isinstance(annotation, AnyValue):
            return True
        root = replace_fallback(annotation)
        return isinstance(root, TypedValue) and root.typ is object

    def _is_permissive_vararg_annotation(self, annotation: Value) -> bool:
        if self._is_permissive_annotated_value(annotation):
            return True
        if (
            isinstance(annotation, GenericValue)
            and annotation.typ is tuple
            and len(annotation.args) == 1
            and self._is_permissive_annotated_value(annotation.args[0])
        ):
            return True
        return False

    def _is_permissive_kwarg_annotation(self, annotation: Value) -> bool:
        if self._is_permissive_annotated_value(annotation):
            return True
        key_annotation: Value | None = None
        if isinstance(annotation, GenericValue) and annotation.typ is dict:
            if len(annotation.args) == 2:
                key_annotation = replace_fallback(annotation.args[0])
        if (
            isinstance(annotation, GenericValue)
            and annotation.typ is dict
            and len(annotation.args) == 2
            and isinstance(key_annotation, TypedValue)
            and key_annotation.typ is str
            and self._is_permissive_annotated_value(annotation.args[1])
        ):
            return True
        return False

    def _has_permissive_constructor_parameters(self, signature: Signature) -> bool:
        params = list(signature.parameters.values())
        if (
            len(params) == 1
            and params[0].kind is ParameterKind.ELLIPSIS
            and isinstance(params[0].annotation, AnyValue)
        ):
            return True
        return (
            len(params) == 2
            and params[0].kind is ParameterKind.VAR_POSITIONAL
            and params[1].kind is ParameterKind.VAR_KEYWORD
            and self._is_permissive_vararg_annotation(params[0].annotation)
            and self._is_permissive_kwarg_annotation(params[1].annotation)
        )

    def _is_passthrough_metaclass_call_signature(
        self, signature: ConcreteSignature, *, instance_type: Value
    ) -> bool:
        signatures = (
            signature.signatures
            if isinstance(signature, OverloadedSignature)
            else [signature]
        )
        for sig in signatures:
            if not self._has_permissive_constructor_parameters(sig):
                return False
            if isinstance(sig.return_value, AnyValue):
                continue
            if instance_type.is_assignable(
                sig.return_value, self
            ) and sig.return_value.is_assignable(instance_type, self):
                continue
            if any(
                isinstance(subval, TypeVarValue)
                for subval in sig.return_value.walk_values()
            ):
                continue
            if self._value_nominal_class_name(
                sig.return_value
            ) == self._value_nominal_class_name(instance_type):
                continue
            return False
        return bool(signatures)

    def _value_nominal_class_name(self, value: Value) -> str | None:
        root = replace_fallback(value)
        if isinstance(root, GenericValue):
            typ = root.typ
        elif isinstance(root, TypedValue):
            typ = root.typ
        else:
            return None
        if isinstance(typ, str):
            return typ.rsplit(".", maxsplit=1)[-1]
        if isinstance(typ, type):
            return typ.__name__
        return None

    def _bind_constructor_like_signature(
        self,
        signature: MaybeSignature,
        *,
        self_value: Value,
        self_annotation_value: Value | None,
    ) -> ConcreteSignature | None:
        if isinstance(signature, BoundMethodSignature):
            return signature.get_signature(
                preserve_impl=True,
                ctx=self,
                self_annotation_value=self_annotation_value,
            )
        concrete = self._as_concrete_signature(signature)
        if concrete is None:
            return None
        if self_annotation_value is None:
            return concrete.bind_self(
                preserve_impl=True,
                self_value=self_value,
                self_annotation_value=None,
                ctx=self,
            )
        bound = make_bound_method(concrete, Composite(self_value), ctx=self)
        if bound is None:
            return None
        return bound.get_signature(
            preserve_impl=True, ctx=self, self_annotation_value=self_annotation_value
        )

    def _synthetic_constructor_return_from_self_annotation(
        self, annotation: Value, *, default: Value, class_type: type | str
    ) -> Value:
        args = _extract_generic_args_from_self_annotation(annotation, class_type)
        if args is not None:
            return GenericValue(class_type, args)
        root = replace_fallback(annotation)
        if isinstance(root, TypedValue) and root.typ == class_type:
            return TypedValue(class_type)
        return default

    def _collapse_constructor_overloads_to_single_generic(
        self, signatures: Sequence[Signature], *, class_type: type | str
    ) -> Signature | None:
        if len(signatures) < 2:
            return None
        params_by_sig = [list(sig.parameters.values()) for sig in signatures]
        if not params_by_sig or any(len(params) != 1 for params in params_by_sig):
            return None
        first_param = params_by_sig[0][0]
        if any(
            param.kind is not first_param.kind or param.name != first_param.name
            for params in params_by_sig
            for param in params
        ):
            return None
        generic_default = first_param.default
        for params in params_by_sig[1:]:
            param_default = params[0].default
            if param_default is None:
                continue
            if generic_default is None:
                generic_default = param_default
            elif generic_default != param_default:
                return None

        constraints: list[type] = []
        for sig, params in zip(signatures, params_by_sig):
            ret = replace_fallback(sig.return_value)
            if (
                not isinstance(ret, GenericValue)
                or ret.typ != class_type
                or len(ret.args) != 1
            ):
                return None
            arg = replace_fallback(ret.args[0])
            if not isinstance(arg, TypedValue) or not isinstance(arg.typ, type):
                return None
            if params[0].annotation != arg:
                return None
            if arg.typ not in constraints:
                constraints.append(arg.typ)

        if len(constraints) < 2:
            return None
        class_name = (
            class_type.__name__
            if isinstance(class_type, type)
            else class_type.rsplit(".", maxsplit=1)[-1]
        )
        typevar = cast(TypeVarType, TypeVar(f"_Ctor_{class_name}_T", *constraints))
        tv_value = TypeVarValue(TypeVarParam(typevar))
        generic_param = SigParameter(
            first_param.name,
            kind=first_param.kind,
            default=generic_default,
            annotation=tv_value,
        )
        return Signature.make([generic_param], GenericValue(class_type, [tv_value]))

    def _signature_allows_init_after_new(
        self, signature: ConcreteSignature, instance_type: Value
    ) -> bool:
        signatures = (
            signature.signatures
            if isinstance(signature, OverloadedSignature)
            else [signature]
        )
        for sig in signatures:
            if not self._new_return_type_allows_init(
                sig.return_value, instance_type=instance_type
            ):
                return False
        return bool(signatures)

    def _new_return_type_allows_init(
        self, return_value: Value, *, instance_type: Value
    ) -> bool:
        if return_value is NO_RETURN_VALUE:
            return False
        saw_return_member = False
        for subvalue in flatten_values(return_value, unwrap_annotated=True):
            saw_return_member = True
            subvalue_root = replace_fallback(subvalue)
            if subvalue_root is NO_RETURN_VALUE:
                return False
            if isinstance(subvalue_root, AnyValue):
                # Any (or a union containing Any) should prevent __init__ checks.
                return False
            if not instance_type.is_assignable(subvalue, self):
                return False
        return saw_return_member

    def _runtime_has_explicit_new_return_annotation(self, typ: type) -> bool:
        direct_new = typ.__dict__.get("__new__")
        if isinstance(direct_new, (staticmethod, classmethod)):
            direct_new = direct_new.__func__
        if not isinstance(direct_new, types.FunctionType):
            return False
        runtime_annotations = safe_getattr(direct_new, "__annotations__", None)
        return isinstance(runtime_annotations, dict) and "return" in runtime_annotations

    def _synthetic_has_explicit_new_return_annotation(
        self,
        value: SyntheticClassObjectValue,
        *,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> bool:
        new_symbol = self.make_type_object(value.class_type.typ).get_declared_symbol(
            "__new__"
        )
        if new_symbol is None or not new_symbol.is_method:
            return False
        new_sig = self._get_synthetic_constructor_method_signature(
            value,
            "__new__",
            use_direct_method=True,
            bound_self_value=self._make_synthetic_constructor_instance_value(value),
            self_annotation_value=value,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        return isinstance(new_sig, (Signature, OverloadedSignature)) and (
            _signature_has_return_annotation(new_sig)
        )

    def _runtime_constructor_instance_value(self, typ: type) -> Value:
        type_params = self.arg_spec_cache.get_type_parameters(typ)
        if type_params:
            return GenericValue(typ, _apply_type_parameter_defaults(type_params, self))
        return TypedValue(typ)

    def _get_runtime_constructor_method_signature(
        self,
        typ: type,
        method_name: str,
        *,
        use_direct_method: bool,
        bound_self_value: Value,
        self_annotation_value: Value | None,
    ) -> ConcreteSignature | None:
        if use_direct_method:
            method_object = typ.__dict__.get(method_name)
        else:
            method_object = safe_getattr(typ, method_name, None)
        if isinstance(method_object, (staticmethod, classmethod)):
            method_object = method_object.__func__
        if method_object is None:
            return None
        method_sig: MaybeSignature = self._get_runtime_overloaded_method_signature(
            typ, method_name
        )
        if method_sig is None:
            method_sig = self.arg_spec_cache.get_argspec(method_object)
        binding_self_annotation = self_annotation_value
        if binding_self_annotation is not None and any(
            isinstance(subval, TypeVarValue)
            for subval in binding_self_annotation.walk_values()
        ):
            binding_self_annotation = None
        if (
            method_name == "__init__"
            and binding_self_annotation is None
            and isinstance(method_sig, (Signature, OverloadedSignature))
        ):
            source_sigs = (
                method_sig.signatures
                if isinstance(method_sig, OverloadedSignature)
                else [method_sig]
            )
            bound_sigs: list[Signature] = []
            for source_sig in source_sigs:
                params = list(source_sig.parameters.values())
                source_self_annotation = params[0].annotation if params else None
                bound = self._bind_constructor_like_signature(
                    source_sig,
                    self_value=bound_self_value,
                    self_annotation_value=source_self_annotation,
                )
                if not isinstance(bound, Signature):
                    continue
                if self._is_uninformative_constructor_signature(bound):
                    continue
                return_value = (
                    self._synthetic_constructor_return_from_self_annotation(
                        source_self_annotation, default=bound_self_value, class_type=typ
                    )
                    if source_self_annotation is not None
                    else bound_self_value
                )
                bound_sigs.append(bound.replace_return_value(return_value))
            if len(bound_sigs) == 1:
                return bound_sigs[0]
            if bound_sigs:
                return OverloadedSignature(bound_sigs)
        bound = self._bind_constructor_like_signature(
            method_sig,
            self_value=bound_self_value,
            self_annotation_value=binding_self_annotation,
        )
        if (
            bound is None
            and method_name == "__new__"
            and binding_self_annotation is not None
        ):
            bound = self._bind_constructor_like_signature(
                method_sig, self_value=bound_self_value, self_annotation_value=None
            )
        if (
            bound is not None
            and self._is_uninformative_constructor_signature(bound)
            and method_name in {"__new__", "__init__"}
        ):
            return None
        if bound is None:
            return None
        if method_name == "__init__":
            return bound.replace_return_value(bound_self_value)
        return bound

    def _get_runtime_constructor_signature(self, typ: type) -> ConcreteSignature | None:
        instance_type = self._runtime_constructor_instance_value(typ)
        has_direct_new = "__new__" in typ.__dict__
        has_direct_init = "__init__" in typ.__dict__
        inherited_init = safe_getattr(typ, "__init__", None)

        new_sig = self._get_runtime_constructor_method_signature(
            typ,
            "__new__",
            use_direct_method=has_direct_new,
            bound_self_value=instance_type,
            self_annotation_value=SubclassValue.make(instance_type),
        )
        init_sig = self._get_runtime_constructor_method_signature(
            typ,
            "__init__",
            use_direct_method=has_direct_init,
            bound_self_value=instance_type,
            self_annotation_value=instance_type,
        )

        if has_direct_new and not has_direct_init:
            init_sig = None
        elif (
            not has_direct_init
            and inherited_init is object.__init__
            and new_sig is not None
        ):
            # Builtin subclasses commonly inherit object.__init__ while exposing
            # their real constructor surface via __new__.
            init_sig = None
        elif (
            not has_direct_init
            and inherited_init is object.__init__
            and new_sig is None
        ):
            return None

        if new_sig is not None and init_sig is not None:
            if self._signature_allows_init_after_new(new_sig, instance_type):
                return init_sig
            return new_sig
        if new_sig is not None:
            return new_sig
        if init_sig is not None:
            return init_sig
        return None

    def _runtime_metaclass_call_overrides_constructor(
        self, typ: type, *, instance_type: Value
    ) -> bool:
        metaclass = type(typ)
        if metaclass is type:
            return False
        if "__call__" not in safe_getattr(metaclass, "__dict__", {}):
            return False
        call_method = safe_getattr(metaclass, "__call__", None)
        if call_method is None:
            return False
        meta_sig: MaybeSignature = self._get_runtime_overloaded_method_signature(
            metaclass, "__call__"
        )
        if meta_sig is None:
            meta_sig = self.arg_spec_cache.get_argspec(call_method)
        concrete = self._bind_constructor_like_signature(
            meta_sig,
            self_value=SubclassValue.make(instance_type),
            self_annotation_value=SubclassValue.make(instance_type),
        )
        if concrete is None or self._is_uninformative_constructor_signature(concrete):
            return False
        return not self._is_passthrough_metaclass_call_signature(
            concrete, instance_type=instance_type
        )

    def _runtime_init_self_annotation_matches(
        self,
        origin: type,
        *,
        instance_type: Value,
        typevar_map: dict[TypeVarLike, Value],
    ) -> bool:
        init_method = safe_getattr(origin, "__init__", None)
        if init_method is None:
            return True
        init_sig = self._as_concrete_signature(
            self.arg_spec_cache.get_argspec(init_method)
        )
        if init_sig is None:
            return True
        signatures = (
            init_sig.signatures
            if isinstance(init_sig, OverloadedSignature)
            else [init_sig]
        )
        checked = False
        for signature in signatures:
            params = list(signature.parameters.values())
            if not params:
                continue
            checked = True
            self_annotation = params[0].annotation.substitute_typevars(typevar_map)
            if self_annotation.is_assignable(instance_type, self):
                return True
        return not checked

    def _runtime_new_cls_annotation_matches(
        self,
        origin: type,
        *,
        class_type_value: Value,
        typevar_map: dict[TypeVarLike, Value],
    ) -> bool:
        new_method = safe_getattr(origin, "__new__", None)
        if new_method is None:
            return True
        if isinstance(new_method, types.FunctionType):
            runtime_annotations = safe_getattr(new_method, "__annotations__", None)
            try:
                runtime_sig = inspect.signature(new_method)
            except (TypeError, ValueError):
                runtime_sig = None
            if (
                isinstance(runtime_annotations, dict)
                and runtime_sig is not None
                and runtime_sig.parameters
            ):
                first_parameter_name = next(iter(runtime_sig.parameters.values())).name
                cls_runtime_annotation = runtime_annotations.get(first_parameter_name)
                if cls_runtime_annotation is not None:
                    cls_annotation = type_from_runtime(
                        cls_runtime_annotation,
                        visitor=self,
                        globals=safe_getattr(new_method, "__globals__", None),
                        suppress_errors=True,
                    ).substitute_typevars(typevar_map)
                    cls_annotation_root = replace_fallback(cls_annotation)
                    if not isinstance(cls_annotation_root, AnyValue):
                        return cls_annotation.is_assignable(class_type_value, self)
        new_sig: MaybeSignature = self._get_runtime_overloaded_method_signature(
            origin, "__new__"
        )
        if new_sig is None:
            new_sig = self.arg_spec_cache.get_argspec(new_method)
        concrete_new_sig = self._as_concrete_signature(new_sig)
        if concrete_new_sig is None:
            return True
        signatures = (
            concrete_new_sig.signatures
            if isinstance(concrete_new_sig, OverloadedSignature)
            else [concrete_new_sig]
        )
        checked = False
        for signature in signatures:
            params = list(signature.parameters.values())
            if not params:
                continue
            checked = True
            cls_annotation = params[0].annotation.substitute_typevars(typevar_map)
            if cls_annotation.is_assignable(class_type_value, self):
                return True
        return not checked

    def _synthetic_init_self_annotation_matches(
        self,
        synthetic_class: SyntheticClassObjectValue,
        *,
        instance_type: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> bool:
        init_symbol = self.make_type_object(
            synthetic_class.class_type.typ
        ).get_declared_symbol("__init__")
        has_direct_init = init_symbol is not None and init_symbol.is_method
        init_sig = self._get_synthetic_constructor_method_signature(
            synthetic_class,
            "__init__",
            use_direct_method=has_direct_init,
            bound_self_value=instance_type,
            self_annotation_value=instance_type,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        if init_sig is None:
            return True
        return not _is_incompatible_constructor_signature(init_sig)

    def _synthetic_new_cls_annotation_matches(
        self,
        synthetic_class: SyntheticClassObjectValue,
        *,
        class_type_value: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> bool:
        new_symbol = self.make_type_object(
            synthetic_class.class_type.typ
        ).get_declared_symbol("__new__")
        has_direct_new = new_symbol is not None and new_symbol.is_method
        if has_direct_new:
            method = (
                get_synthetic_member_initializer(synthetic_class, "__new__", self)
                or UNINITIALIZED_VALUE
            )
            if not isinstance(method, Value):
                return True
        else:
            method = self.get_attribute_from_value(synthetic_class, "__new__")
            if method is UNINITIALIZED_VALUE:
                return True
        method_sig = self.signature_from_value(
            method,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        concrete_sig = self._as_concrete_signature(method_sig)
        if concrete_sig is None:
            return True
        signatures = (
            concrete_sig.signatures
            if isinstance(concrete_sig, OverloadedSignature)
            else [concrete_sig]
        )
        checked = False
        for signature in signatures:
            params = list(signature.parameters.values())
            if not params:
                continue
            checked = True
            if self._synthetic_explicit_cls_annotation_matches_class(
                params[0].annotation, class_type_value=class_type_value
            ):
                return True
        return not checked

    def _synthetic_explicit_cls_annotation_matches_class(
        self, annotation: Value, *, class_type_value: Value
    ) -> bool:
        annotation_root = replace_fallback(annotation)
        class_root = replace_fallback(class_type_value)
        if isinstance(annotation_root, SubclassValue):
            original_annotation_inner = annotation_root.typ
            if isinstance(original_annotation_inner, TypeVarValue):
                return True
            annotation_root = replace_fallback(original_annotation_inner)
            if isinstance(annotation_root, (TypeVarValue, AnyValue)):
                return True
        else:
            return annotation.is_assignable(class_type_value, self)
        if isinstance(class_root, SubclassValue):
            class_root = replace_fallback(class_root.typ)
        else:
            return annotation.is_assignable(class_type_value, self)
        if (
            isinstance(annotation_root, GenericValue)
            and isinstance(class_root, GenericValue)
            and annotation_root.typ == class_root.typ
        ):
            if len(annotation_root.args) != len(class_root.args):
                return False
            for expected_arg, actual_arg in zip(annotation_root.args, class_root.args):
                expected_root = replace_fallback(expected_arg)
                if isinstance(expected_root, TypeVarValue):
                    continue
                if isinstance(
                    expected_root.can_assign(actual_arg, self), CanAssignError
                ):
                    return False
            return True
        if isinstance(annotation_root, TypedValue) and isinstance(
            class_root, TypedValue
        ):
            return annotation_root.typ == class_root.typ
        return annotation.is_assignable(class_type_value, self)

    def _synthetic_explicit_self_annotation_matches_instance(
        self, annotation: Value, *, instance_type: Value, class_type: type | str
    ) -> bool:
        expected_args = _extract_generic_args_from_self_annotation(
            annotation, class_type
        )
        if expected_args is None:
            return True
        instance_root = replace_fallback(instance_type)
        if (
            not isinstance(instance_root, GenericValue)
            or instance_root.typ != class_type
        ):
            return True
        if len(expected_args) != len(instance_root.args):
            return False
        for expected_arg, actual_arg in zip(expected_args, instance_root.args):
            expected_root = replace_fallback(expected_arg)
            if isinstance(expected_root, TypeVarValue):
                continue
            if isinstance(expected_root.can_assign(actual_arg, self), CanAssignError):
                return False
        return True

    def _infer_synthetic_type_params_from_methods(
        self, value: SyntheticClassObjectValue
    ) -> tuple[TypeParam, ...]:
        if not isinstance(value.class_type, TypedValue):
            return ()
        class_type = value.class_type.typ
        for method_name in ("__new__", "__init__"):
            method_value = get_synthetic_member_initializer(value, method_name, self)
            if not isinstance(method_value, CallableValue):
                continue
            signatures = (
                method_value.signature.signatures
                if isinstance(method_value.signature, OverloadedSignature)
                else [method_value.signature]
            )
            for signature in signatures:
                params = list(signature.parameters.values())
                if not params:
                    continue
                args = _extract_generic_args_from_self_annotation(
                    params[0].annotation, class_type
                )
                if args is not None:
                    extracted_args: list[TypeParam] = []
                    for arg in args:
                        if isinstance(arg, TypeVarValue):
                            extracted_args.append(arg.typevar_param)
                        elif isinstance(arg, TypeVarTupleValue):
                            extracted_args.append(arg.typevar_tuple_param)
                        elif isinstance(arg, InputSigValue) and isinstance(
                            arg.input_sig, ParamSpecParam
                        ):
                            extracted_args.append(arg.input_sig)
                    if len(extracted_args) == len(args):
                        return tuple(extracted_args)
        return ()

    def _infer_synthetic_type_params(
        self, value: SyntheticClassObjectValue
    ) -> tuple[TypeParam, ...]:
        inferred: list[TypeParam] = []
        seen: set[object] = set()

        def _record_type_params(candidate: Value) -> None:
            for type_param in iter_type_params_in_value(candidate):
                if type_param.typevar is SelfT or type_param.typevar in seen:
                    continue
                seen.add(type_param.typevar)
                inferred.append(type_param)

        for type_param in self._infer_synthetic_type_params_from_methods(value):
            if type_param.typevar in seen:
                continue
            seen.add(type_param.typevar)
            inferred.append(type_param)

        for base in self.make_type_object(value.class_type.typ).get_direct_bases():
            _record_type_params(base)
        return tuple(inferred)

    def _make_synthetic_constructor_instance_value(
        self, value: SyntheticClassObjectValue, *, apply_default_type_args: bool = True
    ) -> Value:
        tobj = self.make_type_object(value.class_type.typ)
        if tobj.has_any_base():
            return value.class_type
        if isinstance(value.class_type, GenericValue):
            return value.class_type
        if isinstance(value.class_type, TypedValue):
            type_params = self.get_type_parameters(value.class_type.typ)
            if not type_params:
                type_params = list(self._infer_synthetic_type_params(value))
            if type_params:
                args = (
                    _apply_type_parameter_defaults(type_params, self)
                    if apply_default_type_args
                    else [type_param_to_value(type_param) for type_param in type_params]
                )
                return GenericValue(value.class_type.typ, args)
        return value.class_type

    def _get_synthetic_constructor_method_signature(
        self,
        value: SyntheticClassObjectValue,
        method_name: str,
        *,
        use_direct_method: bool,
        bound_self_value: Value,
        self_annotation_value: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> ConcreteSignature | None:
        if use_direct_method:
            method = (
                get_synthetic_member_initializer(value, method_name, self)
                or UNINITIALIZED_VALUE
            )
            if not isinstance(method, Value):
                return None
        else:
            if method_name == "__init__":
                method = self.get_attribute_from_value(value, method_name)
            else:
                method = UNINITIALIZED_VALUE
            if method is UNINITIALIZED_VALUE:
                method = (
                    get_inherited_synthetic_member_initializer(value, method_name, self)
                    or UNINITIALIZED_VALUE
                )
            if method is UNINITIALIZED_VALUE and method_name == "__new__":
                method = self.get_attribute_from_value(value, method_name)
            if method is UNINITIALIZED_VALUE:
                return None
        method_sig = self.signature_from_value(
            method,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        if use_direct_method and method_name in {"__new__", "__init__"}:
            runtime_class = value.runtime_class
            if isinstance(runtime_class, KnownValue) and isinstance(
                runtime_class.val, type
            ):
                runtime_overloads = self._get_runtime_overloaded_method_signature(
                    runtime_class.val, method_name
                )
                if runtime_overloads is not None:
                    method_sig = runtime_overloads
            resolved_method = self.get_attribute_from_value(value, method_name)
            if resolved_method is not UNINITIALIZED_VALUE:
                resolved_sig = self.signature_from_value(
                    resolved_method,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                resolved_concrete = self._as_concrete_signature(resolved_sig)
                direct_concrete = self._as_concrete_signature(method_sig)
                if isinstance(resolved_concrete, OverloadedSignature):
                    method_sig = resolved_sig
                elif resolved_concrete is not None and (
                    direct_concrete is None
                    or self._is_uninformative_constructor_signature(direct_concrete)
                ):
                    method_sig = resolved_sig
        if (
            method_name == "__init__"
            and isinstance(value.class_type, TypedValue)
            and isinstance(method_sig, (Signature, OverloadedSignature))
        ):
            source_sigs = (
                method_sig.signatures
                if isinstance(method_sig, OverloadedSignature)
                else [method_sig]
            )
            bound_sigs: list[Signature] = []
            had_incompatible_self_annotation = False
            enforce_self_compatibility = not any(
                isinstance(subval, TypeVarValue)
                for subval in bound_self_value.walk_values()
            )
            for source_sig in source_sigs:
                params = list(source_sig.parameters.values())
                source_self_annotation = (
                    params[0].annotation if params else self_annotation_value
                )
                if (
                    params
                    and enforce_self_compatibility
                    and not self._synthetic_explicit_self_annotation_matches_instance(
                        source_self_annotation,
                        instance_type=bound_self_value,
                        class_type=value.class_type.typ,
                    )
                ):
                    had_incompatible_self_annotation = True
                    continue
                bound = self._bind_constructor_like_signature(
                    source_sig,
                    self_value=bound_self_value,
                    self_annotation_value=source_self_annotation,
                )
                if bound is None and params and enforce_self_compatibility:
                    had_incompatible_self_annotation = True
                    continue
                if not isinstance(bound, Signature):
                    continue
                if self._is_uninformative_constructor_signature(bound):
                    continue
                return_value = (
                    self._synthetic_constructor_return_from_self_annotation(
                        params[0].annotation,
                        default=bound_self_value,
                        class_type=value.class_type.typ,
                    )
                    if params
                    else bound_self_value
                )
                if (
                    isinstance(return_value, TypedValue)
                    and return_value.typ == value.class_type.typ
                    and isinstance(bound_self_value, GenericValue)
                    and class_keys_match(bound_self_value.typ, value.class_type.typ)
                    and any(
                        isinstance(arg, TypeVarValue) for arg in bound_self_value.args
                    )
                    and any(
                        isinstance(subval, TypeVarValue)
                        and any(
                            isinstance(arg, TypeVarValue)
                            and arg.typevar_param.typevar
                            is subval.typevar_param.typevar
                            for arg in bound_self_value.args
                        )
                        for param in params
                        for subval in param.annotation.walk_values()
                    )
                ):
                    return_value = bound_self_value
                bound_sigs.append(bound.replace_return_value(return_value))
            collapsed = self._collapse_constructor_overloads_to_single_generic(
                bound_sigs, class_type=value.class_type.typ
            )
            if collapsed is not None:
                return collapsed
            combined = _combine_signatures(bound_sigs)
            if combined is not None:
                return combined
            if had_incompatible_self_annotation:
                # Preserve constructor-call failure when explicit self binding fails.
                return Signature.make(
                    [
                        SigParameter(
                            "%self",
                            kind=ParameterKind.POSITIONAL_ONLY,
                            annotation=NO_RETURN_VALUE,
                        )
                    ],
                    bound_self_value,
                )
        bound = self._bind_constructor_like_signature(
            method_sig,
            self_value=bound_self_value,
            self_annotation_value=self_annotation_value,
        )
        if (
            bound is None
            and method_name == "__new__"
            and use_direct_method
            and self_annotation_value is not None
        ):
            bound = self._bind_constructor_like_signature(
                method_sig, self_value=bound_self_value, self_annotation_value=None
            )
        if (
            bound is not None
            and self._is_uninformative_constructor_signature(bound)
            and method_name in {"__new__", "__init__"}
        ):
            return None
        return bound

    def _get_synthetic_metaclass_call_signature(
        self,
        value: SyntheticClassObjectValue,
        *,
        instance_type: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> ConcreteSignature | None:
        metaclass = value.metaclass
        if not isinstance(metaclass, Value):
            return None

        if isinstance(metaclass, SyntheticClassObjectValue):
            # Ignore the default metaclass call behavior; only use an explicit override.
            meta_call = (
                get_synthetic_member_initializer(metaclass, "__call__", self)
                or UNINITIALIZED_VALUE
            )
            if meta_call is UNINITIALIZED_VALUE:
                return None
        else:
            metaclass_root = replace_fallback(metaclass)
            if isinstance(metaclass_root, KnownValue) and isinstance(
                metaclass_root.val, type
            ):
                if metaclass_root.val is type:
                    return None
                if "__call__" not in safe_getattr(metaclass_root.val, "__dict__", {}):
                    return None
            elif isinstance(metaclass_root, TypedValue) and isinstance(
                metaclass_root.typ, type
            ):
                if metaclass_root.typ is type:
                    return None
                if "__call__" not in safe_getattr(metaclass_root.typ, "__dict__", {}):
                    return None
            if get_call_attribute is not None:
                meta_call = get_call_attribute(metaclass)
            else:
                meta_call = self.get_attribute_from_value(metaclass, "__call__")
            if meta_call is UNINITIALIZED_VALUE:
                return None

        meta_sig = self.signature_from_value(
            meta_call,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        concrete = self._bind_constructor_like_signature(
            meta_sig,
            self_value=SubclassValue.make(instance_type),
            self_annotation_value=SubclassValue.make(instance_type),
        )
        if concrete is None or self._is_uninformative_constructor_signature(concrete):
            return None
        if self._is_passthrough_metaclass_call_signature(
            concrete, instance_type=instance_type
        ):
            return None
        return concrete

    def _get_synthetic_constructor_signature(
        self,
        value: SyntheticClassObjectValue,
        *,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
        apply_default_type_args: bool = True,
    ) -> ConcreteSignature:
        if (
            isinstance(value.class_type, TypedValue)
            and isinstance(value.class_type.typ, type)
            and safe_issubclass(value.class_type.typ, enum.Enum)
            and isinstance(
                enum_argspec := self._as_concrete_signature(
                    self.arg_spec_cache.get_argspec(value.class_type.typ)
                ),
                (Signature, OverloadedSignature),
            )
            and not self._is_uninformative_constructor_signature(enum_argspec)
        ):
            return enum_argspec

        type_object = self.make_type_object(value.class_type.typ)
        new_symbol = type_object.get_declared_symbol("__new__")
        init_symbol = type_object.get_declared_symbol("__init__")
        has_direct_new = new_symbol is not None and new_symbol.is_method
        has_direct_init = init_symbol is not None and init_symbol.is_method

        runtime_class = value.runtime_class or UNINITIALIZED_VALUE
        runtime_uses_default_object_constructor = (
            isinstance(runtime_class, KnownValue)
            and isinstance(runtime_class.val, type)
            and safe_getattr(runtime_class.val, "__init__", None) is object.__init__
            and safe_getattr(runtime_class.val, "__new__", None) is object.__new__
        )
        if (
            isinstance(runtime_class, KnownValue)
            and isinstance(
                runtime_argspec := self._as_concrete_signature(
                    self.arg_spec_cache.get_argspec(runtime_class.val)
                ),
                (Signature, OverloadedSignature),
            )
            and not self._is_uninformative_constructor_signature(runtime_argspec)
            and (value.is_dataclass or _signature_has_impl(runtime_argspec))
            and not has_direct_new
            and not has_direct_init
            and not (value.is_dataclass and runtime_uses_default_object_constructor)
        ):
            return runtime_argspec

        default_instance_type = self._make_synthetic_constructor_instance_value(
            value, apply_default_type_args=apply_default_type_args
        )
        instance_type = self._make_synthetic_constructor_instance_value(
            value, apply_default_type_args=False
        )
        dataclass_init_enabled = dataclass_helpers.dataclass_init_enabled(value)

        metaclass_call = self._get_synthetic_metaclass_call_signature(
            value,
            instance_type=instance_type,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        if metaclass_call is not None:
            return metaclass_call

        new_sig = self._get_synthetic_constructor_method_signature(
            value,
            "__new__",
            use_direct_method=has_direct_new,
            bound_self_value=instance_type,
            self_annotation_value=SubclassValue.make(instance_type),
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        init_sig = self._get_synthetic_constructor_method_signature(
            value,
            "__init__",
            use_direct_method=has_direct_init,
            bound_self_value=instance_type,
            self_annotation_value=instance_type,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )

        if has_direct_new and not has_direct_init:
            init_sig = None
        elif not has_direct_init:
            inherited_init = self.get_attribute_from_value(value, "__init__")
            if inherited_init is not UNINITIALIZED_VALUE:
                inherited_init_root = replace_fallback(inherited_init)
                if (
                    isinstance(inherited_init_root, KnownValue)
                    and inherited_init_root.val is object.__init__
                    and new_sig is not None
                ):
                    init_sig = None
                elif (
                    isinstance(inherited_init_root, KnownValue)
                    and inherited_init_root.val is object.__init__
                    and new_sig is None
                ):
                    return Signature.make([], default_instance_type)
        if value.is_dataclass and dataclass_init_enabled and not has_direct_init:
            dataclass_sig = dataclass_helpers.get_synthetic_constructor_signature(
                value,
                instance_type,
                get_field_parameters=self.get_synthetic_dataclass_field_parameters,
            )
            if dataclass_sig is None:
                dataclass_sig = Signature.make([], default_instance_type)
            init_sig = dataclass_sig

        if new_sig is not None and init_sig is not None:
            if self._signature_allows_init_after_new(new_sig, instance_type):
                return init_sig
            return new_sig
        if new_sig is not None:
            return new_sig
        if init_sig is not None:
            return init_sig
        if value.is_dataclass and dataclass_init_enabled:
            dataclass_sig = dataclass_helpers.get_synthetic_constructor_signature(
                value,
                instance_type,
                get_field_parameters=self.get_synthetic_dataclass_field_parameters,
            )
            if dataclass_sig is not None:
                return dataclass_sig
        if get_call_attribute is not None:
            call_method = get_call_attribute(value)
        else:
            call_method = self.get_attribute_from_value(value, "__call__")
        if call_method is not UNINITIALIZED_VALUE:
            call_sig = self.signature_from_value(
                call_method,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            concrete = self._as_concrete_signature(call_sig)
            if (
                concrete is not None
                and not self._is_uninformative_constructor_signature(concrete)
            ):
                return concrete
        return Signature.make([], default_instance_type)

    def get_synthetic_dataclass_field_parameters(
        self,
        value: SyntheticClassObjectValue,
        *,
        include_inherited: bool = True,
        seen: set[int] | None = None,
    ) -> list[SigParameter]:
        entries = self._get_synthetic_dataclass_field_entries(
            value, include_inherited=include_inherited, seen=seen
        )
        params: list[SigParameter] = []
        by_name: dict[str, SigParameter] = {}
        for entry in entries:
            param = entry.parameter
            if param.name in by_name:
                params = [
                    existing for existing in params if existing.name != param.name
                ]
            params.append(param)
            by_name[param.name] = param
        # Dataclass constructors place non-kw-only parameters before kw-only ones,
        # regardless of field declaration order.
        positional_params = [
            param for param in params if param.kind is not ParameterKind.KEYWORD_ONLY
        ]
        kw_only_params = [
            param for param in params if param.kind is ParameterKind.KEYWORD_ONLY
        ]
        return [*positional_params, *kw_only_params]

    def get_synthetic_dataclass_post_init_parameters(
        self, value: SyntheticClassObjectValue
    ) -> list[SigParameter]:
        entries = self._get_synthetic_dataclass_field_entries(
            value, include_inherited=True, seen=None
        )
        return [
            dataclass_replace(
                entry.parameter, kind=ParameterKind.POSITIONAL_OR_KEYWORD, default=None
            )
            for entry in entries
            if entry.is_initvar
        ]

    def _get_synthetic_dataclass_field_entries(
        self,
        value: SyntheticClassObjectValue,
        *,
        include_inherited: bool,
        seen: set[int] | None,
    ) -> list[_DataclassFieldEntry]:
        if seen is None:
            seen = set()
        value_id = id(value)
        if value_id in seen:
            return []
        seen.add(value_id)

        class_type = value.class_type
        if not isinstance(class_type, TypedValue):
            return []
        field_records = self._get_ordered_synthetic_dataclass_field_records(
            value, include_inherited=include_inherited
        )
        tobj = self.make_type_object(class_type.typ)

        entries: list[_DataclassFieldEntry] = []
        for record in field_records:
            symbol = tobj.get_declared_symbol_from_mro(record.field_name, self)
            if symbol is None:
                continue
            field_info = record.field_info
            excluded = symbol.is_method or symbol.is_classvar or (not field_info.init)
            if excluded:
                continue
            attr = symbol.get_effective_type()
            param_name = (
                field_info.alias if field_info.alias is not None else record.field_name
            )
            annotation = field_info.converter_input_type
            if annotation is None:
                annotation = self._synthetic_dataclass_field_annotation(attr)
            has_default = field_info.has_default
            if isinstance(attr, KnownValue):
                default: Value | None = attr if has_default else None
            else:
                default = KnownValue(...) if has_default else None
            entries.append(
                _DataclassFieldEntry(
                    parameter=SigParameter(
                        param_name,
                        (
                            ParameterKind.KEYWORD_ONLY
                            if field_info.kw_only
                            else ParameterKind.POSITIONAL_OR_KEYWORD
                        ),
                        default=default,
                        annotation=annotation,
                    ),
                    is_initvar=symbol.is_initvar,
                )
            )
        return entries

    def _get_ordered_synthetic_dataclass_field_records(
        self, value: SyntheticClassObjectValue, *, include_inherited: bool
    ) -> tuple[DataclassFieldRecord, ...]:
        ordered_names: list[str] = []
        records_by_name: dict[str, DataclassFieldRecord] = {}
        if include_inherited:
            for base_value in self.make_type_object(
                value.class_type.typ
            ).get_direct_bases():
                if not isinstance(base_value, TypedValue):
                    continue
                for record in self.make_type_object(
                    base_value.typ
                ).get_dataclass_fields():
                    if record.field_name not in records_by_name:
                        ordered_names.append(record.field_name)
                    records_by_name[record.field_name] = record

        local_field_names = value.dataclass_field_order
        type_object = self.make_type_object(value.class_type.typ)
        if not local_field_names:
            local_field_names = tuple(
                name
                for name, symbol in type_object.get_declared_symbols().items()
                if symbol.dataclass_field is not None
            )
        for field_name in local_field_names:
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
                ordered_names.append(field_name)
            records_by_name[field_name] = record
        return tuple(records_by_name[name] for name in ordered_names)

    def _synthetic_dataclass_field_annotation(self, attr: Value) -> Value:
        ctx = CheckerAttrContext(
            Composite(AnyValue(AnySource.inference)),
            None,
            "",
            self.options,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=False,
            checker=self,
        )
        descriptor_set_type = _synthetic_descriptor_set_type(attr, ctx)
        if descriptor_set_type is not None:
            return descriptor_set_type
        if isinstance(attr, KnownValue):
            return TypedValue(type(attr.val))
        return attr

    def signature_from_value(
        self,
        value: Value,
        *,
        get_return_override: Callable[[MaybeSignature], Value | None] = lambda _: None,
        get_call_attribute: Callable[[Value], Value] | None = None,
    ) -> MaybeSignature:
        if value is UNINITIALIZED_VALUE:
            return None
        if isinstance(value, PartialValue):
            sig = self._signature_from_partial_subscript(
                value,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if sig is not None:
                return sig
        if (
            isinstance(value, TypeAliasValue)
            and value.runtime_allows_value_call
            and not value.type_arguments
            and not value.alias.get_type_params()
        ):
            alias_value = value.get_value()
            # Explicit TypeAlias declarations can denote class objects (e.g.
            # `Alias: TypeAlias = list`) that should remain callable.
            if isinstance(alias_value, KnownValue) and isinstance(
                alias_value.val, type
            ):
                return self.signature_from_value(
                    alias_value,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
            if isinstance(alias_value, TypedValue) and isinstance(
                alias_value.typ, type
            ):
                return self.signature_from_value(
                    KnownValue(alias_value.typ),
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
        value = replace_fallback(value)
        if isinstance(value, KnownValue):
            origin = safe_getattr(value.val, "__origin__", None)
            args = safe_getattr(value.val, "__args__", None)
            if isinstance(origin, type) and isinstance(args, tuple):
                origin_argspec = self.signature_from_value(
                    KnownValue(origin),
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                if origin_argspec is None:
                    origin_argspec = self.arg_spec_cache.get_argspec(origin)
                if origin_argspec is not None:
                    type_params = self.arg_spec_cache.get_type_parameters(origin)
                    preserve_exact_return = (
                        self._runtime_has_explicit_new_return_annotation(origin)
                    )
                    arg_values = [
                        type_from_runtime(arg, visitor=self, suppress_errors=True)
                        for arg in args
                    ]
                    exact_arg_values = (
                        [
                            (
                                TypedValue(arg)
                                if isinstance(arg, type)
                                else KnownValue(arg)
                            )
                            for arg in args
                        ]
                        if preserve_exact_return
                        else arg_values
                    )
                    specialized_instance_type: Value
                    if exact_arg_values:
                        specialized_instance_type = GenericValue(
                            origin, exact_arg_values
                        )
                    else:
                        specialized_instance_type = TypedValue(origin)
                    typevar_map = {
                        param.typevar: arg
                        for param, arg in zip(type_params, arg_values)
                    }
                    exact_typevar_map = {
                        param.typevar: arg
                        for param, arg in zip(type_params, exact_arg_values)
                    }
                    if not self._runtime_init_self_annotation_matches(
                        origin,
                        instance_type=specialized_instance_type,
                        typevar_map=(
                            exact_typevar_map if preserve_exact_return else typevar_map
                        ),
                    ):
                        return _make_incompatible_constructor_signature(
                            specialized_instance_type
                        )
                    if not self._runtime_new_cls_annotation_matches(
                        origin,
                        class_type_value=SubclassValue.make(specialized_instance_type),
                        typevar_map=(
                            exact_typevar_map if preserve_exact_return else typevar_map
                        ),
                    ):
                        return _make_incompatible_constructor_signature(
                            specialized_instance_type
                        )
                    if typevar_map:
                        specialized_argspec = origin_argspec.substitute_typevars(
                            typevar_map
                        )
                        if (
                            preserve_exact_return
                            and exact_typevar_map
                            and exact_typevar_map != typevar_map
                        ):
                            exact_return_argspec = origin_argspec.substitute_typevars(
                                exact_typevar_map
                            )
                            specialized_argspec = _replace_signature_returns(
                                specialized_argspec, exact_return_argspec
                            )
                        return specialized_argspec
                    return origin_argspec
            cached_argspec = self.arg_spec_cache.get_argspec(value.val)
            argspec = cached_argspec
            if isinstance(value.val, type):
                runtime_instance_type = self._runtime_constructor_instance_value(
                    value.val
                )
                runtime_metaclass_overrides_constructor = (
                    self._runtime_metaclass_call_overrides_constructor(
                        value.val, instance_type=runtime_instance_type
                    )
                )
                preserve_custom_constructor = _signature_uses_custom_constructor(
                    cached_argspec, value.val
                ) and (
                    not _signature_uses_metaclass_call(cached_argspec)
                    or runtime_metaclass_overrides_constructor
                )
                should_preserve_cached_constructor = (
                    _signature_has_impl(cached_argspec)
                    or _signature_returns_typeddict(cached_argspec)
                    or preserve_custom_constructor
                )
                if not should_preserve_cached_constructor:
                    if not runtime_metaclass_overrides_constructor:
                        runtime_constructor_sig = (
                            self._get_runtime_constructor_signature(value.val)
                        )
                        if runtime_constructor_sig is not None:
                            argspec = runtime_constructor_sig
                synthetic_class = self.get_synthetic_class(value.val)
                if synthetic_class is not None:
                    type_object = self.make_type_object(synthetic_class.class_type.typ)
                    new_symbol = type_object.get_declared_symbol("__new__")
                    init_symbol = type_object.get_declared_symbol("__init__")
                    has_direct_new = new_symbol is not None and new_symbol.is_method
                    has_direct_init = init_symbol is not None and init_symbol.is_method
                    synthetic_constructor_sig = (
                        self._get_synthetic_constructor_signature(
                            synthetic_class,
                            get_return_override=get_return_override,
                            get_call_attribute=get_call_attribute,
                        )
                    )
                    concrete_argspec = (
                        self._as_concrete_signature(argspec)
                        if argspec is not None
                        else None
                    )
                    if not should_preserve_cached_constructor:
                        uses_default_object_constructor = (
                            safe_getattr(value.val, "__init__", None) is object.__init__
                            and safe_getattr(value.val, "__new__", None)
                            is object.__new__
                        )
                        synthetic_type_object = self.make_type_object(
                            synthetic_class.class_type.typ
                        )
                        if synthetic_constructor_sig is not None and (
                            synthetic_type_object.is_namedtuple_like()
                            or synthetic_class.is_dataclass
                            or has_direct_new
                            or has_direct_init
                            or argspec is None
                            or uses_default_object_constructor
                            or (
                                concrete_argspec is not None
                                and self._is_uninformative_constructor_signature(
                                    concrete_argspec
                                )
                            )
                        ):
                            argspec = synthetic_constructor_sig
                if _signature_allows_runtime_call(cached_argspec):
                    argspec = _set_signature_allow_call(
                        argspec, allow_call=True, callable_object=value.val
                    )
            if argspec is None:
                if get_call_attribute is not None:
                    method_object = get_call_attribute(value)
                else:
                    method_object = self.get_attribute_from_value(value, "__call__")
                if method_object is UNINITIALIZED_VALUE:
                    return None
                else:
                    return ANY_SIGNATURE
            if isinstance(value, KnownValueWithTypeVars):
                return argspec.substitute_typevars(value.typevars)
            return argspec
        elif isinstance(value, UnboundMethodValue):
            method = value.get_method()
            if method is not None:
                sig: MaybeSignature = None
                if value.attr_name == "__call__" and value.secondary_attr_name is None:
                    owner = self._get_unbound_method_owner(value)
                    if owner is not None:
                        sig = self._get_runtime_overloaded_method_signature(
                            owner, "__call__"
                        )
                if sig is None:
                    sig = self.arg_spec_cache.get_argspec(method)
                if sig is None:
                    # TODO return None here and figure out when the signature is missing
                    # Probably because of cythonized methods
                    return ANY_SIGNATURE
                return_override = get_return_override(sig)
                bound = make_bound_method(
                    sig, value.composite, return_override, ctx=self
                )
                if bound is not None and value.typevars is not None:
                    bound = bound.substitute_typevars(value.typevars)
                return bound
            return None
        elif isinstance(value, CallableValue):
            return value.signature
        elif isinstance(value, SyntheticClassObjectValue):
            if isinstance(value.class_type, TypedDictValue):
                params = [
                    SigParameter(
                        key,
                        ParameterKind.KEYWORD_ONLY,
                        default=None if entry.required else KnownValue(...),
                        annotation=entry.typ,
                    )
                    for key, entry in value.class_type.items.items()
                ]
                if value.class_type.extra_keys is not None:
                    params.append(
                        SigParameter(
                            "%kwargs",
                            ParameterKind.VAR_KEYWORD,
                            annotation=GenericValue(
                                dict, [TypedValue(str), value.class_type.extra_keys]
                            ),
                        )
                    )
                return Signature.make(params, value.class_type)
            constructor_sig = self._get_synthetic_constructor_signature(
                value,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if constructor_sig is not None:
                return constructor_sig
            runtime_class = value.runtime_class
            if isinstance(runtime_class, KnownValue) and isinstance(
                runtime_class.val, type
            ):
                argspec = self.arg_spec_cache.get_argspec(runtime_class.val)
                if argspec is not None:
                    return argspec
            if value.class_type.typ is tuple:
                # Probably an unknown namedtuple
                return ANY_SIGNATURE
            argspec = self.arg_spec_cache.get_argspec(
                value.class_type.typ, allow_synthetic_type=True
            )
            if argspec is None:
                init_attr = get_synthetic_member_initializer(value, "__init__", self)
                if init_attr is not None:
                    init_sig = self.signature_from_value(init_attr)
                    if isinstance(init_sig, BoundMethodSignature):
                        init_sig = init_sig.get_signature(ctx=self)
                    if isinstance(init_sig, (Signature, OverloadedSignature)):
                        bound_init = init_sig.bind_self(
                            self_annotation_value=value.class_type, ctx=self
                        )
                        if bound_init is not None:
                            return _replace_signature_return(
                                bound_init, value.class_type
                            )
                return Signature.make([ELLIPSIS_PARAM], value.class_type)
            return argspec
        elif isinstance(value, TypedValue):
            typ = value.typ
            if typ is collections.abc.Callable or typ is types.FunctionType:
                return ANY_SIGNATURE
            if isinstance(typ, str):
                synthetic_class = self.get_synthetic_class(typ)
                if synthetic_class is not None:
                    synthetic_call = get_synthetic_member_initializer(
                        synthetic_class, "__call__", self
                    )
                    call_symbol = self.make_type_object(
                        synthetic_class.class_type.typ
                    ).get_declared_symbol("__call__")
                    if (
                        synthetic_call is not None
                        and call_symbol is not None
                        and call_symbol.is_method
                    ):
                        normalized_call = normalize_synthetic_descriptor_attribute(
                            synthetic_call
                        )
                        if isinstance(normalized_call, CallableValue):
                            bound = self._bind_synthetic_method(
                                normalized_call.signature, self_annotation_value=value
                            )
                            if bound is not None:
                                return bound
                            return normalized_call.signature
                if get_call_attribute is not None:
                    call_method = get_call_attribute(value)
                else:
                    call_method = self.get_attribute_from_value(value, "__call__")
                if call_method is UNINITIALIZED_VALUE:
                    return None
                return self.signature_from_value(
                    call_method,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
            if getattr(typ.__call__, "__objclass__", None) is type and not issubclass(
                typ, type
            ):
                return None
            call_fn = typ.__call__
            sig = self._get_runtime_overloaded_method_signature(typ, "__call__")
            if sig is None:
                sig = self.arg_spec_cache.get_argspec(call_fn)
            return_override = get_return_override(sig)
            bound_method = make_bound_method(
                sig, Composite(value), return_override, ctx=self
            )
            if bound_method is None:
                return None
            return bound_method.get_signature(ctx=self)

        elif isinstance(value, SubclassValue):
            if isinstance(value.typ, TypedValue):
                if value.typ.typ is tuple:
                    # Probably an unknown namedtuple
                    return ANY_SIGNATURE
                argspec = self.arg_spec_cache.get_argspec(
                    value.typ.typ, allow_synthetic_type=True
                )
                if argspec is None:
                    return ANY_SIGNATURE
                return argspec
            else:
                typevar_bound = value.typ.typevar_param.bound
                if isinstance(typevar_bound, TypeVarValue):
                    typevar_bound = typevar_bound.get_fallback_value()
                if typevar_bound is None:
                    typevar_bound = TypedValue(object)
                bound_subclass = SubclassValue.make(typevar_bound)
                if not isinstance(bound_subclass, SubclassValue):
                    return ANY_SIGNATURE
                argspec = self.signature_from_value(
                    bound_subclass,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                if argspec is None:
                    return ANY_SIGNATURE
                return _replace_signature_return(argspec, value.typ)
        elif isinstance(value, AnyValue):
            return ANY_SIGNATURE
        elif isinstance(value, MultiValuedValue):
            sigs = [
                self.signature_from_value(
                    subval,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                for subval in value.vals
            ]
            if all(sig is not None for sig in sigs):
                # TODO we can't return a Union if we get here
                return ANY_SIGNATURE
            else:
                return None
        else:
            return None

    def _signature_from_partial_subscript(
        self,
        value: PartialValue,
        *,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> MaybeSignature:
        if value.operation is not PartialValueOperation.SUBSCRIPT:
            return None
        if not value.members:
            return None
        root = replace_fallback(value.root)
        class_type: type | str | None = None
        synthetic_root: SyntheticClassObjectValue | None = None
        preserve_exact_return = False
        if isinstance(root, KnownValue) and isinstance(root.val, type):
            class_type = root.val
            synthetic_root = self.get_synthetic_class(root.val)
            origin_argspec = self.signature_from_value(
                KnownValue(root.val),
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if origin_argspec is None:
                origin_argspec = self.arg_spec_cache.get_argspec(root.val)
            elif self._runtime_has_explicit_new_return_annotation(root.val):
                runtime_constructor_sig = self._get_runtime_constructor_signature(
                    root.val
                )
                if runtime_constructor_sig is not None:
                    origin_argspec = runtime_constructor_sig
                    preserve_exact_return = True
        elif isinstance(root, SyntheticClassObjectValue):
            synthetic_root = root
            if isinstance(root.class_type, TypedValue):
                class_type = root.class_type.typ
            origin_argspec = self.signature_from_value(
                root,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if self._synthetic_has_explicit_new_return_annotation(
                root,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            ):
                preserve_exact_return = True
            runtime_class = root.runtime_class
            if (
                isinstance(runtime_class, KnownValue)
                and isinstance(runtime_class.val, type)
                and self._runtime_has_explicit_new_return_annotation(runtime_class.val)
            ):
                runtime_constructor_sig = self._get_runtime_constructor_signature(
                    runtime_class.val
                )
                if runtime_constructor_sig is not None:
                    origin_argspec = runtime_constructor_sig
                    preserve_exact_return = True
        elif isinstance(root, TypedValue) and isinstance(root.typ, str):
            class_type = root.typ
            synthetic_class = self.get_synthetic_class(root.typ)
            if synthetic_class is None:
                return None
            origin_argspec = self.signature_from_value(
                synthetic_class,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
        elif isinstance(root, TypedValue) and isinstance(root.typ, type):
            class_type = root.typ
            synthetic_root = self.get_synthetic_class(root.typ)
            if self._runtime_has_explicit_new_return_annotation(root.typ):
                origin_argspec = self._get_runtime_constructor_signature(root.typ)
                preserve_exact_return = True
            else:
                origin_argspec = self.arg_spec_cache.get_argspec(root.typ)
        else:
            return None
        if synthetic_root is not None:
            synthetic_origin_argspec = self._get_synthetic_constructor_signature(
                synthetic_root,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
                apply_default_type_args=False,
            )
            if synthetic_origin_argspec is not None:
                origin_argspec = synthetic_origin_argspec
        if origin_argspec is None:
            return None
        if class_type is None:
            return origin_argspec
        type_params = self.get_type_parameters(class_type)
        if not type_params:
            type_params = list(_infer_type_params_from_signature(origin_argspec))
        if not type_params and synthetic_root is not None:
            type_params = list(self._infer_synthetic_type_params(synthetic_root))
        explicit_member_values = [
            type_from_value(member, self, value.node, suppress_errors=True)
            for member in value.members
        ]
        member_values = self.arg_spec_cache._specialize_generic_type_params(
            type_params, explicit_member_values
        )
        exact_member_values = (
            [
                (
                    TypedValue(member.val)
                    if isinstance(member, KnownValue) and type(member.val) is type
                    else (
                        member if isinstance(member, TypedValue) else converted_member
                    )
                )
                for member, converted_member in zip(
                    value.members, explicit_member_values
                )
            ]
            if preserve_exact_return and explicit_member_values
            else member_values
        )
        if preserve_exact_return:
            exact_member_values = [
                _promote_constructor_type_arg(member) for member in exact_member_values
            ]
        compatibility_member_values = [
            (
                member
                if (
                    isinstance(converted_member, AnyValue)
                    and converted_member.source is AnySource.error
                    and isinstance(member, TypedValue)
                )
                else exact_member
            )
            for member, converted_member, exact_member in zip(
                value.members, explicit_member_values, exact_member_values
            )
        ]
        if len(compatibility_member_values) < len(member_values):
            compatibility_member_values = [
                *compatibility_member_values,
                *member_values[len(compatibility_member_values) :],
            ]
        typevar_map = {}
        for param, member in zip(type_params, member_values):
            if isinstance(param, ParamSpecParam):
                continue
            typevar_map[param.typevar] = member
        exact_typevar_map = {}
        for param, member in zip(type_params, exact_member_values):
            if isinstance(param, ParamSpecParam):
                continue
            exact_typevar_map[param.typevar] = member
        specialized_instance_type: Value
        if compatibility_member_values:
            specialized_instance_type = GenericValue(
                class_type, compatibility_member_values
            )
        else:
            specialized_instance_type = TypedValue(class_type)
        if (
            synthetic_root is not None
            and not self._synthetic_init_self_annotation_matches(
                synthetic_root,
                instance_type=specialized_instance_type,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
        ):
            return _make_incompatible_constructor_signature(specialized_instance_type)
        if (
            synthetic_root is not None
            and not self._synthetic_new_cls_annotation_matches(
                synthetic_root,
                class_type_value=SubclassValue.make(specialized_instance_type),
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
        ):
            return _make_incompatible_constructor_signature(specialized_instance_type)
        runtime_class_for_synthetic = None
        if synthetic_root is not None and not isinstance(class_type, type):
            runtime_class = synthetic_root.runtime_class
            if isinstance(runtime_class, KnownValue) and isinstance(
                runtime_class.val, type
            ):
                runtime_class_for_synthetic = runtime_class.val
        if runtime_class_for_synthetic is not None:
            runtime_type_params = self.get_type_parameters(runtime_class_for_synthetic)
            runtime_typevar_map = {
                param.typevar: member
                for param, member in zip(runtime_type_params, member_values)
                if not isinstance(param, ParamSpecParam)
            }
            runtime_specialized_instance_type: Value
            if member_values:
                runtime_specialized_instance_type = GenericValue(
                    runtime_class_for_synthetic, member_values
                )
            else:
                runtime_specialized_instance_type = TypedValue(
                    runtime_class_for_synthetic
                )
            if not self._runtime_new_cls_annotation_matches(
                runtime_class_for_synthetic,
                class_type_value=SubclassValue.make(runtime_specialized_instance_type),
                typevar_map=runtime_typevar_map,
            ):
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
        if isinstance(class_type, type):
            if not self._runtime_init_self_annotation_matches(
                class_type,
                instance_type=specialized_instance_type,
                typevar_map=exact_typevar_map if preserve_exact_return else typevar_map,
            ):
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
            if not self._runtime_new_cls_annotation_matches(
                class_type,
                class_type_value=SubclassValue.make(specialized_instance_type),
                typevar_map=exact_typevar_map if preserve_exact_return else typevar_map,
            ):
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
        if not type_params:
            return origin_argspec
        if not typevar_map:
            return origin_argspec
        specialized_argspec = origin_argspec.substitute_typevars(typevar_map)
        if (
            preserve_exact_return
            and exact_typevar_map
            and exact_typevar_map != typevar_map
        ):
            exact_typevar_map = {
                typevar: _promote_constructor_type_arg(member)
                for typevar, member in exact_typevar_map.items()
            }
            exact_return_argspec = origin_argspec.substitute_typevars(exact_typevar_map)
            specialized_argspec = _replace_signature_returns(
                specialized_argspec, exact_return_argspec
            )
        if isinstance(specialized_argspec, OverloadedSignature):
            compatible_sigs = [
                sig
                for sig in specialized_argspec.signatures
                if specialized_instance_type.is_assignable(sig.return_value, self)
            ]
            combined = _combine_signatures(compatible_sigs)
            if combined is None:
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
            specialized_argspec = combined
        return specialized_argspec

    def _bind_synthetic_method(
        self,
        signature: ConcreteSignature,
        *,
        self_annotation_value: Value | None = None,
    ) -> ConcreteSignature | None:
        bound = signature.bind_self(
            self_value=(
                self_annotation_value
                if self_annotation_value is not None
                else AnyValue(AnySource.from_another)
            ),
            self_annotation_value=self_annotation_value,
            ctx=self,
        )
        if bound is None:
            return None
        return bound

    def _specialize_synthetic_classmethod(
        self,
        raw_attr: Value,
        normalized_attr: CallableValue,
        *,
        self_annotation_value: Value | None,
    ) -> CallableValue:
        if self_annotation_value is None:
            return normalized_attr
        if not isinstance(self_annotation_value, (TypedValue, TypeVarValue)):
            return normalized_attr
        raw_attr = replace_fallback(raw_attr)
        if not (
            isinstance(raw_attr, GenericValue)
            and raw_attr.typ is classmethod
            and raw_attr.args
        ):
            return normalized_attr
        inferred = get_tv_map(
            raw_attr.args[0], SubclassValue(self_annotation_value), self
        )
        if isinstance(inferred, CanAssignError):
            return normalized_attr
        inferred = {**inferred, SelfT: self_annotation_value}
        return CallableValue(normalized_attr.signature.substitute_typevars(inferred))

    def get_attribute_from_value(
        self, root_value: Value, attribute: str, *, prefer_typeshed: bool = False
    ) -> Value:
        lookup_root_value: Value | None = None
        if isinstance(root_value, TypeVarValue):
            lookup_root_value = root_value.get_fallback_value()
        elif isinstance(root_value, SubclassValue) and isinstance(
            root_value.typ, TypeVarValue
        ):
            lookup_root_value = SubclassValue.make(root_value.typ.get_fallback_value())
        if is_union(root_value):
            results = [
                self.get_attribute_from_value(
                    subval, attribute, prefer_typeshed=prefer_typeshed
                )
                for subval in flatten_values(root_value)
            ]
            return unite_values(*results)
        ctx = CheckerAttrContext(
            Composite(root_value),
            lookup_root_value,
            attribute,
            self.options,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=prefer_typeshed,
            checker=self,
        )
        return get_attribute(ctx)


@dataclass
class CheckerAttrContext(AttrContext):
    checker: Checker = field(repr=False)

    def get_property_type_from_argspec(self, obj: property) -> Value:
        if obj.fget is None:
            return AnyValue(AnySource.inference)
        getter = set_self(KnownValue(obj.fget), self.get_self_value())
        getter_sig = self.checker.signature_from_value(getter)
        bound = make_bound_method(getter_sig, self.root_composite, ctx=self.checker)
        if bound is None:
            return AnyValue(AnySource.inference)
        concrete = bound.get_signature(ctx=self.checker)
        if concrete is None or not concrete.has_return_value():
            return AnyValue(AnySource.inference)
        return concrete.return_value

    def resolve_name_from_typeshed(self, module: str, name: str) -> Value:
        return self.checker.ts_finder.resolve_name(module, name)

    def get_attribute_from_typeshed(self, typ: type, *, on_class: bool) -> Value:
        return self.checker.ts_finder.get_attribute(typ, self.attr, on_class=on_class)

    def get_attribute_from_typeshed_recursively(
        self, fq_name: str, *, on_class: bool
    ) -> tuple[Value, object]:
        return self.checker.ts_finder.get_attribute_recursively(
            fq_name, self.attr, on_class=on_class
        )

    def should_ignore_none_attributes(self) -> bool:
        return False

    def get_signature(self, obj: object) -> MaybeSignature:
        return self.checker.signature_from_value(KnownValue(obj))

    def signature_from_value(self, value: Value) -> MaybeSignature:
        return self.checker.signature_from_value(value)

    def get_can_assign_context(self) -> CanAssignContext:
        return self.checker

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value]
    ) -> GenericBases:
        return self.checker.get_generic_bases(typ, generic_args)

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        return self.checker.get_synthetic_class(typ)

    def should_include_synthetic_methods(self) -> bool:
        # __call__ has dedicated protocol handling in type_object; exposing
        # synthetic method values here can double-bind callable signatures.
        return self.attr != "__call__"

    def bind_synthetic_instance_attribute(self, attr_name: str, value: Value) -> Value:
        root_value = self.root_value
        resolved_root_value = replace_fallback(root_value)
        if isinstance(resolved_root_value, TypedValue):
            class_key = resolved_root_value.typ
        elif isinstance(resolved_root_value, KnownValue) and not isinstance(
            resolved_root_value.val, type
        ):
            class_key = type(resolved_root_value.val)
        else:
            return value
        tobj = self.checker.make_type_object(class_key)
        symbol = tobj.get_declared_symbol_from_mro(attr_name, self.checker)
        if symbol is None or not symbol.is_method:
            return value
        if symbol.is_staticmethod or symbol.is_classmethod:
            raw_attr = symbol.initializer
            if raw_attr is not None:
                normalized_attr = normalize_synthetic_descriptor_attribute(
                    raw_attr,
                    is_self_returning_classmethod=symbol.returns_self_on_class_access,
                    unknown_descriptor_means_any=False,
                )
                if isinstance(normalized_attr, CallableValue):
                    return self.checker._specialize_synthetic_classmethod(
                        raw_attr, normalized_attr, self_annotation_value=root_value
                    )
            return value
        signature = (
            value.signature
            if isinstance(value, CallableValue)
            else self.signature_from_value(value)
        )
        if isinstance(signature, ConcreteSignature):
            maybe_bound = self.checker._bind_synthetic_method(
                signature, self_annotation_value=self.root_value
            )
            if maybe_bound is not None:
                return CallableValue(maybe_bound)
        return value
