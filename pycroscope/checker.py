"""

The checker maintains global state that is preserved across different modules.

"""

import ast
import collections.abc
import enum
import types
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import InitVar, dataclass, field
from dataclasses import replace as dataclass_replace
from functools import cache
from itertools import chain
from typing import TypeVar, cast

from typing_extensions import assert_never

from . import dataclass as dataclass_helpers
from .analysis_lib import object_from_string
from .annotations import Context, type_from_runtime, type_from_value
from .arg_spec import ArgSpecCache, GenericBases
from .attributes import AttrContext, get_attribute
from .extensions import get_overloads as get_runtime_overloads
from .input_sig import assert_input_sig, coerce_paramspec_specialization_to_input_sig
from .node_visitor import Failure
from .options import Options
from .reexport import ImplicitReexportTracker
from .relations import infer_positional_generic_typevar_map, is_assignable
from .safe import (
    is_direct_namedtuple_class,
    is_typing_name,
    safe_getattr,
    safe_isinstance,
    safe_issubclass,
)
from .shared_options import EnforceNoUnusedCallPatterns, VariableNameValues
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    Argument,
    BoundMethodSignature,
    CheckCallContext,
    ConcreteSignature,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    _promote_constructor_type_arg,
    as_concrete_signature,
    make_bound_method,
)
from .stacked_scopes import Composite
from .suggested_type import CallableTracker
from .type_evaluation import KWARGS
from .type_object import (
    EXCLUDED_PROTOCOL_MEMBERS,
    AttributePolicy,
    TypeObject,
    class_keys_match,
    direct_bases_from_values,
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
    ClassSymbol,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    ParamSpecParam,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
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
    TypeVarMap,
    TypeVarParam,
    TypeVarType,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    _iter_typevar_map_items,
    _typevar_map_from_varlike_pairs,
    flatten_values,
    get_self_param,
    is_union,
    iter_type_params_in_value,
    replace_fallback,
    set_self,
    type_param_to_value,
    typevartuple_value_to_members,
    unite_values,
)

_SyntheticGenericBases = dict[type | str, TypeVarMap]


@cache
def _resolve_runtime_type_key(type_path: str) -> type | None:
    try:
        resolved = object_from_string(type_path)
    except Exception:
        return None
    return resolved if isinstance(resolved, type) else None


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


def _bound_method_self_value_from_typevars(
    typevars: TypeVarMap, typ: type | str
) -> Value | None:
    direct_self = typevars.get_typevar(get_self_param(typ))
    if direct_self is not None:
        return direct_self
    return None


def _apply_type_parameter_defaults(type_params: Sequence[TypeParam]) -> list[Value]:
    specialized: list[Value] = []
    substitutions = TypeVarMap()
    for type_param in type_params:
        if type_param.default is not None:
            value = type_param.default.substitute_typevars(substitutions)
        else:
            value = type_param_to_value(type_param)
        if isinstance(type_param, TypeVarParam):
            substitutions = substitutions.with_typevar(type_param, value)
        elif isinstance(type_param, ParamSpecParam):
            substitutions = substitutions.with_paramspec(
                type_param,
                assert_input_sig(coerce_paramspec_specialization_to_input_sig(value)),
            )
        else:
            substitutions = substitutions.with_typevartuple(
                type_param, typevartuple_value_to_members(value)
            )
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


def _strip_signature_deprecation(signature: MaybeSignature) -> MaybeSignature:
    return _map_maybe_signature(
        signature, lambda sig: dataclass_replace(sig, deprecated=None)
    )


# TODO: Ideally we should set self_param correctly from the beginning instead
# of patching it in later
def _set_missing_signature_self_param(
    signature: MaybeSignature, self_param: TypeVarParam
) -> MaybeSignature:
    def transform(sig: Signature) -> Signature:
        if sig.self_param is not None:
            return sig
        return dataclass_replace(sig, self_param=self_param)

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


def _signature_from_intersection_members(
    signatures: Sequence[MaybeSignature | None],
) -> MaybeSignature:
    """Return a callable signature for an intersection value.

    Non-callable members of an intersection should not block calls through the
    callable members. For example, ``Callable[..., int] & Predicate[...]`` is
    still callable and should preserve the callable branch's signature.
    """
    available_sigs = [sig for sig in signatures if sig is not None]
    if not available_sigs:
        return None
    if len(available_sigs) == 1:
        return _strip_signature_deprecation(available_sigs[0])
    concrete_sigs = list(
        chain.from_iterable(_iter_signature_variants(sig) for sig in available_sigs)
    )
    if not concrete_sigs:
        return None
    if any(sig is ANY_SIGNATURE for sig in concrete_sigs):
        return ANY_SIGNATURE
    combined = _combine_signatures(concrete_sigs)
    if combined is None:
        return None
    return _strip_signature_deprecation(combined)


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
    return as_concrete_signature(signature, ctx.get_can_assign_context())


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


def _matches_constructor_receiver_annotation(
    annotation: Value,
    actual: Value,
    ctx: CanAssignContext,
    *,
    enforce_nongeneric_match: bool,
) -> bool:
    if is_assignable(annotation, actual, ctx):
        return True
    if isinstance(annotation, (AnyValue, TypeVarValue)):
        return True
    if isinstance(annotation, SubclassValue) and isinstance(
        annotation.typ, TypeVarValue
    ):
        return True
    root = replace_fallback(annotation)
    if isinstance(root, AnyValue):
        return True
    if isinstance(root, SubclassValue):
        original_inner = root.typ
        if isinstance(original_inner, TypeVarValue):
            return True
        root = replace_fallback(original_inner)
    inferred = infer_positional_generic_typevar_map(annotation, actual, ctx)
    if inferred and is_assignable(
        annotation.substitute_typevars(inferred), actual, ctx
    ):
        return True
    actual_root = replace_fallback(actual)
    if isinstance(root, SubclassValue):
        if not isinstance(actual_root, SubclassValue):
            return False
        root = replace_fallback(root.typ)
        actual_root = replace_fallback(actual_root.typ)
    if not (
        isinstance(root, GenericValue)
        and isinstance(actual_root, GenericValue)
        and class_keys_match(root.typ, actual_root.typ)
        and len(root.args) == len(actual_root.args)
    ):
        return not enforce_nongeneric_match
    for expected_arg, actual_arg in zip(root.args, actual_root.args):
        expected_root = replace_fallback(expected_arg)
        if isinstance(expected_root, (AnyValue, TypeVarValue)):
            continue
        if not is_assignable(expected_arg, actual_arg, ctx):
            return False
    return True


@dataclass
class Checker:
    raw_options: InitVar[Options | None] = None
    options: Options = field(init=False)
    _arg_spec_cache: ArgSpecCache | None = field(default=None, init=False, repr=False)
    ts_finder: TypeshedFinder = field(init=False, repr=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False, repr=False)
    callable_tracker: CallableTracker = field(init=False, repr=False)
    should_check_unused_call_patterns: bool = field(
        default=False, init=False, repr=False
    )
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
        self.should_check_unused_call_patterns = self.options.get_value_for(
            EnforceNoUnusedCallPatterns
        )

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
        return self.callable_tracker.check(
            self,
            should_check_unused_call_patterns=self.should_check_unused_call_patterns,
        )

    def _canonical_type_object_key(self, typ: type | str) -> type | str:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None and isinstance(
            synthetic_class.class_type, TypedValue
        ):
            class_key = synthetic_class.class_type.typ
            if isinstance(class_key, type):
                return class_key
        if isinstance(typ, str):
            runtime_type = _resolve_runtime_type_key(typ)
            if runtime_type is not None:
                return runtime_type
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
        if cached is None and isinstance(typ, str) and isinstance(canonical_key, type):
            cached = self.type_object_cache.get(
                runtime_type_generic_alias(canonical_key)
            )
        return cached

    def _cache_runtime_type_object_alias(
        self, typ: type, type_object: TypeObject
    ) -> None:
        alias = runtime_type_generic_alias(typ)
        existing = self.type_object_cache.get(alias)
        if existing is None or existing.typ is typ:
            self.type_object_cache[alias] = type_object

    def make_type_object(self, typ: type | str) -> TypeObject:
        try:
            canonical_key = self._canonical_type_object_key(typ)
        except Exception:
            return TypeObject(self, typ)
        cached = self._get_cached_type_object(canonical_key)
        if cached is None:
            cached = TypeObject(self, canonical_key)
            self.type_object_cache[canonical_key] = cached
            self.type_object_cache[typ] = cached
            if isinstance(canonical_key, type):
                self._cache_runtime_type_object_alias(canonical_key, cached)
            if isinstance(typ, type):
                self._cache_runtime_type_object_alias(typ, cached)
        elif isinstance(canonical_key, type) and cached.typ is not canonical_key:
            cached.typ = canonical_key
        self.type_object_cache[canonical_key] = cached
        self.type_object_cache[typ] = cached
        if isinstance(canonical_key, type):
            self._cache_runtime_type_object_alias(canonical_key, cached)
        if isinstance(typ, type):
            self._cache_runtime_type_object_alias(typ, cached)
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
                if tv.typevar_param.is_self and current_class is not None:
                    return self.make_type_object(current_class), True
                # TODO: could be more precise
                return self.make_type_object(object), True
            case _:
                # TODO: should be assert_never(value) but our narrowing isn't good enough
                assert False

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        generic_bases = self.arg_spec_cache.get_generic_bases(typ, generic_args)
        declared_type_params = tuple(self.get_type_parameters(typ))
        substitution_map = TypeVarMap()
        specialized_args = self.arg_spec_cache._specialize_generic_type_params(
            declared_type_params, generic_args
        )
        for type_param, concrete_arg in zip(declared_type_params, specialized_args):
            if isinstance(type_param, ParamSpecParam):
                substitution_map = substitution_map.with_paramspec(
                    type_param,
                    assert_input_sig(
                        coerce_paramspec_specialization_to_input_sig(concrete_arg)
                    ),
                )
            elif isinstance(type_param, TypeVarParam):
                substitution_map = substitution_map.with_typevar(
                    type_param, concrete_arg
                )
            else:
                substitution_map = substitution_map.with_typevartuple(
                    type_param, typevartuple_value_to_members(concrete_arg)
                )

        merged: _SyntheticGenericBases = {
            base: tv_map for base, tv_map in generic_bases.items()
        }
        synthetic_bases = self._get_type_object_generic_bases(typ)
        if synthetic_bases is not None:
            for base, tv_map in synthetic_bases.items():
                substituted_tv_map = _typevar_map_from_varlike_pairs(
                    (tv, value.substitute_typevars(substitution_map))
                    for tv, value in _iter_typevar_map_items(tv_map)
                )
                if base not in merged:
                    merged[base] = TypeVarMap()
                merged[base] = merged[base].merge(substituted_tv_map)
        self._augment_namedtuple_generic_bases(typ, merged, substitution_map)
        return merged

    def get_type_parameters(self, typ: type | str) -> list[TypeParam]:
        declared_type_params = self.make_type_object(typ).get_declared_type_params()
        return list(declared_type_params)

    def register_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> None:
        class_type = synthetic_class.class_type
        if isinstance(class_type, TypedDictValue):
            return
        typ = class_type.typ
        if typ in self.synthetic_classes:
            assert self.synthetic_classes[typ] is synthetic_class, (
                f"Conflicting synthetic classes for key {typ} "
                f"(from {synthetic_class.class_type}):"
                f" {self.synthetic_classes[typ]} vs {synthetic_class}"
            )
        self.synthetic_classes[typ] = synthetic_class
        if isinstance(typ, type):
            alias = runtime_type_generic_alias(typ)
            existing = self.synthetic_classes.get(alias)
            if existing is None or existing.class_type.typ is typ:
                self.synthetic_classes[alias] = synthetic_class

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        synthetic_class = self.synthetic_classes.get(typ)
        if synthetic_class is not None:
            return synthetic_class
        if isinstance(typ, str):
            try:
                canonical_key = self._canonical_type_object_key(typ)
            except Exception:
                return None
            if isinstance(canonical_key, type):
                return self.synthetic_classes.get(
                    runtime_type_generic_alias(canonical_key)
                )
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

    def rekey_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue, old_typ: type | str
    ) -> None:
        """Register an existing synthetic class under its updated runtime type.

        This is a compatibility helper for flows that currently mutate a
        SyntheticClassObjectValue in place from one class key to another. New code
        should prefer creating the right synthetic class up front.
        """
        self.register_synthetic_class(synthetic_class)
        class_type = synthetic_class.class_type
        if isinstance(class_type, TypedDictValue):
            return
        new_typ = class_type.typ
        if old_typ == new_typ:
            return
        old_type_object = self.type_object_cache.get(old_typ)
        if old_type_object is None:
            return
        if old_type_object.typ is not new_typ:
            old_type_object.typ = new_typ
        self.type_object_cache[new_typ] = old_type_object
        if isinstance(new_typ, type):
            self._cache_runtime_type_object_alias(new_typ, old_type_object)

    def register_synthetic_type_bases(
        self,
        typ: type | str,
        base_values: Sequence[Value],
        *,
        declared_type_params: Sequence[TypeParam] = (),
    ) -> None:
        self.make_synthetic_class(typ)
        type_object = self.make_type_object(typ)
        type_object.set_direct_bases(direct_bases_from_values(base_values, self))
        if declared_type_params:
            type_object.set_declared_type_params(declared_type_params)
        else:
            type_object.clear_declared_type_params()

    def register_synthetic_protocol_members(
        self, typ: type | str, members: set[str]
    ) -> None:
        cleaned_members = {
            member
            for member in members
            if member not in EXCLUDED_PROTOCOL_MEMBERS and member != "__slots__"
        }
        type_object = self.make_type_object(typ)
        for member in cleaned_members:
            existing = type_object.get_declared_symbol(member)
            if existing is None:
                type_object.add_declared_symbol(
                    member, ClassSymbol(initializer=AnyValue(AnySource.inference))
                )

    def _get_type_object_generic_bases(
        self, typ: type | str
    ) -> _SyntheticGenericBases | None:
        if self.get_synthetic_class(typ) is None:
            return None
        generic_bases: _SyntheticGenericBases = {}
        for entry in self.make_type_object(typ).get_mro():
            if entry.is_any or entry.tobj is None:
                continue
            base_typ = entry.tobj.typ
            if (
                base_typ is object
                or is_typing_name(base_typ, "Generic")
                or is_typing_name(base_typ, "Protocol")
            ):
                continue
            generic_bases[base_typ] = entry.tv_map
        return generic_bases

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
        tuple_type: type = tuple
        tuple_type_params = self.arg_spec_cache.get_type_parameters(tuple_type)
        if len(tuple_type_params) != 1:
            return
        generic_bases[tuple_type] = generic_bases.get(
            tuple_type, TypeVarMap()
        ).with_value(tuple_type_params[0], tuple_base)

    def _namedtuple_tuple_base(self, typ: type | str) -> SequenceValue | None:
        type_object = self.make_type_object(typ)
        if not type_object.is_namedtuple_like():
            return None
        fields = tuple(type_object.get_namedtuple_fields())
        if not fields:
            return None
        return SequenceValue(tuple, [(False, field.typ) for field in fields])

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
        elif isinstance(typ, type):
            name = safe_getattr(typ, "__name__", None)
            if isinstance(name, str):
                return name
            return None
        else:
            assert_never(typ)

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
        concrete = as_concrete_signature(signature, self)
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
            return GenericValue(typ, _apply_type_parameter_defaults(type_params))
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
            if (
                use_direct_method
                and method_name == "__new__"
                and isinstance(method_object, types.FunctionType)
            ):
                inspect_sig = self.arg_spec_cache._safe_get_signature(method_object)
                if inspect_sig is not None:
                    method_sig = self.arg_spec_cache.from_signature(
                        inspect_sig,
                        function_object=method_object,
                        callable_object=method_object,
                        owner_for_self=typ,
                    )
                else:
                    method_sig = self.arg_spec_cache.get_argspec(method_object)
            else:
                method_sig = self.arg_spec_cache.get_argspec(method_object)
        if use_direct_method and method_name in {"__new__", "__init__"}:
            method_sig = _set_missing_signature_self_param(
                method_sig, get_self_param(typ)
            )
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
                source_self_annotation_for_binding = source_self_annotation
                if source_self_annotation_for_binding is not None and any(
                    isinstance(subval, TypeVarValue)
                    for subval in source_self_annotation_for_binding.walk_values()
                ):
                    source_self_annotation_for_binding = None
                bound = self._bind_constructor_like_signature(
                    source_sig,
                    self_value=bound_self_value,
                    self_annotation_value=source_self_annotation_for_binding,
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
        self, origin: type, *, instance_type: Value, typevar_map: TypeVarMap
    ) -> bool:
        init_method = safe_getattr(origin, "__init__", None)
        if init_method is None:
            return True
        init_sig = as_concrete_signature(
            self.arg_spec_cache.get_argspec(init_method), self
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
            self_annotation_root = replace_fallback(self_annotation)
            if (
                isinstance(self_annotation_root, TypeVarValue)
                and self_annotation_root.typevar_param.is_self
            ):
                return True
            if _matches_constructor_receiver_annotation(
                self_annotation, instance_type, self, enforce_nongeneric_match=False
            ):
                return True
        return not checked

    def _runtime_new_cls_annotation_matches(
        self, origin: type, *, class_type_value: Value, typevar_map: TypeVarMap
    ) -> bool:
        new_method = safe_getattr(origin, "__new__", None)
        if new_method is None:
            return True
        new_sig: MaybeSignature = self._get_runtime_overloaded_method_signature(
            origin, "__new__"
        )
        if new_sig is None:
            new_sig = self.arg_spec_cache.get_argspec(new_method)
        concrete_new_sig = as_concrete_signature(new_sig, self)
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
            if _matches_constructor_receiver_annotation(
                cls_annotation, class_type_value, self, enforce_nongeneric_match=True
            ):
                return True
        return not checked

    def _synthetic_new_cls_annotation_matches(
        self,
        synthetic_class: SyntheticClassObjectValue,
        *,
        class_type_value: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> bool:
        tobj = self.make_type_object(synthetic_class.class_type.typ)
        new_symbol = tobj.get_declared_symbol("__new__")
        has_direct_new = new_symbol is not None and new_symbol.is_method
        if has_direct_new:
            method = get_synthetic_member_initializer(tobj, "__new__")
            if method is None:
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
        concrete_sig = as_concrete_signature(method_sig, self)
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
            if _matches_constructor_receiver_annotation(
                params[0].annotation,
                class_type_value,
                self,
                enforce_nongeneric_match=True,
            ):
                return True
        return not checked

    def _make_synthetic_constructor_instance_value(
        self, value: SyntheticClassObjectValue, *, apply_default_type_args: bool = True
    ) -> Value:
        tobj = self.make_type_object(value.class_type.typ)
        if tobj.has_any_base():
            return value.class_type
        if isinstance(value.class_type, GenericValue):
            return value.class_type
        type_params = self.get_type_parameters(value.class_type.typ)
        if type_params:
            args = (
                _apply_type_parameter_defaults(type_params)
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
        tobj = self.make_type_object(value.class_type.typ)
        if use_direct_method:
            method = (
                get_synthetic_member_initializer(tobj, method_name)
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
                    get_inherited_synthetic_member_initializer(tobj, method_name, self)
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
            method_sig = _set_missing_signature_self_param(
                method_sig, get_self_param(value.class_type.typ)
            )
        binding_self_annotation = self_annotation_value
        if binding_self_annotation is not None and any(
            True for _ in iter_type_params_in_value(binding_self_annotation)
        ):
            binding_self_annotation = None
        if use_direct_method and method_name in {"__new__", "__init__"}:
            resolved_method = self.get_attribute_from_value(value, method_name)
            if resolved_method is not UNINITIALIZED_VALUE:
                resolved_sig = self.signature_from_value(
                    resolved_method,
                    get_return_override=get_return_override,
                    get_call_attribute=get_call_attribute,
                )
                resolved_concrete = as_concrete_signature(resolved_sig, self)
                direct_concrete = as_concrete_signature(method_sig, self)
                if isinstance(resolved_concrete, OverloadedSignature):
                    method_sig = resolved_sig
                elif resolved_concrete is not None and (
                    direct_concrete is None
                    or self._is_uninformative_constructor_signature(direct_concrete)
                ):
                    method_sig = resolved_sig
        if method_name == "__init__" and isinstance(
            method_sig, (Signature, OverloadedSignature)
        ):
            source_sigs = (
                method_sig.signatures
                if isinstance(method_sig, OverloadedSignature)
                else [method_sig]
            )
            bound_sigs: list[Signature] = []
            had_incompatible_self_annotation = False
            enforce_self_compatibility = binding_self_annotation is not None
            for source_sig in source_sigs:
                params = list(source_sig.parameters.values())
                source_self_annotation = (
                    params[0].annotation if params else binding_self_annotation
                )
                source_self_annotation_for_binding = source_self_annotation
                if source_self_annotation_for_binding is not None and any(
                    True
                    for _ in iter_type_params_in_value(
                        source_self_annotation_for_binding
                    )
                ):
                    source_self_annotation_for_binding = None
                if (
                    params
                    and enforce_self_compatibility
                    and source_self_annotation is not None
                    and not _matches_constructor_receiver_annotation(
                        source_self_annotation,
                        bound_self_value,
                        self,
                        enforce_nongeneric_match=False,
                    )
                ):
                    had_incompatible_self_annotation = True
                    continue
                bound = self._bind_constructor_like_signature(
                    source_sig,
                    self_value=bound_self_value,
                    self_annotation_value=source_self_annotation_for_binding,
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
            self_annotation_value=binding_self_annotation,
        )
        if (
            bound is None
            and method_name == "__new__"
            and use_direct_method
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
        return bound

    def _get_synthetic_metaclass_call_signature(
        self,
        value: SyntheticClassObjectValue,
        *,
        instance_type: Value,
        get_return_override: Callable[[MaybeSignature], Value | None],
        get_call_attribute: Callable[[Value], Value] | None,
    ) -> ConcreteSignature | None:
        tobj = value.get_type_object(self)
        metaclass = tobj.get_metaclass()
        if isinstance(metaclass, AnyValue):
            return None

        if isinstance(metaclass.typ, str):
            meta_tobj = self.make_type_object(metaclass.typ)
            # Ignore the default metaclass call behavior; only use an explicit override.
            symbol = meta_tobj.get_declared_symbol("__call__")
            if symbol is None or symbol.initializer is None:
                return None
            meta_call = symbol.initializer
        else:
            if metaclass.typ is type:
                return None
            if "__call__" not in safe_getattr(metaclass.typ, "__dict__", {}):
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
    ) -> ConcreteSignature | None:
        if (
            isinstance(value.class_type.typ, type)
            and safe_issubclass(value.class_type.typ, enum.Enum)
            and isinstance(
                enum_argspec := as_concrete_signature(
                    self.arg_spec_cache.get_argspec(value.class_type.typ), self
                ),
                (Signature, OverloadedSignature),
            )
            and not self._is_uninformative_constructor_signature(enum_argspec)
        ):
            return enum_argspec

        type_object = self.make_type_object(value.class_type.typ)
        if type_object.is_namedtuple_like():
            # Materialize namedtuple fields before reading synthetic constructor
            # symbols. This lazily installs the synthetic __new__ used for generic
            # NamedTuple specializations in fallback analysis.
            type_object.get_namedtuple_fields()
        new_symbol = type_object.get_declared_symbol("__new__")
        init_symbol = type_object.get_declared_symbol("__init__")
        has_direct_new = new_symbol is not None and new_symbol.is_method
        has_direct_init = init_symbol is not None and init_symbol.is_method

        dataclass_info = type_object.get_direct_dataclass_info()

        default_instance_type = self._make_synthetic_constructor_instance_value(
            value, apply_default_type_args=apply_default_type_args
        )
        instance_type = self._make_synthetic_constructor_instance_value(
            value, apply_default_type_args=False
        )
        dataclass_init_enabled = dataclass_info is None or dataclass_info.init

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
        if (
            dataclass_info is not None
            and dataclass_init_enabled
            and not has_direct_init
        ):
            dataclass_sig = dataclass_helpers.get_synthetic_constructor_signature(
                type_object,
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
            concrete = as_concrete_signature(call_sig, self)
            if (
                concrete is not None
                and not self._is_uninformative_constructor_signature(concrete)
            ):
                return concrete
        return Signature.make([], default_instance_type)

    def get_synthetic_dataclass_field_parameters(
        self, typ: type | str
    ) -> list[SigParameter]:
        entries = self._get_synthetic_dataclass_field_entries(typ)
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
        self, typ: type | str
    ) -> list[SigParameter]:
        entries = self._get_synthetic_dataclass_field_entries(typ)
        return [
            dataclass_replace(
                entry.parameter, kind=ParameterKind.POSITIONAL_OR_KEYWORD, default=None
            )
            for entry in entries
            if entry.is_initvar
        ]

    def _get_synthetic_dataclass_field_entries(
        self, typ: type | str
    ) -> list[_DataclassFieldEntry]:
        tobj = self.make_type_object(typ)
        field_records = tobj.get_dataclass_fields()

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

    def _synthetic_dataclass_field_annotation(self, attr: Value) -> Value:
        ctx = CheckerAttrContext(
            Composite(AnyValue(AnySource.inference)),
            None,
            "",
            self.options,
            prefer_typeshed=False,
            checker=self,
        )
        descriptor_set_type = _synthetic_descriptor_set_type(attr, ctx)
        if descriptor_set_type is not None:
            return descriptor_set_type
        if isinstance(attr, KnownValue):
            return TypedValue(type(attr.val))
        return attr

    def get_call_result(
        self,
        callee: Value,
        args: Iterable[Value] = (),
        kwargs: Iterable[tuple[str | None, Value]] = (),
        node: ast.AST | None = None,
    ) -> Value:
        sig = self.signature_from_value(callee)
        if sig is None:
            return AnyValue(AnySource.inference)
        arguments: list[Argument] = []
        for arg in args:
            arguments.append((Composite(arg), None))
        for kwarg_name, kwarg_value in kwargs:
            if kwarg_name is None:
                arguments.append((Composite(kwarg_value), KWARGS))
            else:
                arguments.append((Composite(kwarg_value), kwarg_name))
        ctx = CheckCallContext(
            can_assign_ctx=self, callee=callee, visitor=None, node=node
        )
        return sig.check_call(arguments, ctx)

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
            if (
                isinstance(value, KnownValueWithTypeVars)
                and isinstance(value.val, types.MethodType)
                and safe_isinstance(value.val.__self__, type)
            ):
                receiver_self = _bound_method_self_value_from_typevars(
                    value.typevars, value.val.__self__
                )
                if receiver_self is not None:
                    unbound_sig = self.arg_spec_cache.get_argspec(value.val.__func__)
                    if unbound_sig is not None:
                        bound = make_bound_method(
                            unbound_sig,
                            Composite(SubclassValue.make(receiver_self)),
                            get_return_override(unbound_sig),
                            ctx=self,
                        )
                        if bound is not None:
                            return bound.substitute_typevars(value.typevars)
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
                    typevar_map = _typevar_map_from_varlike_pairs(
                        zip(type_params, arg_values)
                    )
                    exact_typevar_map = _typevar_map_from_varlike_pairs(
                        zip(type_params, exact_arg_values)
                    )
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
                runtime_constructor_sig: ConcreteSignature | None = None
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
                should_preserve_cached_runtime_constructor = (
                    should_preserve_cached_constructor
                    or is_direct_namedtuple_class(value.val)
                )
                if not should_preserve_cached_runtime_constructor:
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
                    synthetic_new_symbol = (
                        type_object.get_synthetic_declared_symbols().get("__new__")
                    )
                    synthetic_init_symbol = (
                        type_object.get_synthetic_declared_symbols().get("__init__")
                    )
                    has_synthetic_direct_new = (
                        synthetic_new_symbol is not None
                        and synthetic_new_symbol.is_method
                    )
                    has_synthetic_direct_init = (
                        synthetic_init_symbol is not None
                        and synthetic_init_symbol.is_method
                    )
                    synthetic_constructor_sig = (
                        self._get_synthetic_constructor_signature(
                            synthetic_class,
                            get_return_override=get_return_override,
                            get_call_attribute=get_call_attribute,
                        )
                    )
                    concrete_argspec = (
                        as_concrete_signature(argspec, self)
                        if argspec is not None
                        else None
                    )
                    should_prefer_synthetic_constructor = not (
                        is_direct_namedtuple_class(value.val)
                        and concrete_argspec is not None
                        and not self._is_uninformative_constructor_signature(
                            concrete_argspec
                        )
                    ) and (
                        concrete_argspec is None
                        or self._is_uninformative_constructor_signature(
                            concrete_argspec
                        )
                        or (
                            synthetic_constructor_sig is not None
                            and not self._is_uninformative_constructor_signature(
                                synthetic_constructor_sig
                            )
                        )
                    )
                    if not should_preserve_cached_constructor:
                        synthetic_param_count = (
                            max(
                                len(sig.parameters)
                                for sig in _iter_signature_variants(
                                    synthetic_constructor_sig
                                )
                            )
                            if synthetic_constructor_sig is not None
                            else 0
                        )
                        runtime_param_count = (
                            max(
                                len(sig.parameters)
                                for sig in _iter_signature_variants(
                                    runtime_constructor_sig
                                )
                            )
                            if isinstance(
                                runtime_constructor_sig,
                                (Signature, OverloadedSignature),
                            )
                            else 0
                        )
                        uses_default_object_constructor = (
                            safe_getattr(value.val, "__init__", None) is object.__init__
                            and safe_getattr(value.val, "__new__", None)
                            is object.__new__
                        )
                        synthetic_type_object = self.make_type_object(
                            synthetic_class.class_type.typ
                        )
                        runtime_has_explicit_constructor = "__init__" in safe_getattr(
                            value.val, "__dict__", {}
                        ) or "__new__" in safe_getattr(value.val, "__dict__", {})
                        should_keep_runtime_constructor = (
                            isinstance(
                                runtime_constructor_sig,
                                (Signature, OverloadedSignature),
                            )
                            and not self._is_uninformative_constructor_signature(
                                runtime_constructor_sig
                            )
                            and (
                                runtime_param_count > synthetic_param_count
                                or (
                                    runtime_has_explicit_constructor
                                    and not has_synthetic_direct_new
                                    and not has_synthetic_direct_init
                                )
                            )
                        )
                        if (
                            synthetic_constructor_sig is not None
                            and should_prefer_synthetic_constructor
                            and not should_keep_runtime_constructor
                            and (
                                synthetic_type_object.is_namedtuple_like()
                                or synthetic_type_object.get_direct_dataclass_info()
                                is not None
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
            if value.secondary_attr_name == "asynq":
                primary_value = dataclass_replace(value, secondary_attr_name=None)
                primary_sig = self.signature_from_value(primary_value)
                if primary_sig is not None:
                    if value.typevars is not None:
                        primary_sig = primary_sig.substitute_typevars(value.typevars)
                    return primary_sig
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
                if value.secondary_attr_name is not None and isinstance(
                    sig, BoundMethodSignature
                ):
                    if value.typevars is not None:
                        sig = sig.substitute_typevars(value.typevars)
                    return sig
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
            if value.class_type.typ is tuple:
                # Probably an unknown namedtuple
                return ANY_SIGNATURE
            argspec = self.arg_spec_cache.get_argspec(
                value.class_type.typ, allow_synthetic_type=True
            )
            if argspec is None:
                return Signature.make([ELLIPSIS_PARAM], value.class_type)
            return argspec
        elif isinstance(value, TypedValue):
            typ = value.typ
            if typ is collections.abc.Callable or typ is types.FunctionType:
                return ANY_SIGNATURE
            if isinstance(typ, str):
                call_access = self.make_type_object(typ).get_attribute(
                    "__call__", AttributePolicy(receiver=value)
                )
                if call_access is not None and call_access.symbol.is_method:
                    if call_access.owner.typ != typ:
                        return ANY_SIGNATURE
                    direct_call_sig = self.signature_from_value(call_access.value)
                    if direct_call_sig is not None:
                        return direct_call_sig
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
        elif isinstance(value, IntersectionValue):
            return _signature_from_intersection_members(
                [
                    self.signature_from_value(
                        subval,
                        get_return_override=get_return_override,
                        get_call_attribute=get_call_attribute,
                    )
                    for subval in value.vals
                ]
            )
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
            preserve_exact_return = self._runtime_has_explicit_new_return_annotation(
                root.val
            )
        elif isinstance(root, SyntheticClassObjectValue):
            synthetic_root = root
            class_type = root.class_type.typ
            origin_argspec = self._get_synthetic_constructor_signature(
                root,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
                apply_default_type_args=False,
            )
            if origin_argspec is None:
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
        elif isinstance(root, SubclassValue) and isinstance(root.typ, TypedValue):
            class_type = root.typ.typ
            synthetic_root = self.get_synthetic_class(class_type)
            origin_argspec = self.signature_from_value(
                KnownValue(class_type),
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if origin_argspec is None:
                origin_argspec = self.arg_spec_cache.get_argspec(class_type)
            if isinstance(class_type, type):
                preserve_exact_return = (
                    self._runtime_has_explicit_new_return_annotation(class_type)
                )
        else:
            return None
        if origin_argspec is None:
            return None
        if class_type is None:
            return origin_argspec
        type_params = self.get_type_parameters(class_type)
        annotation_ctx = Context(can_assign_ctx=self, self_key=class_type)
        explicit_member_values = [
            type_from_value(
                member, node=value.node, ctx=annotation_ctx, suppress_errors=True
            )
            for member in value.members
        ]
        member_values = self.arg_spec_cache._specialize_generic_type_params(
            type_params, explicit_member_values
        )
        receiver_member_values = (
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
        exact_member_values = receiver_member_values
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
        typevar_map = TypeVarMap()
        for param, member in zip(type_params, member_values):
            if isinstance(param, ParamSpecParam):
                continue
            if isinstance(param, TypeVarParam):
                typevar_map = typevar_map.with_typevar(param, member)
            else:
                typevar_map = typevar_map.with_typevartuple(
                    param, typevartuple_value_to_members(member)
                )
        exact_typevar_map = TypeVarMap()
        for param, member in zip(type_params, exact_member_values):
            if isinstance(param, ParamSpecParam):
                continue
            if isinstance(param, TypeVarParam):
                exact_typevar_map = exact_typevar_map.with_typevar(param, member)
            else:
                exact_typevar_map = exact_typevar_map.with_typevartuple(
                    param, typevartuple_value_to_members(member)
                )
        specialized_instance_type: Value
        if compatibility_member_values:
            specialized_instance_type = GenericValue(
                class_type, compatibility_member_values
            )
        else:
            specialized_instance_type = TypedValue(class_type)
        receiver_instance_type: Value
        if receiver_member_values:
            receiver_instance_type = GenericValue(class_type, receiver_member_values)
        else:
            receiver_instance_type = TypedValue(class_type)
        if synthetic_root is not None:
            init_symbol = self.make_type_object(
                synthetic_root.class_type.typ
            ).get_declared_symbol("__init__")
            has_direct_init = init_symbol is not None and init_symbol.is_method
            init_sig = self._get_synthetic_constructor_method_signature(
                synthetic_root,
                "__init__",
                use_direct_method=has_direct_init,
                bound_self_value=receiver_instance_type,
                self_annotation_value=receiver_instance_type,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
            if init_sig is not None and _is_incompatible_constructor_signature(
                init_sig
            ):
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
        if (
            synthetic_root is not None
            and not self._synthetic_new_cls_annotation_matches(
                synthetic_root,
                class_type_value=SubclassValue.make(receiver_instance_type),
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
        ):
            return _make_incompatible_constructor_signature(specialized_instance_type)
        if isinstance(class_type, type):
            if not self._runtime_init_self_annotation_matches(
                class_type,
                instance_type=receiver_instance_type,
                typevar_map=exact_typevar_map if preserve_exact_return else typevar_map,
            ):
                return _make_incompatible_constructor_signature(
                    specialized_instance_type
                )
            if not self._runtime_new_cls_annotation_matches(
                class_type,
                class_type_value=SubclassValue.make(receiver_instance_type),
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
            exact_typevar_map = _typevar_map_from_varlike_pairs(
                (typevar, _promote_constructor_type_arg(member))
                for typevar, member in _iter_typevar_map_items(exact_typevar_map)
            )
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


def get_synthetic_member_initializer(tobj: TypeObject, name: str) -> Value | None:
    symbol = tobj.get_synthetic_declared_symbols().get(name)
    if symbol is None:
        return None
    return symbol.initializer


def get_inherited_synthetic_member_initializer(
    tobj: TypeObject,
    name: str,
    ctx: CanAssignContext,
    *,
    seen: frozenset[type | str] = frozenset(),
) -> Value | None:
    class_key = tobj.typ
    if class_key in seen:
        return None
    seen = seen | {class_key}
    direct = get_synthetic_member_initializer(tobj, name)
    if direct is not None:
        return direct
    for base_value in tobj.get_direct_bases():
        if not isinstance(base_value, TypedValue):
            continue
        base_tobj = ctx.make_type_object(base_value.typ)
        inherited = get_inherited_synthetic_member_initializer(
            base_tobj, name, ctx, seen=seen
        )
        if inherited is not None:
            if isinstance(base_value, GenericValue):
                substitutions = base_tobj.get_substitutions(base_value.args)
                if substitutions:
                    inherited = inherited.substitute_typevars(substitutions)
            return inherited
    return None
