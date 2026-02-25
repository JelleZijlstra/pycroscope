"""

The checker maintains global state that is preserved across different modules.

"""

import ast
import collections.abc
import itertools
import sys
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar, dataclass, field, replace

from .analysis_lib import override
from .annotations import type_from_runtime
from .arg_spec import ArgSpecCache, GenericBases
from .attributes import AttrContext, get_attribute
from .extensions import get_overloads as get_runtime_overloads
from .input_sig import ELLIPSIS as INPUT_SIG_ELLIPSIS
from .input_sig import (
    FullSignature,
    InputSigValue,
    ParamSpecSig,
    extract_type_params,
    wrap_type_param,
)
from .node_visitor import Failure
from .options import Options, PyObjectSequenceOption
from .reexport import ImplicitReexportTracker
from .safe import is_instance_of_typing_name, is_typing_name, safe_getattr
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
    make_bound_method,
)
from .stacked_scopes import Composite
from .suggested_type import CallableTracker
from .type_object import TypeObject, get_mro
from .typeshed import TypeshedFinder
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    GenericValue,
    HasAttrExtension,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    SubclassValue,
    SyntheticClassObjectValue,
    TypeAlias,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeVarLike,
    TypeVarMap,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    annotate_value,
    flatten_values,
    is_union,
    replace_fallback,
    unite_values,
)

_BaseProvider = Callable[[type | super], set[type]]
_SyntheticGenericBases = dict[type | str, dict[TypeVarLike, Value]]


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


@dataclass
class Checker:
    raw_options: InitVar[Options | None] = None
    options: Options = field(init=False)
    arg_spec_cache: ArgSpecCache = field(init=False, repr=False)
    ts_finder: TypeshedFinder = field(init=False, repr=False)
    reexport_tracker: ImplicitReexportTracker = field(init=False, repr=False)
    callable_tracker: CallableTracker = field(init=False, repr=False)
    type_object_cache: dict[type | super | str, TypeObject] = field(
        default_factory=dict, init=False, repr=False
    )
    synthetic_type_bases: dict[str, set[type | str]] = field(
        default_factory=dict, init=False, repr=False
    )
    synthetic_generic_bases: dict[type | str, _SyntheticGenericBases] = field(
        default_factory=dict, init=False, repr=False
    )
    synthetic_class_attributes: dict[str, dict[str, Value]] = field(
        default_factory=dict, init=False, repr=False
    )
    synthetic_protocol_members: dict[type | str, set[str]] = field(
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
    _should_exclude_any: bool = False

    def __post_init__(self, raw_options: Options | None) -> None:
        if raw_options is None:
            self.options = Options.from_option_list()
        else:
            self.options = raw_options
        self.ts_finder = TypeshedFinder.make(self, self.options)
        self.arg_spec_cache = ArgSpecCache(
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

    def get_additional_bases(self, typ: type | super) -> set[type | str]:
        bases: set[type | str] = set()
        for provider in self.options.get_value_for(AdditionalBaseProviders):
            bases |= provider(typ)
        return bases

    def make_type_object(self, typ: type | super | str) -> TypeObject:
        try:
            in_cache = typ in self.type_object_cache
        except Exception:
            return self._build_type_object(typ)
        if in_cache:
            return self.type_object_cache[typ]
        type_object = self._build_type_object(typ)
        self.type_object_cache[typ] = type_object
        return type_object

    def _build_type_object(self, typ: type | super | str) -> TypeObject:
        if isinstance(typ, str):
            # Synthetic type
            bases = self._get_typeshed_bases(typ) | self.synthetic_type_bases.get(
                typ, set()
            )
            is_protocol = any(is_typing_name(base, "Protocol") for base in bases)
            if is_protocol:
                protocol_members = self._get_protocol_members(
                    bases
                ) | self._get_synthetic_protocol_members(typ)
            else:
                protocol_members = set()
            return TypeObject(
                typ,
                bases,
                is_protocol=is_protocol,
                protocol_members=protocol_members,
                is_final=self.ts_finder.is_final(typ),
            )
        elif isinstance(typ, super):
            return TypeObject(typ, self.get_additional_bases(typ))
        else:
            plugin_bases = self.get_additional_bases(typ)
            typeshed_bases = self._get_recursive_typeshed_bases(typ)
            additional_bases = plugin_bases | typeshed_bases
            # Is it marked as a protocol in stubs? If so, use the stub definition.
            if self.ts_finder.is_protocol(typ):
                return TypeObject(
                    typ,
                    additional_bases,
                    is_protocol=True,
                    protocol_members=self._get_protocol_members(typeshed_bases),
                )
            # Is it a protocol at runtime?
            if is_instance_of_typing_name(typ, "_ProtocolMeta") and safe_getattr(
                typ, "_is_protocol", False
            ):
                bases = get_mro(typ)
                members = set(
                    itertools.chain.from_iterable(
                        _extract_protocol_members(base) for base in bases
                    )
                )
                members |= self._get_synthetic_protocol_members(typ)
                return TypeObject(
                    typ, additional_bases, is_protocol=True, protocol_members=members
                )

            is_final = self.ts_finder.is_final(typ)
            return TypeObject(typ, additional_bases, is_final=is_final)

    def _get_recursive_typeshed_bases(self, typ: type | str) -> set[type | str]:
        seen = set()
        to_do = {typ}
        result = set()
        while to_do:
            typ = to_do.pop()
            if typ in seen:
                continue
            bases = self._get_typeshed_bases(typ)
            result |= bases
            to_do |= bases
            seen.add(typ)
        return result

    def _get_typeshed_bases(self, typ: type | str) -> set[type | str]:
        base_values = self.ts_finder.get_bases_recursively(typ)
        return {base.typ for base in base_values if isinstance(base, TypedValue)}

    def _get_protocol_members(self, bases: Iterable[type | str]) -> set[str]:
        members: set[str] = set()
        for base in bases:
            members |= {
                attr
                for attr in self.ts_finder.get_all_attributes(base)
                if attr != "__slots__"
            }
            members |= self._get_synthetic_protocol_members(base)
        return members

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        generic_bases = self.arg_spec_cache.get_generic_bases(typ, generic_args)
        synthetic_bases = self._get_synthetic_generic_bases(typ)
        if synthetic_bases is None:
            return generic_bases

        substitution_map: dict[TypeVarLike, Value] = {}
        synthetic_type_params = synthetic_bases.get(typ, {})
        for i, type_param_value in enumerate(synthetic_type_params.values()):
            if not isinstance(type_param_value, TypeVarValue):
                continue
            try:
                concrete_arg = generic_args[i]
            except IndexError:
                concrete_arg = AnyValue(AnySource.generic_argument)
            substitution_map[type_param_value.typevar] = concrete_arg

        merged = {base: dict(tv_map) for base, tv_map in generic_bases.items()}
        for base, tv_map in synthetic_bases.items():
            substituted_tv_map = {
                tv: value.substitute_typevars(substitution_map)
                for tv, value in tv_map.items()
            }
            merged.setdefault(base, {}).update(substituted_tv_map)
        if isinstance(typ, type) and typ not in merged:
            alias = self._runtime_type_generic_alias(typ)
            for base, tv_map in merged.items():
                if (
                    isinstance(base, type)
                    and self._runtime_type_generic_alias(base) == alias
                ):
                    merged[typ] = dict(tv_map)
                    break
        return merged

    def get_type_parameters(self, typ: type | str) -> list[Value]:
        synthetic_bases = self._get_synthetic_generic_bases(typ)
        if synthetic_bases is not None and typ in synthetic_bases:
            return list(synthetic_bases[typ].values())
        if synthetic_bases is not None and isinstance(typ, type):
            alias = self._runtime_type_generic_alias(typ)
            for base, tv_map in synthetic_bases.items():
                if (
                    isinstance(base, type)
                    and self._runtime_type_generic_alias(base) == alias
                ):
                    return list(tv_map.values())
        return self.arg_spec_cache.get_type_parameters(typ)

    def register_synthetic_type_bases(
        self,
        typ: type | str,
        base_values: Sequence[Value],
        *,
        declared_type_params: Sequence[TypeVarValue] = (),
    ) -> None:
        base_types: set[type | str] = set()
        own_type_params: list[TypeVarLike] = [tv.typevar for tv in declared_type_params]
        merged_generic_bases: _SyntheticGenericBases = {
            typ: {tv.typevar: tv for tv in declared_type_params}
        }
        for base in base_values:
            for subval in flatten_values(replace_fallback(base)):
                converted: Value = subval
                if isinstance(converted, KnownValue):
                    converted = self.arg_spec_cache._type_from_base(converted.val)
                elif isinstance(converted, SyntheticClassObjectValue):
                    converted = converted.class_type
                if not isinstance(converted, TypedValue):
                    continue
                for type_param in extract_type_params(converted):
                    if type_param not in own_type_params:
                        own_type_params.append(type_param)
                base_typ = converted.typ
                base_types.add(base_typ)
                # Preserve direct synthetic bases even when we cannot infer a
                # richer generic mapping for them (common for local synthetic
                # classes with no typeshed entry).
                merged_generic_bases.setdefault(base_typ, {})
                generic_args = (
                    converted.args if isinstance(converted, GenericValue) else ()
                )
                for gb_typ, tv_map in self.get_generic_bases(
                    base_typ, generic_args
                ).items():
                    merged_generic_bases.setdefault(gb_typ, {}).update(tv_map)
        merged_generic_bases[typ].update(
            {
                tv: wrap_type_param(tv)
                for tv in own_type_params
                if tv not in merged_generic_bases[typ]
            }
        )

        if isinstance(typ, str) and base_types:
            self.synthetic_type_bases.setdefault(typ, set()).update(base_types)
        merged_copy = {
            gb_typ: dict(tv_map) for gb_typ, tv_map in merged_generic_bases.items()
        }
        for key in self._iter_generic_override_keys(typ):
            self.synthetic_generic_bases[key] = merged_copy
        self.type_object_cache.pop(typ, None)
        self._relation_cache.clear()

    def _get_type_parameters_for_typ(self, typ: type | str) -> list[Value]:
        type_params = self.arg_spec_cache.get_type_parameters(typ)
        if type_params:
            return type_params
        if isinstance(typ, str):
            synthetic_bases = self.synthetic_generic_bases.get(typ)
            if synthetic_bases is not None:
                return list(synthetic_bases.get(typ, {}).values())
        return []

    def register_synthetic_protocol_members(
        self, typ: type | str, members: Iterable[str]
    ) -> None:
        cleaned_members = {
            member
            for member in members
            if member not in EXCLUDED_PROTOCOL_MEMBERS and member != "__slots__"
        }
        for key in self._iter_generic_override_keys(typ):
            self.synthetic_protocol_members.setdefault(key, set()).update(
                cleaned_members
            )
            self.type_object_cache.pop(key, None)
        self._relation_cache.clear()

    def register_synthetic_class_attributes(
        self, typ: str, attributes: Mapping[str, Value]
    ) -> None:
        normalized = {
            attr: _normalize_synthetic_attribute(value)
            for attr, value in attributes.items()
        }
        self.synthetic_class_attributes.setdefault(typ, {}).update(normalized)
        self.type_object_cache.pop(typ, None)
        self._relation_cache.clear()

    @staticmethod
    def _runtime_type_generic_alias(typ: type) -> str:
        return f"{typ.__module__}.{typ.__qualname__}"

    def _iter_generic_override_keys(self, typ: type | str) -> Iterator[type | str]:
        yield typ
        if isinstance(typ, type):
            yield self._runtime_type_generic_alias(typ)

    def _get_synthetic_generic_bases(
        self, typ: type | str
    ) -> _SyntheticGenericBases | None:
        for key in self._iter_generic_override_keys(typ):
            bases = self.synthetic_generic_bases.get(key)
            if bases is not None:
                if isinstance(typ, type) and isinstance(key, str):
                    declared = bases.get(key)
                    if not declared or not any(
                        isinstance(val, TypeVarValue) for val in declared.values()
                    ):
                        continue
                return bases
        return None

    def _get_synthetic_protocol_members(self, typ: type | str) -> set[str]:
        members: set[str] = set()
        if isinstance(typ, type):
            members |= self.synthetic_protocol_members.get(typ, set())
            members |= self.synthetic_protocol_members.get(
                self._runtime_type_generic_alias(typ), set()
            )
            return members
        return self.synthetic_protocol_members.get(typ, set())

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
    ) -> Iterator[None]:
        """Context manager that notes that left and right can be assumed to be compatible."""
        pair = (left, right)
        self.assumed_compatibilities.append(pair)
        try:
            yield
        finally:
            new_pair = self.assumed_compatibilities.pop()
            assert pair == new_pair

    def can_aliases_assume_compatibility(
        self, left: "TypeAliasValue", right: "TypeAliasValue"
    ) -> bool:
        return (left, right) in self.alias_assumed_compatibilities

    @contextmanager
    def aliases_assume_compatibility(
        self, left: "TypeAliasValue", right: "TypeAliasValue"
    ) -> Iterator[None]:
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

    def set_exclude_any(self) -> AbstractContextManager[None]:
        """Within this context, `Any` is compatible only with itself."""
        return override(self, "_should_exclude_any", True)

    def should_exclude_any(self) -> bool:
        """Whether Any should be compatible only with itself."""
        return self._should_exclude_any

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

    def signature_from_value(
        self,
        value: Value,
        *,
        get_return_override: Callable[[MaybeSignature], Value | None] = lambda _: None,
        get_call_attribute: Callable[[Value], Value] | None = None,
    ) -> MaybeSignature:
        if value is UNINITIALIZED_VALUE:
            return None
        if isinstance(value, AnnotatedValue):
            return self.signature_from_value(
                value.value,
                get_return_override=get_return_override,
                get_call_attribute=get_call_attribute,
            )
        value = replace_fallback(value)
        if isinstance(value, KnownValue):
            origin = safe_getattr(value.val, "__origin__", None)
            args = safe_getattr(value.val, "__args__", None)
            if isinstance(origin, type) and isinstance(args, tuple):
                origin_argspec = self.arg_spec_cache.get_argspec(origin)
                if origin_argspec is not None:
                    type_params = self.arg_spec_cache.get_type_parameters(origin)
                    arg_values = [
                        type_from_runtime(arg, visitor=self, suppress_errors=True)
                        for arg in args
                    ]
                    typevar_map = self._make_typevar_map_from_type_parameters(
                        type_params, arg_values
                    )
                    if typevar_map:
                        return origin_argspec.substitute_typevars(typevar_map)
                    return origin_argspec
            argspec = self.arg_spec_cache.get_argspec(value.val)
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
            runtime_class = value.class_attributes.get("%runtime_class")
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
                return Signature.make(
                    [ELLIPSIS_PARAM], self._make_synthetic_class_instance_value(value)
                )
            return argspec
        elif isinstance(value, TypedValue):
            typ = value.typ
            if typ is collections.abc.Callable or typ is types.FunctionType:
                return ANY_SIGNATURE
            if isinstance(typ, str):
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
                # TODO generic SubclassValue
                return ANY_SIGNATURE
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

    def _synthetic_class_has_any_base(self, value: SyntheticClassObjectValue) -> bool:
        return any(self._synthetic_base_is_any(base) for base in value.base_classes)

    def _synthetic_base_is_any(self, base: Value) -> bool:
        base = replace_fallback(base)
        if isinstance(base, MultiValuedValue):
            return any(self._synthetic_base_is_any(subval) for subval in base.vals)
        if isinstance(base, AnyValue):
            return True
        if isinstance(base, KnownValue):
            return is_typing_name(base.val, "Any")
        if isinstance(base, TypedValue):
            return is_typing_name(base.typ, "Any")
        return False

    def _make_synthetic_class_instance_value(
        self, value: SyntheticClassObjectValue
    ) -> Value:
        if self.make_type_object(value.class_type.typ).is_protocol:
            metadata: list[HasAttrExtension] = []
        else:
            metadata = [
                HasAttrExtension(
                    KnownValue(name), self._make_any_base_attribute(name, attr)
                )
                for name, attr in value.class_attributes.items()
                if not name.startswith("%")
            ]
        if self._synthetic_class_has_any_base(value):
            instance: Value = AnyValue(AnySource.from_another)
        else:
            instance = value.class_type
        if metadata:
            return annotate_value(instance, metadata)
        return instance

    def _make_any_base_attribute(self, name: str, attr: Value) -> Value:
        if isinstance(attr, CallableValue):
            if _is_dunder(name):
                widened = _widen_synthetic_dunder_signature(attr.signature)
                if widened is not attr.signature:
                    return CallableValue(widened, attr.typ)
                return attr
            maybe_bound = self._bind_synthetic_method(attr.signature)
            if maybe_bound is not None:
                return CallableValue(maybe_bound)
            return attr
        return _normalize_synthetic_attribute(attr)

    def _bind_synthetic_method(
        self, signature: ConcreteSignature
    ) -> ConcreteSignature | None:
        def _first_parameter_name(sig: Signature) -> str | None:
            return next(iter(sig.parameters.values())).name if sig.parameters else None

        if isinstance(signature, Signature):
            if _first_parameter_name(signature) not in {"self", "cls"}:
                return None
        elif isinstance(signature, OverloadedSignature):
            if not all(
                _first_parameter_name(sig) in {"self", "cls"}
                for sig in signature.signatures
            ):
                return None
        bound = make_bound_method(
            signature, Composite(AnyValue(AnySource.from_another)), ctx=self
        )
        if bound is None:
            return None
        return bound.get_signature(ctx=self)

    def get_attribute_from_value(
        self, root_value: Value, attribute: str, *, prefer_typeshed: bool = False
    ) -> Value:
        if isinstance(root_value, TypeVarValue):
            root_value = root_value.get_fallback_value()
        if is_union(root_value):
            results = [
                self.get_attribute_from_value(
                    subval, attribute, prefer_typeshed=prefer_typeshed
                )
                for subval in flatten_values(root_value)
            ]
            return unite_values(*results)
        if isinstance(root_value, TypedValue) and isinstance(root_value.typ, str):
            synthetic = self._get_synthetic_instance_attribute(
                root_value.typ, attribute, root_value=root_value
            )
            if synthetic is not None:
                return synthetic
        ctx = CheckerAttrContext(
            Composite(root_value),
            attribute,
            self.options,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=prefer_typeshed,
            checker=self,
        )
        return get_attribute(ctx)

    def _make_typevar_map_from_type_parameters(
        self, type_params: Sequence[Value], args: Sequence[Value]
    ) -> TypeVarMap:
        tv_map: dict[TypeVarLike, Value] = {}
        arg_index_by_key: dict[tuple[str, str], int] = {}
        next_arg_index = 0
        for type_param in type_params:
            if isinstance(type_param, TypeVarValue):
                semantic_key = ("typevar", type_param.typevar.__name__)
            elif isinstance(type_param, InputSigValue) and isinstance(
                type_param.input_sig, ParamSpecSig
            ):
                semantic_key = ("paramspec", type_param.input_sig.param_spec.__name__)
            else:
                continue
            if semantic_key not in arg_index_by_key:
                if next_arg_index >= len(args):
                    continue
                arg_index_by_key[semantic_key] = next_arg_index
                next_arg_index += 1
            arg = args[arg_index_by_key[semantic_key]]
            if isinstance(type_param, TypeVarValue):
                tv_map[type_param.typevar] = arg
            elif isinstance(type_param, InputSigValue) and isinstance(
                type_param.input_sig, ParamSpecSig
            ):
                if isinstance(arg, AnyValue):
                    normalized_arg: Value = InputSigValue(INPUT_SIG_ELLIPSIS)
                elif isinstance(arg, CallableValue) and isinstance(
                    arg.signature, Signature
                ):
                    normalized_arg = InputSigValue(FullSignature(arg.signature))
                else:
                    normalized_arg = arg
                tv_map[type_param.input_sig.param_spec] = normalized_arg
        return tv_map

    def _get_synthetic_instance_attribute(
        self, typ: str, attribute: str, *, root_value: TypedValue | None = None
    ) -> Value | None:
        seen: set[str] = set()
        to_visit = [typ]
        while to_visit:
            current = to_visit.pop()
            if current in seen:
                continue
            seen.add(current)
            attrs = self.synthetic_class_attributes.get(current)
            if attrs is not None and attribute in attrs:
                value = attrs[attribute]
                if isinstance(root_value, GenericValue) and root_value.typ == typ:
                    type_params = self._get_type_parameters_for_typ(typ)
                    tv_map = self._make_typevar_map_from_type_parameters(
                        type_params, root_value.args
                    )
                    if tv_map:
                        value = value.substitute_typevars(tv_map)
                if isinstance(value, CallableValue) and (
                    attribute == "__call__" or not _is_dunder(attribute)
                ):
                    bound = self._bind_synthetic_method(value.signature)
                    if bound is not None:
                        return CallableValue(bound)
                return value
            for base in self.synthetic_type_bases.get(current, ()):
                if isinstance(base, str):
                    to_visit.append(base)
        return None

    def get_synthetic_instance_attribute(
        self, typ: str | TypedValue, attribute: str
    ) -> Value | None:
        if isinstance(typ, TypedValue):
            if isinstance(typ.typ, str):
                return self._get_synthetic_instance_attribute(
                    typ.typ, attribute, root_value=typ
                )
            return None
        return self._get_synthetic_instance_attribute(typ, attribute)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _widen_synthetic_dunder_signature(
    signature: ConcreteSignature,
) -> ConcreteSignature:
    def _widen(sig: Signature) -> Signature:
        if not sig.parameters:
            return sig
        first_name = next(iter(sig.parameters))
        first_param = sig.parameters[first_name]
        if first_param.name not in {"self", "cls"}:
            return sig
        widened_first = replace(
            first_param, annotation=AnyValue(AnySource.from_another)
        )
        if widened_first == first_param:
            return sig
        params = dict(sig.parameters)
        params[first_name] = widened_first
        return replace(sig, parameters=params)

    if isinstance(signature, Signature):
        return _widen(signature)
    if isinstance(signature, OverloadedSignature):
        widened = [_widen(sig) for sig in signature.signatures]
        if all(new is old for new, old in zip(widened, signature.signatures)):
            return signature
        return OverloadedSignature(widened)
    return signature


def _normalize_synthetic_attribute(attr: Value) -> Value:
    if isinstance(attr, AnyValue) and attr.source is AnySource.explicit:
        return AnyValue(AnySource.from_another)
    if isinstance(attr, GenericValue):
        new_args = tuple(_normalize_synthetic_attribute(arg) for arg in attr.args)
        if new_args != attr.args:
            return GenericValue(attr.typ, new_args)
    if isinstance(attr, MultiValuedValue):
        new_vals = tuple(_normalize_synthetic_attribute(val) for val in attr.vals)
        if new_vals != tuple(attr.vals):
            return unite_values(*new_vals)
    return attr


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


def _extract_protocol_members(typ: type) -> set[str]:
    if (
        typ is object
        or is_typing_name(typ, "Generic")
        or is_typing_name(typ, "Protocol")
    ):
        return set()
    members: set[str] = set(typ.__dict__) - EXCLUDED_PROTOCOL_MEMBERS
    # Starting in 3.10 __annotations__ always exists on types
    if sys.version_info >= (3, 10) or hasattr(typ, "__annotations__"):
        members |= set(typ.__annotations__)
    return members


@dataclass
class CheckerAttrContext(AttrContext):
    checker: Checker = field(repr=False)

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

    def get_signature(self, obj: object) -> MaybeSignature:
        return self.checker.signature_from_value(KnownValue(obj))

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value]
    ) -> GenericBases:
        return self.checker.get_generic_bases(typ, generic_args)

    def get_type_parameters(self, typ: type | str) -> list[Value]:
        return self.checker.get_type_parameters(typ)
