"""

The checker maintains global state that is preserved across different modules.

"""

import ast
import collections.abc
import enum
import inspect
import itertools
import sys
import types
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar, dataclass, field
from dataclasses import replace as dataclass_replace
from typing import TypeVar

from .analysis_lib import override
from .annotations import type_from_runtime, type_from_value
from .arg_spec import ArgSpecCache, GenericBases
from .attributes import (
    AttrContext,
    _synthetic_dataclass_converter_input_types,
    _synthetic_dataclass_parameter_annotation_for_field,
    get_attribute,
    normalize_synthetic_descriptor_attribute,
)
from .extensions import get_overloads as get_runtime_overloads
from .input_sig import (
    InputSigValue,
    ParamSpecSig,
    coerce_paramspec_specialization_to_input_sig,
)
from .node_visitor import Failure
from .options import Options, PyObjectSequenceOption
from .reexport import ImplicitReexportTracker
from .safe import (
    is_instance_of_typing_name,
    is_namedtuple_class,
    is_typing_name,
    safe_getattr,
    safe_issubclass,
)
from .shared_options import VariableNameValues
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    BoundMethodSignature,
    ConcreteSignature,
    InvalidSignature,
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
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    GenericValue,
    HasAttrExtension,
    KnownValue,
    KnownValueWithTypeVars,
    MultiValuedValue,
    PartialValue,
    PartialValueOperation,
    SubclassValue,
    SyntheticClassObjectValue,
    TypeAlias,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeVarLike,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    annotate_value,
    flatten_values,
    get_tv_map,
    has_any_base_value,
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


@dataclass(frozen=True)
class _DataclassFieldEntry:
    field_name: str
    parameter: SigParameter
    is_initvar: bool


def _runtime_type_generic_alias(typ: type) -> str:
    return f"{typ.__module__}.{typ.__qualname__}"


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


def _signature_has_return_annotation(signature: ConcreteSignature) -> bool:
    if isinstance(signature, Signature):
        return signature.has_return_annotation
    return all(sig.has_return_annotation for sig in signature.signatures)


def _combine_signatures(signatures: Sequence[Signature]) -> ConcreteSignature | None:
    if not signatures:
        return None
    if len(signatures) == 1:
        return signatures[0]
    return OverloadedSignature(list(signatures))


def _signature_has_no_parameters(signature: ConcreteSignature) -> bool:
    signatures = (
        signature.signatures
        if isinstance(signature, OverloadedSignature)
        else [signature]
    )
    return all(not sig.parameters for sig in signatures)


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
    if isinstance(root, GenericValue) and root.typ == class_type:
        return tuple(root.args)
    return None


def _synthetic_dataclass_init_enabled(value: SyntheticClassObjectValue) -> bool:
    raw_flag = value.class_attributes.get("%dataclass_init")
    if isinstance(raw_flag, KnownValue) and isinstance(raw_flag.val, bool):
        return raw_flag.val
    return True


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
            bases = self._get_typeshed_bases(typ)
            synthetic_class = self.get_synthetic_class(typ)
            if synthetic_class is not None:
                bases |= self._get_type_bases_from_synthetic_class(synthetic_class)
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
        members = {
            attr
            for base in bases
            for attr in self.ts_finder.get_all_attributes(base)
            if attr != "__slots__"
        }
        for base in bases:
            members |= self._get_synthetic_protocol_members(base)
        return members

    def _get_type_bases_from_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> set[type | str]:
        base_types: set[type | str] = set()
        for base in synthetic_class.base_classes:
            for subval in flatten_values(replace_fallback(base)):
                converted: Value = subval
                if isinstance(converted, KnownValue):
                    converted = self.arg_spec_cache._type_from_base(converted.val)
                elif isinstance(converted, SyntheticClassObjectValue):
                    converted = converted.class_type
                if isinstance(converted, TypedValue):
                    base_types.add(converted.typ)
        return base_types

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        generic_bases = self.arg_spec_cache.get_generic_bases(typ, generic_args)
        synthetic_bases = self._get_synthetic_generic_bases(typ)
        if synthetic_bases is None:
            return generic_bases

        substitution_map: dict[TypeVarLike, Value] = {}
        synthetic_type_params = synthetic_bases.get(typ, {})
        if not synthetic_type_params:
            synthetic_class = self.get_synthetic_class(typ)
            if synthetic_class is not None and isinstance(
                synthetic_class.class_type, TypedValue
            ):
                synthetic_type_params = synthetic_bases.get(
                    synthetic_class.class_type.typ, {}
                )
        specialized_args = self.arg_spec_cache._specialize_generic_type_params(
            tuple(synthetic_type_params.values()), generic_args
        )
        for type_param_value, concrete_arg in zip(
            synthetic_type_params.values(), specialized_args
        ):
            if isinstance(type_param_value, TypeVarValue):
                type_param: TypeVarLike = type_param_value.typevar
                is_paramspec = is_instance_of_typing_name(type_param, "ParamSpec")
            elif isinstance(type_param_value, InputSigValue) and isinstance(
                type_param_value.input_sig, ParamSpecSig
            ):
                type_param = type_param_value.input_sig.param_spec
                is_paramspec = True
            else:
                continue
            if is_paramspec:
                concrete_arg = coerce_paramspec_specialization_to_input_sig(
                    concrete_arg
                )
            substitution_map[type_param] = concrete_arg

        merged = {base: dict(tv_map) for base, tv_map in generic_bases.items()}
        for base, tv_map in synthetic_bases.items():
            substituted_tv_map = {
                tv: value.substitute_typevars(substitution_map)
                for tv, value in tv_map.items()
            }
            merged.setdefault(base, {}).update(substituted_tv_map)
        return merged

    def get_type_parameters(self, typ: type | str) -> list[Value]:
        synthetic_bases = self._get_synthetic_generic_bases(typ)
        if synthetic_bases is not None and typ in synthetic_bases:
            return list(synthetic_bases[typ].values())
        if synthetic_bases is not None:
            synthetic_class = self.get_synthetic_class(typ)
            if synthetic_class is not None and isinstance(
                synthetic_class.class_type, TypedValue
            ):
                declared = synthetic_bases.get(synthetic_class.class_type.typ)
                if declared is not None:
                    return list(declared.values())
        return self.arg_spec_cache.get_type_parameters(typ)

    def register_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> None:
        class_type = synthetic_class.class_type
        if not isinstance(class_type, TypedValue):
            return
        typ = class_type.typ
        for key in self._iter_generic_override_keys(typ):
            self.synthetic_classes[key] = synthetic_class
            self.type_object_cache.pop(key, None)

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        for key in self._iter_generic_override_keys(typ):
            synthetic_class = self.synthetic_classes.get(key)
            if synthetic_class is not None:
                return synthetic_class
        return None

    def _ensure_synthetic_class(
        self, typ: type | str, *, base_values: Sequence[Value] = ()
    ) -> SyntheticClassObjectValue:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None:
            return synthetic_class
        if isinstance(typ, str):
            name = typ.rsplit(".", 1)[-1]
        else:
            name = typ.__name__
        synthetic_class = SyntheticClassObjectValue(
            name, TypedValue(typ), base_classes=tuple(base_values)
        )
        self.register_synthetic_class(synthetic_class)
        return synthetic_class

    def register_synthetic_type_bases(
        self,
        typ: type | str,
        base_values: Sequence[Value],
        *,
        declared_type_params: Sequence[TypeVarValue] = (),
    ) -> None:
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
                base_typ = converted.typ
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

        synthetic_class = self._ensure_synthetic_class(typ, base_values=base_values)
        merged_copy = {
            gb_typ: dict(tv_map) for gb_typ, tv_map in merged_generic_bases.items()
        }
        synthetic_class.generic_bases.clear()
        synthetic_class.generic_bases.update(merged_copy)
        for key in self._iter_generic_override_keys(typ):
            self.type_object_cache.pop(key, None)

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
        for member in cleaned_members:
            synthetic_class.class_attributes.setdefault(
                member, AnyValue(AnySource.inference)
            )
        for key in self._iter_generic_override_keys(typ):
            self.type_object_cache.pop(key, None)

    def _iter_generic_override_keys(self, typ: type | str) -> Iterator[type | str]:
        yield typ
        if isinstance(typ, type):
            yield _runtime_type_generic_alias(typ)

    def _get_synthetic_generic_bases(
        self, typ: type | str
    ) -> _SyntheticGenericBases | None:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is not None and synthetic_class.generic_bases:
            return {
                base_typ: dict(tv_map)
                for base_typ, tv_map in synthetic_class.generic_bases.items()
            }
        return None

    def _get_synthetic_protocol_members(self, typ: type | str) -> set[str]:
        synthetic_class = self.get_synthetic_class(typ)
        if synthetic_class is None:
            return set()
        return {
            member
            for member in synthetic_class.class_attributes
            if member not in EXCLUDED_PROTOCOL_MEMBERS
            and member != "__slots__"
            and not member.startswith("%")
        }

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
        self, left: "TypeAliasValue", right: "TypeAliasValue"
    ) -> bool:
        return (left, right) in self.alias_assumed_compatibilities

    @contextmanager
    def aliases_assume_compatibility(
        self, left: "TypeAliasValue", right: "TypeAliasValue"
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
            if isinstance(replace_fallback(sig.return_value), TypeVarValue):
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
            return signature.get_signature(ctx=self)
        concrete = self._as_concrete_signature(signature)
        if concrete is None:
            return None
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
        typevar = TypeVar(f"_Ctor_{class_name}_T", *constraints)
        tv_value = TypeVarValue(typevar)
        generic_param = SigParameter(
            first_param.name,
            kind=first_param.kind,
            default=first_param.default,
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
        if "__new__" not in value.method_attributes:
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
        if not type_params:
            generic_bases = self.arg_spec_cache.get_generic_bases(
                typ, substitute_typevars=False
            )
            type_params = [
                val
                for val in generic_bases.get(typ, {}).values()
                if isinstance(val, TypeVarValue)
            ]
        if type_params:
            return GenericValue(typ, list(type_params))
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
        bound = self._bind_constructor_like_signature(
            method_sig,
            self_value=bound_self_value,
            self_annotation_value=self_annotation_value,
        )
        if (
            bound is None
            and method_name == "__new__"
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
        if bound is None:
            return None
        if method_name == "__init__":
            return bound.replace_return_value(bound_self_value)
        return bound

    def _get_runtime_constructor_signature(self, typ: type) -> ConcreteSignature | None:
        instance_type = self._runtime_constructor_instance_value(typ)
        has_direct_new = "__new__" in typ.__dict__
        has_direct_init = "__init__" in typ.__dict__

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
        has_direct_init = "__init__" in synthetic_class.method_attributes
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
        has_direct_new = "__new__" in synthetic_class.method_attributes
        if has_direct_new:
            method = synthetic_class.class_attributes.get(
                "__new__", UNINITIALIZED_VALUE
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
            annotation_root = replace_fallback(annotation_root.typ)
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
    ) -> tuple[Value, ...]:
        if not isinstance(value.class_type, TypedValue):
            return ()
        class_type = value.class_type.typ
        for method_name in ("__new__", "__init__"):
            method_value = value.class_attributes.get(method_name)
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
                if args is not None and all(
                    isinstance(arg, TypeVarValue) for arg in args
                ):
                    return args
        return ()

    def _make_synthetic_constructor_instance_value(
        self, value: SyntheticClassObjectValue
    ) -> Value:
        if self._synthetic_class_has_any_base(value):
            return self._make_synthetic_class_instance_value(value)
        if isinstance(value.class_type, GenericValue):
            return value.class_type
        if isinstance(value.class_type, TypedValue):
            type_params = self.arg_spec_cache.get_type_parameters(value.class_type.typ)
            if not type_params:
                type_params = list(
                    self._infer_synthetic_type_params_from_methods(value)
                )
            if type_params:
                return GenericValue(value.class_type.typ, type_params)
        return self._make_synthetic_class_instance_value(value)

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
            method = value.class_attributes.get(method_name, UNINITIALIZED_VALUE)
            if not isinstance(method, Value):
                return None
        else:
            method = self.get_attribute_from_value(value, method_name)
            if method is UNINITIALIZED_VALUE:
                return None
        method_sig = self.signature_from_value(
            method,
            get_return_override=get_return_override,
            get_call_attribute=get_call_attribute,
        )
        if use_direct_method and method_name in {"__new__", "__init__"}:
            runtime_class = value.class_attributes.get("%runtime_class")
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
            enforce_self_compatibility = use_direct_method and not any(
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
        metaclass = value.class_attributes.get("%metaclass")
        if not isinstance(metaclass, Value):
            return None

        if isinstance(metaclass, SyntheticClassObjectValue):
            # Ignore the default metaclass call behavior; only use an explicit override.
            meta_call = metaclass.class_attributes.get("__call__", UNINITIALIZED_VALUE)
            if meta_call is UNINITIALIZED_VALUE:
                return None
        else:
            metaclass_root = replace_fallback(metaclass)
            if isinstance(metaclass_root, KnownValue) and isinstance(
                metaclass_root.val, type
            ):
                if "__call__" not in safe_getattr(metaclass_root.val, "__dict__", {}):
                    return None
            elif isinstance(metaclass_root, TypedValue) and isinstance(
                metaclass_root.typ, type
            ):
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
            self_value=SubclassValue(value.class_type),
            self_annotation_value=value,
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

        runtime_class = value.class_attributes.get("%runtime_class")
        runtime_uses_default_object_constructor = (
            isinstance(runtime_class, KnownValue)
            and isinstance(runtime_class.val, type)
            and safe_getattr(runtime_class.val, "__init__", None) is object.__init__
            and safe_getattr(runtime_class.val, "__new__", None) is object.__new__
        )
        if (
            isinstance(runtime_class, KnownValue)
            and (value.is_dataclass or is_namedtuple_class(runtime_class.val))
            and isinstance(
                runtime_argspec := self._as_concrete_signature(
                    self.arg_spec_cache.get_argspec(runtime_class.val)
                ),
                (Signature, OverloadedSignature),
            )
            and not self._is_uninformative_constructor_signature(runtime_argspec)
            and not (value.is_dataclass and runtime_uses_default_object_constructor)
        ):
            return runtime_argspec

        instance_type = self._make_synthetic_constructor_instance_value(value)
        has_direct_new = "__new__" in value.method_attributes
        has_direct_init = "__init__" in value.method_attributes
        dataclass_init_enabled = _synthetic_dataclass_init_enabled(value)

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
            self_annotation_value=value,
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
        if value.is_dataclass and dataclass_init_enabled and not has_direct_init:
            dataclass_sig = self._get_synthetic_dataclass_constructor_signature(
                value, instance_type
            )
            if dataclass_sig is None:
                dataclass_sig = Signature.make([], instance_type)
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
            dataclass_sig = self._get_synthetic_dataclass_constructor_signature(
                value, instance_type
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
        return Signature.make([], instance_type)

    def _get_synthetic_dataclass_constructor_signature(
        self, value: SyntheticClassObjectValue, instance_type: Value
    ) -> Signature | None:
        params = self._get_synthetic_dataclass_field_parameters(value)
        if not params and not value.class_attributes:
            return None
        try:
            return Signature.make(params, instance_type)
        except InvalidSignature:
            return Signature.make([ELLIPSIS_PARAM], instance_type)

    def _get_synthetic_dataclass_field_parameters(
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

        entries_by_field: dict[str, _DataclassFieldEntry] = {}
        field_order: list[str] = []
        if include_inherited:
            for base in value.base_classes:
                for inherited in self._iter_synthetic_dataclass_base_field_entries(
                    base, seen=seen
                ):
                    if inherited.field_name not in entries_by_field:
                        field_order.append(inherited.field_name)
                    entries_by_field[inherited.field_name] = inherited

        classvar_names: set[str] = set()
        classvars = value.class_attributes.get("%classvars")
        if isinstance(classvars, KnownValue) and isinstance(
            classvars.val, (set, frozenset, tuple, list)
        ):
            classvar_names.update(
                name for name in classvars.val if isinstance(name, str)
            )
        default_fields: set[str] = set()
        default_names = value.class_attributes.get("%dataclass_default_fields")
        if isinstance(default_names, KnownValue) and isinstance(
            default_names.val, (set, frozenset, tuple, list)
        ):
            default_fields.update(
                name for name in default_names.val if isinstance(name, str)
            )
        init_false_fields: set[str] = set()
        init_false_names = value.class_attributes.get("%dataclass_init_false_fields")
        if isinstance(init_false_names, KnownValue) and isinstance(
            init_false_names.val, (set, frozenset, tuple, list)
        ):
            init_false_fields.update(
                name for name in init_false_names.val if isinstance(name, str)
            )
        initvar_fields: set[str] = set()
        initvar_names = value.class_attributes.get("%dataclass_initvar_fields")
        if isinstance(initvar_names, KnownValue) and isinstance(
            initvar_names.val, (set, frozenset, tuple, list)
        ):
            initvar_fields.update(
                name for name in initvar_names.val if isinstance(name, str)
            )
        kw_only_fields: set[str] = set()
        kw_only_names = value.class_attributes.get("%dataclass_kw_only_fields")
        if isinstance(kw_only_names, KnownValue) and isinstance(
            kw_only_names.val, (set, frozenset, tuple, list)
        ):
            kw_only_fields.update(
                name for name in kw_only_names.val if isinstance(name, str)
            )
        aliases: dict[str, str] = {}
        alias_names = value.class_attributes.get("%dataclass_field_aliases")
        if isinstance(alias_names, KnownValue) and isinstance(alias_names.val, dict):
            aliases = {
                field_name: alias
                for field_name, alias in alias_names.val.items()
                if isinstance(field_name, str) and isinstance(alias, str)
            }
        converter_input_types = _synthetic_dataclass_converter_input_types(value)
        ordered_fields: list[str] = []
        order = value.class_attributes.get("%dataclass_field_order")
        if isinstance(order, KnownValue) and isinstance(order.val, (tuple, list)):
            ordered_fields = [name for name in order.val if isinstance(name, str)]
        field_names = (
            ordered_fields
            if ordered_fields
            else [
                name
                for name in value.class_attributes
                if not name.startswith("%") and not _is_dunder(name)
            ]
        )

        for name in field_names:
            attr = value.class_attributes.get(name, UNINITIALIZED_VALUE)
            excluded = (
                attr is UNINITIALIZED_VALUE
                or name in value.method_attributes
                or name in classvar_names
                or name in init_false_fields
            )
            if excluded:
                entries_by_field.pop(name, None)
                continue
            param_name = aliases.get(name, name)
            annotation = converter_input_types.get(name)
            if annotation is None:
                annotation = self._synthetic_dataclass_field_annotation(attr)
            if isinstance(attr, KnownValue):
                default: Value | None = attr if name in default_fields else None
            else:
                default = KnownValue(...) if name in default_fields else None
            if name not in field_order:
                field_order.append(name)
            entries_by_field[name] = _DataclassFieldEntry(
                field_name=name,
                parameter=SigParameter(
                    param_name,
                    (
                        ParameterKind.KEYWORD_ONLY
                        if name in kw_only_fields
                        else ParameterKind.POSITIONAL_OR_KEYWORD
                    ),
                    default=default,
                    annotation=annotation,
                ),
                is_initvar=name in initvar_fields,
            )
        return [
            entries_by_field[name] for name in field_order if name in entries_by_field
        ]

    def _synthetic_dataclass_field_annotation(self, attr: Value) -> Value:
        ctx = CheckerAttrContext(
            Composite(AnyValue(AnySource.inference)),
            "",
            self.options,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=False,
            checker=self,
        )
        return _synthetic_dataclass_parameter_annotation_for_field(attr, ctx)

    def _iter_synthetic_dataclass_base_field_entries(
        self, base: Value, *, seen: set[int]
    ) -> list[_DataclassFieldEntry]:
        base = replace_fallback(base)
        synthetic_base: SyntheticClassObjectValue | None = None
        if isinstance(base, SyntheticClassObjectValue):
            synthetic_base = base
        elif isinstance(base, GenericValue):
            if isinstance(base.typ, (type, str)):
                synthetic_base = self.get_synthetic_class(base.typ)
        elif isinstance(base, TypedValue):
            if isinstance(base.typ, (type, str)):
                synthetic_base = self.get_synthetic_class(base.typ)
        elif isinstance(base, KnownValue) and isinstance(base.val, type):
            synthetic_base = self.get_synthetic_class(base.val)
        if synthetic_base is None or not synthetic_base.is_dataclass:
            return []
        return self._get_synthetic_dataclass_field_entries(
            synthetic_base, include_inherited=True, seen=seen
        )

    def _iter_synthetic_dataclass_base_field_parameters(
        self, base: Value, *, seen: set[int]
    ) -> list[SigParameter]:
        return [
            entry.parameter
            for entry in self._iter_synthetic_dataclass_base_field_entries(
                base, seen=seen
            )
        ]

    def _augment_dataclass_constructor_signature_with_local_fields(
        self, init_sig: ConcreteSignature, value: SyntheticClassObjectValue
    ) -> ConcreteSignature:
        extra_params = self._get_synthetic_dataclass_field_parameters(
            value, include_inherited=False
        )
        if not extra_params:
            return init_sig

        def _augment(signature: Signature) -> Signature | None:
            existing = list(signature.parameters.values())
            existing_names = {param.name for param in existing}
            extras = [
                param for param in extra_params if param.name not in existing_names
            ]
            if not extras:
                return signature
            first_non_positional = next(
                (
                    i
                    for i, param in enumerate(existing)
                    if param.kind
                    not in {
                        ParameterKind.POSITIONAL_ONLY,
                        ParameterKind.POSITIONAL_OR_KEYWORD,
                    }
                ),
                len(existing),
            )
            new_params = [
                *existing[:first_non_positional],
                *extras,
                *existing[first_non_positional:],
            ]
            try:
                return dataclass_replace(
                    signature, parameters={param.name: param for param in new_params}
                )
            except InvalidSignature:
                return None

        if isinstance(init_sig, OverloadedSignature):
            augmented = [
                new_sig
                for signature in init_sig.signatures
                if (new_sig := _augment(signature)) is not None
            ]
            if not augmented:
                return init_sig
            if len(augmented) == 1:
                return augmented[0]
            return OverloadedSignature(augmented)
        augmented_sig = _augment(init_sig)
        if augmented_sig is None:
            return init_sig
        return augmented_sig

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
                    if not type_params:
                        generic_bases = self.arg_spec_cache.get_generic_bases(
                            origin, substitute_typevars=False
                        )
                        type_params = [
                            val
                            for val in generic_bases.get(origin, {}).values()
                            if isinstance(val, TypeVarValue)
                        ]
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
                        if isinstance(param, TypeVarValue)
                    }
                    exact_typevar_map = {
                        param.typevar: arg
                        for param, arg in zip(type_params, exact_arg_values)
                        if isinstance(param, TypeVarValue)
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
            argspec = self.arg_spec_cache.get_argspec(value.val)
            if isinstance(value.val, type):
                runtime_instance_type: Value | None = None
                if self._runtime_has_explicit_new_return_annotation(value.val):
                    runtime_instance_type = self._runtime_constructor_instance_value(
                        value.val
                    )
                if (
                    runtime_instance_type is not None
                    and not self._runtime_metaclass_call_overrides_constructor(
                        value.val, instance_type=runtime_instance_type
                    )
                ):
                    runtime_constructor_sig = self._get_runtime_constructor_signature(
                        value.val
                    )
                    if runtime_constructor_sig is not None:
                        argspec = runtime_constructor_sig
                synthetic_class = self.get_synthetic_class(value.val)
                if synthetic_class is not None:
                    has_direct_new = "__new__" in synthetic_class.method_attributes
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
                    if has_direct_new and synthetic_constructor_sig is not None:
                        argspec = synthetic_constructor_sig
                    elif synthetic_class.is_dataclass:
                        uses_default_object_constructor = (
                            safe_getattr(value.val, "__init__", None) is object.__init__
                            and safe_getattr(value.val, "__new__", None)
                            is object.__new__
                        )
                        if synthetic_constructor_sig is not None and (
                            argspec is None
                            or uses_default_object_constructor
                            or (
                                concrete_argspec is not None
                                and self._is_uninformative_constructor_signature(
                                    concrete_argspec
                                )
                            )
                        ):
                            argspec = synthetic_constructor_sig
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
                init_attr = value.class_attributes.get("__init__")
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
                                bound_init,
                                self._make_synthetic_class_instance_value(value),
                            )
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
                typevar_bound = value.typ.bound
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
            if isinstance(root.class_type, TypedValue) and isinstance(
                root.class_type.typ, (type, str)
            ):
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
            runtime_class = root.class_attributes.get("%runtime_class")
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
        if origin_argspec is None:
            return None
        if class_type is None:
            return origin_argspec
        type_params = self.arg_spec_cache.get_type_parameters(class_type)
        if not type_params and isinstance(class_type, type):
            generic_bases = self.arg_spec_cache.get_generic_bases(
                class_type, substitute_typevars=False
            )
            type_params = [
                val
                for val in generic_bases.get(class_type, {}).values()
                if isinstance(val, TypeVarValue)
            ]
        if not type_params and synthetic_root is not None:
            type_params = list(
                self._infer_synthetic_type_params_from_methods(synthetic_root)
            )
        member_values = [
            type_from_value(member, self, value.node, suppress_errors=True)
            for member in value.members
        ]
        exact_member_values = (
            [
                (
                    TypedValue(member.val)
                    if isinstance(member, KnownValue) and type(member.val) is type
                    else (
                        member
                        if isinstance(member, (TypedValue, GenericValue))
                        and isinstance(member.typ, (type, str))
                        else converted_member
                    )
                )
                for member, converted_member in zip(value.members, member_values)
            ]
            if preserve_exact_return
            else member_values
        )
        compatibility_member_values = [
            (
                member
                if (
                    isinstance(converted_member, AnyValue)
                    and converted_member.source is AnySource.error
                    and isinstance(member, (TypedValue, GenericValue))
                    and isinstance(member.typ, (type, str))
                )
                else exact_member
            )
            for member, converted_member, exact_member in zip(
                value.members, member_values, exact_member_values
            )
        ]
        typevar_map = {}
        for param, member in zip(type_params, member_values):
            if not isinstance(param, TypeVarValue):
                continue
            typevar_map[param.typevar] = member
        exact_typevar_map = {}
        for param, member in zip(type_params, exact_member_values):
            if not isinstance(param, TypeVarValue):
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
            runtime_class = synthetic_root.class_attributes.get("%runtime_class")
            if isinstance(runtime_class, KnownValue) and isinstance(
                runtime_class.val, type
            ):
                runtime_class_for_synthetic = runtime_class.val
        if runtime_class_for_synthetic is not None:
            runtime_type_params = self.arg_spec_cache.get_type_parameters(
                runtime_class_for_synthetic
            )
            if not runtime_type_params:
                generic_bases = self.arg_spec_cache.get_generic_bases(
                    runtime_class_for_synthetic, substitute_typevars=False
                )
                runtime_type_params = [
                    val
                    for val in generic_bases.get(
                        runtime_class_for_synthetic, {}
                    ).values()
                    if isinstance(val, TypeVarValue)
                ]
            runtime_typevar_map = {
                param.typevar: member
                for param, member in zip(runtime_type_params, member_values)
                if isinstance(param, TypeVarValue)
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

    def _synthetic_class_has_any_base(self, value: SyntheticClassObjectValue) -> bool:
        return any(has_any_base_value(base) for base in value.base_classes)

    def _make_synthetic_class_instance_value(
        self, value: SyntheticClassObjectValue
    ) -> Value:
        initvar_fields: set[str] = set()
        initvar_names = value.class_attributes.get("%dataclass_initvar_fields")
        if isinstance(initvar_names, KnownValue) and isinstance(
            initvar_names.val, (set, frozenset, tuple, list)
        ):
            initvar_fields.update(
                name for name in initvar_names.val if isinstance(name, str)
            )
        staticmethod_fields = _synthetic_staticmethod_names(value)
        classmethod_fields = _synthetic_classmethod_names(value)
        metadata = [
            HasAttrExtension(
                KnownValue(name),
                self._make_any_base_attribute(
                    name,
                    attr,
                    self_annotation_value=value.class_type,
                    is_staticmethod=name in staticmethod_fields,
                    is_classmethod=name in classmethod_fields,
                ),
            )
            for name, attr in value.class_attributes.items()
            if not name.startswith("%") and name not in initvar_fields
        ]
        if self._synthetic_class_has_any_base(value):
            instance: Value = AnyValue(AnySource.from_another)
        else:
            instance = value.class_type
        if metadata:
            return annotate_value(instance, metadata)
        return instance

    def _make_any_base_attribute(
        self,
        name: str,
        attr: Value,
        *,
        self_annotation_value: Value | None = None,
        is_staticmethod: bool = False,
        is_classmethod: bool = False,
    ) -> Value:
        raw_attr = attr
        attr = normalize_synthetic_descriptor_attribute(attr)
        if isinstance(attr, CallableValue):
            if is_staticmethod:
                return attr
            if is_classmethod:
                return self._specialize_synthetic_classmethod(
                    raw_attr, attr, self_annotation_value=self_annotation_value
                )
            maybe_bound = self._bind_synthetic_method(
                attr.signature, self_annotation_value=self_annotation_value
            )
            if maybe_bound is not None:
                return CallableValue(maybe_bound)
            return attr
        return _normalize_synthetic_attribute(attr)

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
        return CallableValue(normalized_attr.signature.substitute_typevars(inferred))

    def is_synthetic_classmethod_attribute(
        self, synthetic_class: SyntheticClassObjectValue, attr_name: str
    ) -> bool:
        return _is_synthetic_classmethod_attribute(
            synthetic_class, attr_name, self, seen=set()
        )

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


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


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


def _synthetic_staticmethod_names(
    synthetic_class: SyntheticClassObjectValue,
) -> set[str]:
    staticmethods = synthetic_class.class_attributes.get("%staticmethods")
    if isinstance(staticmethods, KnownValue) and isinstance(
        staticmethods.val, (set, frozenset, tuple, list)
    ):
        return {name for name in staticmethods.val if isinstance(name, str)}
    return set()


def _synthetic_classmethod_names(
    synthetic_class: SyntheticClassObjectValue,
) -> set[str]:
    classmethods = synthetic_class.class_attributes.get("%classmethods")
    if isinstance(classmethods, KnownValue) and isinstance(
        classmethods.val, (set, frozenset, tuple, list)
    ):
        return {name for name in classmethods.val if isinstance(name, str)}
    return set()


def _is_synthetic_staticmethod_attribute(
    synthetic_class: SyntheticClassObjectValue,
    attr_name: str,
    checker: Checker,
    *,
    seen: set[int],
) -> bool:
    synthetic_id = id(synthetic_class)
    if synthetic_id in seen:
        return False
    seen.add(synthetic_id)
    if attr_name in _synthetic_staticmethod_names(synthetic_class):
        return True
    for base in synthetic_class.base_classes:
        for base_value in flatten_values(base, unwrap_annotated=True):
            base_value = replace_fallback(base_value)
            synthetic_base: SyntheticClassObjectValue | None = None
            if isinstance(base_value, SyntheticClassObjectValue):
                synthetic_base = base_value
            elif isinstance(base_value, GenericValue) and isinstance(
                base_value.typ, (type, str)
            ):
                synthetic_base = checker.get_synthetic_class(base_value.typ)
            elif isinstance(base_value, TypedValue) and isinstance(
                base_value.typ, (type, str)
            ):
                synthetic_base = checker.get_synthetic_class(base_value.typ)
            elif isinstance(base_value, KnownValue) and isinstance(
                base_value.val, type
            ):
                synthetic_base = checker.get_synthetic_class(base_value.val)
            if synthetic_base is not None and _is_synthetic_staticmethod_attribute(
                synthetic_base, attr_name, checker, seen=seen
            ):
                return True
    return False


def _is_synthetic_classmethod_attribute(
    synthetic_class: SyntheticClassObjectValue,
    attr_name: str,
    checker: Checker,
    *,
    seen: set[int],
) -> bool:
    synthetic_id = id(synthetic_class)
    if synthetic_id in seen:
        return False
    seen.add(synthetic_id)
    if attr_name in _synthetic_classmethod_names(synthetic_class):
        return True
    for base in synthetic_class.base_classes:
        for base_value in flatten_values(base, unwrap_annotated=True):
            base_value = replace_fallback(base_value)
            synthetic_base: SyntheticClassObjectValue | None = None
            if isinstance(base_value, SyntheticClassObjectValue):
                synthetic_base = base_value
            elif isinstance(base_value, GenericValue) and isinstance(
                base_value.typ, (type, str)
            ):
                synthetic_base = checker.get_synthetic_class(base_value.typ)
            elif isinstance(base_value, TypedValue) and isinstance(
                base_value.typ, (type, str)
            ):
                synthetic_base = checker.get_synthetic_class(base_value.typ)
            elif isinstance(base_value, KnownValue) and isinstance(
                base_value.val, type
            ):
                synthetic_base = checker.get_synthetic_class(base_value.val)
            if synthetic_base is not None and _is_synthetic_classmethod_attribute(
                synthetic_base, attr_name, checker, seen=seen
            ):
                return True
    return False


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

    def signature_from_value(self, value: Value) -> MaybeSignature:
        return self.checker.signature_from_value(value)

    def get_can_assign_context(self) -> CanAssignContext:
        return self.checker

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value]
    ) -> GenericBases:
        return self.checker.get_generic_bases(typ, generic_args)

    def get_type_parameters(self, typ: type | str) -> list[Value]:
        return self.checker.get_type_parameters(typ)

    def get_synthetic_class(self, typ: type | str) -> SyntheticClassObjectValue | None:
        return self.checker.get_synthetic_class(typ)

    def should_include_synthetic_methods(self) -> bool:
        # __call__ has dedicated protocol handling in type_object; exposing
        # synthetic method values here can double-bind callable signatures.
        return self.attr != "__call__"

    def bind_synthetic_instance_attribute(self, attr_name: str, value: Value) -> Value:
        value = _normalize_synthetic_attribute(value)
        if isinstance(value, CallableValue):
            root_value = replace_fallback(self.root_value)
            synthetic_root: SyntheticClassObjectValue | None = None
            if isinstance(root_value, GenericValue) and isinstance(root_value.typ, str):
                synthetic_root = self.checker.get_synthetic_class(root_value.typ)
            elif isinstance(root_value, TypedValue) and isinstance(root_value.typ, str):
                synthetic_root = self.checker.get_synthetic_class(root_value.typ)
            if synthetic_root is not None and (
                _is_synthetic_staticmethod_attribute(
                    synthetic_root, attr_name, self.checker, seen=set()
                )
                or _is_synthetic_classmethod_attribute(
                    synthetic_root, attr_name, self.checker, seen=set()
                )
            ):
                raw_attr = synthetic_root.class_attributes.get(attr_name)
                if raw_attr is not None and not any(
                    isinstance(subval, TypeVarValue)
                    for subval in root_value.walk_values()
                ):
                    return self.checker._specialize_synthetic_classmethod(
                        raw_attr, value, self_annotation_value=root_value
                    )
                return value
            maybe_bound = self.checker._bind_synthetic_method(
                value.signature, self_annotation_value=self.root_value
            )
            if maybe_bound is not None:
                return CallableValue(maybe_bound)
        return value
