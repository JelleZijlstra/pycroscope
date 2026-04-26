import ast
import contextlib
import sys
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import Protocol, TypeGuard

from .analysis_lib import override
from .error_code import Error, ErrorCode
from .safe import is_instance_of_typing_name
from .value import (
    AnnotatedValue,
    GenericValue,
    IntersectionValue,
    MultiValuedValue,
    ParamSpecParam,
    PartialValue,
    SequenceValue,
    SubclassValue,
    TypeParam,
    TypeParamOwner,
    TypeVarLike,
    TypeVarMap,
    TypeVarParam,
    TypeVarTupleParam,
    TypeVarTupleValue,
    TypeVarValue,
    Value,
    get_single_typevartuple_param,
    type_param_to_value,
    with_type_param_owner,
)

if sys.version_info >= (3, 12):
    _TYPE_PARAM_AST_NODE_TYPES = (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)
else:
    _TYPE_PARAM_AST_NODE_TYPES = ()

TypeParamIdentity = TypeVarLike | ast.AST


@dataclass
class TypeParamScope:
    """A lexical type-parameter scope.

    `owner` is the definition that binds newly declared parameters in this scope.
    `bindings` maps runtime or AST identities to canonical, owner-bound params.
    `disallowed` contains visible identities that are invalid in this context.
    """

    owner: TypeParamOwner | None
    bindings: dict[TypeParamIdentity, TypeParam]
    disallowed: set[TypeParamIdentity]


@dataclass(frozen=True)
class TypeParamBindingResult:
    type_params: tuple[TypeParam, ...]
    substitutions: TypeVarMap
    aliases: tuple[frozenset[TypeParamIdentity], ...]


class TypeParamVisitor(Protocol):
    in_annotation: bool
    in_type_alias_definition: bool

    def _is_collecting(self) -> bool: ...

    def show_error(
        self,
        node: ast.AST,
        e: str | None = None,
        error_code: Error | None = ErrorCode.invalid_annotation,
    ) -> object | None: ...

    def _show_error_if_checking(
        self,
        node: ast.AST,
        msg: str | None = None,
        error_code: Error | None = ErrorCode.invalid_annotation,
    ) -> object | None: ...

    def resolve_name(
        self,
        node: ast.Name,
        error_node: ast.AST | None = None,
        suppress_errors: bool = False,
    ) -> tuple[Value, object]: ...

    def _merge_type_param_polarities(
        self,
        target: dict[object, set[int]],
        local_polarities: dict[object, set[int]],
        *,
        polarity: int,
    ) -> None: ...

    def get_type_param_from_value(self, value: Value) -> TypeParam | None: ...


@dataclass
class _LegacyTypeParamPolicy:
    allowed_identities: set[object]
    message: str
    error_code: Error = ErrorCode.invalid_annotation
    report_in_collecting: bool = True
    triggered: bool = False


@dataclass
class _VarianceCollectionContext:
    type_param_polarities: dict[object, set[int]]
    polarity: int


def _compose_observed_variance_polarity(polarity: int, modifier: int) -> int:
    if polarity == 0 or modifier == 0:
        return 0
    return polarity * modifier


def _record_variance_polarity(used_polarities: set[int], polarity: int) -> None:
    if polarity == 0:
        used_polarities.update({-1, 1})
    else:
        used_polarities.add(polarity)


def _is_type_param_declaration_node(node: ast.AST) -> bool:
    return bool(_TYPE_PARAM_AST_NODE_TYPES) and isinstance(
        node, _TYPE_PARAM_AST_NODE_TYPES
    )


def _is_type_param_identity(identity: object) -> TypeGuard[TypeParamIdentity]:
    return isinstance(identity, ast.AST) or _is_typevarlike(identity)


def _is_typevarlike(identity: object) -> bool:
    return (
        is_instance_of_typing_name(identity, "TypeVar")
        or is_instance_of_typing_name(identity, "ParamSpec")
        or is_instance_of_typing_name(identity, "TypeVarTuple")
    )


def _typed_identities(identities: Iterable[object]) -> set[TypeParamIdentity]:
    typed: set[TypeParamIdentity] = set()
    for identity in identities:
        if _is_type_param_identity(identity):
            typed.add(identity)
    return typed


def _identity_aliases(
    raw_param: TypeParam, bound_param: TypeParam
) -> frozenset[TypeParamIdentity]:
    aliases: set[TypeParamIdentity] = set()
    if (
        _is_type_param_identity(raw_param.typevar)
        and raw_param.typevar is not bound_param.typevar
    ):
        aliases.add(raw_param.typevar)
    if _is_type_param_identity(bound_param.typevar):
        aliases.add(bound_param.typevar)
    return frozenset(aliases)


def _type_param_from_value(value: Value) -> TypeParam | None:
    if isinstance(value, TypeVarValue):
        return value.typevar_param
    if isinstance(value, TypeVarTupleValue):
        return value.typevar_tuple_param
    typevartuple_param = get_single_typevartuple_param(value)
    if typevartuple_param is not None:
        return typevartuple_param
    from pycroscope.input_sig import InputSigValue

    if isinstance(value, InputSigValue) and isinstance(value.input_sig, ParamSpecParam):
        return value.input_sig
    return None


def _normalize_type_param_identities_in_value(
    value: Value, replacements: Mapping[object, TypeParam]
) -> Value:
    type_param = _type_param_from_value(value)
    if type_param is not None:
        replacement = replacements.get(type_param.typevar)
        if replacement is not None:
            return type_param_to_value(replacement)
        return value
    if isinstance(value, AnnotatedValue):
        normalized = _normalize_type_param_identities_in_value(
            value.value, replacements
        )
        if normalized is value.value:
            return value
        return replace(value, value=normalized)
    if isinstance(value, GenericValue):
        normalized_args = tuple(
            _normalize_type_param_identities_in_value(arg, replacements)
            for arg in value.args
        )
        if normalized_args == value.args:
            return value
        return GenericValue(value.typ, normalized_args, weak=value.weak)
    if isinstance(value, SequenceValue):
        normalized_members = tuple(
            (is_many, _normalize_type_param_identities_in_value(member, replacements))
            for is_many, member in value.members
        )
        if normalized_members == value.members:
            return value
        return replace(value, members=normalized_members)
    if isinstance(value, PartialValue):
        normalized_root = _normalize_type_param_identities_in_value(
            value.root, replacements
        )
        normalized_members = tuple(
            _normalize_type_param_identities_in_value(member, replacements)
            for member in value.members
        )
        normalized_runtime_value = _normalize_type_param_identities_in_value(
            value.runtime_value, replacements
        )
        if (
            normalized_root is value.root
            and normalized_members == value.members
            and normalized_runtime_value is value.runtime_value
        ):
            return value
        return replace(
            value,
            root=normalized_root,
            members=normalized_members,
            runtime_value=normalized_runtime_value,
        )
    if isinstance(value, SubclassValue):
        normalized_typ = _normalize_type_param_identities_in_value(
            value.typ, replacements
        )
        if normalized_typ is value.typ:
            return value
        return replace(value, typ=normalized_typ)
    if isinstance(value, MultiValuedValue):
        normalized_vals = tuple(
            _normalize_type_param_identities_in_value(subval, replacements)
            for subval in value.vals
        )
        if normalized_vals == value.vals:
            return value
        return MultiValuedValue(normalized_vals)
    if isinstance(value, IntersectionValue):
        normalized_vals = tuple(
            _normalize_type_param_identities_in_value(subval, replacements)
            for subval in value.vals
        )
        if normalized_vals == value.vals:
            return value
        return IntersectionValue(normalized_vals)
    return value


def _substitute_typevars_in_type_param(
    type_param: TypeParam, substitutions: TypeVarMap
) -> TypeParam:
    if isinstance(type_param, TypeVarParam):
        return replace(
            type_param,
            bound=(
                None
                if type_param.bound is None
                else type_param.bound.substitute_typevars(substitutions)
            ),
            default=(
                None
                if type_param.default is None
                else type_param.default.substitute_typevars(substitutions)
            ),
            constraints=tuple(
                constraint.substitute_typevars(substitutions)
                for constraint in type_param.constraints
            ),
        )
    if isinstance(type_param, TypeVarTupleParam):
        return replace(
            type_param,
            default=(
                None
                if type_param.default is None
                else type_param.default.substitute_typevars(substitutions)
            ),
        )
    return replace(
        type_param,
        default=(
            None
            if type_param.default is None
            else type_param.default.substitute_typevars(substitutions)
        ),
    )


def _normalize_type_param_identities_in_type_param(
    type_param: TypeParam,
    replacements: Mapping[object, TypeParam],
    normalize_value: Callable[[Value, Mapping[object, TypeParam]], Value],
) -> TypeParam:
    if isinstance(type_param, TypeVarParam):
        return replace(
            type_param,
            bound=(
                None
                if type_param.bound is None
                else normalize_value(type_param.bound, replacements)
            ),
            default=(
                None
                if type_param.default is None
                else normalize_value(type_param.default, replacements)
            ),
            constraints=tuple(
                normalize_value(constraint, replacements)
                for constraint in type_param.constraints
            ),
        )
    if isinstance(type_param, TypeVarTupleParam):
        return replace(
            type_param,
            default=(
                None
                if type_param.default is None
                else normalize_value(type_param.default, replacements)
            ),
        )
    return replace(
        type_param,
        default=(
            None
            if type_param.default is None
            else normalize_value(type_param.default, replacements)
        ),
    )


class ActiveTypeParams:
    def __init__(self, visitor: TypeParamVisitor | None = None) -> None:
        self.visitor = visitor
        self._annotation_allowed_identities: list[set[object]] = []
        self._current_class_type_params: Sequence[TypeParam] | None = None
        self._legacy_policies: list[_LegacyTypeParamPolicy] = []
        self._variance_collections: list[_VarianceCollectionContext] = []
        self._current_class_type_param_polarities: dict[object, set[int]] | None = None
        self._current_is_protocol_class: bool = False
        self._variance_polarity_stack: list[int] = []
        self._variance_is_suspended = 0
        self._variance_outside_annotations = 0
        self._subscript_arg_polarities: list[tuple[tuple[int, bool], ...]] = []
        self._scopes: list[TypeParamScope] = []

    @contextmanager
    def push_scope(self, owner: TypeParamOwner | None = None) -> Generator[None]:
        scope = TypeParamScope(owner, {}, set())
        self._scopes.append(scope)
        try:
            yield
        finally:
            self._scopes.pop()

    if sys.version_info >= (3, 12):

        def current_owner(self) -> TypeParamOwner:
            for scope in reversed(self._scopes):
                if scope.owner is not None:
                    return scope.owner
            raise AssertionError("no active type parameter owner")

    def current_annotation_identities(self) -> set[TypeParamIdentity]:
        identities: set[TypeParamIdentity] = set()
        for allowed in self._annotation_allowed_identities:
            identities.update(_typed_identities(allowed))
        identities.update(self.current_pep695_identities())
        return identities

    def current_pep695_identities(self) -> set[TypeParamIdentity]:
        return set(self.current_pep695_type_params())

    def current_pep695_type_params(self) -> dict[TypeParamIdentity, TypeParam]:
        type_params: dict[TypeParamIdentity, TypeParam] = {}
        for scope in self._scopes:
            type_params.update(scope.bindings)
        return type_params

    def get_type_param(self, identity: object) -> TypeParam | None:
        if not _is_type_param_identity(identity):
            return None
        for scope in reversed(self._scopes):
            type_param = scope.bindings.get(identity)
            if type_param is not None:
                return type_param
        return None

    def declare(
        self,
        type_param: TypeParam,
        *,
        aliases: Iterable[object] = (),
        owner: TypeParamOwner | None = None,
    ) -> TypeParam:
        if owner is None:
            owner = self._scopes[-1].owner if self._scopes else None
        canonical = with_type_param_owner(type_param, owner)
        if not self._scopes:
            self._scopes.append(TypeParamScope(None, {}, set()))
        for identity in (canonical.typevar, *_typed_identities(aliases)):
            self._scopes[-1].bindings[identity] = canonical
        return canonical

    def add_pep695_scope(
        self,
        type_params: Sequence[TypeParam],
        *,
        aliases: Sequence[Iterable[object]] | None = None,
        owner: TypeParamOwner | None = None,
    ) -> None:
        if aliases is None:
            aliases = [()] * len(type_params)
        if len(aliases) != len(type_params):
            raise ValueError("type parameter aliases must match type parameters")
        if not type_params:
            return
        if not self._scopes:
            self._scopes.append(TypeParamScope(owner, {}, set()))
        elif owner is not None and self._scopes[-1].owner is None:
            self._scopes[-1].owner = owner
        for type_param, param_aliases in zip(type_params, aliases):
            self.declare(type_param, aliases=param_aliases, owner=owner)

    def current_class_type_params(self) -> Sequence[TypeParam] | None:
        return self._current_class_type_params

    @contextlib.contextmanager
    def allow_in_annotations(self, identities: Iterable[object]) -> Generator[None]:
        identities = set(identities)
        if not identities:
            yield
            return
        self._annotation_allowed_identities.append(identities)
        try:
            yield
        finally:
            self._annotation_allowed_identities.pop()

    @contextlib.contextmanager
    def push_pep695_scope(
        self,
        type_params: Sequence[TypeParam],
        *,
        aliases: Sequence[Iterable[object]] | None = None,
        owner: TypeParamOwner | None = None,
    ) -> Generator[None]:
        if not type_params:
            yield
            return
        with self.push_scope(owner):
            self.add_pep695_scope(type_params, aliases=aliases)
            yield

    @contextlib.contextmanager
    def push_class_type_params(
        self, type_params: Sequence[TypeParam] | None
    ) -> Generator[None]:
        with override(self, "_current_class_type_params", type_params):
            yield

    @contextlib.contextmanager
    def disallow(self, identities: Iterable[object]) -> Generator[None]:
        disallowed = _typed_identities(identities)
        if not disallowed:
            yield
            return
        with self.push_scope():
            self._scopes[-1].disallowed.update(disallowed)
            yield

    def bind_all(
        self,
        type_params: Sequence[TypeParam],
        owner: TypeParamOwner,
        *,
        aliases: Sequence[Iterable[object]] | None = None,
        normalize_value: (
            Callable[[Value, Mapping[object, TypeParam]], Value] | None
        ) = None,
    ) -> TypeParamBindingResult:
        if aliases is None:
            aliases = [()] * len(type_params)
        if len(aliases) != len(type_params):
            raise ValueError("type parameter aliases must match type parameters")
        owner_bound = tuple(
            with_type_param_owner(param, owner) for param in type_params
        )
        substitutions = TypeVarMap()
        replacements: dict[object, TypeParam] = {}
        all_aliases: list[frozenset[TypeParamIdentity]] = []
        for raw_param, bound_param, extra_aliases in zip(
            type_params, owner_bound, aliases
        ):
            substitutions = substitutions.with_value(
                raw_param, type_param_to_value(bound_param)
            )
            replacements[raw_param.typevar] = bound_param
            replacements[bound_param.typevar] = bound_param
            param_aliases = set(_identity_aliases(raw_param, bound_param))
            param_aliases.update(_typed_identities(extra_aliases))
            for alias in param_aliases:
                replacements[alias] = bound_param
            all_aliases.append(frozenset(param_aliases))
        if normalize_value is None:
            normalize_value = _normalize_type_param_identities_in_value
        return TypeParamBindingResult(
            tuple(
                _normalize_type_param_identities_in_type_param(
                    _substitute_typevars_in_type_param(param, substitutions),
                    replacements,
                    normalize_value,
                )
                for param in owner_bound
            ),
            substitutions,
            tuple(all_aliases),
        )

    @contextlib.contextmanager
    def reject_legacy_type_params(
        self,
        message: str,
        *,
        error_code: Error = ErrorCode.invalid_annotation,
        include_active_pep695: bool = True,
        report_in_collecting: bool = True,
    ) -> Generator[set[object]]:
        allowed_identities: set[object] = set()
        if include_active_pep695:
            for identity in self.current_pep695_identities():
                allowed_identities.add(identity)
        policy = _LegacyTypeParamPolicy(
            allowed_identities,
            message,
            error_code=error_code,
            report_in_collecting=report_in_collecting,
        )
        self._legacy_policies.append(policy)
        try:
            yield allowed_identities
        finally:
            self._legacy_policies.pop()

    @contextlib.contextmanager
    def collect_variance(
        self, type_param_polarities: dict[object, set[int]], *, polarity: int
    ) -> Generator[None]:
        self._variance_collections.append(
            _VarianceCollectionContext(type_param_polarities, polarity)
        )
        try:
            yield
        finally:
            self._variance_collections.pop()

    @contextlib.contextmanager
    def compose_variance(self, polarity: int) -> Generator[None]:
        self._variance_polarity_stack.append(polarity)
        try:
            yield
        finally:
            self._variance_polarity_stack.pop()

    @contextlib.contextmanager
    def suspend_variance(self) -> Generator[None]:
        self._variance_is_suspended += 1
        try:
            yield
        finally:
            self._variance_is_suspended -= 1

    @contextlib.contextmanager
    def allow_variance_outside_annotations(self) -> Generator[None]:
        self._variance_outside_annotations += 1
        try:
            yield
        finally:
            self._variance_outside_annotations -= 1

    def has_variance_collection(self) -> bool:
        return bool(self._variance_collections) and not self._variance_is_suspended

    def current_class_type_param_polarities(self) -> dict[object, set[int]] | None:
        return self._current_class_type_param_polarities

    def current_variance_polarity(self, base_polarity: int) -> int:
        polarity = base_polarity
        for modifier in self._variance_polarity_stack:
            polarity = _compose_observed_variance_polarity(polarity, modifier)
        return polarity

    @contextlib.contextmanager
    def push_class_type_param_variance_collection(
        self,
        type_param_polarities: dict[object, set[int]] | None,
        *,
        is_protocol_class: bool,
    ) -> Generator[None]:
        with (
            override(
                self, "_current_class_type_param_polarities", type_param_polarities
            ),
            override(self, "_current_is_protocol_class", is_protocol_class),
        ):
            yield

    @contextlib.contextmanager
    def suspend_class_type_param_variance_collection(self) -> Generator[None]:
        if self._current_class_type_param_polarities is None:
            yield
            return
        with override(self, "_current_class_type_param_polarities", None):
            yield

    def function_param_type_param_variance_context(
        self, *, parameter_index: int, is_staticmethod: bool
    ) -> AbstractContextManager[None]:
        if self._current_class_type_param_polarities is None:
            return contextlib.nullcontext()
        if (
            self._current_is_protocol_class
            and parameter_index == 0
            and not is_staticmethod
        ):
            return self.suspend_variance()
        return self.local_class_type_param_variance_context(polarity=-1)

    def function_return_type_param_variance_context(
        self,
    ) -> AbstractContextManager[None]:
        if self._current_class_type_param_polarities is None:
            return contextlib.nullcontext()
        return self.local_class_type_param_variance_context(polarity=1)

    @contextlib.contextmanager
    def local_class_type_param_variance_context(
        self, *, polarity: int
    ) -> Generator[None]:
        target = self._current_class_type_param_polarities
        if target is None:
            yield
            return
        visitor = self._require_visitor()
        local_polarities: dict[object, set[int]] = {}
        with (
            self.collect_variance(local_polarities, polarity=1),
            self.compose_variance(polarity),
        ):
            yield
        visitor._merge_type_param_polarities(target, local_polarities, polarity=1)

    @contextlib.contextmanager
    def push_subscript_arg_polarities(
        self, polarities: Sequence[tuple[int, bool]]
    ) -> Generator[None]:
        self._subscript_arg_polarities.append(tuple(polarities))
        try:
            yield
        finally:
            self._subscript_arg_polarities.pop()

    @contextlib.contextmanager
    def consume_subscript_arg_polarities(
        self, arity: int
    ) -> Generator[tuple[tuple[int, bool], ...] | None]:
        polarities = (
            self._subscript_arg_polarities.pop()
            if self._subscript_arg_polarities
            and len(self._subscript_arg_polarities[-1]) == arity
            else None
        )
        try:
            yield polarities
        finally:
            if polarities is not None:
                self._subscript_arg_polarities.append(polarities)

    def observe_value(self, node: ast.AST, value: Value) -> None:
        visitor = self.visitor
        if visitor is None:
            return
        if not (
            self._legacy_policies
            or self._has_disallowed_identities()
            or visitor.in_annotation
            or self._variance_collections
        ):
            return
        if self._variance_is_suspended and not (
            self._legacy_policies or self._has_disallowed_identities()
        ):
            return
        if _is_type_param_declaration_node(node):
            return

        type_param = visitor.get_type_param_from_value(value)
        if type_param is not None:
            self._check_direct_type_param_usage(node, type_param)
            if (
                isinstance(type_param, TypeVarParam)
                and self._variance_collections
                and not self._variance_is_suspended
                and (visitor.in_annotation or self._variance_outside_annotations > 0)
            ):
                for context in self._variance_collections:
                    used_polarities = context.type_param_polarities.get(
                        type_param.typevar
                    )
                    if used_polarities is None:
                        used_polarities = context.type_param_polarities.setdefault(
                            type_param.typevar, set()
                        )
                    _record_variance_polarity(
                        used_polarities,
                        self.current_variance_polarity(context.polarity),
                    )

    def _require_visitor(self) -> TypeParamVisitor:
        if self.visitor is None:
            raise AssertionError(
                "ActiveTypeParams requires a visitor for this operation"
            )
        return self.visitor

    def _show_error(
        self,
        node: ast.AST,
        message: str,
        *,
        error_code: Error,
        report_in_collecting: bool,
    ) -> None:
        visitor = self._require_visitor()
        if report_in_collecting and visitor._is_collecting():
            visitor.show_error(node, message, error_code=error_code)
        else:
            visitor._show_error_if_checking(node, message, error_code=error_code)

    def _check_direct_type_param_usage(
        self, error_node: ast.AST, type_param: TypeParam
    ) -> None:
        visitor = self._require_visitor()
        identity = type_param.typevar
        narrowed_error_node = self._narrow_type_param_error_node(error_node, identity)
        if narrowed_error_node is not error_node:
            return
        for policy in self._legacy_policies:
            if policy.triggered:
                continue
            if identity in policy.allowed_identities:
                continue
            policy.triggered = True
            self._show_error(
                narrowed_error_node,
                policy.message,
                error_code=policy.error_code,
                report_in_collecting=policy.report_in_collecting,
            )
            break

        if not isinstance(type_param, TypeVarParam) or type_param.is_self:
            return
        if self._is_disallowed(identity):
            visitor._show_error_if_checking(
                narrowed_error_node,
                "Type parameter is not valid in this annotation context",
                error_code=ErrorCode.invalid_annotation,
            )
            return
        if visitor.in_type_alias_definition or not visitor.in_annotation:
            return
        if identity in self.current_annotation_identities():
            return
        visitor._show_error_if_checking(
            narrowed_error_node,
            "Type parameter is not valid in this annotation context",
            error_code=ErrorCode.invalid_annotation,
        )

    def _has_disallowed_identities(self) -> bool:
        return any(scope.disallowed for scope in self._scopes)

    def _is_disallowed(self, identity: object) -> bool:
        return any(identity in scope.disallowed for scope in self._scopes)

    def normalize_type_param_identities(
        self,
        type_param: TypeParam,
        replacements: Mapping[object, TypeParam],
        *,
        normalize_value: (
            Callable[[Value, Mapping[object, TypeParam]], Value] | None
        ) = None,
    ) -> TypeParam:
        if normalize_value is None:
            normalize_value = _normalize_type_param_identities_in_value
        return _normalize_type_param_identities_in_type_param(
            type_param, replacements, normalize_value
        )

    # TODO: this is silly, find a better solution to prevent duplicate errors
    def _narrow_type_param_error_node(self, node: ast.AST, identity: object) -> ast.AST:
        visitor = self._require_visitor()
        if isinstance(node, ast.Name):
            return node
        for subnode in ast.walk(node):
            if not isinstance(subnode, ast.Name):
                continue
            resolved, _ = visitor.resolve_name(
                subnode, error_node=subnode, suppress_errors=True
            )
            resolved_type_param = visitor.get_type_param_from_value(resolved)
            if (
                resolved_type_param is not None
                and resolved_type_param.typevar is identity
            ):
                return subnode
        return node
