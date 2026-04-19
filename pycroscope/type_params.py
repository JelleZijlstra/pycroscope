import ast
import contextlib
from collections.abc import Generator, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from .analysis_lib import override
from .error_code import Error, ErrorCode
from .safe import safe_getattr
from .value import TypeParam, TypeVarParam, Value

_TYPE_PARAM_AST_NODE_TYPES = tuple(
    typ
    for typ in (
        safe_getattr(ast, "TypeVar", None),
        safe_getattr(ast, "ParamSpec", None),
        safe_getattr(ast, "TypeVarTuple", None),
    )
    if isinstance(typ, type)
)


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


class ActiveTypeParams:
    def __init__(self, visitor: TypeParamVisitor | None = None) -> None:
        self.visitor = visitor
        self._annotation_allowed_identities: list[set[object]] = []
        self._active_pep695_identities: list[set[object]] = []
        self._active_pep695_type_params: list[dict[object, TypeParam]] = []
        self._current_class_type_params: Sequence[TypeParam] | None = None
        self._disallowed_identities: list[set[object]] = []
        self._legacy_policies: list[_LegacyTypeParamPolicy] = []
        self._variance_collections: list[_VarianceCollectionContext] = []
        self._current_class_type_param_polarities: dict[object, set[int]] | None = None
        self._current_is_protocol_class: bool = False
        self._variance_polarity_stack: list[int] = []
        self._variance_is_suspended = 0
        self._variance_outside_annotations = 0
        self._subscript_arg_polarities: list[tuple[tuple[int, bool], ...]] = []

    def current_annotation_identities(self) -> set[object]:
        identities: set[object] = set()
        for allowed in self._annotation_allowed_identities:
            identities.update(allowed)
        for allowed in self._active_pep695_identities:
            identities.update(allowed)
        return identities

    def current_pep695_identities(self) -> set[object]:
        identities: set[object] = set()
        for allowed in self._active_pep695_identities:
            identities.update(allowed)
        return identities

    def current_pep695_type_params(self) -> dict[object, TypeParam]:
        type_params: dict[object, TypeParam] = {}
        for scope in self._active_pep695_type_params:
            type_params.update(scope)
        return type_params

    def get_type_param(self, identity: object) -> TypeParam | None:
        for scope in reversed(self._active_pep695_type_params):
            type_param = scope.get(identity)
            if type_param is not None:
                return type_param
        return None

    def add_pep695_scope(
        self,
        type_params: Sequence[TypeParam],
        *,
        additional_identities: Sequence[Iterable[object]] | None = None,
    ) -> None:
        if additional_identities is None:
            additional_identities = [()] * len(type_params)
        scope: dict[object, TypeParam] = {}
        identities: set[object] = set()
        for type_param, extra_identities in zip(type_params, additional_identities):
            all_identities = {type_param.typevar, *extra_identities}
            for identity in all_identities:
                scope[identity] = type_param
            identities.update(all_identities)
        if not identities:
            return
        self._active_pep695_identities.append(identities)
        self._active_pep695_type_params.append(scope)

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
        additional_identities: Sequence[Iterable[object]] | None = None,
    ) -> Generator[None]:
        before = len(self._active_pep695_type_params)
        self.add_pep695_scope(type_params, additional_identities=additional_identities)
        if len(self._active_pep695_type_params) == before:
            yield
            return
        try:
            yield
        finally:
            self._active_pep695_type_params.pop()
            self._active_pep695_identities.pop()

    @contextlib.contextmanager
    def push_class_type_params(
        self, type_params: Sequence[TypeParam] | None
    ) -> Generator[None]:
        with override(self, "_current_class_type_params", type_params):
            yield

    @contextlib.contextmanager
    def disallow(self, identities: Iterable[object]) -> Generator[None]:
        identities = set(identities)
        if not identities:
            yield
            return
        self._disallowed_identities.append(identities)
        try:
            yield
        finally:
            self._disallowed_identities.pop()

    @contextlib.contextmanager
    def reject_legacy_type_params(
        self,
        message: str,
        *,
        error_code: Error = ErrorCode.invalid_annotation,
        include_active_pep695: bool = True,
        report_in_collecting: bool = True,
    ) -> Generator[set[object]]:
        allowed_identities = (
            self.current_pep695_identities() if include_active_pep695 else set()
        )
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
            or self._disallowed_identities
            or visitor.in_annotation
            or self._variance_collections
        ):
            return
        if self._variance_is_suspended and not (
            self._legacy_policies or self._disallowed_identities
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
        if any(identity in disallowed for disallowed in self._disallowed_identities):
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
