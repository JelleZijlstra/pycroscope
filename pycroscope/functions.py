"""

Code for understanding function definitions.

"""

import ast
import asyncio
import builtins
import collections.abc
import enum
import sys
import types
from collections.abc import Container, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import TypeAlias, TypeVar

from typing_extensions import Protocol

from pycroscope.input_sig import InputSigValue, ParamSpecSig

from .analysis_lib import is_positional_only_arg_name
from .error_code import ErrorCode
from .extensions import deprecated as deprecated_decorator
from .maybe_asynq import asynq
from .node_visitor import Error, ErrorContext
from .options import Options, PyObjectSequenceOption
from .relations import Relation, has_relation
from .safe import is_instance_of_typing_name, is_typing_name
from .signature import (
    ParameterKind,
    Signature,
    SigParameter,
    mark_ellipsis_style_any_tail_parameters,
)
from .stacked_scopes import Composite
from .value import (
    AnnotationExpr,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    DeprecatedExtension,
    GenericValue,
    KnownValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    Qualifier,
    SelfT,
    SubclassValue,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeVarValue,
    Value,
    annotate_value,
    get_tv_map,
    is_async_iterable,
    is_iterable,
    make_coro_type,
    replace_fallback,
    unite_values,
)

FunctionDefNode = ast.FunctionDef | ast.AsyncFunctionDef
FunctionNode = FunctionDefNode | ast.Lambda
IMPLICIT_CLASSMETHODS = ("__init_subclass__", "__new__")

YieldT = TypeVar("YieldT")
SendT = TypeVar("SendT")
ReturnT = TypeVar("ReturnT")
GeneratorValue = GenericValue(
    collections.abc.Generator,
    [TypeVarValue(YieldT), TypeVarValue(SendT), TypeVarValue(ReturnT)],
)
AsyncGeneratorValue = GenericValue(
    collections.abc.AsyncGenerator, [TypeVarValue(YieldT), TypeVarValue(SendT)]
)


# a list of tuples of (decorator function, applied decorator function, AST node). These are
# different for decorators that take arguments, like @asynq(): the first element will be the
# asynq function and the second will be the result of calling asynq().
DecoratorValues: TypeAlias = list[tuple[Value, Value, ast.expr]]


def _type_param_identities_for_class(enclosing_class: TypedValue | None) -> set[object]:
    if not isinstance(enclosing_class, GenericValue):
        return set()
    identities: set[object] = set()
    for arg in enclosing_class.args:
        if isinstance(arg, TypeVarValue):
            identities.add(arg.typevar)
    return identities


class AsyncFunctionKind(enum.Enum):
    non_async = 0
    normal = 1
    async_proxy = 2
    pure = 3


class FunctionDecorator(enum.Enum):
    classmethod = enum.auto()
    staticmethod = enum.auto()
    decorated_coroutine = enum.auto()
    overload = enum.auto()
    override = enum.auto()
    final = enum.auto()
    evaluated = enum.auto()
    abstractmethod = enum.auto()

    @builtins.classmethod
    def method_kind_for(cls, decorators: Container["FunctionDecorator"]) -> str:
        if cls.classmethod in decorators:
            return "classmethod"
        if cls.staticmethod in decorators:
            return "staticmethod"
        return "instance"


@dataclass(frozen=True)
class ParamInfo:
    param: SigParameter
    node: ast.AST
    is_self: bool = False


@dataclass(frozen=True)
class FunctionInfo:
    """Computed before visiting a function."""

    async_kind: AsyncFunctionKind
    decorator_kinds: frozenset[FunctionDecorator]
    is_nested_in_class: bool
    decorators: DecoratorValues
    node: FunctionNode
    params: Sequence[ParamInfo]
    return_annotation: Value | None
    potential_function: object | None
    type_params: Sequence[TypeVarValue]

    def get_generator_yield_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        if isinstance(self.node, ast.AsyncFunctionDef):
            iterable_val = is_async_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return iterable_val
        else:
            iterable_val = is_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return iterable_val

    def get_generator_send_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        if isinstance(self.node, ast.AsyncFunctionDef):
            tv_map = get_tv_map(AsyncGeneratorValue, self.return_annotation, ctx)
            if not isinstance(tv_map, CanAssignError):
                return tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            # If the return annotation is a non-Generator Iterable, assume the send
            # type is None.
            iterable_val = is_async_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return KnownValue(None)
        else:
            tv_map = get_tv_map(GeneratorValue, self.return_annotation, ctx)
            if not isinstance(tv_map, CanAssignError):
                return tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            # If the return annotation is a non-Generator Iterable, assume the send
            # type is None.
            iterable_val = is_iterable(self.return_annotation, ctx)
            if isinstance(iterable_val, CanAssignError):
                return AnyValue(AnySource.error)
            return KnownValue(None)

    def get_generator_return_type(self, ctx: CanAssignContext) -> Value:
        if self.return_annotation is None:
            return AnyValue(AnySource.unannotated)
        tv_map = get_tv_map(GeneratorValue, self.return_annotation, ctx)
        if not isinstance(tv_map, CanAssignError):
            return tv_map.get(ReturnT, AnyValue(AnySource.generic_argument))
        # If the return annotation is a non-Generator Iterable, assume the return
        # type is None.
        iterable_val = is_iterable(self.return_annotation, ctx)
        if isinstance(iterable_val, CanAssignError):
            return AnyValue(AnySource.error)
        return KnownValue(None)


@dataclass
class FunctionResult:
    """Computed after visiting a function."""

    return_value: Value = AnyValue(AnySource.inference)
    parameters: Sequence[SigParameter] = ()
    has_return: bool = False
    is_generator: bool = False
    has_return_annotation: bool = False


class Context(ErrorContext, CanAssignContext, Protocol):
    options: Options

    def visit_expression(self, node: ast.AST, /) -> Value:
        raise NotImplementedError

    def expr_of_annotation(self, node: ast.expr, /) -> AnnotationExpr:
        raise NotImplementedError

    def check_call(
        self,
        node: ast.AST,
        callee: Value,
        args: Iterable[Composite],
        *,
        allow_call: bool = False,
    ) -> Value:
        raise NotImplementedError

    def catch_errors(self) -> AbstractContextManager[list[Error]]:
        raise NotImplementedError


class AsynqDecorators(PyObjectSequenceOption[object]):
    """Decorators that are equivalent to asynq.asynq."""

    default_value = [asynq.asynq] if asynq is not None else []
    name = "asynq_decorators"


class AsyncProxyDecorators(PyObjectSequenceOption[object]):
    """Decorators that are equivalent to asynq.async_proxy."""

    default_value = [asynq.async_proxy] if asynq is not None else []
    name = "async_proxy_decorators"


_safe_decorators = [classmethod, staticmethod]
if sys.version_info < (3, 11):
    # static analysis: ignore[undefined_attribute]
    _safe_decorators.append(asyncio.coroutine)
if asynq is not None:
    _safe_decorators.append(asynq.asynq)


class SafeDecoratorsForNestedFunctions(PyObjectSequenceOption[object]):
    """These decorators can safely be applied to nested functions."""

    name = "safe_decorators_for_nested_functions"
    default_value = _safe_decorators


def _visit_default(node: ast.AST, ctx: Context) -> Value:
    val = ctx.visit_expression(node)
    if val == KnownValue(...):
        return AnyValue(AnySource.unannotated)
    return val


def _paramspec_identities_from_value(value: Value) -> set[object]:
    identities: set[object] = set()
    for subval in value.walk_values():
        if isinstance(subval, InputSigValue) and isinstance(
            subval.input_sig, ParamSpecSig
        ):
            identities.add(subval.input_sig.param_spec)
        elif isinstance(subval, (ParamSpecArgsValue, ParamSpecKwargsValue)):
            identities.add(subval.param_spec)
        elif isinstance(subval, TypeVarValue) and is_instance_of_typing_name(
            subval.typevar, "ParamSpec"
        ):
            identities.add(subval.typevar)
    return identities


def _paramspec_identities_from_context(ctx: Context) -> set[object]:
    info = getattr(ctx, "current_function_info", None)
    identities: set[object] = set()
    if isinstance(info, FunctionInfo):
        for param_info in info.params:
            identities.update(
                _paramspec_identities_from_value(param_info.param.annotation)
            )
        if info.return_annotation is not None:
            identities.update(_paramspec_identities_from_value(info.return_annotation))
        for type_param in info.type_params:
            if is_instance_of_typing_name(type_param.typevar, "ParamSpec"):
                identities.add(type_param.typevar)
    current_class_type_params = getattr(ctx, "current_class_type_params", None)
    if current_class_type_params is not None:
        for type_param in current_class_type_params:
            if is_instance_of_typing_name(type_param.typevar, "ParamSpec"):
                identities.add(type_param.typevar)
    return identities


def _is_invalid_generic_annotation_value(value: Value) -> bool:
    if isinstance(value, TypeAliasValue):
        target = value.get_value()
        if isinstance(target, GenericValue) and is_typing_name(target.typ, "Generic"):
            return True
    for subval in value.walk_values():
        if isinstance(subval, KnownValue):
            if is_typing_name(subval.val, "Generic"):
                return True
        elif isinstance(subval, TypedValue):
            if is_typing_name(subval.typ, "Generic"):
                return True
        elif isinstance(subval, TypeAliasValue):
            target = subval.get_value()
            if isinstance(target, GenericValue) and is_typing_name(
                target.typ, "Generic"
            ):
                return True
    return False


def compute_parameters(
    node: FunctionNode,
    enclosing_class: TypedValue | None,
    ctx: Context,
    *,
    is_nested_in_class: bool = False,
    is_staticmethod: bool = False,
    is_classmethod: bool = False,
    declared_type_params: Sequence[TypeVarValue] = (),
) -> Sequence[ParamInfo]:
    """Visits and checks the arguments to a function."""
    from .annotations import has_invalid_paramspec_usage

    defaults = [_visit_default(node, ctx) for node in node.args.defaults]
    kw_defaults = [
        None if kw_default is None else _visit_default(kw_default, ctx)
        for kw_default in node.args.kw_defaults
    ]

    posonly_args = getattr(node.args, "posonlyargs", [])
    num_without_defaults = len(node.args.args) + len(posonly_args) - len(defaults)
    vararg_defaults = [None] if node.args.vararg is not None else []
    defaults = [
        *[None] * num_without_defaults,
        *defaults,
        *vararg_defaults,
        *kw_defaults,
    ]
    args: list[tuple[ParameterKind, ast.arg]] = [
        (ParameterKind.POSITIONAL_ONLY, arg) for arg in posonly_args
    ] + [(ParameterKind.POSITIONAL_OR_KEYWORD, arg) for arg in node.args.args]
    if node.args.vararg is not None:
        args.append((ParameterKind.VAR_POSITIONAL, node.args.vararg))
    args += [(ParameterKind.KEYWORD_ONLY, arg) for arg in node.args.kwonlyargs]
    if node.args.kwarg is not None:
        args.append((ParameterKind.VAR_KEYWORD, node.args.kwarg))

    # Support the historical positional-only convention (`__x`) only when
    # no explicit "/" positional-only marker is present.
    if not posonly_args:
        saw_positional_or_keyword = False
        for idx, (kind, arg) in enumerate(args):
            if (
                kind is ParameterKind.POSITIONAL_OR_KEYWORD
                and is_positional_only_arg_name(arg.arg)
            ):
                if saw_positional_or_keyword:
                    ctx.show_error(
                        arg,
                        "Historical positional-only parameter may not follow a"
                        " positional-or-keyword parameter",
                        error_code=ErrorCode.invalid_positional_only,
                    )
                for previous_idx in range(idx + 1):
                    previous_kind, previous_arg = args[previous_idx]
                    if previous_kind is ParameterKind.POSITIONAL_OR_KEYWORD:
                        args[previous_idx] = (
                            ParameterKind.POSITIONAL_ONLY,
                            previous_arg,
                        )
                saw_positional_or_keyword = False
                continue
            is_implicit_method_first_param = (
                idx == 0
                and is_nested_in_class
                and not is_staticmethod
                and not isinstance(node, ast.Lambda)
            )
            if (
                kind is ParameterKind.POSITIONAL_OR_KEYWORD
                and not is_implicit_method_first_param
            ):
                saw_positional_or_keyword = True

    params = []
    tv_index = 1

    paramspecs_in_scope = _paramspec_identities_from_context(ctx)
    for type_param in declared_type_params:
        if is_instance_of_typing_name(type_param.typevar, "ParamSpec"):
            paramspecs_in_scope.add(type_param.typevar)
    seen_paramspec_args: tuple[ast.arg, ParamSpecArgsValue] | None = None
    paramspec_args_has_intervening_param = False
    value: Value | AnnotationExpr
    for idx, (param, default) in enumerate(zip_longest(args, defaults)):
        assert param is not None, "must have more args than defaults"
        kind, arg = param
        is_self = (
            idx == 0
            and enclosing_class is not None
            and not is_staticmethod
            and not isinstance(node, ast.Lambda)
        )
        if arg.annotation is not None:
            value = ctx.expr_of_annotation(arg.annotation)
            inner_value, _ = value.maybe_unqualify(set(Qualifier))
            if (
                inner_value is not None
                and not (
                    isinstance(inner_value, InputSigValue)
                    and isinstance(inner_value.input_sig, ParamSpecSig)
                )
                and not isinstance(
                    inner_value, (ParamSpecArgsValue, ParamSpecKwargsValue)
                )
            ):
                paramspecs_in_scope.update(
                    _paramspec_identities_from_value(inner_value)
                )
            allows_paramspec_component = (
                kind is ParameterKind.VAR_POSITIONAL
                and isinstance(inner_value, ParamSpecArgsValue)
                and inner_value.param_spec in paramspecs_in_scope
            ) or (
                kind is ParameterKind.VAR_KEYWORD
                and isinstance(inner_value, ParamSpecKwargsValue)
                and inner_value.param_spec in paramspecs_in_scope
            )
            if isinstance(inner_value, InputSigValue):
                if isinstance(inner_value.input_sig, ParamSpecSig):
                    ctx.show_error(
                        arg,
                        "ParamSpec cannot be used in this annotation context",
                        error_code=ErrorCode.invalid_annotation,
                    )
                else:
                    ctx.show_error(
                        arg,
                        f"Unrecognized annotation {inner_value}",
                        error_code=ErrorCode.invalid_annotation,
                    )
                value = AnyValue(AnySource.error)
            else:
                if inner_value is not None and _is_invalid_generic_annotation_value(
                    inner_value
                ):
                    ctx.show_error(
                        arg,
                        "Generic[...] is valid only as a base class",
                        error_code=ErrorCode.invalid_annotation,
                    )
                    value = AnyValue(AnySource.error)
                if (
                    inner_value is not None
                    and has_invalid_paramspec_usage(inner_value, ctx)
                    and not allows_paramspec_component
                ):
                    ctx.show_error(
                        arg,
                        "ParamSpec cannot be used in this annotation context",
                        error_code=ErrorCode.invalid_annotation,
                    )
                    value = AnyValue(AnySource.error)
                elif default is not None and inner_value is not None:
                    tv_map = has_relation(
                        inner_value, default, Relation.ASSIGNABLE, ctx
                    )
                    if isinstance(tv_map, CanAssignError):
                        ctx.show_error(
                            arg,
                            f"Default value for argument {arg.arg} incompatible"
                            f" with declared type {inner_value}",
                            error_code=ErrorCode.incompatible_default,
                            detail=tv_map.display(),
                        )
                if (
                    is_self
                    and getattr(node, "name", None) == "__init__"
                    and inner_value is not None
                ):
                    class_type_param_ids = _type_param_identities_for_class(
                        enclosing_class
                    )
                    if class_type_param_ids and any(
                        isinstance(subval, TypeVarValue)
                        and subval.typevar in class_type_param_ids
                        for subval in inner_value.walk_values()
                    ):
                        ctx.show_error(
                            arg,
                            "Class-scoped type variables are not allowed in __init__"
                            " self annotation",
                            error_code=ErrorCode.invalid_annotation,
                        )
        elif is_self:
            assert enclosing_class is not None
            self_tv_value = TypeVarValue(SelfT, bound=enclosing_class)
            if is_classmethod or getattr(node, "name", None) in IMPLICIT_CLASSMETHODS:
                value = SubclassValue(self_tv_value)
            else:
                # normal method
                value = self_tv_value
        else:
            # This is meant to exclude methods in nested classes. It's a bit too
            # conservative for cases such as a function nested in a method nested in a
            # class nested in a function.
            if not isinstance(node, ast.Lambda) and not (
                idx == 0 and not is_staticmethod and is_nested_in_class
            ):
                ctx.show_error(
                    arg,
                    f"Missing type annotation for parameter {arg.arg}",
                    error_code=ErrorCode.missing_parameter_annotation,
                )
            if isinstance(node, ast.Lambda):
                value = TypeVarValue(TypeVar(f"T{tv_index}"))
                tv_index += 1
            else:
                value = AnyValue(AnySource.unannotated)
            if default is not None:
                value = unite_values(value, default)

        value = translate_vararg_type(kind, value, ctx, error_ctx=ctx, node=arg)
        if seen_paramspec_args is not None:
            _, ps_args = seen_paramspec_args
            matches_trailing_kwargs = (
                kind is ParameterKind.VAR_KEYWORD
                and isinstance(value, ParamSpecKwargsValue)
                and value.param_spec is ps_args.param_spec
            )
            if not matches_trailing_kwargs:
                paramspec_args_has_intervening_param = True
        if isinstance(value, ParamSpecArgsValue):
            if kind is ParameterKind.VAR_POSITIONAL:
                if value.param_spec in paramspecs_in_scope:
                    seen_paramspec_args = (arg, value)
                    paramspec_args_has_intervening_param = False
                else:
                    ctx.show_error(
                        arg,
                        "ParamSpec cannot be used in this annotation context",
                        error_code=ErrorCode.invalid_annotation,
                    )
            else:
                ctx.show_error(
                    arg,
                    f"ParamSpec.args must be used on *args, not {arg.arg}",
                    error_code=ErrorCode.invalid_annotation,
                )
        elif isinstance(value, ParamSpecKwargsValue):
            if kind is ParameterKind.VAR_KEYWORD:
                if value.param_spec not in paramspecs_in_scope:
                    ctx.show_error(
                        arg,
                        "ParamSpec cannot be used in this annotation context",
                        error_code=ErrorCode.invalid_annotation,
                    )
                elif seen_paramspec_args is not None:
                    _, ps_args = seen_paramspec_args
                    if ps_args.param_spec is not value.param_spec:
                        ctx.show_error(
                            arg,
                            "The same ParamSpec must be used on *args and **kwargs",
                            error_code=ErrorCode.invalid_annotation,
                        )
                        seen_paramspec_args = None
                    elif paramspec_args_has_intervening_param:
                        ctx.show_error(
                            arg,
                            "ParamSpec.args and ParamSpec.kwargs must be adjacent",
                            error_code=ErrorCode.invalid_annotation,
                        )
                        seen_paramspec_args = None
                    else:
                        seen_paramspec_args = None
                else:
                    ctx.show_error(
                        arg,
                        "ParamSpec.kwargs must be used together with ParamSpec.args",
                        error_code=ErrorCode.invalid_annotation,
                    )
            else:
                ctx.show_error(
                    arg,
                    f"ParamSpec.kwargs must be used on **kwargs, not {arg.arg}",
                    error_code=ErrorCode.invalid_annotation,
                )
        elif kind is ParameterKind.VAR_KEYWORD and isinstance(value, TypedDictValue):
            overlapping_params = [
                existing.param.name
                for existing in params
                if existing.param.kind is not ParameterKind.POSITIONAL_ONLY
                and existing.param.name in value.items
            ]
            for name in overlapping_params:
                ctx.show_error(
                    arg,
                    f"Parameter {name} overlaps with TypedDict key in **kwargs",
                    error_code=ErrorCode.invalid_annotation,
                )
        elif kind is ParameterKind.VAR_KEYWORD and seen_paramspec_args is not None:
            ctx.show_error(
                arg,
                "ParamSpec.args must be used together with ParamSpec.kwargs",
                error_code=ErrorCode.invalid_annotation,
            )
            seen_paramspec_args = None

        param = SigParameter(arg.arg, kind, default, value)
        info = ParamInfo(param, arg, is_self)
        params.append(info)

    if seen_paramspec_args is not None:
        ps_args_arg, _ = seen_paramspec_args
        ctx.show_error(
            ps_args_arg,
            "ParamSpec.args must be used together with ParamSpec.kwargs",
            error_code=ErrorCode.invalid_annotation,
        )

    return params


def translate_vararg_type(
    kind: ParameterKind,
    typ: AnnotationExpr | Value,
    can_assign_ctx: CanAssignContext,
    *,
    error_ctx: ErrorContext | None = None,
    node: ast.AST | None = None,
) -> Value:
    if isinstance(typ, AnnotationExpr):
        inner_typ, qualifiers = typ.unqualify({Qualifier.Unpack})
        has_unpack = Qualifier.Unpack in qualifiers
    else:
        inner_typ = typ
        has_unpack = False
    if kind is ParameterKind.VAR_POSITIONAL:
        if has_unpack:
            if not TypedValue(tuple).is_assignable(inner_typ, can_assign_ctx):
                if error_ctx is not None and node is not None:
                    error_ctx.show_error(
                        node,
                        "Expected tuple type inside Unpack[]",
                        error_code=ErrorCode.invalid_annotation,
                    )
                return AnyValue(AnySource.error)
            return inner_typ
        elif isinstance(inner_typ, ParamSpecArgsValue):
            return inner_typ
        elif isinstance(inner_typ, TypeVarValue) and inner_typ.is_typevartuple():
            if error_ctx is not None and node is not None:
                error_ctx.show_error(
                    node,
                    "TypeVarTuple must be unpacked",
                    error_code=ErrorCode.invalid_annotation,
                )
            return AnyValue(AnySource.error)
        else:
            return GenericValue(tuple, [inner_typ])
    elif kind is ParameterKind.VAR_KEYWORD:
        if has_unpack:
            if isinstance(inner_typ, TypeVarValue):
                if error_ctx is not None and node is not None:
                    error_ctx.show_error(
                        node,
                        "Expected TypedDict type inside Unpack[] for **kwargs",
                        error_code=ErrorCode.invalid_annotation,
                    )
                return AnyValue(AnySource.error)
            inner_typ = replace_fallback(inner_typ)
            if not isinstance(inner_typ, TypedDictValue):
                if error_ctx is not None and node is not None:
                    error_ctx.show_error(
                        node,
                        "Expected TypedDict type inside Unpack[] for **kwargs",
                        error_code=ErrorCode.invalid_annotation,
                    )
                return AnyValue(AnySource.error)
            return inner_typ
        elif isinstance(inner_typ, ParamSpecKwargsValue):
            return inner_typ
        else:
            return GenericValue(dict, [TypedValue(str), inner_typ])
    elif has_unpack:
        if error_ctx is not None and node is not None:
            error_ctx.show_error(
                node,
                "Unpack[] can only be used on *args or **kwargs",
                error_code=ErrorCode.invalid_annotation,
            )
        return AnyValue(AnySource.error)
    return inner_typ


@dataclass
class IsGeneratorVisitor(ast.NodeVisitor):
    """Determine whether an async function is a generator.

    This is important because the return type of async generators
    should not be wrapped in Coroutine.

    We avoid recursing into nested functions, which is why we can't
    just use ast.walk.

    We do not need to check for yield from because it is illegal
    in async generators. We also skip checking nested comprehensions,
    because we error anyway if there is a yield within a comprehension.

    """

    is_generator: bool = False

    def visit_Yield(self, node: ast.Yield) -> None:
        self.is_generator = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass


def _signature_params_from_function_params(
    params: Sequence[ParamInfo],
) -> list[SigParameter]:
    normalized: list[SigParameter] = []
    pending_param: SigParameter | None = None
    pending_paramspec: ParamSpecArgsValue | None = None
    for param_info in params:
        param = param_info.param
        annotation = param.get_annotation()
        if pending_param is not None and pending_paramspec is not None:
            is_matching_kwargs = (
                param.kind is ParameterKind.VAR_KEYWORD
                and isinstance(annotation, ParamSpecKwargsValue)
                and annotation.param_spec is pending_paramspec.param_spec
            )
            if is_matching_kwargs:
                normalized.append(
                    SigParameter(
                        param.name,
                        ParameterKind.PARAM_SPEC,
                        annotation=InputSigValue(ParamSpecSig(annotation.param_spec)),
                    )
                )
                pending_param = None
                pending_paramspec = None
                continue
            normalized.append(pending_param)
            pending_param = None
            pending_paramspec = None
        if param.kind is ParameterKind.VAR_POSITIONAL and isinstance(
            annotation, ParamSpecArgsValue
        ):
            pending_param = param
            pending_paramspec = annotation
            continue
        normalized.append(param)
    if pending_param is not None:
        normalized.append(pending_param)
    return normalized


def compute_value_of_function(
    info: FunctionInfo, ctx: Context, *, result: Value | None = None
) -> Value:
    if result is None:
        result = info.return_annotation
    if result is None:
        result = AnyValue(AnySource.unannotated)
    if isinstance(info.node, ast.AsyncFunctionDef):
        visitor = IsGeneratorVisitor()
        for line in info.node.body:
            visitor.visit(line)
            if visitor.is_generator:
                break
        if not visitor.is_generator:
            result = make_coro_type(result)
    sig = Signature.make(
        mark_ellipsis_style_any_tail_parameters(
            _signature_params_from_function_params(info.params)
        ),
        result,
        has_return_annotation=info.return_annotation is not None,
    )
    val = CallableValue(sig, types.FunctionType)
    for unapplied, decorator, node in reversed(info.decorators):
        deprecated_message = _deprecated_message_from_decorator(unapplied, node, ctx)
        if deprecated_message is not None:
            if isinstance(val, CallableValue) and isinstance(val.signature, Signature):
                val = CallableValue(
                    replace(val.signature, deprecated=deprecated_message), val.typ
                )
            else:
                val = annotate_value(val, [DeprecatedExtension(deprecated_message)])
            continue
        # Special case asynq.asynq until we can create the type automatically
        if (
            asynq is not None
            and unapplied == KnownValue(asynq.asynq)
            and isinstance(val, CallableValue)
        ):
            sig = replace(val.signature, is_asynq=True)
            val = CallableValue(sig, val.typ)
            continue
        allow_call = isinstance(
            unapplied, KnownValue
        ) and SafeDecoratorsForNestedFunctions.contains(unapplied.val, ctx.options)
        val = ctx.check_call(node, decorator, [Composite(val)], allow_call=allow_call)
    return val


def _deprecated_message_from_decorator(
    unapplied: Value, node: ast.expr, ctx: Context
) -> str | None:
    if not (
        isinstance(unapplied, KnownValue)
        and (
            is_typing_name(unapplied.val, "deprecated")
            or unapplied.val is deprecated_decorator
        )
    ):
        return None
    if not isinstance(node, ast.Call) or not node.args:
        return None
    with ctx.catch_errors():
        message_value = ctx.visit_expression(node.args[0])
    if isinstance(message_value, KnownValue) and isinstance(message_value.val, str):
        return message_value.val
    return None
