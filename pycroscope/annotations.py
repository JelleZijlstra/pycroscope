"""

Code for understanding type annotations.

This file contains functions that turn various representations of
Python type annotations into :class:`pycroscope.value.Value` objects.

There are three major functions:

- :func:`type_from_runtime` takes a runtime Python object, for example
  ``type_from_value(int)`` -> ``TypedValue(int)``.
- :func:`type_from_value` takes an existing :class:`pycroscope.value.Value`
  object. For example, evaluating the expression ``int`` will produce
  ``KnownValue(int)``, and calling :func:`type_from_value` on that value
  will produce ``TypedValue(int)``.
- :func:`type_from_ast` takes an AST node and evaluates it into a type.

These functions all rely on each other. For example, when a forward
reference is found in a runtime annotation, the code parses it and calls
:func:`type_from_ast` to evaluate it.

These functions all use :class:`Context` objects to resolve names and
show errors.

"""

import ast
import builtins
import contextlib
import enum
import sys
import types
import typing
from collections.abc import Callable, Container, Generator, Hashable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    NewType,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from weakref import WeakKeyDictionary

import typing_extensions
from typing_extensions import (
    NoDefault,
    ParamSpec,
    Protocol,
    TypedDict,
    TypeIs,
    runtime_checkable,
)

import pycroscope
from pycroscope.annotated_types import get_annotated_types_extension
from pycroscope.input_sig import FullSignature, InputSigValue
from pycroscope.relations import (
    HashableProtoValue,
    Relation,
    has_relation,
    intersect_multi,
)

from . import type_evaluation
from .analysis_lib import object_from_string, override
from .error_code import Error, ErrorCode
from .extensions import (
    AsynqCallable,
    CustomCheck,
    ExternalType,
    Intersection,
    NoReturnGuard,
    Overlapping,
    ParameterTypeGuard,
    TypeGuard,
    deprecated,
)
from .find_unused import used
from .functions import FunctionDefNode
from .node_visitor import ErrorContext
from .safe import is_instance_of_typing_name, is_typing_name, is_union, safe_getattr
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    NO_ARG_SENTINEL,
    InvalidSignature,
    ParameterKind,
    Signature,
    SigParameter,
)
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnnotationExpr,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    CustomCheckExtension,
    DictIncompleteValue,
    Extension,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KVPair,
    MultiValuedValue,
    NewTypeValue,
    NoReturnGuardExtension,
    OverlappingValue,
    ParameterTypeGuardExtension,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    ParamSpecLike,
    ParamSpecParam,
    PartialCallValue,
    PartialValue,
    PartialValueOperation,
    Qualifier,
    SelfTVV,
    SequenceValue,
    SubclassValue,
    SyntheticClassObjectValue,
    TypeAlias,
    TypeAliasValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeGuardExtension,
    TypeIsExtension,
    TypeParam,
    TypeVarLike,
    TypeVarParam,
    TypeVarTupleLike,
    TypeVarTupleParam,
    TypeVarTupleValue,
    TypeVarType,
    TypeVarValue,
    Value,
    Variance,
    annotate_value,
    get_typevar_variance,
    iter_type_params_in_value,
    match_typevar_arguments,
    replace_fallback,
    replace_known_sequence_value,
    unite_values,
)

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor


CONTEXT_MANAGER_TYPES = (typing.ContextManager, contextlib.AbstractContextManager)
ASYNC_CONTEXT_MANAGER_TYPES = (
    typing.AsyncContextManager,
    contextlib.AbstractAsyncContextManager,
)
_SUBSCRIPT_RUNTIME_TYPE = TypedValue(type(list[int]))
_UNION_RUNTIME_TYPE = TypedValue(type(int | str))
_UNPACK_RUNTIME_TYPE = TypedValue(type(typing_extensions.Unpack[int]))
_ENUM_TYPE = getattr(enum, "EnumType", enum.EnumMeta)
_PARTIAL_CALL_TYPE_PARAM_CACHE: WeakKeyDictionary[ast.AST, TypeVarLike] = (
    WeakKeyDictionary()
)


def _is_valid_pep586_literal_value(value: object) -> bool:
    if value is None:
        return True
    if type(value) in (int, bool, str, bytes):
        return True
    if isinstance(value, enum.Enum):
        return True
    return isinstance(type(value), _ENUM_TYPE)


@runtime_checkable
class AnnotationVisitor(ErrorContext, CanAssignContext, Protocol):
    def resolve_name(
        self,
        node: ast.Name,
        error_node: ast.AST | None = None,
        suppress_errors: bool = False,
    ) -> tuple[Value, object]:
        raise NotImplementedError

    def invalid_self_annotation_message(self, annotation: ast.AST) -> str | None:
        raise NotImplementedError

    def get_type_alias_cache(self) -> dict[object, TypeAlias]:
        raise NotImplementedError


@dataclass
class Context:
    """A context for evaluating annotations.

    The base implementation does very little. Subclass this to do something more useful.

    """

    should_suppress_errors: bool = field(default=False, init=False)
    """While this is True, no annotation errors are emitted."""
    should_allow_undefined_names: bool = field(default=False, init=False)
    """While this is True, unresolved names may evaluate to an unknown type."""
    _being_evaluated: dict[int, Value] = field(default_factory=dict, init=False)
    _invalid_self_nodes: set[int] = field(default_factory=set, init=False)
    visitor: AnnotationVisitor | None = field(default=None, kw_only=True)
    can_assign_ctx: CanAssignContext | None = field(default=None, kw_only=True)

    def suppress_errors(self) -> AbstractContextManager[None]:
        """Temporarily suppress all annotation-evaluation errors."""
        return override(self, "should_suppress_errors", True)

    def allow_undefined_names(self) -> AbstractContextManager[None]:
        """Temporarily treat undefined annotation names as unknown types."""
        return override(self, "should_allow_undefined_names", True)

    def is_being_evaluted(self, obj: object) -> Value | None:
        return self._being_evaluated.get(id(obj))

    @contextlib.contextmanager
    def add_evaluation(
        self, obj: object, name: str, module: str, evaluator: Callable[[], Value]
    ) -> Generator[None, None, None]:
        """Temporarily add an object to the set of objects being evaluated.

        This is used to prevent infinite recursion when evaluating forward references.

        """
        obj_id = id(obj)
        value = TypeAliasValue(
            name, module, self.get_type_alias(name, evaluator, lambda: [])
        )
        self._being_evaluated[obj_id] = value
        try:
            yield
        finally:
            del self._being_evaluated[obj_id]

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: ast.AST | None = None,
    ) -> None:
        """Show an error found while evaluating an annotation."""
        pass

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`pycroscope.value.Value` corresponding to a name."""
        return AnyValue(AnySource.inference)

    def get_error_node(self) -> ast.AST | None:
        """Return the node that should be used as fallback for annotation errors."""
        return None

    def invalid_self_annotation_message(self, node: ast.AST) -> str | None:
        return None

    def maybe_show_invalid_self_annotation(self, node: ast.AST | None = None) -> None:
        if self.should_suppress_errors:
            return
        node = node or self.get_error_node()
        if node is None:
            return
        message = self.invalid_self_annotation_message(node)
        if message is None:
            return
        if id(node) in self._invalid_self_nodes:
            return
        self._invalid_self_nodes.add(id(node))
        self.show_error(message, ErrorCode.invalid_annotation, node=node)

    def handle_undefined_name(self, name: str) -> Value:
        if self.should_allow_undefined_names and not self.should_suppress_errors:
            self.show_error(
                f"Undefined name {name!r} used in annotation", ErrorCode.undefined_name
            )
            return AnyValue(AnySource.error)
        if self.should_suppress_errors:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {name!r} used in annotation", ErrorCode.undefined_name
        )
        return AnyValue(AnySource.error)

    def get_name_from_globals(self, name: str, globals: Mapping[str, Any]) -> Value:
        if name in globals:
            return KnownValue(globals[name])
        elif hasattr(builtins, name):
            return KnownValue(getattr(builtins, name))
        return self.handle_undefined_name(name)

    def get_attribute(self, root_value: Value, node: ast.Attribute) -> Value:
        if isinstance(root_value, KnownValue):
            try:
                value = KnownValue(getattr(root_value.val, node.attr))
                if _is_self_annotation_value(value):
                    self.maybe_show_invalid_self_annotation()
                return value
            except AttributeError:
                self.show_error(
                    f"{root_value.val!r} has no attribute {node.attr!r}", node=node
                )
                return AnyValue(AnySource.error)
        elif self.visitor is not None:
            with self.visitor.catch_errors():
                value = self.visitor.get_attribute_from_value(root_value, node.attr)
            if value != AnyValue(AnySource.error):
                if _is_self_annotation_value(value):
                    self.maybe_show_invalid_self_annotation()
                return value
        elif not isinstance(root_value, AnyValue):
            self.show_error(f"Cannot resolve annotation {root_value}", node=node)
        return AnyValue(AnySource.error)

    def get_type_alias(
        self,
        key: object,
        evaluator: typing.Callable[[], Value],
        evaluate_type_params: typing.Callable[[], Sequence[TypeParam]],
    ) -> TypeAlias:
        return TypeAlias(evaluator, evaluate_type_params)


@dataclass
class RuntimeEvaluator(type_evaluation.Evaluator, Context):
    globals: Mapping[str, object] = field(repr=False)

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(node, ctx=self, error_on_unrecognized=False)

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`pycroscope.value.Value` corresponding to a name."""
        return self.get_name_from_globals(node.id, self.globals)


@dataclass
class SyntheticEvaluator(type_evaluation.Evaluator):
    annotations_context: Context

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self.annotations_context)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(
            node, ctx=self.annotations_context, error_on_unrecognized=False
        )

    @classmethod
    def from_visitor(
        cls,
        node: FunctionDefNode,
        visitor: "pycroscope.name_check_visitor.NameCheckVisitor",
        return_annotation: Value,
    ) -> "SyntheticEvaluator":
        return cls(
            node,
            return_annotation,
            _DefaultContext(visitor, node, use_name_node_for_error=True),
        )


@used  # part of an API
def type_from_ast(
    ast_node: ast.AST,
    visitor: Optional["pycroscope.name_check_visitor.NameCheckVisitor"] = None,
    ctx: Context | None = None,
) -> Value:
    """Given an AST node representing an annotation, return a
    :class:`pycroscope.value.Value`.

    :param ast_node: AST node to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
    return _type_from_ast(ast_node, ctx)


@used  # part of an API
def annotation_expr_from_ast(
    ast_node: ast.AST,
    visitor: Optional["pycroscope.name_check_visitor.NameCheckVisitor"] = None,
    ctx: Context | None = None,
    suppress_errors: bool = False,
) -> AnnotationExpr:
    """Given an AST node representing an annotation, return a
    ``AnnotationExpr``."""
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
    if suppress_errors:
        with ctx.suppress_errors():
            return _annotation_expr_from_ast(ast_node, ctx)
    return _annotation_expr_from_ast(ast_node, ctx)


@used  # part of an API
def type_from_annotations(
    annotations: Mapping[str, object],
    key: str,
    *,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
) -> Value | None:
    try:
        annotation = annotations[key]
    except Exception:
        # Malformed __annotations__
        return None
    else:
        maybe_val = type_from_runtime(annotation, globals=globals, ctx=ctx)
        if maybe_val != AnyValue(AnySource.incomplete_annotation):
            return maybe_val
    return None


@used  # part of an API
def annotation_expr_from_annotations(
    annotations: Mapping[str, object],
    key: str,
    *,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
) -> AnnotationExpr | None:
    try:
        annotation = annotations[key]
    except Exception:
        # Malformed __annotations__
        return None
    else:
        maybe_val = annotation_expr_from_runtime(annotation, globals=globals, ctx=ctx)
        return maybe_val


def type_from_runtime(
    val: object,
    visitor: CanAssignContext | None = None,
    node: ast.AST | None = None,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
    suppress_errors: bool = False,
    allow_undefined_names: bool = False,
) -> Value:
    """Given a runtime annotation object, return a
    :class:`pycroscope.value.Value`.

    :param val: Object to evaluate. This will usually come from an
                ``__annotations__`` dictionary.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param globals: Dictionary of global variables that can be used
                    to resolve names. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """

    if ctx is None:
        ctx = _DefaultContext(visitor, node, globals)
    if suppress_errors and allow_undefined_names:
        with ctx.suppress_errors():
            with ctx.allow_undefined_names():
                return _type_from_runtime(val, ctx)
    if suppress_errors:
        with ctx.suppress_errors():
            return _type_from_runtime(val, ctx)
    if allow_undefined_names:
        with ctx.allow_undefined_names():
            return _type_from_runtime(val, ctx)
    return _type_from_runtime(val, ctx)


def annotation_expr_from_runtime(
    val: object,
    *,
    visitor: Optional["pycroscope.name_check_visitor.NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
    suppress_errors: bool = False,
    allow_undefined_names: bool = False,
) -> AnnotationExpr:
    if ctx is None:
        ctx = _DefaultContext(visitor, node, globals)
    if suppress_errors and allow_undefined_names:
        with ctx.suppress_errors():
            with ctx.allow_undefined_names():
                return _annotation_expr_from_runtime(val, ctx)
    if suppress_errors:
        with ctx.suppress_errors():
            return _annotation_expr_from_runtime(val, ctx)
    if allow_undefined_names:
        with ctx.allow_undefined_names():
            return _annotation_expr_from_runtime(val, ctx)
    return _annotation_expr_from_runtime(val, ctx)


def type_from_value(
    value: Value,
    visitor: CanAssignContext | None = None,
    node: ast.AST | None = None,
    ctx: Context | None = None,
    suppress_errors: bool = False,
    allow_undefined_names: bool = False,
) -> Value:
    """Given a :class:`pycroscope.value.Value` representing an annotation,
    return a :class:`pycroscope.value.Value` representing the type.

    The input value represents an expression, the output value represents
    a type. For example, the :term:`impl` of ``typing.cast(typ, val)``
    calls :func:`type_from_value` on the value it receives for its
    `typ` argument and returns the result.

    :param value: :class:`pycroscope.value.Value` to evaluate.

    :param visitor: Visitor class to use. This is used in the default
                    :class:`Context` to resolve names and show errors.
                    This is ignored if `ctx` is given.

    :param node: AST node that the annotation derives from. This is
                 used for showing errors. Ignored if `ctx` is given.

    :param ctx: :class:`Context` to use for evaluation.

    """
    if ctx is None:
        ctx = _DefaultContext(visitor, node)
    if suppress_errors and allow_undefined_names:
        with ctx.suppress_errors():
            with ctx.allow_undefined_names():
                return _type_from_value(value, ctx)
    if suppress_errors:
        with ctx.suppress_errors():
            return _type_from_value(value, ctx)
    if allow_undefined_names:
        with ctx.allow_undefined_names():
            return _type_from_value(value, ctx)
    return _type_from_value(value, ctx)


def annotation_expr_from_value(
    value: Value,
    *,
    visitor: Optional["pycroscope.name_check_visitor.NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    ctx: Context | None = None,
    suppress_errors: bool = False,
    allow_undefined_names: bool = False,
) -> AnnotationExpr:
    if ctx is None:
        ctx = _DefaultContext(visitor, node)
    if suppress_errors and allow_undefined_names:
        with ctx.suppress_errors():
            with ctx.allow_undefined_names():
                return _annotation_expr_from_value(value, ctx)
    if suppress_errors:
        with ctx.suppress_errors():
            return _annotation_expr_from_value(value, ctx)
    if allow_undefined_names:
        with ctx.allow_undefined_names():
            return _annotation_expr_from_value(value, ctx)
    return _annotation_expr_from_value(value, ctx)


def value_from_ast(
    ast_node: ast.AST,
    ctx: Context | None = None,
    *,
    visitor: Optional["pycroscope.name_check_visitor.NameCheckVisitor"] = None,
    error_on_unrecognized: bool = True,
) -> Value:
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
    try:
        val = _Visitor(ctx).visit(ast_node)
    except _UnsupportedAnnotationExpression:
        if error_on_unrecognized:
            ctx.show_error("Invalid type annotation", node=ast_node)
        return AnyValue(AnySource.error)
    return val


def _type_from_ast(node: ast.AST, ctx: Context) -> Value:
    val = value_from_ast(node, ctx)
    return _type_from_value(val, ctx)


def _annotation_expr_from_ast(node: ast.AST, ctx: Context) -> AnnotationExpr:
    val = value_from_ast(node, ctx)
    return _annotation_expr_from_value(val, ctx)


def _annotation_expr_from_runtime(val: object, ctx: Context) -> AnnotationExpr:
    if isinstance(val, str):
        if (result := ctx.is_being_evaluted(val)) is not None:
            return AnnotationExpr(ctx, result)
        final_value = AnyValue(AnySource.inference)
        with ctx.add_evaluation(val, val, "<unknown>", lambda: final_value):
            final_expr = _eval_forward_ref(val, ctx)
            final_value = final_expr.to_value(allow_qualifiers=True, allow_empty=True)
        return final_expr
    elif is_typing_name(val, "Final"):
        return AnnotationExpr(ctx, None, [(Qualifier.Final, None)])
    elif is_typing_name(val, "ClassVar"):
        return AnnotationExpr(ctx, None, [(Qualifier.ClassVar, None)])
    elif is_typing_name(val, "TypeAlias"):
        return AnnotationExpr(ctx, None, [(Qualifier.TypeAlias, None)])
    elif isinstance(val, InitVar):
        if isinstance(val.type, tuple):
            ctx.show_error("InitVar[] requires a single argument")
            return AnnotationExpr(ctx, AnyValue(AnySource.error))
        return AnnotationExpr(
            ctx, _type_from_runtime(val.type, ctx), [(Qualifier.InitVar, None)]
        )
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        if (result := ctx.is_being_evaluted(val)) is not None:
            return AnnotationExpr(ctx, result)
        final_value = AnyValue(AnySource.inference)
        with ctx.add_evaluation(
            val,
            val.__forward_arg__,  # static analysis: ignore[undefined_attribute]
            val.__forward_module__,  # static analysis: ignore[undefined_attribute]
            lambda: final_value,
        ):
            # Forward refs may be defined in a different file and errors can be misattributed.
            with ctx.suppress_errors():
                # static analysis: ignore[undefined_attribute]
                final_expr = _eval_forward_ref(val.__forward_arg__, ctx)
                final_value = final_expr.to_value(allow_qualifiers=True)
        return final_expr
    elif isinstance(val, GenericAlias) and getattr(val, "__unpacked__", False):
        inner = _type_from_runtime(val, ctx)
        return AnnotationExpr(ctx, inner, qualifiers=[(Qualifier.Unpack, None)])
    else:
        if not is_instance_of_typing_name(
            val, "ParamSpecArgs"
        ) and not is_instance_of_typing_name(val, "ParamSpecKwargs"):
            origin = get_origin(val)
            if origin is not None:
                args = get_args(val)
                return _annotation_expr_of_origin_args(origin, args, val, ctx)
        return AnnotationExpr(ctx, _type_from_runtime(val, ctx))


def _type_from_runtime(val: Any, ctx: Context) -> Value:
    if isinstance(val, str):
        if (result := ctx.is_being_evaluted(val)) is not None:
            return result
        final_value = AnyValue(AnySource.inference)
        with ctx.add_evaluation(val, val, "<unknown>", lambda: final_value):
            final_value = _eval_forward_ref(val, ctx).to_value()
        return final_value
    elif is_instance_of_typing_name(val, "ParamSpecArgs"):
        origin = get_origin(val)
        if is_instance_of_typing_name(origin, "ParamSpec"):
            return ParamSpecArgsValue(origin)
        return AnyValue(AnySource.inference)
    elif is_instance_of_typing_name(val, "ParamSpecKwargs"):
        origin = get_origin(val)
        if is_instance_of_typing_name(origin, "ParamSpec"):
            return ParamSpecKwargsValue(origin)
        return AnyValue(AnySource.inference)
    origin = get_origin(val)
    if origin is None:
        maybe_origin = getattr(val, "__origin__", None)
        maybe_args = getattr(val, "__args__", None)
        if maybe_origin is not None and isinstance(maybe_args, tuple):
            return _value_of_origin_args(maybe_origin, maybe_args, val, ctx)
    if origin is not None:
        args = get_args(val)
        return _value_of_origin_args(origin, args, val, ctx)
    # Can't use is_typeddict() here because we still want to support
    # mypy_extensions.TypedDict
    elif is_instance_of_typing_name(val, "_TypedDictMeta"):
        required_keys = getattr(val, "__required_keys__", None)
        readonly_keys = getattr(val, "__readonly_keys__", None)
        total = getattr(val, "__total__", True)
        extra_keys = None
        # Deprecated
        if hasattr(val, "__extra_keys__"):
            extra_keys = _annotation_expr_from_runtime(val.__extra_keys__, ctx)
        # typing_extensions 4.12
        # static analysis: ignore[value_always_true]
        if isinstance(val, typing_extensions._TypedDictMeta) and not hasattr(
            typing_extensions, "TypeForm"
        ):
            if hasattr(val, "__closed__") and val.__closed__:
                extra_keys = _annotation_expr_from_runtime(val.__extra_items__, ctx)
        else:
            # Newer typing-extensions
            if hasattr(val, "__closed__") and val.__closed__:
                extra_keys = AnnotationExpr(ctx, NO_RETURN_VALUE)
            elif hasattr(val, "__extra_items__") and not is_typing_name(
                val.__extra_items__, "NoExtraItems"
            ):
                extra_keys = _annotation_expr_from_runtime(val.__extra_items__, ctx)
        if extra_keys is None:
            extra_keys_val = None
            extra_readonly = False
        else:
            extra_keys_val, qualifiers = extra_keys.unqualify({Qualifier.ReadOnly})
            extra_readonly = Qualifier.ReadOnly in qualifiers
        return TypedDictValue(
            {
                key: _get_typeddict_value(
                    value, ctx, key, required_keys, total, readonly_keys
                )
                for key, value in val.__annotations__.items()
            },
            extra_keys=extra_keys_val,
            extra_keys_readonly=extra_readonly,
        )
    elif val is AsynqCallable:
        return CallableValue(Signature.make([ELLIPSIS_PARAM], is_asynq=True))
    elif val is Intersection:
        ctx.show_error("Intersection[] is missing arguments")
        return AnyValue(AnySource.error)
    elif val is Overlapping:
        ctx.show_error("Overlapping[] is missing arguments")
        return AnyValue(AnySource.error)
    elif is_typing_name(val, "Any"):
        return AnyValue(AnySource.explicit)
    elif is_typing_name(val, "TypeForm"):
        # Bare TypeForm is equivalent to TypeForm[Any].
        return TypeFormValue(AnyValue(AnySource.explicit))
    elif isinstance(val, type):
        return _maybe_typed_value(val)
    elif val is None:
        return KnownValue(None)
    elif is_typing_name(val, "NoReturn") or is_typing_name(val, "Never"):
        return NO_RETURN_VALUE
    elif is_typing_name(val, "Self"):
        return SelfTVV
    elif is_typing_name(val, "LiteralString"):
        return TypedValue(str, literal_only=True)
    elif hasattr(val, "__supertype__"):
        supertype = _type_from_runtime(val.__supertype__, ctx)
        return NewTypeValue(val.__name__, supertype, val)
    elif is_instance_of_typing_name(val, "ParamSpec"):
        return InputSigValue(ParamSpecParam(val))
    elif is_typevarlike(val):
        type_param = make_type_param(val, ctx=ctx)
        if isinstance(type_param, TypeVarParam):
            return TypeVarValue(type_param)
        if isinstance(type_param, TypeVarTupleParam):
            return TypeVarTupleValue(type_param)
        return InputSigValue(type_param)
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        if (result := ctx.is_being_evaluted(val)) is not None:
            return result
        final_value = AnyValue(AnySource.inference)
        with ctx.add_evaluation(
            val, val.__forward_arg__, val.__forward_module__, lambda: final_value
        ):
            # Forward refs may be defined in a different file and errors can be misattributed.
            with ctx.suppress_errors():
                final_value = _eval_forward_ref(val.__forward_arg__, ctx).to_value()
        return final_value
    elif is_instance_of_typing_name(val, "TypeAliasType"):
        alias = ctx.get_type_alias(
            val,
            lambda: type_from_runtime(val.__value__, ctx=ctx),
            lambda: tuple(make_type_param(tv, ctx) for tv in val.__type_params__),
        )
        return TypeAliasValue(val.__name__, val.__module__, alias)
    elif val is Ellipsis:
        # valid in Callable[..., ]
        return AnyValue(AnySource.explicit)
    elif isinstance(val, TypeGuard):
        return AnnotatedValue(
            TypedValue(bool),
            [TypeGuardExtension(_type_from_runtime(val.guarded_type, ctx))],
        )
    elif isinstance(val, AsynqCallable):
        params = _callable_args_from_runtime(val.args, "AsynqCallable", ctx)
        sig = Signature.make(
            params, _type_from_runtime(val.return_type, ctx), is_asynq=True
        )
        return CallableValue(sig)
    elif isinstance(val, Intersection):
        if not val.args:
            ctx.show_error("Intersection[] is missing arguments")
            return AnyValue(AnySource.error)
        return IntersectionValue(
            tuple(_type_from_runtime(subval, ctx) for subval in val.args)
        )
    elif isinstance(val, Overlapping):
        return OverlappingValue(_type_from_runtime(val.arg, ctx))
    elif isinstance(val, ExternalType):
        try:
            typ = object_from_string(val.type_path)
        except Exception:
            ctx.show_error(f"Cannot resolve type {val.type_path!r}")
            return AnyValue(AnySource.error)
        return _type_from_runtime(typ, ctx)
    elif is_typing_name(val, "TypedDict"):
        return KnownValue(TypedDict)
    elif is_typing_name(val, "NamedTuple"):
        return TypedValue(tuple)
    else:
        ctx.show_error(f"Invalid type annotation {val}")
        return AnyValue(AnySource.error)


def make_type_param_from_value(
    value: Value,
    ctx: Context | None = None,
    *,
    visitor: "NameCheckVisitor | None" = None,
    node: ast.AST | None = None,
) -> TypeParam | None:
    if ctx is None:
        assert visitor is not None, "visitor must be provided if ctx is not"
        ctx = _DefaultContext(visitor, node)
    if isinstance(value, PartialCallValue):
        runtime_val = replace_fallback(value.runtime_value)
        if isinstance(runtime_val, TypedValue):
            if is_typing_name(runtime_val.typ, "TypeVar"):
                name, default = _extract_common_type_param_args(value, ctx)
                if name is None:
                    return None
                tv = typing.cast(
                    TypeVarType,
                    _synthetic_type_param_for_partial_call(
                        value, name, lambda name: TypeVar(name)
                    ),
                )
                if value.arguments["bound"] is NO_ARG_SENTINEL:
                    bound = None
                else:
                    bound = type_from_value(value.arguments["bound"], ctx=ctx)
                if (
                    isinstance(value.arguments["constraints"], SequenceValue)
                    and value.arguments["constraints"].members
                ):
                    constraints = [
                        type_from_value(constraint, ctx=ctx)
                        for constraint in (
                            value.arguments["constraints"].get_member_sequence() or ()
                        )
                    ]
                else:
                    constraints = None
                infer_variance = _extract_boolean_arg(value, "infer_variance")
                covariant = _extract_boolean_arg(value, "covariant")
                contravariant = _extract_boolean_arg(value, "contravariant")
                if covariant is None or contravariant is None:
                    return None
                match (infer_variance, covariant, contravariant):
                    case (True, _, _):
                        variance = Variance.INFERRED
                    case (_, True, _):
                        variance = Variance.COVARIANT
                    case (_, _, True):
                        variance = Variance.CONTRAVARIANT
                    case _:
                        variance = Variance.INVARIANT
                return TypeVarParam(
                    tv,
                    bound=bound,
                    constraints=tuple(constraints) if constraints is not None else (),
                    variance=variance,
                    default=default,
                )
            elif is_typing_name(runtime_val.typ, "ParamSpec"):
                name, default = _extract_common_type_param_args(value, ctx)
                if name is None:
                    return None
                ps = typing.cast(
                    ParamSpecLike,
                    _synthetic_type_param_for_partial_call(
                        value, name, lambda name: ParamSpec(name)
                    ),
                )
                return ParamSpecParam(ps, default=default)
            elif is_typing_name(runtime_val.typ, "TypeVarTuple"):
                name, default = _extract_common_type_param_args(value, ctx)
                if name is None:
                    return None
                tvt = typing.cast(
                    TypeVarTupleLike,
                    _synthetic_type_param_for_partial_call(
                        value, name, lambda name: typing_extensions.TypeVarTuple(name)
                    ),
                )
                return TypeVarTupleParam(tvt, default=default)
    value = replace_fallback(value)
    if isinstance(value, KnownValue) and is_typevarlike(value.val):
        return make_type_param(value.val, ctx=ctx)
    return None


def _extract_boolean_arg(pcv: PartialCallValue, arg_name: str) -> bool | None:
    arg_val = pcv.arguments.get(arg_name, NO_ARG_SENTINEL)
    if arg_val is NO_ARG_SENTINEL:
        return None
    if isinstance(arg_val, KnownValue) and isinstance(arg_val.val, bool):
        return arg_val.val
    return None


def _synthetic_type_param_for_partial_call(
    pcv: PartialCallValue, name: str, factory: Callable[[str], TypeVarLike]
) -> TypeVarLike:
    if pcv.node is None:
        return factory(name)
    cached = _PARTIAL_CALL_TYPE_PARAM_CACHE.get(pcv.node)
    if cached is None:
        cached = factory(name)
        _PARTIAL_CALL_TYPE_PARAM_CACHE[pcv.node] = cached
    return cached


def _extract_common_type_param_args(
    pcv: PartialCallValue, ctx: Context
) -> tuple[str | None, Value | None]:
    name_val = pcv.arguments["name"]
    if isinstance(name_val, KnownValue) and isinstance(name_val.val, str):
        name = name_val.val
    else:
        name = None
    default_val = pcv.arguments.get("default", NO_ARG_SENTINEL)
    if default_val is NO_ARG_SENTINEL:
        default = None
    else:
        default = type_from_value(default_val, ctx=ctx)
    return name, default


def _type_param_component_from_runtime(val: object, ctx: Context) -> Value:
    if is_instance_of_typing_name(val, "TypeVar"):
        type_param = make_type_param(val, ctx=ctx)
        assert isinstance(type_param, TypeVarParam)
        return TypeVarValue(type_param)
    if is_instance_of_typing_name(val, "TypeVarTuple"):
        type_param = make_type_param(val, ctx=ctx)
        assert isinstance(type_param, TypeVarTupleParam)
        return TypeVarTupleValue(type_param)
    if is_instance_of_typing_name(val, "ParamSpec"):
        type_param = make_type_param(val, ctx=ctx)
        assert isinstance(type_param, ParamSpecParam)
        return InputSigValue(type_param)
    if isinstance(val, tuple):
        return SequenceValue(
            tuple,
            [
                (False, _type_param_component_from_runtime(member, ctx))
                for member in val
            ],
        )
    if isinstance(val, list):
        return SequenceValue(
            list,
            [
                (False, _type_param_component_from_runtime(member, ctx))
                for member in val
            ],
        )
    return _type_from_runtime(val, ctx)


def make_type_param(
    tv: TypeVarLike,
    ctx: Context | None = None,
    *,
    visitor: "pycroscope.name_check_visitor.NameCheckVisitor | None" = None,
    node: ast.AST | None = None,
) -> TypeParam:
    if ctx is None:
        assert visitor is not None, "visitor must be provided if ctx is not"
        ctx = _DefaultContext(visitor, node)
    runtime_default = getattr(tv, "__default__", NoDefault)
    if runtime_default is not NoDefault:
        default = _type_param_component_from_runtime(runtime_default, ctx)
    else:
        default = None
    if isinstance(tv, (TypeVar, typing_extensions.TypeVar)):
        if getattr(tv, "__bound__", None) is not None:
            bound = _type_from_runtime(tv.__bound__, ctx)
        else:
            bound = None
        if getattr(tv, "__constraints__", ()):
            constraints = tuple(
                _type_from_runtime(constraint, ctx) for constraint in tv.__constraints__
            )
        else:
            constraints = ()
        return TypeVarParam(
            tv,
            bound=bound,
            constraints=constraints,
            default=default,
            variance=get_typevar_variance(tv),
        )
    if is_instance_of_typing_name(tv, "ParamSpec"):
        return ParamSpecParam(tv, default=default)
    if is_instance_of_typing_name(tv, "TypeVarTuple"):
        return TypeVarTupleParam(tv, default=default)
    raise TypeError(f"Unsupported type parameter: {tv!r}")


def _get_can_assign_context(ctx: Context) -> CanAssignContext | None:
    return ctx.can_assign_ctx


def _is_assignable_for_alias_arg(expected: Value, actual: Value, ctx: Context) -> bool:
    can_assign_ctx = _get_can_assign_context(ctx)
    if can_assign_ctx is None:
        return True

    result = has_relation(expected, actual, Relation.ASSIGNABLE, can_assign_ctx)
    return not isinstance(result, CanAssignError)


def _iter_alias_arg_possibilities(actual: Value) -> Sequence[Value]:
    if isinstance(actual, TypeVarValue):
        if actual.typevar_param.constraints:
            return actual.typevar_param.constraints
        return (actual.get_upper_bound_value(),)
    return (actual,)


def _is_alias_arg_compatible_with_bound(
    expected: Value, actual: Value, ctx: Context
) -> bool:
    return all(
        _is_assignable_for_alias_arg(expected, possibility, ctx)
        for possibility in _iter_alias_arg_possibilities(actual)
    )


def _is_alias_arg_compatible_with_constraints(
    constraints: Sequence[Value], actual: Value, ctx: Context
) -> bool:
    return all(
        any(
            _is_assignable_for_alias_arg(constraint, possibility, ctx)
            for constraint in constraints
        )
        for possibility in _iter_alias_arg_possibilities(actual)
    )


def _get_generic_type_parameters_for_annotation(
    typ: type, ctx: Context
) -> Sequence[TypeParam]:
    runtime_type_params = getattr(typ, "__parameters__", ())
    if isinstance(runtime_type_params, tuple) and runtime_type_params:
        return tuple(make_type_param(tp, ctx=ctx) for tp in runtime_type_params)
    can_assign_ctx = _get_can_assign_context(ctx)
    if can_assign_ctx is None:
        return ()
    return can_assign_ctx.get_type_parameters(typ)


def is_typevarlike(obj: object) -> TypeIs[TypeVarLike]:
    return (
        is_instance_of_typing_name(obj, "TypeVar")
        or is_instance_of_typing_name(obj, "TypeVarTuple")
        or is_instance_of_typing_name(obj, "ParamSpec")
    )


def _make_runtime_type_alias_value(
    alias_value: Value, type_params: Sequence[TypeParam], module: str = "typing"
) -> TypeAliasValue:
    normalized_type_params = tuple(dict.fromkeys(type_params))
    alias = TypeAlias(
        evaluator=lambda alias_value=alias_value: alias_value,
        evaluate_type_params=lambda normalized_type_params=normalized_type_params: (
            normalized_type_params
        ),
    )
    return TypeAliasValue(
        "<runtime_generic_alias>", module, alias, runtime_allows_value_call=True
    )


def _infer_alias_type_params_from_value(alias_value: Value) -> tuple[TypeParam, ...]:
    inferred_type_params: list[TypeParam] = []
    seen_type_params: set[object] = set()
    for extracted in iter_type_params_in_value(alias_value):
        identity = extracted.typevar
        if identity in seen_type_params:
            continue
        seen_type_params.add(identity)
        inferred_type_params.append(extracted)
    return tuple(inferred_type_params)


def _runtime_type_alias_from_runtime_value(
    runtime_value: object, ctx: Context
) -> TypeAliasValue | None:
    runtime_type_params = getattr(runtime_value, "__parameters__", ())
    if not isinstance(runtime_type_params, tuple) or not runtime_type_params:
        return None

    origin = get_origin(runtime_value)
    args: tuple[object, ...]
    if origin is not None:
        args = get_args(runtime_value)
    else:
        maybe_origin = getattr(runtime_value, "__origin__", None)
        maybe_args = getattr(runtime_value, "__args__", None)
        if maybe_origin is None or not isinstance(maybe_args, tuple):
            return None
        origin = maybe_origin
        args = maybe_args

    alias_value = _value_of_origin_args(origin, args, runtime_value, ctx)
    inferred_type_params = _infer_alias_type_params_from_value(alias_value)
    if inferred_type_params:
        type_params = inferred_type_params
    else:
        type_params = tuple(
            make_type_param(param, ctx=ctx) for param in runtime_type_params
        )
    return _make_runtime_type_alias_value(
        alias_value,
        type_params,
        module=safe_getattr(runtime_value, "__module__", "typing") or "typing",
    )


def _runtime_type_alias_from_partial_value(
    partial_value: PartialValue, ctx: Context
) -> TypeAliasValue | None:
    if partial_value.operation is not PartialValueOperation.SUBSCRIPT:
        return None
    alias_value = _type_from_subscripted_value(
        partial_value.root, partial_value.members, ctx
    )
    inferred_type_params = _infer_alias_type_params_from_value(alias_value)
    if not inferred_type_params:
        return None
    return _make_runtime_type_alias_value(alias_value, inferred_type_params)


# TODO: I think this ends up the same as just returning match_typevar_arguments()
def _match_type_alias_arg_values(
    type_params: Sequence[TypeParam], args_vals: Sequence[Value]
) -> Sequence[tuple[TypeParam, Value]] | None:
    matched = match_typevar_arguments(type_params, args_vals)
    if matched is None:
        return None
    values = [value for _, value in matched]
    return list(zip(type_params, values))


def _is_paramspec_annotation(value: Value) -> bool:
    return isinstance(value, InputSigValue) and isinstance(
        value.input_sig, ParamSpecParam
    )


def has_invalid_paramspec_usage(
    value: Value, can_assign_ctx: CanAssignContext | None
) -> bool:
    if _is_paramspec_annotation(value):
        return True
    if isinstance(value, (ParamSpecArgsValue, ParamSpecKwargsValue)):
        return True
    if isinstance(value, AnnotatedValue):
        return has_invalid_paramspec_usage(value.value, can_assign_ctx)
    if isinstance(value, MultiValuedValue):
        return any(
            has_invalid_paramspec_usage(subval, can_assign_ctx) for subval in value.vals
        )
    if isinstance(value, TypeAliasValue):
        alias_type_params = tuple(value.alias.get_type_params())
        if len(alias_type_params) == len(value.type_arguments):
            for type_param, type_arg in zip(alias_type_params, value.type_arguments):
                if _is_paramspec_annotation(type_arg):
                    if not isinstance(type_param, ParamSpecParam):
                        return True
                elif has_invalid_paramspec_usage(type_arg, can_assign_ctx):
                    return True
            return False
        return any(
            has_invalid_paramspec_usage(type_arg, can_assign_ctx)
            for type_arg in value.type_arguments
        )
    if isinstance(value, CallableValue):
        signature = value.signature
        if not isinstance(signature, Signature):
            return False
        if has_invalid_paramspec_usage(signature.return_value, can_assign_ctx):
            return True
        for param in signature.parameters.values():
            annotation = param.annotation
            if _is_paramspec_annotation(annotation):
                if param.kind is not ParameterKind.PARAM_SPEC:
                    return True
            elif has_invalid_paramspec_usage(annotation, can_assign_ctx):
                return True
        return False
    if isinstance(value, GenericValue):
        type_params: Sequence[TypeParam] = ()
        if can_assign_ctx is not None:
            type_params = can_assign_ctx.get_type_parameters(value.typ)
        for i, arg in enumerate(value.args):
            if _is_paramspec_annotation(arg):
                if i >= len(type_params) or not isinstance(
                    type_params[i], ParamSpecParam
                ):
                    return True
            elif has_invalid_paramspec_usage(arg, can_assign_ctx):
                return True
        return False
    return False


def _type_from_runtime_type_alias_arg(
    arg: object, type_param: TypeParam, ctx: Context
) -> Value:
    if isinstance(type_param, ParamSpecParam):
        if is_typing_name(get_origin(arg), "Concatenate"):
            concatenate_members = [
                _type_from_runtime(member, ctx) for member in get_args(arg)
            ]
            return _paramspec_value_from_concatenate_members(concatenate_members, ctx)
        if isinstance(arg, tuple):
            return SequenceValue(
                tuple, [(False, _type_from_runtime(member, ctx)) for member in arg]
            )
        if isinstance(arg, list):
            return SequenceValue(
                list, [(False, _type_from_runtime(member, ctx)) for member in arg]
            )
    return _type_from_runtime(arg, ctx)


def _type_from_value_type_alias_arg(
    arg: Value, type_param: TypeParam, ctx: Context
) -> Value:
    if isinstance(type_param, ParamSpecParam):
        concatenate_value = _maybe_paramspec_concatenate_value(arg, ctx)
        if concatenate_value is not None:
            return concatenate_value
        if isinstance(arg, SequenceValue) and arg.typ in (list, tuple):
            members = arg.get_member_sequence()
            if members is not None:
                return SequenceValue(
                    arg.typ,
                    [(False, _type_from_value(member, ctx)) for member in members],
                )
        return arg
    if isinstance(type_param, TypeVarTupleParam):
        if arg == KnownValue(()):
            return SequenceValue(tuple, [])
        if isinstance(arg, KnownValue) and isinstance(arg.val, tuple):
            return SequenceValue(
                tuple, [(False, _type_from_runtime(member, ctx)) for member in arg.val]
            )
        if isinstance(arg, SequenceValue) and arg.typ is tuple:
            members = arg.get_member_sequence()
            if members is not None:
                return SequenceValue(
                    tuple,
                    [(False, _type_from_value(member, ctx)) for member in members],
                )
        expr = _annotation_expr_from_value(arg, ctx)
        unpacked, qualifiers = expr.unqualify({Qualifier.Unpack})
        if Qualifier.Unpack in qualifiers:
            unpacked_members = _unpack_value(unpacked)
            if unpacked_members is None:
                ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                return AnyValue(AnySource.error)
            if (
                len(unpacked_members) == 1
                and unpacked_members[0][0]
                and isinstance(unpacked_members[0][1], TypeVarTupleValue)
            ):
                return unpacked_members[0][1]
            return SequenceValue(tuple, list(unpacked_members))
    return _type_from_alias_argument_value(arg, ctx)


def _type_from_alias_argument_value(arg: Value, ctx: Context) -> Value:
    if type(arg) is TypedValue:
        return arg
    if isinstance(arg, MultiValuedValue):
        return unite_values(
            *[_type_from_alias_argument_value(member, ctx) for member in arg.vals]
        )
    if isinstance(arg, IntersectionValue):
        return IntersectionValue(
            tuple(_type_from_alias_argument_value(member, ctx) for member in arg.vals)
        )
    return _type_from_value(arg, ctx)


def _maybe_paramspec_concatenate_value(arg: Value, ctx: Context) -> Value | None:
    if (
        isinstance(arg, PartialValue)
        and arg.operation is PartialValueOperation.SUBSCRIPT
        and isinstance(arg.root, KnownValue)
        and is_typing_name(arg.root.val, "Concatenate")
    ):
        concatenate_members = [_type_from_value(member, ctx) for member in arg.members]
        return _paramspec_value_from_concatenate_members(concatenate_members, ctx)
    if isinstance(arg, KnownValue) and is_typing_name(
        get_origin(arg.val), "Concatenate"
    ):
        concatenate_members = [
            _type_from_runtime(member, ctx) for member in get_args(arg.val)
        ]
        return _paramspec_value_from_concatenate_members(concatenate_members, ctx)
    return None


def _normalize_paramspec_generic_arg_in_context(
    arg: Value, *, allow_flat_form: bool, ctx: Context
) -> Value:
    if isinstance(arg, KnownValue):
        arg = replace_known_sequence_value(arg)
    concatenate_value = _maybe_paramspec_concatenate_value(arg, ctx)
    if concatenate_value is not None:
        return concatenate_value
    if isinstance(arg, InputSigValue):
        return arg
    if isinstance(arg, SequenceValue) and arg.typ in (list, tuple):
        return arg
    if arg == KnownValue(Ellipsis):
        return AnyValue(AnySource.ellipsis_callable)
    if isinstance(arg, AnyValue):
        return arg
    if isinstance(arg, KnownValue) and is_instance_of_typing_name(arg.val, "ParamSpec"):
        return InputSigValue(ParamSpecParam(arg.val))
    if allow_flat_form:
        return SequenceValue(tuple, [(False, arg)])
    ctx.show_error(
        "ParamSpec specialization must use list form, Concatenate[..., P], P, or ...",
        error_code=ErrorCode.invalid_annotation,
    )
    return AnyValue(AnySource.error)


def _normalize_paramspec_generic_args_in_context(
    type_params: Sequence[TypeParam], args: Sequence[Value], ctx: Context
) -> list[Value]:
    if len(type_params) == 1 and isinstance(type_params[0], ParamSpecParam):
        if len(args) == 1:
            return [
                _normalize_paramspec_generic_arg_in_context(
                    args[0], allow_flat_form=True, ctx=ctx
                )
            ]
        return [SequenceValue(tuple, [(False, arg) for arg in args])]
    if len(type_params) == len(args):
        return [
            (
                _normalize_paramspec_generic_arg_in_context(
                    arg, allow_flat_form=False, ctx=ctx
                )
                if isinstance(type_param, ParamSpecParam)
                else arg
            )
            for type_param, arg in zip(type_params, args)
        ]
    return list(args)


def _ensure_annotation_context(
    ctx: Context | CanAssignContext, node: ast.AST | None
) -> Context:
    if isinstance(ctx, Context):
        return ctx
    return _DefaultContext(ctx, node)


def _normalize_paramspec_generic_arg(
    arg: Value,
    *,
    allow_flat_form: bool,
    ctx: Context | CanAssignContext,
    node: ast.AST | None = None,
) -> Value:
    return _normalize_paramspec_generic_arg_in_context(
        arg, allow_flat_form=allow_flat_form, ctx=_ensure_annotation_context(ctx, node)
    )


def _normalize_paramspec_generic_args(
    type_params: Sequence[TypeParam],
    args: Sequence[Value],
    ctx: Context | CanAssignContext,
    node: ast.AST | None = None,
) -> list[Value]:
    return _normalize_paramspec_generic_args_in_context(
        type_params, args, ctx=_ensure_annotation_context(ctx, node)
    )


def _normalize_generic_unpack_members(
    members: Sequence[Value], ctx: Context
) -> list[tuple[bool, Value]] | None:
    normalized_members: list[tuple[bool, Value]] = []
    saw_unpack = False
    for member in members:
        if (
            isinstance(member, KnownValue)
            and isinstance(member.val, GenericAlias)
            and getattr(member.val, "__unpacked__", False)
        ):
            saw_unpack = True
            unpacked = _type_from_runtime(member.val, ctx)
            unpacked_members = _unpack_value(unpacked)
            if unpacked_members is None:
                ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                normalized_members.append((False, AnyValue(AnySource.error)))
                continue
            normalized_members.extend(unpacked_members)
            continue
        if _is_unpack_annotation_member(member):
            saw_unpack = True
            expr = _annotation_expr_from_value(member, ctx)
            unpacked, qualifiers = expr.unqualify({Qualifier.Unpack})
            if Qualifier.Unpack not in qualifiers:
                normalized_members.append((False, _type_from_value(unpacked, ctx)))
                continue
            unpacked_members = _unpack_value(unpacked)
            if unpacked_members is None:
                ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                normalized_members.append((False, AnyValue(AnySource.error)))
                continue
            for is_many, unpacked_member in unpacked_members:
                normalized_members.append(
                    (is_many, _type_from_alias_argument_value(unpacked_member, ctx))
                )
            continue
        normalized_members.append((False, _type_from_value(member, ctx)))
    if not saw_unpack:
        return None
    return normalized_members


def _pack_typevartuple_args_from_unpack_members(
    type_params: Sequence[TypeParam], members: Sequence[Value], ctx: Context
) -> list[Value] | None:
    if not any(_is_unpack_annotation_member(member) for member in members):
        return None
    normalized_members = _normalize_generic_unpack_members(members, ctx)
    if normalized_members is None or not any(
        is_many for is_many, _ in normalized_members
    ):
        return None
    if (
        len(normalized_members) == 1
        and normalized_members[0][0]
        and isinstance(normalized_members[0][1], TypeVarTupleValue)
    ):
        return None

    variadic_indexes = [
        i
        for i, type_param in enumerate(type_params)
        if isinstance(type_param, TypeVarTupleParam)
    ]
    if len(variadic_indexes) != 1:
        return None

    variadic_index = variadic_indexes[0]
    if len(normalized_members) == 1 and normalized_members[0][0]:
        repeated_member = normalized_members[0][1]
        return [
            (
                SequenceValue(tuple, [(True, repeated_member)])
                if i == variadic_index
                else repeated_member
            )
            for i in range(len(type_params))
        ]
    minimum_args = len(type_params) - 1
    if len(normalized_members) < minimum_args:
        return None

    suffix_count = len(type_params) - variadic_index - 1
    variadic_end = len(normalized_members) - suffix_count
    if variadic_end < variadic_index:
        return None

    prefix_members = normalized_members[:variadic_index]
    suffix_members = normalized_members[variadic_end:]
    if any(is_many for is_many, _ in prefix_members) or any(
        is_many for is_many, _ in suffix_members
    ):
        return None

    packed: list[Value] = []
    for i in range(len(type_params)):
        if i < variadic_index:
            packed.append(prefix_members[i][1])
        elif i == variadic_index:
            packed.append(
                SequenceValue(tuple, normalized_members[variadic_index:variadic_end])
            )
        else:
            suffix_index = i - variadic_index - 1
            packed.append(suffix_members[suffix_index][1])
    return packed


def _pack_typevartuple_runtime_args(
    type_params: Sequence[TypeParam], args: Sequence[object], ctx: Context
) -> list[Value] | None:
    if not any(_is_unpack_runtime_arg(arg) for arg in args):
        return None
    normalized_members: list[tuple[bool, Value]] = []
    for arg in args:
        if _is_unpack_runtime_arg(arg):
            expr = _annotation_expr_from_runtime(arg, ctx)
            unpacked, qualifiers = expr.unqualify({Qualifier.Unpack})
            if Qualifier.Unpack not in qualifiers:
                normalized_members.append((False, unpacked))
                continue
            unpacked_members = _unpack_value(unpacked)
            if unpacked_members is None:
                ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                normalized_members.append((False, AnyValue(AnySource.error)))
                continue
            normalized_members.extend(unpacked_members)
            continue
        normalized_members.append((False, _type_from_runtime(arg, ctx)))
    if not any(is_many for is_many, _ in normalized_members):
        return None
    if (
        len(normalized_members) == 1
        and normalized_members[0][0]
        and isinstance(normalized_members[0][1], TypeVarTupleValue)
    ):
        return None

    variadic_indexes = [
        i
        for i, type_param in enumerate(type_params)
        if isinstance(type_param, TypeVarTupleParam)
    ]
    if len(variadic_indexes) != 1:
        return None

    variadic_index = variadic_indexes[0]
    minimum_args = len(type_params) - 1
    if len(normalized_members) < minimum_args:
        return None

    suffix_count = len(type_params) - variadic_index - 1
    variadic_end = len(normalized_members) - suffix_count
    if variadic_end < variadic_index:
        return None

    prefix_members = normalized_members[:variadic_index]
    suffix_members = normalized_members[variadic_end:]
    if any(is_many for is_many, _ in prefix_members) or any(
        is_many for is_many, _ in suffix_members
    ):
        return None

    packed: list[Value] = []
    for i in range(len(type_params)):
        if i < variadic_index:
            packed.append(prefix_members[i][1])
        elif i == variadic_index:
            packed.append(
                SequenceValue(tuple, normalized_members[variadic_index:variadic_end])
            )
        else:
            suffix_index = i - variadic_index - 1
            packed.append(suffix_members[suffix_index][1])
    return packed


def _validate_type_alias_arg_values(
    type_params: Sequence[TypeParam], args_vals: Sequence[Value], ctx: Context
) -> list[Value]:
    normalized_args = list(args_vals)
    matched_args = _match_type_alias_arg_values(type_params, args_vals)
    if matched_args is None:
        ctx.show_error(
            f"Expected {len(type_params)} type arguments for type alias,"
            f" got {len(args_vals)}"
        )
        return normalized_args
    for i, (type_param, arg) in enumerate(matched_args):
        if isinstance(type_param, ParamSpecParam):
            normalized = _normalize_paramspec_generic_arg(
                arg, allow_flat_form=len(type_params) == 1, ctx=ctx
            )
            if len(type_params) == len(args_vals):
                normalized_args[i] = normalized
            continue
        if not isinstance(type_param, TypeVarParam):
            continue
        if type_param.bound is not None and not _is_alias_arg_compatible_with_bound(
            type_param.bound, arg, ctx
        ):
            ctx.show_error(f"Type argument {arg} is not compatible with {type_param}")
        elif type_param.constraints and not _is_alias_arg_compatible_with_constraints(
            type_param.constraints, arg, ctx
        ):
            constraint_list = ", ".join(
                str(constraint) for constraint in type_param.constraints
            )
            ctx.show_error(
                f"Type argument {arg} is not compatible with constraints ({constraint_list})"
            )
    return normalized_args


def _paramspec_value_from_concatenate_members(
    concatenate_members: Sequence[Value], ctx: Context
) -> Value:
    if not concatenate_members:
        ctx.show_error("Concatenate[] must have at least one type argument")
        return AnyValue(AnySource.error)
    tail = concatenate_members[-1]
    if isinstance(tail, InputSigValue):
        tail_annotation = tail
    else:
        ctx.show_error(f"Last argument to Concatenate must be a ParamSpec, got {tail}")
        return AnyValue(AnySource.error)
    annotations = [*concatenate_members[:-1], tail_annotation]
    params = [
        SigParameter(
            f"@{i}",
            kind=(
                ParameterKind.PARAM_SPEC
                if i == len(annotations) - 1
                else ParameterKind.POSITIONAL_ONLY
            ),
            annotation=annotation,
        )
        for i, annotation in enumerate(annotations)
    ]
    return InputSigValue(
        FullSignature(Signature.make(params, AnyValue(AnySource.generic_argument)))
    )


def _callable_args_from_runtime(
    arg_types: Any, label: str, ctx: Context
) -> Sequence[SigParameter]:
    if arg_types is Ellipsis or arg_types == [Ellipsis]:
        return [ELLIPSIS_PARAM]
    elif type(arg_types) in (tuple, list):
        if len(arg_types) == 1:
            (arg,) = arg_types
            if arg is Ellipsis:
                return [ELLIPSIS_PARAM]
            elif is_typing_name(get_origin(arg), "Concatenate"):
                return _args_from_concatenate(arg, ctx)
            elif is_instance_of_typing_name(arg, "ParamSpec"):
                param_spec = InputSigValue(ParamSpecParam(arg))
                param = SigParameter(
                    "__P", kind=ParameterKind.PARAM_SPEC, annotation=param_spec
                )
                return [param]
        types: list[Value] = []
        for arg in arg_types:
            if _is_unpack_runtime_arg(arg):
                expr = _annotation_expr_from_runtime(arg, ctx)
                unpacked, qualifiers = expr.unqualify({Qualifier.Unpack})
                if Qualifier.Unpack not in qualifiers:
                    types.append(unpacked)
                    continue
                unpacked_members = _unpack_value(unpacked)
                if unpacked_members is None:
                    ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                    types.append(AnyValue(AnySource.error))
                    continue
                for is_many, member in unpacked_members:
                    if is_many:
                        # Callable argument lists support unpacked TypeVarTuple
                        # placeholders, but not generic unbounded tuples.
                        if isinstance(member, TypeVarTupleValue):
                            types.append(member)
                        else:
                            ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                            types.append(AnyValue(AnySource.error))
                    else:
                        types.append(member)
                continue
            types.append(_type_from_runtime(arg, ctx))
        params = [
            SigParameter(
                f"@{i}",
                kind=(
                    ParameterKind.PARAM_SPEC
                    if isinstance(typ, InputSigValue)
                    else ParameterKind.POSITIONAL_ONLY
                ),
                annotation=typ,
            )
            for i, typ in enumerate(types)
        ]
        return params
    elif is_instance_of_typing_name(arg_types, "ParamSpec"):
        param_spec = InputSigValue(ParamSpecParam(arg_types))
        param = SigParameter(
            "__P", kind=ParameterKind.PARAM_SPEC, annotation=param_spec
        )
        return [param]
    elif is_typing_name(get_origin(arg_types), "Concatenate"):
        return _args_from_concatenate(arg_types, ctx)
    else:
        ctx.show_error(f"Invalid arguments to {label}: {arg_types!r}")
        return [ELLIPSIS_PARAM]


def _args_from_concatenate(concatenate: Any, ctx: Context) -> Sequence[SigParameter]:
    types = [_type_from_runtime(arg, ctx) for arg in concatenate.__args__]
    params = [
        SigParameter(
            f"@{i}",
            kind=(
                ParameterKind.PARAM_SPEC
                if i == len(types) - 1
                else ParameterKind.POSITIONAL_ONLY
            ),
            annotation=annotation,
        )
        for i, annotation in enumerate(types)
    ]
    return params


def _get_typeddict_value(
    value: Value,
    ctx: Context,
    key: str,
    required_keys: Container[str] | None,
    total: bool,
    readonly_keys: Container[str] | None,
) -> TypedDictEntry:
    ann_expr = _annotation_expr_from_runtime(value, ctx)
    val, qualifiers = ann_expr.unqualify(
        {Qualifier.ReadOnly, Qualifier.Required, Qualifier.NotRequired},
        mutually_exclusive_qualifiers=((Qualifier.Required, Qualifier.NotRequired),),
    )
    if required_keys is None:
        required = total
    else:
        required = key in required_keys
    if readonly_keys is None:
        readonly = False
    else:
        readonly = key in readonly_keys
    if Qualifier.ReadOnly in qualifiers:
        readonly = True
    if Qualifier.Required in qualifiers:
        required = True
    if Qualifier.NotRequired in qualifiers:
        required = False
    return TypedDictEntry(required=required, readonly=readonly, typ=val)


def _eval_forward_ref(val: str, ctx: Context) -> AnnotationExpr:
    parse_source = val
    if "\n" in val or "\r" in val:
        # Per the typing spec, multiline string annotations are parsed as if
        # they were parenthesized.
        parse_source = f"({val})"
    try:
        tree = ast.parse(parse_source, mode="eval")
    except SyntaxError:
        ctx.show_error(f"Syntax error in type annotation: {val}")
        return AnnotationExpr(ctx, AnyValue(AnySource.error))

    node = ctx.get_error_node()
    if node is not None and hasattr(node, "lineno") and node.lineno > 1:
        ast.increment_lineno(tree, node.lineno - 1)
    return _annotation_expr_from_ast(tree.body, ctx)


def _annotation_expr_from_value(value: Value, ctx: Context) -> AnnotationExpr:
    if isinstance(value, KnownValue):
        return _annotation_expr_from_runtime(value.val, ctx)
    elif isinstance(value, PartialValue):
        if value.operation is PartialValueOperation.SUBSCRIPT:
            return _annotation_expr_from_subscripted_value(
                value.root, value.node, value.members, ctx
            )
        if value.operation is PartialValueOperation.UNPACK:
            inner = _annotation_expr_from_value(value.root, ctx)
            return inner.add_qualifier(Qualifier.Unpack, value.node)
        return AnnotationExpr(ctx, _type_from_value(value, ctx))
    else:
        return AnnotationExpr(ctx, _type_from_value(value, ctx))


def _is_self_annotation_value(value: Value) -> bool:
    return isinstance(value, KnownValue) and is_typing_name(value.val, "Self")


def _type_from_value(value: Value, ctx: Context) -> Value:
    if isinstance(value, KnownValue):
        return _type_from_runtime(value.val, ctx)
    elif isinstance(value, TypedDictValue):
        return value
    elif isinstance(value, SyntheticClassObjectValue):
        return value.class_type
    elif isinstance(value, (TypeVarValue, TypeVarTupleValue, TypeAliasValue)):
        return value
    elif isinstance(value, MultiValuedValue):
        return unite_values(*[_type_from_value(val, ctx) for val in value.vals])
    elif isinstance(value, TypeFormValue):
        return value
    elif isinstance(value, AnnotatedValue):
        return _type_from_value(value.value, ctx)
    elif isinstance(value, PartialValue):
        if value.operation is PartialValueOperation.SUBSCRIPT:
            return _type_from_subscripted_value(value.root, value.members, ctx)
        if value.operation is PartialValueOperation.BITOR:
            return _type_from_bitor_value(value.root, value.members, ctx)
        return value.get_fallback_value()
    elif isinstance(value, PartialCallValue):
        type_param = make_type_param_from_value(value, ctx=ctx)
        if isinstance(type_param, TypeVarParam):
            return TypeVarValue(type_param)
        if isinstance(type_param, TypeVarTupleParam):
            return TypeVarTupleValue(type_param)
        if isinstance(type_param, ParamSpecParam):
            return InputSigValue(type_param)
        ctx.show_error(f"Unrecognized annotation {value}")
        return AnyValue(AnySource.error)
    elif isinstance(value, AnyValue):
        return value
    elif isinstance(value, InputSigValue):
        return value
    else:
        ctx.show_error(f"Unrecognized annotation {value}")
        return AnyValue(AnySource.error)


def _type_from_bitor_value(
    root: Value, members: Sequence[Value], ctx: Context
) -> Value:
    if not members:
        ctx.show_error("Invalid type expression using |")
        return AnyValue(AnySource.error)
    values = [
        _type_from_value(root, ctx),
        *[_type_from_value(member, ctx) for member in members],
    ]
    return unite_values(*values)


def _require_exact_argument_count(
    args: Sequence[object], expected: int, construct: str, ctx: Context
) -> bool:
    if len(args) != expected:
        target = f"{construct}[]"
        if expected == 1:
            message = f"{target} requires a single argument"
        else:
            message = (
                f"{target} requires exactly {_argument_count_word(expected)} arguments"
            )
        ctx.show_error(message)
        return False
    return True


def _argument_count_word(count: int) -> str:
    if count == 1:
        return "one"
    if count == 2:
        return "two"
    return str(count)


def _require_min_argument_count(
    args: Sequence[object], minimum: int, construct: str, ctx: Context
) -> bool:
    if len(args) < minimum:
        target = f"{construct}[]"
        ctx.show_error(
            f"{target} requires at least {_argument_count_word(minimum)} arguments"
        )
        return False
    return True


def _annotation_expr_from_subscripted_value(
    root: Value, node: ast.AST, members: Sequence[Value], ctx: Context
) -> AnnotationExpr:
    root_val: object | None = None
    if isinstance(root, KnownValue):
        root_val = root.val
    elif isinstance(root, TypedValue):
        root_val = root.typ
    elif isinstance(root, GenericValue):
        root_val = root.typ

    if root_val is not None:
        if is_typing_name(root_val, "Annotated"):
            if not _require_min_argument_count(members, 2, "Annotated", ctx):
                return AnnotationExpr(ctx, AnyValue(AnySource.error))
            origin, *metadata = members
            origin_expr = _annotation_expr_from_value(origin, ctx)
            return origin_expr.add_metadata(translate_annotated_metadata(metadata, ctx))
        for qualifier in (
            Qualifier.Required,
            Qualifier.NotRequired,
            Qualifier.ReadOnly,
            Qualifier.Final,
            Qualifier.ClassVar,
            Qualifier.InitVar,
            Qualifier.Unpack,
        ):
            is_qualifier = is_typing_name(root_val, qualifier.name)
            if not is_qualifier and qualifier is Qualifier.InitVar:
                is_qualifier = root_val is InitVar
            if is_qualifier:
                if not _require_exact_argument_count(members, 1, qualifier.name, ctx):
                    return AnnotationExpr(ctx, AnyValue(AnySource.error))
                inner = _annotation_expr_from_value(members[0], ctx)
                return inner.add_qualifier(qualifier, node)
    val = _type_from_subscripted_value(root, members, ctx)
    return AnnotationExpr(ctx, val)


def _type_from_subscripted_value(
    root: Value, members: Sequence[Value], ctx: Context
) -> Value:
    if isinstance(root, GenericValue):
        root_type_param_list: list[TypeParam] = []
        for arg in root.args:
            if isinstance(arg, TypeVarValue):
                root_type_param_list.append(arg.typevar_param)
            elif isinstance(arg, TypeVarTupleValue):
                root_type_param_list.append(arg.typevar_tuple_param)
            elif isinstance(arg, InputSigValue) and isinstance(
                arg.input_sig, ParamSpecParam
            ):
                root_type_param_list.append(arg.input_sig)
            else:
                root_type_param_list = []
                break
        root_type_params: tuple[TypeParam, ...] | None = (
            tuple(root_type_param_list)
            if len(root_type_param_list) == len(root.args)
            else None
        )
        typed_members: list[Value] | None = None
        packed_variadic_members = (
            _pack_typevartuple_args_from_unpack_members(root_type_params, members, ctx)
            if root_type_params is not None
            else None
        )
        if packed_variadic_members is not None:
            typed_members = packed_variadic_members
        elif (
            not members
            and root_type_params is not None
            and len(root_type_params) == 1
            and isinstance(root_type_params[0], TypeVarTupleParam)
        ):
            typed_members = [SequenceValue(tuple, [])]
        elif root_type_params is not None and len(root_type_params) == len(members):
            typed_members = [
                _type_from_value_type_alias_arg(member, root_arg, ctx)
                for member, root_arg in zip(members, root_type_params)
            ]
        if typed_members is not None:
            if root_type_params is not None:
                typed_members = _normalize_paramspec_generic_args(
                    root_type_params, typed_members, ctx
                )
            if is_typing_name(root.typ, "Generic") or is_typing_name(
                root.typ, "Protocol"
            ):
                if root_type_params is not None:
                    alias = TypeAlias(
                        evaluator=lambda root=root: root,
                        evaluate_type_params=lambda generic_type_params=tuple(
                            root_type_params
                        ): generic_type_params,
                    )
                    return TypeAliasValue(
                        "<generic_alias>", "typing", alias, tuple(typed_members)
                    )
            return GenericValue(root.typ, typed_members)
        if (
            root_type_params is not None
            and len(root_type_params) == 1
            and isinstance(root_type_params[0], ParamSpecParam)
        ):
            typed_members = [_type_from_value(member, ctx) for member in members]
            typed_members = _normalize_paramspec_generic_args(
                root_type_params, typed_members, ctx
            )
            return GenericValue(root.typ, typed_members)
    if isinstance(root, PartialValue):
        runtime_alias = _runtime_type_alias_from_partial_value(root, ctx)
        if runtime_alias is not None:
            return _type_from_subscripted_value(runtime_alias, members, ctx)
        root_type = _type_from_value(root, ctx)
        return _type_from_subscripted_value(root_type, members, ctx)
    elif isinstance(root, MultiValuedValue):
        return unite_values(
            *[
                _type_from_subscripted_value(subval, members, ctx)
                for subval in root.vals
            ]
        )
    if isinstance(root, SyntheticClassObjectValue) and isinstance(
        root.class_type, TypedValue
    ):
        synthetic_typ = root.class_type.typ
        can_assign_ctx = _get_can_assign_context(ctx)
        synthetic_type_params: Sequence[TypeParam] = ()
        if can_assign_ctx is not None:
            synthetic_type_params = can_assign_ctx.get_type_parameters(synthetic_typ)
        packed_variadic_members = _pack_typevartuple_args_from_unpack_members(
            synthetic_type_params, members, ctx
        )
        if packed_variadic_members is not None:
            typed_members = packed_variadic_members
        elif (
            not members
            and len(synthetic_type_params) == 1
            and isinstance(synthetic_type_params[0], TypeVarTupleParam)
        ):
            typed_members = [SequenceValue(tuple, [])]
        elif len(synthetic_type_params) == len(members):
            typed_members = [
                _type_from_value_type_alias_arg(elt, type_param, ctx)
                for elt, type_param in zip(members, synthetic_type_params)
            ]
        else:
            typed_members = [_type_from_value(elt, ctx) for elt in members]
        typed_members = _normalize_paramspec_generic_args(
            synthetic_type_params, typed_members, ctx
        )
        return GenericValue(synthetic_typ, typed_members)

    if isinstance(root, TypedValue) and isinstance(root.typ, str):
        synthetic_typ = root.typ
        can_assign_ctx = _get_can_assign_context(ctx)
        synthetic_type_params_for_str: Sequence[TypeParam] = ()
        if can_assign_ctx is not None:
            synthetic_type_params_for_str = can_assign_ctx.get_type_parameters(
                synthetic_typ
            )
        packed_variadic_members = _pack_typevartuple_args_from_unpack_members(
            synthetic_type_params_for_str, members, ctx
        )
        if packed_variadic_members is not None:
            typed_members = packed_variadic_members
        elif (
            not members
            and len(synthetic_type_params_for_str) == 1
            and isinstance(synthetic_type_params_for_str[0], TypeVarTupleParam)
        ):
            typed_members = [SequenceValue(tuple, [])]
        elif len(synthetic_type_params_for_str) == len(members):
            typed_members = [
                _type_from_value_type_alias_arg(elt, type_param, ctx)
                for elt, type_param in zip(members, synthetic_type_params_for_str)
            ]
        else:
            typed_members = [_type_from_value(elt, ctx) for elt in members]
        typed_members = _normalize_paramspec_generic_args(
            synthetic_type_params_for_str, typed_members, ctx
        )
        return GenericValue(synthetic_typ, typed_members)
    if isinstance(root, TypeAliasValue):
        type_params = tuple(root.alias.get_type_params())
        type_arguments_are_packed = False
        if any(_is_unpack_annotation_member(member) for member in members):
            normalized_unpack_members = _normalize_generic_unpack_members(members, ctx)
        else:
            normalized_unpack_members = None
        saw_unpack = normalized_unpack_members is not None
        has_unbounded_unpack = (
            saw_unpack
            and normalized_unpack_members is not None
            and any(is_many for is_many, _ in normalized_unpack_members)
        )
        packed_variadic_members = _pack_typevartuple_args_from_unpack_members(
            type_params, members, ctx
        )
        if packed_variadic_members is not None:
            args_vals = packed_variadic_members
            type_arguments_are_packed = True
        elif (
            saw_unpack
            and normalized_unpack_members is not None
            and not has_unbounded_unpack
        ):
            unpacked_members = [member for _, member in normalized_unpack_members]
            if len(unpacked_members) == len(type_params):
                args_vals = [
                    _type_from_value_type_alias_arg(member, type_param, ctx)
                    for member, type_param in zip(unpacked_members, type_params)
                ]
            else:
                args_vals = [
                    _type_from_alias_argument_value(member, ctx)
                    for member in unpacked_members
                ]
        elif len(members) == len(type_params):
            args_vals = [
                _type_from_value_type_alias_arg(member, type_param, ctx)
                for member, type_param in zip(members, type_params)
            ]
        else:
            args_vals = [
                _type_from_alias_argument_value(member, ctx) for member in members
            ]
        if has_unbounded_unpack and packed_variadic_members is None:
            ctx.show_error("Unpacked TypeVarTuple cannot specialize this type alias")
        args_vals = _validate_type_alias_arg_values(type_params, args_vals, ctx)
        alias_value = TypeAliasValue(
            root.name,
            root.module,
            root.alias,
            tuple(args_vals),
            runtime_allows_value_call=root.runtime_allows_value_call,
            uses_type_alias_object_semantics=root.uses_type_alias_object_semantics,
            is_specialized=True,
            type_arguments_are_packed=type_arguments_are_packed,
        )
        if root.runtime_allows_value_call:
            # Explicit `TypeAlias` declarations should behave like expanded types in
            # annotation contexts so generic solving can bind function/class type
            # variables precisely even when runtime module loading fails.
            return alias_value.get_value()
        return alias_value

    assert isinstance(root, Value)
    if not isinstance(root, KnownValue):
        if root != AnyValue(AnySource.error):
            ctx.show_error(f"Cannot resolve subscripted annotation: {root}")
        return AnyValue(AnySource.error)
    runtime_alias = _runtime_type_alias_from_runtime_value(root.val, ctx)
    if runtime_alias is not None:
        return _type_from_subscripted_value(runtime_alias, members, ctx)
    root = root.val
    if is_instance_of_typing_name(root, "TypeAliasType"):
        alias_object = root
        runtime_type_params = tuple(alias_object.__type_params__)
        type_params = tuple(make_type_param(tp, ctx=ctx) for tp in runtime_type_params)
        if len(members) == len(type_params):
            args_vals = [
                _type_from_value_type_alias_arg(member, type_param, ctx)
                for member, type_param in zip(members, type_params)
            ]
        else:
            args_vals = [_type_from_value(member, ctx) for member in members]
        args_vals = _validate_type_alias_arg_values(type_params, args_vals, ctx)
        alias = ctx.get_type_alias(
            root,
            lambda: type_from_runtime(alias_object.__value__, ctx=ctx),
            lambda: tuple(
                make_type_param(tv, ctx) for tv in alias_object.__type_params__
            ),
        )
        return TypeAliasValue(
            alias_object.__name__,
            alias_object.__module__,
            alias,
            tuple(args_vals),
            is_specialized=True,
        )
    if root is typing.Union:
        return unite_values(*[_type_from_value(elt, ctx) for elt in members])
    elif is_typing_name(root, "Literal"):
        if not all(isinstance(elt, KnownValue) for elt in members):
            ctx.show_error(
                f"Arguments to Literal[] must be literals, not {members}",
                error_code=ErrorCode.invalid_literal,
            )
            return AnyValue(AnySource.error)
        known_members = [elt for elt in members if isinstance(elt, KnownValue)]
        invalid_members = [
            elt for elt in known_members if not _is_valid_pep586_literal_value(elt.val)
        ]
        if invalid_members:
            invalid_values = ", ".join(repr(elt.val) for elt in invalid_members)
            ctx.show_error(
                "Arguments to Literal[] must be None, bool, int, str, bytes, or enum"
                f" members; got {invalid_values}",
                error_code=ErrorCode.invalid_literal,
            )
        return unite_values(*members)
    elif _is_tuple(root):
        if (
            len(members) == 2
            and members[1] == KnownValue(Ellipsis)
            and not _is_unpack_annotation_member(members[0])
        ):
            return GenericValue(tuple, [_type_from_value(members[0], ctx)])
        elif len(members) == 1 and members[0] == KnownValue(()):
            return SequenceValue(tuple, [])
        else:
            if any(member == KnownValue(Ellipsis) for member in members):
                ctx.show_error("Ellipsis can be used only in tuple[T, ...]")
            exprs = [_annotation_expr_from_value(arg, ctx) for arg in members]
            return _make_sequence_value(tuple, exprs, ctx)
    elif root is typing.Optional:
        if not _require_exact_argument_count(members, 1, "Optional", ctx):
            return AnyValue(AnySource.error)
        return unite_values(KnownValue(None), _type_from_value(members[0], ctx))
    elif root is typing.Type or root is type:
        if not _require_exact_argument_count(members, 1, "Type", ctx):
            return AnyValue(AnySource.error)
        argument = _type_from_value(members[0], ctx)
        return SubclassValue.make(argument)
    elif is_typing_name(root, "Annotated"):
        if not _require_min_argument_count(members, 2, "Annotated", ctx):
            return AnyValue(AnySource.error)
        origin, *metadata = members
        return _make_annotated(_type_from_value(origin, ctx), metadata, ctx)
    elif is_typing_name(root, "TypeGuard"):
        if not _require_exact_argument_count(members, 1, "TypeGuard", ctx):
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "TypeIs"):
        if not _require_exact_argument_count(members, 1, "TypeIs", ctx):
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeIsExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "TypeForm"):
        if not _require_exact_argument_count(members, 1, "TypeForm", ctx):
            return AnyValue(AnySource.error)
        return TypeFormValue(_type_from_value(members[0], ctx))
    elif is_typing_name(root, "Required"):
        ctx.show_error("Required[] used in unsupported context")
        return AnyValue(AnySource.error)
    elif is_typing_name(root, "NotRequired"):
        ctx.show_error("NotRequired[] used in unsupported context")
        return AnyValue(AnySource.error)
    elif is_typing_name(root, "ReadOnly"):
        ctx.show_error("ReadOnly[] used in unsupported context")
        return AnyValue(AnySource.error)
    elif is_typing_name(root, "Final"):
        if not _require_exact_argument_count(members, 1, "Final", ctx):
            return AnyValue(AnySource.error)
        return _type_from_value(members[0], ctx)
    elif is_typing_name(root, "Unpack"):
        ctx.show_error("Unpack[] used in unsupported context")
        return AnyValue(AnySource.error)
    elif root is Callable or root is typing.Callable:
        if not _require_exact_argument_count(members, 2, "Callable", ctx):
            return AnyValue(AnySource.error)
        args, return_value = members
        return _make_callable_from_value(args, return_value, ctx)
    elif root is AsynqCallable:
        if not _require_exact_argument_count(members, 2, "AsynqCallable", ctx):
            return AnyValue(AnySource.error)
        args, return_value = members
        return _make_callable_from_value(args, return_value, ctx, is_asynq=True)
    elif root is Intersection:
        if not members:
            ctx.show_error("Intersection[] is missing arguments")
            return AnyValue(AnySource.error)
        return IntersectionValue(
            tuple(_type_from_value(subval, ctx) for subval in members)
        )
    elif root is Overlapping:
        if not _require_exact_argument_count(members, 1, "Overlapping", ctx):
            return AnyValue(AnySource.error)
        return OverlappingValue(_type_from_value(members[0], ctx))
    elif isinstance(root, type):
        type_params = _get_generic_type_parameters_for_annotation(root, ctx)
        packed_variadic_members = _pack_typevartuple_args_from_unpack_members(
            type_params, members, ctx
        )
        if packed_variadic_members is not None:
            typed_members = packed_variadic_members
        elif (
            not members
            and len(type_params) == 1
            and isinstance(type_params[0], TypeVarTupleParam)
        ):
            typed_members = [SequenceValue(tuple, [])]
        elif len(type_params) == len(members):
            typed_members = [
                _type_from_value_type_alias_arg(elt, type_param, ctx)
                for elt, type_param in zip(members, type_params)
            ]
        else:
            typed_members = [_type_from_value(elt, ctx) for elt in members]
        typed_members = _normalize_paramspec_generic_args(
            type_params, typed_members, ctx
        )
        return GenericValue(root, typed_members)
    elif is_typing_name(root, "ClassVar"):
        ctx.show_error("ClassVar[] used in unsupported context")
        return AnyValue(AnySource.error)
    else:
        origin = get_origin(root)
        if isinstance(origin, type):
            typed_members = [_type_from_value(elt, ctx) for elt in members]
            if is_typing_name(origin, "Generic") or is_typing_name(origin, "Protocol"):
                runtime_type_params = tuple(get_args(root))
                if len(runtime_type_params) == len(members) and all(
                    is_typevarlike(type_param) for type_param in runtime_type_params
                ):
                    typed_runtime_type_params = tuple(
                        make_type_param(type_param, ctx)
                        for type_param in runtime_type_params
                    )

                    def _evaluate_runtime_generic_alias(
                        origin: type = origin,
                        runtime_type_params: tuple[
                            TypeParam, ...
                        ] = typed_runtime_type_params,
                    ) -> Value:
                        runtime_generic_args: list[Value] = []
                        for type_param in runtime_type_params:
                            if isinstance(type_param, TypeVarParam):
                                runtime_generic_args.append(TypeVarValue(type_param))
                            elif isinstance(type_param, TypeVarTupleParam):
                                runtime_generic_args.append(
                                    TypeVarTupleValue(type_param)
                                )
                            else:
                                runtime_generic_args.append(InputSigValue(type_param))
                        return GenericValue(origin, runtime_generic_args)

                    def _evaluate_runtime_generic_alias_type_params(
                        runtime_type_params: tuple[
                            TypeParam, ...
                        ] = typed_runtime_type_params,
                    ) -> tuple[TypeParam, ...]:
                        return runtime_type_params

                    alias = TypeAlias(
                        evaluator=_evaluate_runtime_generic_alias,
                        evaluate_type_params=_evaluate_runtime_generic_alias_type_params,
                    )
                    return TypeAliasValue(
                        "<runtime_generic_alias>", "typing", alias, tuple(typed_members)
                    )
            return GenericValue(origin, typed_members)
        ctx.show_error(f"Unrecognized subscripted annotation: {root}")
        return AnyValue(AnySource.error)


def _maybe_get_extra(origin: type) -> type | str:
    # ContextManager is defined oddly and we lose the Protocol if we don't use
    # synthetic types.
    if any(origin is cls for cls in CONTEXT_MANAGER_TYPES):
        return "contextlib.AbstractContextManager"
    elif any(origin is cls for cls in ASYNC_CONTEXT_MANAGER_TYPES):
        return "contextlib.AbstractAsyncContextManager"
    else:
        return origin


def _type_alias_cache_key(key: object) -> object:
    """Return a hashable cache key for runtime type alias objects."""
    try:
        hash(key)
    except TypeError:
        if isinstance(key, list):
            return ("list", tuple(_type_alias_cache_key(item) for item in key))
        if isinstance(key, tuple):
            return ("tuple", tuple(_type_alias_cache_key(item) for item in key))
        origin = get_origin(key)
        if origin is not None:
            return (
                "origin_args",
                _type_alias_cache_key(origin),
                tuple(_type_alias_cache_key(arg) for arg in get_args(key)),
            )
        return ("object_id", type(key), id(key))
    else:
        return key


class _DefaultContext(Context):
    def __init__(
        self,
        visitor: CanAssignContext | None,
        node: ast.AST | None,
        globals: Mapping[str, object] | None = None,
        use_name_node_for_error: bool = False,
    ) -> None:
        super().__init__(can_assign_ctx=visitor)
        if isinstance(visitor, AnnotationVisitor):
            self.visitor = visitor
        self.node = node
        self.globals = globals
        self.use_name_node_for_error = use_name_node_for_error

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: ast.AST | None = None,
    ) -> None:
        if self.should_suppress_errors and not (
            self.should_allow_undefined_names and error_code is ErrorCode.undefined_name
        ):
            return
        if node is None:
            node = self.node
        if self.visitor is not None and node is not None:
            self.visitor.show_error(node, message, error_code)

    def get_error_node(self) -> ast.AST | None:
        return self.node

    def get_name(self, node: ast.Name) -> Value:
        if self.visitor is not None:
            if self.should_allow_undefined_names:
                val, defining_scope, _ = self.visitor.scopes.get_with_scope(
                    node.id, None, self.visitor.state, can_assign_ctx=self.visitor
                )
                if val is UNINITIALIZED_VALUE:
                    if self.visitor._is_collecting():
                        return AnyValue(AnySource.inference)
                    self.show_error(
                        f"Undefined name {node.id!r} used in annotation",
                        ErrorCode.undefined_name,
                        node=node,
                    )
                    return AnyValue(AnySource.error)
                if self.visitor.in_annotation and defining_scope is not None:
                    declared_type = defining_scope.get_declared_type(node.id)
                    if isinstance(declared_type, TypeAliasValue):
                        val = declared_type
                if _is_self_annotation_value(val):
                    self.maybe_show_invalid_self_annotation()
                return val
            val, _ = self.visitor.resolve_name(
                node,
                error_node=node if self.use_name_node_for_error else self.node,
                suppress_errors=(
                    self.should_suppress_errors or self.should_allow_undefined_names
                ),
            )
            if self.should_allow_undefined_names and val == AnyValue(AnySource.error):
                # Allow unresolved forward refs without suppressing other
                # annotation errors.
                val = AnyValue(AnySource.inference)
            if _is_self_annotation_value(val):
                self.maybe_show_invalid_self_annotation()
            return val
        elif self.globals is not None:
            if node.id in self.globals:
                val = KnownValue(self.globals[node.id])
                if _is_self_annotation_value(val):
                    self.maybe_show_invalid_self_annotation()
                return val
            elif hasattr(builtins, node.id):
                val = KnownValue(getattr(builtins, node.id))
                if _is_self_annotation_value(val):
                    self.maybe_show_invalid_self_annotation()
                return val
        if self.should_allow_undefined_names and not self.should_suppress_errors:
            self.show_error(
                f"Undefined name {node.id!r} used in annotation",
                ErrorCode.undefined_name,
                node=node,
            )
            return AnyValue(AnySource.error)
        if self.should_suppress_errors:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {node.id!r} used in annotation",
            ErrorCode.undefined_name,
            node=node,
        )
        return AnyValue(AnySource.error)

    def invalid_self_annotation_message(self, node: ast.AST) -> str | None:
        if self.visitor is None:
            return None
        return self.visitor.invalid_self_annotation_message(node)

    def get_type_alias(
        self,
        key: object,
        evaluator: typing.Callable[[], Value],
        evaluate_type_params: typing.Callable[[], Sequence[TypeParam]],
    ) -> TypeAlias:
        if self.visitor is not None:
            cache = self.visitor.get_type_alias_cache()
            if cache is not None:
                cache_key = _type_alias_cache_key(key)
                if cache_key in cache:
                    return cache[cache_key]
                alias = super().get_type_alias(key, evaluator, evaluate_type_params)
                cache[cache_key] = alias
                return alias
        return super().get_type_alias(key, evaluator, evaluate_type_params)


@dataclass
class _RuntimeAnnotationsContext(Context):
    owner: object
    node: ast.AST | None = None

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: ast.AST | None = None,
    ) -> None:
        if self.should_suppress_errors and not (
            self.should_allow_undefined_names and error_code is ErrorCode.undefined_name
        ):
            return
        error_node = node or self.node
        if self.visitor is not None and error_node is not None:
            self.visitor.show_error(error_node, message, error_code)

    def get_error_node(self) -> ast.AST | None:
        return self.node

    def get_name(self, node: ast.Name) -> Value:
        try:
            if isinstance(self.owner, types.ModuleType):
                globals_dict = self.owner.__dict__
            else:
                globals_dict = sys.modules[self.owner.__module__].__dict__
        except Exception:
            if self.visitor is None:
                return AnyValue(AnySource.error)
            value, _ = self.visitor.resolve_name(
                node, error_node=self.node, suppress_errors=self.should_suppress_errors
            )
        else:
            value = self.get_name_from_globals(node.id, globals_dict)
        if _is_self_annotation_value(value):
            self.maybe_show_invalid_self_annotation()
        return value

    def invalid_self_annotation_message(self, node: ast.AST) -> str | None:
        if self.visitor is None:
            return None
        return self.visitor.invalid_self_annotation_message(node)


@dataclass(frozen=True)
class DecoratorValue(Value):
    decorator: object
    args: tuple[Value, ...]


class _UnsupportedAnnotationExpression(Exception):
    pass


class _Visitor(ast.NodeVisitor):
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def generic_visit(self, node: ast.AST) -> Value:
        if isinstance(node, ast.expr):
            raise _UnsupportedAnnotationExpression
        raise NotImplementedError(f"no visitor implemented for {node!r}")

    def visit_Name(self, node: ast.Name) -> Value:
        return self.ctx.get_name(node)

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        value = self.visit(node.value)
        index = self.visit(node.slice)
        if isinstance(index, SequenceValue):
            members = index.get_member_sequence()
            if members is None:
                # TODO support unpacking here
                return AnyValue(AnySource.inference)
            members = tuple(members)
        else:
            members = (index,)
        return PartialValue(
            PartialValueOperation.SUBSCRIPT,
            value,
            node.value,
            members,
            _SUBSCRIPT_RUNTIME_TYPE,
        )

    def visit_Attribute(self, node: ast.Attribute) -> Value:
        root_value = self.visit(node.value)
        return self.ctx.get_attribute(root_value, node)

    def visit_Tuple(self, node: ast.Tuple) -> Value:
        elts = [(False, self.visit(elt)) for elt in node.elts]
        return SequenceValue(tuple, elts)

    def visit_List(self, node: ast.List) -> Value:
        elts = [(False, self.visit(elt)) for elt in node.elts]
        return SequenceValue(list, elts)

    def visit_Set(self, node: ast.Set) -> Value:
        elts = [(False, self.visit(elt)) for elt in node.elts]
        return SequenceValue(set, elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        kvpairs = []
        for key_node, value_node in zip(node.keys, node.values):
            if key_node is None:
                # Just skip ** unpacking in stubs for now.
                kvpairs.append(
                    KVPair(AnyValue(AnySource.inference), AnyValue(AnySource.inference))
                )
                continue
            kvpairs.append(KVPair(self.visit(key_node), self.visit(value_node)))
        return DictIncompleteValue(dict, kvpairs)

    def visit_Constant(self, node: ast.Constant) -> Value:
        return KnownValue(node.value)

    def visit_Expr(self, node: ast.Expr) -> Value:
        return self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Value:
        if isinstance(node.op, ast.BitOr):
            return PartialValue(
                PartialValueOperation.SUBSCRIPT,
                KnownValue(Union),
                node,
                (self.visit(node.left), self.visit(node.right)),
                _UNION_RUNTIME_TYPE,
            )
        raise _UnsupportedAnnotationExpression

    def visit_Starred(self, node: ast.Starred) -> Value:
        value = self.visit(node.value)
        return PartialValue(
            PartialValueOperation.UNPACK, value, node, (), _UNPACK_RUNTIME_TYPE
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value:
        # Only int and float negation on literals are supported.
        if isinstance(node.op, ast.USub):
            operand = self.visit(node.operand)
            if isinstance(operand, KnownValue) and isinstance(
                operand.val, (int, float)
            ):
                return KnownValue(-operand.val)
        raise _UnsupportedAnnotationExpression

    def visit_Call(self, node: ast.Call) -> Value:
        func = self.visit(node.func)
        if not isinstance(func, KnownValue):
            raise _UnsupportedAnnotationExpression
        if func.val == NewType:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            args = []
            kwargs = {}
            for arg_value in arg_values:
                if isinstance(arg_value, KnownValue):
                    args.append(arg_value.val)
                else:
                    raise _UnsupportedAnnotationExpression
            for name, kwarg_value in kwarg_values:
                if name is None:
                    if isinstance(kwarg_value, KnownValue) and isinstance(
                        kwarg_value.val, dict
                    ):
                        kwargs.update(kwarg_value.val)
                    else:
                        raise _UnsupportedAnnotationExpression
                else:
                    if isinstance(kwarg_value, KnownValue):
                        kwargs[name] = kwarg_value.val
                    else:
                        raise _UnsupportedAnnotationExpression
            return KnownValue(func.val(*args, **kwargs))
        elif is_typing_name(func.val, "TypeVar"):
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "TypeVar() requires at least one argument", node=node
                )
                return AnyValue(AnySource.error)
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error("TypeVar name must be a literal", node=node.args[0])
                return AnyValue(AnySource.error)

            def _typevar_arg_to_type(arg_value: Value) -> Value:
                # String bounds/constraints may contain forward refs to names that
                # are defined later in the file.
                allow_undefined_names = isinstance(
                    arg_value, KnownValue
                ) and isinstance(arg_value.val, str)
                return type_from_value(
                    arg_value, ctx=self.ctx, allow_undefined_names=allow_undefined_names
                )

            constraints = []
            for arg_value in arg_values[1:]:
                constraints.append(_typevar_arg_to_type(arg_value))
            bound = default = None
            covariant = False
            contravariant = False
            infer_variance = False
            for name, kwarg_value in kwarg_values:
                if name in ("covariant", "contravariant", "infer_variance"):
                    if not isinstance(kwarg_value, KnownValue) or not isinstance(
                        kwarg_value.val, bool
                    ):
                        self.ctx.show_error(
                            f"TypeVar kwarg {name} must be a bool literal", node=node
                        )
                        return AnyValue(AnySource.error)
                    if name == "covariant":
                        covariant = kwarg_value.val
                    elif name == "contravariant":
                        contravariant = kwarg_value.val
                    elif name == "infer_variance":
                        infer_variance = kwarg_value.val
                elif name == "bound":
                    bound = _typevar_arg_to_type(kwarg_value)
                elif name == "default":
                    default = _typevar_arg_to_type(kwarg_value)
                else:
                    self.ctx.show_error(f"Unrecognized TypeVar kwarg {name}", node=node)
                    return AnyValue(AnySource.error)
            try:
                kwargs = {"covariant": covariant, "contravariant": contravariant}
                if infer_variance:
                    kwargs_with_infer = {**kwargs, "infer_variance": True}
                    tv = typing.cast(
                        TypeVarType,
                        typing_extensions.TypeVar(name_val.val, **kwargs_with_infer),
                    )
                else:
                    tv = typing.cast(TypeVarType, TypeVar(name_val.val, **kwargs))
            except Exception as e:
                self.ctx.show_error(str(e), node=node)
                return AnyValue(AnySource.error)
            if covariant:
                variance = Variance.COVARIANT
            elif contravariant:
                variance = Variance.CONTRAVARIANT
            else:
                variance = Variance.INVARIANT
            return TypeVarValue(
                TypeVarParam(
                    tv,
                    bound=bound,
                    constraints=tuple(constraints),
                    default=default,
                    variance=variance,
                )
            )
        elif is_typing_name(func.val, "ParamSpec"):
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "ParamSpec() requires at least one argument", node=node
                )
                return AnyValue(AnySource.error)
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error(
                    "ParamSpec name must be a literal", node=node.args[0]
                )
                return AnyValue(AnySource.error)
            for name, _ in kwarg_values:
                # TODO support defaults
                self.ctx.show_error(f"Unrecognized ParamSpec kwarg {name}", node=node)
                return AnyValue(AnySource.error)
            tv = ParamSpec(name_val.val)
            return InputSigValue(ParamSpecParam(tv))
        elif is_typing_name(func.val, "deprecated") or func.val is deprecated:
            if node.keywords:
                self.ctx.show_error(
                    "deprecated() does not accept keyword arguments", node=node
                )
                return AnyValue(AnySource.error)
            arg_values = tuple(self.visit(arg) for arg in node.args)
            return DecoratorValue(deprecated, arg_values)
        elif isinstance(func.val, type):
            if func.val is object:
                return AnyValue(AnySource.inference)
            return TypedValue(func.val)
        raise _UnsupportedAnnotationExpression


def _is_tuple(typ: object) -> bool:
    return typ is tuple or is_typing_name(typ, "Tuple")


def _is_unpack_annotation_member(member: Value) -> bool:
    if isinstance(member, KnownValue):
        origin = get_origin(member.val)
        return is_typing_name(origin, "Unpack") or (
            isinstance(member.val, GenericAlias)
            and getattr(member.val, "__unpacked__", False)
        )
    if isinstance(member, PartialValue):
        if member.operation is PartialValueOperation.UNPACK:
            return True
        return (
            member.operation is PartialValueOperation.SUBSCRIPT
            and isinstance(member.root, KnownValue)
            and is_typing_name(member.root.val, "Unpack")
        )
    return False


def _is_unpack_runtime_arg(arg: object) -> bool:
    origin = get_origin(arg)
    return is_typing_name(origin, "Unpack") or (
        isinstance(arg, GenericAlias) and getattr(arg, "__unpacked__", False)
    )


def _annotation_expr_of_origin_args(
    origin: object, args: Sequence[object], val: object, ctx: Context
) -> AnnotationExpr:
    if is_typing_name(origin, "Annotated"):
        if not _require_min_argument_count(args, 2, "Annotated", ctx):
            return AnnotationExpr(ctx, AnyValue(AnySource.error))
        origin, *metadata = args
        inner = _annotation_expr_from_runtime(origin, ctx)
        meta = translate_annotated_metadata(
            [KnownValue(data) for data in metadata], ctx
        )
        return inner.add_metadata(meta)
    for qualifier in (
        Qualifier.Required,
        Qualifier.NotRequired,
        Qualifier.ReadOnly,
        Qualifier.ClassVar,
        Qualifier.Final,
        Qualifier.Unpack,
    ):
        if is_typing_name(origin, qualifier.name):
            if not _require_exact_argument_count(args, 1, qualifier.name, ctx):
                return AnnotationExpr(ctx, AnyValue(AnySource.error))
            inner = _annotation_expr_from_runtime(args[0], ctx)
            return inner.add_qualifier(qualifier, None)
    val = _value_of_origin_args(origin, args, val, ctx)
    return AnnotationExpr(ctx, val)


def _value_of_origin_args(
    origin: object, args: Sequence[object], val: object, ctx: Context
) -> Value:
    if is_typing_name(origin, "Unpack"):
        if not _require_exact_argument_count(args, 1, "Unpack", ctx):
            return AnyValue(AnySource.error)
        return _type_from_runtime(args[0], ctx)
    if origin is type:
        if not args:
            return TypedValue(type)
        if not _require_exact_argument_count(args, 1, "Type", ctx):
            return AnyValue(AnySource.error)
        return SubclassValue.make(_type_from_runtime(args[0], ctx))
    elif _is_tuple(origin):
        if not args:
            return SequenceValue(tuple, [])
        elif (
            len(args) == 2
            and args[1] is Ellipsis
            and not _is_unpack_runtime_arg(args[0])
        ):
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceValue(tuple, [])
        else:
            if any(arg is Ellipsis for arg in args):
                ctx.show_error("Ellipsis can be used only in tuple[T, ...]")
            exprs = [_annotation_expr_from_runtime(arg, ctx) for arg in args]
            return _make_sequence_value(tuple, exprs, ctx)
    elif is_union(origin):
        vals = [_type_from_runtime(arg, ctx) for arg in args]
        assert all(isinstance(val, Value) for val in vals), args
        return unite_values(*vals)
    elif origin is Callable or is_typing_name(origin, "Callable"):
        if len(args) == 0:
            return CallableValue(ANY_SIGNATURE)
        if not _require_exact_argument_count(args, 2, "Callable", ctx):
            return AnyValue(AnySource.error)
        arg_types, return_type = args
        params = _callable_args_from_runtime(arg_types, "Callable", ctx)
        sig = Signature.make(params, _type_from_runtime(return_type, ctx))
        return CallableValue(sig)
    elif is_typing_name(origin, "Annotated"):
        if not _require_min_argument_count(args, 2, "Annotated", ctx):
            return AnyValue(AnySource.error)
        origin, *metadata = args
        return _make_annotated(
            _type_from_runtime(origin, ctx),
            [KnownValue(data) for data in metadata],
            ctx,
        )
    elif isinstance(origin, type):
        runtime_origin = origin
        origin = _maybe_get_extra(origin)
        type_params = _get_generic_type_parameters_for_annotation(runtime_origin, ctx)
        is_empty_typevartuple_specialization = (
            val is not runtime_origin
            and not args
            and len(type_params) == 1
            and isinstance(type_params[0], TypeVarTupleParam)
        )
        if args or is_empty_typevartuple_specialization:
            packed_variadic_members = _pack_typevartuple_runtime_args(
                type_params, args, ctx
            )
            if packed_variadic_members is not None:
                args_vals = packed_variadic_members
            elif is_empty_typevartuple_specialization:
                args_vals = [SequenceValue(tuple, [])]
            elif len(type_params) == len(args):
                args_vals = [
                    _type_from_runtime_type_alias_arg(arg, type_param, ctx)
                    for arg, type_param in zip(args, type_params)
                ]
            else:
                args_vals = [_type_from_runtime(val, ctx) for val in args]
            args_vals = _normalize_paramspec_generic_args(type_params, args_vals, ctx)
            return GenericValue(origin, args_vals)
        else:
            return _maybe_typed_value(origin)
    if is_typing_name(origin, "Literal"):
        invalid_args = [arg for arg in args if not _is_valid_pep586_literal_value(arg)]
        if invalid_args:
            invalid_values = ", ".join(repr(arg) for arg in invalid_args)
            ctx.show_error(
                "Arguments to Literal[] must be None, bool, int, str, bytes, or enum"
                f" members; got {invalid_values}",
                error_code=ErrorCode.invalid_literal,
            )
        if len(args) == 1:
            return KnownValue(args[0])
        return unite_values(*[KnownValue(arg) for arg in args])
    elif is_typing_name(origin, "TypeGuard"):
        if not _require_exact_argument_count(args, 1, "TypeGuard", ctx):
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "TypeIs"):
        if not _require_exact_argument_count(args, 1, "TypeIs", ctx):
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeIsExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "TypeForm"):
        if not _require_exact_argument_count(args, 1, "TypeForm", ctx):
            return AnyValue(AnySource.error)
        return TypeFormValue(_type_from_runtime(args[0], ctx))
    elif is_instance_of_typing_name(origin, "TypeAliasType"):
        alias_object = origin
        type_params = tuple(
            make_type_param(type_param, ctx)
            for type_param in alias_object.__type_params__
        )
        if len(args) == len(type_params):
            args_vals = [
                _type_from_runtime_type_alias_arg(arg, type_param, ctx)
                for arg, type_param in zip(args, type_params)
            ]
        else:
            args_vals = [_type_from_runtime(val, ctx) for val in args]
        args_vals = _validate_type_alias_arg_values(type_params, args_vals, ctx)
        alias = ctx.get_type_alias(
            val,
            lambda: type_from_runtime(alias_object.__value__, ctx=ctx),
            lambda: tuple(
                make_type_param(type_param, ctx)
                for type_param in alias_object.__type_params__
            ),
        )
        return TypeAliasValue(
            alias_object.__name__, alias_object.__module__, alias, tuple(args_vals)
        )
    else:
        ctx.show_error(
            f"Unrecognized annotation {origin}[{', '.join(map(repr, args))}]"
        )
        return AnyValue(AnySource.error)


def _maybe_typed_value(val: type | str) -> Value:
    if val is type(None):
        return KnownValue(None)
    elif val is Hashable:
        return HashableProtoValue
    elif val is Callable or is_typing_name(val, "Callable"):
        return CallableValue(ANY_SIGNATURE)
    elif val is float:
        return TypedValue(float) | TypedValue(int)
    elif val is complex:
        return TypedValue(complex) | TypedValue(float) | TypedValue(int)
    return TypedValue(val)


def _make_sequence_value(
    typ: type, members: Sequence[AnnotationExpr], ctx: Context
) -> SequenceValue:
    pairs = []
    for expr in members:
        val, qualifiers = expr.unqualify({Qualifier.Unpack})
        if Qualifier.Unpack in qualifiers:
            elements = _unpack_value(val)
            if elements is None:
                ctx.show_error(f"Invalid usage of Unpack with {val}")
                elements = [(True, AnyValue(AnySource.error))]
            pairs += elements
        else:
            if isinstance(val, TypeVarTupleValue):
                ctx.show_error("TypeVarTuple must be unpacked")
                val = AnyValue(AnySource.error)
            pairs.append((False, val))
    if typ is tuple and sum(is_many for is_many, _ in pairs) > 1:
        ctx.show_error("Only one unbounded tuple can be used inside a tuple type")
    return SequenceValue(typ, pairs)


def _unpack_value(value: Value) -> Sequence[tuple[bool, Value]] | None:
    if isinstance(value, SequenceValue) and value.typ is tuple:
        return value.members
    elif isinstance(value, GenericValue) and value.typ is tuple:
        return [(True, value.args[0])]
    elif isinstance(value, TypeVarTupleValue):
        return [(True, value)]
    elif isinstance(value, TypedValue) and value.typ is tuple:
        return [(True, AnyValue(AnySource.generic_argument))]
    return None


def _make_callable_from_value(
    args: Value, return_value: Value, ctx: Context, is_asynq: bool = False
) -> Value:
    return_annotation = _type_from_value(return_value, ctx)
    if isinstance(args, KnownValue):
        args = replace_known_sequence_value(args)
    if args == KnownValue(Ellipsis):
        return CallableValue(
            Signature.make(
                [ELLIPSIS_PARAM], return_annotation=return_annotation, is_asynq=is_asynq
            )
        )
    elif isinstance(args, KnownValue) and args.val == [Ellipsis]:
        ctx.show_error(
            "Ellipsis must be used directly in Callable[..., T], not in Callable[[...], T]"
        )
        return AnyValue(AnySource.error)
    elif isinstance(args, KnownValue):
        params = _callable_args_from_runtime(args.val, "Callable", ctx)
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    elif isinstance(args, SequenceValue):
        members = args.get_member_sequence()
        if (
            args.typ is list
            and members is not None
            and len(members) == 1
            and members[0] == KnownValue(Ellipsis)
        ):
            ctx.show_error(
                "Ellipsis must be used directly in Callable[..., T], not in Callable[[...], T]"
            )
            return AnyValue(AnySource.error)
        normalized_args: list[tuple[bool, Value]] = []
        for is_many, arg in args.members:
            if _is_unpack_annotation_member(arg):
                expr = _annotation_expr_from_value(arg, ctx)
                unpacked, qualifiers = expr.unqualify({Qualifier.Unpack})
                if Qualifier.Unpack not in qualifiers:
                    normalized_args.append((False, unpacked))
                    continue
                unpacked_members = _unpack_value(unpacked)
                if unpacked_members is None:
                    ctx.show_error(f"Invalid usage of Unpack with {unpacked}")
                    normalized_args.append((False, AnyValue(AnySource.error)))
                    continue
                for unpacked_is_many, member in unpacked_members:
                    if unpacked_is_many and isinstance(member, TypeVarTupleValue):
                        normalized_args.append((True, member))
                        continue
                    normalized_args.append((unpacked_is_many, member))
                continue
            normalized_args.append(
                (is_many, arg if is_many else _type_from_value(arg, ctx))
            )
        if any(is_many for is_many, _ in normalized_args):
            params = [
                SigParameter(
                    "@args",
                    kind=ParameterKind.VAR_POSITIONAL,
                    annotation=SequenceValue(tuple, normalized_args),
                )
            ]
        else:
            params = [
                SigParameter(
                    f"@{i}", kind=ParameterKind.POSITIONAL_ONLY, annotation=annotation
                )
                for i, (_is_many, annotation) in enumerate(normalized_args)
            ]
        try:
            sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        except InvalidSignature as e:
            ctx.show_error(str(e))
            return AnyValue(AnySource.error)
        return CallableValue(sig)
    elif isinstance(args, KnownValue) and is_instance_of_typing_name(
        args.val, "ParamSpec"
    ):
        annotation = InputSigValue(ParamSpecParam(args.val))
        params = [
            SigParameter("__P", kind=ParameterKind.PARAM_SPEC, annotation=annotation)
        ]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    elif isinstance(args, InputSigValue):
        params = [SigParameter("__P", kind=ParameterKind.PARAM_SPEC, annotation=args)]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    elif (
        isinstance(args, PartialValue)
        and args.operation is PartialValueOperation.SUBSCRIPT
        and isinstance(args.root, KnownValue)
        and is_typing_name(args.root.val, "Concatenate")
    ):
        annotations = [_type_from_value(arg, ctx) for arg in args.members]
        params = [
            SigParameter(
                f"@{i}",
                kind=(
                    ParameterKind.PARAM_SPEC
                    if i == len(annotations) - 1
                    else ParameterKind.POSITIONAL_ONLY
                ),
                annotation=annotation,
            )
            for i, annotation in enumerate(annotations)
        ]
        sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        return CallableValue(sig)
    else:
        ctx.show_error(f"Unrecognized Callable type argument {args}")
        return AnyValue(AnySource.error)


def translate_annotated_metadata(
    metadata: Sequence[Value], ctx: Context
) -> tuple[Sequence[Value], Sequence[Extension]]:
    metadata_objs: list[Extension] = []
    intersects: list[Value] = []
    for entry in metadata:
        if isinstance(entry, KnownValue):
            if isinstance(entry.val, ParameterTypeGuard):
                metadata_objs.append(
                    ParameterTypeGuardExtension(
                        entry.val.varname,
                        _type_from_runtime(entry.val.guarded_type, ctx),
                    )
                )
                continue
            elif isinstance(entry.val, NoReturnGuard):
                metadata_objs.append(
                    NoReturnGuardExtension(
                        entry.val.varname,
                        _type_from_runtime(entry.val.guarded_type, ctx),
                    )
                )
                continue
            elif isinstance(entry.val, CustomCheck):
                metadata_objs.append(CustomCheckExtension(entry.val))
                continue
            annotated_types_extensions = list(get_annotated_types_extension(entry.val))
            if annotated_types_extensions:
                for obj in annotated_types_extensions:
                    if isinstance(obj, Value):
                        intersects.append(obj)
                    else:
                        metadata_objs.append(obj)
    return intersects, metadata_objs


def _make_annotated(origin: Value, metadata: Sequence[Value], ctx: Context) -> Value:
    intersects, metadata_objs = translate_annotated_metadata(metadata, ctx)
    if intersects:
        if ctx.can_assign_ctx is not None:
            origin = intersect_multi([origin, *intersects], ctx=ctx.can_assign_ctx)
        else:
            origin = IntersectionValue((origin, *intersects))
    return annotate_value(origin, metadata_objs)


_CONTEXT_MANAGER_TYPES = {
    "typing.AsyncContextManager",
    "typing.ContextManager",
    "contextlib.AbstractContextManager",
    "contextlib.AbstractAsyncContextManager",
    *CONTEXT_MANAGER_TYPES,
    *ASYNC_CONTEXT_MANAGER_TYPES,
}


def is_context_manager_type(typ: str | type) -> bool:
    return typ in _CONTEXT_MANAGER_TYPES
