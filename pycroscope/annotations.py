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
    cast,
    get_args,
    get_origin,
)

import typing_extensions
from typing_extensions import NoDefault, ParamSpec, TypedDict

from pycroscope.annotated_types import get_annotated_types_extension
from pycroscope.input_sig import InputSigValue, ParamSpecSig
from pycroscope.relations import HashableProtoValue

from . import type_evaluation
from .analysis_lib import object_from_string, override
from .error_code import Error, ErrorCode
from .extensions import (
    AsynqCallable,
    CustomCheck,
    ExternalType,
    HasAttrGuard,
    Intersection,
    NoReturnGuard,
    ParameterTypeGuard,
    TypeGuard,
    deprecated,
)
from .find_unused import used
from .functions import FunctionDefNode
from .node_visitor import ErrorContext
from .safe import is_instance_of_typing_name, is_typing_name, is_union
from .signature import (
    ANY_SIGNATURE,
    ELLIPSIS_PARAM,
    InvalidSignature,
    ParameterKind,
    Signature,
    SigParameter,
)
from .value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnnotationExpr,
    AnySource,
    AnyValue,
    CallableValue,
    CustomCheckExtension,
    DictIncompleteValue,
    Extension,
    GenericValue,
    HasAttrGuardExtension,
    IntersectionValue,
    KnownValue,
    KVPair,
    MultiValuedValue,
    NewTypeValue,
    NoReturnGuardExtension,
    ParameterTypeGuardExtension,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    Qualifier,
    SelfTVV,
    SequenceValue,
    SubclassValue,
    TypeAlias,
    TypeAliasValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeGuardExtension,
    TypeIsExtension,
    TypeVarLike,
    TypeVarValue,
    Value,
    annotate_value,
    unite_values,
)

if TYPE_CHECKING:
    from .name_check_visitor import NameCheckVisitor


CONTEXT_MANAGER_TYPES = (typing.ContextManager, contextlib.AbstractContextManager)
ASYNC_CONTEXT_MANAGER_TYPES = (
    typing.AsyncContextManager,
    contextlib.AbstractAsyncContextManager,
)


@dataclass
class Context:
    """A context for evaluating annotations.

    The base implementation does very little. Subclass this to do something more useful.

    """

    should_suppress_undefined_names: bool = field(default=False, init=False)
    """While this is True, no errors are shown for undefined names."""
    _being_evaluated: dict[int, Value] = field(default_factory=dict, init=False)

    def suppress_undefined_names(self) -> AbstractContextManager[None]:
        """Temporarily suppress errors about undefined names."""
        return override(self, "should_suppress_undefined_names", True)

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

    def handle_undefined_name(self, name: str) -> Value:
        if self.should_suppress_undefined_names:
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
                return KnownValue(getattr(root_value.val, node.attr))
            except AttributeError:
                self.show_error(
                    f"{root_value.val!r} has no attribute {node.attr!r}", node=node
                )
                return AnyValue(AnySource.error)
        elif not isinstance(root_value, AnyValue):
            self.show_error(f"Cannot resolve annotation {root_value}", node=node)
        return AnyValue(AnySource.error)

    def get_type_alias(
        self,
        key: object,
        evaluator: typing.Callable[[], Value],
        evaluate_type_params: typing.Callable[[], Sequence[TypeVarLike]],
    ) -> TypeAlias:
        return TypeAlias(evaluator, evaluate_type_params)


@dataclass
class RuntimeEvaluator(type_evaluation.Evaluator, Context):
    globals: Mapping[str, object] = field(repr=False)
    func: typing.Callable[..., Any]

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(node, ctx=self, error_on_unrecognized=False)

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`pycroscope.value.Value` corresponding to a name."""
        return self.get_name_from_globals(node.id, self.globals)


@dataclass
class SyntheticEvaluator(type_evaluation.Evaluator):
    error_ctx: ErrorContext
    annotations_context: Context

    def show_error(
        self,
        message: str,
        error_code: Error = ErrorCode.invalid_annotation,
        node: ast.AST | None = None,
    ) -> None:
        self.error_ctx.show_error(node or self.node, message, error_code=error_code)

    def evaluate_type(self, node: ast.AST) -> Value:
        return type_from_ast(node, ctx=self.annotations_context)

    def evaluate_value(self, node: ast.AST) -> Value:
        return value_from_ast(
            node, ctx=self.annotations_context, error_on_unrecognized=False
        )

    def get_name(self, node: ast.Name) -> Value:
        """Return the :class:`pycroscope.value.Value` corresponding to a name."""
        return self.annotations_context.get_name(node)

    @classmethod
    def from_visitor(
        cls,
        node: FunctionDefNode,
        visitor: "NameCheckVisitor",
        return_annotation: Value,
    ) -> "SyntheticEvaluator":
        return cls(
            node,
            return_annotation,
            visitor,
            _DefaultContext(visitor, node, use_name_node_for_error=True),
        )


@used  # part of an API
def type_from_ast(
    ast_node: ast.AST,
    visitor: Optional["NameCheckVisitor"] = None,
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
    visitor: Optional["NameCheckVisitor"] = None,
    ctx: Context | None = None,
) -> AnnotationExpr:
    """Given an AST node representing an annotation, return a
    ``AnnotationExpr``."""
    if ctx is None:
        ctx = _DefaultContext(visitor, ast_node)
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
    visitor: Optional["NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
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
    return _type_from_runtime(val, ctx)


def annotation_expr_from_runtime(
    val: object,
    *,
    visitor: Optional["NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    globals: Mapping[str, object] | None = None,
    ctx: Context | None = None,
) -> AnnotationExpr:
    if ctx is None:
        ctx = _DefaultContext(visitor, node, globals)
    return _annotation_expr_from_runtime(val, ctx)


def type_from_value(
    value: Value,
    visitor: Optional["NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    ctx: Context | None = None,
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
    return _type_from_value(value, ctx)


def annotation_expr_from_value(
    value: Value,
    *,
    visitor: Optional["NameCheckVisitor"] = None,
    node: ast.AST | None = None,
    ctx: Context | None = None,
) -> AnnotationExpr:
    if ctx is None:
        ctx = _DefaultContext(visitor, node)
    return _annotation_expr_from_value(value, ctx)


def value_from_ast(
    ast_node: ast.AST, ctx: Context, *, error_on_unrecognized: bool = True
) -> Value:
    val = _Visitor(ctx).visit(ast_node)
    if val is None:
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
        return AnnotationExpr(ctx, None, [(Qualifier.Final, None)])
    elif is_typing_name(val, "TypeAlias"):
        return AnnotationExpr(ctx, None, [(Qualifier.TypeAlias, None)])
    elif isinstance(val, InitVar):
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
            # This is necessary because the forward ref may be defined in a different file, in
            # which case we don't know which names are valid in it.
            with ctx.suppress_undefined_names():
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
        return ParamSpecArgsValue(get_origin(val))
    elif is_instance_of_typing_name(val, "ParamSpecKwargs"):
        return ParamSpecKwargsValue(get_origin(val))
    origin = get_origin(val)
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
    elif is_typing_name(type(val), "TypeVar"):
        tv = cast(TypeVar, val)
        return make_type_var_value(tv, ctx)
    elif is_instance_of_typing_name(val, "ParamSpec"):
        return InputSigValue(ParamSpecSig(val))
    elif is_instance_of_typing_name(val, "_ForwardRef") or is_instance_of_typing_name(
        val, "ForwardRef"
    ):
        if (result := ctx.is_being_evaluted(val)) is not None:
            return result
        final_value = AnyValue(AnySource.inference)
        with ctx.add_evaluation(
            val, val.__forward_arg__, val.__forward_module__, lambda: final_value
        ):
            # This is necessary because the forward ref may be defined in a different file, in
            # which case we don't know which names are valid in it.
            with ctx.suppress_undefined_names():
                final_value = _eval_forward_ref(val.__forward_arg__, ctx).to_value()
        return final_value
    elif is_instance_of_typing_name(val, "TypeAliasType"):
        alias = ctx.get_type_alias(
            val,
            lambda: type_from_runtime(val.__value__, ctx=ctx),
            lambda: val.__type_params__,
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


def make_type_var_value(tv: TypeVarLike, ctx: Context) -> TypeVarValue:
    if (
        isinstance(tv, (TypeVar, typing_extensions.TypeVar))
        and getattr(tv, "__bound__", None) is not None
    ):
        bound = _type_from_runtime(tv.__bound__, ctx)
    else:
        bound = None
    if isinstance(tv, (TypeVar, typing_extensions.TypeVar)) and getattr(
        tv, "__constraints__", ()
    ):
        constraints = tuple(
            _type_from_runtime(constraint, ctx) for constraint in tv.__constraints__
        )
    else:
        constraints = ()
    if hasattr(tv, "__default__") and tv.__default__ is not NoDefault:
        default = _type_from_runtime(tv.__default__, ctx)
    else:
        default = None
    return TypeVarValue(
        tv,
        bound=bound,
        constraints=constraints,
        default=default,
        is_typevartuple=is_instance_of_typing_name(tv, "TypeVarTuple"),
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
                param_spec = InputSigValue(ParamSpecSig(arg))
                param = SigParameter(
                    "__P", kind=ParameterKind.PARAM_SPEC, annotation=param_spec
                )
                return [param]
        types = [_type_from_runtime(arg, ctx) for arg in arg_types]
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
        param_spec = InputSigValue(ParamSpecSig(arg_types))
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
        {Qualifier.ReadOnly, Qualifier.Required, Qualifier.NotRequired}
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
    try:
        tree = ast.parse(val, mode="eval")
    except SyntaxError:
        ctx.show_error(f"Syntax error in type annotation: {val}")
        return AnnotationExpr(ctx, AnyValue(AnySource.error))
    else:
        return _annotation_expr_from_ast(tree.body, ctx)


def _annotation_expr_from_value(value: Value, ctx: Context) -> AnnotationExpr:
    if isinstance(value, KnownValue):
        return _annotation_expr_from_runtime(value.val, ctx)
    elif isinstance(value, _SubscriptedValue):
        return _annotation_expr_from_subscripted_value(
            value.root, value.root_node, value.members, ctx
        )
    else:
        return AnnotationExpr(ctx, _type_from_value(value, ctx))


def _type_from_value(value: Value, ctx: Context) -> Value:
    if isinstance(value, KnownValue):
        return _type_from_runtime(value.val, ctx)
    elif isinstance(value, (TypeVarValue, TypeAliasValue)):
        return value
    elif isinstance(value, MultiValuedValue):
        return unite_values(*[_type_from_value(val, ctx) for val in value.vals])
    elif isinstance(value, TypeFormValue):
        return value
    elif isinstance(value, AnnotatedValue):
        return _type_from_value(value.value, ctx)
    elif isinstance(value, _SubscriptedValue):
        return _type_from_subscripted_value(value.root, value.members, ctx)
    elif isinstance(value, AnyValue):
        return value
    elif isinstance(value, SubclassValue) and value.exactly:
        return value.typ
    elif isinstance(value, TypedValue) and isinstance(value.typ, str):
        # Synthetic type
        return value
    elif isinstance(value, InputSigValue):
        return value
    else:
        ctx.show_error(f"Unrecognized annotation {value}")
        return AnyValue(AnySource.error)


def _annotation_expr_from_subscripted_value(
    root: Value | None, root_node: ast.AST, members: Sequence[Value], ctx: Context
) -> AnnotationExpr:
    if not isinstance(root, KnownValue):
        val = _type_from_subscripted_value(root, members, ctx)
        return AnnotationExpr(ctx, val)
    root_val = root.val
    if is_typing_name(root_val, "Annotated"):
        origin, *metadata = members
        origin_expr = _annotation_expr_from_value(origin, ctx)
        return origin_expr.add_metadata(translate_annotated_metadata(metadata, ctx))
    for qualifier in (
        Qualifier.Required,
        Qualifier.NotRequired,
        Qualifier.ReadOnly,
        Qualifier.ClassVar,
        Qualifier.InitVar,
        Qualifier.Unpack,
    ):
        if is_typing_name(root_val, qualifier.name):
            if len(members) != 1:
                ctx.show_error(f"{qualifier.name}[] requires a single argument")
                return AnnotationExpr(ctx, AnyValue(AnySource.error))
            inner = _annotation_expr_from_value(members[0], ctx)
            return inner.add_qualifier(qualifier, root_node)
    val = _type_from_subscripted_value(root, members, ctx)
    return AnnotationExpr(ctx, val)


def _type_from_subscripted_value(
    root: Value | None, members: Sequence[Value], ctx: Context
) -> Value:
    if isinstance(root, GenericValue):
        if len(root.args) == len(members):
            return GenericValue(
                root.typ, [_type_from_value(member, ctx) for member in members]
            )
    if isinstance(root, _SubscriptedValue):
        root_type = _type_from_value(root, ctx)
        return _type_from_subscripted_value(root_type, members, ctx)
    elif isinstance(root, MultiValuedValue):
        return unite_values(
            *[
                _type_from_subscripted_value(subval, members, ctx)
                for subval in root.vals
            ]
        )
    if (
        isinstance(root, SubclassValue)
        and root.exactly
        and isinstance(root.typ, TypedValue)
    ):
        return GenericValue(
            root.typ.typ, [_type_from_value(elt, ctx) for elt in members]
        )

    if isinstance(root, TypedValue) and isinstance(root.typ, str):
        return GenericValue(root.typ, [_type_from_value(elt, ctx) for elt in members])

    assert isinstance(root, Value)
    if not isinstance(root, KnownValue):
        if root != AnyValue(AnySource.error):
            ctx.show_error(f"Cannot resolve subscripted annotation: {root}")
        return AnyValue(AnySource.error)
    root = root.val
    if root is typing.Union:
        return unite_values(*[_type_from_value(elt, ctx) for elt in members])
    elif is_typing_name(root, "Literal"):
        if all(isinstance(elt, KnownValue) for elt in members):
            return unite_values(*members)
        else:
            ctx.show_error(f"Arguments to Literal[] must be literals, not {members}")
            return AnyValue(AnySource.error)
    elif _is_tuple(root):
        if len(members) == 2 and members[1] == KnownValue(Ellipsis):
            return GenericValue(tuple, [_type_from_value(members[0], ctx)])
        elif len(members) == 1 and members[0] == KnownValue(()):
            return SequenceValue(tuple, [])
        else:
            exprs = [_annotation_expr_from_value(arg, ctx) for arg in members]
            return _make_sequence_value(tuple, exprs, ctx)
    elif root is typing.Optional:
        if len(members) != 1:
            ctx.show_error("Optional[] takes only one argument")
            return AnyValue(AnySource.error)
        return unite_values(KnownValue(None), _type_from_value(members[0], ctx))
    elif root is typing.Type or root is type:
        if len(members) != 1:
            ctx.show_error("Type[] takes only one argument")
            return AnyValue(AnySource.error)
        argument = _type_from_value(members[0], ctx)
        return SubclassValue.make(argument)
    elif is_typing_name(root, "Annotated"):
        origin, *metadata = members
        return _make_annotated(_type_from_value(origin, ctx), metadata, ctx)
    elif is_typing_name(root, "TypeGuard"):
        if len(members) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "TypeIs"):
        if len(members) != 1:
            ctx.show_error("TypeIs requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeIsExtension(_type_from_value(members[0], ctx))]
        )
    elif is_typing_name(root, "TypeForm"):
        if len(members) != 1:
            ctx.show_error("TypeForm requires a single argument")
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
    elif is_typing_name(root, "Unpack"):
        ctx.show_error("Unpack[] used in unsupported context")
        return AnyValue(AnySource.error)
    elif root is Callable or root is typing.Callable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx)
        ctx.show_error("Callable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif root is AsynqCallable:
        if len(members) == 2:
            args, return_value = members
            return _make_callable_from_value(args, return_value, ctx, is_asynq=True)
        ctx.show_error("AsynqCallable requires exactly two arguments")
        return AnyValue(AnySource.error)
    elif root is Intersection:
        if not members:
            ctx.show_error("Intersection[] is missing arguments")
            return AnyValue(AnySource.error)
        return IntersectionValue(
            tuple(_type_from_value(subval, ctx) for subval in members)
        )
    elif isinstance(root, type):
        return GenericValue(root, [_type_from_value(elt, ctx) for elt in members])
    elif is_typing_name(root, "ClassVar"):
        ctx.show_error("ClassVar[] used in unsupported context")
        return AnyValue(AnySource.error)
    else:
        origin = get_origin(root)
        if isinstance(origin, type):
            return GenericValue(origin, [_type_from_value(elt, ctx) for elt in members])
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
        visitor: "NameCheckVisitor",
        node: ast.AST | None,
        globals: Mapping[str, object] | None = None,
        use_name_node_for_error: bool = False,
    ) -> None:
        super().__init__()
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
        if node is None:
            node = self.node
        if self.visitor is not None and node is not None:
            self.visitor.show_error(node, message, error_code)

    def get_name(self, node: ast.Name) -> Value:
        if self.visitor is not None:
            val, _ = self.visitor.resolve_name(
                node,
                error_node=node if self.use_name_node_for_error else self.node,
                suppress_errors=self.should_suppress_undefined_names,
            )
            return val
        elif self.globals is not None:
            if node.id in self.globals:
                return KnownValue(self.globals[node.id])
            elif hasattr(builtins, node.id):
                return KnownValue(getattr(builtins, node.id))
        if self.should_suppress_undefined_names:
            return AnyValue(AnySource.inference)
        self.show_error(
            f"Undefined name {node.id!r} used in annotation",
            ErrorCode.undefined_name,
            node=node,
        )
        return AnyValue(AnySource.error)

    def get_type_alias(
        self,
        key: object,
        evaluator: typing.Callable[[], Value],
        evaluate_type_params: typing.Callable[[], Sequence[TypeVarLike]],
    ) -> TypeAlias:
        if self.visitor is not None:
            cache = self.visitor.checker.type_alias_cache
            cache_key = _type_alias_cache_key(key)
            if cache_key in cache:
                return cache[cache_key]
            alias = super().get_type_alias(key, evaluator, evaluate_type_params)
            cache[cache_key] = alias
            return alias
        return super().get_type_alias(key, evaluator, evaluate_type_params)


@dataclass(frozen=True)
class _SubscriptedValue(AnyValue):
    """Represents certain values encountered while parsing an AST directly, primarily in stubs.

    Ideally should not exist but currently it helps support some use cases.

    """

    root: Value | None
    root_node: ast.AST
    members: tuple[Value, ...]


@dataclass(frozen=True)
class DecoratorValue(Value):
    decorator: object
    args: tuple[Value, ...]


class _Visitor(ast.NodeVisitor):
    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.expr):
            # Unsupported annotation expression: let callers turn this into a
            # normal "invalid annotation" error rather than crashing.
            return None
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
        return _SubscriptedValue(AnySource.inference, value, node.value, members)

    def visit_Attribute(self, node: ast.Attribute) -> Value | None:
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
        keys = [self.visit(key) if key is not None else None for key in node.keys]
        values = [self.visit(value) for value in node.values]
        kvpairs = []
        for key, value in zip(keys, values):
            if key is None:
                # Just skip ** unpacking in stubs for now.
                kvpairs.append(
                    KVPair(AnyValue(AnySource.inference), AnyValue(AnySource.inference))
                )
            else:
                kvpairs.append(KVPair(key, value))
        return DictIncompleteValue(dict, kvpairs)

    def visit_Constant(self, node: ast.Constant) -> Value:
        return KnownValue(node.value)

    def visit_Expr(self, node: ast.Expr) -> Value:
        return self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Value | None:
        if isinstance(node.op, ast.BitOr):
            return _SubscriptedValue(
                AnySource.inference,
                KnownValue(Union),
                node,
                (self.visit(node.left), self.visit(node.right)),
            )
        else:
            return None

    def visit_Starred(self, node: ast.Starred) -> Value:
        value = self.visit(node.value)
        return _SubscriptedValue(
            AnySource.inference, KnownValue(typing_extensions.Unpack), node, (value,)
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value | None:
        # Only int and float negation on literals are supported.
        if isinstance(node.op, ast.USub):
            operand = self.visit(node.operand)
            if isinstance(operand, KnownValue) and isinstance(
                operand.val, (int, float)
            ):
                return KnownValue(-operand.val)
        return None

    def visit_Call(self, node: ast.Call) -> Value | None:
        func = self.visit(node.func)
        if not isinstance(func, KnownValue):
            return None
        if func.val == NewType:
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            args = []
            kwargs = {}
            for arg_value in arg_values:
                if isinstance(arg_value, KnownValue):
                    args.append(arg_value.val)
                else:
                    return None
            for name, kwarg_value in kwarg_values:
                if name is None:
                    if isinstance(kwarg_value, KnownValue) and isinstance(
                        kwarg_value.val, dict
                    ):
                        kwargs.update(kwarg_value.val)
                    else:
                        return None
                else:
                    if isinstance(kwarg_value, KnownValue):
                        kwargs[name] = kwarg_value.val
                    else:
                        return None
            return KnownValue(func.val(*args, **kwargs))
        elif is_typing_name(func.val, "TypeVar"):
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "TypeVar() requires at least one argument", node=node
                )
                return None
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error("TypeVar name must be a literal", node=node.args[0])
                return None
            constraints = []
            for arg_value in arg_values[1:]:
                constraints.append(_type_from_value(arg_value, self.ctx))
            bound = default = None
            for name, kwarg_value in kwarg_values:
                if name in ("covariant", "contravariant", "infer_variance"):
                    continue
                elif name == "bound":
                    bound = _type_from_value(kwarg_value, self.ctx)
                elif name == "default":
                    default = _type_from_value(kwarg_value, self.ctx)
                else:
                    self.ctx.show_error(f"Unrecognized TypeVar kwarg {name}", node=node)
                    return None
            tv = TypeVar(name_val.val)
            return TypeVarValue(
                tv, bound=bound, constraints=tuple(constraints), default=default
            )
        elif is_typing_name(func.val, "ParamSpec"):
            arg_values = [self.visit(arg) for arg in node.args]
            kwarg_values = [(kw.arg, self.visit(kw.value)) for kw in node.keywords]
            if not arg_values:
                self.ctx.show_error(
                    "ParamSpec() requires at least one argument", node=node
                )
                return None
            name_val = arg_values[0]
            if not isinstance(name_val, KnownValue):
                self.ctx.show_error(
                    "ParamSpec name must be a literal", node=node.args[0]
                )
                return None
            for name, _ in kwarg_values:
                # TODO support defaults
                self.ctx.show_error(f"Unrecognized ParamSpec kwarg {name}", node=node)
                return None
            tv = ParamSpec(name_val.val)
            return InputSigValue(ParamSpecSig(tv))
        elif is_typing_name(func.val, "deprecated") or func.val is deprecated:
            if node.keywords:
                self.ctx.show_error(
                    "deprecated() does not accept keyword arguments", node=node
                )
                return None
            arg_values = tuple(self.visit(arg) for arg in node.args)
            return DecoratorValue(deprecated, arg_values)
        elif isinstance(func.val, type):
            if func.val is object:
                return AnyValue(AnySource.inference)
            return TypedValue(func.val)
        else:
            return None


def _is_tuple(typ: object) -> bool:
    return typ is tuple or is_typing_name(typ, "Tuple")


def _annotation_expr_of_origin_args(
    origin: object, args: Sequence[object], val: object, ctx: Context
) -> AnnotationExpr:
    if is_typing_name(origin, "Annotated"):
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
            if len(args) != 1:
                ctx.show_error(f"{qualifier.name}[] requires a single argument")
                return AnnotationExpr(ctx, AnyValue(AnySource.error))
            inner = _annotation_expr_from_runtime(args[0], ctx)
            return inner.add_qualifier(qualifier, None)
    val = _value_of_origin_args(origin, args, val, ctx)
    return AnnotationExpr(ctx, val)


def _value_of_origin_args(
    origin: object, args: Sequence[object], val: object, ctx: Context
) -> Value:
    if origin is type or origin is type:
        if not args:
            return TypedValue(type)
        return SubclassValue.make(_type_from_runtime(args[0], ctx))
    elif _is_tuple(origin):
        if not args:
            return SequenceValue(tuple, [])
        elif len(args) == 2 and args[1] is Ellipsis:
            return GenericValue(tuple, [_type_from_runtime(args[0], ctx)])
        elif len(args) == 1 and args[0] == ():
            return SequenceValue(tuple, [])
        else:
            exprs = [_annotation_expr_from_runtime(arg, ctx) for arg in args]
            return _make_sequence_value(tuple, exprs, ctx)
    elif is_union(origin):
        vals = [_type_from_runtime(arg, ctx) for arg in args]
        assert all(isinstance(val, Value) for val in vals), args
        return unite_values(*vals)
    elif origin is Callable or is_typing_name(origin, "Callable"):
        if len(args) == 0:
            return CallableValue(ANY_SIGNATURE)
        *arg_types, return_type = args
        if len(arg_types) == 1 and isinstance(arg_types[0], list):
            arg_types = arg_types[0]
        params = _callable_args_from_runtime(arg_types, "Callable", ctx)
        sig = Signature.make(params, _type_from_runtime(return_type, ctx))
        return CallableValue(sig)
    elif is_typing_name(origin, "Annotated"):
        origin, *metadata = args
        return _make_annotated(
            _type_from_runtime(origin, ctx),
            [KnownValue(data) for data in metadata],
            ctx,
        )
    elif isinstance(origin, type):
        origin = _maybe_get_extra(origin)
        if args:
            args_vals = [_type_from_runtime(val, ctx) for val in args]
            return GenericValue(origin, args_vals)
        else:
            return _maybe_typed_value(origin)
    if is_typing_name(origin, "Literal"):
        if len(args) == 1:
            return KnownValue(args[0])
        else:
            return unite_values(*[KnownValue(arg) for arg in args])
    elif is_typing_name(origin, "TypeGuard"):
        if len(args) != 1:
            ctx.show_error("TypeGuard requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeGuardExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "TypeIs"):
        if len(args) != 1:
            ctx.show_error("TypeIs requires a single argument")
            return AnyValue(AnySource.error)
        return AnnotatedValue(
            TypedValue(bool), [TypeIsExtension(_type_from_runtime(args[0], ctx))]
        )
    elif is_typing_name(origin, "TypeForm"):
        if len(args) != 1:
            ctx.show_error("TypeForm requires a single argument")
            return AnyValue(AnySource.error)
        return TypeFormValue(_type_from_runtime(args[0], ctx))
    elif is_instance_of_typing_name(origin, "TypeAliasType"):
        args_vals = [_type_from_runtime(val, ctx) for val in args]
        alias_object = cast(Any, origin)
        alias = ctx.get_type_alias(
            val,
            lambda: type_from_runtime(alias_object.__value__, ctx=ctx),
            lambda: alias_object.__type_params__,
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
            pairs.append((False, val))
    return SequenceValue(typ, pairs)


def _unpack_value(value: Value) -> Sequence[tuple[bool, Value]] | None:
    if isinstance(value, SequenceValue) and value.typ is tuple:
        return value.members
    elif isinstance(value, GenericValue) and value.typ is tuple:
        return [(True, value.args[0])]
    elif isinstance(value, TypedValue) and value.typ is tuple:
        return [(True, AnyValue(AnySource.generic_argument))]
    return None


def _make_callable_from_value(
    args: Value, return_value: Value, ctx: Context, is_asynq: bool = False
) -> Value:
    return_annotation = _type_from_value(return_value, ctx)
    if args == KnownValue(Ellipsis):
        return CallableValue(
            Signature.make(
                [ELLIPSIS_PARAM], return_annotation=return_annotation, is_asynq=is_asynq
            )
        )
    elif isinstance(args, SequenceValue):
        params = []
        for i, (is_many, arg) in enumerate(args.members):
            annotation = _type_from_value(arg, ctx)
            if is_many:
                param = SigParameter(
                    f"@{i}",
                    kind=ParameterKind.VAR_POSITIONAL,
                    annotation=GenericValue(tuple, [annotation]),
                )
            else:
                param = SigParameter(
                    f"@{i}", kind=ParameterKind.POSITIONAL_ONLY, annotation=annotation
                )
            params.append(param)
        try:
            sig = Signature.make(params, return_annotation, is_asynq=is_asynq)
        except InvalidSignature as e:
            ctx.show_error(str(e))
            return AnyValue(AnySource.error)
        return CallableValue(sig)
    elif isinstance(args, KnownValue) and is_instance_of_typing_name(
        args.val, "ParamSpec"
    ):
        annotation = InputSigValue(ParamSpecSig(args.val))
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
        isinstance(args, _SubscriptedValue)
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
) -> list[Value | Extension]:
    metadata_objs: list[Value | Extension] = []
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
            elif isinstance(entry.val, HasAttrGuard):
                metadata_objs.append(
                    HasAttrGuardExtension(
                        entry.val.varname,
                        _type_from_runtime(entry.val.attribute_name, ctx),
                        _type_from_runtime(entry.val.attribute_type, ctx),
                    )
                )
                continue
            elif isinstance(entry.val, CustomCheck):
                metadata_objs.append(CustomCheckExtension(entry.val))
                continue
            annotated_types_extensions = list(get_annotated_types_extension(entry.val))
            if annotated_types_extensions:
                metadata_objs.extend(annotated_types_extensions)
            else:
                metadata_objs.append(entry)
    return metadata_objs


def _make_annotated(origin: Value, metadata: Sequence[Value], ctx: Context) -> Value:
    metadata_objs = translate_annotated_metadata(metadata, ctx)
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
