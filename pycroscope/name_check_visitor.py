"""

The core of the pycroscope type checker.

:class:`NameCheckVisitor` is the AST visitor that powers pycroscope's
type inference. It is the central object that invokes other parts of
the system.

"""

import abc
import ast
import asyncio
import collections
import collections.abc
import contextlib
import dataclasses
import enum
import itertools
import logging
import operator
import os
import pickle
import sys
import time
import traceback
import types
import typing
from abc import abstractmethod
from argparse import SUPPRESS, ArgumentParser
from collections.abc import Callable, Container, Generator, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from itertools import chain
from pathlib import Path
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    ClassVar,
    Optional,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)
from unittest.mock import ANY

import typeshed_client
from typing_extensions import NoDefault, Protocol, assert_never, is_typeddict

from pycroscope.input_sig import (
    ActualArguments,
    InputSigValue,
    ParamSpecSig,
    extract_type_params,
)

from . import attributes, format_strings, importer, node_visitor, type_evaluation
from .analysis_lib import (
    get_attribute_path,
    get_subclasses_recursively,
    is_cython_class,
    object_from_string,
    override,
)
from .annotated_types import Ge, Gt, Le, Lt
from .annotations import (
    AnnotationExpr,
    Qualifier,
    SyntheticEvaluator,
    annotation_expr_from_annotations,
    annotation_expr_from_ast,
    annotation_expr_from_runtime,
    annotation_expr_from_value,
    has_invalid_paramspec_usage,
    is_context_manager_type,
    is_instance_of_typing_name,
    is_typing_name,
    type_from_value,
    value_from_ast,
)
from .arg_spec import (
    ArgSpecCache,
    ClassesSafeToInstantiate,
    IgnoredCallees,
    UnwrapClass,
    is_dot_asynq_function,
)
from .asynq_checker import AsynqChecker
from .boolability import Boolability, get_boolability
from .checker import Checker, CheckerAttrContext
from .error_code import Error, ErrorCode
from .extensions import (
    AsynqCallable,
    ParameterTypeGuard,
    assert_error,
    evaluated,
    overload,
    patch_typing_overload,
    real_overload,
)
from .find_unused import UnusedObjectFinder, used
from .functions import (
    IMPLICIT_CLASSMETHODS,
    AsyncFunctionKind,
    AsyncProxyDecorators,
    AsynqDecorators,
    DecoratorValues,
    FunctionDecorator,
    FunctionDefNode,
    FunctionInfo,
    FunctionNode,
    FunctionResult,
    GeneratorValue,
    ReturnT,
    SendT,
    YieldT,
    compute_parameters,
    compute_value_of_function,
)
from .maybe_asynq import asynq, qcore
from .options import (
    BooleanOption,
    ConcatenatedOption,
    ConfigOption,
    IntegerOption,
    InvalidConfigOption,
    Options,
    PyObjectSequenceOption,
    StringSequenceOption,
    add_arguments,
)
from .patma import PatmaVisitor
from .predicates import EqualsPredicate, InPredicate
from .reexport import ImplicitReexportTracker
from .relations import (
    Relation,
    check_hashability,
    has_relation,
    intersect_multi,
    is_assignable,
    is_subtype,
)
from .safe import (
    all_of_type,
    is_dataclass_type,
    is_hashable,
    is_namedtuple_class,
    is_typing_name,
    safe_getattr,
    safe_hasattr,
    safe_isinstance,
    safe_issubclass,
    should_disable_runtime_call_for_namedtuple_class,
)
from .shared_options import EnforceNoUnused, ImportPaths, Paths
from .signature import (
    ANY_SIGNATURE,
    ARGS,
    KWARGS,
    Argument,
    BoundArgs,
    BoundMethodSignature,
    ConcreteSignature,
    InvalidSignature,
    MaybeSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    SigParameter,
    preprocess_args,
)
from .stacked_scopes import (
    EMPTY_ORIGIN,
    FALSY_CONSTRAINT,
    LEAVES_LOOP,
    LEAVES_SCOPE,
    NULL_CONSTRAINT,
    TRUTHY_CONSTRAINT,
    AbstractConstraint,
    AndConstraint,
    Composite,
    CompositeIndex,
    Constraint,
    ConstraintType,
    EquivalentConstraint,
    FunctionScope,
    OrConstraint,
    PredicateProvider,
    Scope,
    ScopeType,
    StackedScopes,
    SubScope,
    Varname,
    VarnameOrigin,
    VarnameWithOrigin,
    VisitorState,
    annotate_with_constraint,
    constrain_value,
    extract_constraints,
)
from .suggested_type import (
    CallArgs,
    display_suggested_type,
    prepare_type,
    should_suggest_type,
)
from .type_object import TypeObject, get_mro
from .typeshed import TypeshedFinder
from .value import (
    NO_RETURN_VALUE,
    SYS_PLATFORM_EXTENSION,
    SYS_VERSION_INFO_EXTENSION,
    UNINITIALIZED_VALUE,
    VOID,
    AlwaysPresentExtension,
    AnnotatedValue,
    AnySource,
    AnyValue,
    AssertErrorExtension,
    AsyncTaskIncompleteValue,
    CallableValue,
    CanAssign,
    CanAssignError,
    ConstraintExtension,
    CustomCheckExtension,
    DataclassTransformDecoratorExtension,
    DataclassTransformExtension,
    DataclassTransformInfo,
    DefiniteValueExtension,
    DeprecatedExtension,
    DictIncompleteValue,
    GenericBases,
    GenericValue,
    HasAttrExtension,
    IntersectionValue,
    KnownValue,
    KnownValueWithTypeVars,
    KVPair,
    MultiValuedValue,
    NoReturnConstraintExtension,
    OverlapMode,
    PartialValue,
    PartialValueOperation,
    PredicateValue,
    ReferencingValue,
    SelfT,
    SequenceValue,
    SkipDeprecatedExtension,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    SysPlatformExtension,
    SysVersionInfoExtension,
    TypeAlias,
    TypeAliasValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeGuardExtension,
    TypeIsExtension,
    TypeVarLike,
    TypeVarMap,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    Variance,
    annotate_value,
    concrete_values_from_iterable,
    flatten_values,
    get_tv_map,
    get_typevar_variance,
    has_any_base_value,
    is_async_iterable,
    is_iterable,
    is_union,
    kv_pairs_from_mapping,
    make_coro_type,
    replace_fallback,
    replace_known_sequence_value,
    set_self,
    stringify_object,
    unannotate_value,
    unite_and_simplify,
    unite_values,
    unpack_values,
)
from .yield_checker import YieldChecker

if sys.version_info >= (3, 11):
    TryNode = ast.Try | ast.TryStar
else:
    TryNode = ast.Try


def _strip_predicate_intersection(value: Value) -> Value:
    """Remove predicate-only refinements when invoking runtime dunder methods."""
    if not isinstance(value, IntersectionValue):
        return value
    vals = [subval for subval in value.vals if not isinstance(subval, PredicateValue)]
    if not vals or len(vals) == len(value.vals):
        return value
    if len(vals) == 1:
        return vals[0]
    return IntersectionValue(tuple(vals))


T = TypeVar("T")
U = TypeVar("U")
T_co = TypeVar("T_co", covariant=True)
U_co = TypeVar("U_co", covariant=True)
AwaitableValue = GenericValue(collections.abc.Awaitable, [TypeVarValue(T)])
KnownNone = KnownValue(None)
ExceptionValue = TypedValue(BaseException) | SubclassValue(TypedValue(BaseException))
ExceptionOrNone = ExceptionValue | KnownNone


class _SupportsDescriptorGet(Protocol):
    def __get__(
        self, instance: object, owner: type[object] | None = None, /
    ) -> object: ...


_TYPING_CONSTRUCTS_WITH_NAME_ARG: dict[str, str] = {
    "TypeVar": "name",
    "TypeVarTuple": "name",
    "ParamSpec": "name",
    "NewType": "name",
    "NamedTuple": "typename",
    "TypedDict": "typename",
}


def _is_known_none_annotation(value: Value) -> bool:
    return replace_fallback(value) == KnownNone


BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD = {
    ast.Add: ("addition", "__add__", "__iadd__", "__radd__"),
    ast.Sub: ("subtraction", "__sub__", "__isub__", "__rsub__"),
    ast.Mult: ("multiplication", "__mul__", "__imul__", "__rmul__"),
    ast.Div: ("division", "__truediv__", "__itruediv__", "__rtruediv__"),
    ast.Mod: ("modulo", "__mod__", "__imod__", "__rmod__"),
    ast.Pow: ("exponentiation", "__pow__", "__ipow__", "__rpow__"),
    ast.LShift: ("left-shifting", "__lshift__", "__ilshift__", "__rlshift__"),
    ast.RShift: ("right-shifting", "__rshift__", "__irshift__", "__rrshift__"),
    ast.BitOr: ("bitwise OR", "__or__", "__ior__", "__ror__"),
    ast.BitXor: ("bitwise XOR", "__xor__", "__ixor__", "__rxor__"),
    ast.BitAnd: ("bitwise AND", "__and__", "__iand__", "__rand__"),
    ast.FloorDiv: ("floor division", "__floordiv__", "__ifloordiv__", "__rfloordiv__"),
    ast.MatMult: ("matrix multiplication", "__matmul__", "__imatmul__", "__rmatmul__"),
    ast.Eq: ("equality", "__eq__", None, "__eq__"),
    ast.NotEq: ("inequality", "__ne__", None, "__ne__"),
    ast.Lt: ("less than", "__lt__", None, "__gt__"),
    ast.LtE: ("less than or equal", "__le__", None, "__ge__"),
    ast.Gt: ("greater than", "__gt__", None, "__lt__"),
    ast.GtE: ("greater than or equal", "__ge__", None, "__le__"),
    ast.In: ("contains", "__contains__", None, None),
    ast.NotIn: ("contains", "__contains__", None, None),
}

# Certain special methods are expected to return NotImplemented if they
# can't handle a particular argument, so that the interpreter can
# try some other call. To support thiis, such methods are allowed to
# return NotImplemented, even if their return annotation says otherwise.
# This is pieced together from the CPython source code, including:
# - Methods defined as SLOT1BIN in Objects/typeobject.c
# - Objects/abstract.c also does the binops
# - Rich comparison in object.c and typeobject.c
METHODS_ALLOWING_NOTIMPLEMENTED = {
    *[
        method
        for method in itertools.chain.from_iterable(
            data[1:] for data in BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD.values()
        )
        if method is not None
    ],
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__length_hint__",  # Objects/abstract.c
    "__subclasshook__",  # Modules/_abc.c
}

UNARY_OPERATION_TO_DESCRIPTION_AND_METHOD = {
    ast.Invert: ("inversion", "__invert__"),
    ast.UAdd: ("unary positive", "__pos__"),
    ast.USub: ("unary negation", "__neg__"),
}


def _in(a: object, b: Container[object]) -> bool:
    return operator.contains(b, a)


def _not_in(a: object, b: Container[object]) -> bool:
    return not operator.contains(b, a)


COMPARATOR_TO_OPERATOR = {
    ast.Eq: (operator.eq, operator.ne, None),
    ast.NotEq: (operator.ne, operator.eq, None),
    ast.Lt: (operator.lt, operator.ge, Lt),
    ast.LtE: (operator.le, operator.gt, Le),
    ast.Gt: (operator.gt, operator.le, Gt),
    ast.GtE: (operator.ge, operator.lt, Ge),
    ast.Is: (operator.is_, operator.is_not, None),
    ast.IsNot: (operator.is_not, operator.is_, None),
    ast.In: (_in, _not_in, None),
    ast.NotIn: (_not_in, _in, None),
}
_NEG_OPERATOR_TO_AST = {
    neg_op: node_cls for node_cls, (_, neg_op, _) in COMPARATOR_TO_OPERATOR.items()
}
AST_TO_REVERSE = {
    node_cls: _NEG_OPERATOR_TO_AST[op]
    for node_cls, (op, _, _) in COMPARATOR_TO_OPERATOR.items()
}

SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL = [KnownValue(property)]
if sys.version_info < (3, 11):
    SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL.append(KnownValue(asyncio.coroutine))
if asynq is not None:
    SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL.append(KnownValue(asynq.asynq))

SYNTHETIC_PROPERTY_GETTER_PREFIX = "%property_getter:"
SYNTHETIC_PROPERTY_SETTER_PREFIX = "%property_setter:"


class CustomContextManager(Protocol[T_co, U_co]):
    def __enter__(self) -> T_co:
        raise NotImplementedError

    def __exit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: types.TracebackType | None,
    ) -> U_co:
        raise NotImplementedError


class AsyncCustomContextManager(Protocol[T_co, U_co]):
    async def __aenter__(self) -> T_co:
        raise NotImplementedError

    async def __aexit__(
        self,
        __exc_type: type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: types.TracebackType | None,
    ) -> U_co:
        raise NotImplementedError


def _contains_unpack_annotation_value(value: Value) -> bool:
    if isinstance(value, PartialValue):
        if value.operation is PartialValueOperation.UNPACK:
            return True
        if (
            value.operation is PartialValueOperation.SUBSCRIPT
            and isinstance(value.root, KnownValue)
            and is_typing_name(value.root.val, "Unpack")
        ):
            return True
        return _contains_unpack_annotation_value(value.root) or any(
            _contains_unpack_annotation_value(member) for member in value.members
        )
    if isinstance(value, SequenceValue):
        if any(is_many for is_many, _ in value.members):
            return True
        return any(
            _contains_unpack_annotation_value(member) for _, member in value.members
        )
    if isinstance(value, KnownValue):
        origin = get_origin(value.val)
        if is_typing_name(origin, "Unpack"):
            return True
        if isinstance(value.val, (tuple, list)):
            return any(
                _contains_unpack_annotation_value(KnownValue(member))
                for member in value.val
            )
    return False


def _is_typevartuple_annotation_value(value: Value) -> bool:
    if isinstance(value, TypeVarValue):
        return value.is_typevartuple
    if not isinstance(value, KnownValue):
        return False
    return is_instance_of_typing_name(value.val, "TypeVarTuple") or is_typing_name(
        type(value.val), "TypeVarTuple"
    )


def _count_typevartuple_type_param_arg(value: Value) -> tuple[int, int]:
    if isinstance(value, PartialValue):
        if (
            value.operation is PartialValueOperation.UNPACK
            and _is_typevartuple_annotation_value(value.root)
        ):
            return (0, 1)
        if (
            value.operation is PartialValueOperation.SUBSCRIPT
            and isinstance(value.root, KnownValue)
            and is_typing_name(value.root.val, "Unpack")
            and len(value.members) == 1
            and _is_typevartuple_annotation_value(value.members[0])
        ):
            return (0, 1)
    if _is_typevartuple_annotation_value(value):
        return (1, 0)
    return (0, 0)


@dataclass(init=False)
class _AttrContext(CheckerAttrContext):
    visitor: "NameCheckVisitor"
    node: ast.AST | None
    ignore_none: bool = False
    record_reads: bool = True

    # Needs to be implemented explicitly to work around Cython limitations
    def __init__(
        self,
        root_composite: Composite,
        attr: str,
        visitor: "NameCheckVisitor",
        *,
        node: ast.AST | None,
        ignore_none: bool = False,
        skip_mro: bool = False,
        skip_unwrap: bool = False,
        prefer_typeshed: bool = False,
        record_reads: bool = True,
    ) -> None:
        super().__init__(
            root_composite,
            attr,
            visitor.options,
            skip_mro=skip_mro,
            skip_unwrap=skip_unwrap,
            prefer_typeshed=prefer_typeshed,
            checker=visitor.checker,
        )
        self.node = node
        self.visitor = visitor
        self.ignore_none = ignore_none
        self.record_reads = record_reads

    def record_usage(self, obj: object, val: Value) -> None:
        self.visitor._maybe_record_usage(obj, self.attr, val)

    def record_attr_read(self, obj: type) -> None:
        if self.record_reads and self.node is not None:
            self.visitor._record_type_attr_read(obj, self.attr, self.node)

    def get_property_type_from_argspec(self, obj: property) -> Value:
        return self.visitor.resolve_property(obj, self.root_composite, self.node)

    def should_ignore_none_attributes(self) -> bool:
        return self.ignore_none

    def should_include_synthetic_methods(self) -> bool:
        return self.attr != "__call__"

    def _synthetic_instance_attr_is_method(
        self,
        synthetic_class: SyntheticClassObjectValue,
        attr_name: str,
        *,
        seen: set[int],
    ) -> bool:
        synthetic_id = id(synthetic_class)
        if synthetic_id in seen:
            return False
        seen.add(synthetic_id)
        if attr_name in synthetic_class.method_attributes:
            return True
        for base in synthetic_class.base_classes:
            base = replace_fallback(base)
            synthetic_base: SyntheticClassObjectValue | None = None
            if isinstance(base, SyntheticClassObjectValue):
                synthetic_base = base
            elif isinstance(base, GenericValue) and isinstance(base.typ, (type, str)):
                synthetic_base = self.checker.get_synthetic_class(base.typ)
            elif isinstance(base, TypedValue) and isinstance(base.typ, (type, str)):
                synthetic_base = self.checker.get_synthetic_class(base.typ)
            elif isinstance(base, KnownValue) and isinstance(base.val, type):
                synthetic_base = self.checker.get_synthetic_class(base.val)
            if synthetic_base is not None and self._synthetic_instance_attr_is_method(
                synthetic_base, attr_name, seen=seen
            ):
                return True
        return False

    def bind_synthetic_instance_attribute(self, attr_name: str, value: Value) -> Value:
        # Treat synthetic instance methods like bound methods in both expression
        # and relation contexts. Callable fields that aren't methods should
        # remain regular callables when accessed on an instance.
        if isinstance(value, CallableValue):
            root_value = replace_fallback(self.root_composite.value)
            if isinstance(root_value, AnnotatedValue):
                root_value = replace_fallback(root_value.value)
            synthetic_typ: str | None = None
            generic_args: Sequence[Value] = ()
            if isinstance(root_value, GenericValue) and isinstance(root_value.typ, str):
                synthetic_typ = root_value.typ
                generic_args = root_value.args
            elif isinstance(root_value, TypedValue) and isinstance(root_value.typ, str):
                synthetic_typ = root_value.typ
            if synthetic_typ is None:
                return super().bind_synthetic_instance_attribute(attr_name, value)
            synthetic_class = self.checker.get_synthetic_class(synthetic_typ)
            is_dunder = attr_name.startswith("__") and attr_name.endswith("__")
            if is_dunder and attr_name != "__init__":
                return super().bind_synthetic_instance_attribute(attr_name, value)
            should_bind = synthetic_class is not None and (
                self._synthetic_instance_attr_is_method(
                    synthetic_class, attr_name, seen=set()
                )
            )
            if not should_bind:
                return super().bind_synthetic_instance_attribute(attr_name, value)
            if synthetic_class is not None:
                if self.checker.is_synthetic_classmethod_attribute(
                    synthetic_class, attr_name
                ):
                    # classmethod attributes are already descriptor-adjusted by
                    # synthetic attribute normalization; binding again drops one
                    # real parameter.
                    return super().bind_synthetic_instance_attribute(attr_name, value)
                raw_attr = synthetic_class.class_attributes.get(attr_name)
                if raw_attr is not None:
                    raw_attr = replace_fallback(raw_attr)
                    if (
                        isinstance(raw_attr, GenericValue)
                        and raw_attr.typ in {classmethod, staticmethod}
                    ) or (
                        isinstance(raw_attr, KnownValue)
                        and isinstance(raw_attr.val, (classmethod, staticmethod))
                    ):
                        # classmethod/staticmethod are already descriptor-adjusted by
                        # synthetic attribute normalization; binding again drops one
                        # real parameter.
                        return super().bind_synthetic_instance_attribute(
                            attr_name, value
                        )
            bound_signature = value.signature.bind_self(
                self_annotation_value=None,
                self_value=self.root_composite.value,
                ctx=self.checker,
            )
            if bound_signature is not None:
                generic_bases = self.checker.get_generic_bases(
                    synthetic_typ, generic_args
                )
                declared = generic_bases.get(synthetic_typ, {})
                if declared:
                    bound_signature = bound_signature.substitute_typevars(declared)
                return CallableValue(bound_signature)
        return super().bind_synthetic_instance_attribute(attr_name, value)

    def clone_for_attribute_lookup(
        self, root_composite: Composite, attr: str
    ) -> "_AttrContext":
        return _AttrContext(
            root_composite,
            attr,
            self.visitor,
            node=self.node,
            ignore_none=self.ignore_none,
            skip_mro=False,
            skip_unwrap=False,
            prefer_typeshed=False,
            record_reads=self.record_reads,
        )


@dataclass
class _SyntheticTypedDictContext:
    total: bool
    bases: list[TypedDictValue]
    inherited_extra_keys: Value | None
    inherited_extra_keys_readonly: bool
    extra_keys: Value | None
    extra_keys_readonly: bool
    local_items: dict[str, tuple[TypedDictEntry, ast.AST]] = dataclass_field(
        default_factory=dict
    )


@dataclass(frozen=True)
class DataclassInfo:
    init: bool
    eq: bool
    frozen: bool | None
    unsafe_hash: bool
    match_args: bool
    order: bool
    slots: bool
    kw_only_default: bool
    field_specifiers: tuple[Value, ...]

    @classmethod
    def from_transform_info_and_options(
        cls, info: DataclassTransformInfo, keywords: Mapping[str, ast.expr]
    ) -> "DataclassInfo":
        return DataclassInfo(
            init=_extract_bool(keywords.get("init"), True),
            eq=_extract_bool(keywords.get("eq"), info.eq_default),
            frozen=_extract_bool(keywords.get("frozen"), info.frozen_default),
            unsafe_hash=_extract_bool(keywords.get("unsafe_hash"), False),
            match_args=_extract_bool(keywords.get("match_args"), True),
            order=_extract_bool(keywords.get("order"), info.order_default),
            slots=_extract_bool(keywords.get("slots"), False),
            kw_only_default=_extract_bool(
                keywords.get("kw_only"), info.kw_only_default
            ),
            field_specifiers=info.field_specifiers,
        )


@dataclass(frozen=True)
class _ClassDataclassSemantics:
    is_dataclass: bool
    init: bool | None
    eq: bool | None
    frozen: bool | None
    unsafe_hash: bool | None
    match_args: bool | None
    order: bool | None
    slots: bool | None
    kw_only_default: bool | None
    field_specifiers: tuple[Value, ...]
    is_transform_provider: bool
    transform_info: DataclassTransformInfo | None = None


class ComprehensionLengthInferenceLimit(IntegerOption):
    """If we iterate over something longer than this, we don't try to infer precise
    types for comprehensions. Increasing this can hurt performance."""

    default_value = 25
    name = "comprehension_length_inference_limit"


class UnionSimplificationLimit(IntegerOption):
    """We may simplify unions with more than this many values."""

    default_value = 100
    name = "union_simplification_limit"


class OutputFormatOption(ConfigOption[node_visitor.OutputFormat]):
    """Output format for reported errors (`\"detailed\"` or `\"concise\"`)."""

    default_value = "detailed"
    name = "output_format"

    @classmethod
    def parse(cls, data: object, source_path: Path) -> node_visitor.OutputFormat:
        if data == "concise":
            return "concise"
        if data == "detailed":
            return "detailed"
        raise InvalidConfigOption.from_parser(cls, "'concise' or 'detailed'", data)

    @classmethod
    def create_command_line_option(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--output-format",
            choices=["concise", "detailed"],
            default=SUPPRESS,
            help=cls.__doc__,
            action="store",
        )


class DisallowCallsToDunders(StringSequenceOption):
    """Set of dunder methods (e.g., '{"__lshift__"}') that pycroscope is not allowed to call on
    objects."""

    name = "disallow_calls_to_dunders"


class DisallowedImports(StringSequenceOption):
    """List of imports that will trigger an error.

    Entries may be top-level modules (e.g., "os") or dotted submodule paths (e.g., "os.path").

    """

    name = "disallowed_imports"


class ForLoopAlwaysEntered(BooleanOption):
    """If True, we assume that for loops are always entered at least once,
    which affects the potentially_undefined_name check. This will miss
    some bugs but also remove some annoying false positives."""

    name = "for_loop_always_entered"


class IgnoreNoneAttributes(BooleanOption):
    """If True, we ignore None when type checking attribute access on a Union
    type."""

    name = "ignore_none_attributes"


class UnimportableModules(StringSequenceOption):
    """Do not attempt to import these modules if they are imported within a function."""

    default_value = []
    name = "unimportable_modules"


class ExtraBuiltins(StringSequenceOption):
    """Even if these variables are undefined, no errors are shown."""

    name = "extra_builtins"
    default_value = ["__IPYTHON__"]  # special global defined in IPython


class IgnoredPaths(ConcatenatedOption[Sequence[str]]):
    """Attribute accesses on these do not result in errors."""

    name = "ignored_paths"
    default_value = ()

    # too complicated and this option isn't too useful anyway
    should_create_command_line_option = False

    @classmethod
    def parse(cls, data: object, source_path: Path) -> Sequence[Sequence[str]]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(cls, "sequence", data)
        for sublist in data:
            if not isinstance(sublist, (list, tuple)):
                raise InvalidConfigOption.from_parser(cls, "sequence", sublist)
            for elt in sublist:
                if not isinstance(elt, str):
                    raise InvalidConfigOption.from_parser(cls, "string", elt)
        return data


class IgnoredEndOfReference(StringSequenceOption):
    """When these attributes are accessed but they don't exist, the error is ignored."""

    name = "ignored_end_of_reference"
    default_value = [
        # these are created by the mock module
        "call_count",
        "assert_has_calls",
        "reset_mock",
        "called",
        "assert_called_once",
        "assert_called_once_with",
        "assert_called_with",
        "count",
        "assert_any_call",
        "assert_not_called",
    ]


class IgnoredForIncompatibleOverride(StringSequenceOption):
    """These attributes are not checked for incompatible overrides."""

    name = "ignored_for_incompatible_overrides"
    default_value = ["__init__", "__eq__", "__ne__"]


class IgnoredUnusedAttributes(StringSequenceOption):
    """When these attributes are unused, they are not listed as such by the unused attribute
    finder."""

    name = "ignored_unused_attributes"
    default_value = [
        # ABCs
        "_abc_cache",
        "_abc_negative_cache",
        "__abstractmethods__",
        "_abc_negative_cache_version",
        "_abc_registry",
        # Python core
        "__module__",
        "__doc__",
        "__init__",
        "__dict__",
        "__weakref__",
        "__enter__",
        "__exit__",
        "__metaclass__",
    ]


class IgnoredUnusedClassAttributes(ConcatenatedOption[tuple[type, set[str]]]):
    """List of pairs of (class, set of attribute names). When these attribute names are seen as
    unused on a child or base class of the class, they are not listed."""

    name = "ignored_unused_class_attributes"
    default_value = []
    should_create_command_line_option = False  # too complicated

    @classmethod
    def parse(cls, data: object, source_path: Path) -> Sequence[tuple[type, set[str]]]:
        if not isinstance(data, (list, tuple)):
            raise InvalidConfigOption.from_parser(
                cls, "sequence of (type, [attribute]) pairs", data
            )
        final = []
        for elt in data:
            if not isinstance(elt, (list, tuple)) or len(elt) != 2:
                raise InvalidConfigOption.from_parser(
                    cls, "sequence of (type, [attribute]) pairs", elt
                )
            typ, attrs = elt
            try:
                obj = object_from_string(typ)
            except Exception:
                raise InvalidConfigOption.from_parser(
                    cls, "path to Python object", typ
                ) from None
            if not isinstance(obj, type):
                raise InvalidConfigOption.from_parser(cls, "type", obj)
            if not isinstance(attrs, (list, tuple)):
                raise InvalidConfigOption.from_parser(
                    cls, "sequence of attributes", attrs
                )
            for attr in attrs:
                if not isinstance(attr, str):
                    raise InvalidConfigOption.from_parser(cls, "attribute string", attr)
            final.append((obj, set(attrs)))
        return final


class CheckForDuplicateValues(PyObjectSequenceOption[type]):
    """For subclasses of these classes, we error if multiple attributes have the same
    value. This is used for the duplicate_enum check."""

    name = "check_for_duplicate_values"
    default_value = [enum.Enum]


class AllowDuplicateValues(PyObjectSequenceOption[type]):
    """For subclasses of these classes, we do not error if multiple attributes have the same
    value. This overrides CheckForDuplicateValues."""

    name = "allow_duplicate_values"
    default_value = []


def should_check_for_duplicate_values(cls: object, options: Options) -> bool:
    if not isinstance(cls, type):
        return False
    positive_list = tuple(options.get_value_for(CheckForDuplicateValues))
    if not safe_issubclass(cls, positive_list):
        return False
    negative_list = tuple(options.get_value_for(AllowDuplicateValues))
    if safe_issubclass(cls, negative_list):
        return False
    return True


def _anything_to_any(obj: object) -> Value | None:
    if obj is ANY:
        return AnyValue(AnySource.explicit)
    if qcore is not None and obj is qcore.testing.Anything:
        return AnyValue(AnySource.explicit)
    return None


class TransformGlobals(PyObjectSequenceOption[Callable[[object], Value | None]]):
    """Transform global variables."""

    name = "transform_globals"
    default_value = [_anything_to_any]


class IgnoredTypesForAttributeChecking(PyObjectSequenceOption[type]):
    """Used in the check for object attributes that are accessed but not set. In general, the check
    will only alert about attributes that don't exist when it has visited all the base classes of
    the class with the possibly missing attribute. However, these classes are never going to be
    visited (since they're builtin), but they don't set any attributes that we rely on.
    """

    name = "ignored_types_for_attribute_checking"
    default_value = [object, abc.ABC]


class ClassAttributeChecker:
    """Helper class to keep track of attributes that are read and set on instances."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        should_check_unused_attributes: bool = False,
        should_serialize: bool = False,
        options: Options = Options.from_option_list(),
        ts_finder: TypeshedFinder | None = None,
    ) -> None:
        self.options = options
        # we might not have examined all parent classes when looking for attributes set
        # we dump them here. in case the callers want to extend coverage.
        self.unexamined_base_classes = set()
        self.modules_examined = set()
        self.enabled = enabled
        self.should_check_unused_attributes = should_check_unused_attributes
        self.should_serialize = should_serialize
        self.all_failures = []
        self.types_with_dynamic_attrs = set()
        self.filename_to_visitor = {}
        # Dictionary from type to list of (attr_name, node, filename) tuples
        self.attributes_read = collections.defaultdict(list)
        # Dictionary from type to set of attributes that are set on that class
        self.attributes_set = collections.defaultdict(set)
        # Used for attribute value inference
        self.attribute_values = collections.defaultdict(dict)
        # Classes that we have examined the AST for
        self.classes_examined = {
            self.serialize_type(typ)
            for typ in self.options.get_value_for(IgnoredTypesForAttributeChecking)
        }
        self.ts_finder = ts_finder

    def __enter__(self) -> Optional["ClassAttributeChecker"]:
        if self.enabled:
            return self
        else:
            return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        if exc_type is None and self.enabled:
            self.check_attribute_reads()

            if self.should_check_unused_attributes:
                self.check_unused_attributes()

    def record_attribute_read(
        self, typ: type, attr_name: str, node: ast.AST, visitor: "NameCheckVisitor"
    ) -> None:
        """Records that attribute attr_name was accessed on type typ."""
        self.filename_to_visitor[visitor.filename] = visitor
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.attributes_read[serialized].append((attr_name, node, visitor.filename))

    def record_attribute_set(
        self, typ: type, attr_name: str, node: ast.AST, value: Value
    ) -> None:
        """Records that attribute attr_name was set on type typ."""
        serialized = self.serialize_type(typ)
        if serialized is None:
            return
        self.attributes_set[serialized].add(attr_name)
        self.merge_attribute_value(serialized, attr_name, value)

    def merge_attribute_value(
        self, serialized: object, attr_name: str, value: Value
    ) -> None:
        try:
            pickle.loads(pickle.dumps(value))
        except Exception:
            # If we can't serialize it, don't attempt to store it.
            value = AnyValue(AnySource.inference)
        scope = self.attribute_values[serialized]
        if attr_name not in scope:
            scope[attr_name] = value
        elif scope[attr_name] == value:
            pass
        else:
            scope[attr_name] = unite_values(scope[attr_name], value)

    def record_type_has_dynamic_attrs(self, typ: type) -> None:
        serialized = self.serialize_type(typ)
        if serialized is not None:
            self.types_with_dynamic_attrs.add(serialized)

    def record_class_examined(self, cls: type) -> None:
        """Records that we examined the attributes of class cls."""
        serialized = self.serialize_type(cls)
        if serialized is not None:
            self.classes_examined.add(serialized)

    def record_module_examined(self, module_name: str) -> None:
        self.modules_examined.add(module_name)

    def serialize_type(self, typ: type) -> object:
        """Serialize a type so it is pickleable.

        We do this to make it possible to pass ClassAttributeChecker objects around
        to parallel workers.

        """
        if not self.should_serialize:
            try:
                hash(typ)
            except Exception:
                return None  # ignore non-hashable types
            else:
                return typ
        if isinstance(typ, super):
            typ = typ.__self_class__
        if isinstance(safe_getattr(typ, "__module__", None), str) and isinstance(
            safe_getattr(typ, "__name__", None), str
        ):
            module = typ.__module__
            name = typ.__name__
            if module not in sys.modules:
                return None
            actual = safe_getattr(sys.modules[module], name, None)
            if UnwrapClass.unwrap(actual, self.options) is typ:
                return (module, name)
        return None

    def unserialize_type(self, serialized: Any) -> type | None:
        if not self.should_serialize:
            return serialized
        module, name = serialized
        if module not in sys.modules:
            __import__(module)
        try:
            actual = getattr(sys.modules[module], name)
            return UnwrapClass.unwrap(actual, self.options)
        except AttributeError:
            # We've seen this happen when we import different modules under the same name.
            return None

    def get_attribute_value(self, typ: type, attr_name: str) -> Value:
        """Gets the current recorded value of the attribute."""
        for base_typ in get_mro(typ):
            serialized_base = self.serialize_type(base_typ)
            if serialized_base is None:
                continue
            value = self.attribute_values[serialized_base].get(attr_name)
            if value is not None:
                return value
        return AnyValue(AnySource.inference)

    def check_attribute_reads(self) -> None:
        """Checks that all recorded attribute reads refer to valid attributes.

        This is done by checking for each read whether the class has the attribute or whether any
        code sets the attribute on a class instance, among other conditions.

        """
        for serialized, attrs_read in sorted(
            self.attributes_read.items(), key=self._cls_sort
        ):
            typ = self.unserialize_type(serialized)
            if typ is None:
                continue
            # we setattr on it with an unresolved value, so we don't know what attributes this may
            # have
            if any(
                self.serialize_type(base_cls) in self.types_with_dynamic_attrs
                for base_cls in get_mro(typ)
            ):
                continue

            for attr_name, node, filename in sorted(
                attrs_read, key=lambda data: data[0]
            ):
                self._check_attribute_read(
                    typ, attr_name, node, self.filename_to_visitor[filename]
                )

    def check_unused_attributes(self) -> None:
        """Attempts to find attributes.

        This relies on comparing the set of attributes read on each class with the attributes in the
        class's ``__dict__``. It has many false positives and should be considered experimental.

        Some known causes of false positives:

        - Methods called in base classes of children (mixins)
        - Special methods like ``__eq__``
        - Insufficiently powerful type inference

        """
        all_attrs_read = collections.defaultdict(set)

        def _add_attrs(typ: Any, attr_names_read: set[str]) -> None:
            if typ is None:
                return
            all_attrs_read[typ] |= attr_names_read
            for base_cls in typ.__bases__:
                all_attrs_read[base_cls] |= attr_names_read
            if isinstance(typ, type):
                for child_cls in get_subclasses_recursively(typ):
                    all_attrs_read[child_cls] |= attr_names_read

        for serialized, attrs_read in self.attributes_read.items():
            attr_names_read = {attr_name for attr_name, _, _ in attrs_read}
            _add_attrs(self.unserialize_type(serialized), attr_names_read)

        for typ, attrs in self.options.get_value_for(IgnoredUnusedClassAttributes):
            _add_attrs(typ, attrs)

        ignored = set(self.options.get_value_for(IgnoredUnusedAttributes))
        for typ, attrs_read in sorted(all_attrs_read.items(), key=self._cls_sort):
            if self.serialize_type(typ) not in self.classes_examined:
                continue
            existing_attrs = set(typ.__dict__.keys())
            for attr in existing_attrs - attrs_read - ignored:
                # server calls will always show up as unused here
                if safe_getattr(safe_getattr(typ, attr, None), "server_call", False):
                    continue
                print(f"Unused method: {typ!r}.{attr}")

    # sort by module + name in order to get errors in a reasonable order
    def _cls_sort(self, pair: tuple[Any, Any]) -> tuple[str, ...]:
        typ = pair[0]
        if hasattr(typ, "__name__") and isinstance(typ.__name__, str):
            return (str(typ.__module__), str(typ.__name__))
        else:
            return (str(typ), "")

    def _check_attribute_read(
        self, typ: type, attr_name: str, node: ast.AST, visitor: "NameCheckVisitor"
    ) -> None:
        # class itself has the attribute
        if hasattr(typ, attr_name):
            return
        # the attribute is in __annotations__, e.g. a dataclass
        if _has_annotation_for_attr(typ, attr_name) or attributes.get_attrs_attribute(
            typ,
            attributes.AttrContext(
                Composite(TypedValue(typ)),
                attr_name,
                visitor.options,
                skip_unwrap=False,
                skip_mro=False,
                prefer_typeshed=False,
            ),
        ):
            return

        # name mangling
        if attr_name.startswith("__") and hasattr(typ, f"_{typ.__name__}{attr_name}"):
            return

        # can't be sure whether it exists if class has __getattr__
        if hasattr(typ, "__getattr__") or (
            typ.__getattribute__ is not object.__getattribute__
        ):
            return

        # instances of old-style classes have __class__ available, even though the class doesn't
        if attr_name == "__class__":
            return

        serialized = self.serialize_type(typ)

        # it was set on an instance of the class
        if attr_name in self.attributes_set[serialized]:
            return

        # web browser test classes
        if attr_name == "browser" and hasattr(typ, "_pre_setup"):
            return

        base_classes_examined = {typ}
        any_base_classes_unexamined = False
        for base_cls in get_mro(typ):
            # the attribute is in __annotations__, e.g. a dataclass
            if _has_annotation_for_attr(base_cls, attr_name):
                return

            if self._should_reject_unexamined(base_cls):
                self.unexamined_base_classes.add(base_cls)
                any_base_classes_unexamined = True
                continue

            # attribute was set on the base class
            if attr_name in self.attributes_set[
                self.serialize_type(base_cls)
            ] or hasattr(base_cls, attr_name):
                return

            base_classes_examined.add(base_cls)

        if any_base_classes_unexamined:
            return

        if not isinstance(typ, type):
            # old-style class; don't want to support
            return

        # if it's on a child class it's also ok
        for child_cls in get_subclasses_recursively(typ):
            # also check the child classes' base classes, because mixins sometimes use attributes
            # defined on other parents of their child classes
            for base_cls in get_mro(child_cls):
                if base_cls in base_classes_examined:
                    continue

                if attr_name in self.attributes_set[
                    self.serialize_type(base_cls)
                ] or hasattr(base_cls, attr_name):
                    return

                if self._should_reject_unexamined(base_cls):
                    visitor.log(
                        logging.INFO,
                        "Rejecting because of unexamined child base class",
                        (typ, base_cls, attr_name),
                    )
                    return

                base_classes_examined.add(base_cls)

        # If the class has only known attributes, we already reported the error.
        if _has_only_known_attributes(self.ts_finder, typ):
            return

        message = visitor.show_error(
            node,
            f"Attribute {attr_name} of type {typ} probably does not exist",
            ErrorCode.attribute_is_never_set,
        )
        # message can be None if the error is intercepted by error code settings or ignore
        # directives
        if message is not None:
            self.all_failures.append(message)

    def _should_reject_unexamined(self, base_cls: type) -> bool:
        """Whether an undefined attribute should be ignored because base_cls was not examined.

        This is to keep the script from concluding that an attribute does not exist because it was
        defined on a base class whose AST was not examined.

        In two cases we still want to throw an error for undefined components even if a base class
        was not examined:
        - If the base class's module was examined, it is probably a wrapper class created by a
          decorator that does not set additional attributes.
        - If the base class is a Cython class, it should not set any attributes that are not defined
          on the class.

        """
        result = (
            self.serialize_type(base_cls) not in self.classes_examined
            and base_cls.__module__ not in self.modules_examined
            and not is_cython_class(base_cls)
        )
        if not result:
            self.unexamined_base_classes.add(base_cls)
        return result


_AstType = type[ast.AST] | tuple[type[ast.AST], ...]


class StackedContexts:
    """Object to keep track of a stack of states.

    This is used to indicate all the AST node types that are parents of the node being examined.

    """

    contexts: list[ast.AST]

    def __init__(self) -> None:
        self.contexts = []

    def includes(self, typ: _AstType) -> bool:
        return any(isinstance(val, typ) for val in self.contexts)

    def nth_parent(self, n: int) -> ast.AST | None:
        return self.contexts[-n] if len(self.contexts) >= n else None

    def nearest_enclosing(self, typ: _AstType) -> ast.AST | None:
        for node in reversed(self.contexts):
            if isinstance(node, typ):
                return node
        return None

    @contextlib.contextmanager
    def add(self, value: ast.AST) -> Generator[None]:
        """Context manager to add a context to the stack."""
        self.contexts.append(value)
        try:
            yield
        finally:
            self.contexts.pop()


@dataclass(frozen=True)
class _PendingOverload:
    node: FunctionDefNode
    signature: ConcreteSignature | None
    decorator_kinds: frozenset[FunctionDecorator]
    dataclass_transform_info: DataclassTransformInfo | None = None


@dataclass
class _PendingOverloadBlock:
    name: str
    scope: Scope
    overloads: list[_PendingOverload] = dataclass_field(default_factory=list)


@dataclass
class _EnumMemberTracker:
    by_value: dict[object, str] = dataclass_field(default_factory=dict)
    by_name: dict[str, object] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class _DataclassFieldCallOptions:
    init: bool | None = None
    kw_only: bool | None = None
    alias: str | None = None
    has_default: bool = False
    default_factory: Value | None = None


@dataclass
class _DataclassFieldInferenceCallContext:
    checker: "NameCheckVisitor"
    errors: list[str] = dataclass_field(default_factory=list)

    @property
    def visitor(self) -> "NameCheckVisitor":
        return self.checker

    @property
    def can_assign_ctx(self) -> "NameCheckVisitor":
        return self.checker

    def on_error(
        self,
        message: str,
        *,
        code: Error = ErrorCode.incompatible_call,
        node: ast.AST | None = None,
        detail: str | None = None,
        replacement: node_visitor.Replacement | None = None,
    ) -> None:
        self.errors.append(message)


@used  # exposed as an API
class CallSiteCollector:
    """Class to record function calls with their origin."""

    def __init__(self) -> None:
        self.map = collections.defaultdict(list)

    def record_call(self, caller: object, callee: object) -> None:
        try:
            self.map[callee].append(caller)
        except TypeError:
            # Unhashable callee. This is mostly calls to bound versions of list.append. We could get
            # the unbound method, but that doesn't seem very useful, so we just ignore it.
            pass


class NameCheckVisitor(node_visitor.ReplacingNodeVisitor):
    """Visitor class that infers the type and value of Python objects and detects errors."""

    error_code_enum = ErrorCode
    config_filename: ClassVar[str | None] = None
    """Path (relative to this class's file) to a pyproject.toml config file."""

    _argspec_to_retval: dict[int, tuple[Value, MaybeSignature]]
    _pending_overload_blocks: dict[int, _PendingOverloadBlock]
    _synthetic_classes_by_name: dict[str, SyntheticClassObjectValue]
    _synthetic_abstract_methods: dict[str, set[str]]
    _synthetic_final_methods: dict[str, set[str]]
    _dataclass_field_call_options_by_node: dict[int, _DataclassFieldCallOptions]
    _function_decorator_kinds_by_node: dict[
        ast.FunctionDef | ast.AsyncFunctionDef, frozenset[FunctionDecorator]
    ]
    _function_returns_self_by_node: dict[ast.FunctionDef | ast.AsyncFunctionDef, bool]
    _type_alias_first_definition_by_scope: dict[int, dict[str, ast.AST]]
    _type_alias_unguarded_refs_by_scope: dict[int, dict[str, set[str]]]
    _method_cache: dict[type[ast.AST], Callable[[Any], Value | None]]
    _name_node_to_statement: dict[ast.AST, ast.AST | None] | None
    _should_exclude_any: bool
    _statement_types: set[type[ast.AST]]
    ann_assign_type: tuple[Value | None, bool] | None
    annotate: bool
    arg_spec_cache: ArgSpecCache
    async_kind: AsyncFunctionKind
    asynq_checker: AsynqChecker
    attribute_checker: ClassAttributeChecker
    being_assigned: Value | None
    checker: Checker
    collector: CallSiteCollector | None
    current_class: type | str | None
    current_dataclass_info: DataclassInfo | None
    current_class_key: type | str | None
    current_class_type_params: Sequence[TypeVarValue] | None
    _active_pep695_type_params: list[set[object]]
    current_enum_members: _EnumMemberTracker | None
    current_function: object | None
    current_function_info: FunctionInfo | None
    current_function_name: str | None
    current_synthetic_typeddict: _SyntheticTypedDictContext | None
    error_for_implicit_any: bool
    expected_return_value: Value | None
    future_imports: set[str]
    in_annotation: bool
    in_comprehension_body: bool
    in_union_decomposition: bool
    import_name_to_node: dict[str, ast.Import | ast.ImportFrom]
    is_async_def: bool
    is_compiled: bool
    is_generator: bool
    match_subject: Composite
    module: types.ModuleType | None
    node_context: StackedContexts
    options: Options
    reexport_tracker: ImplicitReexportTracker
    return_values: list[Value | None]
    scopes: StackedScopes
    state: VisitorState
    unused_finder: UnusedObjectFinder
    final_class_keys: set[type | str]
    final_member_names_by_class: dict[type | str, set[str]]
    final_members_initialized_in_init: dict[type | str, set[str]]
    final_members_requiring_init: dict[type | str, dict[str, ast.AST]]
    enum_class_keys: set[type | str]
    enum_value_type_by_class: dict[type | str, Value]
    yield_checker: YieldChecker

    def __init__(
        self,
        filename: str,
        contents: str,
        tree: ast.Module,
        *,
        settings: Mapping[Error, bool] | None = None,
        fail_after_first: bool = False,
        verbosity: int = logging.CRITICAL,
        unused_finder: UnusedObjectFinder | None = None,
        module: types.ModuleType | None = None,
        attribute_checker: ClassAttributeChecker | None = None,
        collector: CallSiteCollector | None = None,
        annotate: bool = False,
        add_ignores: bool = False,
        checker: Checker,
        is_code_only: bool = False,
    ) -> None:
        super().__init__(
            filename,
            contents,
            tree,
            settings,
            fail_after_first=fail_after_first,
            verbosity=verbosity,
            add_ignores=add_ignores,
            is_code_only=is_code_only,
        )
        self.checker = checker

        # State (to use in with override())
        self.state = VisitorState.collect_names
        # value currently being assigned
        self.being_assigned = AnyValue(AnySource.inference)
        self.ann_assign_type = None
        # current match target
        self.match_subject = Composite(AnyValue(AnySource.inference))
        # current class (for inferring the type of cls and self arguments)
        self.current_class = None
        self.current_dataclass_info = None
        self.current_class_key = None
        self.current_class_type_params = None
        self._active_pep695_type_params = []
        self.current_synthetic_typeddict = None
        self.current_function_name = None
        self.current_function_info = None

        # async
        self.async_kind = AsyncFunctionKind.non_async
        self.is_generator = False  # set to True if this function is a generator
        # if true, we annotate each node we visit with its inferred value
        self.annotate = annotate
        # true if we're in the body of a comprehension's loop
        self.in_comprehension_body = False
        self.options = checker.options

        if module is not None:
            self.module = module
            self.is_compiled = False
        else:
            self.module, self.is_compiled = self._load_module()

        if self.module is not None and hasattr(self.module, "__name__"):
            module_path = tuple(self.module.__name__.split("."))
            self.options = checker.options.for_module(module_path)
        self.output_format = self.options.get_value_for(OutputFormatOption)

        # Data storage objects
        self.unused_finder = unused_finder
        self.attribute_checker = attribute_checker
        self.arg_spec_cache = checker.arg_spec_cache
        self.reexport_tracker = checker.reexport_tracker
        if (
            self.attribute_checker is not None
            and self.module is not None
            and not self.is_compiled
        ):
            self.attribute_checker.record_module_examined(self.module.__name__)

        self.scopes = build_stacked_scopes(
            self.module,
            simplification_limit=self.options.get_value_for(UnionSimplificationLimit),
            options=self.options,
        )
        self.node_context = StackedContexts()
        self.asynq_checker = AsynqChecker(
            self.options, self.module, self.show_error, self.log, self.replace_node
        )
        self.yield_checker = YieldChecker(self)
        self.current_function = None
        self.expected_return_value = None
        self.current_enum_members = None
        self.is_async_def = False
        self.in_annotation = False
        self.in_union_decomposition = False
        self.collector = collector
        self.import_name_to_node = {}
        self.future_imports = set()  # active future imports in this file
        self.return_values = []
        self.error_for_implicit_any = self.options.is_error_code_enabled(
            ErrorCode.implicit_any
        )

        self._name_node_to_statement = None
        # Cache the return values of functions within this file, so that we can use them to
        # infer types. Previously, we cached this globally, but that makes things non-
        # deterministic because we'll start depending on the order modules are checked.
        self._argspec_to_retval = {}
        self._pending_overload_blocks = {}
        self._synthetic_classes_by_name = {}
        self._synthetic_abstract_methods = {}
        self._synthetic_final_methods = {}
        self._dataclass_field_call_options_by_node = {}
        self._function_decorator_kinds_by_node = {}
        self._function_returns_self_by_node = {}
        self._type_alias_first_definition_by_scope = {}
        self._type_alias_unguarded_refs_by_scope = {}
        self._method_cache = {}
        self._statement_types = set()
        self._should_exclude_any = False
        self.final_class_keys = set()
        self.final_member_names_by_class = {}
        self.final_members_initialized_in_init = {}
        self.final_members_requiring_init = {}
        self.enum_class_keys = set()
        self.enum_value_type_by_class = {}
        self._fill_method_cache()

    def get_local_return_value(self, sig: MaybeSignature) -> Value | None:
        val, saved_sig = self._argspec_to_retval.get(id(sig), (None, None))
        if sig is not saved_sig:
            return None
        return val

    def make_type_object(self, typ: type | super | str) -> TypeObject:
        return self.checker.make_type_object(typ)

    def can_assume_compatibility(self, left: TypeObject, right: TypeObject) -> bool:
        return self.checker.can_assume_compatibility(left, right)

    def assume_compatibility(
        self, left: TypeObject, right: TypeObject
    ) -> AbstractContextManager[None]:
        return self.checker.assume_compatibility(left, right)

    def can_aliases_assume_compatibility(
        self, left: TypeAliasValue, right: TypeAliasValue
    ) -> bool:
        return self.checker.can_aliases_assume_compatibility(left, right)

    def aliases_assume_compatibility(
        self, left: TypeAliasValue, right: TypeAliasValue
    ) -> AbstractContextManager[None]:
        return self.checker.aliases_assume_compatibility(left, right)

    def get_relation_cache(self) -> dict[object, object] | None:
        return self.checker.get_relation_cache()

    def has_active_relation_assumptions(self) -> bool:
        return self.checker.has_active_relation_assumptions()

    def get_type_alias_cache(self) -> dict[object, TypeAlias]:
        return self.checker.get_type_alias_cache()

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""
        pass

    def set_exclude_any(self) -> AbstractContextManager[None]:
        """Within this context, `Any` is compatible only with itself."""
        return override(self, "_should_exclude_any", True)

    def should_exclude_any(self) -> bool:
        """Whether Any should be compatible only with itself."""
        return self._should_exclude_any

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence[Value] = ()
    ) -> GenericBases:
        return self.checker.get_generic_bases(typ, generic_args)

    def get_type_parameters(self, typ: type | str) -> Sequence[Value]:
        return self.checker.get_type_parameters(typ)

    def get_signature(
        self, obj: object, is_asynq: bool = False
    ) -> ConcreteSignature | None:
        return self.checker.get_signature(obj, is_asynq=is_asynq)

    def __reduce_ex__(self, proto: object) -> object:
        # Only pickle the attributes needed to get error reporting working
        return self.__class__, (self.filename, self.contents, self.tree, self.settings)

    def _get_import_failure_lineno(self, error: BaseException) -> int | None:
        target_filename = _maybe_normalize_filename(self.filename)

        tb = error.__traceback__
        lineno = None
        while tb is not None:
            frame_filename = _maybe_normalize_filename(tb.tb_frame.f_code.co_filename)
            if frame_filename == target_filename:
                lineno = tb.tb_lineno
            tb = tb.tb_next
        if lineno is not None:
            return lineno

        lineno = getattr(error, "lineno", None)
        if not isinstance(lineno, int) or lineno <= 0:
            return None

        error_filename = getattr(error, "filename", None)
        if error_filename is None:
            return lineno
        if _maybe_normalize_filename(error_filename) == target_filename:
            return lineno
        return None

    def _get_import_failure_node(
        self, error: BaseException
    ) -> ast.AST | node_visitor._FakeNode | None:
        lineno = self._get_import_failure_lineno(error)
        if lineno is not None:
            return node_visitor._FakeNode(lineno=lineno, col_offset=0)
        if self.tree is not None and self.tree.body:
            return self.tree.body[0]
        return None

    def _load_module(self) -> tuple[types.ModuleType | None, bool]:
        """Sets the module_path and module for this file."""
        if not self.filename:
            return None, False
        self.log(logging.INFO, "Checking file", (self.filename, os.getpid()))
        if self.is_code_only:
            mod_dict = {}
            try:
                exec(compile(self.contents, self.filename, "exec"), mod_dict)
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                node = self._get_import_failure_node(e)
                self.show_error(
                    node,
                    f"Failed to execute code due to {e!r}",
                    error_code=ErrorCode.import_failed,
                )
                return None, False
            mod = types.ModuleType(self.filename)
            mod.__dict__.update(mod_dict)
            return mod, False
        import_paths = self.options.get_value_for(ImportPaths)

        try:
            return importer.load_module_from_file(
                self.filename, import_paths=[str(p) for p in import_paths]
            )
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            # don't re-raise the error, just proceed without a module object
            # this can happen with scripts that aren't intended to be imported
            if not self.has_file_level_ignore():
                node = self._get_import_failure_node(e)
                failure = self.show_error(
                    node,
                    f"Failed to import {self.filename} due to {e!r}",
                    error_code=ErrorCode.import_failed,
                )
                if failure is not None:
                    # Don't print a traceback if the error was suppressed.
                    traceback.print_exc()
            return None, False

    def check(self, ignore_missing_module: bool = False) -> list[node_visitor.Failure]:
        """Run the visitor on this module."""
        start_time = time.time()
        try:
            if self.is_compiled:
                # skip compiled (Cythonized) files because pycroscope will misinterpret the
                # AST in some cases (for example, if a function was cdefed)
                return []
            if self.tree is None:
                return self.all_failures
            if self.module is None and not ignore_missing_module:
                # Keep checking so we can surface non-import-related issues too.
                self.log(
                    logging.INFO,
                    "Continuing check despite missing module",
                    self.filename,
                )
            with override(self, "state", VisitorState.collect_names):
                self.visit(self.tree)
            self._pending_overload_blocks.clear()
            with override(self, "state", VisitorState.check_names):
                self.visit(self.tree)
                self._flush_pending_overload_blocks()
            self._pending_overload_blocks.clear()
            # This doesn't deal correctly with errors from the attribute checker. Therefore,
            # leaving this check disabled by default for now.
            self.show_errors_for_unused_ignores(ErrorCode.unused_ignore)
            self.show_errors_for_bare_ignores(ErrorCode.bare_ignore)
            if (
                self.module is not None
                and self.unused_finder is not None
                and not self.has_file_level_ignore()
            ):
                self.unused_finder.record_module_visited(self.module)
            if self.module is not None and self.module.__name__ is not None:
                self.reexport_tracker.record_module_completed(self.module.__name__)
        except node_visitor.VisitorError:
            raise
        except Exception as e:
            self.show_error(
                None,
                f"{traceback.format_exc()}\nInternal error: {e!r}",
                error_code=ErrorCode.internal_error,
            )
        # Recover memory used for the AST. We keep the visitor object around later in order
        # to show ClassAttributeChecker errors, but those don't need the full AST.
        self.tree = None
        self._argspec_to_retval.clear()
        end_time = time.time()
        message = f"{self.filename} took {end_time - start_time:.2f} s"
        self.logger.log(logging.INFO, message)
        return self.all_failures

    def visit(self, node: ast.AST) -> Value:
        """Visit a node and return the :class:`pycroscope.value.Value` corresponding
        to the node."""
        # inline self.node_context.add and the superclass's visit() for performance
        node_type = type(node)
        method = self._method_cache[node_type]
        self.node_context.contexts.append(node)
        try:
            # This part inlines ReplacingNodeVisitor.visit
            if node_type in self._statement_types:
                # inline override here
                old_statement = self.current_statement
                try:
                    self.current_statement = node
                    ret = method(node)
                finally:
                    self.current_statement = old_statement
            else:
                ret = method(node)
        except node_visitor.VisitorError:
            raise
        except Exception as e:
            self.show_error(
                node,
                f"{traceback.format_exc()}\nInternal error: {e!r}",
                error_code=ErrorCode.internal_error,
            )
            ret = AnyValue(AnySource.error)
        finally:
            self.node_context.contexts.pop()
        if ret is None:
            ret = VOID
        if self.annotate:
            node.inferred_value = ret
        if self.error_for_implicit_any:
            for val in ret.walk_values():
                if isinstance(val, AnyValue) and val.source is not AnySource.explicit:
                    self._show_error_if_checking(
                        node,
                        f"Inferred value contains Any: {ret}",
                        ErrorCode.implicit_any,
                    )
        return ret

    def generic_visit(self, node: ast.AST) -> None:
        # Inlined version of ast.Visitor.generic_visit for performance.
        for field in node._fields:
            try:
                value = getattr(node, field)
            except AttributeError:
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def _fill_method_cache(self) -> None:
        for typ in get_subclasses_recursively(ast.AST):
            method = "visit_" + typ.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[typ] = visitor
            if issubclass(typ, ast.stmt):
                self._statement_types.add(typ)

    def _is_collecting(self) -> bool:
        return self.state == VisitorState.collect_names

    def _is_checking(self) -> bool:
        return self.state == VisitorState.check_names

    def _show_error_if_checking(
        self,
        node: ast.AST,
        msg: str | None = None,
        error_code: Error | None = None,
        *,
        replacement: node_visitor.Replacement | None = None,
        detail: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        """We usually should show errors only in the check_names state to avoid duplicate errors."""
        if self._is_checking():
            self.show_error(
                node,
                msg,
                error_code=error_code,
                replacement=replacement,
                detail=detail,
                extra_metadata=extra_metadata,
            )

    def _set_name_in_scope(
        self,
        varname: str,
        node: ast.AST,
        value: Value | None = AnyValue(AnySource.inference),
        *,
        private: bool = False,
        lookup_node: object = None,
    ) -> tuple[Value, VarnameOrigin]:
        if lookup_node is None:
            lookup_node = node

        current_scope = self.scopes.current_scope()
        if self.ann_assign_type is not None:
            expected_type, is_final = self.ann_assign_type
            if not current_scope.set_declared_type(
                varname, expected_type, is_final, node
            ):
                self._show_error_if_checking(
                    node, f"{varname} already declared", ErrorCode.already_declared
                )

        else:
            declared_type = current_scope.get_declared_type(varname)
            if declared_type is not None and value is not None:
                can_assign = has_relation(
                    declared_type, value, Relation.ASSIGNABLE, self
                )
                if isinstance(can_assign, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        f"Incompatible assignment: expected {declared_type}, got"
                        f" {value}",
                        error_code=ErrorCode.incompatible_assignment,
                        detail=can_assign.display(),
                    )
            if current_scope.is_final(varname):
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to final name {varname}",
                    ErrorCode.incompatible_assignment,
                )

        scope_type = current_scope.scope_type
        if scope_type == ScopeType.module_scope:
            if (
                self.module is not None
                and self.module.__name__ is not None
                and not private
            ):
                self.reexport_tracker.record_exported_attribute(
                    self.module.__name__, varname
                )
            if varname in current_scope.variables:
                existing, _ = current_scope.get_local(
                    varname, lookup_node, self.state, can_assign_ctx=self
                )
                if self._is_checking() and value is not None:
                    if self.module is None:
                        # If module import failed, collected module-scope values are
                        # provisional. Prefer check-pass static inference, but keep
                        # declared types for annotated names.
                        declared_type = current_scope.get_declared_type(varname)
                        if declared_type is not None:
                            current_scope.variables[varname] = declared_type
                            return declared_type, EMPTY_ORIGIN
                        current_scope.variables[varname] = value
                        return value, EMPTY_ORIGIN
                    # Runtime module loading can populate values that differ from
                    # static typing semantics for typing helper calls.
                    if isinstance(existing, KnownValue) and isinstance(
                        value, TypeFormValue
                    ):
                        current_scope.variables[varname] = value
                        return value, EMPTY_ORIGIN
                    if isinstance(existing, TypeAliasValue) and not isinstance(
                        value, TypeAliasValue
                    ):
                        # Keep explicit TypeAlias declarations usable at runtime
                        # (e.g. ListAlias()) while still preserving alias metadata
                        # in declared_types for annotation contexts.
                        current_scope.variables[varname] = value
                        return value, EMPTY_ORIGIN
                    # Generic aliases like list[int]() evaluate at runtime to plain
                    # instances (e.g. []), so preserve static generic information.
                    if isinstance(existing, KnownValue) and isinstance(
                        value, GenericValue
                    ):
                        if isinstance(value.typ, type) and isinstance(
                            existing.val, value.typ
                        ):
                            current_scope.variables[varname] = value
                            return value, EMPTY_ORIGIN
                    if isinstance(value, AnnotatedValue) and value.has_metadata_of_type(
                        DataclassTransformExtension
                    ):
                        current_scope.variables[varname] = value
                        return value, EMPTY_ORIGIN
                return existing, EMPTY_ORIGIN
        if scope_type == ScopeType.class_scope:
            if value is not None:
                self._check_for_incompatible_overrides(varname, node, value)
            self._check_for_class_variable_redefinition(varname, node)
        if value is None:
            return AnyValue(AnySource.inference), EMPTY_ORIGIN
        origin = current_scope.set(varname, value, lookup_node, self.state)
        if scope_type == ScopeType.class_scope:
            self._set_synthetic_class_attribute(varname, value)
        return value, origin

    def _get_synthetic_class_for_current_scope(
        self,
    ) -> SyntheticClassObjectValue | None:
        current_class = self.current_class
        if isinstance(current_class, str):
            return self._synthetic_classes_by_name.get(current_class)
        if isinstance(current_class, type):
            return self.checker.get_synthetic_class(current_class)
        return None

    def _set_synthetic_class_attribute(self, name: str, value: Value) -> None:
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return
        synthetic_name = name
        synthetic_value = value
        if isinstance(self.current_class, (type, str)) and self._is_enum_class_key(
            self.current_class
        ):
            class_name = self._current_class_name_from_context()
            if class_name is not None:
                mangled = _mangle_private_enum_name(class_name, name)
                if mangled is not None:
                    synthetic_name = mangled
                    if isinstance(synthetic_value, KnownValue):
                        synthetic_value = TypedValue(type(synthetic_value.val))
        synthetic_class.class_attributes[synthetic_name] = synthetic_value
        if not synthetic_name.startswith("%"):
            self._discard_synthetic_instance_only_annotation_name(synthetic_name)

    def _record_synthetic_classvar_name(self, name: str) -> None:
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return
        existing = synthetic_class.class_attributes.get("%classvars")
        classvar_names: set[str] = set()
        if isinstance(existing, KnownValue) and isinstance(
            existing.val, (set, frozenset, tuple, list)
        ):
            classvar_names.update(
                item for item in existing.val if isinstance(item, str)
            )
        classvar_names.add(name)
        synthetic_class.class_attributes["%classvars"] = KnownValue(
            frozenset(classvar_names)
        )

    def _record_synthetic_instance_only_annotation_name(self, name: str) -> None:
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return
        existing = synthetic_class.class_attributes.get("%instance_only_annotations")
        names: set[str] = set()
        if isinstance(existing, KnownValue) and isinstance(
            existing.val, (set, frozenset, tuple, list)
        ):
            names.update(item for item in existing.val if isinstance(item, str))
        names.add(name)
        synthetic_class.class_attributes["%instance_only_annotations"] = KnownValue(
            frozenset(names)
        )

    def _discard_synthetic_instance_only_annotation_name(self, name: str) -> None:
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return
        existing = synthetic_class.class_attributes.get("%instance_only_annotations")
        if not isinstance(existing, KnownValue) or not isinstance(
            existing.val, (set, frozenset, tuple, list)
        ):
            return
        names = {item for item in existing.val if isinstance(item, str)}
        if name not in names:
            return
        names.discard(name)
        if names:
            synthetic_class.class_attributes["%instance_only_annotations"] = KnownValue(
                frozenset(names)
            )
        else:
            synthetic_class.class_attributes.pop("%instance_only_annotations", None)

    def _record_synthetic_property_metadata(
        self, node: FunctionDefNode, info: FunctionInfo
    ) -> None:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return
        if self.scopes.scope_type() is not ScopeType.class_scope:
            return
        class_name = self._current_class_name_from_context()
        if class_name is None:
            return
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return

        mangled_name = _mangle_class_attribute_name(class_name, node.name)
        getter_key = f"{SYNTHETIC_PROPERTY_GETTER_PREFIX}{mangled_name}"
        if any(
            isinstance(unapplied, KnownValue) and unapplied.val is property
            for unapplied, _, _ in info.decorators
        ):
            getter_value = (
                info.return_annotation
                if info.return_annotation is not None
                else AnyValue(AnySource.unannotated)
            )
            synthetic_class.class_attributes[getter_key] = getter_value

        setter_target_name: str | None = None
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "setter"
                and isinstance(decorator.value, ast.Name)
            ):
                setter_target_name = decorator.value.id
                break
        if setter_target_name is None:
            return

        mangled_target = _mangle_class_attribute_name(class_name, setter_target_name)
        if FunctionDecorator.staticmethod in info.decorator_kinds:
            value_index = 0
        else:
            value_index = 1
        if value_index >= len(info.params):
            setter_value: Value = AnyValue(AnySource.unannotated)
        else:
            setter_annotation = info.params[value_index].param.annotation
            if setter_annotation is None:
                setter_value = AnyValue(AnySource.unannotated)
            else:
                setter_value = setter_annotation
        synthetic_class.class_attributes[
            f"{SYNTHETIC_PROPERTY_SETTER_PREFIX}{mangled_target}"
        ] = setter_value

    def _is_classvar_member(self, class_key: type | str, attr_name: str) -> bool:
        return attr_name in self._classvar_names_for_class_key(class_key, set())

    def _is_direct_classvar_member(self, class_key: type | str, attr_name: str) -> bool:
        synthetic_class: SyntheticClassObjectValue | None = None
        if isinstance(class_key, type):
            synthetic_class = self.checker.get_synthetic_class(class_key)
        elif isinstance(class_key, str):
            synthetic_class = self._synthetic_classes_by_name.get(class_key)
        if synthetic_class is not None and attr_name in _classvar_names_from_mapping(
            synthetic_class.class_attributes
        ):
            return True
        if isinstance(class_key, type):
            return attr_name in self._runtime_classvar_names_for_class(class_key)
        return False

    def _classvar_names_for_class_key(
        self, class_key: type | str, seen: set[type | str]
    ) -> set[str]:
        if class_key in seen:
            return set()
        seen.add(class_key)
        names: set[str] = set()
        synthetic_class: SyntheticClassObjectValue | None = None
        if isinstance(class_key, type):
            synthetic_class = self.checker.get_synthetic_class(class_key)
        elif isinstance(class_key, str):
            synthetic_class = self._synthetic_classes_by_name.get(class_key)
        if synthetic_class is not None:
            names.update(_classvar_names_from_mapping(synthetic_class.class_attributes))
            for base in synthetic_class.base_classes:
                for base_value in flatten_values(base, unwrap_annotated=True):
                    base_key: type | str | None = None
                    if isinstance(base_value, SyntheticClassObjectValue):
                        class_type = base_value.class_type
                        if isinstance(class_type, TypedValue) and isinstance(
                            class_type.typ, (type, str)
                        ):
                            base_key = class_type.typ
                    elif isinstance(base_value, GenericValue) and isinstance(
                        base_value.typ, (type, str)
                    ):
                        base_key = base_value.typ
                    elif isinstance(base_value, TypedValue) and isinstance(
                        base_value.typ, (type, str)
                    ):
                        base_key = base_value.typ
                    elif isinstance(base_value, KnownValue) and isinstance(
                        base_value.val, type
                    ):
                        base_key = base_value.val
                    if base_key is not None:
                        names.update(self._classvar_names_for_class_key(base_key, seen))
        if isinstance(class_key, type):
            names.update(self._runtime_classvar_names_for_class(class_key))
            for base_class in self.checker.get_generic_bases(class_key):
                if base_class != class_key:
                    names.update(self._classvar_names_for_class_key(base_class, seen))
        else:
            for base_class in self.checker.get_generic_bases(class_key):
                if base_class != class_key:
                    names.update(self._classvar_names_for_class_key(base_class, seen))
        return names

    def _runtime_classvar_names_for_class(self, cls: type) -> set[str]:
        annotations = safe_getattr(cls, "__annotations__", None)
        if not isinstance(annotations, Mapping):
            return set()
        names: set[str] = set()
        for name, annotation in annotations.items():
            if not isinstance(name, str):
                continue
            if _is_runtime_classvar_annotation(annotation):
                names.add(name)
        return names

    def _class_attribute_names_for_class_key(self, class_key: type | str) -> set[str]:
        names: set[str] = set()
        synthetic_class: SyntheticClassObjectValue | None = None
        if isinstance(class_key, type):
            synthetic_class = self.checker.get_synthetic_class(class_key)
        elif isinstance(class_key, str):
            synthetic_class = self._synthetic_classes_by_name.get(class_key)
        if synthetic_class is not None:
            names.update(
                name
                for name in synthetic_class.class_attributes
                if isinstance(name, str) and not name.startswith("%")
            )
        if isinstance(class_key, type):
            class_dict = safe_getattr(class_key, "__dict__", None)
            if isinstance(class_dict, Mapping):
                names.update(name for name in class_dict if isinstance(name, str))
        return names

    def _is_instance_only_member(self, class_key: type | str, attr_name: str) -> bool:
        return attr_name in self._instance_only_annotation_names_for_class_key(
            class_key, set()
        )

    def _instance_only_annotation_names_for_class_key(
        self, class_key: type | str, seen: set[type | str]
    ) -> set[str]:
        if class_key in seen:
            return set()
        seen.add(class_key)
        names: set[str] = set()
        blocked_names = self._class_attribute_names_for_class_key(class_key)
        synthetic_class: SyntheticClassObjectValue | None = None
        if isinstance(class_key, type):
            synthetic_class = self.checker.get_synthetic_class(class_key)
        elif isinstance(class_key, str):
            synthetic_class = self._synthetic_classes_by_name.get(class_key)
        if synthetic_class is not None:
            names.update(
                _instance_only_names_from_mapping(synthetic_class.class_attributes)
            )
            for base in synthetic_class.base_classes:
                for base_value in flatten_values(base, unwrap_annotated=True):
                    base_key: type | str | None = None
                    if isinstance(base_value, SyntheticClassObjectValue):
                        class_type = base_value.class_type
                        if isinstance(class_type, TypedValue) and isinstance(
                            class_type.typ, (type, str)
                        ):
                            base_key = class_type.typ
                    elif isinstance(base_value, GenericValue) and isinstance(
                        base_value.typ, (type, str)
                    ):
                        base_key = base_value.typ
                    elif isinstance(base_value, TypedValue) and isinstance(
                        base_value.typ, (type, str)
                    ):
                        base_key = base_value.typ
                    elif isinstance(base_value, KnownValue) and isinstance(
                        base_value.val, type
                    ):
                        base_key = base_value.val
                    if base_key is not None:
                        names.update(
                            name
                            for name in (
                                self._instance_only_annotation_names_for_class_key(
                                    base_key, seen
                                )
                            )
                            if name not in blocked_names
                        )
        if isinstance(class_key, type):
            names.update(
                self._runtime_instance_only_annotation_names_for_class(class_key)
            )
            for base_class in self.checker.get_generic_bases(class_key):
                if base_class != class_key:
                    names.update(
                        name
                        for name in self._instance_only_annotation_names_for_class_key(
                            base_class, seen
                        )
                        if name not in blocked_names
                    )
        else:
            for base_class in self.checker.get_generic_bases(class_key):
                if base_class != class_key:
                    names.update(
                        name
                        for name in self._instance_only_annotation_names_for_class_key(
                            base_class, seen
                        )
                        if name not in blocked_names
                    )
        return names

    def _runtime_instance_only_annotation_names_for_class(self, cls: type) -> set[str]:
        annotations = safe_getattr(cls, "__annotations__", None)
        if not isinstance(annotations, Mapping):
            return set()
        class_dict = safe_getattr(cls, "__dict__", None)
        names: set[str] = set()
        for name, annotation in annotations.items():
            if not isinstance(name, str):
                continue
            if _is_runtime_classvar_annotation(annotation):
                continue
            if isinstance(class_dict, Mapping) and name in class_dict:
                continue
            names.add(name)
        return names

    def _contains_classvar_type_parameter(self, value: Value) -> bool:
        for subval in value.walk_values():
            if isinstance(subval, TypeVarValue):
                if subval.typevar is SelfT:
                    continue
                return True
            if isinstance(subval, InputSigValue) and isinstance(
                subval.input_sig, ParamSpecSig
            ):
                return True
        return False

    def _record_synthetic_dataclass_field_metadata(
        self,
        name: str,
        *,
        has_default: bool,
        init: bool,
        initvar: bool,
        kw_only: bool,
        alias: str | None,
    ) -> None:
        synthetic_class = self._get_synthetic_class_for_current_scope()
        if synthetic_class is None:
            return

        existing_order = synthetic_class.class_attributes.get("%dataclass_field_order")
        field_order: list[str] = []
        if isinstance(existing_order, KnownValue) and isinstance(
            existing_order.val, (tuple, list)
        ):
            field_order = [item for item in existing_order.val if isinstance(item, str)]
        if name not in field_order:
            field_order.append(name)

        existing_defaults = synthetic_class.class_attributes.get(
            "%dataclass_default_fields"
        )
        default_fields: set[str] = set()
        if isinstance(existing_defaults, KnownValue) and isinstance(
            existing_defaults.val, (set, frozenset, tuple, list)
        ):
            default_fields.update(
                item for item in existing_defaults.val if isinstance(item, str)
            )
        if has_default:
            default_fields.add(name)
        else:
            default_fields.discard(name)

        existing_init_false = synthetic_class.class_attributes.get(
            "%dataclass_init_false_fields"
        )
        init_false_fields: set[str] = set()
        if isinstance(existing_init_false, KnownValue) and isinstance(
            existing_init_false.val, (set, frozenset, tuple, list)
        ):
            init_false_fields.update(
                item for item in existing_init_false.val if isinstance(item, str)
            )
        if not init:
            init_false_fields.add(name)
        else:
            init_false_fields.discard(name)

        existing_initvar = synthetic_class.class_attributes.get(
            "%dataclass_initvar_fields"
        )
        initvar_fields: set[str] = set()
        if isinstance(existing_initvar, KnownValue) and isinstance(
            existing_initvar.val, (set, frozenset, tuple, list)
        ):
            initvar_fields.update(
                item for item in existing_initvar.val if isinstance(item, str)
            )
        if initvar:
            initvar_fields.add(name)
        else:
            initvar_fields.discard(name)

        existing_kw_only = synthetic_class.class_attributes.get(
            "%dataclass_kw_only_fields"
        )
        kw_only_fields: set[str] = set()
        if isinstance(existing_kw_only, KnownValue) and isinstance(
            existing_kw_only.val, (set, frozenset, tuple, list)
        ):
            kw_only_fields.update(
                item for item in existing_kw_only.val if isinstance(item, str)
            )
        if kw_only:
            kw_only_fields.add(name)
        else:
            kw_only_fields.discard(name)

        existing_aliases = synthetic_class.class_attributes.get(
            "%dataclass_field_aliases"
        )
        aliases: dict[str, str] = {}
        if isinstance(existing_aliases, KnownValue) and isinstance(
            existing_aliases.val, Mapping
        ):
            aliases.update(
                {
                    str(field_name): alias_name
                    for field_name, alias_name in existing_aliases.val.items()
                    if isinstance(field_name, str) and isinstance(alias_name, str)
                }
            )
        if alias is not None:
            aliases[name] = alias
        else:
            aliases.pop(name, None)

        synthetic_class.class_attributes["%dataclass_field_order"] = KnownValue(
            tuple(field_order)
        )
        synthetic_class.class_attributes["%dataclass_default_fields"] = KnownValue(
            frozenset(default_fields)
        )
        synthetic_class.class_attributes["%dataclass_init_false_fields"] = KnownValue(
            frozenset(init_false_fields)
        )
        synthetic_class.class_attributes["%dataclass_initvar_fields"] = KnownValue(
            frozenset(initvar_fields)
        )
        synthetic_class.class_attributes["%dataclass_kw_only_fields"] = KnownValue(
            frozenset(kw_only_fields)
        )
        synthetic_class.class_attributes["%dataclass_field_aliases"] = KnownValue(
            dict(aliases)
        )

    def _get_base_class_attributes(
        self, varname: str, node: ast.AST
    ) -> Iterable[tuple[type | str, Value]]:
        yield from self._get_base_class_attributes_for(
            self.current_class, varname, node
        )

    def _has_base_attribute(self, varname: str, node: ast.AST) -> bool:
        return self._has_base_attribute_for(self.current_class, varname, node)

    def _has_base_attribute_for(
        self, current_class: type | str | None, varname: str, node: ast.AST
    ) -> bool:
        for base_class, base_value in self._get_base_class_attributes_for(
            current_class, varname, node
        ):
            base_value = replace_fallback(base_value)
            if isinstance(base_value, AnyValue):
                if self._base_class_has_any_base(base_class, set()):
                    return True
                continue
            return True
        if current_class is None:
            return False
        return self._base_class_has_any_base(current_class, set())

    def _base_class_has_any_base(
        self, base_class: type | str, seen: set[type | str]
    ) -> bool:
        if isinstance(base_class, type):
            return has_any_base_value(TypedValue(base_class))
        if base_class in seen:
            return False
        seen.add(base_class)

        synthetic_class = self._synthetic_classes_by_name.get(base_class)
        if synthetic_class is not None and any(
            self._base_value_has_any_base(base, seen)
            for base in synthetic_class.base_classes
        ):
            return True

        for ancestor in self.checker.get_generic_bases(base_class):
            if ancestor != base_class and self._base_class_has_any_base(ancestor, seen):
                return True
        return False

    def _base_value_has_any_base(
        self, base_value: Value, seen: set[type | str]
    ) -> bool:
        if has_any_base_value(base_value):
            return True
        base_value = replace_fallback(base_value)
        if isinstance(base_value, SyntheticClassObjectValue):
            class_type = base_value.class_type
            if isinstance(class_type, TypedValue) and isinstance(class_type.typ, str):
                return self._base_class_has_any_base(class_type.typ, seen)
            return False
        if isinstance(base_value, GenericValue):
            if isinstance(base_value.typ, str):
                return self._base_class_has_any_base(base_value.typ, seen)
            return False
        if isinstance(base_value, TypedValue):
            if isinstance(base_value.typ, str):
                return self._base_class_has_any_base(base_value.typ, seen)
        return False

    def _get_base_class_attributes_for(
        self, current_class: type | str | None, varname: str, node: ast.AST
    ) -> Iterable[tuple[type | str, Value]]:
        if current_class is None:
            return
        if isinstance(current_class, str):
            synthetic_class = self._synthetic_classes_by_name.get(current_class)
            if synthetic_class is not None:
                for base in synthetic_class.base_classes:
                    for base_class_value in flatten_values(base, unwrap_annotated=True):
                        if isinstance(base_class_value, SyntheticClassObjectValue):
                            base_class: type | str = base_class_value.class_type.typ
                            root_value: Value = base_class_value
                        elif isinstance(base_class_value, TypedValue):
                            base_class = base_class_value.typ
                            root_value = (
                                self._synthetic_classes_by_name.get(
                                    base_class, base_class_value
                                )
                                if isinstance(base_class, str)
                                else TypedValue(base_class)
                            )
                        elif isinstance(base_class_value, KnownValue) and isinstance(
                            base_class_value.val, type
                        ):
                            base_class = base_class_value.val
                            root_value = TypedValue(base_class)
                        else:
                            continue
                        if base_class == current_class:
                            continue
                        ctx = _AttrContext(
                            Composite(root_value),
                            varname,
                            self,
                            node=node,
                            skip_mro=True,
                            skip_unwrap=True,
                            record_reads=False,
                        )
                        base_value = attributes.get_attribute(ctx)
                        if base_value is not UNINITIALIZED_VALUE:
                            yield base_class, base_value
        for base_class in self.checker.get_generic_bases(current_class):
            if base_class == current_class:
                continue
            if isinstance(base_class, str):
                base_class_value: Value = self._synthetic_classes_by_name.get(
                    base_class, TypedValue(base_class)
                )
            else:
                base_class_value = TypedValue(base_class)
            ctx = _AttrContext(
                Composite(base_class_value),
                varname,
                self,
                node=node,
                skip_mro=True,
                skip_unwrap=True,
                record_reads=False,
            )
            base_value = attributes.get_attribute(ctx)
            if base_value is not UNINITIALIZED_VALUE:
                yield base_class, base_value

    def _check_for_incompatible_overrides(
        self, varname: str, node: ast.AST, value: Value
    ) -> None:
        if self.current_class is None:
            return
        if self._is_current_class_dataclass() and varname == "__post_init__":
            # Dataclasses synthesize the expected __post_init__ contract from InitVar
            # fields, so generic override rules are too strict here.
            return
        if varname in self.options.get_value_for(IgnoredForIncompatibleOverride):
            return
        if varname.startswith("__") and not varname.endswith("__"):
            return
        is_annotated_assignment = (
            isinstance(self.current_statement, ast.AnnAssign)
            and isinstance(self.current_statement.target, ast.Name)
            and self.current_statement.target.id == varname
        )
        base_attributes = list(self._get_base_class_attributes(varname, node))
        child_is_classvar = is_annotated_assignment and self._is_direct_classvar_member(
            self.current_class, varname
        )
        saw_final_member = False
        for base_class, base_value in base_attributes:
            if self._is_final_member(base_class, varname, base_value):
                saw_final_member = True
                self._show_error_if_checking(
                    node,
                    f"Cannot override final attribute {varname}",
                    ErrorCode.invalid_annotation,
                )
        if saw_final_member:
            return
        for base_class, base_value in base_attributes:
            if is_annotated_assignment:
                base_is_classvar = self._is_direct_classvar_member(base_class, varname)
                if child_is_classvar and not base_is_classvar:
                    self._show_error_if_checking(
                        node,
                        f"Class variable {varname} cannot override instance variable from"
                        f" base class {base_class}",
                        ErrorCode.incompatible_override,
                    )
                    continue
                if base_is_classvar and not child_is_classvar:
                    self._show_error_if_checking(
                        node,
                        f"Instance variable {varname} cannot override class variable from"
                        f" base class {base_class}",
                        ErrorCode.incompatible_override,
                    )
                    continue
            can_assign = self._can_assign_to_base(base_value, value, base_class, node)
            if isinstance(can_assign, CanAssignError):
                error = CanAssignError(
                    children=[
                        CanAssignError(f"Base class: {self.display_value(base_value)}"),
                        CanAssignError(f"Child class: {self.display_value(value)}"),
                        can_assign,
                    ]
                )
                self._show_error_if_checking(
                    node,
                    f"Value of {varname} incompatible with base class {base_class}",
                    ErrorCode.incompatible_override,
                    detail=str(error),
                )

    def _base_class_key_from_value(self, base_value: Value) -> type | str | None:
        base_value = replace_fallback(base_value)
        if isinstance(base_value, AnnotatedValue):
            return self._base_class_key_from_value(base_value.value)
        if isinstance(base_value, SubclassValue) and isinstance(
            base_value.typ, TypedValue
        ):
            if isinstance(base_value.typ.typ, (type, str)):
                return base_value.typ.typ
            return None
        if isinstance(base_value, SyntheticClassObjectValue):
            class_type = base_value.class_type
            if isinstance(class_type, TypedValue):
                return class_type.typ
            return None
        if isinstance(base_value, GenericValue):
            if isinstance(base_value.typ, (type, str)):
                return base_value.typ
            return None
        if isinstance(base_value, TypedValue):
            if isinstance(base_value.typ, (type, str)):
                return base_value.typ
            return None
        if isinstance(base_value, KnownValue):
            if isinstance(base_value.val, type):
                return base_value.val
            return None
        return None

    def display_value(self, value: Value) -> str:
        return self.checker.display_value(value)

    def resolve_property(
        self, obj: property, root_composite: Composite, node: ast.AST | None
    ) -> Value:
        if obj.fget is None:
            return UNINITIALIZED_VALUE

        getter = set_self(KnownValue(obj.fget), root_composite.value)
        return self.check_call(node, getter, [root_composite])

    def _can_assign_to_base(
        self,
        base_value: Value,
        child_value: Value,
        base_class: type | str,
        node: ast.AST,
    ) -> CanAssign:
        if base_value is UNINITIALIZED_VALUE:
            return {}
        if isinstance(base_value, KnownValue):
            if isinstance(base_value.val, property):
                return self._can_assign_to_base_property(
                    base_value.val, child_value, base_class, node
                )
            if callable(base_value.val):
                callable_result = self._can_assign_to_base_callable(
                    base_value, child_value
                )
                if callable_result is None:
                    return {}
                return callable_result
        if isinstance(base_value, CallableValue):
            callable_result = self._can_assign_to_base_callable(base_value, child_value)
            if callable_result is not None:
                return callable_result
        return has_relation(base_value, child_value, Relation.ASSIGNABLE, self)

    def _can_assign_to_base_property(
        self,
        base_property: property,
        child_value: Value,
        base_class: type | str,
        node: ast.AST,
    ) -> CanAssign:
        if isinstance(child_value, KnownValue) and isinstance(
            child_value.val, property
        ):
            if base_property.fset is not None and child_value.val.fset is None:
                return CanAssignError(
                    "Property is settable on base class but not on child class"
                )
            if base_property.fdel is not None and child_value.val.fdel is None:
                return CanAssignError(
                    "Property is settable on base class but not on child class"
                )
            assert self.current_class is not None
            child_value = self.resolve_property(
                child_value.val, Composite(TypedValue(self.current_class)), node
            )
        base_value = self.resolve_property(
            base_property, Composite(TypedValue(base_class)), node
        )
        get_direction = has_relation(base_value, child_value, Relation.ASSIGNABLE, self)
        if isinstance(get_direction, CanAssignError):
            return get_direction
        if base_property.fset is not None:
            # settable properties behave invariantly, so we need to check both directions
            return has_relation(child_value, base_value, Relation.ASSIGNABLE, self)
        else:
            return get_direction

    def _can_assign_to_base_callable(
        self, base_value: Value, child_value: Value
    ) -> CanAssign | None:
        base_sig = self.signature_from_value(base_value)
        if base_sig is ANY_SIGNATURE:
            return None
        if not isinstance(base_sig, (Signature, OverloadedSignature)):
            return None
        child_sig = self.signature_from_value(child_value)
        if child_sig is ANY_SIGNATURE:
            return None
        if not isinstance(child_sig, (Signature, OverloadedSignature)):
            return CanAssignError(f"{child_value} is not callable")
        base_bound = base_sig.bind_self(ctx=self)
        if base_bound is None:
            return None
        child_bound = child_sig.bind_self(ctx=self)
        if child_bound is None:
            return CanAssignError(f"{child_value} is missing a receiver argument")
        return base_bound.can_assign(child_bound, self)

    def _check_for_class_variable_redefinition(
        self, varname: str, node: ast.AST
    ) -> None:
        current_scope = self.scopes.current_scope()
        if varname not in current_scope.variables:
            return

        # Exclude cases where we do @<property>.setter. During synthetic class
        # analysis current_class can be a string, so inspect the existing scope
        # value directly rather than accessing current_class.__dict__.
        existing_value = replace_fallback(current_scope.variables[varname])
        for subval in flatten_values(existing_value):
            if isinstance(subval, KnownValue) and isinstance(subval.val, property):
                return
            if (
                isinstance(subval, (GenericValue, TypedValue))
                and subval.typ is property
            ):
                return

        # allow augmenting an attribute
        if isinstance(self.current_statement, ast.AugAssign):
            return

        if isinstance(self.current_statement, ast.ClassDef):
            for subval in flatten_values(existing_value):
                if (
                    isinstance(subval, SyntheticClassObjectValue)
                    and subval.name == self.current_statement.name
                ):
                    return

        self.show_error(
            node,
            f"Name {varname} is already defined",
            error_code=ErrorCode.class_variable_redefinition,
        )

    def resolve_name(
        self,
        node: ast.Name,
        error_node: ast.AST | None = None,
        suppress_errors: bool = False,
    ) -> tuple[Value, VarnameOrigin]:
        """Resolves a Name node to a value.

        :param node: Node to resolve the name from
        :type node: ast.AST

        :param error_node: If given, this AST node is used instead of `node`
                           for displaying errors.
        :type error_node: Optional[ast.AST]

        :param suppress_errors: If True, do not produce errors if the name is
                                undefined.
        :type suppress_errors: bool

        """
        if error_node is None:
            error_node = node
        value, defining_scope, origin = self.scopes.get_with_scope(
            node.id, node, self.state, can_assign_ctx=self
        )
        if defining_scope is not None:
            if defining_scope.scope_type in (
                ScopeType.module_scope,
                ScopeType.class_scope,
            ):
                if defining_scope.scope_object is not None:
                    self._maybe_record_usage(
                        defining_scope.scope_object, node.id, value
                    )
            if self.in_annotation:
                declared_type = defining_scope.get_declared_type(node.id)
                if isinstance(declared_type, TypeAliasValue):
                    value = declared_type
        if value is UNINITIALIZED_VALUE:
            if suppress_errors or node.id in self.options.get_value_for(ExtraBuiltins):
                self.log(logging.INFO, "ignoring undefined name", node.id)
            else:
                self._show_error_if_checking(
                    error_node, f"Undefined name: {node.id}", ErrorCode.undefined_name
                )
            return AnyValue(AnySource.error), origin
        if isinstance(value, InputSigValue):
            # ParamSpecs are stored as InputSigValues and cannot be converted to a
            # gradual fallback.
            value_for_subvals = value
        else:
            value_for_subvals = replace_fallback(value)
        if isinstance(value_for_subvals, MultiValuedValue):
            subvals = value_for_subvals.vals
        else:
            subvals = None

        if subvals is not None:
            if any(subval is UNINITIALIZED_VALUE for subval in subvals):
                self._show_error_if_checking(
                    error_node,
                    f"{node.id} may be used uninitialized",
                    ErrorCode.possibly_undefined_name,
                )
                new_mvv = MultiValuedValue(
                    [
                        (
                            AnyValue(AnySource.error)
                            if subval is UNINITIALIZED_VALUE
                            else subval
                        )
                        for subval in subvals
                    ]
                )
                if isinstance(value, AnnotatedValue):
                    return AnnotatedValue(new_mvv, value.metadata), origin
                else:
                    return new_mvv, origin
        return value, origin

    def _get_first_import_node(self) -> ast.stmt:
        return min(self.import_name_to_node.values(), key=lambda node: node.lineno)

    def _generic_visit_list(self, lst: Iterable[ast.AST]) -> list[Value]:
        return [self.visit(node) for node in lst]

    def _is_write_ctx(self, ctx: ast.AST) -> bool:
        return isinstance(ctx, (ast.Store, ast.Param))

    def _is_read_ctx(self, ctx: ast.AST) -> bool:
        return isinstance(ctx, (ast.Load, ast.Del))

    def _is_enum_class_key(self, class_key: type | str | None) -> bool:
        if class_key is None:
            return False
        if class_key in self.enum_class_keys:
            return True
        if isinstance(class_key, type):
            return safe_issubclass(class_key, enum.Enum)
        return False

    @contextlib.contextmanager
    def _set_current_class(self, current_class: type | str | None) -> Generator[None]:
        should_track_members = should_check_for_duplicate_values(
            current_class, self.options
        )
        if not should_track_members and self._is_enum_class_key(current_class):
            should_track_members = True
        if should_track_members:
            current_enum_members = _EnumMemberTracker()
        else:
            current_enum_members = None
        with (
            override(self, "current_class", current_class),
            override(self.asynq_checker, "current_class", current_class),
            override(self, "current_enum_members", current_enum_members),
        ):
            yield

    def visit_decorator_list(self, decorators: list[ast.expr]) -> DecoratorValues:
        result = []
        for decorator in decorators:
            if isinstance(decorator, ast.Call):
                callee = self.visit(decorator.func)
                value = self.visit_Call(decorator, callee=callee)
                if self.annotate:
                    decorator.inferred_value = value
            else:
                callee = value = self.visit(decorator)
            result.append((callee, value, decorator))
        return result

    def visit_ClassDef(self, node: ast.ClassDef) -> Value:
        decorator_values = self.visit_decorator_list(node.decorator_list)
        class_obj = self._get_current_class_object(node)
        class_key: type | str = (
            class_obj
            if class_obj is not None
            else self._get_synthetic_class_fq_name(node)
        )
        self.enum_value_type_by_class.pop(class_key, None)
        if any(
            self._is_final_decorator_value(value) for _, value, _ in decorator_values
        ):
            self.final_class_keys.add(class_key)
        if sys.version_info >= (3, 12) and node.type_params:
            ctx = self.scopes.add_scope(
                ScopeType.annotation_scope, scope_node=node, scope_object=class_obj
            )
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            if sys.version_info >= (3, 12) and node.type_params:
                declared_type_params = node.type_params
                type_param_values = list(
                    self.visit_type_param_values(declared_type_params)
                )
            else:
                type_param_values = []
                declared_type_params = []
            if self._is_checking() and type_param_values:
                legacy_typevars = self._legacy_typevars_in_nodes(
                    [
                        *declared_type_params,
                        *node.bases,
                        *(kw.value for kw in node.keywords),
                    ],
                    type_param_values,
                )
                if legacy_typevars:
                    self._show_error_if_checking(
                        node,
                        "Class definition cannot combine old-style TypeVar declarations"
                        " with type parameter syntax",
                        error_code=ErrorCode.invalid_annotation,
                    )
            base_values = self._generic_visit_list(node.bases)
            if self._is_checking():
                self._check_typevartuple_usage_in_type_parameter_bases(
                    node, base_values
                )
                if sys.version_info >= (3, 12) and declared_type_params:
                    self._check_pep695_type_parameter_base_compatibility(
                        node, base_values
                    )
                self._check_duplicate_type_params_in_generic_bases(node, base_values)
                self._check_inconsistent_generic_base_specialization(node, base_values)
                self._check_protocol_base_validity(node, base_values)
                for base_node, base_value in zip(node.bases, base_values):
                    if _is_type_alias_base_value(base_value):
                        self._show_error_if_checking(
                            base_node,
                            "Type aliases cannot be used as base classes",
                            error_code=ErrorCode.invalid_base,
                        )
                    if _is_newtype_base_value(base_value):
                        self._show_error_if_checking(
                            base_node,
                            "NewType types cannot be used as base classes",
                            error_code=ErrorCode.invalid_base,
                        )
            if any(
                is_subtype(SubclassValue(TypedValue(enum.Enum)), base, self)
                for base in base_values
            ):
                self.enum_class_keys.add(class_key)
            self._check_for_final_base_classes(node, base_values)
            keyword_values = [(kw, self.visit(kw.value)) for kw in node.keywords]
            dataclass_semantics = self._get_class_dataclass_semantics(
                node,
                class_obj=class_obj,
                base_values=base_values,
                keyword_values=keyword_values,
                decorator_values=decorator_values,
            )
            direct_dataclass_transform_info = self._get_direct_dataclass_transform_info(
                decorator_values
            )
            dataclass_transform_info = self._get_class_transform_provider_info(
                class_obj=class_obj,
                base_values=base_values,
                keyword_values=keyword_values,
                direct_dataclass_transform_info=direct_dataclass_transform_info,
            )
            if self._is_checking() and dataclass_semantics is not None:
                self._check_dataclass_inheritance(
                    node, base_values, dataclass_semantics.frozen
                )
                self._check_dataclass_slots_definition(node, dataclass_semantics)
            synthetic_typeddict = self._make_synthetic_typeddict_context(
                node, base_values, keyword_values
            )
            fallback_runtime_class: type | None = None
            synthetic_enum_runtime_class: type | None = None
            if (
                class_obj is None
                and synthetic_typeddict is None
                and self._is_checking()
                and not node.keywords
            ):
                fallback_runtime_class = self._make_dataclass_related_fallback_class(
                    node, base_values, decorator_values
                )
                if fallback_runtime_class is None:
                    fallback_runtime_class = (
                        self._make_namedtuple_related_fallback_class(node, base_values)
                    )
                if fallback_runtime_class is None:
                    fallback_runtime_class = self._make_enum_related_fallback_class(
                        node, base_values
                    )
                if fallback_runtime_class is not None:
                    if safe_issubclass(fallback_runtime_class, enum.Enum):
                        synthetic_enum_runtime_class = fallback_runtime_class
                    else:
                        class_obj = fallback_runtime_class
            synthetic_class = None
            synthetic_fq_name: str | None = None
            class_scope_object: type | str | None = class_obj
            dataclass_metadata_class: SyntheticClassObjectValue | None = None
            dataclass_check_class: SyntheticClassObjectValue | None = None
            synthetic_base_values = tuple(
                self._base_values_for_generic_analysis(node, base_values)
            ) or (KnownValue(object),)
            if synthetic_typeddict is not None:
                self._validate_typeddict_class_syntax(node)
            elif class_obj is None:
                synthetic_fq_name = self._get_synthetic_class_fq_name(node)
                if synthetic_enum_runtime_class is not None:
                    synthetic_class_type: TypedValue = TypedValue(
                        synthetic_enum_runtime_class
                    )
                    class_scope_object = synthetic_enum_runtime_class
                else:
                    synthetic_class_type = TypedValue(synthetic_fq_name)
                    class_scope_object = synthetic_fq_name
                synthetic_class = SyntheticClassObjectValue(
                    node.name,
                    synthetic_class_type,
                    base_classes=synthetic_base_values,
                    is_dataclass=dataclass_semantics is not None,
                    dataclass_frozen=(
                        dataclass_semantics.frozen
                        if dataclass_semantics is not None
                        else None
                    ),
                    dataclass_order=(
                        dataclass_semantics.order
                        if dataclass_semantics is not None
                        else None
                    ),
                )
                if dataclass_transform_info is not None:
                    synthetic_class.class_attributes["%dataclass_transform"] = (
                        KnownValue(True)
                    )
                    synthetic_class.class_attributes[
                        "%dataclass_transform_eq_default"
                    ] = KnownValue(dataclass_transform_info.eq_default)
                    synthetic_class.class_attributes[
                        "%dataclass_transform_frozen_default"
                    ] = KnownValue(dataclass_transform_info.frozen_default)
                    synthetic_class.class_attributes[
                        "%dataclass_transform_kw_only_default"
                    ] = KnownValue(dataclass_transform_info.kw_only_default)
                    synthetic_class.class_attributes[
                        "%dataclass_transform_order_default"
                    ] = KnownValue(dataclass_transform_info.order_default)
                    synthetic_class.class_attributes[
                        "%dataclass_transform_field_specifiers"
                    ] = KnownValue(tuple(dataclass_transform_info.field_specifiers))
                synthetic_methods = self._get_synthetic_method_attributes(node)
                synthetic_class.method_attributes.update(synthetic_methods)
                for method_name in synthetic_methods:
                    synthetic_class.class_attributes.setdefault(
                        method_name, AnyValue(AnySource.from_another)
                    )
                _record_dataclass_slots_flag(synthetic_class, dataclass_semantics)
                self._synthetic_classes_by_name[synthetic_fq_name] = synthetic_class
                self.checker.register_synthetic_class(synthetic_class)
                dataclass_metadata_class = synthetic_class
                if self._is_checking():
                    self._synthetic_abstract_methods[synthetic_fq_name] = set()
                    # Bind the class name while checking its body so references
                    # like "return C.attr" or string annotations mentioning C
                    # resolve even when no runtime class object exists.
                    self.scopes.set(node.name, synthetic_class, node, self.state)
            elif (
                dataclass_semantics is not None or dataclass_transform_info is not None
            ):
                existing = self.checker.get_synthetic_class(class_obj)
                if existing is None:
                    existing = SyntheticClassObjectValue(
                        node.name,
                        TypedValue(class_obj),
                        base_classes=synthetic_base_values,
                        is_dataclass=dataclass_semantics is not None,
                        dataclass_frozen=(
                            dataclass_semantics.frozen
                            if dataclass_semantics is not None
                            else None
                        ),
                        dataclass_order=(
                            dataclass_semantics.order
                            if dataclass_semantics is not None
                            else None
                        ),
                    )
                    self.checker.register_synthetic_class(existing)
                else:
                    existing = replace(
                        existing,
                        base_classes=synthetic_base_values,
                        is_dataclass=dataclass_semantics is not None,
                        dataclass_frozen=(
                            dataclass_semantics.frozen
                            if dataclass_semantics is not None
                            else None
                        ),
                        dataclass_order=(
                            dataclass_semantics.order
                            if dataclass_semantics is not None
                            else None
                        ),
                    )
                    self.checker.register_synthetic_class(existing)
                if dataclass_transform_info is not None:
                    existing.class_attributes["%dataclass_transform"] = KnownValue(True)
                    existing.class_attributes["%dataclass_transform_eq_default"] = (
                        KnownValue(dataclass_transform_info.eq_default)
                    )
                    existing.class_attributes["%dataclass_transform_frozen_default"] = (
                        KnownValue(dataclass_transform_info.frozen_default)
                    )
                    existing.class_attributes[
                        "%dataclass_transform_kw_only_default"
                    ] = KnownValue(dataclass_transform_info.kw_only_default)
                    existing.class_attributes["%dataclass_transform_order_default"] = (
                        KnownValue(dataclass_transform_info.order_default)
                    )
                    existing.class_attributes[
                        "%dataclass_transform_field_specifiers"
                    ] = KnownValue(tuple(dataclass_transform_info.field_specifiers))
                else:
                    existing.class_attributes.pop("%dataclass_transform", None)
                    existing.class_attributes.pop(
                        "%dataclass_transform_eq_default", None
                    )
                    existing.class_attributes.pop(
                        "%dataclass_transform_frozen_default", None
                    )
                    existing.class_attributes.pop(
                        "%dataclass_transform_kw_only_default", None
                    )
                    existing.class_attributes.pop(
                        "%dataclass_transform_order_default", None
                    )
                    existing.class_attributes.pop(
                        "%dataclass_transform_field_specifiers", None
                    )
                _record_dataclass_slots_flag(existing, dataclass_semantics)
                dataclass_metadata_class = existing
            generic_class_key = (
                class_scope_object
                if isinstance(class_scope_object, (type, str))
                else class_key
            )
            runtime_class_for_type_params = (
                class_scope_object if isinstance(class_scope_object, type) else None
            )
            is_protocol_class = self._is_protocol_class(base_values, class_scope_object)
            effective_type_param_values = (
                type_param_values
                if type_param_values
                else self._type_params_from_base_values(base_values)
            )
            analyzed_base_values = self._base_values_for_generic_analysis(
                node, base_values
            )
            if (
                not type_param_values
                and self.module is None
                and not effective_type_param_values
                and analyzed_base_values is not base_values
            ):
                effective_type_param_values = (
                    self._order_type_params_by_base_annotation_appearance(
                        node.bases,
                        self._type_params_from_base_values(analyzed_base_values),
                    )
                )
            if not effective_type_param_values and is_protocol_class:
                # Runtime protocol classes often expose no generic parameters;
                # recover class type parameters from protocol bases.
                protocol_type_params = self._type_params_from_base_values_for_methods(
                    base_values
                )
                if protocol_type_params and all(
                    not (
                        is_instance_of_typing_name(type_param.typevar, "ParamSpec")
                        or type_param.is_typevartuple
                    )
                    for type_param in protocol_type_params
                ):
                    effective_type_param_values = protocol_type_params
            if not effective_type_param_values and self.module is None:
                # In static-fallback mode we can lose Generic[...] type arguments
                # from base values; recover from base annotation expressions.
                recovered_type_param_values = (
                    self._type_params_from_base_annotations_for_default_rules(
                        node.bases
                    )
                )
                if any(
                    is_instance_of_typing_name(type_param.typevar, "ParamSpec")
                    or type_param.is_typevartuple
                    for type_param in recovered_type_param_values
                ):
                    effective_type_param_values = recovered_type_param_values
            registered_type_param_values = self._align_type_params_with_runtime_class(
                runtime_class_for_type_params, effective_type_param_values
            )
            has_explicit_constructor = any(
                isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
                and statement.name in {"__init__", "__new__"}
                for statement in node.body
            )
            if (
                not registered_type_param_values
                and not type_param_values
                and self.module is None
                and dataclass_semantics is None
                and not has_explicit_constructor
            ):
                recovered_registration_type_params = (
                    self._type_params_from_base_annotations_for_default_rules(
                        node.bases
                    )
                )
                if recovered_registration_type_params:
                    registered_type_param_values = (
                        self._align_type_params_with_runtime_class(
                            runtime_class_for_type_params,
                            self._order_type_params_by_base_annotation_appearance(
                                node.bases, recovered_registration_type_params
                            ),
                        )
                    )
            if (
                registered_type_param_values
                and not type_param_values
                and self.module is None
                and dataclass_semantics is None
                and not has_explicit_constructor
            ):
                registered_type_param_values = (
                    self._order_type_params_by_base_annotation_appearance(
                        node.bases, registered_type_param_values
                    )
                )
            method_type_params = (
                type_param_values
                if type_param_values
                else self._type_params_from_base_values_for_methods(base_values)
            )
            if (
                not type_param_values
                and self.module is None
                and analyzed_base_values is not base_values
            ):
                method_type_params = (
                    self._order_type_params_by_base_annotation_appearance(
                        node.bases,
                        self._type_params_from_base_values_for_methods(
                            analyzed_base_values
                        ),
                    )
                )
            if (
                not method_type_params
                and effective_type_param_values
                and any(
                    is_instance_of_typing_name(type_param.typevar, "ParamSpec")
                    or type_param.is_typevartuple
                    for type_param in effective_type_param_values
                )
            ):
                method_type_params = effective_type_param_values
            method_type_param_values = self._align_type_params_with_runtime_class(
                runtime_class_for_type_params, method_type_params
            )
            should_register_generic_bases = synthetic_typeddict is None and (
                isinstance(generic_class_key, str) or bool(registered_type_param_values)
            )
            base_values_for_registration = (
                base_values
                if (base_values or class_obj is not None)
                else [KnownValue(object)]
            )
            if self.module is None and node.bases:
                # In static fallback mode, runtime subscripting of class bases can lose
                # generic information. Recover base values from annotation syntax.
                base_values_for_registration = list(
                    self._base_values_for_generic_analysis(node, base_values)
                )
            if should_register_generic_bases:
                self.checker.register_synthetic_type_bases(
                    generic_class_key,
                    base_values_for_registration,
                    declared_type_params=registered_type_param_values,
                )
            with (
                self._active_pep695_type_param_scope(type_param_values),
                override(self, "current_synthetic_typeddict", synthetic_typeddict),
                override(self, "current_class_key", class_key),
                override(self, "current_dataclass_info", dataclass_semantics),
                override(
                    self, "current_class_type_params", tuple(method_type_param_values)
                ),
            ):
                value, class_scope_values = self._visit_class_and_get_value(
                    node, class_scope_object
                )
            if type_param_values and synthetic_typeddict is None:
                inferred_type_params = self._infer_class_type_param_variances(
                    node,
                    type_param_values,
                    base_values,
                    dataclass_semantics=dataclass_semantics,
                )
                registered_inferred_type_params = (
                    self._align_type_params_with_runtime_class(
                        runtime_class_for_type_params, inferred_type_params
                    )
                )
                self.checker.register_synthetic_type_bases(
                    generic_class_key,
                    base_values_for_registration,
                    declared_type_params=registered_inferred_type_params,
                )
            if self._is_checking() and synthetic_typeddict is None:
                declared_type_params = (
                    type_param_values
                    if type_param_values
                    else effective_type_param_values
                )
                if not declared_type_params:
                    declared_type_params = (
                        self._type_params_from_base_annotations_for_default_rules(
                            node.bases
                        )
                    )
                self._check_class_type_param_default_rules(node, declared_type_params)
                self._check_class_base_type_param_variances(
                    node, declared_type_params, base_values, class_scope_object
                )
                self._check_protocol_type_param_variances(
                    node, declared_type_params, base_values, class_scope_object
                )
            if fallback_runtime_class is not None and class_scope_values is not None:
                _populate_fallback_runtime_class(
                    fallback_runtime_class, class_scope_values
                )
            metaclass_value = next(
                (
                    value
                    for keyword, value in keyword_values
                    if keyword.arg == "metaclass"
                ),
                None,
            )
            if synthetic_typeddict is not None:
                typeddict_value = self._build_synthetic_typeddict_value(
                    synthetic_typeddict, node
                )
                if class_obj is None:
                    value = SyntheticClassObjectValue(
                        node.name, typeddict_value, base_classes=tuple(base_values)
                    )
                    self.checker.register_synthetic_class(value)
            elif synthetic_class is not None:
                if class_scope_values is None:
                    value = synthetic_class
                else:
                    class_attributes = {
                        name: value
                        for name, value in class_scope_values.items()
                        if not name.startswith("%")
                    }
                    self_returning_classmethods = (
                        self._get_synthetic_self_returning_classmethods(node)
                    )
                    classmethods = self._get_synthetic_classmethod_attributes(node)
                    # Keep the prebound synthetic class object and fill in the
                    # discovered class attributes on that same object so
                    # references captured during class analysis (including bases
                    # of later synthetic classes) see the enriched attributes.
                    synthetic_class.class_attributes.update(class_attributes)
                    if self_returning_classmethods:
                        synthetic_class.class_attributes["%self_classmethods"] = (
                            KnownValue(frozenset(self_returning_classmethods))
                        )
                    else:
                        synthetic_class.class_attributes.pop("%self_classmethods", None)
                    if classmethods:
                        synthetic_class.class_attributes["%classmethods"] = KnownValue(
                            frozenset(classmethods)
                        )
                    else:
                        synthetic_class.class_attributes.pop("%classmethods", None)
                    staticmethods = self._get_synthetic_staticmethod_attributes(node)
                    if staticmethods:
                        synthetic_class.class_attributes["%staticmethods"] = KnownValue(
                            frozenset(staticmethods)
                        )
                    else:
                        synthetic_class.class_attributes.pop("%staticmethods", None)
                    if isinstance(metaclass_value, Value):
                        synthetic_class.class_attributes["%metaclass"] = metaclass_value
                    synthetic_class.method_attributes.clear()
                    synthetic_class.method_attributes.update(
                        self._get_synthetic_method_attributes(node)
                    )
                    self._apply_dataclass_slots_semantics(
                        synthetic_class, dataclass_semantics
                    )
                    self._apply_dataclass_hash_semantics(
                        synthetic_class, dataclass_semantics
                    )
                    self._apply_synthetic_enum_semantics(
                        node, synthetic_class, class_key
                    )
                    self.checker.register_synthetic_class(synthetic_class)
                    value = synthetic_class
                if dataclass_semantics is not None:
                    dataclass_check_class = synthetic_class
                if isinstance(metaclass_value, Value):
                    synthetic_class.class_attributes["%metaclass"] = metaclass_value
                if synthetic_fq_name is not None:
                    self._synthetic_classes_by_name[synthetic_fq_name] = synthetic_class
            elif (
                dataclass_metadata_class is not None
                and class_scope_values is not None
                and class_obj is not None
            ):
                class_attributes = {
                    name: value
                    for name, value in class_scope_values.items()
                    if not name.startswith("%")
                }
                self_returning_classmethods = (
                    self._get_synthetic_self_returning_classmethods(node)
                )
                classmethods = self._get_synthetic_classmethod_attributes(node)
                dataclass_metadata_class.class_attributes.update(class_attributes)
                if self_returning_classmethods:
                    dataclass_metadata_class.class_attributes["%self_classmethods"] = (
                        KnownValue(frozenset(self_returning_classmethods))
                    )
                else:
                    dataclass_metadata_class.class_attributes.pop(
                        "%self_classmethods", None
                    )
                if classmethods:
                    dataclass_metadata_class.class_attributes["%classmethods"] = (
                        KnownValue(frozenset(classmethods))
                    )
                else:
                    dataclass_metadata_class.class_attributes.pop("%classmethods", None)
                staticmethods = self._get_synthetic_staticmethod_attributes(node)
                if staticmethods:
                    dataclass_metadata_class.class_attributes["%staticmethods"] = (
                        KnownValue(frozenset(staticmethods))
                    )
                else:
                    dataclass_metadata_class.class_attributes.pop(
                        "%staticmethods", None
                    )
                if isinstance(metaclass_value, Value):
                    dataclass_metadata_class.class_attributes["%metaclass"] = (
                        metaclass_value
                    )
                dataclass_metadata_class.method_attributes.clear()
                dataclass_metadata_class.method_attributes.update(
                    self._get_synthetic_method_attributes(node)
                )
                self._apply_dataclass_slots_semantics(
                    dataclass_metadata_class, dataclass_semantics
                )
                self._apply_dataclass_hash_semantics(
                    dataclass_metadata_class, dataclass_semantics
                )
                self.checker.register_synthetic_class(dataclass_metadata_class)
                if dataclass_semantics is not None:
                    dataclass_check_class = dataclass_metadata_class
            if (
                self._is_checking()
                and dataclass_semantics is not None
                and dataclass_check_class is not None
            ):
                self._check_dataclass_field_default_order(
                    node, dataclass_check_class, decorator_values
                )
                self._check_dataclass_post_init_signature(node, dataclass_check_class)
        value_to_store: Value | None = value
        if (
            class_obj is None
            and self._is_collecting()
            and self.scopes.scope_type() == ScopeType.module_scope
            and synthetic_typeddict is None
        ):
            # In the collect phase we don't always visit top-level class bodies.
            # Avoid storing placeholder values that later get unioned with the
            # check-phase result.
            value_to_store = None
        value, _ = self._set_name_in_scope(node.name, node, value_to_store)
        self._finalize_synthetic_abstract_members(
            node, class_key, is_protocol_class=is_protocol_class
        )
        self._check_for_uninitialized_final_members(class_key)
        return value

    def _make_synthetic_typeddict_context(
        self,
        node: ast.ClassDef,
        base_values: Sequence[Value],
        keyword_values: Sequence[tuple[ast.keyword, Value]],
    ) -> _SyntheticTypedDictContext | None:
        has_typeddict_base = False
        bases = []
        invalid_non_typeddict_bases = []
        for base_node, base_value in zip(node.bases, base_values):
            if _is_typeddict_marker_base(base_value):
                has_typeddict_base = True
                continue
            base_typed_dict = self._typed_dict_base_value(base_value, base_node)
            if base_typed_dict is not None:
                has_typeddict_base = True
                bases.append(base_typed_dict)
                continue
            if self._is_typeddict_generic_base(base_value):
                continue
            invalid_non_typeddict_bases.append(base_node)
        if not has_typeddict_base:
            return None
        for base_node in invalid_non_typeddict_bases:
            self._show_error_if_checking(
                base_node,
                "TypedDict classes may only inherit from TypedDict types and Generic",
                error_code=ErrorCode.invalid_base,
            )
        inherited_extra_keys: Value | None = None
        inherited_extra_keys_readonly = False
        for base in bases:
            if base.extra_keys is None:
                continue
            inherited_extra_keys = base.extra_keys
            inherited_extra_keys_readonly = base.extra_keys_readonly
            break
        total = True
        closed: bool | None = None
        explicit_extra_keys: Value | None = None
        explicit_extra_keys_readonly = False
        for keyword, _ in keyword_values:
            if keyword.arg == "total":
                bool_value = _get_bool_literal(keyword.value)
                if bool_value is None:
                    self._show_error_if_checking(
                        keyword.value,
                        "TypedDict total= argument must be a bool literal",
                        error_code=ErrorCode.invalid_annotation,
                    )
                else:
                    total = bool_value
                continue
            if keyword.arg == "closed":
                bool_value = _get_bool_literal(keyword.value)
                if bool_value is None:
                    self._show_error_if_checking(
                        keyword.value,
                        'Argument to "closed" must be a literal True or False',
                        error_code=ErrorCode.invalid_annotation,
                    )
                else:
                    closed = bool_value
                continue
            if keyword.arg == "extra_items":
                extra_items_expr = annotation_expr_from_ast(keyword.value, visitor=self)
                extra_items_value, qualifiers = extra_items_expr.unqualify(
                    {Qualifier.ReadOnly, Qualifier.Required, Qualifier.NotRequired},
                    mutually_exclusive_qualifiers=(
                        (Qualifier.Required, Qualifier.NotRequired),
                    ),
                )
                if Qualifier.Required in qualifiers:
                    self._show_error_if_checking(
                        keyword.value,
                        "'extra_items' value cannot be 'Required[...]'",
                        error_code=ErrorCode.invalid_annotation,
                    )
                if Qualifier.NotRequired in qualifiers:
                    self._show_error_if_checking(
                        keyword.value,
                        "'extra_items' value cannot be 'NotRequired[...]'",
                        error_code=ErrorCode.invalid_annotation,
                    )
                explicit_extra_keys = extra_items_value
                explicit_extra_keys_readonly = Qualifier.ReadOnly in qualifiers
                continue
            if keyword.arg == "metaclass":
                self._show_error_if_checking(
                    keyword,
                    "TypedDict definitions cannot specify a metaclass",
                    error_code=ErrorCode.invalid_annotation,
                )
            else:
                if keyword.arg is None:
                    message = "TypedDict definitions do not support **kwargs"
                else:
                    message = (
                        f"Unexpected keyword argument {keyword.arg!r}"
                        " in TypedDict definition"
                    )
                self._show_error_if_checking(
                    keyword, message, error_code=ErrorCode.invalid_annotation
                )

        if closed is False and inherited_extra_keys is not None:
            self._show_error_if_checking(
                node,
                "Cannot set 'closed=False' when superclass is closed or has 'extra_items'",
                error_code=ErrorCode.invalid_annotation,
            )

        if (
            closed is True
            and inherited_extra_keys is not None
            and inherited_extra_keys is not NO_RETURN_VALUE
            and not inherited_extra_keys_readonly
        ):
            self._show_error_if_checking(
                node,
                "Cannot set 'closed=True' when superclass has non-read-only 'extra_items'",
                error_code=ErrorCode.invalid_annotation,
            )

        if (
            explicit_extra_keys is not None
            and inherited_extra_keys is not None
            and inherited_extra_keys is not NO_RETURN_VALUE
            and not inherited_extra_keys_readonly
        ):
            left_to_right = has_relation(
                inherited_extra_keys, explicit_extra_keys, Relation.ASSIGNABLE, self
            )
            right_to_left = has_relation(
                explicit_extra_keys, inherited_extra_keys, Relation.ASSIGNABLE, self
            )
            if isinstance(left_to_right, CanAssignError) or isinstance(
                right_to_left, CanAssignError
            ):
                self._show_error_if_checking(
                    node,
                    "Cannot change 'extra_items' type unless it is 'ReadOnly' in the superclass",
                    error_code=ErrorCode.invalid_annotation,
                )

        if explicit_extra_keys is None:
            if closed is True:
                extra_keys = NO_RETURN_VALUE
                extra_keys_readonly = False
            elif closed is False:
                extra_keys = None
                extra_keys_readonly = False
            else:
                extra_keys = inherited_extra_keys
                extra_keys_readonly = inherited_extra_keys_readonly
        else:
            extra_keys = explicit_extra_keys
            extra_keys_readonly = explicit_extra_keys_readonly

        return _SyntheticTypedDictContext(
            total=total,
            bases=bases,
            inherited_extra_keys=inherited_extra_keys,
            inherited_extra_keys_readonly=inherited_extra_keys_readonly,
            extra_keys=extra_keys,
            extra_keys_readonly=extra_keys_readonly,
        )

    def _validate_typeddict_class_syntax(self, node: ast.ClassDef) -> None:
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._show_error_if_checking(
                    statement,
                    "Methods are not allowed in TypedDict definitions",
                    error_code=ErrorCode.invalid_annotation,
                )

    def _is_typeddict_generic_base(self, base_value: Value) -> bool:
        if isinstance(base_value, MultiValuedValue):
            return all(
                self._is_typeddict_generic_base(subval) for subval in base_value.vals
            )
        if isinstance(base_value, KnownValue):
            origin = safe_getattr(base_value.val, "__origin__", None)
            return origin is not None and is_typing_name(origin, "Generic")
        return False

    def _is_namedtuple_generic_base(self, base_value: Value) -> bool:
        if isinstance(base_value, MultiValuedValue):
            return all(
                self._is_namedtuple_generic_base(subval) for subval in base_value.vals
            )
        if isinstance(base_value, GenericValue):
            return is_typing_name(base_value.typ, "Generic")
        if isinstance(base_value, KnownValue):
            origin = safe_getattr(base_value.val, "__origin__", None)
            return origin is not None and is_typing_name(origin, "Generic")
        return False

    def _make_namedtuple_related_fallback_class(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> type | None:
        has_namedtuple_marker_base = False
        runtime_bases: list[object] = []
        namedtuple_base_fields: set[str] = set()
        invalid_non_generic_bases: list[ast.expr] = []
        for base_node, base_value in zip(node.bases, base_values):
            if _is_namedtuple_marker_base(base_value):
                has_namedtuple_marker_base = True
                continue
            base = replace_fallback(base_value)
            if isinstance(base, KnownValue):
                runtime_base = base.val
            else:
                runtime_base = self._runtime_annotation_from_value(base_value)
                if runtime_base is typing.Any:
                    return None
            if isinstance(runtime_base, type) and is_namedtuple_class(runtime_base):
                namedtuple_base_fields.update(runtime_base._fields)
            if has_namedtuple_marker_base and not self._is_namedtuple_generic_base(
                base_value
            ):
                invalid_non_generic_bases.append(base_node)
            runtime_bases.append(runtime_base)

        if not has_namedtuple_marker_base and not namedtuple_base_fields:
            return None

        if invalid_non_generic_bases:
            for base_node in invalid_non_generic_bases:
                self._show_error_if_checking(
                    base_node,
                    "NamedTuple classes may only inherit from NamedTuple and Generic",
                    error_code=ErrorCode.invalid_base,
                )
            return None

        annotations: dict[str, object] = {}
        defaults: dict[str, object] = {}
        saw_default = False
        has_namedtuple_field_error = False
        for statement in node.body:
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.simple
            ):
                continue
            field_name = statement.target.id
            annotation_expr = annotation_expr_from_ast(
                statement.annotation, visitor=self
            )
            field_type, _ = annotation_expr.unqualify(
                {
                    Qualifier.ClassVar,
                    Qualifier.Final,
                    Qualifier.InitVar,
                    Qualifier.ReadOnly,
                    Qualifier.Required,
                    Qualifier.NotRequired,
                },
                mutually_exclusive_qualifiers=(
                    (Qualifier.Required, Qualifier.NotRequired),
                ),
            )
            annotations[field_name] = self._runtime_annotation_from_value(field_type)
            if statement.value is not None:
                defaults[field_name] = self._runtime_default_from_expr(statement.value)

            if has_namedtuple_marker_base:
                if field_name.startswith("_"):
                    has_namedtuple_field_error = True
                    self._show_error_if_checking(
                        statement.target,
                        "NamedTuple field names cannot start with an underscore",
                        error_code=ErrorCode.invalid_annotation,
                    )
                has_default = statement.value is not None
                if saw_default and not has_default:
                    has_namedtuple_field_error = True
                    self._show_error_if_checking(
                        statement,
                        "NamedTuple fields without defaults cannot follow fields with defaults",
                        error_code=ErrorCode.invalid_annotation,
                    )
                saw_default = saw_default or has_default
            elif field_name in namedtuple_base_fields:
                self._show_error_if_checking(
                    statement.target,
                    f"Field {field_name!r} conflicts with base NamedTuple field",
                    error_code=ErrorCode.incompatible_override,
                )

        if has_namedtuple_marker_base and has_namedtuple_field_error:
            return None

        module_name = self.module.__name__ if self.module is not None else self.filename
        qualname = self._get_class_qualname_from_name(node.name)

        if has_namedtuple_marker_base:
            field_names = tuple(annotations.keys())
            default_values = tuple(
                defaults[field_name]
                for field_name in field_names
                if field_name in defaults
            )
            namedtuple_defaults: tuple[object, ...] | None = (
                default_values if default_values else None
            )
            try:
                namedtuple_base = collections.namedtuple(
                    node.name,
                    field_names,
                    defaults=namedtuple_defaults,
                    module=module_name,
                )
            except Exception:
                self.log(
                    logging.INFO,
                    "unable to synthesize namedtuple runtime base",
                    node.name,
                )
                return None
            try:
                namedtuple_base.__qualname__ = qualname
                if annotations:
                    namedtuple_base.__annotations__ = annotations
            except Exception:
                pass

            if not runtime_bases:
                return namedtuple_base

            def exec_namedtuple_body(ns: dict[str, object]) -> None:
                ns["__module__"] = module_name
                ns["__qualname__"] = qualname
                if annotations:
                    ns["__annotations__"] = annotations

            try:
                return types.new_class(
                    node.name,
                    (namedtuple_base, *tuple(runtime_bases)),
                    {},
                    exec_namedtuple_body,
                )
            except Exception:
                self.log(
                    logging.INFO,
                    "unable to synthesize generic namedtuple runtime class",
                    node.name,
                )
                return None

        def exec_body(ns: dict[str, object]) -> None:
            ns["__module__"] = module_name
            ns["__qualname__"] = qualname
            if annotations:
                ns["__annotations__"] = annotations
            ns.update(defaults)

        try:
            return types.new_class(node.name, tuple(runtime_bases), {}, exec_body)
        except Exception:
            self.log(logging.INFO, "unable to synthesize runtime class", node.name)
            return None

    def _make_dataclass_related_fallback_class(
        self,
        node: ast.ClassDef,
        base_values: Sequence[Value],
        decorator_values: DecoratorValues,
    ) -> type | None:
        options = self._get_dataclass_decorator_options(decorator_values)
        if options is None:
            return None
        if not self._should_build_dataclass_kw_only_fallback(node, options):
            return None

        runtime_bases = []
        for base_value in base_values:
            runtime_base = self._runtime_base_from_value(
                base_value, allow_synthetic_class_base=True
            )
            if runtime_base is None:
                return None
            runtime_bases.append(runtime_base)

        annotations: dict[str, object] = {}
        defaults: dict[str, object] = {}
        marker = getattr(dataclasses, "KW_ONLY", None)
        for statement in node.body:
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.simple
            ):
                continue
            field_name = statement.target.id
            if marker is not None and self._is_dataclass_kw_only_marker_annotation(
                statement.annotation
            ):
                annotations[field_name] = marker
                continue
            annotation_expr = annotation_expr_from_ast(
                statement.annotation, visitor=self, suppress_errors=True
            )
            field_type, _ = annotation_expr.unqualify(
                {
                    Qualifier.ClassVar,
                    Qualifier.Final,
                    Qualifier.InitVar,
                    Qualifier.ReadOnly,
                    Qualifier.Required,
                    Qualifier.NotRequired,
                },
                mutually_exclusive_qualifiers=(
                    (Qualifier.Required, Qualifier.NotRequired),
                ),
            )
            annotations[field_name] = self._runtime_annotation_from_value(field_type)
            if statement.value is not None:
                defaults[field_name] = self._runtime_dataclass_default_from_expr(
                    statement.value
                )

        module_name = self.module.__name__ if self.module is not None else self.filename
        qualname = self._get_class_qualname_from_name(node.name)

        def exec_body(ns: dict[str, object]) -> None:
            ns["__module__"] = module_name
            ns["__qualname__"] = qualname
            ns["__annotations__"] = annotations
            for name, default in defaults.items():
                ns[name] = default

        try:
            runtime_class = types.new_class(
                node.name, tuple(runtime_bases), {}, exec_body
            )
            return dataclass(**options)(runtime_class)
        except Exception:
            self.log(
                logging.INFO, "unable to synthesize dataclass runtime class", node.name
            )
            return None

    def _make_enum_related_fallback_class(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> type | None:
        runtime_bases = self._runtime_enum_bases_from_values(
            base_values, allow_synthetic_class_base=False
        )
        if runtime_bases is None:
            return None

        members: dict[str, object] = {}
        for statement in node.body:
            if isinstance(statement, ast.Expr):
                if isinstance(statement.value, ast.Constant) and isinstance(
                    statement.value.value, str
                ):
                    continue
                return None
            if isinstance(statement, ast.Pass):
                continue
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(
                    statement.targets[0], ast.Name
                ):
                    return None
                member_name = statement.targets[0].id
                if member_name.startswith("_"):
                    continue
                with self.catch_errors() as errors:
                    member_value = self.visit(statement.value)
                if errors:
                    return None
            elif isinstance(statement, ast.AnnAssign):
                if (
                    not isinstance(statement.target, ast.Name)
                    or statement.value is None
                    or not statement.simple
                ):
                    return None
                member_name = statement.target.id
                if member_name.startswith("_"):
                    continue
                with self.catch_errors() as errors:
                    member_value = self.visit(statement.value)
                if errors:
                    return None
            else:
                return None
            if not isinstance(member_value, KnownValue):
                return None
            members[member_name] = member_value.val

        if not members:
            return None

        module_name = self.module.__name__ if self.module is not None else self.filename
        qualname = self._get_class_qualname_from_name(node.name)

        def exec_body(ns: dict[str, object]) -> None:
            ns["__module__"] = module_name
            ns["__qualname__"] = qualname
            ns.update(members)

        try:
            return types.new_class(node.name, tuple(runtime_bases), {}, exec_body)
        except Exception:
            self.log(logging.INFO, "unable to synthesize enum runtime class", node.name)
            return None

    def _runtime_base_from_value(
        self, base_value: Value, *, allow_synthetic_class_base: bool
    ) -> type | None:
        base = replace_fallback(base_value)
        if allow_synthetic_class_base and isinstance(base, SyntheticClassObjectValue):
            runtime_class = base.class_attributes.get("%runtime_class")
            if isinstance(runtime_class, KnownValue) and isinstance(
                runtime_class.val, type
            ):
                return runtime_class.val
            class_type = base.class_type
            if isinstance(class_type, TypedValue) and isinstance(class_type.typ, type):
                return class_type.typ
            return None
        if isinstance(base, KnownValue) and isinstance(base.val, type):
            return base.val
        if isinstance(base, TypedValue) and isinstance(base.typ, type):
            return base.typ
        runtime_annotation = self._runtime_annotation_from_value(base_value)
        if isinstance(runtime_annotation, type):
            return runtime_annotation
        return None

    def _runtime_enum_bases_from_values(
        self, base_values: Sequence[Value], *, allow_synthetic_class_base: bool
    ) -> list[type] | None:
        runtime_bases: list[type] = []
        has_enum_base = False
        for base_value in base_values:
            runtime_base = self._runtime_enum_base_from_value(
                base_value, allow_synthetic_class_base=allow_synthetic_class_base
            )
            if runtime_base is None:
                return None
            if safe_issubclass(runtime_base, enum.Enum):
                has_enum_base = True
            runtime_bases.append(runtime_base)
        if not has_enum_base:
            return None
        return runtime_bases

    def _runtime_enum_base_from_value(
        self, base_value: Value, *, allow_synthetic_class_base: bool
    ) -> type | None:
        return self._runtime_base_from_value(
            base_value, allow_synthetic_class_base=allow_synthetic_class_base
        )

    def _runtime_annotation_from_value(self, value: Value) -> object:
        if isinstance(value, AnnotatedValue):
            return self._runtime_annotation_from_value(value.value)
        if isinstance(value, TypeVarValue):
            return value.typevar
        if isinstance(value, KnownValue):
            return value.val
        if isinstance(value, GenericValue):
            if isinstance(value.typ, str):
                return typing.Any
            runtime_args = tuple(
                self._runtime_annotation_from_value(arg) for arg in value.args
            )
            runtime_typ: Any = value.typ
            try:
                if len(runtime_args) == 1:
                    return runtime_typ[runtime_args[0]]
                return runtime_typ[runtime_args]
            except Exception:
                return typing.Any
        if isinstance(value, TypedValue):
            return value.typ if not isinstance(value.typ, str) else typing.Any
        if isinstance(value, MultiValuedValue):
            runtime_members = [
                self._runtime_annotation_from_value(subval) for subval in value.vals
            ]
            if not runtime_members:
                return typing.Any
            result = runtime_members[0]
            for member in runtime_members[1:]:
                try:
                    runtime_result: Any = result
                    result = runtime_result | member
                except Exception:
                    return typing.Any
            return result
        return typing.Any

    def _runtime_default_from_expr(self, expr: ast.expr) -> object:
        value = self.visit(expr)
        if isinstance(value, KnownValue):
            return value.val
        return object()

    def _maybe_make_runtime_dataclass_field_default(
        self, expr: ast.expr
    ) -> object | None:
        if not isinstance(expr, ast.Call):
            return None
        callee = self.visit(expr.func)
        if not (isinstance(callee, KnownValue) and callee.val is dataclass_field):
            return None
        if expr.args:
            return None
        kwargs: dict[str, object] = {}
        for kw in expr.keywords:
            if kw.arg is None:
                return None
            value = self.visit(kw.value)
            if not isinstance(value, KnownValue):
                return None
            kwargs[kw.arg] = value.val
        try:
            return cast(Any, dataclass_field)(**kwargs)
        except Exception:
            return None

    def _runtime_dataclass_default_from_expr(self, expr: ast.expr) -> object:
        if (
            field_default := self._maybe_make_runtime_dataclass_field_default(expr)
        ) is not None:
            return field_default
        return self._runtime_default_from_expr(expr)

    def _apply_synthetic_enum_semantics(
        self,
        node: ast.ClassDef,
        synthetic_class: SyntheticClassObjectValue,
        class_key: type | str,
    ) -> None:
        if not self._is_enum_class_key(class_key):
            return

        class_attributes = synthetic_class.class_attributes
        enum_value_type = self.enum_value_type_by_class.get(class_key)
        ignore_names = _enum_ignore_names(class_attributes.get("_ignore_"))
        member_literal_values: dict[str, object] = {}
        member_order: list[str] = []
        missing: Value | None = None

        for member_name, statement in _iter_enum_assignment_candidates(node):
            stmt_forced_member, stmt_forced_nonmember = (
                _enum_statement_member_decorators(statement)
            )
            value = class_attributes.get(member_name, missing)
            if value is missing:
                if stmt_forced_nonmember:
                    continue
                if (
                    isinstance(
                        statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    )
                    and not stmt_forced_member
                ):
                    continue
                value = AnyValue(AnySource.inference)
                unwrapped = value
                forced_member = stmt_forced_member
                forced_nonmember = False
            else:
                assert value is not None
                unwrapped, forced_member, forced_nonmember = (
                    _unwrap_enum_member_wrapper(value)
                )

            if member_name in ignore_names:
                continue
            if member_name.startswith("_"):
                continue
            if member_name.startswith("__") and not member_name.endswith("__"):
                continue
            if isinstance(statement, ast.AnnAssign) and statement.value is None:
                continue
            if stmt_forced_nonmember:
                continue
            if (
                isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
                and not stmt_forced_member
            ):
                continue
            if forced_nonmember:
                continue
            if not forced_member and _is_nonmember_enum_assignment_value(
                unwrapped, self
            ):
                continue
            if isinstance(statement, ast.AnnAssign):
                self._show_error_if_checking(
                    statement.target,
                    "Enum members should not be explicitly annotated",
                    error_code=ErrorCode.invalid_annotation,
                )

            if (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and isinstance(statement.value, ast.Name)
                and statement.value.id in member_literal_values
            ):
                literal_value = member_literal_values[statement.value.id]
            else:
                literal_value = _runtime_object_for_enum_member(unwrapped)
            if literal_value is Ellipsis:
                literal_value = object()
            member_literal_values[member_name] = literal_value
            member_order.append(member_name)

            if (
                enum_value_type is not None
                and isinstance(unwrapped, KnownValue)
                and not isinstance(unwrapped.val, tuple)
            ):
                self._check_declared_enum_value_type(
                    enum_value_type, unwrapped, statement
                )

        for attr_name in list(class_attributes):
            mangled = _mangle_private_enum_name(node.name, attr_name)
            if mangled is None:
                continue
            class_attributes.setdefault(mangled, class_attributes[attr_name])
            del class_attributes[attr_name]

        runtime_enum = self._make_synthetic_enum_runtime_class(
            node, synthetic_class.base_classes, member_literal_values, member_order
        )
        if runtime_enum is None:
            return

        object.__setattr__(synthetic_class, "class_type", TypedValue(runtime_enum))
        for member_name in member_order:
            try:
                class_attributes[member_name] = KnownValue(
                    getattr(runtime_enum, member_name)
                )
            except Exception:
                continue

    def _check_declared_enum_value_type(
        self, expected_type: Value, actual_value: Value, node: ast.AST
    ) -> None:
        can_assign = has_relation(
            expected_type, actual_value, Relation.ASSIGNABLE, self
        )
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node,
                f"Enum member value must be assignable to {expected_type}",
                error_code=ErrorCode.invalid_annotation,
                detail=can_assign.display(),
            )

    def _make_synthetic_enum_runtime_class(
        self,
        node: ast.ClassDef,
        base_values: Sequence[Value],
        member_literal_values: Mapping[str, object],
        member_order: Sequence[str],
    ) -> type | None:
        runtime_bases = self._runtime_enum_bases_from_values(
            base_values, allow_synthetic_class_base=True
        )
        if runtime_bases is None:
            return None

        members = {name: member_literal_values[name] for name in member_order}
        module_name = self.module.__name__ if self.module is not None else self.filename
        qualname = self._get_class_qualname_from_name(node.name)

        def exec_body(ns: dict[str, object]) -> None:
            ns["__module__"] = module_name
            ns["__qualname__"] = qualname
            ns.update(members)

        try:
            return types.new_class(node.name, tuple(runtime_bases), {}, exec_body)
        except Exception:
            self.log(
                logging.INFO,
                "unable to synthesize enum runtime class from static analysis",
                node.name,
            )
            return None

    def _typed_dict_base_value(
        self, base_value: Value, base_node: ast.AST
    ) -> TypedDictValue | None:
        if isinstance(base_value, MultiValuedValue):
            typeddict_values = [
                typed_dict
                for subval in base_value.vals
                if (typed_dict := self._typed_dict_base_value(subval, base_node))
                is not None
            ]
            if typeddict_values:
                return max(
                    typeddict_values,
                    key=lambda value: (
                        len(value.items),
                        int(value.extra_keys is not None),
                    ),
                )
            return None
        if isinstance(base_value, SyntheticClassObjectValue) and isinstance(
            base_value.class_type, TypedDictValue
        ):
            return base_value.class_type
        if isinstance(base_value, KnownValue):
            if isinstance(base_value.val, type) and is_typeddict(base_value.val):
                typed_dict_type = type_from_value(base_value, self, base_node)
                if isinstance(typed_dict_type, TypedDictValue):
                    return typed_dict_type
        return None

    def _is_dataclass_decorator_target(self, target: ast.expr) -> bool:
        if isinstance(target, ast.Attribute):
            return target.attr == "dataclass"
        if not isinstance(target, ast.Name):
            return False
        if target.id == "dataclass":
            return True
        value = self.scopes.get(target.id, target, self.state, can_assign_ctx=self)
        return _is_dataclass_decorator_value(value)

    def _get_dataclass_decorator_options(
        self, decorator_values: DecoratorValues
    ) -> dict[str, bool] | None:
        for decorator, _, expr in decorator_values:
            if _is_known_decorator(decorator, dataclasses.dataclass):
                options = {}
                if isinstance(expr, ast.Call):
                    for kw in expr.keywords:
                        if kw.arg is None:
                            continue
                        value = _get_bool_literal(kw.value)
                        if value is None:
                            continue
                        options[kw.arg] = value
                return options
        return None

    def _get_direct_dataclass_transform_info(
        self, decorator_values: DecoratorValues
    ) -> DataclassTransformInfo | None:
        infos: list[DataclassTransformInfo] = []
        for _, value, _ in decorator_values:
            _, exts = unannotate_value(value, DataclassTransformDecoratorExtension)
            if exts:
                for ext in exts:
                    infos.append(ext.info)
        if not infos:
            return None
        return _merge_dataclass_transform_infos(infos)

    def _get_dataclass_transform_info_from_runtime_object(
        self, obj: object
    ) -> DataclassTransformInfo | None:
        raw = safe_getattr(obj, "__dataclass_transform__", None)
        if not isinstance(raw, Mapping):
            return None

        eq_default = raw.get("eq_default", True)
        if not isinstance(eq_default, bool):
            eq_default = True
        frozen_default = raw.get("frozen_default", False)
        if not isinstance(frozen_default, bool):
            frozen_default = False
        kw_only_default = raw.get("kw_only_default", False)
        if not isinstance(kw_only_default, bool):
            kw_only_default = False
        order_default = raw.get("order_default", False)
        if not isinstance(order_default, bool):
            order_default = False

        field_specifier_values: list[Value] = []
        raw_field_specifiers = raw.get("field_specifiers", ())
        if isinstance(raw_field_specifiers, (tuple, list, set, frozenset)):
            for field_specifier in raw_field_specifiers:
                value = KnownValue(field_specifier)
                if value not in field_specifier_values:
                    field_specifier_values.append(value)

        return DataclassTransformInfo(
            eq_default=eq_default,
            frozen_default=frozen_default,
            kw_only_default=kw_only_default,
            order_default=order_default,
            field_specifiers=tuple(field_specifier_values),
        )

    def _get_dataclass_transform_info_from_synthetic_class(
        self, value: SyntheticClassObjectValue
    ) -> DataclassTransformInfo | None:
        transform_marker = value.class_attributes.get("%dataclass_transform")
        if not (
            isinstance(transform_marker, KnownValue) and transform_marker.val is True
        ):
            return None

        eq_default: bool = True
        raw_eq_default = value.class_attributes.get("%dataclass_transform_eq_default")
        if isinstance(raw_eq_default, KnownValue) and isinstance(
            raw_eq_default.val, bool
        ):
            eq_default = raw_eq_default.val

        frozen_default: bool = False
        raw_frozen_default = value.class_attributes.get(
            "%dataclass_transform_frozen_default"
        )
        if isinstance(raw_frozen_default, KnownValue) and isinstance(
            raw_frozen_default.val, bool
        ):
            frozen_default = raw_frozen_default.val

        kw_only_default: bool = False
        raw_kw_only_default = value.class_attributes.get(
            "%dataclass_transform_kw_only_default"
        )
        if isinstance(raw_kw_only_default, KnownValue) and isinstance(
            raw_kw_only_default.val, bool
        ):
            kw_only_default = raw_kw_only_default.val

        order_default: bool = False
        raw_order_default = value.class_attributes.get(
            "%dataclass_transform_order_default"
        )
        if isinstance(raw_order_default, KnownValue) and isinstance(
            raw_order_default.val, bool
        ):
            order_default = raw_order_default.val

        field_specifiers: tuple[Value, ...] = ()
        raw_field_specifiers = value.class_attributes.get(
            "%dataclass_transform_field_specifiers"
        )
        if isinstance(raw_field_specifiers, KnownValue) and isinstance(
            raw_field_specifiers.val, (tuple, list)
        ):
            values = [
                item for item in raw_field_specifiers.val if isinstance(item, Value)
            ]
            field_specifiers = tuple(values)

        return DataclassTransformInfo(
            eq_default=eq_default,
            frozen_default=frozen_default,
            kw_only_default=kw_only_default,
            order_default=order_default,
            field_specifiers=field_specifiers,
        )

    def _get_dataclass_transform_info_from_value(
        self, value: Value
    ) -> DataclassTransformInfo | None:
        if value is UNINITIALIZED_VALUE:
            return None
        if isinstance(value, (ReferencingValue, InputSigValue)):
            return None
        if isinstance(value, AnnotatedValue):
            infos = [
                extension.info
                for extension in value.get_metadata_of_type(DataclassTransformExtension)
            ]
            if (
                value_info := self._get_dataclass_transform_info_from_value(value.value)
            ) is not None:
                infos.append(value_info)
            if not infos:
                return None
            return _merge_dataclass_transform_infos(infos)
        if isinstance(value, PartialValue):
            if (
                info := self._get_dataclass_transform_info_from_value(
                    value.runtime_value
                )
            ) is not None:
                return info
            return self._get_dataclass_transform_info_from_value(value.root)
        value = replace_fallback(value)
        if isinstance(value, MultiValuedValue):
            infos = [
                info
                for subval in value.vals
                if (info := self._get_dataclass_transform_info_from_value(subval))
                is not None
            ]
            if not infos:
                return None
            return _merge_dataclass_transform_infos(infos)
        if isinstance(value, SyntheticClassObjectValue):
            return self._get_dataclass_transform_info_from_synthetic_class(value)
        if isinstance(value, KnownValue):
            if (
                info := self._get_dataclass_transform_info_from_runtime_object(
                    value.val
                )
            ) is not None:
                return info
            origin = safe_getattr(value.val, "__origin__", None)
            if isinstance(origin, type):
                synthetic_class = self.checker.get_synthetic_class(origin)
                if (
                    synthetic_class is not None
                    and (
                        synthetic_info := self._get_dataclass_transform_info_from_synthetic_class(
                            synthetic_class
                        )
                    )
                    is not None
                ):
                    return synthetic_info
                return self._get_dataclass_transform_info_from_runtime_object(origin)
            return None
        if isinstance(value, SubclassValue) and isinstance(value.typ, TypedValue):
            return self._get_dataclass_transform_info_from_value(value.typ)
        if isinstance(value, (TypedValue, GenericValue)):
            if isinstance(value.typ, str):
                synthetic_class = self.checker.get_synthetic_class(value.typ)
                if synthetic_class is None:
                    return None
                return self._get_dataclass_transform_info_from_synthetic_class(
                    synthetic_class
                )
            return self._get_dataclass_transform_info_from_runtime_object(value.typ)
        return None

    def _get_class_transform_provider_info(
        self,
        *,
        class_obj: type | None,
        base_values: Sequence[Value],
        keyword_values: Sequence[tuple[ast.keyword, Value]],
        direct_dataclass_transform_info: DataclassTransformInfo | None,
    ) -> DataclassTransformInfo | None:
        infos = [
            info
            for base_value in base_values
            if (info := self._get_dataclass_transform_info_from_value(base_value))
            is not None
        ]
        metaclass_transform_info = next(
            (
                info
                for keyword, value in keyword_values
                if keyword.arg == "metaclass"
                and (info := self._get_dataclass_transform_info_from_value(value))
                is not None
            ),
            None,
        )
        if metaclass_transform_info is not None:
            infos.append(metaclass_transform_info)
        if direct_dataclass_transform_info is not None:
            infos.append(direct_dataclass_transform_info)
        if (
            class_obj is not None
            and (
                runtime_transform_info := self._get_dataclass_transform_info_from_runtime_object(
                    class_obj
                )
            )
            is not None
        ):
            infos.append(runtime_transform_info)
        if not infos:
            return None
        return _merge_dataclass_transform_infos(infos)

    def _get_class_dataclass_semantics(
        self,
        node: ast.ClassDef,
        *,
        class_obj: type | None,
        base_values: Sequence[Value],
        keyword_values: Sequence[tuple[ast.keyword, Value]],
        decorator_values: DecoratorValues,
    ) -> DataclassInfo | None:
        final_info: DataclassInfo | None = None
        dataclass_options = self._get_dataclass_decorator_options(decorator_values)
        if dataclass_options is not None:
            final_info = DataclassInfo(
                init=dataclass_options.get("init", True),
                eq=dataclass_options.get("eq", True),
                frozen=dataclass_options.get("frozen", False),
                unsafe_hash=dataclass_options.get("unsafe_hash", False),
                match_args=dataclass_options.get("match_args", True),
                order=dataclass_options.get("order", False),
                slots=dataclass_options.get("slots", False),
                kw_only_default=dataclass_options.get("kw_only", False),
                field_specifiers=(KnownValue(dataclass_field),),
            )

        for target_value, _, decorator in decorator_values:
            info = self._get_dataclass_transform_info_from_value(target_value)
            if info is None:
                continue
            if final_info is not None:
                self._show_error_if_checking(
                    decorator,
                    "Multiple dataclass transforms on the same class are not supported",
                    ErrorCode.multiple_dataclass_transform,
                )
                continue
            if isinstance(decorator, ast.Call):
                keywords = _extract_keywords(decorator.keywords)
            else:
                keywords = {}
            final_info = DataclassInfo.from_transform_info_and_options(info, keywords)

        base_transform_infos = [
            info
            for base_value in base_values
            if (info := self._get_dataclass_transform_info_from_value(base_value))
            is not None
        ]
        metaclass_transform_info = next(
            (
                info
                for keyword, value in keyword_values
                if keyword.arg == "metaclass"
                and (info := self._get_dataclass_transform_info_from_value(value))
                is not None
            ),
            None,
        )

        if base_transform_infos or metaclass_transform_info:
            class_keywords = _extract_keywords(node.keywords)
        else:
            class_keywords = {}

        for base_transform_info in base_transform_infos:
            if final_info is not None:
                self._show_error_if_checking(
                    node,
                    "Cannot combine dataclass transform from base class with dataclass decorator",
                    ErrorCode.multiple_dataclass_transform,
                )
                continue
            final_info = DataclassInfo.from_transform_info_and_options(
                base_transform_info, class_keywords
            )

        if metaclass_transform_info is not None:
            if final_info is not None:
                self._show_error_if_checking(
                    node,
                    "Cannot combine dataclass transform from metaclass with "
                    "dataclass decorator or base class transform",
                    ErrorCode.multiple_dataclass_transform,
                )
            else:
                final_info = DataclassInfo.from_transform_info_and_options(
                    metaclass_transform_info, class_keywords
                )
                if "frozen" not in class_keywords:
                    # Per the typing spec, a class that directly specifies a
                    # dataclass_transform-decorated metaclass is considered neither frozen
                    # nor non-frozen for inheritance checks.
                    final_info = replace(final_info, frozen=None)

        if (
            final_info is None
            and class_obj is not None
            and is_dataclass_type(class_obj)
        ):
            dataclass_params = safe_getattr(class_obj, "__dataclass_params__", None)
            init = safe_getattr(dataclass_params, "init", None)
            if not isinstance(init, bool):
                init = True
            eq = safe_getattr(dataclass_params, "eq", None)
            if not isinstance(eq, bool):
                eq = True
            frozen = safe_getattr(dataclass_params, "frozen", None)
            if not isinstance(frozen, bool):
                frozen = False
            unsafe_hash = safe_getattr(dataclass_params, "unsafe_hash", None)
            if not isinstance(unsafe_hash, bool):
                unsafe_hash = False
            match_args = "__match_args__" in class_obj.__dict__
            order = safe_getattr(dataclass_params, "order", None)
            if not isinstance(order, bool):
                order = True
            slots = "__slots__" in class_obj.__dict__
            return DataclassInfo(
                init=init,
                eq=eq,
                frozen=frozen,
                unsafe_hash=unsafe_hash,
                match_args=match_args,
                order=order,
                slots=slots,
                kw_only_default=False,
                field_specifiers=(KnownValue(dataclass_field),),
            )

        return final_info

    def _apply_dataclass_hash_semantics(
        self,
        synthetic_class: SyntheticClassObjectValue,
        semantics: DataclassInfo | None,
    ) -> None:
        if semantics is None:
            return
        if "__hash__" in synthetic_class.class_attributes:
            return
        synthesized_hash = _synthesize_dataclass_hash_attribute(semantics)
        if synthesized_hash is not None:
            synthetic_class.class_attributes["__hash__"] = synthesized_hash

    def _check_dataclass_slots_definition(
        self, node: ast.ClassDef, semantics: DataclassInfo
    ) -> None:
        if semantics.slots is not True:
            return
        if not _class_body_defines_slots(node):
            return
        self._show_error_if_checking(
            node,
            "Class cannot define __slots__ when dataclass slots=True",
            error_code=ErrorCode.invalid_annotation,
        )

    def _apply_dataclass_slots_semantics(
        self,
        synthetic_class: SyntheticClassObjectValue,
        semantics: DataclassInfo | None,
    ) -> None:
        if semantics is None:
            return
        _record_dataclass_slots_flag(synthetic_class, semantics)
        if semantics.slots is not True:
            return
        if "__slots__" in synthetic_class.class_attributes:
            return
        slot_names = self._dataclass_slot_names_from_synthetic_class(synthetic_class)
        if slot_names is None:
            return
        synthetic_class.class_attributes["__slots__"] = KnownValue(slot_names)

    def _dataclass_slot_names_from_synthetic_class(
        self, synthetic_class: SyntheticClassObjectValue
    ) -> tuple[str, ...] | None:
        local_names = _known_string_sequence_values(
            synthetic_class.class_attributes.get("%dataclass_field_order")
        )
        if local_names is None or not local_names:
            classvar_names = set(
                _known_string_sequence_values(
                    synthetic_class.class_attributes.get("%classvars")
                )
                or ()
            )
            local_names = tuple(
                name
                for name in synthetic_class.class_attributes
                if not name.startswith("%")
                and not (name.startswith("__") and name.endswith("__"))
                and name not in synthetic_class.method_attributes
                and name not in classvar_names
            )
        initvar_names = set(
            _known_string_sequence_values(
                synthetic_class.class_attributes.get("%dataclass_initvar_fields")
            )
            or ()
        )
        return tuple(name for name in local_names if name not in initvar_names)

    def _is_dataclass_kw_only_marker_annotation(self, node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            value = self.scopes.get(node.id, node, self.state, can_assign_ctx=self)
            return _is_dataclass_kw_only_marker_value(value)
        if isinstance(node, ast.Attribute):
            with self.catch_errors():
                value = self.visit(node)
            return _is_dataclass_kw_only_marker_value(value)
        return False

    def _is_dataclass_field_call(self, expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        with self.catch_errors():
            callee = self.visit(expr.func)
        return self._is_dataclass_field_callee(callee)

    def _is_dataclass_field_callee(self, callee: Value) -> bool:
        if isinstance(callee, KnownValue) and callee.val is dataclass_field:
            return True
        if self.current_dataclass_info is None:
            return False
        return any(
            _value_matches_dataclass_field_specifier(callee, field_specifier)
            for field_specifier in self.current_dataclass_info.field_specifiers
        )

    def _get_bound_args_for_dataclass_field_signature(
        self, signature: Signature, actual_args: ActualArguments, *, is_overload: bool
    ) -> BoundArgs | None:
        ctx = _DataclassFieldInferenceCallContext(self)
        bound_args = signature.bind_arguments(actual_args, ctx)
        if bound_args is None:
            return None
        ret = signature.check_call_preprocessed(
            actual_args, ctx, is_overload=is_overload
        )
        if ret.is_error or ret.remaining_arguments is not None:
            return None
        return bound_args

    def _get_dataclass_field_call_bound_args_from_resolved_call(
        self,
        callee: Value,
        args: Sequence[Composite],
        keywords: Sequence[tuple[str | None, Composite]],
        node: ast.Call,
    ) -> BoundArgs | None:
        signature = self.signature_from_value(callee, node)
        if not isinstance(signature, (Signature, OverloadedSignature)):
            return None

        arguments = _arguments_from_call_composites(args, keywords)
        preprocess_ctx = _DataclassFieldInferenceCallContext(self)
        actual_args = preprocess_args(arguments, preprocess_ctx)
        if actual_args is None:
            return None

        if isinstance(signature, Signature):
            return self._get_bound_args_for_dataclass_field_signature(
                signature, actual_args, is_overload=False
            )

        last = len(signature.signatures) - 1
        for i, overload_sig in enumerate(signature.signatures):
            bound_args = self._get_bound_args_for_dataclass_field_signature(
                overload_sig, actual_args, is_overload=i != last
            )
            if bound_args is not None:
                return bound_args
        return None

    def _infer_dataclass_field_call_options_from_resolved_call(
        self,
        callee: Value,
        args: Sequence[Composite],
        keywords: Sequence[tuple[str | None, Composite]],
        node: ast.Call,
    ) -> _DataclassFieldCallOptions | None:
        if not self._is_dataclass_field_callee(callee):
            return None
        bound_args = self._get_dataclass_field_call_bound_args_from_resolved_call(
            callee, args, keywords, node
        )
        if bound_args is None:
            return _DataclassFieldCallOptions()
        init = _dataclass_field_bound_bool_arg(bound_args, "init")
        kw_only = _dataclass_field_bound_bool_arg(
            bound_args, "kw_only", include_default=False
        )
        alias = _dataclass_field_bound_str_arg(
            bound_args, "alias", include_default=False
        )
        has_default = False
        default_value = _dataclass_field_bound_arg(
            bound_args, "default", include_default=False
        )
        if default_value is not None and not _is_absent_dataclass_default_value(
            default_value
        ):
            has_default = True
        default_factory = None
        for key in ("default_factory", "factory"):
            factory_value = _dataclass_field_bound_arg(
                bound_args, key, include_default=False
            )
            if factory_value is None:
                continue
            if _is_absent_dataclass_default_value(factory_value):
                continue
            has_default = True
            default_factory = factory_value
            break
        return _DataclassFieldCallOptions(
            init=init,
            kw_only=kw_only,
            alias=alias,
            has_default=has_default,
            default_factory=default_factory,
        )

    def _should_build_dataclass_kw_only_fallback(
        self, node: ast.ClassDef, options: Mapping[str, bool]
    ) -> bool:
        if options.get("kw_only", False):
            return True
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            if self._is_dataclass_kw_only_marker_annotation(statement.annotation):
                return True
            value_expr = statement.value
            if (
                isinstance(value_expr, ast.Call)
                and self._is_dataclass_field_call(value_expr)
                and any(kw.arg == "kw_only" for kw in value_expr.keywords)
            ):
                return True
        return False

    def _slot_state_for_synthetic_class(
        self,
        synthetic_class: SyntheticClassObjectValue,
        *,
        seen: set[int] | None = None,
    ) -> tuple[frozenset[str], bool] | None:
        if seen is None:
            seen = set()
        synthetic_id = id(synthetic_class)
        if synthetic_id in seen:
            return frozenset(), False
        seen.add(synthetic_id)

        slot_names: set[str] = set()
        has_dict = False
        if "__slots__" in synthetic_class.class_attributes:
            names = _known_string_sequence_values(
                synthetic_class.class_attributes.get("__slots__")
            )
            if names is None:
                return None
            normalized_names, local_has_dict = _normalize_slot_names(names)
            slot_names.update(normalized_names)
            has_dict = has_dict or local_has_dict
        elif synthetic_class.class_attributes.get("%dataclass_slots") == KnownValue(
            True
        ):
            dataclass_slot_names = self._dataclass_slot_names_from_synthetic_class(
                synthetic_class
            )
            if dataclass_slot_names is None:
                return None
            slot_names.update(dataclass_slot_names)
        else:
            has_dict = True

        for base in synthetic_class.base_classes:
            base_state = self._slot_state_for_base_value(base, seen=seen)
            if base_state is None:
                return None
            base_slots, base_has_dict = base_state
            slot_names.update(base_slots)
            has_dict = has_dict or base_has_dict
        return frozenset(slot_names), has_dict

    def _slot_state_for_runtime_type(
        self, typ: type
    ) -> tuple[frozenset[str], bool] | None:
        if typ is object:
            return frozenset(), False
        if attributes.may_have_dynamic_attributes(typ):
            return frozenset(), True
        slot_names: set[str] = set()
        has_dict = False
        saw_slots = False
        for base in typ.__mro__:
            if base is object:
                continue
            if "__slots__" not in base.__dict__:
                has_dict = True
                continue
            names = _slot_names_from_runtime_slots(base.__dict__["__slots__"])
            if names is None:
                return None
            normalized_names, local_has_dict = _normalize_slot_names(names)
            slot_names.update(normalized_names)
            has_dict = has_dict or local_has_dict
            saw_slots = True
        if not saw_slots:
            has_dict = True
        return frozenset(slot_names), has_dict

    def _slot_state_for_base_value(
        self, value: Value, *, seen: set[int] | None = None
    ) -> tuple[frozenset[str], bool] | None:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._slot_state_for_base_value(value.value, seen=seen)
        if isinstance(value, SyntheticClassObjectValue):
            return self._slot_state_for_synthetic_class(value, seen=seen)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            states = {
                state
                for subval in value.vals
                if (state := self._slot_state_for_base_value(subval, seen=seen))
                is not None
            }
            if len(states) == 1:
                return next(iter(states))
            return None
        if isinstance(value, KnownValue) and isinstance(value.val, type):
            return self._slot_state_for_runtime_type(value.val)
        if isinstance(value, (TypedValue, GenericValue)) and isinstance(
            value.typ, (type, str)
        ):
            return self._slot_state_for_type(value.typ, seen=seen)
        return None

    def _slot_state_for_type(
        self, typ: type | str, *, seen: set[int] | None = None
    ) -> tuple[frozenset[str], bool] | None:
        if isinstance(typ, str):
            synthetic_class = self.checker.get_synthetic_class(typ)
            if synthetic_class is None:
                return None
            return self._slot_state_for_synthetic_class(synthetic_class, seen=seen)
        return self._slot_state_for_runtime_type(typ)

    def _slot_state_for_instance_value(
        self, value: Value
    ) -> tuple[frozenset[str], bool] | None:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._slot_state_for_instance_value(value.value)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            states = {
                state
                for subval in value.vals
                if (state := self._slot_state_for_instance_value(subval)) is not None
            }
            if len(states) == 1:
                return next(iter(states))
            return None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return None
            return self._slot_state_for_runtime_type(type(value.val))
        if isinstance(value, (TypedValue, GenericValue)):
            if isinstance(value.typ, (type, str)):
                if isinstance(value.typ, type) and safe_getattr(
                    value.typ, "__abstractmethods__", None
                ):
                    return None
                return self._slot_state_for_type(value.typ)
        return None

    def _is_assignment_to_non_slot_attribute(
        self, root_value: Value, attr_name: str
    ) -> bool:
        slot_state = self._slot_state_for_instance_value(root_value)
        if slot_state is None:
            return False
        slot_names, has_dict = slot_state
        if has_dict:
            return False
        return attr_name not in slot_names

    def _get_dataclass_status_for_type(
        self, typ: type | str
    ) -> tuple[bool, bool | None]:
        synthetic_class = self.checker.get_synthetic_class(typ)
        if synthetic_class is not None and synthetic_class.is_dataclass:
            return True, synthetic_class.dataclass_frozen
        if not isinstance(typ, type) or not is_dataclass_type(typ):
            return False, None
        dataclass_params = safe_getattr(typ, "__dataclass_params__", None)
        frozen = safe_getattr(dataclass_params, "frozen", None)
        if not isinstance(frozen, bool):
            frozen = None
        return True, frozen

    def _get_dataclass_order_status_for_type(
        self, typ: type | str
    ) -> tuple[bool, bool | None]:
        synthetic_class = self.checker.get_synthetic_class(typ)
        if synthetic_class is not None and synthetic_class.is_dataclass:
            return True, synthetic_class.dataclass_order
        if not isinstance(typ, type) or not is_dataclass_type(typ):
            return False, None
        dataclass_params = safe_getattr(typ, "__dataclass_params__", None)
        order = safe_getattr(dataclass_params, "order", None)
        if not isinstance(order, bool):
            order = None
        return True, order

    def _get_dataclass_status_for_class_value(
        self, value: Value
    ) -> tuple[bool, bool | None]:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._get_dataclass_status_for_class_value(value.value)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            statuses = {
                self._get_dataclass_status_for_class_value(subval)
                for subval in value.vals
            }
            if len(statuses) == 1:
                return next(iter(statuses))
            return False, None
        if isinstance(value, SyntheticClassObjectValue):
            if value.is_dataclass:
                return True, value.dataclass_frozen
            if isinstance(value.class_type, TypedValue) and isinstance(
                value.class_type.typ, (type, str)
            ):
                return self._get_dataclass_status_for_type(value.class_type.typ)
            return False, None
        if isinstance(value, SubclassValue) and isinstance(value.typ, TypedValue):
            if isinstance(value.typ.typ, (type, str)):
                return self._get_dataclass_status_for_type(value.typ.typ)
            return False, None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return self._get_dataclass_status_for_type(value.val)
            return False, None
        if isinstance(value, (TypedValue, GenericValue)):
            if isinstance(value.typ, (type, str)):
                return self._get_dataclass_status_for_type(value.typ)
        return False, None

    def _get_dataclass_status_for_instance_value(
        self, value: Value
    ) -> tuple[bool, bool | None]:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._get_dataclass_status_for_instance_value(value.value)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            statuses = {
                self._get_dataclass_status_for_instance_value(subval)
                for subval in value.vals
            }
            if len(statuses) == 1:
                return next(iter(statuses))
            return False, None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return False, None
            return self._get_dataclass_status_for_type(type(value.val))
        if isinstance(value, (TypedValue, GenericValue)):
            if isinstance(value.typ, (type, str)):
                return self._get_dataclass_status_for_type(value.typ)
        return False, None

    def _get_dataclass_order_info_for_instance_value(
        self, value: Value
    ) -> tuple[type | str, bool | None] | None:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._get_dataclass_order_info_for_instance_value(value.value)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            infos = {
                info
                for subval in value.vals
                if (info := self._get_dataclass_order_info_for_instance_value(subval))
                is not None
            }
            if len(infos) == 1:
                return next(iter(infos))
            return None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return None
            typ: type | str = type(value.val)
            is_dataclass, order = self._get_dataclass_order_status_for_type(typ)
            if not is_dataclass:
                return None
            return typ, order
        if isinstance(value, (TypedValue, GenericValue)):
            if not isinstance(value.typ, (type, str)):
                return None
            typ = value.typ
            is_dataclass, order = self._get_dataclass_order_status_for_type(typ)
            if not is_dataclass:
                return None
            return typ, order
        return None

    def _check_dataclass_order_comparison(
        self, op: ast.cmpop, lhs: Value, rhs: Value, parent_node: ast.AST
    ) -> None:
        if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return
        lhs_info = self._get_dataclass_order_info_for_instance_value(lhs)
        rhs_info = self._get_dataclass_order_info_for_instance_value(rhs)
        if lhs_info is None or rhs_info is None:
            return
        lhs_type, lhs_order = lhs_info
        rhs_type, rhs_order = rhs_info
        if lhs_order is not True or rhs_order is not True or lhs_type != rhs_type:
            description, _, _, _ = BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD[type(op)]
            lhs_shown = (
                lhs_type if isinstance(lhs_type, str) else stringify_object(lhs_type)
            )
            rhs_shown = (
                rhs_type if isinstance(rhs_type, str) else stringify_object(rhs_type)
            )
            self._show_error_if_checking(
                parent_node,
                f"Unsupported operands for {description}: {lhs_shown} and {rhs_shown}",
                error_code=ErrorCode.unsupported_operation,
            )

    def _check_dataclass_inheritance(
        self,
        node: ast.ClassDef,
        base_values: Sequence[Value],
        frozen_dataclass: bool | None,
    ) -> None:
        if frozen_dataclass is None:
            return
        for base_value in base_values:
            is_base_dataclass, base_frozen = self._get_dataclass_status_for_class_value(
                base_value
            )
            if not is_base_dataclass or base_frozen is None:
                continue
            if base_frozen == frozen_dataclass:
                continue
            if frozen_dataclass:
                message = "Frozen dataclass cannot inherit from a non-frozen dataclass"
            else:
                message = "Non-frozen dataclass cannot inherit from a frozen dataclass"
            self._show_error_if_checking(
                node, message, error_code=ErrorCode.invalid_base
            )
            return

    def _check_dataclass_post_init_signature(
        self, node: ast.ClassDef, dataclass_class: SyntheticClassObjectValue
    ) -> None:
        post_init_node = _get_dataclass_post_init_node(node)
        if post_init_node is None:
            return
        post_init_value = dataclass_class.class_attributes.get("__post_init__")
        if post_init_value is None:
            return
        post_init_params = self.checker.get_synthetic_dataclass_post_init_parameters(
            dataclass_class
        )
        expected_params = [
            SigParameter(
                "self",
                ParameterKind.POSITIONAL_ONLY,
                annotation=AnyValue(AnySource.inference),
            ),
            *post_init_params,
        ]
        try:
            expected_signature = Signature.make(
                expected_params, AnyValue(AnySource.inference)
            )
        except InvalidSignature:
            return
        expected_value = CallableValue(expected_signature)
        can_assign = self._can_assign_to_base_callable(expected_value, post_init_value)
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                post_init_node,
                "Dataclass __post_init__ is incompatible with InitVar fields",
                error_code=ErrorCode.incompatible_override,
                detail=can_assign.display(),
            )

    def _check_dataclass_field_default_order(
        self,
        node: ast.ClassDef,
        dataclass_class: SyntheticClassObjectValue,
        decorator_values: DecoratorValues,
    ) -> None:
        raw_init = dataclass_class.class_attributes.get("%dataclass_init")
        if isinstance(raw_init, KnownValue) and raw_init.val is False:
            return
        error_node = next(
            (
                deco_node
                for deco_value, _, deco_node in decorator_values
                if deco_value == KnownValue(dataclass)
            ),
            node,
        )

        field_order = _known_string_sequence_values(
            dataclass_class.class_attributes.get("%dataclass_field_order")
        )
        if not field_order:
            return
        default_fields = set(
            _known_string_sequence_values(
                dataclass_class.class_attributes.get("%dataclass_default_fields")
            )
            or ()
        )
        init_false_fields = set(
            _known_string_sequence_values(
                dataclass_class.class_attributes.get("%dataclass_init_false_fields")
            )
            or ()
        )
        kw_only_fields = set(
            _known_string_sequence_values(
                dataclass_class.class_attributes.get("%dataclass_kw_only_fields")
            )
            or ()
        )

        saw_default = False
        for field_name in field_order:
            if field_name in init_false_fields or field_name in kw_only_fields:
                continue
            if field_name in default_fields:
                saw_default = True
                continue
            if saw_default:
                self._show_error_if_checking(
                    error_node,
                    "Dataclass fields without defaults cannot follow fields with defaults",
                    error_code=ErrorCode.invalid_annotation,
                )
                return

    def _synthetic_type_params_for_variance(
        self, typ: type | str, arity: int
    ) -> Sequence[TypeVarLike] | None:
        if not isinstance(typ, str):
            return None
        synthetic_class = self.checker.get_synthetic_class(typ)
        if synthetic_class is None:
            return None
        for base in synthetic_class.base_classes:
            runtime_annotation = self._runtime_annotation_from_value(base)
            origin = typing.get_origin(runtime_annotation)
            if origin is None or not (
                is_typing_name(origin, "Generic") or is_typing_name(origin, "Protocol")
            ):
                continue
            type_params = typing.get_args(runtime_annotation)
            if len(type_params) != arity:
                continue
            if all(
                is_instance_of_typing_name(type_param, "TypeVar")
                for type_param in type_params
            ):
                return cast(tuple[TypeVarLike, ...], type_params)
        return None

    def _value_for_variance_annotation(self, annotation: ast.expr) -> Value:
        annotation_expr = annotation_expr_from_ast(
            annotation, self, suppress_errors=True
        )
        value, _ = annotation_expr.maybe_unqualify(
            {
                Qualifier.ClassVar,
                Qualifier.Final,
                Qualifier.ReadOnly,
                Qualifier.Required,
                Qualifier.NotRequired,
                Qualifier.Unpack,
                Qualifier.InitVar,
                Qualifier.TypeAlias,
            }
        )
        if value is None:
            return AnyValue(AnySource.inference)
        return value

    def _collect_type_param_polarities_from_value(
        self,
        value: Value,
        type_param_polarities: Mapping[object, set[int]],
        *,
        polarity: int,
    ) -> None:
        if isinstance(value, TypeVarValue):
            used_polarities = type_param_polarities.get(value.typevar)
            if used_polarities is not None:
                _record_variance_polarity(used_polarities, polarity)
            return
        if isinstance(value, GenericValue):
            type_parameters = self.get_type_parameters(value.typ)
            if len(type_parameters) == len(value.args):
                for arg, type_param in zip(value.args, type_parameters):
                    if isinstance(type_param, TypeVarValue):
                        param_variance = type_param.variance
                    else:
                        param_variance = Variance.INVARIANT
                    self._collect_type_param_polarities_from_value(
                        arg,
                        type_param_polarities,
                        polarity=_compose_variance_polarity(polarity, param_variance),
                    )
            else:
                synthetic_type_params = self._synthetic_type_params_for_variance(
                    value.typ, len(value.args)
                )
                if synthetic_type_params is None:
                    for arg in value.args:
                        self._collect_type_param_polarities_from_value(
                            arg, type_param_polarities, polarity=0
                        )
                else:
                    for arg, type_param in zip(value.args, synthetic_type_params):
                        self._collect_type_param_polarities_from_value(
                            arg,
                            type_param_polarities,
                            polarity=_compose_variance_polarity(
                                polarity, get_typevar_variance(type_param)
                            ),
                        )
            return
        if isinstance(value, AnnotatedValue):
            self._collect_type_param_polarities_from_value(
                value.value, type_param_polarities, polarity=polarity
            )
            return
        if isinstance(value, MultiValuedValue):
            for subval in value.vals:
                self._collect_type_param_polarities_from_value(
                    subval, type_param_polarities, polarity=polarity
                )
            return
        if isinstance(value, TypeAliasValue):
            alias_type_params = value.alias.get_type_params()
            if value.type_arguments and len(alias_type_params) == len(
                value.type_arguments
            ):
                for alias_type_param, type_argument in zip(
                    alias_type_params, value.type_arguments
                ):
                    type_param = (
                        alias_type_param.typevar
                        if isinstance(alias_type_param, TypeVarValue)
                        else alias_type_param
                    )
                    self._collect_type_param_polarities_from_value(
                        type_argument,
                        type_param_polarities,
                        polarity=_compose_variance_polarity(
                            polarity, get_typevar_variance(type_param)
                        ),
                    )
                return
            self._collect_type_param_polarities_from_value(
                value.get_value(), type_param_polarities, polarity=polarity
            )
            return
        if isinstance(value, SubclassValue):
            self._collect_type_param_polarities_from_value(
                value.typ, type_param_polarities, polarity=polarity
            )
            return
        if isinstance(value, SequenceValue):
            members = value.get_member_sequence()
            if members is not None:
                for member in members:
                    self._collect_type_param_polarities_from_value(
                        member, type_param_polarities, polarity=polarity
                    )
            return
        if isinstance(value, CallableValue):
            signature = value.signature
            signatures = (
                signature.signatures
                if isinstance(signature, OverloadedSignature)
                else [signature]
            )
            for sig in signatures:
                if not isinstance(sig, Signature):
                    continue
                for param in sig.parameters.values():
                    self._collect_type_param_polarities_from_value(
                        param.annotation,
                        type_param_polarities,
                        polarity=-polarity if polarity else 0,
                    )
                self._collect_type_param_polarities_from_value(
                    sig.return_value, type_param_polarities, polarity=polarity
                )
            return

    def _infer_class_type_param_variances(
        self,
        node: ast.ClassDef,
        type_params: Sequence[TypeVarValue],
        base_values: Sequence[Value],
        *,
        is_protocol: bool = False,
        dataclass_semantics: DataclassInfo | None = None,
    ) -> Sequence[TypeVarValue]:
        if not type_params:
            return type_params
        type_param_polarities = {tp.typevar: set() for tp in type_params}
        for base_node in node.bases:
            base = self._value_for_variance_annotation(base_node)
            if is_protocol and _is_protocol_base(base):
                continue
            self._collect_type_param_polarities_from_value(
                base, type_param_polarities, polarity=1
            )

        frozen_dataclass = (
            dataclass_semantics is not None and dataclass_semantics.frozen
        )
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if statement.name in {"__init__", "__new__"}:
                    continue
                decorator_kinds = self._function_decorator_kinds_by_node.get(
                    statement, frozenset()
                )
                is_staticmethod = FunctionDecorator.staticmethod in decorator_kinds
                all_args = [
                    *statement.args.posonlyargs,
                    *statement.args.args,
                    *statement.args.kwonlyargs,
                ]
                if statement.args.vararg is not None:
                    all_args.append(statement.args.vararg)
                if statement.args.kwarg is not None:
                    all_args.append(statement.args.kwarg)
                for i, arg in enumerate(all_args):
                    if is_protocol and i == 0 and not is_staticmethod:
                        continue
                    if arg.annotation is None:
                        continue
                    annotation_value = self._value_for_variance_annotation(
                        arg.annotation
                    )
                    self._collect_type_param_polarities_from_value(
                        annotation_value, type_param_polarities, polarity=-1
                    )
                if statement.returns is not None:
                    return_value = self._value_for_variance_annotation(
                        statement.returns
                    )
                    self._collect_type_param_polarities_from_value(
                        return_value, type_param_polarities, polarity=1
                    )
            elif (
                isinstance(statement, ast.AnnAssign)
                and statement.annotation is not None
            ):
                if not isinstance(statement.target, ast.Name):
                    continue
                annotation_expr = self.expr_of_annotation(statement.annotation)
                value, qualifiers = annotation_expr.maybe_unqualify(
                    {Qualifier.ClassVar, Qualifier.Final, Qualifier.ReadOnly}
                )
                if value is None or Qualifier.ClassVar in qualifiers:
                    continue
                attribute_polarity = (
                    1
                    if frozen_dataclass
                    or Qualifier.Final in qualifiers
                    or Qualifier.ReadOnly in qualifiers
                    else 0
                )
                self._collect_type_param_polarities_from_value(
                    value, type_param_polarities, polarity=attribute_polarity
                )

        inferred_type_params = []
        for type_param in type_params:
            polarities = type_param_polarities[type_param.typevar]
            if polarities == {1} or (is_protocol and not polarities):
                variance = Variance.COVARIANT
            elif polarities == {-1}:
                variance = Variance.CONTRAVARIANT
            else:
                variance = Variance.INVARIANT
            inferred_type_params.append(replace(type_param, variance=variance))
        return inferred_type_params

    def _check_protocol_type_param_variances(
        self,
        node: ast.ClassDef,
        type_params: Sequence[TypeVarValue],
        base_values: Sequence[Value],
        class_scope_object: type | str | None,
    ) -> None:
        if sys.version_info >= (3, 12) and node.type_params:
            # PEP 695 class type parameters infer variance, so explicit
            # protocol variance checks for legacy TypeVars don't apply.
            return
        if not self._is_protocol_class(base_values, class_scope_object):
            return
        if not type_params and isinstance(class_scope_object, (type, str)):
            type_params = tuple(
                type_param
                for type_param in self.get_type_parameters(class_scope_object)
                if isinstance(type_param, TypeVarValue)
            )
        if not type_params:
            return
        checked_type_params = [
            type_param
            for type_param in type_params
            if is_instance_of_typing_name(type_param.typevar, "TypeVar")
        ]
        if not checked_type_params:
            return
        inferred_type_params = self._infer_class_type_param_variances(
            node,
            checked_type_params,
            base_values,
            is_protocol=True,
            dataclass_semantics=None,
        )
        for declared_type_param, inferred_type_param in zip(
            checked_type_params, inferred_type_params
        ):
            if declared_type_param.variance is inferred_type_param.variance:
                continue
            type_param_name = safe_getattr(
                declared_type_param.typevar,
                "__name__",
                str(declared_type_param.typevar),
            )
            self._show_error_if_checking(
                node,
                f"{type_param_name} should be "
                f"{inferred_type_param.variance.display_name()}",
                error_code=ErrorCode.invalid_annotation,
            )

    def _check_class_base_type_param_variances(
        self,
        node: ast.ClassDef,
        type_params: Sequence[TypeVarValue],
        base_values: Sequence[Value],
        class_scope_object: type | str | None,
    ) -> None:
        if self._is_protocol_class(base_values, class_scope_object):
            return
        analyzed_bases = [
            self._value_for_variance_annotation(base_node) for base_node in node.bases
        ]
        typevars_to_check = {
            type_param.typevar
            for type_param in type_params
            if is_instance_of_typing_name(type_param.typevar, "TypeVar")
        }
        for analyzed_base in analyzed_bases:
            for subval in flatten_values(analyzed_base):
                for walked in self._walk_values_for_variance_typevar_collection(subval):
                    if not isinstance(walked, TypeVarValue):
                        continue
                    if not is_instance_of_typing_name(walked.typevar, "TypeVar"):
                        continue
                    typevars_to_check.add(walked.typevar)
        if not typevars_to_check:
            return

        sorted_typevars = sorted(typevars_to_check, key=str)
        for base_node, analyzed_base in zip(node.bases, analyzed_bases):
            if _is_variance_declaration_base(analyzed_base):
                continue
            type_param_polarities: dict[object, set[int]] = {
                typevar: set() for typevar in sorted_typevars
            }
            for subval in flatten_values(analyzed_base):
                self._collect_type_param_polarities_from_value(
                    subval, type_param_polarities, polarity=1
                )
            for typevar in sorted_typevars:
                used_polarities = type_param_polarities[typevar]
                variance = get_typevar_variance(typevar)
                if _variance_is_compatible_with_usage(variance, used_polarities):
                    continue
                type_param_name = safe_getattr(typevar, "__name__", str(typevar))
                self._show_error_if_checking(
                    base_node,
                    f"{type_param_name} has incompatible variance in base class",
                    error_code=ErrorCode.invalid_annotation,
                )
                break

    def _walk_values_for_variance_typevar_collection(
        self, value: Value
    ) -> Iterable[Value]:
        if isinstance(value, TypeAliasValue):
            yield value
            for type_argument in value.type_arguments:
                yield from self._walk_values_for_variance_typevar_collection(
                    type_argument
                )
            return
        for walked in value.walk_values():
            if walked is value:
                yield walked
            elif isinstance(walked, TypeAliasValue):
                yield from self._walk_values_for_variance_typevar_collection(walked)
            else:
                yield walked

    def _is_protocol_class(
        self, base_values: Sequence[Value], class_scope_object: type | str | None
    ) -> bool:
        if isinstance(class_scope_object, (type, str)):
            if self.checker.make_type_object(class_scope_object).is_protocol:
                return True
        return any(_is_protocol_base(base_value) for base_value in base_values)

    def _align_type_params_with_runtime_class(
        self, class_obj: type | None, type_params: Sequence[TypeVarValue]
    ) -> Sequence[TypeVarValue]:
        if class_obj is None:
            return type_params
        runtime_type_params = safe_getattr(class_obj, "__type_params__", ())
        if len(runtime_type_params) != len(type_params):
            return type_params
        aligned = []
        for type_param, runtime_type_param in zip(type_params, runtime_type_params):
            aligned.append(replace(type_param, typevar=runtime_type_param))
        return aligned

    def _type_param_has_default(self, type_param: TypeVarValue) -> bool:
        if type_param.default is not None:
            return True
        runtime_default = safe_getattr(type_param.typevar, "__default__", NoDefault)
        return runtime_default is not NoDefault

    def _check_class_type_param_default_rules(
        self, node: ast.ClassDef, type_params: Sequence[TypeVarValue]
    ) -> None:
        seen_default = False
        previous_was_typevartuple = False
        for type_param in type_params:
            has_default = self._type_param_has_default(type_param)
            is_typevartuple = type_param.is_typevartuple
            is_typevar = is_instance_of_typing_name(type_param.typevar, "TypeVar")
            if seen_default and not has_default:
                self._show_error_if_checking(
                    node,
                    "non-default TypeVars cannot follow ones with defaults",
                    error_code=ErrorCode.invalid_annotation,
                )
                return
            if previous_was_typevartuple and has_default and is_typevar:
                self._show_error_if_checking(
                    node,
                    "TypeVars with defaults cannot follow TypeVarTuples",
                    error_code=ErrorCode.invalid_annotation,
                )
                return
            seen_default = seen_default or has_default
            previous_was_typevartuple = is_typevartuple

    def _type_param_from_expr_for_default_rules(
        self, node: ast.expr
    ) -> TypeVarValue | None:
        if isinstance(node, ast.Starred):
            return self._type_param_from_expr_for_default_rules(node.value)
        if isinstance(node, ast.Subscript):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "Unpack"
                or isinstance(node.value, ast.Attribute)
                and node.value.attr == "Unpack"
            ):
                unpacked = node.slice
                if isinstance(unpacked, ast.Tuple) and len(unpacked.elts) == 1:
                    unpacked = unpacked.elts[0]
                return self._type_param_from_expr_for_default_rules(unpacked)
            return None
        if not isinstance(node, ast.Name):
            return None
        value = self.scopes.get(node.id, node, self.state, can_assign_ctx=self)
        return _type_param_value_from_value(value)

    def _type_params_from_base_annotations_for_default_rules(
        self, base_nodes: Sequence[ast.expr]
    ) -> Sequence[TypeVarValue]:
        seen: set[object] = set()
        type_params: list[TypeVarValue] = []
        for base_node in base_nodes:
            if not isinstance(base_node, ast.Subscript):
                continue
            maybe_members = (
                base_node.slice.elts
                if isinstance(base_node.slice, ast.Tuple)
                else [base_node.slice]
            )
            for member in maybe_members:
                maybe_type_param = self._type_param_from_expr_for_default_rules(member)
                if maybe_type_param is None:
                    continue
                if maybe_type_param.typevar in seen:
                    continue
                seen.add(maybe_type_param.typevar)
                type_params.append(maybe_type_param)
        return type_params

    def _type_params_from_base_values(
        self, base_values: Sequence[Value]
    ) -> Sequence[TypeVarValue]:
        seen: set[object] = set()
        type_params = list(
            self._type_params_from_protocol_shorthand_base_values(base_values)
        )
        for type_param in type_params:
            seen.add(type_param.typevar)
        for base in base_values:
            for subval in flatten_values(base):
                for walked in subval.walk_values():
                    if isinstance(walked, TypeVarValue):
                        type_param = walked
                    elif isinstance(walked, InputSigValue) and isinstance(
                        walked.input_sig, ParamSpecSig
                    ):
                        type_param = TypeVarValue(walked.input_sig.param_spec)
                    else:
                        continue
                    if type_param.typevar in seen:
                        continue
                    seen.add(type_param.typevar)
                    type_params.append(type_param)
        return type_params

    def _order_type_params_by_base_annotation_appearance(
        self, base_nodes: Sequence[ast.expr], type_params: Sequence[TypeVarValue]
    ) -> Sequence[TypeVarValue]:
        annotation_order = self._type_params_from_base_annotations_for_default_rules(
            base_nodes
        )
        if not annotation_order:
            return type_params
        by_identity = {type_param.typevar: type_param for type_param in type_params}
        merged = [
            by_identity.get(type_param.typevar, type_param)
            for type_param in annotation_order
        ]
        seen = {type_param.typevar for type_param in merged}
        for type_param in type_params:
            if type_param.typevar in seen:
                continue
            seen.add(type_param.typevar)
            merged.append(type_param)
        return merged

    def _type_params_from_protocol_shorthand_base_values(
        self, base_values: Sequence[Value]
    ) -> Sequence[TypeVarValue]:
        """Return type parameters from a ``Protocol[...]`` shorthand base.

        The typing spec dictates that this base controls parameter ordering.
        """
        seen: set[object] = set()
        type_params: list[TypeVarValue] = []
        for base in base_values:
            for subval in flatten_values(base):
                maybe_type_params: list[TypeVarValue] = []
                if isinstance(subval, GenericValue) and is_typing_name(
                    subval.typ, "Protocol"
                ):
                    for arg in subval.args:
                        maybe_type_param = _type_param_value_from_value(arg)
                        if maybe_type_param is not None:
                            maybe_type_params.append(maybe_type_param)
                else:
                    runtime_annotation = self._runtime_annotation_from_value(subval)
                    origin = typing.get_origin(runtime_annotation)
                    if origin is not None and is_typing_name(origin, "Protocol"):
                        for runtime_arg in typing.get_args(runtime_annotation):
                            if not (
                                is_instance_of_typing_name(runtime_arg, "TypeVar")
                                or is_instance_of_typing_name(
                                    runtime_arg, "TypeVarTuple"
                                )
                                or is_instance_of_typing_name(runtime_arg, "ParamSpec")
                            ):
                                continue
                            maybe_type_params.append(
                                TypeVarValue(
                                    runtime_arg,
                                    variance=get_typevar_variance(runtime_arg),
                                    is_typevartuple=is_instance_of_typing_name(
                                        runtime_arg, "TypeVarTuple"
                                    ),
                                )
                            )
                if maybe_type_params and not all(
                    is_instance_of_typing_name(type_param.typevar, "TypeVar")
                    for type_param in maybe_type_params
                ):
                    continue
                for maybe_type_param in maybe_type_params:
                    if maybe_type_param.typevar in seen:
                        continue
                    seen.add(maybe_type_param.typevar)
                    type_params.append(maybe_type_param)
                if type_params:
                    return type_params
        return type_params

    def _type_params_from_base_values_for_methods(
        self, base_values: Sequence[Value]
    ) -> Sequence[TypeVarValue]:
        type_params_from_bases = list(self._type_params_from_base_values(base_values))
        if type_params_from_bases and all(
            not (
                is_instance_of_typing_name(type_param.typevar, "ParamSpec")
                or type_param.is_typevartuple
            )
            for type_param in type_params_from_bases
        ):
            return type_params_from_bases
        seen: set[object] = set()
        type_params: list[TypeVarValue] = []
        for type_param in type_params_from_bases:
            if type_param.typevar in seen:
                continue
            seen.add(type_param.typevar)
            type_params.append(type_param)
        for base in base_values:
            for subval in flatten_values(base):
                runtime_annotation = self._runtime_annotation_from_value(subval)
                for runtime_arg in typing.get_args(runtime_annotation):
                    if not (
                        is_instance_of_typing_name(runtime_arg, "TypeVar")
                        or is_instance_of_typing_name(runtime_arg, "TypeVarTuple")
                        or is_instance_of_typing_name(runtime_arg, "ParamSpec")
                    ):
                        continue
                    if runtime_arg in seen:
                        continue
                    seen.add(runtime_arg)
                    type_params.append(
                        TypeVarValue(
                            runtime_arg,
                            variance=get_typevar_variance(runtime_arg),
                            is_typevartuple=is_instance_of_typing_name(
                                runtime_arg, "TypeVarTuple"
                            ),
                        )
                    )
        return type_params

    def _base_values_for_generic_analysis(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> Sequence[Value]:
        if self.module is not None:
            return base_values
        if not any(isinstance(base_node, ast.Subscript) for base_node in node.bases):
            return base_values
        if node.bases and all(
            isinstance(base_node, ast.Subscript)
            and self._is_type_parameter_base(base_value)
            for base_node, base_value in zip(node.bases, base_values)
        ):
            # Generic[T] / Protocol[T] bases usually preserve enough runtime
            # information even in static fallback mode.
            return base_values
        analyzed_bases = [
            self._value_for_variance_annotation(base_node) for base_node in node.bases
        ]
        if analyzed_bases:
            return analyzed_bases
        return base_values

    def _check_duplicate_type_params_in_generic_bases(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        analyzed_bases = self._base_values_for_generic_analysis(node, base_values)
        for base_node, base_value in zip(node.bases, analyzed_bases):
            if not isinstance(base_node, ast.Subscript):
                continue
            if not self._is_generic_type_parameter_base(base_value):
                continue
            seen: set[object] = set()
            for arg_value in self._type_param_base_arg_values(base_node.slice):
                type_param = _type_param_identity(arg_value)
                if type_param is None:
                    continue
                if type_param in seen:
                    self._show_error_if_checking(
                        base_node.slice,
                        "Type parameter list cannot contain duplicate type variables",
                        error_code=ErrorCode.invalid_annotation,
                    )
                    return
                seen.add(type_param)

    def _check_inconsistent_generic_base_specialization(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        analyzed_bases = self._base_values_for_generic_analysis(node, base_values)
        seen_mappings: dict[type | str, dict[TypeVarLike, Value]] = {}
        for base_node, base_value in zip(node.bases, analyzed_bases):
            for subval in flatten_values(replace_fallback(base_value)):
                converted: Value = subval
                if isinstance(converted, KnownValue):
                    converted = self.arg_spec_cache._type_from_base(converted.val)
                elif isinstance(converted, SyntheticClassObjectValue):
                    converted = converted.class_type
                if not isinstance(converted, TypedValue):
                    continue
                base_typ = converted.typ
                generic_args = (
                    converted.args if isinstance(converted, GenericValue) else ()
                )
                for gb_typ, tv_map in self.checker.get_generic_bases(
                    base_typ, generic_args
                ).items():
                    existing_map = seen_mappings.setdefault(gb_typ, {})
                    for type_param, value in tv_map.items():
                        existing = existing_map.get(type_param)
                        if existing is None:
                            existing_map[type_param] = value
                        elif (
                            existing != value
                            and not isinstance(existing, AnyValue)
                            and not isinstance(value, AnyValue)
                        ):
                            existing_identity = _type_param_identity(existing)
                            value_identity = _type_param_identity(value)
                            if (
                                existing_identity is None
                                or value_identity is None
                                or existing_identity == value_identity
                            ):
                                continue
                            self._show_error_if_checking(
                                base_node,
                                "Inconsistent type variable order in base classes",
                                error_code=ErrorCode.invalid_annotation,
                            )
                            return

    def _check_typevartuple_usage_in_type_parameter_bases(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        for base, base_value in zip(node.bases, base_values):
            if not isinstance(base, ast.Subscript):
                continue
            if self._is_type_parameter_base(base_value):
                is_type_param_base = True
            else:
                root_value = safe_getattr(base.value, "inferred_value", None)
                if root_value is None:
                    root_value = value_from_ast(
                        base.value, visitor=self, error_on_unrecognized=False
                    )
                is_type_param_base = self._is_type_parameter_base(root_value)
            if not is_type_param_base:
                continue
            arg_values = self._type_param_base_arg_values(base.slice)
            bare_count = 0
            unpacked_count = 0
            for arg_value in arg_values:
                bare, unpacked = _count_typevartuple_type_param_arg(arg_value)
                bare_count += bare
                unpacked_count += unpacked
            bare_count = max(bare_count, self._count_typevartuple_name_args(base.slice))
            starred_count = _count_starred_type_param_args(base.slice)
            if bare_count:
                self._show_error_if_checking(
                    base.slice,
                    "TypeVarTuple must be unpacked",
                    error_code=ErrorCode.invalid_annotation,
                )
            if unpacked_count > 1 or starred_count > 1:
                self._show_error_if_checking(
                    base.slice,
                    "Only one TypeVarTuple can be used in a type parameter list",
                    error_code=ErrorCode.invalid_annotation,
                )

    def _check_pep695_type_parameter_base_compatibility(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        analyzed_bases = self._base_values_for_generic_analysis(node, base_values)
        for base_node, base_value in zip(node.bases, analyzed_bases):
            if not isinstance(base_node, ast.Subscript):
                continue
            if not self._is_type_parameter_base(base_value):
                continue
            self._show_error_if_checking(
                base_node,
                "Class definition cannot specialize Generic or Protocol bases"
                " when using type parameter syntax",
                error_code=ErrorCode.invalid_annotation,
            )
            return

    def _type_param_base_arg_values(self, slice_node: ast.AST) -> Sequence[Value]:
        args = safe_getattr(slice_node, "inferred_value", None)
        if args is None:
            args = value_from_ast(slice_node, visitor=self, error_on_unrecognized=False)
        if isinstance(args, SequenceValue) and args.typ is tuple:
            members = args.get_member_sequence()
            if members is not None:
                return members
            return [member for _, member in args.members]
        return [args]

    def _count_typevartuple_name_args(self, slice_node: ast.AST) -> int:
        if isinstance(slice_node, ast.Tuple):
            arg_nodes: Sequence[ast.AST] = slice_node.elts
        else:
            arg_nodes = [slice_node]
        count = 0
        for arg_node in arg_nodes:
            if not isinstance(arg_node, ast.Name):
                continue
            resolved, _ = self.resolve_name(arg_node, suppress_errors=True)
            if _is_typevartuple_annotation_value(resolved):
                count += 1
        return count

    def _is_type_parameter_base(self, value: Value) -> bool:
        for subval in flatten_values(replace_fallback(value)):
            candidate: object
            if isinstance(subval, SyntheticClassObjectValue):
                subval = subval.class_type
            if isinstance(subval, GenericValue):
                candidate = subval.typ
            elif isinstance(subval, TypedValue):
                candidate = subval.typ
            elif isinstance(subval, KnownValue):
                candidate = subval.val
            else:
                continue
            origin = get_origin(candidate)
            if origin is not None:
                candidate = origin
            if is_typing_name(candidate, "Generic") or is_typing_name(
                candidate, "Protocol"
            ):
                return True
        return False

    def _is_generic_type_parameter_base(self, value: Value) -> bool:
        for subval in flatten_values(replace_fallback(value)):
            candidate: object
            if isinstance(subval, SyntheticClassObjectValue):
                subval = subval.class_type
            if isinstance(subval, GenericValue):
                candidate = subval.typ
            elif isinstance(subval, TypedValue):
                candidate = subval.typ
            elif isinstance(subval, KnownValue):
                candidate = subval.val
            else:
                continue
            origin = get_origin(candidate)
            if origin is not None:
                candidate = origin
            if is_typing_name(candidate, "Generic"):
                return True
        return False

    def _check_protocol_base_validity(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        if not any(_is_protocol_base(base_value) for base_value in base_values):
            return
        for base_node, base_value in zip(node.bases, base_values):
            if _is_protocol_base(base_value):
                continue
            if self._is_valid_non_protocol_base_for_protocol(base_value):
                continue
            self._show_error_if_checking(
                base_node,
                "Protocols can only inherit from protocol bases",
                error_code=ErrorCode.invalid_base,
            )

    def _is_valid_non_protocol_base_for_protocol(self, base_value: Value) -> bool:
        for subval in flatten_values(replace_fallback(base_value)):
            if isinstance(subval, SyntheticClassObjectValue):
                subval = subval.class_type
            if isinstance(subval, TypedValue):
                typ = subval.typ
                if is_typing_name(typ, "Generic"):
                    return True
                if isinstance(typ, str):
                    return self.checker.make_type_object(typ).is_protocol
                if isinstance(typ, type):
                    # For runtime classes, let runtime semantics decide.
                    return True
                continue
            if isinstance(subval, KnownValue):
                if is_typing_name(subval.val, "Generic"):
                    return True
                if isinstance(subval.val, type):
                    # For runtime classes, let runtime semantics decide.
                    return True
        return True

    def _get_local_object(self, name: str, node: ast.AST) -> Value:
        if self.scopes.scope_type() == ScopeType.module_scope:
            return self.scopes.get(name, node, self.state, can_assign_ctx=self)
        elif (
            self.scopes.scope_type() == ScopeType.class_scope
            and self.current_class is not None
            and hasattr(self.current_class, "__dict__")
        ):
            runtime_obj = self.current_class.__dict__.get(name)
            if isinstance(runtime_obj, type):
                return KnownValue(runtime_obj)
        return AnyValue(AnySource.inference)

    def _get_current_class_object(self, node: ast.ClassDef) -> type | None:
        cls_obj = self._get_local_object(node.name, node)

        module = self.module
        if isinstance(cls_obj, MultiValuedValue) and module is not None:
            # if there are multiple, see if there is only one that matches this module
            possible_values = [
                val
                for val in cls_obj.vals
                if isinstance(val, KnownValue)
                and isinstance(val.val, type)
                and safe_getattr(val.val, "__module__", None) == module.__name__
            ]
            if len(possible_values) == 1:
                cls_obj = possible_values[0]

        if isinstance(cls_obj, KnownValue):
            cls_obj = KnownValue(UnwrapClass.unwrap(cls_obj.val, self.options))
            current_class = cls_obj.val
            if isinstance(current_class, type):
                self._record_class_examined(current_class)
                return current_class
            else:
                return None
        else:
            return None

    def _get_synthetic_class_fq_name(self, node: ast.ClassDef) -> str:
        return self._get_synthetic_class_fq_name_from_name(node.name)

    def _get_synthetic_class_fq_name_from_name(self, name: str) -> str:
        if self.module is not None and hasattr(self.module, "__name__"):
            module_name = self.module.__name__
        else:
            module_name = self.filename

        qualname = self._get_class_qualname_from_name(name)
        if module_name:
            return ".".join((module_name, qualname))
        return qualname

    def _get_class_qualname_from_name(self, name: str) -> str:
        qualname_parts = []
        for context in self.node_context.contexts:
            if isinstance(context, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname_parts.extend((context.name, "<locals>"))
            elif isinstance(context, ast.Lambda):
                qualname_parts.extend(("<lambda>", "<locals>"))
            elif isinstance(context, ast.ClassDef):
                qualname_parts.append(context.name)
        if not qualname_parts or qualname_parts[-1] != name:
            qualname_parts.append(name)
        return ".".join(qualname_parts)

    def _get_synthetic_method_attributes(self, node: ast.ClassDef) -> set[str]:
        method_attributes = set()
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_attributes.add(
                    _mangle_class_attribute_name(node.name, stmt.name)
                )
        return method_attributes

    def _get_synthetic_staticmethod_attributes(self, node: ast.ClassDef) -> set[str]:
        staticmethod_attributes = set()
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_kinds = self._function_decorator_kinds_by_node.get(
                stmt, frozenset()
            )
            if FunctionDecorator.staticmethod not in decorator_kinds:
                continue
            staticmethod_attributes.add(
                _mangle_class_attribute_name(node.name, stmt.name)
            )
        return staticmethod_attributes

    def _get_synthetic_classmethod_attributes(self, node: ast.ClassDef) -> set[str]:
        classmethod_attributes = set()
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_kinds = self._function_decorator_kinds_by_node.get(
                stmt, frozenset()
            )
            if FunctionDecorator.classmethod not in decorator_kinds:
                continue
            classmethod_attributes.add(
                _mangle_class_attribute_name(node.name, stmt.name)
            )
        return classmethod_attributes

    def _return_annotation_node_contains_self(self, annotation: ast.AST | None) -> bool:
        if annotation is None:
            return False
        if isinstance(annotation, ast.Name):
            return annotation.id == "Self"
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == "Self"
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval")
            except SyntaxError:
                return False
            return self._return_annotation_node_contains_self(parsed.body)
        return any(
            self._return_annotation_node_contains_self(child)
            for child in ast.iter_child_nodes(annotation)
        )

    def _get_synthetic_self_returning_classmethods(
        self, node: ast.ClassDef
    ) -> set[str]:
        self_returning_classmethods = set()
        for stmt in node.body:
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_kinds = self._function_decorator_kinds_by_node.get(
                stmt, frozenset()
            )
            if FunctionDecorator.classmethod not in decorator_kinds:
                continue
            if not self._function_returns_self_by_node.get(stmt, False):
                continue
            self_returning_classmethods.add(
                _mangle_class_attribute_name(node.name, stmt.name)
            )
        return self_returning_classmethods

    def _current_class_name_from_context(self) -> str | None:
        for context in reversed(self.node_context.contexts):
            if isinstance(context, ast.ClassDef):
                return context.name
        return None

    def _is_current_class_dataclass(self) -> bool:
        return self.current_dataclass_info is not None

    def _visit_class_and_get_value(
        self, node: ast.ClassDef, current_class: type | str | None
    ) -> tuple[Value, Mapping[str, Value] | None]:
        if self._is_collecting():
            # If this is a nested class, we need to run the collecting phase to get data
            # about names accessed from the class.
            if (
                len(self.scopes.scopes) > 2
                or self.current_synthetic_typeddict is not None
            ):
                with (
                    self.scopes.add_scope(
                        ScopeType.class_scope,
                        scope_node=node,
                        scope_object=current_class,
                    ),
                    self._set_current_class(current_class),
                ):
                    self._generic_visit_list(node.body)
            return AnyValue(AnySource.inference), None
        else:
            with (
                self.scopes.add_scope(
                    ScopeType.class_scope, scope_node=None, scope_object=current_class
                ),
                self._set_current_class(current_class),
            ):
                self._generic_visit_list(node.body)
                self._flush_pending_overload_block_for_scope(
                    self.scopes.current_scope()
                )
                class_scope_values = dict(self.scopes.current_scope().variables)

            if isinstance(current_class, type):
                return KnownValue(current_class), class_scope_values
            return AnyValue(AnySource.inference), class_scope_values

        return AnyValue(AnySource.inference), None

    def _build_synthetic_typeddict_value(
        self, context: _SyntheticTypedDictContext, node: ast.ClassDef
    ) -> TypedDictValue:
        items: dict[str, TypedDictEntry] = {}
        for base in context.bases:
            for key, incoming in base.items.items():
                if key not in items:
                    items[key] = incoming
                    continue
                merged = self._merge_typeddict_base_entries(
                    key, items[key], incoming, node
                )
                if merged is not None:
                    items[key] = merged

        for key, (entry, entry_node) in context.local_items.items():
            if key in items:
                if not self._is_valid_typeddict_override(key, items[key], entry):
                    self._show_error_if_checking(
                        entry_node,
                        f"Incompatible TypedDict override for key {key!r}",
                        error_code=ErrorCode.invalid_annotation,
                    )
                items[key] = entry
            else:
                base_extra_keys = context.inherited_extra_keys
                if base_extra_keys is NO_RETURN_VALUE:
                    self._show_error_if_checking(
                        entry_node,
                        f'"{node.name}" is a closed TypedDict; extra key {key!r} not allowed',
                        error_code=ErrorCode.invalid_annotation,
                    )
                elif base_extra_keys is not None:
                    if context.inherited_extra_keys_readonly:
                        can_assign = has_relation(
                            base_extra_keys, entry.typ, Relation.ASSIGNABLE, self
                        )
                        if isinstance(can_assign, CanAssignError):
                            self._show_error_if_checking(
                                entry_node,
                                f"{entry.typ} is not assignable to 'extra_items'"
                                f" type {base_extra_keys}",
                                error_code=ErrorCode.invalid_annotation,
                            )
                    else:
                        if entry.required:
                            self._show_error_if_checking(
                                entry_node,
                                f"Required key {key!r} is not known to base TypedDict",
                                error_code=ErrorCode.invalid_annotation,
                            )
                        base_to_entry = has_relation(
                            base_extra_keys, entry.typ, Relation.ASSIGNABLE, self
                        )
                        entry_to_base = has_relation(
                            entry.typ, base_extra_keys, Relation.ASSIGNABLE, self
                        )
                        if isinstance(base_to_entry, CanAssignError) or isinstance(
                            entry_to_base, CanAssignError
                        ):
                            self._show_error_if_checking(
                                entry_node,
                                f"{entry.typ} is not consistent with 'extra_items'"
                                f" type {base_extra_keys}",
                                error_code=ErrorCode.invalid_annotation,
                            )
                items[key] = entry

        return TypedDictValue(
            items=items,
            extra_keys=context.extra_keys,
            extra_keys_readonly=context.extra_keys_readonly,
        )

    def _merge_typeddict_base_entries(
        self, key: str, left: TypedDictEntry, right: TypedDictEntry, node: ast.AST
    ) -> TypedDictEntry | None:
        if left.readonly != right.readonly:
            self._show_error_if_checking(
                node,
                f"TypedDict base classes define key {key!r} with incompatible mutability",
                error_code=ErrorCode.invalid_annotation,
            )
            return None

        if left.readonly:
            if left.required != right.required:
                self._show_error_if_checking(
                    node,
                    f"TypedDict base classes define key {key!r} with incompatible requiredness",
                    error_code=ErrorCode.invalid_annotation,
                )
                return None
            if isinstance(
                has_relation(left.typ, right.typ, Relation.ASSIGNABLE, self),
                CanAssignError,
            ):
                if isinstance(
                    has_relation(right.typ, left.typ, Relation.ASSIGNABLE, self),
                    CanAssignError,
                ):
                    self._show_error_if_checking(
                        node,
                        f"TypedDict base classes define key {key!r} with incompatible types",
                        error_code=ErrorCode.invalid_annotation,
                    )
                    return None
                return right
            return left

        if left.required != right.required:
            self._show_error_if_checking(
                node,
                f"TypedDict base classes define key {key!r} with incompatible requiredness",
                error_code=ErrorCode.invalid_annotation,
            )
            return None
        if isinstance(
            has_relation(left.typ, right.typ, Relation.ASSIGNABLE, self), CanAssignError
        ) or isinstance(
            has_relation(right.typ, left.typ, Relation.ASSIGNABLE, self), CanAssignError
        ):
            self._show_error_if_checking(
                node,
                f"TypedDict base classes define key {key!r} with incompatible types",
                error_code=ErrorCode.invalid_annotation,
            )
            return None
        return left

    def _is_valid_typeddict_override(
        self, key: str, base: TypedDictEntry, override_entry: TypedDictEntry
    ) -> bool:
        if not base.readonly:
            if override_entry.readonly:
                return False
            if base.required != override_entry.required:
                return False
            if isinstance(
                has_relation(base.typ, override_entry.typ, Relation.ASSIGNABLE, self),
                CanAssignError,
            ):
                return False
            if isinstance(
                has_relation(override_entry.typ, base.typ, Relation.ASSIGNABLE, self),
                CanAssignError,
            ):
                return False
            return True

        if base.required and not override_entry.required:
            return False
        if isinstance(
            has_relation(base.typ, override_entry.typ, Relation.SUBTYPE, self),
            CanAssignError,
        ):
            return False
        return True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Value:
        return self.visit_FunctionDef(node)

    @contextlib.contextmanager
    def compute_function_info(
        self,
        node: FunctionNode,
        *,
        is_nested_in_class: bool = False,
        enclosing_class: TypedValue | None = None,
        potential_function: object | None = None,
    ) -> Generator[FunctionInfo, None, None]:
        """Visits a function's decorator list."""
        async_kind = AsyncFunctionKind.non_async
        decorator_kinds: set[FunctionDecorator] = set()
        decorators = []
        for decorator in [] if isinstance(node, ast.Lambda) else node.decorator_list:
            # We have to descend into the Call node because the result of
            # asynq.asynq() is a one-off function that we can't test against.
            # This means that the decorator will be visited more than once, which seems OK.
            if isinstance(decorator, ast.Call):
                decorator_value = self.visit_expression(decorator)
                callee = self.visit_expression(decorator.func)
                if isinstance(callee, KnownValue):
                    if AsynqDecorators.contains(callee.val, self.options):
                        if any(kw.arg == "pure" for kw in decorator.keywords):
                            async_kind = AsyncFunctionKind.pure
                        else:
                            async_kind = AsyncFunctionKind.normal
                    elif AsyncProxyDecorators.contains(callee.val, self.options):
                        # @async_proxy(pure=True) is a noop, so don't treat it specially
                        if not any(kw.arg == "pure" for kw in decorator.keywords):
                            async_kind = AsyncFunctionKind.async_proxy
                decorators.append((callee, decorator_value, decorator))
            else:
                decorator_value = self.visit_expression(decorator)
                if isinstance(decorator_value, KnownValue):
                    val = decorator_value.val
                    if val is classmethod:
                        decorator_kinds.add(FunctionDecorator.classmethod)
                    elif val is staticmethod:
                        decorator_kinds.add(FunctionDecorator.staticmethod)
                    elif sys.version_info < (3, 11) and val is (
                        asyncio.coroutine  # static analysis: ignore[undefined_attribute]
                    ):
                        decorator_kinds.add(FunctionDecorator.decorated_coroutine)
                    elif val is real_overload or val is overload:
                        decorator_kinds.add(FunctionDecorator.overload)
                    elif val is abstractmethod:
                        decorator_kinds.add(FunctionDecorator.abstractmethod)
                    elif val is evaluated:
                        decorator_kinds.add(FunctionDecorator.evaluated)
                    elif is_typing_name(val, "override"):
                        decorator_kinds.add(FunctionDecorator.override)
                    elif is_typing_name(val, "final"):
                        decorator_kinds.add(FunctionDecorator.final)
                decorators.append((decorator_value, decorator_value, decorator))
        if (
            sys.version_info >= (3, 12)
            and not isinstance(node, ast.Lambda)
            and node.type_params
        ):
            ctx = self.scopes.add_scope(
                ScopeType.annotation_scope,
                scope_node=node,
                scope_object=potential_function,
            )
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            declared_type_params = cast(
                Sequence[ast.AST], getattr(node, "type_params", ())
            )
            if (
                sys.version_info >= (3, 12)
                and not isinstance(node, ast.Lambda)
                and declared_type_params
            ):
                type_params = self.visit_type_param_values(declared_type_params)
            else:
                type_params = []
            if type_params and not isinstance(node, ast.Lambda):
                annotation_nodes: list[ast.AST] = [*declared_type_params]
                annotation_nodes.extend(
                    arg.annotation
                    for arg in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    )
                    if arg.annotation is not None
                )
                if (
                    node.args.vararg is not None
                    and node.args.vararg.annotation is not None
                ):
                    annotation_nodes.append(node.args.vararg.annotation)
                if (
                    node.args.kwarg is not None
                    and node.args.kwarg.annotation is not None
                ):
                    annotation_nodes.append(node.args.kwarg.annotation)
                if node.returns is not None:
                    annotation_nodes.append(node.returns)
                legacy_typevars = self._legacy_typevars_in_nodes(
                    annotation_nodes, type_params
                )
                if legacy_typevars:
                    self._show_error_if_checking(
                        node,
                        "Function definition cannot combine old-style TypeVar"
                        " declarations with type parameter syntax",
                        error_code=ErrorCode.invalid_annotation,
                    )
            params = compute_parameters(
                node,
                enclosing_class,
                self,
                is_nested_in_class=is_nested_in_class,
                is_classmethod=FunctionDecorator.classmethod in decorator_kinds,
                is_staticmethod=FunctionDecorator.staticmethod in decorator_kinds,
                declared_type_params=type_params,
            )
            if isinstance(node, ast.Lambda) or node.returns is None:
                return_annotation = None
            else:
                return_annotation = self.value_of_annotation(node.returns)
                if isinstance(return_annotation, InputSigValue):
                    if isinstance(return_annotation.input_sig, ParamSpecSig):
                        self.show_error(
                            node.returns,
                            "ParamSpec cannot be used in this annotation context",
                            error_code=ErrorCode.invalid_annotation,
                        )
                    else:
                        self.show_error(
                            node.returns,
                            f"Unrecognized annotation {return_annotation}",
                            error_code=ErrorCode.invalid_annotation,
                        )
                    return_annotation = AnyValue(AnySource.error)
            yield FunctionInfo(
                async_kind=async_kind,
                decorator_kinds=frozenset(decorator_kinds),
                is_nested_in_class=is_nested_in_class,
                decorators=decorators,
                node=node,
                params=params,
                return_annotation=return_annotation,
                potential_function=potential_function,
                type_params=type_params,
            )

    def visit_FunctionDef(self, node: FunctionDefNode) -> Value:
        potential_function = self._get_potential_function(node)
        with self.compute_function_info(
            node,
            # If we set the current_class in the collecting phase,
            # the self argument of nested methods with an unannotated
            # first argument is incorrectly inferred.
            enclosing_class=self._get_enclosing_class_value_for_method(),
            is_nested_in_class=self.node_context.includes(ast.ClassDef),
            potential_function=potential_function,
        ) as info:
            self._function_decorator_kinds_by_node[node] = info.decorator_kinds
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                contains_self_return = _return_annotation_contains_self(
                    info.return_annotation
                )
                if not contains_self_return:
                    contains_self_return = self._return_annotation_node_contains_self(
                        node.returns
                    )
                self._function_returns_self_by_node[node] = contains_self_return
            self.yield_checker.reset_yield_checks()

            direct_dataclass_transform_info: DataclassTransformInfo | None = None
            if not isinstance(node, ast.Lambda):
                direct_dataclass_transform_info = (
                    self._get_direct_dataclass_transform_info(info.decorators)
                )

            if FunctionDecorator.final in info.decorator_kinds:
                in_class = self.node_context.includes(ast.ClassDef)
                if not in_class:
                    self._show_error_if_checking(
                        node,
                        "@final is not allowed on non-method functions",
                        error_code=ErrorCode.invalid_annotation,
                    )
                elif (
                    FunctionDecorator.overload in info.decorator_kinds
                    and not self.filename.endswith(".pyi")
                ):
                    self._show_error_if_checking(
                        node,
                        "@final should be applied only to the overload implementation",
                        error_code=ErrorCode.invalid_annotation,
                    )
                class_key = self.current_class_key
                if class_key is not None and (
                    FunctionDecorator.overload not in info.decorator_kinds
                    or self.filename.endswith(".pyi")
                ):
                    self._record_final_member(class_key, node.name)

            if node.returns is None:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.missing_return_annotation
                )

            info_for_computed_value = info
            if FunctionDecorator.overload in info.decorator_kinds:
                decorators = [
                    decorator
                    for decorator in info.decorators
                    if not self._is_overload_decorator(decorator[0])
                ]
                if len(decorators) != len(info.decorators):
                    info_for_computed_value = replace(info, decorators=decorators)
            computed_function = compute_value_of_function(info_for_computed_value, self)
            static_overload_signature: OverloadedSignature | None = None
            overload_dataclass_transform_info: DataclassTransformInfo | None = None
            if (
                FunctionDecorator.overload not in info.decorator_kinds
                and FunctionDecorator.evaluated not in info.decorator_kinds
            ):
                static_overload_signature = self._get_pending_overload_signature(node)
                overload_dataclass_transform_info = (
                    self._get_pending_overload_dataclass_transform_info(node)
                )
            self._check_overload_implementation_consistency(
                node,
                info,
                computed_function,
                direct_dataclass_transform_info=direct_dataclass_transform_info,
            )
            if potential_function is None:
                if static_overload_signature is not None:
                    val = CallableValue(static_overload_signature, types.FunctionType)
                else:
                    val = computed_function
            else:
                val = KnownValue(potential_function)
            if (
                FunctionDecorator.overload not in info.decorator_kinds
                and FunctionDecorator.evaluated not in info.decorator_kinds
            ):
                merged_dataclass_transform_info = direct_dataclass_transform_info
                if overload_dataclass_transform_info is not None:
                    if merged_dataclass_transform_info is None:
                        merged_dataclass_transform_info = (
                            overload_dataclass_transform_info
                        )
                    else:
                        merged_dataclass_transform_info = (
                            _merge_dataclass_transform_infos(
                                [
                                    merged_dataclass_transform_info,
                                    overload_dataclass_transform_info,
                                ]
                            )
                        )
                if merged_dataclass_transform_info is not None:
                    val = annotate_value(
                        val,
                        [DataclassTransformExtension(merged_dataclass_transform_info)],
                    )
                self._set_name_in_scope(node.name, node, val)
                self._record_synthetic_property_metadata(node, info)

            if (
                node.name in METHODS_ALLOWING_NOTIMPLEMENTED
                and info.return_annotation is not None
            ):
                expected_return = info.return_annotation | KnownValue(NotImplemented)
            else:
                expected_return = info.return_annotation
            if isinstance(expected_return, AnnotatedValue):
                expected_return, _ = unannotate_value(expected_return, TypeIsExtension)
                expected_return, _ = unannotate_value(
                    expected_return, TypeGuardExtension
                )

            with (
                self.asynq_checker.set_func_name(
                    node.name,
                    async_kind=info.async_kind,
                    is_classmethod=FunctionDecorator.classmethod
                    in info.decorator_kinds,
                ),
                self._active_pep695_type_param_scope(info.type_params),
                override(self, "yield_checker", YieldChecker(self)),
                override(self, "is_async_def", isinstance(node, ast.AsyncFunctionDef)),
                override(self, "current_function_name", node.name),
                override(self, "current_function", potential_function),
                override(self, "expected_return_value", expected_return),
                override(self, "current_function_info", info),
            ):
                result = self._visit_function_body(info)

        self.check_typeis(info)

        if (
            not result.has_return
            and not self._allow_missing_return(info)
            and node.returns is not None
            and (
                info.return_annotation is None
                or self._return_annotation_has_invalid_error(node.returns)
                or not self._return_annotation_allows_implicit_none(
                    info.return_annotation
                )
            )
        ):
            if info.return_annotation is NO_RETURN_VALUE:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.no_return_may_return
                )
            else:
                self._show_error_if_checking(node, error_code=ErrorCode.missing_return)

        if (
            FunctionDecorator.override in info.decorator_kinds
            and FunctionDecorator.overload not in info.decorator_kinds
        ):
            if self.scopes.scope_type() is not ScopeType.class_scope:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.invalid_override_decorator
                )
            else:
                if not self._has_base_attribute(node.name, node):
                    self._show_error_if_checking(
                        node, error_code=ErrorCode.override_does_not_override
                    )
        if (
            FunctionDecorator.final in info.decorator_kinds
            and FunctionDecorator.overload not in info.decorator_kinds
            and isinstance(self.current_class, str)
            and self.scopes.scope_type() is ScopeType.class_scope
        ):
            self._synthetic_final_methods.setdefault(self.current_class, set()).add(
                node.name
            )
        if (
            self._is_checking()
            and isinstance(self.current_class, str)
            and self.scopes.scope_type() is ScopeType.class_scope
        ):
            abstract_methods = self._synthetic_abstract_methods.setdefault(
                self.current_class, set()
            )
            if FunctionDecorator.abstractmethod in info.decorator_kinds:
                abstract_methods.add(node.name)
            else:
                abstract_methods.discard(node.name)

        if info.return_annotation is not None:
            assert node.returns is not None
            if result.is_generator:
                if isinstance(node, ast.FunctionDef):
                    if info.async_kind is AsyncFunctionKind.non_async:
                        can_assign = has_relation(
                            TypedValue(collections.abc.Iterable),
                            info.return_annotation,
                            Relation.ASSIGNABLE,
                            self,
                        )
                        if isinstance(can_assign, CanAssignError):
                            self._show_error_if_checking(
                                node.returns,
                                "Generator function must return an iterable",
                                error_code=ErrorCode.generator_return,
                                detail=can_assign.display(),
                            )
                else:
                    can_assign = has_relation(
                        TypedValue(collections.abc.AsyncIterable),
                        info.return_annotation,
                        Relation.ASSIGNABLE,
                        self,
                    )
                    if isinstance(can_assign, CanAssignError):
                        self._show_error_if_checking(
                            node.returns,
                            "Async generator function must return an async iterable",
                            error_code=ErrorCode.generator_return,
                            detail=can_assign.display(),
                        )

        if node.returns is None:
            if (
                result.has_return
                and FunctionDecorator.overload not in info.decorator_kinds
                and FunctionDecorator.abstractmethod not in info.decorator_kinds
            ):
                prepared = prepare_type(result.return_value, self)
                if should_suggest_type(prepared):
                    detail, metadata = display_suggested_type(
                        prepared, self.scopes, self
                    )
                    self._show_error_if_checking(
                        node,
                        error_code=ErrorCode.suggested_return_type,
                        detail=detail,
                        extra_metadata=metadata,
                    )

            if info.async_kind == AsyncFunctionKind.normal and _is_asynq_future(
                result.return_value
            ):
                self._show_error_if_checking(
                    node, error_code=ErrorCode.task_needs_yield
                )

        self._set_argspec_to_retval(val, info, result)
        return val

    def _get_enclosing_class_value_for_method(self) -> TypedValue | None:
        if self.current_class is None or not self._is_checking():
            return None
        class_value: TypedValue = TypedValue(self.current_class)
        type_params = (
            list(self.current_class_type_params)
            if self.current_class_type_params
            else []
        )
        if not type_params:
            generic_bases = self.checker.get_generic_bases(self.current_class, ())
            declared = generic_bases.get(self.current_class)
            if declared:
                type_params = [
                    val for val in declared.values() if isinstance(val, TypeVarValue)
                ]
        if type_params:
            return GenericValue(self.current_class, type_params)
        return class_value

    def _get_pending_overload_signature(
        self, node: FunctionDefNode
    ) -> OverloadedSignature | None:
        if not self._is_checking():
            return None
        current_scope = self.scopes.current_scope()
        scope_key = id(current_scope)
        pending_block = self._pending_overload_blocks.get(scope_key)
        if pending_block is not None and pending_block.scope is not current_scope:
            self._pending_overload_blocks.pop(scope_key, None)
            pending_block = None
        if pending_block is None or pending_block.name != node.name:
            return None
        signatures: list[Signature] = []
        for pending in pending_block.overloads:
            if pending.signature is None:
                continue
            if isinstance(pending.signature, Signature):
                signatures.append(pending.signature)
            else:
                signatures.extend(pending.signature.signatures)
        if signatures:
            return OverloadedSignature(signatures)
        return None

    def _get_pending_overload_dataclass_transform_info(
        self, node: FunctionDefNode
    ) -> DataclassTransformInfo | None:
        if not self._is_checking():
            return None
        current_scope = self.scopes.current_scope()
        scope_key = id(current_scope)
        pending_block = self._pending_overload_blocks.get(scope_key)
        if pending_block is not None and pending_block.scope is not current_scope:
            self._pending_overload_blocks.pop(scope_key, None)
            pending_block = None
        if pending_block is None or pending_block.name != node.name:
            return None
        infos = [
            pending.dataclass_transform_info
            for pending in pending_block.overloads
            if pending.dataclass_transform_info is not None
        ]
        if not infos:
            return None
        return _merge_dataclass_transform_infos(infos)

    def _flush_pending_overload_blocks(self) -> None:
        if not self._is_checking():
            return
        for scope_key, pending_block in list(self._pending_overload_blocks.items()):
            self._finalize_pending_overload_block(pending_block, implementation=None)
            self._pending_overload_blocks.pop(scope_key, None)

    def _flush_pending_overload_block_for_scope(self, scope: Scope) -> None:
        if not self._is_checking():
            return
        scope_key = id(scope)
        pending_block = self._pending_overload_blocks.get(scope_key)
        if pending_block is None or pending_block.scope is not scope:
            return
        self._pending_overload_blocks.pop(scope_key, None)
        self._finalize_pending_overload_block(pending_block, implementation=None)

    def _check_overload_implementation_consistency(
        self,
        node: FunctionDefNode,
        info: FunctionInfo,
        computed_function: Value,
        *,
        direct_dataclass_transform_info: DataclassTransformInfo | None,
    ) -> None:
        if not self._is_checking():
            return
        current_scope = self.scopes.current_scope()
        scope_key = id(current_scope)
        pending_block = self._pending_overload_blocks.get(scope_key)
        if pending_block is not None and pending_block.scope is not current_scope:
            # A scope exited with a dangling overload block and a new scope reused
            # its id(); discard the stale block.
            self._pending_overload_blocks.pop(scope_key, None)
            pending_block = None
        signature = self._signature_for_overload_consistency(info, computed_function)

        if pending_block is not None and (
            pending_block.name != node.name
            or FunctionDecorator.overload not in info.decorator_kinds
        ):
            self._pending_overload_blocks.pop(scope_key, None)
            implementation = None
            if (
                pending_block.name == node.name
                and FunctionDecorator.overload not in info.decorator_kinds
            ):
                implementation = (node, info, signature)
            self._finalize_pending_overload_block(pending_block, implementation)
            pending_block = None
            if implementation is not None:
                return

        if FunctionDecorator.overload not in info.decorator_kinds:
            return

        if pending_block is None:
            pending_block = _PendingOverloadBlock(node.name, current_scope)
            self._pending_overload_blocks[scope_key] = pending_block
        pending_block.overloads.append(
            _PendingOverload(
                node=node,
                signature=signature,
                decorator_kinds=info.decorator_kinds,
                dataclass_transform_info=direct_dataclass_transform_info,
            )
        )

    def _finalize_pending_overload_block(
        self,
        pending_block: _PendingOverloadBlock,
        implementation: (
            tuple[FunctionDefNode, FunctionInfo, ConcreteSignature | None] | None
        ),
    ) -> None:
        if not pending_block.overloads:
            return
        if implementation is None:
            self._materialize_overload_block_value(pending_block)
        should_check_consistency = self._validate_overload_block(
            pending_block, implementation
        )
        if not should_check_consistency or implementation is None:
            return
        node, _, implementation_signature = implementation
        if implementation_signature is None:
            return

        for pending in pending_block.overloads:
            if pending.signature is None:
                continue
            parameter_error = self._check_overload_parameter_compatibility(
                pending.signature, implementation_signature
            )
            return_error = self._check_overload_return_compatibility(
                pending.signature, implementation_signature
            )
            if parameter_error is not None or return_error is not None:
                detail_lines = []
                if parameter_error is not None:
                    detail_lines.append(
                        "Implementation signature does not accept all overload inputs."
                    )
                    detail_lines.append(parameter_error.display())
                if return_error is not None:
                    detail_lines.append(
                        "Implementation return type does not include all overload returns."
                    )
                    detail_lines.append(return_error.display())
                detail = "\n".join(detail_lines)
                self._show_error_if_checking(
                    pending.node,
                    f"Overload for {node.name!r} is inconsistent with implementation",
                    error_code=ErrorCode.inconsistent_overload,
                    detail=detail,
                )

    def _materialize_overload_block_value(
        self, pending_block: _PendingOverloadBlock
    ) -> None:
        signatures: list[Signature] = []
        transform_infos: list[DataclassTransformInfo] = []
        for pending in pending_block.overloads:
            signature = pending.signature
            if isinstance(signature, Signature):
                signatures.append(signature)
            elif isinstance(signature, OverloadedSignature):
                signatures.extend(signature.signatures)
            if pending.dataclass_transform_info is not None:
                transform_infos.append(pending.dataclass_transform_info)
        if signatures:
            value: Value = CallableValue(
                OverloadedSignature(signatures), types.FunctionType
            )
            if transform_infos:
                value = annotate_value(
                    value,
                    [
                        DataclassTransformExtension(
                            _merge_dataclass_transform_infos(transform_infos)
                        )
                    ],
                )
            pending_block.scope.variables[pending_block.name] = value

    def _validate_overload_block(
        self,
        pending_block: _PendingOverloadBlock,
        implementation: (
            tuple[FunctionDefNode, FunctionInfo, ConcreteSignature | None] | None
        ),
    ) -> bool:
        overloads = pending_block.overloads
        impl_node: FunctionDefNode | None
        impl_info: FunctionInfo | None
        impl_signature: ConcreteSignature | None
        if implementation is None:
            impl_node = None
            impl_info = None
            impl_signature = None
        else:
            impl_node, impl_info, impl_signature = implementation

        should_check_consistency = (
            implementation is not None
            and impl_signature is not None
            and any(pending.signature is not None for pending in overloads)
        )

        if len(overloads) < 2:
            should_check_consistency = False
            self._show_error_if_checking(
                overloads[0].node,
                "At least two overload signatures are required",
                error_code=ErrorCode.invalid_annotation,
            )

        class_scope_object = pending_block.scope.scope_object
        in_protocol_class = (
            pending_block.scope.scope_type is ScopeType.class_scope
            and isinstance(class_scope_object, (type, str))
            and self.checker.make_type_object(class_scope_object).is_protocol
        )
        needs_implementation = (
            not self.filename.endswith(".pyi")
            and not in_protocol_class
            and not all(
                FunctionDecorator.abstractmethod in pending.decorator_kinds
                for pending in overloads
            )
        )
        if implementation is None and needs_implementation:
            should_check_consistency = False
            self._show_error_if_checking(
                overloads[0].node,
                "Overloaded function is missing an implementation",
                error_code=ErrorCode.invalid_annotation,
            )

        overload_kinds = {
            _method_decorator_kind(decorator_kinds=pending.decorator_kinds)
            for pending in overloads
        }
        kind_mismatch = False
        if len(overload_kinds) > 1:
            kind_mismatch = True
            self._show_error_if_checking(
                overloads[0].node,
                "@staticmethod/@classmethod usage must be consistent across overloads",
                error_code=ErrorCode.invalid_annotation,
            )
        elif impl_info is not None:
            (overload_kind,) = overload_kinds
            impl_kind = _method_decorator_kind(
                decorator_kinds=impl_info.decorator_kinds
            )
            if impl_kind != overload_kind:
                kind_mismatch = True
                self._show_error_if_checking(
                    impl_node if impl_node is not None else overloads[0].node,
                    "Overload implementation has incompatible @staticmethod/@classmethod decorator",
                    error_code=ErrorCode.invalid_annotation,
                )
        if kind_mismatch:
            should_check_consistency = False

        overload_final_positions = [
            i
            for i, pending in enumerate(overloads)
            if FunctionDecorator.final in pending.decorator_kinds
        ]
        overload_override_positions = [
            i
            for i, pending in enumerate(overloads)
            if FunctionDecorator.override in pending.decorator_kinds
        ]

        placement_error = False
        effective_is_final = False
        effective_is_override = False
        if impl_info is not None:
            effective_is_final = FunctionDecorator.final in impl_info.decorator_kinds
            effective_is_override = (
                FunctionDecorator.override in impl_info.decorator_kinds
            )
            if overload_final_positions:
                placement_error = True
                self._show_error_if_checking(
                    overloads[overload_final_positions[0]].node,
                    "@final should be applied only to the overload implementation",
                    error_code=ErrorCode.invalid_annotation,
                )
            if overload_override_positions:
                placement_error = True
                self._show_error_if_checking(
                    overloads[overload_override_positions[0]].node,
                    error_code=ErrorCode.invalid_override_decorator,
                )
        else:
            if overload_final_positions:
                if overload_final_positions != [0]:
                    placement_error = True
                    idx = overload_final_positions[0]
                    if idx == 0 and len(overload_final_positions) > 1:
                        idx = overload_final_positions[1]
                    self._show_error_if_checking(
                        overloads[idx].node,
                        "@final should appear only on the first overload",
                        error_code=ErrorCode.invalid_annotation,
                    )
                else:
                    effective_is_final = True
            if overload_override_positions:
                if overload_override_positions != [0]:
                    placement_error = True
                    idx = overload_override_positions[0]
                    if idx == 0 and len(overload_override_positions) > 1:
                        idx = overload_override_positions[1]
                    self._show_error_if_checking(
                        overloads[idx].node,
                        error_code=ErrorCode.invalid_override_decorator,
                    )
                else:
                    effective_is_override = True
        if placement_error:
            should_check_consistency = False

        representative_node = impl_node if impl_node is not None else overloads[0].node
        class_context: type | str | None
        if pending_block.scope.scope_type is ScopeType.class_scope and isinstance(
            pending_block.scope.scope_object, (type, str)
        ):
            class_context = pending_block.scope.scope_object
        else:
            class_context = None
        if effective_is_override:
            if class_context is None:
                self._show_error_if_checking(
                    representative_node, error_code=ErrorCode.invalid_override_decorator
                )
            elif not self._has_base_attribute_for(
                class_context, pending_block.name, representative_node
            ):
                self._show_error_if_checking(
                    representative_node, error_code=ErrorCode.override_does_not_override
                )

        if class_context is not None and self._is_final_base_method(
            class_context, pending_block.name, representative_node
        ):
            self._show_error_if_checking(
                representative_node,
                "Cannot override a final method",
                error_code=ErrorCode.invalid_annotation,
            )

        if effective_is_final and isinstance(class_context, str):
            self._synthetic_final_methods.setdefault(class_context, set()).add(
                pending_block.name
            )
        return should_check_consistency

    def _is_final_base_method(
        self, current_class: type | str, method_name: str, node: ast.AST
    ) -> bool:
        for base_class in self.checker.get_generic_bases(current_class):
            if isinstance(
                base_class, str
            ) and method_name in self._synthetic_final_methods.get(base_class, set()):
                return True
        for base_class, base_value in self._get_base_class_attributes_for(
            current_class, method_name, node
        ):
            if isinstance(
                base_class, str
            ) and method_name in self._synthetic_final_methods.get(base_class, set()):
                return True
            for subval in flatten_values(base_value, unwrap_annotated=True):
                if isinstance(subval, KnownValue) and safe_getattr(
                    subval.val, "__final__", False
                ):
                    return True
        return False

    def _signature_for_overload_consistency(
        self, info: FunctionInfo, computed_function: Value
    ) -> ConcreteSignature | None:
        if FunctionDecorator.overload in info.decorator_kinds:
            decorators = [
                decorator
                for decorator in info.decorators
                if not self._is_overload_decorator(decorator[0])
            ]
            if len(decorators) != len(info.decorators):
                info = replace(info, decorators=decorators)
                computed_function = compute_value_of_function(info, self)

        signature = self.signature_from_value(computed_function)
        if isinstance(signature, BoundMethodSignature):
            signature = signature.get_signature(ctx=self)
        if isinstance(signature, (Signature, OverloadedSignature)):
            return signature
        return None

    def _check_overload_parameter_compatibility(
        self,
        overload_signature: ConcreteSignature,
        implementation_signature: ConcreteSignature,
    ) -> CanAssignError | None:
        any_return = AnyValue(AnySource.marker)
        can_assign = overload_signature.replace_return_value(any_return).can_assign(
            implementation_signature.replace_return_value(any_return), self
        )
        if isinstance(can_assign, CanAssignError):
            return can_assign
        return None

    def _check_overload_return_compatibility(
        self,
        overload_signature: ConcreteSignature,
        implementation_signature: ConcreteSignature,
    ) -> CanAssignError | None:
        can_assign = has_relation(
            implementation_signature.return_value,
            overload_signature.return_value,
            Relation.ASSIGNABLE,
            self,
        )
        if isinstance(can_assign, CanAssignError):
            return can_assign
        return None

    def _is_overload_decorator(self, value: Value) -> bool:
        return isinstance(value, KnownValue) and value.val in (real_overload, overload)

    def _is_final_decorator_value(self, value: Value) -> bool:
        return isinstance(value, KnownValue) and is_typing_name(value.val, "final")

    def _finalize_synthetic_abstract_members(
        self, node: ast.ClassDef, class_key: type | str, *, is_protocol_class: bool
    ) -> None:
        if not self._is_checking() or not isinstance(class_key, str):
            return

        required = set(self._synthetic_abstract_methods.get(class_key, set()))
        for base_key in self.checker.get_generic_bases(class_key):
            if base_key == class_key:
                continue
            required |= self._required_abstract_members_for_base(base_key)

        if is_protocol_class:
            required |= self._protocol_required_members_from_class_body(node)

        provided = self._concrete_member_names_for_current_class(
            node, class_key, is_protocol_class=is_protocol_class
        )
        for base_key in self.checker.get_generic_bases(class_key):
            if base_key == class_key:
                continue
            provided |= self._concrete_member_names_for_base(base_key)

        self._synthetic_abstract_methods[class_key] = required - provided

    def _required_abstract_members_for_base(self, class_key: type | str) -> set[str]:
        if isinstance(class_key, str):
            return set(self._synthetic_abstract_methods.get(class_key, set()))
        abstract_methods = safe_getattr(class_key, "__abstractmethods__", ())
        if not isinstance(abstract_methods, (set, frozenset, tuple, list)):
            return set()
        return {name for name in abstract_methods if isinstance(name, str)}

    def _concrete_member_names_for_current_class(
        self, node: ast.ClassDef, class_key: str, *, is_protocol_class: bool
    ) -> set[str]:
        provided = self._concrete_member_names_for_base(class_key)
        provided -= self._uninitialized_classvar_names_from_class_body(node)
        if is_protocol_class:
            provided -= self._protocol_required_members_from_class_body(node)
        return provided

    def _concrete_member_names_for_base(self, class_key: type | str) -> set[str]:
        if isinstance(class_key, str):
            synthetic_class = self.checker.get_synthetic_class(class_key)
            if synthetic_class is None:
                return set()
            provided = {
                name
                for name in synthetic_class.class_attributes
                if not name.startswith("%")
            }
            return provided - self._required_abstract_members_for_base(class_key)

        class_dict = safe_getattr(class_key, "__dict__", None)
        if not isinstance(class_dict, Mapping):
            return set()
        provided = {name for name in class_dict if isinstance(name, str)}
        return provided - self._required_abstract_members_for_base(class_key)

    def _protocol_required_members_from_class_body(
        self, node: ast.ClassDef
    ) -> set[str]:
        required = set(self._uninitialized_classvar_names_from_class_body(node))
        for statement in node.body:
            if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_kinds = self._function_decorator_kinds_by_node.get(
                statement, frozenset()
            )
            if (
                FunctionDecorator.abstractmethod in decorator_kinds
                or self._has_only_docstring_and_stub(statement)
            ):
                required.add(statement.name)
        return required

    def _uninitialized_classvar_names_from_class_body(
        self, node: ast.ClassDef
    ) -> set[str]:
        classvars = set()
        for statement in node.body:
            if (
                isinstance(statement, ast.AnnAssign)
                and statement.value is None
                and isinstance(statement.target, ast.Name)
                and self._annotation_has_classvar_qualifier(statement.annotation)
            ):
                classvars.add(statement.target.id)
        return classvars

    def _annotation_has_classvar_qualifier(self, annotation: ast.expr) -> bool:
        try:
            annotation_expr = annotation_expr_from_ast(
                annotation, visitor=self, suppress_errors=True
            )
        except Exception:
            return False
        _, qualifiers = annotation_expr.maybe_unqualify(set(Qualifier))
        return Qualifier.ClassVar in qualifiers

    def _record_final_member(self, class_key: type | str, member_name: str) -> None:
        self.final_member_names_by_class.setdefault(class_key, set()).add(member_name)

    def _check_for_final_base_classes(
        self, node: ast.ClassDef, base_values: Sequence[Value]
    ) -> None:
        if not self._is_checking():
            return
        for base in base_values:
            if self._is_final_base_value(base):
                self._show_error_if_checking(
                    node,
                    "Cannot inherit from final class",
                    ErrorCode.invalid_annotation,
                )
                return

    def _is_final_base_value(self, base_value: Value) -> bool:
        base_key = self._base_class_key_from_value(base_value)
        if base_key is None:
            return False
        if base_key in self.final_class_keys:
            return True
        if isinstance(base_key, type):
            if safe_issubclass(base_key, enum.Enum):
                try:
                    if bool(base_key.__members__):
                        return True
                except Exception:
                    pass
            if getattr(base_key, "__final__", False):
                return True
        try:
            return self.checker.ts_finder.is_final(base_key)
        except Exception:
            return False

    def _check_for_uninitialized_final_members(self, class_key: type | str) -> None:
        if not self._is_checking():
            self.final_members_requiring_init.pop(class_key, None)
            self.final_members_initialized_in_init.pop(class_key, None)
            return
        required = self.final_members_requiring_init.pop(class_key, {})
        initialized = self.final_members_initialized_in_init.pop(class_key, set())
        for name, node in required.items():
            if name in initialized:
                continue
            self._show_error_if_checking(
                node,
                "Final class attributes without initializers must be assigned in __init__",
                error_code=ErrorCode.invalid_annotation,
            )

    def _is_allowed_instance_final_annotation_target(
        self, target: ast.Attribute, root_value: Value
    ) -> bool:
        return (
            self.current_function_name == "__init__"
            and self._is_current_method_receiver_node(target.value)
            and self._is_class_object_attribute_root(root_value) is False
        )

    def _current_method_receiver_name(self) -> str | None:
        info = self.current_function_info
        if info is None:
            return None
        for param_info in info.params:
            if param_info.is_self:
                return param_info.param.name
        return None

    def _is_current_method_receiver_node(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Name):
            return False
        receiver_name = self._current_method_receiver_name()
        return receiver_name is not None and node.id == receiver_name

    def _class_key_for_attribute_target(
        self, node: ast.Attribute, root_value: Value
    ) -> type | str | None:
        class_key = self._class_key_from_attribute_root_value(root_value)
        if class_key is not None:
            return class_key
        if (
            self.current_class_key is not None
            and self._is_current_method_receiver_node(node.value)
        ):
            return self.current_class_key
        return None

    def _class_key_from_attribute_root_value(
        self, root_value: Value
    ) -> type | str | None:
        if (
            isinstance(root_value, PartialValue)
            and root_value.operation is PartialValueOperation.SUBSCRIPT
            and self._is_class_object_attribute_root(root_value.root) is True
        ):
            return self._class_key_from_attribute_root_value(root_value.root)
        root_value = replace_fallback(root_value)
        if isinstance(root_value, AnnotatedValue):
            return self._class_key_from_attribute_root_value(root_value.value)
        if isinstance(root_value, KnownValue):
            if isinstance(root_value.val, type):
                return root_value.val
            origin = get_origin(root_value.val)
            if isinstance(origin, type):
                return origin
            return type(root_value.val)
        if isinstance(root_value, GenericValue):
            if isinstance(root_value.typ, (type, str)):
                return root_value.typ
            return None
        if isinstance(root_value, MultiValuedValue):
            class_keys = {
                class_key
                for subval in root_value.vals
                if (class_key := self._class_key_from_attribute_root_value(subval))
                is not None
            }
            if len(class_keys) == 1:
                return next(iter(class_keys))
            return None
        return self._base_class_key_from_value(root_value)

    def _frozen_dataclass_instance_status(self, value: Value) -> bool | None:
        is_dataclass_class, frozen = self._get_dataclass_status_for_instance_value(
            value
        )
        if not is_dataclass_class:
            return False
        return frozen

    def _is_assignment_to_frozen_dataclass_attribute(self, root_value: Value) -> bool:
        return self._frozen_dataclass_instance_status(root_value) is True

    def _is_assignment_to_final_attribute(
        self, node: ast.Attribute, root_value: Value
    ) -> bool:
        if self.ann_assign_type is not None and self.ann_assign_type[1]:
            return False
        class_key = self._class_key_for_attribute_target(node, root_value)
        if class_key is None:
            return False
        if not self._is_final_member(class_key, node.attr):
            return False
        required = self.final_members_requiring_init.get(class_key)
        if self.current_function_name == "__init__" and required is not None:
            if node.attr in required:
                self.final_members_initialized_in_init.setdefault(class_key, set()).add(
                    node.attr
                )
                return False
        return True

    def _is_class_object_attribute_root(self, value: Value) -> bool | None:
        if (
            isinstance(value, PartialValue)
            and value.operation is PartialValueOperation.SUBSCRIPT
            and self._is_class_object_attribute_root(value.root) is True
        ):
            # Preserve class-object-ness for subscripted class objects such as C[T].
            return True
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._is_class_object_attribute_root(value.value)
        if isinstance(value, MultiValuedValue):
            kinds = {
                result
                for subval in value.vals
                if (result := self._is_class_object_attribute_root(subval)) is not None
            }
            if len(kinds) == 1:
                return next(iter(kinds))
            return None
        if isinstance(value, IntersectionValue):
            kinds = {
                result
                for subval in value.vals
                if (result := self._is_class_object_attribute_root(subval)) is not None
            }
            if len(kinds) == 1:
                return next(iter(kinds))
            return None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return True
            return isinstance(get_origin(value.val), type)
        if isinstance(value, SyntheticClassObjectValue):
            return True
        if isinstance(value, SubclassValue):
            return True
        if isinstance(value, TypedValue):
            if isinstance(value.typ, type):
                return safe_issubclass(value.typ, type)
            if isinstance(value.typ, str):
                return value.typ in {"builtins.type", "type"}
            return False
        if isinstance(value, GenericValue):
            if isinstance(value.typ, type):
                return safe_issubclass(value.typ, type)
            if isinstance(value.typ, str):
                return value.typ in {"builtins.type", "type"}
            return False
        if isinstance(
            value,
            (
                AnyValue,
                SyntheticModuleValue,
                UnboundMethodValue,
                TypeFormValue,
                PredicateValue,
            ),
        ):
            return False
        assert_never(value)

    def _is_assignment_to_classvar_through_instance(
        self, node: ast.Attribute, root_value: Value
    ) -> bool:
        class_key = self._class_key_for_attribute_target(node, root_value)
        if class_key is None or not self._is_classvar_member(class_key, node.attr):
            return False
        is_class_object = self._is_class_object_attribute_root(root_value)
        return is_class_object is False

    def _is_assignment_to_instance_member_through_class(
        self, node: ast.Attribute, root_value: Value
    ) -> bool:
        class_key = self._class_key_for_attribute_target(node, root_value)
        if class_key is None or not self._is_instance_only_member(class_key, node.attr):
            return False
        is_class_object = self._is_class_object_attribute_root(root_value)
        return is_class_object is True

    def _class_key_from_type_call_expr(
        self, call_expr: ast.AST | None
    ) -> type | str | None:
        call_node = call_expr
        if (
            not isinstance(call_node, ast.Call)
            or len(call_node.args) != 1
            or call_node.keywords
        ):
            return None
        is_builtin_type_name = (
            isinstance(call_node.func, ast.Name) and call_node.func.id == "type"
        )
        with self.catch_errors():
            callee_value = replace_fallback(
                self.composite_from_node(call_node.func).value
            )
        is_type_callee = (
            (isinstance(callee_value, KnownValue) and callee_value.val is type)
            or (isinstance(callee_value, TypedValue) and callee_value.typ is type)
            or (
                isinstance(callee_value, SubclassValue)
                and isinstance(callee_value.typ, TypedValue)
                and callee_value.typ.typ is type
            )
        )
        if not is_type_callee and not is_builtin_type_name:
            return None
        with self.catch_errors():
            argument_value = self.composite_from_node(call_node.args[0]).value
        return self._class_key_from_attribute_root_value(argument_value)

    def _is_subscripted_class_alias_value(self, value: Value) -> bool:
        if (
            isinstance(value, PartialValue)
            and value.operation is PartialValueOperation.SUBSCRIPT
        ):
            return self._is_class_object_attribute_root(value.root) is True
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._is_subscripted_class_alias_value(value.value)
        if isinstance(value, MultiValuedValue):
            return any(
                self._is_subscripted_class_alias_value(subval) for subval in value.vals
            )
        if isinstance(value, IntersectionValue):
            return any(
                self._is_subscripted_class_alias_value(subval) for subval in value.vals
            )
        if isinstance(value, KnownValue):
            return isinstance(get_origin(value.val), type)
        return False

    def _should_check_plain_class_object_instance_member_access(
        self, class_key: type | str
    ) -> bool:
        if self._is_enum_class_key(class_key):
            return False
        if isinstance(class_key, type):
            is_dataclass, _ = self._get_dataclass_status_for_type(class_key)
            if is_dataclass:
                return False
        elif isinstance(class_key, str):
            synthetic_class = self._synthetic_classes_by_name.get(class_key)
            if synthetic_class is not None and synthetic_class.is_dataclass:
                return False
        return True

    def _is_instance_member_accessed_through_class(
        self, root_composite: Composite, attr_name: str, node: ast.AST | None = None
    ) -> bool:
        call_expr: ast.AST | None = root_composite.node
        if isinstance(node, ast.Attribute):
            call_expr = node.value
        type_call_key = self._class_key_from_type_call_expr(call_expr)
        if type_call_key is not None and self._is_instance_only_member(
            type_call_key, attr_name
        ):
            return True
        if not self._is_subscripted_class_alias_value(root_composite.value):
            class_key = self._class_key_from_attribute_root_value(root_composite.value)
            if (
                class_key is None
                or self._is_class_object_attribute_root(root_composite.value)
                is not True
                or not self._should_check_plain_class_object_instance_member_access(
                    class_key
                )
            ):
                return False
            return self._is_instance_only_member(class_key, attr_name)
        class_key = self._class_key_from_attribute_root_value(root_composite.value)
        return class_key is not None and self._is_instance_only_member(
            class_key, attr_name
        )

    def _namedtuple_fields_for_simple_attribute_root(
        self, value: Value
    ) -> set[str] | None:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._namedtuple_fields_for_simple_attribute_root(value.value)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            return None
        if isinstance(value, KnownValue):
            if isinstance(value.val, type):
                return None
            typ = type(value.val)
        elif isinstance(value, GenericValue):
            if not isinstance(value.typ, type):
                return None
            typ = value.typ
        elif isinstance(value, TypedValue):
            if not isinstance(value.typ, type):
                return None
            typ = value.typ
        elif isinstance(
            value,
            (
                AnyValue,
                SyntheticClassObjectValue,
                SyntheticModuleValue,
                UnboundMethodValue,
                SubclassValue,
                TypeFormValue,
                PredicateValue,
            ),
        ):
            return None
        else:
            assert_never(value)
        if not is_namedtuple_class(typ):
            return None
        fields = safe_getattr(typ, "_fields", None)
        if not isinstance(fields, tuple):
            return None
        if not all(isinstance(field, str) for field in fields):
            return None
        return set(fields)

    def _is_namedtuple_field_attribute(self, root_value: Value, attr_name: str) -> bool:
        value = replace_fallback(root_value)
        if isinstance(value, AnnotatedValue):
            return self._is_namedtuple_field_attribute(value.value, attr_name)
        if isinstance(value, (MultiValuedValue, IntersectionValue)):
            if not value.vals:
                return False
            return all(
                self._is_namedtuple_field_attribute(subval, attr_name)
                for subval in value.vals
            )
        fields = self._namedtuple_fields_for_simple_attribute_root(value)
        return fields is not None and attr_name in fields

    def _show_namedtuple_attribute_mutation_error(self, node: ast.Attribute) -> None:
        self._show_error_if_checking(
            node,
            f"Cannot mutate NamedTuple field {node.attr!r}",
            error_code=ErrorCode.incompatible_assignment,
        )

    def _is_final_member(
        self, class_key: type | str, member_name: str, member_value: Value | None = None
    ) -> bool:
        if member_name in self.final_member_names_by_class.get(class_key, set()):
            return True
        if isinstance(class_key, type):
            class_dict = safe_getattr(class_key, "__dict__", {})
            if isinstance(class_dict, Mapping):
                runtime_member = class_dict.get(member_name)
                if getattr(runtime_member, "__final__", False):
                    return True
        if isinstance(member_value, KnownValue) and getattr(
            member_value.val, "__final__", False
        ):
            return True
        if isinstance(class_key, str):
            try:
                return self.checker.ts_finder.is_final_attribute(class_key, member_name)
            except Exception:
                return False
        return False

    def _is_final_imported_name(self, source_module: Value, alias_name: str) -> bool:
        if not isinstance(source_module, KnownValue):
            return False
        module = source_module.val
        if not isinstance(module, types.ModuleType):
            return False
        annotations = safe_getattr(module, "__annotations__", None)
        if not isinstance(annotations, Mapping) or alias_name not in annotations:
            return False
        try:
            expr = annotation_expr_from_runtime(
                annotations[alias_name],
                visitor=self,
                globals=module.__dict__,
                suppress_errors=True,
            )
        except Exception:
            return False
        _, qualifiers = expr.maybe_unqualify({Qualifier.Final})
        return Qualifier.Final in qualifiers

    def _is_forbidden_annotated_call(self, callee: Value) -> bool:
        if not isinstance(callee, KnownValue):
            return False
        if is_typing_name(callee.val, "Annotated"):
            return True
        return is_typing_name(get_origin(callee.val), "Annotated")

    def _allow_missing_return(self, info: FunctionInfo) -> bool:
        if (
            FunctionDecorator.overload in info.decorator_kinds
            or FunctionDecorator.evaluated in info.decorator_kinds
        ):
            return True
        node = info.node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if not self._has_only_docstring_and_stub(node):
            return False
        if FunctionDecorator.abstractmethod in info.decorator_kinds:
            return True
        return (
            self.current_class is not None
            and self.checker.make_type_object(self.current_class).is_protocol
        )

    def _return_annotation_allows_implicit_none(self, value: Value) -> bool:
        if _is_known_none_annotation(value):
            return True
        can_assign = has_relation(value, KnownNone, Relation.ASSIGNABLE, self)
        return not isinstance(can_assign, CanAssignError)

    def _return_annotation_has_invalid_error(self, annotation: ast.expr) -> bool:
        return any(
            (subnode, ErrorCode.invalid_annotation) in self.seen_errors
            for subnode in ast.walk(annotation)
        )

    def _has_only_docstring_and_stub(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        body = node.body
        if not body:
            return False
        if len(body) == 1:
            candidate = body[0]
        elif (
            len(body) == 2
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            candidate = body[1]
        else:
            return False
        if isinstance(candidate, ast.Pass):
            return True
        return (
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Constant)
            and candidate.value.value is Ellipsis
        )

    def check_typeis(self, info: FunctionInfo) -> None:
        if info.return_annotation is None:
            return
        assert isinstance(info.node, (ast.FunctionDef, ast.AsyncFunctionDef))
        assert info.node.returns is not None
        _, ti = unannotate_value(info.return_annotation, TypeIsExtension)
        for type_is in ti:
            param = self._get_typeis_parameter(info)
            if param is None:
                self._show_error_if_checking(
                    info.node,
                    "TypeIs must be used on a function taking at least one positional"
                    " parameter",
                    error_code=ErrorCode.invalid_typeguard,
                )
                continue
            can_assign = has_relation(
                param.annotation, type_is.guarded_type, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    info.node.returns,
                    f"TypeIs narrowed type {type_is.guarded_type} is incompatible "
                    f"with parameter {param.name}",
                    error_code=ErrorCode.typeis_must_be_subtype,
                    detail=can_assign.display(),
                )
        _, tg = unannotate_value(info.return_annotation, TypeGuardExtension)
        for _ in tg:
            param = self._get_typeis_parameter(info)
            if param is None:
                self._show_error_if_checking(
                    info.node,
                    "TypeGuard must be used on a function taking at least one"
                    " positional parameter",
                    error_code=ErrorCode.invalid_typeguard,
                )

    def _get_typeis_parameter(self, info: FunctionInfo) -> SigParameter | None:
        index = 0
        if FunctionDecorator.classmethod in info.decorator_kinds or (
            info.is_nested_in_class
            and FunctionDecorator.classmethod not in info.decorator_kinds
            and FunctionDecorator.staticmethod not in info.decorator_kinds
        ):
            index = 1
        if len(info.params) <= index:
            return None
        param = info.params[index].param
        if param.kind not in (
            ParameterKind.POSITIONAL_ONLY,
            ParameterKind.POSITIONAL_OR_KEYWORD,
        ):
            return None
        return param

    def _set_argspec_to_retval(
        self, val: Value, info: FunctionInfo, result: FunctionResult
    ) -> None:
        if isinstance(info.node, ast.Lambda) or info.node.returns is not None:
            return
        if info.async_kind == AsyncFunctionKind.async_proxy:
            # Don't attempt to infer the return value of async_proxy functions, since it will be
            # set within the Future returned. Without this, we'll incorrectly infer the return
            # value to be the Future instead of the Future's value.
            return
        if info.node.decorator_list and not (
            len(info.decorators) == 1
            and info.decorators[0][0] in SAFE_DECORATORS_FOR_ARGSPEC_TO_RETVAL
        ):
            return  # With decorators we don't know what it will return
        return_value = result.return_value

        if result.is_generator and return_value == KnownNone:
            return_value = AnyValue(AnySource.inference)

        # pure async functions are otherwise incorrectly inferred as returning whatever the
        # underlying function returns
        if info.async_kind == AsyncFunctionKind.pure:
            task_cls = _get_task_cls(info.potential_function)
            return_value = AsyncTaskIncompleteValue(task_cls, return_value)

        if isinstance(info.node, ast.AsyncFunctionDef) or (
            FunctionDecorator.decorated_coroutine in info.decorator_kinds
        ):
            return_value = make_coro_type(return_value)

        if isinstance(val, KnownValue) and isinstance(val.val, property):
            fget = val.val.fget
            if fget is None:
                return
            val = KnownValue(fget)

        sig = self.signature_from_value(val)
        if sig is None or sig.has_return_value():
            return
        self._argspec_to_retval[id(sig)] = (return_value, sig)

    def _get_potential_function(self, node: FunctionDefNode) -> object | None:
        scope_type = self.scopes.scope_type()
        if scope_type == ScopeType.module_scope and self.module is not None:
            potential_function = safe_getattr(self.module, node.name, None)
        elif scope_type == ScopeType.class_scope and isinstance(
            self.current_class, type
        ):
            potential_function = safe_getattr(self.current_class, node.name, None)
        else:
            potential_function = None

        if (
            potential_function is not None
            and self.options.is_error_code_enabled_anywhere(
                ErrorCode.suggested_parameter_type
            )
        ):
            sig = self.signature_from_value(KnownValue(potential_function))
            if isinstance(sig, Signature):
                self.checker.callable_tracker.record_callable(
                    node, potential_function, sig, scopes=self.scopes, ctx=self
                )
        return potential_function

    def record_call(self, callable: object, arguments: CallArgs) -> None:
        if self.options.is_error_code_enabled_anywhere(
            ErrorCode.suggested_parameter_type
        ):
            self.checker.callable_tracker.record_call(callable, arguments)

    def visit_Lambda(self, node: ast.Lambda) -> Value:
        with self.asynq_checker.set_func_name("<lambda>"):
            with self.compute_function_info(node) as info:
                with override(self, "current_function_info", info):
                    result = self._visit_function_body(info)
            return compute_value_of_function(info, self, result=result.return_value)

    def _visit_function_body(self, function_info: FunctionInfo) -> FunctionResult:
        is_collecting = self._is_collecting()
        node = function_info.node

        class_ctx = (
            contextlib.nullcontext()
            if not self.scopes.is_nested_function()
            else override(self, "current_class", None)
        )
        with class_ctx:
            self._check_method_first_arg(node, function_info=function_info)
        infos = function_info.params
        params = [info.param for info in infos]

        if is_collecting and not self.scopes.contains_scope_of_type(
            ScopeType.function_scope
        ):
            return FunctionResult(parameters=params)

        if FunctionDecorator.evaluated in function_info.decorator_kinds:
            if self._is_collecting() or isinstance(node, ast.Lambda):
                return FunctionResult(parameters=params)
            with self.scopes.allow_only_module_scope():
                # The return annotation doesn't actually matter for validation.
                evaluator = SyntheticEvaluator.from_visitor(
                    node, self, AnyValue(AnySource.marker)
                )
                ctx = type_evaluation.EvalContext(
                    variables={param.name: param.annotation for param in params},
                    positions={param.name: type_evaluation.DEFAULT for param in params},
                    can_assign_context=self,
                    tv_map={},
                )
                for error in evaluator.validate(ctx):
                    self.show_error(
                        error.node, error.message, error_code=ErrorCode.bad_evaluator
                    )
                if self.annotate:
                    with (
                        self.catch_errors(),
                        self.scopes.add_scope(
                            ScopeType.function_scope, scope_node=node
                        ),
                    ):
                        self._generic_visit_list(node.body)
            return FunctionResult(parameters=params)

        # We pass in the node to add_scope() and visit the body once in collecting
        # mode if in a nested function, so that constraints on nonlocals in the outer
        # scope propagate into this scope. This means that we'll use the constraints
        # of the place where the function is defined, not those of where the function
        # is called, which is strictly speaking wrong but should be fine in practice.
        with (
            self.scopes.add_scope(ScopeType.function_scope, scope_node=node),
            override(self, "is_generator", False),
            override(self, "async_kind", function_info.async_kind),
            override(self, "_name_node_to_statement", {}),
        ):
            scope = self.scopes.current_scope()
            assert isinstance(scope, FunctionScope)

            for info in infos:
                if info.is_self:
                    # we need this for the implementation of super()
                    self.scopes.set(
                        "%first_arg",
                        info.param.annotation,
                        "%first_arg",
                        VisitorState.check_names,
                    )
                self.scopes.set(
                    info.param.name,
                    info.param.annotation,
                    info.node,
                    VisitorState.check_names,
                )

            with (
                override(self, "state", VisitorState.collect_names),
                override(self, "return_values", []),
                self.yield_checker.set_function_node(node),
            ):
                if isinstance(node, ast.Lambda):
                    self.visit(node.body)
                else:
                    self._generic_visit_list(node.body)
                scope.get_local(LEAVES_SCOPE, node, self.state, can_assign_ctx=self)
            if is_collecting:
                return FunctionResult(is_generator=self.is_generator, parameters=params)

            # otherwise we may end up using results from the last yield (generated during the
            # collect state) to evaluate the first one visited during the check state
            self.yield_checker.reset_yield_checks()

            with (
                override(self, "current_class", None),
                override(self, "state", VisitorState.check_names),
                override(self, "return_values", []),
                self.yield_checker.set_function_node(node),
            ):
                if isinstance(node, ast.Lambda):
                    return_values = [self.visit(node.body)]
                else:
                    self._generic_visit_list(node.body)
                    return_values = self.return_values
                return_set, _ = scope.get_local(
                    LEAVES_SCOPE, node, self.state, can_assign_ctx=self
                )

            self._check_function_unused_vars(scope)
            return self._compute_return_type(
                node, return_values, return_set, function_info, params
            )

    def _compute_return_type(
        self,
        node: FunctionNode,
        return_values: Sequence[Value | None],
        return_set: Value,
        info: FunctionInfo,
        params: Sequence[SigParameter],
    ) -> FunctionResult:
        # Ignore generators for now.
        if (
            isinstance(return_set, AnyValue)
            or return_set is NO_RETURN_VALUE
            or (self.is_generator and info.async_kind is not AsyncFunctionKind.normal)
        ):
            has_return = True
        elif return_set is UNINITIALIZED_VALUE:
            has_return = False
        else:
            assert False, return_set
        # if the return value was never set, the function returns None
        if not return_values:
            return FunctionResult(KnownNone, params, has_return, self.is_generator)
        # None is added to return_values if the function raises an error.
        return_values = [val for val in return_values if val is not None]
        # If it only ever raises an error, we don't know what it returns. Strictly
        # this should perhaps be NoReturnValue, but that leads to issues because
        # in practice this condition often occurs in abstract methods that just
        # raise NotImplementedError.
        if not return_values:
            ret = AnyValue(AnySource.inference)
        else:
            ret = unite_values(*return_values)
        if isinstance(node, ast.Lambda):
            has_return_annotation = False
        else:
            has_return_annotation = node.returns is not None
        return FunctionResult(
            ret,
            params,
            has_return=has_return,
            is_generator=self.is_generator,
            has_return_annotation=has_return_annotation,
        )

    def _check_function_unused_vars(
        self, scope: FunctionScope, enclosing_statement: ast.stmt | None = None
    ) -> None:
        """Shows errors for any unused variables in the function."""
        all_def_nodes = set(
            chain.from_iterable(scope.name_to_all_definition_nodes.values())
        )
        all_used_def_nodes = set(
            chain.from_iterable(scope.usage_to_definition_nodes.values())
        )
        all_unused_nodes = all_def_nodes - all_used_def_nodes
        for unused in all_unused_nodes:
            # Ignore names not defined through a Name node (e.g., function arguments)
            if not isinstance(unused, ast.Name) or not self._is_write_ctx(unused.ctx):
                continue
            # Ignore names that are meant to be ignored
            if unused.id.startswith("_"):
                continue
            # Ignore names involved in global and similar declarations
            if unused.id in scope.accessed_from_special_nodes:
                continue
            replacement = None
            if self._name_node_to_statement is not None:
                # Ignore some names defined in unpacking assignments. This should behave as follows:
                #   a, b = c()  # error only if a and b are both unused
                #   a, b = yield c.asynq()  # same
                #   a, b = yield (func1.asynq(), func2.asynq())  # error if either a or b is unused
                #   [None for i in range(3)]  # error
                #   [a for a, b in pairs]  # no error
                #   [None for a, b in pairs]  # error
                statement = self._name_node_to_statement.get(unused)
                if isinstance(statement, ast.Assign):
                    # it's an assignment
                    if not (
                        isinstance(statement.value, ast.Yield)
                        and isinstance(statement.value.value, ast.Tuple)
                    ):
                        # but not an assignment originating from yielding a tuple (which is
                        # probably an async yield)

                        # We need to loop over the targets to handle code like "a, b = c = func()".
                        # If the target containing our unused variable is a tuple and some of its
                        # members are not unused, ignore it.
                        partly_used_target = False
                        for target in statement.targets:
                            if (
                                isinstance(target, (ast.List, ast.Tuple))
                                and _contains_node(target.elts, unused)
                                and not _all_names_unused(target.elts, all_unused_nodes)
                            ):
                                partly_used_target = True
                                break
                        if partly_used_target:
                            continue
                    if len(statement.targets) == 1 and not isinstance(
                        statement.targets[0], (ast.List, ast.Tuple)
                    ):
                        replacement = self.remove_node(unused, statement)
                elif isinstance(statement, ast.comprehension):
                    if isinstance(statement.target, ast.Tuple):
                        if not _all_names_unused(
                            statement.target.elts, all_unused_nodes
                        ):
                            continue
                    else:
                        replacement = self.replace_node(
                            unused,
                            ast.Name(id="_", ctx=ast.Store()),
                            enclosing_statement,
                        )
                elif isinstance(statement, ast.AnnAssign):
                    # Ignore bare annotations (`x: int`), which don't assign a value.
                    # But treat `x: int = value` like a regular assignment for
                    # unused-variable reporting.
                    if statement.value is None:
                        continue
                    replacement = self.remove_node(unused, statement)
            if all(
                node in all_unused_nodes
                for node in scope.name_to_all_definition_nodes[unused.id]
            ):
                self._show_error_if_checking(
                    unused,
                    f"Variable {unused.id} is never accessed",
                    error_code=ErrorCode.unused_variable,
                    replacement=replacement,
                )
            else:
                self._show_error_if_checking(
                    unused,
                    f"Assigned value of {unused.id} is never accessed",
                    error_code=ErrorCode.unused_assignment,
                    replacement=replacement,
                )

    def value_of_annotation(self, node: ast.expr) -> Value:
        expr = self.expr_of_annotation(node)
        val, _ = expr.unqualify()
        return val

    def expr_of_annotation(self, node: ast.expr) -> AnnotationExpr:
        with override(self, "state", VisitorState.collect_names):
            annotated_type = self._visit_annotation(node)
        return self._expr_of_annotation_type(annotated_type, node)

    def _visit_annotation(self, node: ast.AST) -> Value:
        with override(self, "in_annotation", True):
            val = self.visit(node)
            if self._is_invalid_generic_annotation_node(node):
                self._show_error_if_checking(
                    node,
                    "Generic[...] is valid only as a base class",
                    error_code=ErrorCode.invalid_annotation,
                )
            self.check_for_missing_generic_params(node, val)
            return val

    def _is_invalid_generic_annotation_node(self, node: ast.AST) -> bool:
        target = node.value if isinstance(node, ast.Subscript) else node
        value = value_from_ast(target, visitor=self, error_on_unrecognized=False)
        for subval in flatten_values(replace_fallback(value)):
            candidate: object
            if isinstance(subval, SyntheticClassObjectValue):
                subval = subval.class_type
            if isinstance(subval, KnownValue):
                candidate = subval.val
            elif isinstance(subval, TypedValue):
                candidate = subval.typ
            else:
                continue
            if is_typing_name(candidate, "Generic"):
                return True
        return False

    def check_for_missing_generic_params(self, node: ast.AST, value: Value) -> None:
        if not isinstance(value, KnownValue):
            return
        val = value.val
        if not safe_isinstance(val, type):
            args = get_args(val)
            if args:
                return
            val = get_origin(val)
            if not safe_isinstance(val, type):
                return
            if val is tuple and value.val is not tuple:
                # tuple[()]
                return
        if isinstance(val, GenericAlias):
            return
        generic_params = self.arg_spec_cache.get_type_parameters(val)
        if not generic_params:
            return
        self.show_error(
            node,
            f"Missing type parameters for generic type {stringify_object(value.val)}",
            error_code=ErrorCode.missing_generic_parameters,
        )

    def _expr_of_annotation_type(self, val: Value, node: ast.AST) -> AnnotationExpr:
        """Given a value encountered in a type annotation, return a type."""
        return annotation_expr_from_value(
            val, visitor=self, node=node, suppress_errors=self._is_collecting()
        )

    def _check_method_first_arg(
        self, node: FunctionNode, function_info: FunctionInfo
    ) -> None:
        """Makes sure the first argument to a method is self or cls."""
        if not isinstance(self.current_class, type):
            return
        # staticmethods have no restrictions
        if FunctionDecorator.staticmethod in function_info.decorator_kinds:
            return
        # try to confirm that it's actually a method
        if isinstance(node, ast.Lambda) or not hasattr(self.current_class, node.name):
            return
        if node.name in IMPLICIT_CLASSMETHODS:
            return
        first_must_be = (
            "cls"
            if FunctionDecorator.classmethod in function_info.decorator_kinds
            else "self"
        )

        if len(node.args.args) < 1 or len(node.args.defaults) == len(node.args.args):
            self.show_error(
                node,
                "Method must have at least one non-keyword argument",
                ErrorCode.method_first_arg,
            )
        elif node.args.args[0].arg != first_must_be:
            self.show_error(
                node,
                f"First argument to method should be {first_must_be}",
                ErrorCode.method_first_arg,
            )

    def visit_Global(self, node: ast.Global) -> None:
        if self.scopes.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_global)
            return

        module_scope = self.scopes.module_scope()
        for name in node.names:
            if self.unused_finder is not None and module_scope.scope_object is not None:
                assert isinstance(module_scope.scope_object, types.ModuleType)
                self.unused_finder.record(
                    module_scope.scope_object, name, module_scope.scope_object.__name__
                )
            with (
                override(self, "ann_assign_type", (None, True))
                if module_scope.is_final(name)
                else contextlib.nullcontext()
            ):
                self._set_name_in_scope(
                    name, node, ReferencingValue(module_scope, name)
                )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        if self.scopes.scope_type() != ScopeType.function_scope:
            self._show_error_if_checking(node, error_code=ErrorCode.bad_nonlocal)
            return

        for name in node.names:
            defining_scope = self.scopes.get_nonlocal_scope(
                name, self.scopes.current_scope()
            )
            if defining_scope is None:
                # this is a SyntaxError, so it might be impossible to reach this branch
                self._show_error_if_checking(
                    node,
                    f"nonlocal name {name} does not exist in any enclosing scope",
                    error_code=ErrorCode.bad_nonlocal,
                )
                defining_scope = self.scopes.module_scope()
            with (
                override(self, "ann_assign_type", (None, True))
                if defining_scope.is_final(name)
                else contextlib.nullcontext()
            ):
                self._set_name_in_scope(
                    name, node, ReferencingValue(defining_scope, name)
                )

    def check_deprecation(self, node: ast.AST, value: Value) -> bool:
        if isinstance(value, AnnotatedValue):
            if value.has_metadata_of_type(SkipDeprecatedExtension):
                return False
            for metadata in value.get_metadata_of_type(DeprecatedExtension):
                self._show_error_if_checking(
                    node,
                    f"{value} is deprecated: {metadata.deprecation_message}",
                    error_code=ErrorCode.deprecated,
                )
                return True
            return self.check_deprecation(node, value.value)
        if isinstance(value, InputSigValue):
            return False
        value = replace_fallback(value)
        if isinstance(value, UnboundMethodValue):
            method = value.get_method()
            if method is None:
                return False
            return self.check_deprecation(node, KnownValue(method))
        if isinstance(value, CallableValue):
            if not isinstance(value.signature, Signature):
                return False
            if value.signature.deprecated is None:
                return False
            deprecated = value.signature.deprecated
        elif isinstance(value, KnownValue):
            deprecated = safe_getattr(value.val, "__deprecated__", None)
            if deprecated is None:
                return False
        else:
            return False
        if not safe_isinstance(deprecated, str):
            # happens with Mock objects
            return False
        self._show_error_if_checking(
            node,
            f"{value} is deprecated: {deprecated}",
            error_code=ErrorCode.deprecated,
        )
        return True

    # Imports

    def check_for_disallowed_import(
        self, node: ast.AST, name: str, *, check_parents: bool = True
    ) -> None:
        disallowed = self.options.get_value_for(DisallowedImports)
        parts = name.split(".") if check_parents else [name]
        for i in range(len(parts)):
            name_to_check = ".".join(parts[: i + 1])
            if name_to_check in disallowed:
                self._show_error_if_checking(
                    node,
                    f"Disallowed import of module {name!r}",
                    error_code=ErrorCode.disallowed_import,
                )
                break

    def visit_Import(self, node: ast.Import) -> None:
        self.generic_visit(node)
        if self.scopes.scope_type() == ScopeType.module_scope:
            for name in node.names:
                self.import_name_to_node[name.name] = node

        for alias in node.names:
            self.check_for_disallowed_import(node, alias.name)
            self._try_to_import(alias.name)
            # "import a.b" sets the name "a", but "import a.b as c" sets "c" to the value "a.b"
            varname = (
                alias.name if alias.asname is not None else alias.name.split(".")[0]
            )
            mod = self._get_module(varname, node)
            self._set_alias_in_scope(alias, mod, node=node)

    def _set_alias_in_scope(
        self,
        alias: ast.alias,
        value: Value,
        *,
        force_public: bool = False,
        is_final: bool = False,
        node: ast.AST,
    ) -> None:
        if self.check_deprecation(alias, value):
            value = annotate_value(value, [SkipDeprecatedExtension()])
        annotation_ctx = (
            override(self, "ann_assign_type", (None, True))
            if is_final
            else contextlib.nullcontext()
        )
        with annotation_ctx:
            if alias.asname is not None:
                self._set_name_in_scope(
                    alias.asname,
                    alias,
                    value,
                    private=not force_public and alias.asname != alias.name,
                )
            else:
                self._set_name_in_scope(
                    alias.name.split(".")[0], alias, value, private=not force_public
                )

    def _get_module(self, name: str, node: ast.AST) -> Value:
        if name not in sys.modules:
            self._try_to_import(name)
        if name in sys.modules:
            # import a.b.c only succeeds if a.b.c is a module that
            # exists, but it doesn't return the module a.b.c, it
            # follows the attribute chain. But this isn't true for
            # ImportFrom.
            if isinstance(node, ast.ImportFrom):
                return KnownValue(sys.modules[name])
            pieces = name.split(".")
            base_module = sys.modules.get(pieces[0])
            for piece in pieces[1:]:
                if not safe_hasattr(base_module, piece):
                    self._show_error_if_checking(
                        node,
                        f"Cannot import {name} because {piece} is not an attribute"
                        f" of {base_module!r}",
                        error_code=ErrorCode.import_failed,
                    )
                    return AnyValue(AnySource.unresolved_import)
                base_module = getattr(base_module, piece)
            return KnownValue(base_module)
        else:
            # TODO: Maybe get the module from stubs?
            self._show_error_if_checking(
                node, f"Cannot import {name}", error_code=ErrorCode.import_failed
            )
            return AnyValue(AnySource.unresolved_import)

    def _try_to_import(self, module_name: str) -> None:
        try:
            __import__(module_name)
        except Exception:
            self._try_to_import_stub(module_name)

    def _try_to_import_stub(self, module_name: str) -> None:
        parts = module_name.split(".")
        search_paths = [
            *[Path(path) for path in self.options.get_value_for(ImportPaths)],
            Path.cwd(),
        ]
        for entry in sys.path:
            try:
                search_paths.append(Path(entry))
            except TypeError:
                continue
        for base in search_paths:
            module_file = base.joinpath(*parts).with_suffix(".pyi")
            package_file = base.joinpath(*parts, "__init__.pyi")
            for candidate in (module_file, package_file):
                if not candidate.exists():
                    continue
                try:
                    importer.import_module(module_name, candidate.resolve())
                except Exception:
                    continue
                return

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.generic_visit(node)
        # this is used to decide where to add additional imports (after the first import), so
        # exclude __future__ imports
        if (
            self.scopes.scope_type() == ScopeType.module_scope
            and node.module
            and node.module != "__future__"
        ):
            self.import_name_to_node[node.module] = node
        if node.module == "__future__":
            for name in node.names:
                self.future_imports.add(name.name)

        if node.module is not None and node.level == 0:
            self.check_for_disallowed_import(node, node.module)
            for alias in node.names:
                self.check_for_disallowed_import(
                    alias, f"{node.module}.{alias.name}", check_parents=False
                )

        self._maybe_record_usages_from_import(node)

        # See if we can get the names from the stub instead
        if (
            node.module is not None
            and node.level == 0
            # pycroscope.extensions has a stub only for the purpose of other stubs
            # it shouldn't be used for runtime imports
            and node.module != "pycroscope.extensions"
        ):
            path = typeshed_client.ModulePath(tuple(node.module.split(".")))
            finder = self.checker.ts_finder
            mod = finder.resolver.get_module(path)
            if mod.exists:
                for alias in node.names:
                    val = finder.resolve_name(node.module, alias.name)
                    if val is UNINITIALIZED_VALUE:
                        self._show_error_if_checking(
                            node,
                            f"Cannot import name {alias.name!r} from {node.module!r}",
                            ErrorCode.import_failed,
                        )
                        val = AnyValue(AnySource.error)
                    self._set_alias_in_scope(alias, val, node=node)
                return

        is_init = self.filename.endswith("/__init__.py")
        source_module = self._get_import_from_module(node)

        # from .a import b implicitly sets a in the parent module's namespace.
        # We allow relying on this behavior.
        if (
            is_init
            and node.module is not None
            and "." not in node.module
            and node.level == 1
        ):
            self._set_name_in_scope(node.module, node, source_module, private=False)

        for alias in node.names:
            if alias.name == "*":
                if isinstance(source_module, KnownValue) and isinstance(
                    source_module.val, types.ModuleType
                ):
                    for name, val in source_module.val.__dict__.items():
                        if name.startswith("_"):
                            continue
                        with (
                            override(self, "ann_assign_type", (None, True))
                            if self._is_final_imported_name(source_module, name)
                            else contextlib.nullcontext()
                        ):
                            self._set_name_in_scope(
                                name, alias, KnownValue(val), private=False
                            )
                else:
                    self._show_error_if_checking(
                        node,
                        f"Cannot import * from unresolved module {node.module!r}",
                        ErrorCode.invalid_import,
                    )
                continue
            val = self._get_import_from_value(source_module, alias.name, node)
            self._set_alias_in_scope(
                alias,
                val,
                force_public=is_init and node.level == 1,
                is_final=self._is_final_imported_name(source_module, alias.name),
                node=node,
            )

    def _get_import_from_value(
        self, source_module: Value, alias_name: str, node: ast.ImportFrom
    ) -> Value:
        val = self.get_attribute_from_value(source_module, alias_name)
        if val is not UNINITIALIZED_VALUE:
            return val
        if isinstance(source_module, KnownValue) and isinstance(
            source_module.val, types.ModuleType
        ):
            name = f"{source_module.val.__name__}.{alias_name}"
            self._try_to_import(name)
            val = self.get_attribute_from_value(source_module, alias_name)
            if val is not UNINITIALIZED_VALUE:
                return val
            self._try_to_import_stub(source_module.val.__name__)
            refreshed_source_module = self._get_module(source_module.val.__name__, node)
            val = self.get_attribute_from_value(refreshed_source_module, alias_name)
            if val is not UNINITIALIZED_VALUE:
                return val

        self._show_error_if_checking(
            node,
            f"Cannot import name {alias_name!r} from {node.module!r}",
            ErrorCode.import_failed,
        )
        return AnyValue(AnySource.error)

    def _get_import_from_module(self, node: ast.ImportFrom) -> Value:
        if node.level > 0:
            if self.module is None:
                return AnyValue(AnySource.unresolved_import)
            level = node.level
            if self.filename.endswith("/__init__.py"):
                level -= 1

            current_module_path = [
                str(part) for part in self.module.__name__.split(".")
            ]
            if level >= len(current_module_path):
                self._show_error_if_checking(
                    node,
                    "Attempted relative import beyond top-level package",
                    error_code=ErrorCode.invalid_import,
                )
                return AnyValue(AnySource.error)
            if level:
                current_module_path = current_module_path[:-level]
            if node.module is not None:
                current_module_path.append(node.module)
            module_name = ".".join(current_module_path)
        else:
            # Should be disallowed by the AST
            if node.module is None:
                self._show_error_if_checking(
                    node,
                    "Attempted absolute import without module name",
                    error_code=ErrorCode.invalid_import,
                )
                return AnyValue(AnySource.error)
            module_name = node.module
        return self._get_module(module_name, node)

    def _maybe_record_usages_from_import(self, node: ast.ImportFrom) -> None:
        if self.unused_finder is None or self.module is None:
            return
        if self._is_unimportable_module(node):
            return
        if node.level == 0:
            module_name = node.module
        else:
            if self.filename.endswith("/__init__.py"):
                this_module_name = self.module.__name__ + ".__init__"
            else:
                this_module_name = self.module.__name__
            parent_module_name = this_module_name.rsplit(".", maxsplit=node.level)[0]
            if node.module is not None:
                module_name = parent_module_name + "." + node.module
            else:
                module_name = parent_module_name
        if module_name is None:
            return
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = __import__(module_name)
            except Exception:
                return
        for alias in node.names:
            if alias.name == "*":
                self.unused_finder.record_import_star(module, self.module)
            else:
                self.unused_finder.record(module, alias.name, self.module.__name__)

    def _is_unimportable_module(self, node: ast.Import | ast.ImportFrom) -> bool:
        unimportable = self.options.get_value_for(UnimportableModules)
        if isinstance(node, ast.ImportFrom):
            # the split is needed for cases like "from foo.bar import baz" if foo is unimportable
            return node.module is not None and node.module.split(".")[0] in unimportable
        else:
            # need the split if the code is "import foo.bar as bar" if foo is unimportable
            return any(name.name.split(".")[0] in unimportable for name in node.names)

    # Comprehensions

    def visit_DictComp(self, node: ast.DictComp) -> Value:
        return self._visit_sequence_comp(node, dict)

    def visit_ListComp(self, node: ast.ListComp) -> Value:
        return self._visit_sequence_comp(node, list)

    def visit_SetComp(self, node: ast.SetComp) -> Value:
        return self._visit_sequence_comp(node, set)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Value:
        return self._visit_sequence_comp(node, types.GeneratorType)

    def visit_comprehension(
        self, node: ast.comprehension, iterable_type: Value | None = None
    ) -> None:
        if iterable_type is None:
            is_async = bool(node.is_async)
            iterable_type = self._member_value_of_iterator(node.iter, is_async)
            if not isinstance(iterable_type, Value):
                iterable_type = unite_and_simplify(
                    *iterable_type,
                    limit=self.options.get_value_for(UnionSimplificationLimit),
                )
        with override(self, "in_comprehension_body", True):
            with override(self, "being_assigned", iterable_type):
                self.visit(node.target)
            for cond in node.ifs:
                _, constraint = self.constraint_from_condition(cond)
                self.add_constraint(cond, constraint)

    def _visit_sequence_comp(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        typ: type,
    ) -> Value:
        # the iteree of the first generator is executed in the enclosing scope
        is_async = bool(node.generators[0].is_async)
        iterable_type = self._member_value_of_iterator(
            node.generators[0].iter, is_async
        )
        if self.state == VisitorState.collect_names:
            # Visit it once to get usage nodes for usage of nested variables. This enables
            # us to inherit constraints on nested variables.
            # Strictly speaking this is unsafe to do for generator expressions, which may
            # be evaluated at a different place in the function than where they are defined,
            # but that is unlikely to be an issue in practice.
            with (
                self.scopes.add_scope(ScopeType.function_scope, scope_node=node),
                override(self, "_name_node_to_statement", {}),
            ):
                return self._visit_comprehension_inner(node, typ, iterable_type)

        with (
            self.scopes.add_scope(ScopeType.function_scope, scope_node=node),
            override(self, "_name_node_to_statement", {}),
        ):
            scope = self.scopes.current_scope()
            assert isinstance(scope, FunctionScope)
            for state in (VisitorState.collect_names, VisitorState.check_names):
                with override(self, "state", state):
                    ret = self._visit_comprehension_inner(node, typ, iterable_type)
            stmt = self.node_context.nearest_enclosing(ast.stmt)
            assert isinstance(stmt, ast.stmt)
            self._check_function_unused_vars(scope, enclosing_statement=stmt)
        return ret

    def _visit_comprehension_inner(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        typ: type,
        iterable_type: Value | Sequence[Value],
    ) -> Value:
        if not isinstance(iterable_type, Value):
            # If it is a simple comprehension (only one generator, no ifs) and we know
            # the exact iterated values, we try to infer an IncompleteValue instead.
            if (
                len(node.generators) == 1
                and not node.generators[0].ifs
                and 0
                < len(iterable_type)
                <= self.options.get_value_for(ComprehensionLengthInferenceLimit)
            ):
                generator = node.generators[0]
                if isinstance(node, ast.DictComp):
                    items = []
                    self.node_context.contexts.append(generator)
                    try:
                        for val in iterable_type:
                            self.visit_comprehension(generator, iterable_type=val)
                            with override(self, "in_comprehension_body", True):
                                # PEP 572 mandates that the key be evaluated first.
                                key = self.visit(node.key)
                                value = self.visit(node.value)
                                items.append(KVPair(key, value))
                    finally:
                        self.node_context.contexts.pop()
                    return DictIncompleteValue(typ, items)
                elif isinstance(node, (ast.ListComp, ast.SetComp)):
                    elts = []
                    self.node_context.contexts.append(generator)
                    try:
                        for val in iterable_type:
                            self.visit_comprehension(generator, iterable_type=val)
                            with override(self, "in_comprehension_body", True):
                                elts.append((False, self.visit(node.elt)))
                    finally:
                        self.node_context.contexts.pop()
                    return SequenceValue(typ, elts)

            iterable_type = unite_and_simplify(
                *iterable_type,
                limit=self.options.get_value_for(UnionSimplificationLimit),
            )
        # need to visit the generator expression first so that we know of variables
        # created in them
        for i, generator in enumerate(node.generators):
            # for generators after the first one, compute the iterable_type inside
            # the comprehension's scope
            self.node_context.contexts.append(generator)
            try:
                self.visit_comprehension(
                    generator, iterable_type=iterable_type if i == 0 else None
                )
            finally:
                self.node_context.contexts.pop()

        if isinstance(node, ast.DictComp):
            with override(self, "in_comprehension_body", True):
                key_value = self.visit(node.key)
                value_value = self.visit(node.value)

                hashability = check_hashability(key_value, self)
                if isinstance(hashability, CanAssignError):
                    self._show_error_if_checking(
                        node.key,
                        "Dictionary key is not hashable",
                        ErrorCode.unhashable_key,
                        detail=str(hashability),
                    )
                    key_value = AnyValue(AnySource.error)
            return DictIncompleteValue(
                dict, [KVPair(key_value, value_value, is_many=True)]
            )

        with override(self, "in_comprehension_body", True):
            member_value = self.visit(node.elt)

            if typ is set:
                hashability = check_hashability(member_value, self)
                if isinstance(hashability, CanAssignError):
                    self._show_error_if_checking(
                        node.elt,
                        "Set member is not hashable",
                        ErrorCode.unhashable_key,
                        detail=str(hashability),
                    )
                    member_value = AnyValue(AnySource.error)

        if typ is types.GeneratorType:
            return GenericValue(typ, [member_value, KnownValue(None), KnownValue(None)])
        # Returning a SequenceValue here instead of a GenericValue allows
        # later code to modify this container.
        return SequenceValue(typ, [(True, member_value)])

    # Literals and displays

    def _is_literal_string_compatible(self, val: Value) -> bool:
        return is_subtype(TypedValue(str, literal_only=True), val, self)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> Value:
        elements = self._generic_visit_list(node.values)
        if all(self._is_literal_string_compatible(elt) for elt in elements):
            fallback = TypedValue(str, literal_only=True)
        else:
            fallback = TypedValue(str)
        limit = self.options.get_value_for(UnionSimplificationLimit)
        possible_values: list[list[str]] = [[]]
        for elt in elements:
            subvals = list(flatten_values(elt))
            # Bail out if the list of possible values gets too long.
            if len(possible_values) * len(subvals) > limit:
                return TypedValue(str)
            to_add = []
            for subval in subvals:
                if not isinstance(subval, KnownValue):
                    return fallback
                if not isinstance(subval.val, str):
                    return fallback
                to_add.append(subval.val)
            possible_values = [
                lst + [new_elt] for lst in possible_values for new_elt in to_add
            ]
        return unite_values(*[KnownValue("".join(lst)) for lst in possible_values])

    def visit_FormattedValue(self, node: ast.FormattedValue) -> Value:
        val = self.visit(node.value)
        value_is_literal_string = self._is_literal_string_compatible(val)
        format_spec_val = (
            self.visit(node.format_spec) if node.format_spec else KnownValue("")
        )
        format_spec_is_literal_string = (
            self._is_literal_string_compatible(format_spec_val)
            if node.format_spec
            else True
        )
        if isinstance(format_spec_val, KnownValue) and isinstance(
            format_spec_val.val, str
        ):
            format_spec = format_spec_val.val
            possible_vals = []
            for subval in flatten_values(val):
                possible_vals.append(
                    self._visit_single_formatted_value(subval, node, format_spec)
                )
            result = unite_and_simplify(
                *possible_vals,
                limit=self.options.get_value_for(UnionSimplificationLimit),
            )
        else:
            # TODO: statically check whether the format specifier is valid.
            result = TypedValue(str)

        if not (value_is_literal_string and format_spec_is_literal_string):
            return TypedValue(str)

        if all(
            isinstance(subval, KnownValue) and isinstance(subval.val, str)
            for subval in flatten_values(result)
        ):
            return result
        return TypedValue(str, literal_only=True)

    def _visit_single_formatted_value(
        self, val: Value, node: ast.FormattedValue, format_spec: str
    ) -> Value:
        if not isinstance(val, KnownValue):
            return TypedValue(str)
        output = val.val
        if node.conversion != -1:
            unsupported_conversion = False
            try:
                if node.conversion == ord("a"):
                    output = ascii(output)
                elif node.conversion == ord("s"):
                    output = str(output)
                elif node.conversion == ord("r"):
                    output = repr(output)
                else:
                    unsupported_conversion = True
            except Exception:
                # str/repr/ascii failed
                return TypedValue(str)
            if unsupported_conversion:
                raise NotImplementedError(
                    f"Unsupported conversion specifier {node.conversion}"
                )
        try:
            output = format(output, format_spec)
        except Exception:
            # format failed
            return TypedValue(str)
        return KnownValue(output)

    def visit_Constant(self, node: ast.Constant) -> Value:
        if isinstance(node.value, str):
            self._maybe_show_missing_f_error(node, node.value)
        return KnownValue(node.value)

    def _maybe_show_missing_f_error(self, node: ast.AST, s: str | bytes) -> None:
        """Show an error if this string was probably meant to be an f-string."""
        if isinstance(s, bytes):
            return
        if "{" not in s:
            return
        f_str = "f" + repr(s)
        try:
            f_str_ast = ast.parse(f_str)
        except SyntaxError:
            return
        names = {
            subnode.id
            for subnode in ast.walk(f_str_ast)
            if isinstance(subnode, ast.Name)
        }
        # TODO:
        # - use nearest_enclosing() to find the Call node
        # - don't suggest this if there's (a lot of?) stuff after :
        # - some false positives with SQL queries
        # - if there are implicitly concatenated strings, the errors are correct, but
        #   we point to the wrong line and give wrong suggested fixes because the AST is weird
        if names and all(self._name_exists(name) for name in names):
            parent = self.node_context.nth_parent(2)
            if parent is not None:
                # the string is immediately .format()ed, probably doesn't need to be an f-string
                if isinstance(parent, ast.Attribute) and parent.attr == "format":
                    return
                # probably a docstring
                elif isinstance(parent, ast.Expr):
                    return
                # Probably a function that does template-like interpolation itself. In practice
                # this covers our translation API (translate("hello {user}", user=...)).
                elif isinstance(parent, ast.Call):
                    keywords = {kw.arg for kw in parent.keywords if kw.arg is not None}
                    if names <= keywords:
                        return
            stmt = f_str_ast.body[0]
            assert isinstance(stmt, ast.Expr), f"unexpected ast {ast.dump(f_str_ast)}"
            self._show_error_if_checking(
                node,
                error_code=ErrorCode.missing_f,
                replacement=self.replace_node(node, stmt.value),
            )

    def _name_exists(self, name: str) -> bool:
        try:
            val = self.scopes.get(
                name, None, VisitorState.check_names, can_assign_ctx=self
            )
        except KeyError:
            return False
        else:
            return val is not UNINITIALIZED_VALUE

    def visit_Dict(self, node: ast.Dict) -> Value:
        ret = {}
        all_pairs: list[KVPair] = []
        has_non_literal = False
        for key_node, value_node in zip(node.keys, node.values):
            value_val = self.visit(value_node)
            # ** unpacking
            if key_node is None:
                has_non_literal = True
                new_pairs = kv_pairs_from_mapping(value_val, self)
                if isinstance(new_pairs, CanAssignError):
                    self._show_error_if_checking(
                        value_node,
                        f"{value_val} is not a mapping",
                        ErrorCode.unsupported_operation,
                        detail=str(new_pairs),
                    )
                    return TypedValue(dict)
                all_pairs += new_pairs
                continue
            key_val = self.visit(key_node)

            hashability = check_hashability(key_val, self)
            if isinstance(hashability, CanAssignError):
                self._show_error_if_checking(
                    key_node,
                    "Dictionary key is not hashable",
                    ErrorCode.unhashable_key,
                    detail=str(hashability),
                )

            all_pairs.append(KVPair(key_val, value_val))
            if not isinstance(key_val, KnownValue) or not isinstance(
                value_val, KnownValue
            ):
                has_non_literal = True
            value = value_val.val if isinstance(value_val, KnownValue) else None

            if not isinstance(key_val, KnownValue):
                continue

            key = key_val.val

            try:
                already_exists = key in ret
            except TypeError:
                continue

            if already_exists:
                self._show_error_if_checking(
                    key_node,
                    f"Duplicate dictionary key {key!r}",
                    ErrorCode.duplicate_dict_key,
                )
            ret[key] = value

        if has_non_literal:
            return DictIncompleteValue(dict, all_pairs)
        else:
            return KnownValue(ret)

    def visit_Set(self, node: ast.Set) -> Value:
        return self._visit_display_read(node, set)

    def visit_List(self, node: ast.List) -> Value | None:
        return self._visit_display(node, list)

    def visit_Tuple(self, node: ast.Tuple) -> Value | None:
        return self._visit_display(node, tuple)

    def _visit_display(self, node: ast.List | ast.Tuple, typ: type) -> Value | None:
        if self._is_write_ctx(node.ctx):
            target_length = 0
            post_starred_length = None
            for target in node.elts:
                if isinstance(target, ast.Starred):
                    if post_starred_length is not None:
                        # This is a SyntaxError at runtime so it should never happen
                        self.show_error(
                            node,
                            "Two starred expressions in assignment",
                            error_code=ErrorCode.unexpected_node,
                        )
                        with override(
                            self, "being_assigned", AnyValue(AnySource.error)
                        ):
                            return self.generic_visit(node)
                    else:
                        post_starred_length = 0
                elif post_starred_length is not None:
                    post_starred_length += 1
                else:
                    target_length += 1

            assert (
                self.being_assigned is not None
            ), "annotated assignment can only have a single target"
            being_assigned = unpack_values(
                self.being_assigned, self, target_length, post_starred_length
            )
            if isinstance(being_assigned, CanAssignError):
                self.show_error(
                    node,
                    f"Cannot unpack {self.being_assigned}",
                    ErrorCode.bad_unpack,
                    detail=str(being_assigned),
                )
                with override(self, "being_assigned", AnyValue(AnySource.error)):
                    return self.generic_visit(node)

            for target, value in zip(node.elts, being_assigned):
                with override(self, "being_assigned", value):
                    self.visit(target)
            return None
        else:
            return self._visit_display_read(node, typ)

    def _visit_display_read(
        self, node: ast.Set | ast.List | ast.Tuple, typ: type
    ) -> Value:
        if typ is tuple and self.in_annotation:
            elts = []
            for elt in node.elts:
                val = self.visit(elt)
                self.check_for_missing_generic_params(elt, val)
                elts.append(val)
        else:
            elts = [self.visit(elt) for elt in node.elts]
        return self._maybe_make_sequence(typ, elts, node, elt_nodes=node.elts)

    def _maybe_make_sequence(
        self,
        typ: type,
        elts: Sequence[Value],
        node: ast.AST,
        elt_nodes: Sequence[ast.AST] | None = None,
    ) -> Value:
        values = []
        for i, elt in enumerate(elts):
            if (
                isinstance(elt, PartialValue)
                and elt.operation is PartialValueOperation.UNPACK
            ):
                vals = concrete_values_from_iterable(elt.root, self)
                if isinstance(vals, CanAssignError):
                    self.show_error(
                        elt.node,
                        f"{elt.root} is not iterable",
                        ErrorCode.unsupported_operation,
                        detail=str(vals),
                    )
                    new_vals = [(True, AnyValue(AnySource.error))]
                elif isinstance(vals, Value):
                    # single value
                    new_vals = [(True, vals)]
                else:
                    new_vals = [(False, val) for val in vals]
                if typ is set:
                    for _, val in new_vals:
                        hashability = check_hashability(val, self)
                        if isinstance(hashability, CanAssignError):
                            if elt_nodes:
                                error_node = elt_nodes[i]
                            else:
                                error_node = node
                            self._show_error_if_checking(
                                error_node,
                                "Set element is not hashable",
                                ErrorCode.unhashable_key,
                                detail=str(hashability),
                            )

                values += new_vals
            else:
                if typ is set:
                    hashability = check_hashability(elt, self)
                    if isinstance(hashability, CanAssignError):
                        if elt_nodes:
                            error_node = elt_nodes[i]
                        else:
                            error_node = node
                        self._show_error_if_checking(
                            error_node,
                            "Set element is not hashable",
                            ErrorCode.unhashable_key,
                            detail=str(hashability),
                        )
                values.append((False, elt))

        return SequenceValue.make_or_known(typ, values)

    # Operations

    def visit_BoolOp(self, node: ast.BoolOp) -> Value:
        # Visit an AND or OR expression.

        # We want to show an error if the left operand in a BoolOp is always true,
        # so we use constraint_from_condition.

        # Within the BoolOp itself we set additional constraints: for an AND
        # clause we know that if it is executed, all constraints to its left must
        # be true, so we set a positive constraint; for OR it is the opposite, so
        # we set a negative constraint.

        is_and = isinstance(node.op, ast.And)
        stack = contextlib.ExitStack()
        scopes = []
        out_constraints = []
        values = []
        constraint = NULL_CONSTRAINT
        definite_value = None
        with stack:
            for i, condition in enumerate(node.values):
                is_last = i == len(node.values) - 1
                scope = stack.enter_context(self.scopes.subscope())
                scopes.append(scope)
                if is_and:
                    self.add_constraint(condition, constraint)
                else:
                    self.add_constraint(condition, constraint.invert())

                new_value, constraint = self.constraint_from_condition(
                    condition, check_boolability=not is_last
                )
                new_def_val = _extract_definite_value(new_value)
                if is_and and new_def_val is False:
                    definite_value = False
                    stack.enter_context(self.catch_errors())
                elif not is_and and new_def_val is True:
                    definite_value = True
                    stack.enter_context(self.catch_errors())
                out_constraints.append(constraint)

                if is_last:
                    values.append(new_value)
                elif is_and:
                    values.append(
                        constrain_value(new_value, FALSY_CONSTRAINT, ctx=self)
                    )
                else:
                    values.append(
                        constrain_value(new_value, TRUTHY_CONSTRAINT, ctx=self)
                    )

        self.scopes.combine_subscopes(scopes)
        out = unite_values(*values)
        if definite_value is not None:
            out = annotate_value(out, [DefiniteValueExtension(definite_value)])
        if is_and:
            constraint = AndConstraint.make(reversed(out_constraints))
            return annotate_with_constraint(out, constraint)
        else:
            # For OR conditions, no need to add a constraint here; we'll
            # return a Union and extract_constraints() will combine them.
            return out

    def visit_Compare(self, node: ast.Compare) -> Value:
        nodes = [node.left, *node.comparators]
        vals = [self._visit_possible_constraint(node) for node in nodes]
        results = []
        constraints = []
        for i, (rhs_node, rhs) in enumerate(zip(nodes, vals)):
            if i == 0:
                continue
            op = node.ops[i - 1]
            lhs_node = nodes[i - 1]
            lhs = vals[i - 1]
            result = self._visit_single_compare(lhs_node, lhs, op, rhs_node, rhs, node)
            constraints.append(extract_constraints(result))
            result, _ = unannotate_value(result, ConstraintExtension)
            results.append(result)
        return annotate_with_constraint(
            unite_values(*results), AndConstraint.make(constraints)
        )

    def check_for_unsafe_comparison(
        self, op: ast.cmpop, lhs: Value, rhs: Value, parent_node: ast.AST
    ) -> None:
        if lhs == KnownNone or rhs == KnownNone:
            return
        if isinstance(op, (ast.Eq, ast.NotEq)):
            mode = OverlapMode.EQ
        elif isinstance(op, (ast.Is, ast.IsNot)):
            mode = OverlapMode.IS
        else:
            return
        if self._is_enum_value_for_unsafe_comparison(
            lhs
        ) and self._is_enum_value_for_unsafe_comparison(rhs):
            return
        if KnownNone.is_assignable(lhs, self) or KnownNone.is_assignable(rhs, self):
            return
        error = lhs.can_overlap(rhs, self, mode)
        if error is None:
            return
        lhs_shown = lhs
        rhs_shown = rhs
        for ignored_extension in (
            SysPlatformExtension,
            SysVersionInfoExtension,
            ConstraintExtension,
            DefiniteValueExtension,
        ):
            lhs_shown, _ = unannotate_value(lhs_shown, ignored_extension)
            rhs_shown, _ = unannotate_value(rhs_shown, ignored_extension)
        self._show_error_if_checking(
            msg=f"Comparison between objects that do not overlap: {lhs_shown} and {rhs_shown}",
            error_code=ErrorCode.unsafe_comparison,
            detail=str(error),
            node=parent_node,
        )

    def _is_enum_value_for_unsafe_comparison(self, value: Value) -> bool:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._is_enum_value_for_unsafe_comparison(value.value)
        if isinstance(value, MultiValuedValue):
            return all(
                self._is_enum_value_for_unsafe_comparison(subval)
                for subval in value.vals
            )
        if isinstance(value, SyntheticClassObjectValue):
            return self._is_enum_value_for_unsafe_comparison(value.class_type)
        if isinstance(value, KnownValue):
            return safe_isinstance(value.val, enum.Enum)
        if isinstance(value, TypedValue):
            if isinstance(value.typ, type):
                return safe_issubclass(value.typ, enum.Enum)
            if isinstance(value.typ, str):
                return self._is_enum_class_key(value.typ)
            return False
        if isinstance(value, SubclassValue):
            return self._is_enum_value_for_unsafe_comparison(value.typ)
        return False

    def _visit_single_compare(
        self,
        lhs_node: ast.expr,
        lhs: Value,
        op: ast.cmpop,
        rhs_node: ast.expr,
        rhs: Value,
        parent_node: ast.AST,
    ) -> Value:
        self.check_for_unsafe_comparison(op, lhs, rhs, parent_node)
        self._check_dataclass_order_comparison(op, lhs, rhs, parent_node)

        lhs_constraint = extract_constraints(lhs)
        rhs_constraint = extract_constraints(rhs)
        rhs = replace_fallback(rhs)
        definite_value = None
        if isinstance(lhs, AnnotatedValue):
            if (
                SYS_PLATFORM_EXTENSION in lhs.metadata
                and isinstance(rhs, KnownValue)
                and isinstance(op, (ast.Eq, ast.NotEq))
            ):
                op_func, _, _ = COMPARATOR_TO_OPERATOR[type(op)]
                definite_value = op_func(sys.platform, rhs.val)
            elif (
                SYS_VERSION_INFO_EXTENSION in lhs.metadata
                and isinstance(rhs, KnownValue)
                and isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE))
            ):
                op_func, _, _ = COMPARATOR_TO_OPERATOR[type(op)]
                definite_value = op_func(sys.version_info, rhs.val)
        lhs = replace_fallback(lhs)
        if isinstance(lhs_constraint, PredicateProvider) and isinstance(
            rhs, KnownValue
        ):
            constraint = self._constraint_from_predicate_provider(
                lhs_constraint, rhs.val, op
            )
        elif isinstance(rhs_constraint, PredicateProvider) and isinstance(
            lhs, KnownValue
        ):
            constraint = self._constraint_from_predicate_provider(
                rhs_constraint, lhs.val, op
            )
        elif isinstance(rhs, KnownValue):
            constraint = self._constraint_from_compare_op(
                lhs_node, rhs.val, op, is_right=True
            )
        elif isinstance(lhs, KnownValue):
            constraint = self._constraint_from_compare_op(
                rhs_node, lhs.val, op, is_right=False
            )
        else:
            constraint = NULL_CONSTRAINT
        if isinstance(op, (ast.Is, ast.IsNot)):
            # is and is not always return a boolean and don't forward to a dunder.
            val = TypedValue(bool)
        elif isinstance(op, (ast.In, ast.NotIn)):
            self._visit_binop_internal(
                rhs_node,
                Composite(rhs),
                op,
                lhs_node,
                Composite(lhs),
                parent_node,
                allow_call=False,
            )
            # These always return a bool, regardless of what the dunder does.
            val = TypedValue(bool)
        else:
            val = self._visit_binop_internal(
                lhs_node,
                Composite(lhs),
                op,
                rhs_node,
                Composite(rhs),
                parent_node,
                allow_call=False,
            )

        if definite_value is not None:
            val = annotate_value(val, [DefiniteValueExtension(definite_value)])
        return annotate_with_constraint(val, constraint)

    def _constraint_from_compare_op(
        self, constrained_node: ast.AST, other_val: Any, op: ast.AST, *, is_right: bool
    ) -> AbstractConstraint:
        varname = self.composite_from_node(constrained_node).varname
        if varname is None:
            return NULL_CONSTRAINT
        if isinstance(op, (ast.Is, ast.IsNot)):
            predicate = EqualsPredicate(other_val, self, use_is=True)
            positive = isinstance(op, ast.Is)
            return Constraint(varname, ConstraintType.predicate, positive, predicate)
        elif isinstance(op, (ast.Eq, ast.NotEq)):
            predicate = EqualsPredicate(other_val, self)
            positive = isinstance(op, ast.Eq)
            return Constraint(varname, ConstraintType.predicate, positive, predicate)
        elif isinstance(op, (ast.In, ast.NotIn)) and is_right:
            try:
                predicate_vals = list(other_val)
                predicate_types = {type(val) for val in predicate_vals}
                if len(predicate_types) == 1:
                    pattern_type = next(iter(predicate_types))
                else:
                    pattern_type = object
            except Exception:
                return NULL_CONSTRAINT
            predicate = InPredicate(other_val, pattern_type, self)
            positive = isinstance(op, ast.In)
            return Constraint(varname, ConstraintType.predicate, positive, predicate)
        else:
            positive_operator, negative_operator, ext = COMPARATOR_TO_OPERATOR[type(op)]

            def predicate_func(value: Value, positive: bool) -> Value | None:
                op = positive_operator if positive else negative_operator
                if isinstance(value, KnownValue):
                    try:
                        if is_right:
                            result = op(value.val, other_val)
                        else:
                            result = op(other_val, value.val)
                    except Exception:
                        pass
                    else:
                        if not result:
                            return None
                    return value
                if ext is not None and positive:
                    return annotate_value(value, [CustomCheckExtension(ext(other_val))])
                return value

            return Constraint(varname, ConstraintType.predicate, True, predicate_func)

    def _constraint_from_predicate_provider(
        self, pred: PredicateProvider, other_val: Any, op: ast.AST
    ) -> Constraint:
        positive_operator, negative_operator, _ = COMPARATOR_TO_OPERATOR[type(op)]

        def predicate_func(value: Value, positive: bool) -> Value | None:
            predicate_value = pred.provider(value)
            if isinstance(predicate_value, KnownValue):
                operator = positive_operator if positive else negative_operator
                try:
                    result = operator(predicate_value.val, other_val)
                except Exception:
                    pass
                else:
                    if not result:
                        return None
            if pred.value_transformer is not None:
                op_cls = type(op)
                if not positive:
                    if op_cls in AST_TO_REVERSE:
                        op_cls = AST_TO_REVERSE[op_cls]
                    else:
                        return value
                return pred.value_transformer(value, op_cls, other_val, self)
            return value

        return Constraint(pred.varname, ConstraintType.predicate, True, predicate_func)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Value:
        if isinstance(node.op, ast.Not):
            # not doesn't have its own special method
            val, constraint = self.constraint_from_condition(node.operand)
            definite_value = _extract_definite_value(val)
            boolability = get_boolability(val)
            if boolability.is_safely_true():
                val = KnownValue(False)
            elif boolability.is_safely_false():
                val = KnownValue(True)
            else:
                val = TypedValue(bool)
            if definite_value is not None:
                val = annotate_value(val, [DefiniteValueExtension(not definite_value)])
            return annotate_with_constraint(val, constraint.invert())
        else:
            operand = self.composite_from_node(node.operand)
            _, method = UNARY_OPERATION_TO_DESCRIPTION_AND_METHOD[type(node.op)]
            val, _ = self._check_dunder_call(node, operand, method, [], allow_call=True)
            return val

    def visit_BinOp(self, node: ast.BinOp) -> Value:
        left = self.composite_from_node(node.left)
        right = self.composite_from_node(node.right)
        return self._visit_binop_internal(
            node.left, left, node.op, node.right, right, node
        )

    def _visit_binop_internal(
        self,
        left_node: ast.expr,
        left_composite: Composite,
        op: ast.AST,
        right_node: ast.expr,
        right_composite: Composite,
        source_node: ast.AST,
        is_inplace: bool = False,
        allow_call: bool = True,
    ) -> Value:
        left = left_composite.value
        right = right_composite.value
        if self.in_annotation and isinstance(op, ast.BitOr):
            # Accept PEP 604 (int | None) in annotations
            self.check_for_missing_generic_params(left_node, left)
            self.check_for_missing_generic_params(right_node, right)
            if isinstance(left, KnownValue) and isinstance(right, KnownValue):
                return KnownValue(Union[left.val, right.val])  # noqa: UP007
            # In static fallback mode class objects may be synthetic and cannot
            # be OR'd at runtime. Preserve both sides for annotation evaluation.
            return unite_values(left, right)

        if (
            isinstance(op, ast.Mod)
            and isinstance(left, KnownValue)
            and isinstance(left.val, (bytes, str))
        ):
            value, replacement_node = format_strings.check_string_format(
                left_node,
                left.val,
                right_node,
                right,
                self._show_error_if_checking,
                self,
            )
            if replacement_node is not None and isinstance(source_node, ast.BinOp):
                replacement = self.replace_node(source_node, replacement_node)
                self._show_error_if_checking(
                    source_node,
                    error_code=ErrorCode.use_fstrings,
                    replacement=replacement,
                )
            return value

        _, method, imethod, _ = BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD[type(op)]
        allow_call = allow_call and method not in self.options.get_value_for(
            DisallowCallsToDunders
        )

        if is_inplace:
            assert imethod is not None, f"no inplace method available for {op}"
            with self.catch_errors() as inplace_errors:
                # Not _check_dunder_call_or_catch because if the call doesn't
                # typecheck it normally returns NotImplemented and we try the
                # non-inplace method next.
                inplace_result, _ = self._check_dunder_call(
                    source_node,
                    left_composite,
                    imethod,
                    [right_composite],
                    allow_call=allow_call,
                )
            if not inplace_errors:
                return inplace_result

        possibilities = []
        for subval in flatten_values(left):
            result = self._visit_binop_no_mvv(
                Composite(subval, left_composite.varname, left_composite.node),
                op,
                right_composite,
                source_node,
                allow_call,
            )
            possibilities.append(result)
        return unite_values(*possibilities)

    def _visit_binop_no_mvv(
        self,
        left_composite: Composite,
        op: ast.AST,
        right_composite: Composite,
        source_node: ast.AST,
        allow_call: bool = True,
    ) -> Value:
        left = left_composite.value
        right = right_composite.value
        description, method, _, rmethod = BINARY_OPERATION_TO_DESCRIPTION_AND_METHOD[
            type(op)
        ]
        if rmethod is None:
            # "in" falls back to __iter__ and then to __getitem__ if __contains__ is not defined
            if method == "__contains__":
                contains_result_or_errors = self._check_dunder_call_or_catch(
                    source_node,
                    left_composite,
                    method,
                    [right_composite],
                    allow_call=allow_call,
                )
                if isinstance(contains_result_or_errors, Value):
                    return contains_result_or_errors

                iterable_type = is_iterable(left, self)
                if isinstance(iterable_type, Value):
                    can_assign = has_relation(
                        iterable_type, right, Relation.ASSIGNABLE, self
                    )
                    if isinstance(can_assign, CanAssignError):
                        self._show_error_if_checking(
                            source_node,
                            "Unsupported operand for 'in'",
                            ErrorCode.incompatible_argument,
                            detail=str(can_assign),
                        )
                        return TypedValue(bool)
                    else:
                        return TypedValue(bool)

                getitem_result = self._check_dunder_call_or_catch(
                    source_node,
                    left_composite,
                    "__getitem__",
                    [right_composite],
                    allow_call=allow_call,
                )
                if isinstance(getitem_result, Value):
                    return TypedValue(bool)  # Always returns a bool
                self.show_caught_errors(contains_result_or_errors)
                return TypedValue(bool)

            result, _ = self._check_dunder_call(
                source_node,
                left_composite,
                method,
                [right_composite],
                allow_call=allow_call,
            )
            return result

        with self.catch_errors() as left_errors:
            left_result, _ = self._check_dunder_call(
                source_node,
                left_composite,
                method,
                [right_composite],
                allow_call=allow_call,
            )

        with self.catch_errors() as right_errors:
            right_result, _ = self._check_dunder_call(
                source_node,
                right_composite,
                rmethod,
                [left_composite],
                allow_call=allow_call,
            )
        if left_errors:
            if right_errors:
                self.show_error(
                    source_node,
                    f"Unsupported operands for {description}: {left} and {right}",
                    error_code=ErrorCode.unsupported_operation,
                )
                return AnyValue(AnySource.error)
            return right_result
        else:
            if right_errors:
                return left_result
            # The interesting case: neither threw an error. Naively we might
            # want to return the left result, but that fails in a case like
            # this:
            #     df: Any
            #     1 + df
            # because this would return "int" (which is what int.__add__ returns),
            # and "df" might be an object that implements __radd__.
            # Instead, we return Any if right is Any.
            if isinstance(right, AnyValue):
                return AnyValue(AnySource.from_another)
            return left_result

    # Indexing

    def visit_Slice(self, node: ast.Slice) -> Value:
        lower = self.visit(node.lower) if node.lower is not None else None
        upper = self.visit(node.upper) if node.upper is not None else None
        step = self.visit(node.step) if node.step is not None else None

        if all(
            val is None or isinstance(val, KnownValue) for val in (lower, upper, step)
        ):
            return KnownValue(
                slice(
                    lower.val if isinstance(lower, KnownValue) else None,
                    upper.val if isinstance(upper, KnownValue) else None,
                    step.val if isinstance(step, KnownValue) else None,
                )
            )
        else:
            return TypedValue(slice)

    # These two are unused in 3.9 and higher

    # Control flow

    def visit_Await(self, node: ast.Await) -> Value:
        composite = self.composite_from_node(node.value)
        return_value = self.unpack_awaitable(composite, node.value)
        if return_value is NO_RETURN_VALUE:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        return return_value

    def unpack_awaitable(self, composite: Composite, node: ast.AST) -> Value:
        tv_map = get_tv_map(AwaitableValue, composite.value, self)
        if isinstance(tv_map, CanAssignError):
            result, _ = self._check_dunder_call(node, composite, "__await__", [])
            return result
        else:
            return tv_map.get(T, AnyValue(AnySource.generic_argument))

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Value:
        self.is_generator = True
        value = self.visit(node.value)
        tv_map = get_tv_map(GeneratorValue, value, self)
        if isinstance(tv_map, CanAssignError):
            can_assign = get_tv_map(AwaitableValue, value, self)
            if not isinstance(can_assign, CanAssignError):
                tv_map = {
                    ReturnT: can_assign.get(T, AnyValue(AnySource.generic_argument))
                }
            else:
                iterable_type = is_iterable(value, self)
                if isinstance(iterable_type, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        f"Cannot use {value} in yield from",
                        error_code=ErrorCode.bad_yield_from,
                        detail=can_assign.display(),
                    )
                    tv_map = {ReturnT: AnyValue(AnySource.error)}
                else:
                    tv_map = {YieldT: iterable_type}

        if self.current_function_info is not None:
            expected_yield = self.current_function_info.get_generator_yield_type(self)
            yield_type = tv_map.get(YieldT, AnyValue(AnySource.generic_argument))
            can_assign = has_relation(
                expected_yield, yield_type, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    f"Cannot yield from {value} (expected {expected_yield})",
                    error_code=ErrorCode.incompatible_yield,
                    detail=can_assign.display(),
                )

            expected_send = self.current_function_info.get_generator_send_type(self)
            send_type = tv_map.get(SendT, AnyValue(AnySource.generic_argument))
            can_assign = has_relation(
                send_type, expected_send, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    f"Cannot send {send_type} to a generator (expected"
                    f" {expected_send})",
                    error_code=ErrorCode.incompatible_yield,
                    detail=can_assign.display(),
                )

        return tv_map.get(ReturnT, AnyValue(AnySource.generic_argument))

    def visit_Yield(self, node: ast.Yield) -> Value:
        if self._is_checking():
            if self.in_comprehension_body:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.yield_in_comprehension
                )

            ctx = self.yield_checker.check_yield(node, self.current_statement)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            if node.value is not None:
                value = self.visit(node.value)
            else:
                value = KnownValue(None)

        if node.value is None and self.async_kind in (
            AsyncFunctionKind.normal,
            AsyncFunctionKind.pure,
        ):
            self._show_error_if_checking(node, error_code=ErrorCode.yield_without_value)
        self.is_generator = True

        # unwrap the results of async yields
        if self.async_kind != AsyncFunctionKind.non_async:
            return self._unwrap_yield_result(node, value)
        if self.current_function_info is None:
            return AnyValue(AnySource.inference)
        yield_type = self.current_function_info.get_generator_yield_type(self)
        can_assign = has_relation(yield_type, value, Relation.ASSIGNABLE, self)
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node,
                f"Cannot assign value of type {value} to yield expression of type"
                f" {yield_type}",
                error_code=ErrorCode.incompatible_yield,
                detail=can_assign.display(),
            )
        return self.current_function_info.get_generator_send_type(self)

    def _unwrap_yield_result(self, node: ast.AST, value: Value) -> Value:
        assert asynq is not None
        value = replace_fallback(value)
        if isinstance(value, AsyncTaskIncompleteValue):
            return value.value
        elif isinstance(value, TypedValue) and (
            # asynq only supports exactly list and tuple, not subclasses
            # https://github.com/quora/asynq/blob/b07682d8b11e53e4ee5c585020cc9033e239c7eb/asynq/async_task.py#L446
            value.get_type_object().is_exactly({list, tuple})
        ):
            if isinstance(value, SequenceValue) and isinstance(value.typ, type):
                values = [
                    (is_many, self._unwrap_yield_result(node, member))
                    for is_many, member in value.members
                ]
                return SequenceValue.make_or_known(value.typ, values)
            elif isinstance(value, GenericValue):
                member_value = self._unwrap_yield_result(node, value.get_arg(0))
                return GenericValue(value.typ, [member_value])
            else:
                return TypedValue(value.typ)
        elif isinstance(value, TypedValue) and value.get_type_object().is_exactly(
            {dict}
        ):
            if isinstance(value, DictIncompleteValue):
                pairs = [
                    KVPair(
                        pair.key,
                        self._unwrap_yield_result(node, pair.value),
                        pair.is_many,
                        pair.is_required,
                    )
                    for pair in value.kv_pairs
                ]
                return DictIncompleteValue(value.typ, pairs)
            elif isinstance(value, GenericValue):
                val = self._unwrap_yield_result(node, value.get_arg(1))
                return GenericValue(value.typ, [value.get_arg(0), val])
            else:
                return TypedValue(dict)
        elif isinstance(value, KnownValue) and isinstance(value.val, asynq.ConstFuture):
            return KnownValue(value.val.value())
        elif isinstance(value, KnownValue) and value.val is None:
            return value  # we're allowed to yield None
        elif isinstance(value, KnownValue) and isinstance(value.val, (list, tuple)):
            values = [
                self._unwrap_yield_result(node, KnownValue(elt)) for elt in value.val
            ]
            return KnownValue(values)
        elif isinstance(value, AnyValue):
            return AnyValue(AnySource.from_another)
        elif isinstance(value, MultiValuedValue):
            return unite_values(
                *[self._unwrap_yield_result(node, val) for val in value.vals]
            )
        elif _is_asynq_future(value):
            return AnyValue(AnySource.inference)
        else:
            self._show_error_if_checking(
                node,
                f"Invalid value yielded: {value}",
                error_code=ErrorCode.bad_async_yield,
            )
            return AnyValue(AnySource.error)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is None:
            value = KnownNone
        else:
            value = self.visit(node.value)
        self.return_values.append(value)
        self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        if (
            self.expected_return_value is NO_RETURN_VALUE
            and value is not NO_RETURN_VALUE
        ):
            self._show_error_if_checking(
                node, error_code=ErrorCode.no_return_may_return
            )
        elif self.is_generator and self.async_kind == AsyncFunctionKind.non_async:
            if self.current_function_info is not None:
                expected = self.current_function_info.get_generator_return_type(self)
                can_assign = has_relation(expected, value, Relation.ASSIGNABLE, self)
                if isinstance(can_assign, CanAssignError):
                    self._show_error_if_checking(
                        node,
                        f"Incompatible return type: expected {expected}, got {value}",
                        error_code=ErrorCode.incompatible_return_value,
                        detail=can_assign.display(),
                    )
        elif self.expected_return_value is not None:
            can_assign = has_relation(
                self.expected_return_value, value, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    f"Incompatible return type: expected {self.expected_return_value}, got {value}",
                    error_code=ErrorCode.incompatible_return_value,
                    detail=can_assign.display(),
                )
        if (
            self.expected_return_value == KnownNone
            and value != KnownNone
            and value is not NO_RETURN_VALUE
        ):
            self._show_error_if_checking(
                node,
                "Function declared as returning None may not return a value",
                error_code=ErrorCode.incompatible_return_value,
            )

    def visit_Raise(self, node: ast.Raise) -> None:
        # we need to record this in the return value so that functions that always raise
        # NotImplementedError aren't inferred as returning None
        self.return_values.append(None)
        self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

        raised_expr = node.exc

        if raised_expr is not None:
            raised_value = self.visit(raised_expr)
            can_assign = has_relation(
                ExceptionValue, raised_value, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node, error_code=ErrorCode.bad_exception, detail=str(can_assign)
                )

        if node.cause is not None:
            cause_value = self.visit(node.cause)
            can_assign = has_relation(
                ExceptionOrNone, cause_value, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    "Invalid object in raise from",
                    error_code=ErrorCode.bad_exception,
                    detail=str(can_assign),
                )

    def visit_Assert(self, node: ast.Assert) -> None:
        test = self._visit_possible_constraint(node.test)
        constraint = extract_constraints(test)
        if node.msg is not None:
            with self.scopes.subscope():
                self.add_constraint(node, constraint.invert())
                self.visit(node.msg)
        self.add_constraint(node, constraint)
        # code after an assert False is unreachable
        boolability = get_boolability(test)
        if boolability is Boolability.value_always_false:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))
        # We don't check value_always_true here; it's fine to have an assertion
        # that the type checker statically thinks is True.
        self._check_boolability(test, node, disabled={ErrorCode.value_always_true})

    def add_constraint(self, node: object, constraint: AbstractConstraint) -> None:
        if constraint is NULL_CONSTRAINT:
            return  # save some work
        self.scopes.current_scope().add_constraint(constraint, node, self.state)

    def _visit_possible_constraint(self, node: ast.AST) -> Value:
        if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
            composite = self.composite_from_node(node)
            if composite.varname is not None:
                constraint = Constraint(
                    composite.varname, ConstraintType.is_truthy, True, None
                )
                existing = extract_constraints(composite.value)
                new_value, _ = unannotate_value(composite.value, ConstraintExtension)
                return annotate_with_constraint(
                    new_value, EquivalentConstraint.make([constraint, existing])
                )
            else:
                return composite.value
        else:
            return self.visit(node)

    def visit_Break(self, node: ast.Break) -> None:
        self._set_name_in_scope(LEAVES_LOOP, node, AnyValue(AnySource.marker))

    def visit_Continue(self, node: ast.Continue) -> None:
        self._set_name_in_scope(LEAVES_LOOP, node, AnyValue(AnySource.marker))

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        iterated_value = self._member_value_of_iterator(
            node.iter, is_async=isinstance(node, ast.AsyncFor)
        )
        if self.options.get_value_for(ForLoopAlwaysEntered):
            always_entered = True
        elif isinstance(iterated_value, Value):
            iterated_value, present = unannotate_value(
                iterated_value, AlwaysPresentExtension
            )
            always_entered = bool(present)
        else:
            always_entered = len(iterated_value) > 0
        if not isinstance(iterated_value, Value):
            iterated_value = unite_and_simplify(
                *iterated_value,
                limit=self.options.get_value_for(UnionSimplificationLimit),
            )
        with self.scopes.subscope() as body_scope:
            with self.scopes.loop_scope():
                with override(self, "being_assigned", iterated_value):
                    # assume that node.target is not affected by variable assignments in the body
                    # one could write some contortion like
                    # for (a if a[0] == 1 else b)[0] in range(2):
                    #   b = [1]
                    # but that doesn't seem worth supporting
                    self.visit(node.target)
                self._generic_visit_list(node.body)
        self._handle_loop_else(node.orelse, body_scope, always_entered)

        # in loops, variables may have their first read before their first write
        # see e.g. test_stacked_scopes.TestLoop.test_conditional_in_loop
        # to get all the definition nodes in that case, visit the body twice in the collecting
        # phase
        if self.state == VisitorState.collect_names:
            with self.scopes.subscope():
                with override(self, "being_assigned", iterated_value):
                    self.visit(node.target)
                self._generic_visit_list(node.body)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        # see comments under For for discussion

        # We don't check boolability here because "while True" is legitimate and common.
        test, constraint = self.constraint_from_condition(
            node.test, check_boolability=False
        )
        always_entered = get_boolability(test) in (
            Boolability.value_always_true,
            Boolability.type_always_true,
        )
        with self.scopes.subscope() as body_scope:
            with self.scopes.loop_scope() as loop_scopes:
                # The "node" argument need not be an AST node but must be unique.
                self.add_constraint((node, 1), constraint)
                self._generic_visit_list(node.body)
        self._handle_loop_else(node.orelse, body_scope, always_entered)

        if self.state == VisitorState.collect_names:
            test, constraint = self.constraint_from_condition(
                node.test, check_boolability=False
            )
            with self.scopes.subscope():
                self.add_constraint((node, 2), constraint)
                self._generic_visit_list(node.body)

        if always_entered and all(LEAVES_LOOP not in scope for scope in loop_scopes):
            # This means the code following the loop is unreachable.
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

    def _handle_loop_else(
        self, orelse: list[ast.stmt], body_scope: SubScope, always_entered: bool
    ) -> None:
        if always_entered:
            self.scopes.combine_subscopes([body_scope])
            # Replace body_scope with a dummy scope, because body_scope
            # should always execute and has already been combined in.
            with self.scopes.subscope() as body_scope:
                pass
        with self.scopes.subscope() as else_scope:
            self._generic_visit_list(orelse)
        self.scopes.combine_subscopes([body_scope, else_scope])

    def _member_value_of_iterator(
        self, node: ast.AST, is_async: bool = False
    ) -> Value | Sequence[Value]:
        """Analyze an iterator AST node.

        Returns a tuple of two values:
        - A Value object representing a member of the iterator.
        - The number of elements in the iterator, or None if the number is unknown.

        """
        composite = self.composite_from_node(node)
        if is_async:
            value = is_async_iterable(composite.value, self)
            if isinstance(value, CanAssignError):
                self._show_error_if_checking(
                    node,
                    f"{composite.value} is not async iterable",
                    ErrorCode.unsupported_operation,
                    detail=str(value),
                )
                return AnyValue(AnySource.error)
            return value
        iterated = composite.value
        result = concrete_values_from_iterable(iterated, self)
        if isinstance(result, CanAssignError):
            self._show_error_if_checking(
                node,
                f"{iterated} is not iterable",
                ErrorCode.unsupported_operation,
                detail=str(result),
            )
            return AnyValue(AnySource.error)
        return result

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) == 1:
            with self.scopes.subscope():
                context = self.visit(node.items[0].context_expr)
            if isinstance(context, AnnotatedValue) and context.has_metadata_of_type(
                AssertErrorExtension
            ):
                self._visit_assert_errors_block(node)
                return

        self.visit_single_cm(node.items, node.body, is_async=False)

    def _visit_assert_errors_block(self, node: ast.With) -> None:
        with self.catch_errors() as caught:
            self._generic_visit_list(node.body)
        if not caught:
            self._show_error_if_checking(
                node,
                "No errors found in assert_error() block",
                error_code=ErrorCode.inference_failure,
            )

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_single_cm(node.items, node.body, is_async=True)

    def visit_single_cm(
        self,
        items: list[ast.withitem],
        body: Iterable[ast.AST],
        *,
        is_async: bool = False,
    ) -> None:
        if len(items) == 0:
            self._generic_visit_list(body)
            return
        first_item = items[0]
        can_suppress = self.visit_withitem(first_item, is_async)
        if can_suppress:
            with self.scopes.suppressing_subscope():
                self.visit_single_cm(items[1:], body, is_async=is_async)
        else:
            self.visit_single_cm(items[1:], body, is_async=is_async)

    def visit_withitem(self, node: ast.withitem, is_async: bool = False) -> bool:
        context = self.visit(node.context_expr)
        if is_async:
            protocol = AsyncCustomContextManager
        else:
            protocol = CustomContextManager
        val = GenericValue(protocol, [TypeVarValue(T_co), TypeVarValue(U_co)])
        can_assign = get_tv_map(val, context, self)
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node.context_expr,
                f"{context} is not a context manager",
                detail=str(can_assign),
                error_code=ErrorCode.invalid_context_manager,
            )
            assigned = AnyValue(AnySource.error)
            can_suppress = False
        else:
            assigned = can_assign.get(T_co, AnyValue(AnySource.generic_argument))
            exit_assigned = can_assign.get(U_co, AnyValue(AnySource.generic_argument))
            exit_boolability = get_boolability(exit_assigned)
            exit_is_bool_subtype = not isinstance(
                has_relation(
                    TypedValue(bool), exit_assigned, Relation.ASSIGNABLE, self
                ),
                CanAssignError,
            )
            can_suppress = (
                exit_is_bool_subtype and not exit_boolability.is_safely_false()
            )
            if isinstance(exit_assigned, AnyValue) or (
                isinstance(context, TypedValue) and is_context_manager_type(context.typ)
            ):
                # cannot easily infer what the context manager will do,
                # assume it does not suppress exceptions.
                can_suppress = False
        if node.optional_vars is not None:
            with override(self, "being_assigned", assigned):
                self.visit(node.optional_vars)
        return can_suppress

    def visit_try_except(self, node: TryNode, *, is_try_star: bool = False) -> None:
        with self.scopes.subscope():
            with self.scopes.subscope() as dummy_scope:
                pass

            with self.scopes.subscope() as failure_scope:
                with self.scopes.suppressing_subscope() as success_scope:
                    self._generic_visit_list(node.body)

            with self.scopes.subscope() as else_scope:
                self.yield_checker.reset_yield_checks()
                self.scopes.combine_subscopes([success_scope])
                self._generic_visit_list(node.orelse)

            except_scopes = []
            for handler in node.handlers:
                with self.scopes.subscope() as except_scope:
                    except_scopes.append(except_scope)
                    # reset yield checks between branches to avoid incorrect errors when we yield
                    # both in the try and the except block
                    self.yield_checker.reset_yield_checks()
                    # With except*, multiple except* blocks may run, so we need
                    # to combine not just the failure scope, but also the previous
                    # except_scopes.
                    if is_try_star:
                        subscopes = [dummy_scope, failure_scope, *except_scopes]
                    else:
                        subscopes = [dummy_scope, failure_scope]
                    self.scopes.combine_subscopes(subscopes)
                    self.visit(handler)

        self.scopes.combine_subscopes([else_scope, *except_scopes])

    def visit_Try(self, node: TryNode, *, is_try_star: bool = False) -> None:
        if node.finalbody:
            with self.scopes.subscope() as failure_scope:
                with self.scopes.suppressing_subscope() as success_scope:
                    self.visit_try_except(node, is_try_star=is_try_star)

            # If the try block fails
            with self.scopes.subscope():
                self.scopes.combine_subscopes([failure_scope])
                self._generic_visit_list(node.finalbody)

            # For the case where execution continues after the try-finally
            self.scopes.combine_subscopes([success_scope])
            self._generic_visit_list(node.finalbody)
        else:
            # Life is much simpler without finally
            self.visit_try_except(node, is_try_star=is_try_star)
        self.yield_checker.reset_yield_checks()

    if sys.version_info >= (3, 11):

        def visit_TryStar(self, node: ast.TryStar) -> None:
            self.visit_Try(node, is_try_star=True)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            typ = self.visit(node.type)
            is_try_star = not isinstance(self.node_context.contexts[-2], ast.Try)
            possible_types = self._extract_exception_types(
                typ, node, is_try_star=is_try_star
            )
            if node.name is not None:
                to_assign = unite_values(*[typ for _, typ in possible_types])
                if is_try_star and sys.version_info >= (3, 11):
                    if all(is_exception for is_exception, _ in possible_types):
                        base = ExceptionGroup
                    else:
                        base = BaseExceptionGroup
                    to_assign = GenericValue(base, [to_assign])
                self._set_name_in_scope(node.name, node, value=to_assign, private=True)

        self._generic_visit_list(node.body)

    def _extract_exception_types(
        self, typ: Value, node: ast.AST, is_try_star: bool = False
    ) -> list[tuple[bool, Value]]:
        possible_types = []
        for subval in flatten_values(typ, unwrap_annotated=True):
            subval = replace_known_sequence_value(subval)
            if isinstance(subval, SequenceValue) and subval.typ is tuple:
                for _, elt in subval.members:
                    possible_types += self._extract_exception_types(
                        elt, node, is_try_star=is_try_star
                    )
                continue
            elif isinstance(subval, GenericValue) and subval.typ is tuple:
                possible_types += self._extract_exception_types(
                    subval.args[0], node, is_try_star=is_try_star
                )
                continue
            elif (
                isinstance(subval, SubclassValue)
                and isinstance(subval.typ, TypedValue)
                and isinstance(subval.typ.typ, type)
            ):
                subval = KnownValue(subval.typ.typ)
            if isinstance(subval, KnownValue):
                if isinstance(subval.val, type) and issubclass(
                    subval.val, BaseException
                ):
                    if (
                        is_try_star
                        and sys.version_info >= (3, 11)
                        and issubclass(subval.val, BaseExceptionGroup)
                    ):
                        self._show_error_if_checking(
                            node,
                            "ExceptionGroup cannot be used as the type in an"
                            f" except* clause: {subval.val!r}",
                            error_code=ErrorCode.bad_except_handler,
                        )
                    is_exception = issubclass(subval.val, Exception)
                    possible_types.append((is_exception, TypedValue(subval.val)))
                else:
                    self._show_error_if_checking(
                        node,
                        f"{subval!r} is not an exception class",
                        error_code=ErrorCode.bad_except_handler,
                    )
                    possible_types.append((False, TypedValue(BaseException)))
            else:
                # TODO consider raising an error for except classes
                # that cannot be statically resolved.
                possible_types.append((False, TypedValue(BaseException)))
        return possible_types

    def visit_If(self, node: ast.If) -> None:
        val, constraint = self.constraint_from_condition(node.test)
        definite_value = _extract_definite_value(val)
        # reset yield checks to avoid incorrect errors when we yield in both the condition and one
        # of the blocks
        self.yield_checker.reset_yield_checks()
        with self._subscope_and_maybe_supress(definite_value is False) as body_scope:
            self.add_constraint(node, constraint)
            self._generic_visit_list(node.body)
        self.yield_checker.reset_yield_checks()

        with self._subscope_and_maybe_supress(definite_value is True) as else_scope:
            self.add_constraint(node, constraint.invert())
            self._generic_visit_list(node.orelse)
        self.scopes.combine_subscopes([body_scope, else_scope])
        self.yield_checker.reset_yield_checks()

    def visit_IfExp(self, node: ast.IfExp) -> Value:
        val, constraint = self.constraint_from_condition(node.test)
        definite_value = _extract_definite_value(val)
        with self._subscope_and_maybe_supress(definite_value is False) as if_scope:
            self.add_constraint(node, constraint)
            then_val = self.visit(node.body)
        with self._subscope_and_maybe_supress(definite_value is True) as else_scope:
            self.add_constraint(node, constraint.invert())
            else_val = self.visit(node.orelse)
        self.scopes.combine_subscopes([if_scope, else_scope])
        return unite_values(then_val, else_val)

    @contextlib.contextmanager
    def _subscope_and_maybe_supress(self, should_suppress: bool) -> Generator[SubScope]:
        with self.scopes.subscope() as scope:
            if should_suppress:
                with self.catch_errors():
                    yield scope
            else:
                yield scope

    def constraint_from_condition(
        self, node: ast.AST, check_boolability: bool = True
    ) -> tuple[Value, AbstractConstraint]:
        condition = self._visit_possible_constraint(node)
        constraint = extract_constraints(condition)
        if self._is_collecting():
            return condition, constraint
        if check_boolability:
            disabled = set()
        else:
            disabled = {ErrorCode.type_always_true, ErrorCode.value_always_true}
        self._check_boolability(condition, node, disabled=disabled)
        return condition, constraint

    def _check_boolability(
        self, value: Value, node: ast.AST, *, disabled: Container[Error] = frozenset()
    ) -> None:
        boolability = get_boolability(value)
        if boolability is Boolability.erroring_bool:
            if ErrorCode.type_does_not_support_bool not in disabled:
                self.show_error(
                    node,
                    f"{value} does not support bool()",
                    error_code=ErrorCode.type_does_not_support_bool,
                )
        elif boolability is Boolability.type_always_true:
            if ErrorCode.type_always_true not in disabled:
                self._show_error_if_checking(
                    node,
                    f"{value} is always True because it does not provide __bool__",
                    error_code=ErrorCode.type_always_true,
                )
        elif boolability in (
            Boolability.value_always_true,
            Boolability.value_always_true_mutable,
        ):
            if ErrorCode.value_always_true not in disabled:
                self.show_error(
                    node,
                    f"{value} is always True",
                    error_code=ErrorCode.value_always_true,
                )

    def visit_Expr(self, node: ast.Expr) -> Value:
        value = self.visit(node.value)
        if _is_asynq_future(value):
            new_node = ast.Expr(value=ast.Yield(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.task_needs_yield, replacement=replacement
            )
        # If the value is an awaitable or is assignable to asyncio.Future, show
        # an error about a missing await.
        elif value.is_type(collections.abc.Awaitable) or value.is_type(asyncio.Future):
            if self.is_async_def:
                new_node = ast.Expr(value=ast.Await(value=node.value))
            else:
                new_node = ast.Expr(value=ast.YieldFrom(value=node.value))
            replacement = self.replace_node(node, new_node)
            self._show_error_if_checking(
                node, error_code=ErrorCode.missing_await, replacement=replacement
            )
        elif value.is_type(collections.abc.Generator) or value.is_type(
            collections.abc.AsyncGenerator
        ):
            self._show_error_if_checking(
                node,
                f"Must use {value} (for example, by iterating over it)",
                error_code=ErrorCode.must_use,
            )
        return value

    # Assignments
    def visit_NamedExpr(self, node: ast.NamedExpr) -> Value:
        composite = self.composite_from_walrus(node)
        return composite.value

    def composite_from_walrus(self, node: ast.NamedExpr) -> Composite:
        rhs = self.visit(node.value)
        with override(self, "being_assigned", rhs):
            if self.in_comprehension_body:
                ctx = self.scopes.ignore_topmost_scope()
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                return self.composite_from_node(node.target)

    def visit_Assign(self, node: ast.Assign) -> None:
        is_yield = isinstance(node.value, ast.Yield)
        runtime_type_alias_value = self._make_runtime_type_alias_assignment_value(node)
        if runtime_type_alias_value is not None:
            if self.annotate:
                with self.catch_errors():
                    self.visit(node.value)
            value = runtime_type_alias_value
        elif (
            self.current_enum_members is not None
            and self.current_function_name is None
            and isinstance(node.value, ast.Name)
            and node.value.id in self.current_enum_members.by_name
        ):
            value = KnownValue(self.current_enum_members.by_name[node.value.id])
        else:
            value = self.visit(node.value)

        with (
            override(self, "being_assigned", value),
            self.yield_checker.check_yield_result_assignment(is_yield),
        ):
            # syntax like 'x = y = 0' results in multiple targets
            self._generic_visit_list(node.targets)

        if (
            self.current_enum_members is not None
            and self.current_function_name is None
            and isinstance(value, KnownValue)
            and is_hashable(value.val)
        ):
            names = [
                target.id for target in node.targets if isinstance(target, ast.Name)
            ]
            is_alias = isinstance(node.value, ast.Name) and (
                node.value.id in self.current_enum_members.by_name
            )
            if value.val in self.current_enum_members.by_value and not is_alias:
                self._show_error_if_checking(
                    node,
                    "Duplicate enum member: {} is used for both {} and {}".format(
                        value.val,
                        self.current_enum_members.by_value[value.val],
                        ", ".join(names),
                    ),
                    error_code=ErrorCode.duplicate_enum_member,
                )
            else:
                for name in names:
                    self.current_enum_members.by_name[name] = value.val
                    self.current_enum_members.by_value.setdefault(value.val, name)

        enum_value_type = (
            self.enum_value_type_by_class.get(self.current_class_key)
            if self.current_class_key is not None
            else None
        )
        if (
            enum_value_type is not None
            and self.current_function_name is not None
            and isinstance(value, Value)
        ):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_value_"
                    and self._is_current_method_receiver_node(target.value)
                ):
                    self._check_declared_enum_value_type(enum_value_type, value, node)

    def is_in_typeddict_definition(self) -> bool:
        return (
            is_instance_of_typing_name(self.current_class, "_TypedDictMeta")
            or self.current_synthetic_typeddict is not None
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        explicit_type_alias_assignment_value: TypeAliasValue | None = None
        if self.current_synthetic_typeddict is None:
            annotation = self._visit_annotation(node.annotation)
            if (
                self._is_current_class_dataclass()
                and _is_dataclass_kw_only_marker_value(annotation)
            ):
                if node.value is not None:
                    self._show_error_if_checking(
                        node,
                        "dataclasses.KW_ONLY marker cannot have a default value",
                        error_code=ErrorCode.invalid_annotation,
                    )
                if self.annotate:
                    node.target.inferred_value = AnyValue(AnySource.inference)
                return
            if self._is_subscripted_type_alias_annotation(node.annotation):
                expr = annotation_expr_from_ast(
                    node.annotation, visitor=self, suppress_errors=self._is_collecting()
                )
            else:
                expr = self._expr_of_annotation_type(annotation, node.annotation)
        else:
            # Still visit the annotation node so ast_annotator can attach
            # inferred values to all annotation expressions.
            self._visit_annotation(node.annotation)
            expr = annotation_expr_from_ast(
                node.annotation, visitor=self, suppress_errors=self._is_collecting()
            )
        if self.is_in_typeddict_definition():
            expected_type, qualifiers = expr.unqualify(
                {Qualifier.Required, Qualifier.NotRequired, Qualifier.ReadOnly},
                mutually_exclusive_qualifiers=(
                    (Qualifier.Required, Qualifier.NotRequired),
                ),
            )
        else:
            # TODO: validate these qualifiers more
            qualifiers = {
                Qualifier.Final,
                Qualifier.ClassVar,
                Qualifier.TypeAlias,
                Qualifier.InitVar,
            }
            expected_type, qualifiers = expr.maybe_unqualify(qualifiers)
        if Qualifier.TypeAlias in qualifiers and node.value is not None:
            alias_expr = annotation_expr_from_ast(
                node.value, self, suppress_errors=self._is_collecting()
            )
            alias_type, alias_qualifiers = alias_expr.maybe_unqualify(set(Qualifier))
            if Qualifier.ClassVar in alias_qualifiers:
                self._show_error_if_checking(
                    node.value,
                    "ClassVar cannot be used in type aliases",
                    error_code=ErrorCode.invalid_annotation,
                )
            if isinstance(alias_type, InputSigValue):
                if isinstance(alias_type.input_sig, ParamSpecSig):
                    message = "ParamSpec cannot be used in this annotation context"
                else:
                    message = f"Unrecognized annotation {alias_type}"
                self._show_error_if_checking(
                    node.value, message, error_code=ErrorCode.invalid_annotation
                )
            elif alias_type is not None and has_invalid_paramspec_usage(
                alias_type, self
            ):
                self._show_error_if_checking(
                    node.value,
                    "ParamSpec cannot be used in this annotation context",
                    error_code=ErrorCode.invalid_annotation,
                )
            if isinstance(node.target, ast.Name) and alias_type is not None:
                type_params = self._infer_type_alias_type_params(node.value, alias_type)
                explicit_type_alias_assignment_value = TypeAliasValue(
                    node.target.id,
                    self.module.__name__ if self.module is not None else "",
                    TypeAlias(
                        lambda value_node=node.value: annotation_expr_from_ast(
                            value_node, visitor=self, suppress_errors=True
                        ).to_value(allow_qualifiers=True, allow_empty=True),
                        lambda type_params=tuple(type_params): type_params,
                    ),
                    runtime_allows_value_call=True,
                )
            # `TypeAlias` marks this assignment as an alias declaration, not a
            # variable declaration of the marker type itself.
            expected_type = None
        if self.current_synthetic_typeddict is not None and isinstance(
            node.target, ast.Name
        ):
            self._record_synthetic_typeddict_item(
                node.target.id, expected_type, qualifiers, node
            )
        if (
            isinstance(node.target, ast.Name)
            and node.target.id == "_value_"
            and self.current_class_key is not None
            and self._is_enum_class_key(self.current_class_key)
            and expected_type is not None
        ):
            self.enum_value_type_by_class[self.current_class_key] = expected_type

        if isinstance(expected_type, InputSigValue):
            if isinstance(expected_type.input_sig, ParamSpecSig):
                self._show_error_if_checking(
                    node.annotation,
                    "ParamSpec cannot be used in this annotation context",
                    error_code=ErrorCode.invalid_annotation,
                )
            else:
                self._show_error_if_checking(
                    node.annotation,
                    f"Unrecognized annotation {expected_type}",
                    error_code=ErrorCode.invalid_annotation,
                )
            expected_type = AnyValue(AnySource.error)
        elif expected_type is not None and has_invalid_paramspec_usage(
            expected_type, self
        ):
            self._show_error_if_checking(
                node.annotation,
                "ParamSpec cannot be used in this annotation context",
                error_code=ErrorCode.invalid_annotation,
            )
            expected_type = AnyValue(AnySource.error)

        # TODO: handle TypeAlias and ClassVar
        is_final = Qualifier.Final in qualifiers
        has_classvar = Qualifier.ClassVar in qualifiers
        if has_classvar and self.scopes.scope_type() != ScopeType.class_scope:
            self._show_error_if_checking(
                node.annotation,
                "ClassVar can only be used for assignments in class body",
                error_code=ErrorCode.invalid_annotation,
            )
        if (
            has_classvar
            and expected_type is not None
            and self._contains_classvar_type_parameter(expected_type)
        ):
            self._show_error_if_checking(
                node.annotation,
                "ClassVar type cannot include type parameters",
                error_code=ErrorCode.classvar_type_parameters,
            )
        if (
            has_classvar
            and self.scopes.scope_type() == ScopeType.class_scope
            and isinstance(node.target, ast.Name)
        ):
            self._record_synthetic_classvar_name(node.target.id)
        is_current_class_dataclass = self._is_current_class_dataclass()
        has_default = node.value is not None
        init = True
        kw_only = (
            self.current_dataclass_info is not None
            and self.current_dataclass_info.kw_only_default
        )
        alias: str | None = None
        is_dataclass_field_call = False
        dataclass_default_factory: Value | None = None
        dataclass_field_name: str | None = None
        if (
            is_current_class_dataclass
            and self.scopes.scope_type() == ScopeType.class_scope
            and isinstance(node.target, ast.Name)
            and not has_classvar
        ):
            dataclass_field_name = node.target.id
        if (
            has_classvar
            and is_final
            and not (is_current_class_dataclass and _is_dataclass_classvar_final(expr))
        ):
            self._show_error_if_checking(
                node.annotation,
                "Final cannot be combined with ClassVar",
                error_code=ErrorCode.invalid_annotation,
            )
        if is_final and node.value is None and expected_type is None:
            self._show_error_if_checking(
                node.annotation,
                "Final annotation without assignment requires an explicit type",
                error_code=ErrorCode.invalid_annotation,
            )

        if is_final and self.scopes.scope_type() == ScopeType.class_scope:
            class_key = self.current_class_key
            if class_key is not None and isinstance(node.target, ast.Name):
                self._record_final_member(class_key, node.target.id)
                if node.value is None and (
                    has_classvar or not is_current_class_dataclass
                ):
                    self.final_members_requiring_init.setdefault(class_key, {})[
                        node.target.id
                    ] = node

        if is_final and isinstance(node.target, ast.Attribute):
            target_root_value = self.composite_from_node(node.target.value).value
            if not self._is_allowed_instance_final_annotation_target(
                node.target, target_root_value
            ):
                self._show_error_if_checking(
                    node.annotation,
                    "Final instance attributes may be declared only in __init__",
                    error_code=ErrorCode.invalid_annotation,
                )
            elif self.current_class_key is not None:
                self._record_final_member(self.current_class_key, node.target.attr)

        is_class_annotation_without_value = (
            node.value is None
            and self.current_synthetic_typeddict is None
            and isinstance(node.target, ast.Name)
            and self.scopes.scope_type() == ScopeType.class_scope
        )
        if is_class_annotation_without_value:
            self._set_synthetic_class_attribute(
                node.target.id, expected_type or AnyValue(AnySource.error)
            )

        if node.value is not None:
            if explicit_type_alias_assignment_value is not None:
                if self.annotate:
                    with self.catch_errors():
                        self.visit(node.value)
                is_yield = False
                alias_runtime_value = explicit_type_alias_assignment_value.get_value()
                if isinstance(alias_runtime_value, TypedValue) and isinstance(
                    alias_runtime_value.typ, type
                ):
                    value = KnownValue(alias_runtime_value.typ)
                else:
                    value = explicit_type_alias_assignment_value
            else:
                is_yield = isinstance(node.value, ast.Yield)
                value = self.visit(node.value)
                if (
                    dataclass_field_name is not None
                    and isinstance(node.value, ast.Call)
                    and (
                        inferred_options := self._dataclass_field_call_options_by_node.pop(
                            id(node.value), None
                        )
                    )
                    is not None
                ):
                    is_dataclass_field_call = True
                    dataclass_default_factory = inferred_options.default_factory
                    has_default = inferred_options.has_default or any(
                        kw.arg in {"default", "default_factory", "factory"}
                        for kw in node.value.keywords
                    )
                    if inferred_options.init is not None:
                        init = inferred_options.init
                    if inferred_options.kw_only is not None:
                        kw_only = inferred_options.kw_only
                    if inferred_options.alias is not None:
                        alias = inferred_options.alias
                    init_keyword = next(
                        (kw.value for kw in node.value.keywords if kw.arg == "init"),
                        None,
                    )
                    if isinstance(init_keyword, ast.Constant) and isinstance(
                        init_keyword.value, bool
                    ):
                        init = init_keyword.value
                    kw_only_keyword = next(
                        (kw.value for kw in node.value.keywords if kw.arg == "kw_only"),
                        None,
                    )
                    if isinstance(kw_only_keyword, ast.Constant) and isinstance(
                        kw_only_keyword.value, bool
                    ):
                        kw_only = kw_only_keyword.value
                    alias_keyword = next(
                        (kw.value for kw in node.value.keywords if kw.arg == "alias"),
                        None,
                    )
                    if isinstance(alias_keyword, ast.Constant) and isinstance(
                        alias_keyword.value, str
                    ):
                        alias = alias_keyword.value

                if expected_type is not None:
                    if not (is_current_class_dataclass and is_dataclass_field_call):
                        can_assign = has_relation(
                            expected_type, value, Relation.ASSIGNABLE, self
                        )
                        if isinstance(can_assign, CanAssignError):
                            self._show_error_if_checking(
                                node,
                                f"Incompatible assignment: expected {expected_type}, got"
                                f" {value}",
                                error_code=ErrorCode.incompatible_assignment,
                                detail=can_assign.display(),
                            )
                    # We set the declared type on initial assignment, so that the
                    # annotation can be used to adjust pycroscope's type inference.
                    value = expected_type

                    if (
                        is_current_class_dataclass
                        and is_dataclass_field_call
                        and dataclass_default_factory is not None
                    ):
                        default_factory_sig = self.signature_from_value(
                            dataclass_default_factory
                        )
                        default_factory_return = _callable_return_type_from_signature(
                            default_factory_sig, checker=self
                        )
                        if default_factory_return is not None:
                            can_assign_return = has_relation(
                                expected_type,
                                default_factory_return,
                                Relation.ASSIGNABLE,
                                self,
                            )
                            if isinstance(can_assign_return, CanAssignError):
                                self._show_error_if_checking(
                                    node,
                                    "Dataclass default_factory return type is incompatible"
                                    f" with field type {expected_type}",
                                    error_code=ErrorCode.incompatible_assignment,
                                    detail=can_assign_return.display(),
                                )

        else:
            is_yield = False
            value = None

        if dataclass_field_name is not None:
            self._record_synthetic_dataclass_field_metadata(
                dataclass_field_name,
                has_default=has_default,
                init=init,
                initvar=Qualifier.InitVar in qualifiers,
                kw_only=kw_only,
                alias=alias,
            )

        ann_assign_declared_type = expected_type
        if explicit_type_alias_assignment_value is not None:
            ann_assign_declared_type = explicit_type_alias_assignment_value

        with (
            override(self, "being_assigned", value),
            self.yield_checker.check_yield_result_assignment(is_yield),
            override(self, "ann_assign_type", (ann_assign_declared_type, is_final)),
        ):
            self.visit(node.target)

        if is_class_annotation_without_value and not has_classvar:
            assert isinstance(node.target, ast.Name)
            self._record_synthetic_instance_only_annotation_name(node.target.id)

    def _record_synthetic_typeddict_item(
        self,
        key: str,
        expected_type: Value | None,
        qualifiers: Container[Qualifier],
        node: ast.AST,
    ) -> None:
        context = self.current_synthetic_typeddict
        if context is None:
            return
        required = context.total
        readonly = Qualifier.ReadOnly in qualifiers
        if Qualifier.Required in qualifiers:
            required = True
        if Qualifier.NotRequired in qualifiers:
            required = False
        context.local_items[key] = (
            TypedDictEntry(
                typ=expected_type or AnyValue(AnySource.error),
                required=required,
                readonly=readonly,
            ),
            node,
        )

    def _is_subscripted_type_alias_annotation(self, node: ast.expr) -> bool:
        if not isinstance(node, ast.Subscript):
            return False
        if not isinstance(node.value, ast.Name):
            return False
        resolved, _ = self.resolve_name(
            node.value, error_node=node.value, suppress_errors=True
        )
        for subval in flatten_values(resolved, unwrap_annotated=True):
            if isinstance(subval, TypeAliasValue):
                return True
            if isinstance(subval, KnownValue) and (
                is_typing_name(subval.val, "TypeAliasType")
                or is_instance_of_typing_name(subval.val, "TypeAliasType")
            ):
                return True
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        is_yield = isinstance(node.value, ast.Yield)
        rhs = self.composite_from_node(node.value)

        if isinstance(node.target, ast.Name):
            lhs = self.composite_from_name(node.target, force_read=True)
        else:
            lhs = Composite(AnyValue(AnySource.inference), None, node.target)

        value = self._visit_binop_internal(
            node.target, lhs, node.op, node.value, rhs, node, is_inplace=True
        )

        with (
            override(self, "being_assigned", value),
            self.yield_checker.check_yield_result_assignment(is_yield),
        ):
            # syntax like 'x = y = 0' results in multiple targets
            self.visit(node.target)

    def visit_type_param_values(
        self, type_params: Sequence[ast.AST]
    ) -> Sequence[TypeVarValue]:
        type_param_values = []
        for param in type_params:
            value = self.visit(param)
            if isinstance(value, TypeVarValue):
                type_param_values.append(value)
            elif isinstance(value, InputSigValue) and isinstance(
                value.input_sig, ParamSpecSig
            ):
                type_param_values.append(TypeVarValue(value.input_sig.param_spec))
            else:
                assert False, f"unexpected type parameter value: {value!r}"
        assert all_of_type(type_param_values, TypeVarValue)
        return type_param_values

    def _current_scope_key(self) -> int:
        return id(self.scopes.current_scope())

    @contextlib.contextmanager
    def _active_pep695_type_param_scope(
        self, type_params: Sequence[TypeVarValue]
    ) -> Generator[None]:
        if not type_params:
            yield
            return
        type_param_identities = {param.typevar for param in type_params}
        self._active_pep695_type_params.append(type_param_identities)
        try:
            yield
        finally:
            self._active_pep695_type_params.pop()

    def _record_type_alias_structure(
        self, name: str, alias_node: ast.AST, value_node: ast.AST
    ) -> None:
        scope_key = self._current_scope_key()
        first_by_name = self._type_alias_first_definition_by_scope.setdefault(
            scope_key, {}
        )
        first_by_name.setdefault(name, alias_node)
        refs_by_name = self._type_alias_unguarded_refs_by_scope.setdefault(
            scope_key, {}
        )
        refs_by_name.setdefault(name, set()).update(
            _collect_unguarded_type_alias_refs(value_node)
        )

    def _type_alias_has_unguarded_cycle(self, name: str) -> bool:
        scope_refs = self._type_alias_unguarded_refs_by_scope.get(
            self._current_scope_key(), {}
        )
        if name not in scope_refs:
            return False
        visited: set[str] = set()

        def reaches_target(current: str) -> bool:
            if current == name:
                return True
            if current in visited:
                return False
            visited.add(current)
            for dep in scope_refs.get(current, ()):
                if dep in scope_refs and reaches_target(dep):
                    return True
            return False

        return any(
            dep in scope_refs and reaches_target(dep)
            for dep in scope_refs.get(name, ())
        )

    def _legacy_typevars_in_nodes(
        self,
        nodes: Iterable[ast.AST],
        declared_type_params: Sequence[TypeVarValue],
        *,
        include_active_type_params: bool = True,
    ) -> set[str]:
        declared = {param.typevar for param in declared_type_params}
        if include_active_type_params:
            declared.update(chain.from_iterable(self._active_pep695_type_params))
        legacy: set[str] = set()
        for node in nodes:
            for subnode in ast.walk(node):
                if not isinstance(subnode, ast.Name) or not isinstance(
                    subnode.ctx, ast.Load
                ):
                    continue
                resolved, _ = self.resolve_name(
                    subnode, error_node=subnode, suppress_errors=True
                )
                for subval in flatten_values(resolved, unwrap_annotated=True):
                    identity = _type_param_identity(subval)
                    if identity is None:
                        continue
                    if identity not in declared:
                        legacy.add(subnode.id)
                    break
        return legacy

    def _legacy_typevars_in_alias_expr(
        self, value_node: ast.AST, declared_type_params: Sequence[TypeVarValue]
    ) -> set[str]:
        return self._legacy_typevars_in_nodes(
            [value_node], declared_type_params, include_active_type_params=False
        )

    def _infer_type_alias_type_params(
        self, value_node: ast.AST, value: Value
    ) -> tuple[TypeVarLike, ...]:
        type_params: list[TypeVarLike] = []
        seen_identities: set[object] = set()

        def maybe_record_type_param(resolved: Value) -> None:
            for subval in flatten_values(resolved, unwrap_annotated=True):
                identity = _type_param_identity(subval)
                if identity is None or identity in seen_identities:
                    continue
                seen_identities.add(identity)
                if isinstance(subval, TypeVarValue):
                    type_params.append(subval.typevar)
                elif isinstance(subval, InputSigValue) and isinstance(
                    subval.input_sig, ParamSpecSig
                ):
                    type_params.append(subval.input_sig.param_spec)
                elif isinstance(subval, KnownValue) and (
                    is_instance_of_typing_name(subval.val, "TypeVar")
                    or is_instance_of_typing_name(subval.val, "TypeVarTuple")
                    or is_instance_of_typing_name(subval.val, "ParamSpec")
                ):
                    type_params.append(cast(TypeVarLike, subval.val))
                break

        def walk(node: ast.AST, *, allow_string_parse: bool = True) -> None:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                resolved, _ = self.resolve_name(
                    node, error_node=node, suppress_errors=True
                )
                maybe_record_type_param(resolved)
                return
            if (
                allow_string_parse
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                parse_source = node.value
                if "\n" in parse_source or "\r" in parse_source:
                    parse_source = f"({parse_source})"
                try:
                    parsed = ast.parse(parse_source, mode="eval")
                except SyntaxError:
                    return
                walk(parsed.body, allow_string_parse=False)
                return
            for child in ast.iter_child_nodes(node):
                walk(child, allow_string_parse=allow_string_parse)

        walk(value_node)
        for type_param in extract_type_params(value):
            if type_param in seen_identities:
                continue
            seen_identities.add(type_param)
            type_params.append(type_param)
        return tuple(type_params)

    def _resolve_call_target_value(self, node: ast.expr) -> Value:
        if isinstance(node, ast.Name):
            resolved, _ = self.resolve_name(node, error_node=node, suppress_errors=True)
            return resolved
        if isinstance(node, ast.Attribute):
            root_value = self._resolve_call_target_value(node.value)
            if root_value is UNINITIALIZED_VALUE or any(
                subval is UNINITIALIZED_VALUE
                for subval in flatten_values(root_value, unwrap_annotated=True)
            ):
                return AnyValue(AnySource.inference)
            if isinstance(root_value, AnyValue) and root_value == AnyValue(
                AnySource.error
            ):
                return root_value
            resolved = self.get_attribute(Composite(root_value), node.attr, None)
            if resolved is UNINITIALIZED_VALUE:
                return AnyValue(AnySource.inference)
            return resolved
        return AnyValue(AnySource.inference)

    def _is_typealiastype_call(self, node: ast.Call) -> bool:
        return _is_typealiastype_value(self._resolve_call_target_value(node.func))

    def _extract_runtime_type_alias_type_params(
        self, node: ast.Call
    ) -> Sequence[TypeVarLike | TypeVarValue]:
        type_params_keyword = next(
            (keyword for keyword in node.keywords if keyword.arg == "type_params"), None
        )
        if type_params_keyword is None:
            return []
        if not isinstance(type_params_keyword.value, ast.Tuple):
            self._show_error_if_checking(
                type_params_keyword.value,
                "type_params argument to TypeAliasType must be a literal tuple",
                error_code=ErrorCode.invalid_annotation,
            )
            return []
        params: list[TypeVarLike | TypeVarValue] = []
        for elt in type_params_keyword.value.elts:
            value = self.visit(elt)
            if isinstance(value, TypeVarValue):
                params.append(value)
                continue
            if isinstance(value, InputSigValue) and isinstance(
                value.input_sig, ParamSpecSig
            ):
                params.append(value.input_sig.param_spec)
                continue
            identity = _type_param_identity(value)
            if identity is not None:
                params.append(cast(TypeVarLike, identity))
                continue
            self._show_error_if_checking(
                elt,
                "TypeAliasType type_params must contain only type parameters",
                error_code=ErrorCode.invalid_annotation,
            )
        return params

    def _make_runtime_type_alias_assignment_value(
        self, node: ast.Assign
    ) -> Value | None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return None
        if not isinstance(node.value, ast.Call) or not self._is_typealiastype_call(
            node.value
        ):
            return None
        value_node = _get_runtime_type_alias_value_node(node.value)
        if value_node is None:
            return None
        name = node.targets[0].id
        if self._is_collecting():
            self._record_type_alias_structure(name, node, value_node)
        type_params = self._extract_runtime_type_alias_type_params(node.value)
        declared_type_params = list(
            _runtime_type_alias_declared_type_params(type_params)
        )
        if self.current_class_type_params is not None:
            declared_type_params.extend(self.current_class_type_params)
        declared_type_params = list(dict.fromkeys(declared_type_params))
        has_circular_definition = False
        if _runtime_type_alias_self_reference(value_node, name):
            has_circular_definition = True
        if self._type_alias_has_unguarded_cycle(name):
            has_circular_definition = True
        if self._is_checking():
            if has_circular_definition:
                self._show_error_if_checking(
                    node,
                    f"Type alias {name} has a circular definition",
                    error_code=ErrorCode.invalid_annotation,
                )
            legacy_typevars = self._legacy_typevars_in_nodes(
                [value_node], declared_type_params
            )
            if legacy_typevars:
                if type_params:
                    message = (
                        "Type alias cannot combine old-style TypeVar declarations"
                        " with type statement parameters"
                    )
                else:
                    message = (
                        "Type alias must declare type parameters in the"
                        " type statement"
                    )
                self._show_error_if_checking(
                    node, message, error_code=ErrorCode.invalid_annotation
                )
        if has_circular_definition:
            evaluator = lambda: AnyValue(AnySource.error)
        else:
            evaluator = lambda: annotation_expr_from_ast(
                value_node, visitor=self, suppress_errors=True
            ).to_value(allow_qualifiers=True, allow_empty=True)
        return TypeAliasValue(
            name,
            self.module.__name__ if self.module is not None else "",
            TypeAlias(evaluator, lambda: tuple(type_params)),
        )

    if sys.version_info >= (3, 12):

        def _pep695_type_param_expr_needs_string_forward_ref(
            self, node: ast.expr
        ) -> bool:
            root: ast.expr = node
            while isinstance(root, ast.Subscript):
                root = root.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                return False
            resolved, _ = self.resolve_name(root, error_node=root, suppress_errors=True)
            return isinstance(resolved, AnyValue) and resolved.source is AnySource.error

        def _type_from_pep695_type_param_expr(self, node: ast.expr) -> Value:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                return type_from_value(
                    KnownValue(node.value), self, node, suppress_errors=True
                )
            if self._pep695_type_param_expr_needs_string_forward_ref(node):
                return type_from_value(
                    KnownValue(ast.unparse(node)), self, node, suppress_errors=True
                )
            return type_from_value(self.visit(node), self, node)

        def visit_TypeAlias(self, node: ast.TypeAlias) -> Value:
            assert isinstance(node.name, ast.Name)
            name = node.name.id
            if self._is_collecting():
                self._record_type_alias_structure(name, node, node.value)
            alias_val = self._get_local_object(name, node)
            if isinstance(alias_val, KnownValue) and isinstance(
                alias_val.val, typing.TypeAliasType
            ):
                alias_obj = alias_val.val
            else:
                alias_obj = None
            type_param_values = []
            disallow_in_function = self.scopes.scope_type() == ScopeType.function_scope
            if self._is_checking():
                if disallow_in_function:
                    self._show_error_if_checking(
                        node,
                        "Type alias statements are not allowed inside functions",
                        error_code=ErrorCode.invalid_annotation,
                    )
                first_by_name = self._type_alias_first_definition_by_scope.get(
                    self._current_scope_key(), {}
                )
                first_definition = first_by_name.get(name)
                if first_definition is not None and first_definition is not node:
                    self._show_error_if_checking(
                        node,
                        f"Type alias {name} is already defined",
                        error_code=ErrorCode.invalid_annotation,
                    )
                if self._type_alias_has_unguarded_cycle(name):
                    self._show_error_if_checking(
                        node,
                        f"Type alias {name} has a circular definition",
                        error_code=ErrorCode.invalid_annotation,
                    )
                with self.scopes.add_scope(
                    ScopeType.annotation_scope, scope_node=node, scope_object=alias_obj
                ):
                    if node.type_params:
                        type_param_values = self.visit_type_param_values(
                            node.type_params
                        )
                        with self.scopes.add_scope(
                            ScopeType.annotation_scope,
                            scope_node=node,
                            scope_object=alias_obj,
                        ):
                            value = self.visit(node.value)
                    else:
                        value = self.visit(node.value)
                legacy_typevars = self._legacy_typevars_in_alias_expr(
                    node.value, type_param_values
                )
                if legacy_typevars:
                    if node.type_params:
                        message = (
                            "Type alias cannot combine old-style TypeVar declarations"
                            " with type statement parameters"
                        )
                    else:
                        message = (
                            "Type alias must declare type parameters in the"
                            " type statement"
                        )
                    self._show_error_if_checking(
                        node, message, error_code=ErrorCode.invalid_annotation
                    )
            else:
                value = None
            if alias_obj is None:
                if value is None or disallow_in_function:
                    alias_val = AnyValue(AnySource.inference)
                else:
                    alias_val = TypeAliasValue(
                        name,
                        self.module.__name__ if self.module is not None else "",
                        TypeAlias(
                            lambda: type_from_value(value, self, node),
                            lambda: tuple(type_param_values),
                        ),
                    )
            set_value, _ = self._set_name_in_scope(name, node, alias_val)
            return set_value

        def visit_TypeVar(self, node: ast.TypeVar) -> Value:
            bound = constraints = default = None
            if node.bound is not None:
                if isinstance(node.bound, ast.Tuple):
                    constraints = [
                        self._type_from_pep695_type_param_expr(elt)
                        for elt in node.bound.elts
                    ]
                    if len(constraints) < 2:
                        self._show_error_if_checking(
                            node.bound,
                            "TypeVar constraints must contain at least two types",
                            error_code=ErrorCode.invalid_annotation,
                        )
                    for elt, constraint in zip(node.bound.elts, constraints):
                        message = self._typevar_invalid_bound_message(
                            constraint, is_constraint=True
                        )
                        if message is not None:
                            self._show_error_if_checking(
                                elt, message, error_code=ErrorCode.invalid_annotation
                            )
                else:
                    bound = self._type_from_pep695_type_param_expr(node.bound)
                    message = self._typevar_invalid_bound_message(bound)
                    if message is not None:
                        self._show_error_if_checking(
                            node.bound, message, error_code=ErrorCode.invalid_annotation
                        )
            if sys.version_info >= (3, 13):
                if node.default_value is not None:
                    default = self._type_from_pep695_type_param_expr(node.default_value)
            tv = TypeVar(node.name)
            typevar = TypeVarValue(
                tv,
                bound=bound,
                constraints=tuple(constraints) if constraints is not None else (),
                default=default if default is not None else None,
            )
            self._check_typevar_default_constraints(typevar, node)
            self._set_name_in_scope(node.name, node, typevar)
            return typevar

        def visit_ParamSpec(self, node: ast.ParamSpec) -> Value:
            ps = typing.ParamSpec(node.name)
            typevar = InputSigValue(ParamSpecSig(ps))
            self._set_name_in_scope(node.name, node, typevar)
            return typevar

        def visit_TypeVarTuple(self, node: ast.TypeVarTuple) -> Value:
            tv = TypeVar(node.name)
            typevar = TypeVarValue(tv, is_typevartuple=True)
            self._set_name_in_scope(node.name, node, typevar)
            return typevar

    def visit_Name(self, node: ast.Name, force_read: bool = False) -> Value:
        return self.composite_from_name(node, force_read=force_read).value

    def composite_from_name(
        self, node: ast.Name, force_read: bool = False
    ) -> Composite:
        if force_read or self._is_read_ctx(node.ctx):
            self.yield_checker.record_usage(node.id, node)
            value, origin = self.resolve_name(node)
            varname_value = self.checker.maybe_get_variable_name_value(node.id)
            if varname_value is not None and self._should_use_varname_value(value):
                value = varname_value
            self.check_deprecation(node, value)
            return Composite(value, VarnameWithOrigin(node.id, origin), node)
        elif self._is_write_ctx(node.ctx):
            if self._name_node_to_statement is not None:
                statement = self.node_context.nearest_enclosing(
                    (ast.stmt, ast.comprehension)
                )
                self._name_node_to_statement[node] = statement

            value = self.being_assigned
            if (
                value is None
                and self.scopes.current_scope().scope_type == ScopeType.class_scope
                and self.ann_assign_type is not None
            ):
                ann_assign_type, _ = self.ann_assign_type
                if ann_assign_type is not None:
                    value = ann_assign_type
            if value is not None:
                self.yield_checker.record_assignment(node.id)
            value, origin = self._set_name_in_scope(node.id, node, value=value)
            varname = VarnameWithOrigin(node.id, origin)
            constraint = Constraint(varname, ConstraintType.is_truthy, True, None)
            value = annotate_with_constraint(value, constraint)
            return Composite(value, varname, node)
        else:
            # not sure when (if ever) the other contexts can happen
            self.show_error(node, f"Bad context: {node.ctx}", ErrorCode.unexpected_node)
            return Composite(AnyValue(AnySource.error), None, node)

    def visit_Starred(self, node: ast.Starred) -> Value:
        val = self.visit(node.value)
        return PartialValue(
            PartialValueOperation.UNPACK, val, node, (), AnyValue(AnySource.inference)
        )

    def visit_arg(self, node: ast.arg) -> None:
        self.yield_checker.record_assignment(node.arg)
        # it's none only for AnnAssign nodes without a value
        assert self.being_assigned is not None, "should not happen"
        self._set_name_in_scope(node.arg, node, value=self.being_assigned)

    def _should_use_varname_value(self, value: Value) -> bool:
        """Returns whether a value should be replaced with VariableNameValue.

        VariableNameValues are used for things like uids that are represented as integers, but
        in places where we don't necessarily have precise annotations. Therefore, we replace
        only AnyValue.

        """
        return (
            isinstance(value, AnyValue) and value.source is not AnySource.variable_name
        )

    def visit_Subscript(self, node: ast.Subscript) -> Value:
        return self.composite_from_subscript(node).value

    def composite_from_subscript(self, node: ast.Subscript) -> Composite:
        if self.annotate and not hasattr(node, "inferred_value"):
            node.inferred_value = AnyValue(AnySource.inference)
        root_composite = self.composite_from_node(node.value)
        index_composite = self.composite_from_node(node.slice)
        index = index_composite.value
        self.check_for_missing_generic_params(node.slice, index)
        if (
            root_composite.varname is not None
            and isinstance(index, KnownValue)
            and is_hashable(index.val)
        ):
            varname = self._extend_composite(root_composite, index, node)
        else:
            varname = None
        if isinstance(root_composite.value, MultiValuedValue):
            values = [
                self._composite_from_subscript_no_mvv(
                    node,
                    Composite(val, root_composite.varname, root_composite.node),
                    index_composite,
                    varname,
                )
                for val in root_composite.value.vals
            ]
            return_value = unite_values(*values)
        else:
            return_value = self._composite_from_subscript_no_mvv(
                node, root_composite, index_composite, varname
            )
        composite = Composite(return_value, varname, node)
        if self.annotate:
            node.inferred_value = composite.value
        return composite

    def _composite_from_subscript_no_mvv(
        self,
        node: ast.Subscript,
        root_composite: Composite,
        index_composite: Composite,
        composite_var: VarnameWithOrigin | None,
    ) -> Value:
        value = root_composite.value
        index = index_composite.value

        if isinstance(node.ctx, ast.Store):
            if self.ann_assign_type is not None:
                self._show_error_if_checking(
                    node, error_code=ErrorCode.invalid_annotated_assignment
                )
            if self.being_assigned is None:
                assert (
                    self.ann_assign_type is not None
                ), "should only happen in AnnAssign"
                return AnyValue(AnySource.inference)
            if (
                composite_var is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                self.scopes.set(
                    composite_var.get_varname(), self.being_assigned, node, self.state
                )
            self._check_dunder_call(
                node.value,
                root_composite,
                "__setitem__",
                [index_composite, Composite(self.being_assigned, None, node)],
            )
            return self.being_assigned
        elif isinstance(node.ctx, ast.Load):
            if value == KnownValue(type):
                # "type[int]" is legal, but neither
                # type.__getitem__ nor type.__class_getitem__ exists at runtime. Support
                # it directly instead.
                if isinstance(index, KnownValue):
                    return_value = KnownValue(type[index.val])
                else:
                    return_value = PartialValue(
                        PartialValueOperation.SUBSCRIPT,
                        value,
                        node,
                        (index,),
                        TypedValue(types.GenericAlias),
                    )
            else:
                stripped_value = _strip_predicate_intersection(value)
                stripped_root = (
                    root_composite
                    if stripped_value is value
                    else Composite(
                        stripped_value, root_composite.varname, root_composite.node
                    )
                )
                if self.in_annotation and (
                    isinstance(stripped_root.value, TypeAliasValue)
                    or (
                        self.module is None
                        and _should_use_static_annotation_subscript_on_import_failure(
                            stripped_root.value
                        )
                    )
                    or (
                        isinstance(stripped_root.value, KnownValue)
                        and (
                            is_typing_name(stripped_root.value.val, "TypeAliasType")
                            or is_instance_of_typing_name(
                                stripped_root.value.val, "TypeAliasType"
                            )
                        )
                    )
                    or (
                        isinstance(stripped_root.value, KnownValue)
                        and is_typing_name(stripped_root.value.val, "Literal")
                        and not _is_runtime_literal_index(index)
                    )
                    or _contains_unpack_annotation_value(index)
                    or _should_use_static_annotation_subscript(stripped_root.value)
                ):
                    return_value = PartialValue(
                        PartialValueOperation.SUBSCRIPT,
                        stripped_root.value,
                        node,
                        self._maybe_unpack_tuple(index, node),
                        TypedValue(types.GenericAlias),
                    )
                else:
                    synthetic_subscript = self._maybe_subscript_synthetic_class(
                        stripped_root.value, index, node
                    )
                    if synthetic_subscript is not None:
                        return_value = synthetic_subscript
                    else:
                        with self.catch_errors():
                            getitem = self._get_dunder(
                                node.value, stripped_root.value, "__getitem__"
                            )
                        if getitem is not UNINITIALIZED_VALUE:
                            return_value = self.check_call(
                                node.value,
                                getitem,
                                [stripped_root, index_composite],
                                allow_call=True,
                            )
                        else:
                            # If there was no __getitem__, try __class_getitem__ in 3.7+
                            cgi = self.get_attribute(
                                Composite(stripped_root.value),
                                "__class_getitem__",
                                node.value,
                            )
                            if cgi is UNINITIALIZED_VALUE:
                                self._show_error_if_checking(
                                    node,
                                    f"Object {value} does not support subscripting",
                                    error_code=ErrorCode.unsupported_operation,
                                )
                                return_value = AnyValue(AnySource.error)
                            else:
                                runtime_return_value = self.check_call(
                                    node.value, cgi, [index_composite], allow_call=True
                                )
                                if isinstance(runtime_return_value, KnownValue):
                                    return_value = runtime_return_value
                                else:
                                    return_value = PartialValue(
                                        PartialValueOperation.SUBSCRIPT,
                                        value,
                                        node,
                                        self._maybe_unpack_tuple(index),
                                        runtime_return_value,
                                    )

                if (
                    self._should_use_varname_value(return_value)
                    and isinstance(index, KnownValue)
                    and isinstance(index.val, str)
                ):
                    varname_value = self.checker.maybe_get_variable_name_value(
                        index.val
                    )
                    if varname_value is not None:
                        return_value = varname_value

            if (
                composite_var is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                local_value = self._get_composite(
                    composite_var.get_varname(), node, return_value
                )
                if local_value is not UNINITIALIZED_VALUE:
                    return_value = local_value
            return return_value
        elif isinstance(node.ctx, ast.Del):
            result, _ = self._check_dunder_call(
                node.value, root_composite, "__delitem__", [index_composite]
            )
            return result
        else:
            self.show_error(
                node,
                f"Unexpected subscript context: {node.ctx}",
                ErrorCode.unexpected_node,
            )
            return AnyValue(AnySource.error)

    def _maybe_unpack_tuple(
        self, value: Value, node: ast.AST | None = None
    ) -> tuple[Value, ...]:
        if isinstance(value, SequenceValue) and value.typ is tuple:
            members = value.get_member_sequence()
            if members is not None:
                return tuple(members)
            if node is not None and self.in_annotation:
                unpacked_members = []
                for is_many, member in value.members:
                    if is_many:
                        unpacked_members.append(
                            PartialValue(
                                PartialValueOperation.UNPACK,
                                member,
                                node,
                                (),
                                AnyValue(AnySource.inference),
                            )
                        )
                    else:
                        unpacked_members.append(member)
                return tuple(unpacked_members)
            return (AnyValue(AnySource.inference),)
        elif isinstance(value, KnownValue) and isinstance(value.val, tuple):
            return tuple(KnownValue(member) for member in value.val)
        return (value,)

    def _maybe_subscript_synthetic_class(
        self, value: Value, index: Value, node: ast.Subscript
    ) -> Value | None:
        if isinstance(value, GenericValue) and value.typ is not type:
            # Generic instances should use __getitem__; this path is for class
            # specialization (C[T]).
            return None

        def _normalize_member(member: Value) -> Value:
            normalized = type_from_value(member, self, node, suppress_errors=True)
            if (
                isinstance(normalized, AnyValue)
                and normalized.source is AnySource.error
            ):
                sequence_normalized = replace_known_sequence_value(member)
                if sequence_normalized != member:
                    return sequence_normalized
                if isinstance(member, (TypedValue, GenericValue)) and isinstance(
                    member.typ, (type, str)
                ):
                    return member
            return normalized

        synthetic_typ: type | str
        root_for_partial: Value
        if isinstance(value, KnownValue) and isinstance(value.val, type):
            synthetic_class = self.checker.get_synthetic_class(value.val)
            if (
                synthetic_class is None
                or not isinstance(synthetic_class.class_type, TypedValue)
                or not isinstance(synthetic_class.class_type.typ, str)
            ):
                return None
            synthetic_typ = synthetic_class.class_type.typ
            root_for_partial = TypedValue(synthetic_typ)
        elif isinstance(value, SyntheticClassObjectValue):
            class_type = value.class_type
            if not isinstance(class_type, TypedValue) or not isinstance(
                class_type.typ, str
            ):
                return None
            synthetic_typ = class_type.typ
            root_for_partial = value
        elif isinstance(value, TypedValue) and isinstance(value.typ, str):
            synthetic_typ = value.typ
            root_for_partial = value
        else:
            return None

        members = self._maybe_unpack_tuple(index)
        normalized_members = tuple(_normalize_member(member) for member in members)
        if self.checker.get_synthetic_class(synthetic_typ) is None:
            return None
        type_parameters = self.checker.get_type_parameters(synthetic_typ)
        variadic_type_param_indexes = [
            i
            for i, type_param in enumerate(type_parameters)
            if isinstance(type_param, TypeVarValue) and type_param.is_typevartuple
        ]
        if len(variadic_type_param_indexes) > 1:
            return None
        expected_len: int | None
        if variadic_type_param_indexes:
            # A TypeVarTuple can absorb any number of type arguments.
            expected_len = len(type_parameters) - 1
            is_valid_type_arg_count = len(normalized_members) >= expected_len
        else:
            expected_len = len(type_parameters)
            is_valid_type_arg_count = expected_len == len(normalized_members)

        if type_parameters and not is_valid_type_arg_count:
            if variadic_type_param_indexes:
                expected_type_arg_message = (
                    f"Expected at least {expected_len} type arguments for "
                    f"{stringify_object(synthetic_typ)}"
                )
            else:
                expected_type_arg_message = (
                    f"Expected {expected_len} type arguments for "
                    f"{stringify_object(synthetic_typ)}"
                )
            self._show_error_if_checking(
                node, expected_type_arg_message, error_code=ErrorCode.invalid_annotation
            )
            return AnyValue(AnySource.error)
        generic_bases = self.checker.get_generic_bases(synthetic_typ, ())
        if not generic_bases.get(synthetic_typ):
            return None
        return PartialValue(
            PartialValueOperation.SUBSCRIPT,
            root_for_partial,
            node,
            members,
            GenericValue(synthetic_typ, list(normalized_members)),
        )

    def _get_dunder(self, node: ast.AST, callee_val: Value, method_name: str) -> Value:
        synthetic_lookup_val = callee_val
        if isinstance(callee_val, AnnotatedValue):
            is_dunder = method_name.startswith("__") and method_name.endswith("__")
            has_explicit_method = any(
                extension.attribute_name == KnownValue(method_name)
                for extension in callee_val.get_metadata_of_type(HasAttrExtension)
            )
            if has_explicit_method and not is_dunder:
                return self.get_attribute(
                    Composite(callee_val),
                    method_name,
                    node,
                    ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
                )
            synthetic_lookup_val = callee_val.value
        fallback_lookup_val = callee_val.get_type_value()
        if isinstance(synthetic_lookup_val, TypedValue) and isinstance(
            synthetic_lookup_val.typ, str
        ):
            synthetic_class = self.checker.get_synthetic_class(synthetic_lookup_val.typ)
            if synthetic_class is not None:
                method_object = self.get_attribute(
                    Composite(synthetic_class),
                    method_name,
                    node,
                    ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
                )
                if method_object is not UNINITIALIZED_VALUE:
                    synthetic_typevars = self._get_synthetic_instance_typevars(
                        synthetic_lookup_val
                    )
                    if synthetic_typevars:
                        if isinstance(method_object, KnownValueWithTypeVars):
                            merged_typevars = {
                                typevar: value.substitute_typevars(synthetic_typevars)
                                for typevar, value in method_object.typevars.items()
                            }
                            merged_typevars.update(synthetic_typevars)
                            method_object = KnownValueWithTypeVars(
                                method_object.val, merged_typevars
                            )
                        elif isinstance(method_object, KnownValue):
                            method_object = KnownValueWithTypeVars(
                                method_object.val, synthetic_typevars
                            )
                        else:
                            method_object = method_object.substitute_typevars(
                                synthetic_typevars
                            )
                    return method_object

        method_object = self.get_attribute(
            Composite(fallback_lookup_val),
            method_name,
            node,
            ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
        )
        if method_object is UNINITIALIZED_VALUE:
            self.show_error(
                node,
                f"Object of type {callee_val} does not support {method_name!r}",
                error_code=ErrorCode.unsupported_operation,
            )
        return method_object

    def _get_synthetic_instance_typevars(self, value: TypedValue) -> TypeVarMap:
        if not isinstance(value.typ, str):
            return {}
        if not isinstance(value, GenericValue):
            return {}
        generic_args = value.args
        declared_type_params = self.checker.get_type_parameters(value.typ)
        if not declared_type_params:
            return {}
        typevars: list[TypeVarLike] = []
        for type_param in declared_type_params:
            if isinstance(type_param, TypeVarValue):
                typevars.append(type_param.typevar)
            elif isinstance(type_param, InputSigValue) and isinstance(
                type_param.input_sig, ParamSpecSig
            ):
                typevars.append(type_param.input_sig.param_spec)
        return dict(zip(typevars, generic_args))

    def _check_dunder_call_or_catch(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> Value | list[node_visitor.Error]:
        """Use this for checking a dunder call that may fall back to another.

        There are three cases:
        - The dunder does not exist. We want to defer the error, in case the fallback
          exists.
        - The dunder exists and the call typechecks. We want to return its result.
        - The dunder exists, but the call doesn't typecheck. We want to show the error
          immediately and return Any.

        """
        with self.catch_errors() as errors:
            result, exists = self._check_dunder_call(
                node, callee_composite, method_name, args, allow_call=allow_call
            )
        if not errors:
            return result
        elif exists:
            # Inplace method exists, but it doesn't accept these arguments
            self.show_caught_errors(errors)
            return result
        else:
            return errors

    def _check_dunder_call(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> tuple[Value, bool]:
        val = replace_fallback(callee_composite.value)
        if isinstance(val, MultiValuedValue):
            composites = [
                Composite(subval, callee_composite.varname, callee_composite.node)
                for subval in val.vals
            ]
            with override(self, "in_union_decomposition", True):
                values_and_exists = [
                    self._check_dunder_call_no_mvv(
                        node, composite, method_name, args, allow_call
                    )
                    for composite in composites
                ]
            values = [value for value, _ in values_and_exists]
            # TODO: We should do something more complex when unions are involved.
            exists = all(exists for _, exists in values_and_exists)
            return (
                unite_and_simplify(
                    *values, limit=self.options.get_value_for(UnionSimplificationLimit)
                ),
                exists,
            )
        return self._check_dunder_call_no_mvv(
            node, callee_composite, method_name, args, allow_call
        )

    def _check_dunder_call_no_mvv(
        self,
        node: ast.AST,
        callee_composite: Composite,
        method_name: str,
        args: Iterable[Composite],
        allow_call: bool = False,
    ) -> tuple[Value, bool]:
        stripped_value = _strip_predicate_intersection(callee_composite.value)
        stripped_callee = (
            callee_composite
            if stripped_value is callee_composite.value
            else Composite(
                stripped_value, callee_composite.varname, callee_composite.node
            )
        )
        method_object = self._get_dunder(node, stripped_callee.value, method_name)
        if method_object is UNINITIALIZED_VALUE:
            return AnyValue(AnySource.error), False
        return_value = self.check_call(
            node, method_object, [stripped_callee, *args], allow_call=allow_call
        )
        return return_value, True

    def _get_composite(self, composite: Varname, node: ast.AST, value: Value) -> Value:
        local_value = self.scopes.get(
            composite, node, self.state, fallback_value=value, can_assign_ctx=self
        )
        if isinstance(local_value, MultiValuedValue):
            vals = [val for val in local_value.vals if val is not UNINITIALIZED_VALUE]
            if vals:
                return unite_values(*vals)
            else:
                return NO_RETURN_VALUE
        return local_value

    def visit_Attribute(self, node: ast.Attribute) -> Value:
        return self.composite_from_attribute(node).value

    def _extend_composite(
        self, root_composite: Composite, index: CompositeIndex, node: ast.AST
    ) -> VarnameWithOrigin | None:
        varname = root_composite.get_extended_varname(index)
        if varname is None:
            return None
        origin = self.scopes.current_scope().get_origin(varname, node, self.state)
        return root_composite.get_extended_varname_with_origin(index, origin)

    def composite_from_attribute(self, node: ast.Attribute) -> Composite:
        if isinstance(node.value, ast.Name):
            attr_str = f"{node.value.id}.{node.attr}"
            if self._is_write_ctx(node.ctx):
                self.yield_checker.record_assignment(attr_str)
            else:
                self.yield_checker.record_usage(attr_str, node)

        root_composite = self.composite_from_node(node.value)
        composite = self._extend_composite(root_composite, node.attr, node)
        if self._is_write_ctx(node.ctx):
            # TODO: We should do something here if we're in an AnnAssign, e.g.
            # note the type in the class's namespace.
            if self.being_assigned is None:
                assert (
                    self.ann_assign_type is not None
                ), "should only happen in AnnAssign"
                return Composite(AnyValue(AnySource.inference), composite, node)
            is_final_assignment = self._is_assignment_to_final_attribute(
                node, root_composite.value
            )
            if is_final_assignment:
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to final name {node.attr}",
                    error_code=ErrorCode.incompatible_assignment,
                )
            is_classvar_instance_assignment = (
                not is_final_assignment
                and self._is_assignment_to_classvar_through_instance(
                    node, root_composite.value
                )
            )
            if is_classvar_instance_assignment:
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to class variable {node.attr!r} via instance",
                    error_code=ErrorCode.incompatible_assignment,
                )
            is_instance_member_class_assignment = (
                not is_final_assignment
                and not is_classvar_instance_assignment
                and self._is_assignment_to_instance_member_through_class(
                    node, root_composite.value
                )
            )
            if is_instance_member_class_assignment:
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to instance attribute {node.attr!r} via class object",
                    error_code=ErrorCode.incompatible_assignment,
                )
            is_frozen_dataclass_assignment = (
                not is_final_assignment
                and not is_classvar_instance_assignment
                and not is_instance_member_class_assignment
                and self._is_assignment_to_frozen_dataclass_attribute(
                    root_composite.value
                )
            )
            if is_frozen_dataclass_assignment:
                self._show_error_if_checking(
                    node,
                    "Dataclass is frozen",
                    error_code=ErrorCode.incompatible_assignment,
                )
            is_namedtuple_field = self._is_namedtuple_field_attribute(
                root_composite.value, node.attr
            )
            if is_namedtuple_field:
                self._show_namedtuple_attribute_mutation_error(node)
            is_slots_assignment = (
                not is_final_assignment
                and not is_classvar_instance_assignment
                and not is_instance_member_class_assignment
                and not is_frozen_dataclass_assignment
                and not is_namedtuple_field
                and self._is_assignment_to_non_slot_attribute(
                    root_composite.value, node.attr
                )
            )
            if is_slots_assignment:
                self._show_error_if_checking(
                    node,
                    f"Cannot assign to attribute {node.attr!r}; it is not in __slots__",
                    error_code=ErrorCode.incompatible_assignment,
                )
            if (
                not is_final_assignment
                and not is_classvar_instance_assignment
                and not is_instance_member_class_assignment
                and not is_frozen_dataclass_assignment
                and not is_namedtuple_field
                and not is_slots_assignment
            ):
                self._check_attribute_assignment_type(node, root_composite)
                self._record_synthetic_attr_set(node, root_composite.value)
            should_record_type_attr_set = (
                not is_final_assignment
                and not is_classvar_instance_assignment
                and not is_instance_member_class_assignment
                and not is_frozen_dataclass_assignment
                and not is_namedtuple_field
                and not is_slots_assignment
            )
            if (
                composite is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                self.scopes.set(
                    composite.get_varname(), self.being_assigned, node, self.state
                )

            if isinstance(root_composite.value, TypedValue):
                typ = root_composite.value.typ
                if isinstance(typ, type):
                    if should_record_type_attr_set:
                        self._record_type_attr_set(
                            typ, node.attr, node, self.being_assigned
                        )
            elif isinstance(root_composite.value, GenericValue):
                typ = root_composite.value.typ
                if isinstance(typ, type):
                    if should_record_type_attr_set:
                        self._record_type_attr_set(
                            typ, node.attr, node, self.being_assigned
                        )
            elif isinstance(root_composite.value, KnownValue) and not isinstance(
                root_composite.value.val, type
            ):
                if should_record_type_attr_set:
                    self._record_type_attr_set(
                        type(root_composite.value.val),
                        node.attr,
                        node,
                        self.being_assigned,
                    )
            return Composite(self.being_assigned, composite, node)
        elif self._is_read_ctx(node.ctx):
            if self._is_checking():
                if (
                    isinstance(root_composite.value, KnownValue)
                    and isinstance(root_composite.value.val, types.ModuleType)
                    and root_composite.value.val.__name__ is not None
                ):
                    self.reexport_tracker.record_attribute_accessed(
                        root_composite.value.val.__name__, node.attr, node, self
                    )
            value = self.get_attribute(
                root_composite,
                node.attr,
                node,
                use_fallback=True,
                ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
            )
            if isinstance(node.ctx, ast.Del) and self._is_namedtuple_field_attribute(
                root_composite.value, node.attr
            ):
                self._show_namedtuple_attribute_mutation_error(node)
            self.check_deprecation(node, value)
            if self._should_use_varname_value(value):
                varname_value = self.checker.maybe_get_variable_name_value(node.attr)
                if varname_value is not None:
                    return Composite(varname_value, composite, node)
            if (
                composite is not None
                and self.scopes.scope_type() == ScopeType.function_scope
            ):
                local_value = self._get_composite(composite.get_varname(), node, value)
                if local_value is not UNINITIALIZED_VALUE:
                    value = local_value
            if root_composite.value == KnownValue(sys):
                if node.attr == "platform":
                    value = annotate_value(value, [SYS_PLATFORM_EXTENSION])
                elif node.attr == "version_info":
                    value = annotate_value(value, [SYS_VERSION_INFO_EXTENSION])
            return Composite(value, composite, node)
        else:
            self.show_error(node, "Unknown context", ErrorCode.unexpected_node)
            return Composite(AnyValue(AnySource.error), composite, node)

    def _get_attribute_value_for_assignment(
        self, root_composite: Composite, attr: str, node: ast.Attribute
    ) -> Value:
        lookup_composite = root_composite
        root_value = replace_fallback(root_composite.value)
        if isinstance(root_value, KnownValue) and not isinstance(root_value.val, type):
            lookup_composite = Composite(
                TypedValue(type(root_value.val)),
                root_composite.varname,
                root_composite.node,
            )
            root_value = lookup_composite.value
        if isinstance(root_value, (MultiValuedValue, IntersectionValue)):
            return UNINITIALIZED_VALUE
        ctx = _AttrContext(
            lookup_composite,
            attr,
            self,
            node=node,
            ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
            record_reads=False,
        )
        return attributes.get_attribute(ctx)

    def _normalize_expected_attribute_type_for_assignment(
        self, value: Value
    ) -> Value | None:
        if value is UNINITIALIZED_VALUE:
            return None
        value = replace_fallback(value)
        if isinstance(value, AnyValue):
            return None
        if isinstance(value, AnnotatedValue):
            normalized = self._normalize_expected_attribute_type_for_assignment(
                value.value
            )
            if normalized is None:
                return None
            return annotate_value(normalized, value.metadata)
        if isinstance(value, MultiValuedValue):
            normalized_vals: list[Value] = []
            saw_non_known = False
            for subval in value.vals:
                subval = replace_fallback(subval)
                if isinstance(subval, KnownValue):
                    normalized_vals.append(TypedValue(type(subval.val)))
                    continue
                normalized_subval = (
                    self._normalize_expected_attribute_type_for_assignment(subval)
                )
                if normalized_subval is not None:
                    normalized_vals.append(normalized_subval)
                    saw_non_known = True
            if not normalized_vals or not saw_non_known:
                return None
            return unite_values(*normalized_vals)
        if isinstance(value, KnownValue):
            return None
        if isinstance(value, SequenceValue):
            return value
        return value

    def _check_attribute_assignment_type(
        self, node: ast.Attribute, root_composite: Composite
    ) -> None:
        if self.being_assigned is None:
            return
        if (
            self.current_class_key is not None
            and self._is_current_method_receiver_node(node.value)
            and self._class_key_for_attribute_target(node, root_composite.value)
            == self.current_class_key
            and not self._is_protocol_base_member_assignment(
                self.current_class_key, node.attr
            )
        ):
            return
        expected = self._get_attribute_value_for_assignment(
            root_composite, node.attr, node
        )
        expected_type = self._normalize_expected_attribute_type_for_assignment(expected)
        if expected_type is None:
            return
        can_assign = has_relation(
            expected_type, self.being_assigned, Relation.ASSIGNABLE, self
        )
        if isinstance(can_assign, CanAssignError):
            self._show_error_if_checking(
                node,
                f"Incompatible assignment: expected {expected_type}, got"
                f" {self.being_assigned}",
                error_code=ErrorCode.incompatible_assignment,
                detail=can_assign.display(),
            )

    def _is_protocol_base_member_assignment(
        self, class_key: type | str, member_name: str
    ) -> bool:
        for base_key in self.checker.get_generic_bases(class_key):
            if base_key == class_key:
                continue
            type_object = self.checker.make_type_object(base_key)
            if member_name in type_object.protocol_members:
                return True
        return False

    def get_attribute(
        self,
        root_composite: Composite,
        attr: str,
        node: ast.AST | None = None,
        *,
        ignore_none: bool = False,
        use_fallback: bool = False,
        prefer_typeshed: bool = False,
    ) -> Value:
        """Get an attribute of this value.

        Returns :data:`pycroscope.value.UNINITIALIZED_VALUE` if the attribute cannot be found.

        """
        if isinstance(root_composite.value, TypeVarValue):
            root_composite = Composite(
                value=root_composite.value.get_fallback_value(),
                varname=root_composite.varname,
                node=root_composite.node,
            )
        if self._is_instance_member_accessed_through_class(root_composite, attr, node):
            if node is not None:
                self._show_error_if_checking(
                    node,
                    f"{root_composite.value} has no attribute {attr!r}",
                    ErrorCode.undefined_attribute,
                )
                return AnyValue(AnySource.error)
            return UNINITIALIZED_VALUE
        if is_union(root_composite.value):
            results = []
            for subval in flatten_values(root_composite.value):
                composite = Composite(
                    subval, root_composite.varname, root_composite.node
                )
                subresult = self.get_attribute(
                    composite,
                    attr,
                    node,
                    ignore_none=ignore_none,
                    use_fallback=use_fallback,
                )
                if (
                    subresult is UNINITIALIZED_VALUE
                    and use_fallback
                    and node is not None
                ):
                    subresult = self._get_attribute_fallback(subval, attr, node)
                results.append(subresult)
            return unite_values(*results)
        elif isinstance(root_composite.value, IntersectionValue):
            # If the value is an intersection, we need to get the attribute from each
            # of the intersection's values.
            results = []
            for subval in root_composite.value.vals:
                composite = Composite(
                    subval, root_composite.varname, root_composite.node
                )
                subresult = self.get_attribute(
                    composite, attr, node, ignore_none=ignore_none, use_fallback=False
                )
                if (
                    subresult is UNINITIALIZED_VALUE
                    and use_fallback
                    and node is not None
                ):
                    subresult = self._get_attribute_fallback(
                        subval, attr, node, allow_error=False
                    )
                if subresult is not UNINITIALIZED_VALUE:
                    results.append(subresult)
            if not results:
                if node is not None:
                    self._show_error_if_checking(
                        node,
                        f"Intersection value {root_composite.value} has no attribute {attr!r}",
                        ErrorCode.undefined_attribute,
                    )
                return AnyValue(AnySource.error)
            return intersect_multi(results, self)
        ctx = _AttrContext(
            root_composite,
            attr,
            self,
            node=node,
            ignore_none=ignore_none,
            prefer_typeshed=prefer_typeshed,
        )
        result = attributes.get_attribute(ctx)
        if (
            result is UNINITIALIZED_VALUE
            and node is not None
            and isinstance(root_composite.value, TypeAliasValue)
            and attributes._is_type_alias_symbol(ctx)
        ):
            self._show_error_if_checking(
                node,
                f"{root_composite.value} has no attribute {attr!r}",
                ErrorCode.undefined_attribute,
            )
            return AnyValue(AnySource.error)
        if result is UNINITIALIZED_VALUE and use_fallback and node is not None:
            return self._get_attribute_fallback(root_composite.value, attr, node)
        return result

    def get_attribute_from_value(
        self, root_value: Value, attribute: str, *, prefer_typeshed: bool = False
    ) -> Value:
        return self.get_attribute(
            Composite(root_value), attribute, prefer_typeshed=prefer_typeshed
        )

    def _get_attribute_fallback(
        self, root_value: Value, attr: str, node: ast.AST, *, allow_error: bool = True
    ) -> Value:
        # We don't throw an error in many
        # cases where we're not quite sure whether an attribute
        # will exist.
        root_value = replace_fallback(root_value)
        if isinstance(root_value, UnboundMethodValue):
            if self._should_ignore_val(node):
                return AnyValue(AnySource.error)
        elif isinstance(root_value, KnownValue):
            # super calls on mixin classes may use attributes that are defined only on child classes
            if isinstance(root_value.val, super):
                subclasses = get_subclasses_recursively(root_value.val.__thisclass__)
                if any(
                    hasattr(cls, attr)
                    for cls in subclasses
                    if cls is not root_value.val.__thisclass__
                ):
                    return AnyValue(AnySource.inference)

            # Ignore objects that override __getattr__.
            # typing alias objects expose __getattr__ but still should not be
            # treated as having arbitrary attributes.
            has_dynamic_getattr = _static_hasattr(root_value.val, "__getattr__")
            if has_dynamic_getattr and _is_typing_alias_value(root_value.val):
                has_dynamic_getattr = False
            if not _has_only_known_attributes(
                self.checker.ts_finder, root_value.val
            ) and (has_dynamic_getattr or self._should_ignore_val(node)):
                return AnyValue(AnySource.inference)
        elif isinstance(root_value, TypedValue):
            root_type = root_value.typ
            if isinstance(root_type, type) and not _has_only_known_attributes(
                self.checker.ts_finder, root_type
            ):
                if not self._is_dataclass_initvar_attribute(root_type, attr):
                    return self._maybe_get_attr_value(root_type, attr)
        elif isinstance(root_value, SubclassValue):
            if isinstance(root_value.typ, TypedValue):
                root_type = root_value.typ.typ
                if isinstance(root_type, type) and not _has_only_known_attributes(
                    self.checker.ts_finder, root_type
                ):
                    if not self._is_dataclass_initvar_attribute(root_type, attr):
                        return self._maybe_get_attr_value(root_type, attr)
            else:
                return AnyValue(AnySource.inference)
        elif isinstance(root_value, MultiValuedValue):
            return unite_values(
                *[
                    self._get_attribute_fallback(
                        val, attr, node, allow_error=allow_error
                    )
                    for val in root_value.vals
                ]
            )
        if allow_error:
            self._show_error_if_checking(
                node,
                f"{root_value} has no attribute {attr!r}",
                ErrorCode.undefined_attribute,
            )
            return AnyValue(AnySource.error)
        else:
            return UNINITIALIZED_VALUE

    def composite_from_node(self, node: ast.AST) -> Composite:
        if isinstance(node, ast.Attribute):
            composite = self.composite_from_attribute(node)
        elif isinstance(node, ast.Name):
            composite = self.composite_from_name(node)
        elif isinstance(node, ast.Subscript):
            composite = self.composite_from_subscript(node)
        elif isinstance(node, ast.Slice):
            # These don't have a .lineno attribute, which would otherwise cause trouble.
            composite = Composite(self.visit(node), None, None)
        elif isinstance(node, ast.NamedExpr):
            composite = self.composite_from_walrus(node)
        else:
            composite = Composite(self.visit(node), None, node)
        if self.annotate:
            node.inferred_value = composite.value
        return composite

    def varname_for_constraint(self, node: ast.AST) -> VarnameWithOrigin | None:
        """Given a node, returns a variable name that could be used in a local scope."""
        composite = self.composite_from_node(node)
        return composite.varname

    def varname_for_self_constraint(self, node: ast.AST) -> VarnameWithOrigin | None:
        """Helper for constraints on self from method calls.

        Given an ``ast.Call`` node representing a method call, return the variable name
        to be used for a constraint on the self object.

        """
        if not isinstance(node, ast.Call):
            return None
        if isinstance(node.func, ast.Attribute):
            return self.varname_for_constraint(node.func.value)
        else:
            return None

    def _should_ignore_val(self, node: ast.AST) -> bool:
        if node is not None:
            path = get_attribute_path(node)
            if path is not None:
                ignored_paths = self.options.get_value_for(IgnoredPaths)
                for ignored_path in ignored_paths:
                    if path[: len(ignored_path)] == ignored_path:
                        return True
                if path[-1] in self.options.get_value_for(IgnoredEndOfReference):
                    self.log(logging.INFO, "Ignoring end of reference", path)
                    return True
        return False

    # Call nodes

    def visit_Call(self, node: ast.Call, *, callee: Value | None = None) -> Value:
        callee_wrapped = callee if callee is not None else self.visit(node.func)
        args = [self.composite_from_node(arg) for arg in node.args]
        if node.keywords:
            keywords = [
                (kw.arg, self.composite_from_node(kw.value)) for kw in node.keywords
            ]
        else:
            keywords = []

        if (
            self._is_current_class_dataclass()
            and self.scopes.scope_type() == ScopeType.class_scope
        ):
            inferred_options = (
                self._infer_dataclass_field_call_options_from_resolved_call(
                    callee_wrapped, args, keywords, node
                )
            )
            if inferred_options is not None:
                self._dataclass_field_call_options_by_node[id(node)] = inferred_options
            else:
                self._dataclass_field_call_options_by_node.pop(id(node), None)

        if self._is_forbidden_annotated_call(callee_wrapped):
            self._show_error_if_checking(
                node,
                "Annotated cannot be called",
                error_code=ErrorCode.invalid_annotation,
            )
            return AnyValue(AnySource.error)

        self._check_unsafe_super_protocol_call(node)

        return_value = self.check_call(
            node, callee_wrapped, args, keywords, allow_call=self.in_annotation
        )

        if self._is_checking():
            self.yield_checker.record_call(callee_wrapped, node)
            self.asynq_checker.check_call(callee_wrapped, node)

        if self.collector is not None:
            callee_val = None
            if isinstance(callee_wrapped, UnboundMethodValue):
                callee_val = callee_wrapped.get_method()
            elif isinstance(callee_wrapped, KnownValue):
                callee_val = callee_wrapped.val
            elif isinstance(callee_wrapped, SubclassValue) and isinstance(
                callee_wrapped.typ, TypedValue
            ):
                callee_val = callee_wrapped.typ.typ

            if callee_val is not None:
                caller = (
                    self.current_function
                    if self.current_function is not None
                    else self.module
                )
                if caller is not None:
                    self.collector.record_call(caller, callee_val)

        if (
            isinstance(callee_wrapped, KnownValue)
            and callee_wrapped.val is assert_error
        ):
            return annotate_value(return_value, [AssertErrorExtension()])
        return return_value

    def _check_unsafe_super_protocol_call(self, node: ast.Call) -> None:
        if not self._is_checking() or self.module is not None:
            return
        member_name = self._super_call_member_name(node)
        if member_name is None:
            return
        class_key = self.current_class_key
        if class_key is None:
            return
        for base_key in self._direct_base_class_keys(class_key):
            if member_name in self._required_abstract_members_for_base(base_key):
                self._show_error_if_checking(
                    node.func,
                    "Call to abstract protocol member via super() has no default implementation",
                    error_code=ErrorCode.bad_super_call,
                )
                return

    def _super_call_member_name(self, node: ast.Call) -> str | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        inner_call = node.func.value
        if not isinstance(inner_call, ast.Call):
            return None
        if (
            not isinstance(inner_call.func, ast.Name)
            or inner_call.func.id != "super"
            or inner_call.args
            or inner_call.keywords
        ):
            return None
        return node.func.attr

    def _direct_base_class_keys(self, class_key: type | str) -> list[type | str]:
        if isinstance(class_key, str):
            synthetic_class = self.checker.get_synthetic_class(class_key)
            if synthetic_class is None:
                return []
            keys: list[type | str] = []
            for base_value in synthetic_class.base_classes:
                base_key = self._base_class_key_from_value(base_value)
                if base_key is not None:
                    keys.append(base_key)
            return keys
        return [base for base in safe_getattr(class_key, "__bases__", ())]

    def _can_perform_call(
        self, args: Iterable[Value], keywords: Iterable[tuple[str | None, Value]]
    ) -> Annotated[
        bool,
        ParameterTypeGuard["args", Iterable[KnownValue]],
        ParameterTypeGuard["keywords", Iterable[tuple[str, KnownValue]]],
    ]:
        """Returns whether all of the arguments were inferred successfully."""
        return all(isinstance(arg, KnownValue) for arg in args) and all(
            keyword is not None and isinstance(arg, KnownValue)
            for keyword, arg in keywords
        )

    def _is_invalid_typevar_bound(self, value: Value) -> bool:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._is_invalid_typevar_bound(value.value)
        if isinstance(value, MultiValuedValue):
            return any(self._is_invalid_typevar_bound(subval) for subval in value.vals)
        if isinstance(value, KnownValue):
            if is_typing_name(value.val, "TypedDict"):
                return True
            value = type_from_value(value, self, suppress_errors=True)
        if any(isinstance(subval, TypeVarValue) for subval in value.walk_values()):
            return True
        return False

    def _check_invalid_typevar_bound(
        self,
        callee: Value,
        keywords: Sequence[tuple[str | None, Composite]],
        node: ast.AST | None = None,
    ) -> None:
        if not (
            isinstance(callee, KnownValue) and is_typing_name(callee.val, "TypeVar")
        ):
            return
        for keyword, composite in keywords:
            if keyword != "bound":
                continue
            bound = composite.value
            message = self._typevar_invalid_bound_message(bound)
            if message is None:
                continue
            error_node = composite.node if composite.node is not None else node
            if error_node is None:
                continue
            self._show_error_if_checking(
                error_node, message, error_code=ErrorCode.invalid_annotation
            )

    def _typevar_invalid_bound_message(
        self, value: Value, *, is_constraint: bool = False
    ) -> str | None:
        if not self._is_invalid_typevar_bound(value):
            return None
        if isinstance(value, KnownValue) and is_typing_name(value.val, "TypedDict"):
            kind = "constraint" if is_constraint else "bound"
            return f"TypedDict cannot be used as a TypeVar {kind}"
        if is_constraint:
            return "TypeVar constraint cannot be parameterized by type variables"
        return "TypeVar bound cannot be parameterized by type variables"

    def _check_typevar_default_constraints(
        self, typevar: TypeVarValue, node: ast.AST | None
    ) -> None:
        if node is None or typevar.default is None:
            return
        if typevar.bound is not None:
            can_assign = has_relation(
                typevar.bound, typevar.default, Relation.ASSIGNABLE, self
            )
            if isinstance(can_assign, CanAssignError):
                self._show_error_if_checking(
                    node,
                    "the bound and default are incompatible",
                    error_code=ErrorCode.invalid_annotation,
                )
                return
        if typevar.constraints:

            def _default_matches_constraint(constraint: Value, default: Value) -> bool:
                if constraint == default:
                    return True
                if isinstance(constraint, TypedValue) and isinstance(
                    default, TypedValue
                ):
                    return constraint.typ == default.typ
                if isinstance(constraint, KnownValue) and isinstance(
                    default, KnownValue
                ):
                    return constraint.val == default.val
                return False

            for constraint in typevar.constraints:
                if _default_matches_constraint(constraint, typevar.default):
                    return
            self._show_error_if_checking(
                node,
                "TypeVar default must be one of its constraints",
                error_code=ErrorCode.invalid_annotation,
            )

    def _maybe_build_typevar_call_value(
        self,
        callee: Value,
        args: Iterable[Composite],
        keywords: Sequence[tuple[str | None, Composite]],
    ) -> TypeVarValue | None:
        if not (
            isinstance(callee, KnownValue) and is_typing_name(callee.val, "TypeVar")
        ):
            return None

        args = tuple(args)
        if not args:
            return None
        name_arg = args[0].value
        if not (isinstance(name_arg, KnownValue) and isinstance(name_arg.val, str)):
            return None

        def _typevar_arg_to_type(composite: Composite) -> Value:
            value = composite.value
            suppress_errors = isinstance(value, KnownValue) and isinstance(
                value.val, str
            )
            return type_from_value(
                value, self, composite.node, suppress_errors=suppress_errors
            )

        constraints = [_typevar_arg_to_type(arg) for arg in args[1:]]
        bound = default = None
        covariant = False
        contravariant = False
        infer_variance = False
        for keyword, composite in keywords:
            if keyword is None:
                return None
            kwarg_value = composite.value
            if keyword in ("covariant", "contravariant", "infer_variance"):
                if not (
                    isinstance(kwarg_value, KnownValue)
                    and isinstance(kwarg_value.val, bool)
                ):
                    return None
                if keyword == "covariant":
                    covariant = kwarg_value.val
                elif keyword == "contravariant":
                    contravariant = kwarg_value.val
                else:
                    infer_variance = kwarg_value.val
            elif keyword == "bound":
                bound = _typevar_arg_to_type(composite)
            elif keyword == "default":
                default = _typevar_arg_to_type(composite)
            else:
                return None

        try:
            if infer_variance:
                kwargs_with_infer = {
                    "covariant": covariant,
                    "contravariant": contravariant,
                    "infer_variance": True,
                }
                typevar = cast(Any, TypeVar)(name_arg.val, **kwargs_with_infer)
            else:
                typevar = TypeVar(
                    name_arg.val, covariant=covariant, contravariant=contravariant
                )
        except Exception:
            return None

        if covariant:
            variance = Variance.COVARIANT
        elif contravariant:
            variance = Variance.CONTRAVARIANT
        else:
            variance = Variance.INVARIANT
        return TypeVarValue(
            typevar,
            bound=bound,
            constraints=tuple(constraints),
            default=default,
            variance=variance,
        )

    def _call_assignment_target_name(self, node: ast.AST | None) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        parent = self.node_context.nth_parent(2)
        if (
            isinstance(parent, ast.Assign)
            and parent.value is node
            and len(parent.targets) == 1
        ):
            target = parent.targets[0]
            if isinstance(target, ast.Name):
                return target.id
        if (
            isinstance(parent, ast.AnnAssign)
            and parent.value is node
            and isinstance(parent.target, ast.Name)
        ):
            return parent.target.id
        return None

    def _maybe_get_name_arg_in_call(
        self,
        args: Sequence[Composite],
        keywords: Sequence[tuple[str | None, Composite]],
        keyword_name: str,
    ) -> tuple[str, ast.AST | None] | None:
        if args:
            positional_name = replace_fallback(args[0].value)
            if isinstance(positional_name, KnownValue) and isinstance(
                positional_name.val, str
            ):
                return positional_name.val, args[0].node
        for keyword, composite in keywords:
            if keyword != keyword_name:
                continue
            keyword_value = replace_fallback(composite.value)
            if isinstance(keyword_value, KnownValue) and isinstance(
                keyword_value.val, str
            ):
                return keyword_value.val, composite.node
            break
        return None

    def _check_assignment_target_name_match(
        self,
        node: ast.AST | None,
        callee: Value,
        args: Sequence[Composite],
        keywords: Sequence[tuple[str | None, Composite]],
    ) -> None:
        if not isinstance(callee, KnownValue):
            return
        assigned_name = self._call_assignment_target_name(node)
        if assigned_name is None:
            return
        for construct_name, name_keyword in _TYPING_CONSTRUCTS_WITH_NAME_ARG.items():
            if not is_typing_name(callee.val, construct_name):
                continue
            maybe_name_arg = self._maybe_get_name_arg_in_call(
                args, keywords, name_keyword
            )
            if maybe_name_arg is None:
                return
            name_arg_value, name_arg_node = maybe_name_arg
            if name_arg_value != assigned_name:
                error_node = name_arg_node if name_arg_node is not None else node
                if error_node is None:
                    return
                self._show_error_if_checking(
                    error_node,
                    f"{construct_name} name argument must match the assignment target"
                    " name",
                    error_code=ErrorCode.incompatible_call,
                )
            return

    def check_call(
        self,
        node: ast.AST | None,
        callee: Value,
        args: Iterable[Composite],
        keywords: Iterable[tuple[str | None, Composite]] = (),
        *,
        allow_call: bool = False,
    ) -> Value:
        if isinstance(callee, MultiValuedValue):
            with override(self, "in_union_decomposition", True):
                values = [
                    self._check_call_no_mvv(
                        node, val, args, keywords, allow_call=allow_call
                    )
                    for val in callee.vals
                ]

            pairs = [
                unannotate_value(val, NoReturnConstraintExtension) for val in values
            ]
            val = unite_values(*[val for val, _ in pairs])
            constraint = OrConstraint.make(
                [
                    AndConstraint.make(ext.constraint for ext in exts)
                    for _, exts in pairs
                ]
            )
        else:
            val = self._check_call_no_mvv(
                node, callee, args, keywords, allow_call=allow_call
            )
            val, nru_extensions = unannotate_value(val, NoReturnConstraintExtension)
            constraint = AndConstraint.make(ext.constraint for ext in nru_extensions)
        self.add_constraint(node, constraint)
        return val

    def _get_instantiable_protocol_class_name(self, value: Value) -> str | None:
        value = replace_fallback(value)
        if isinstance(value, AnnotatedValue):
            return self._get_instantiable_protocol_class_name(value.value)
        if isinstance(value, SubclassValue):
            # ``type[Proto]`` may refer to concrete implementers. Rejecting calls
            # here causes spurious "Cannot instantiate protocol class" errors.
            return None
        if self._is_class_object_attribute_root(value) is not True:
            return None
        class_key = self._base_class_key_from_value(value)
        if class_key is None:
            return None
        if self.checker.make_type_object(class_key).is_protocol:
            if isinstance(value, SyntheticClassObjectValue):
                return value.name
            if isinstance(class_key, type):
                return class_key.__name__
            return class_key.rsplit(".", 1)[-1]
        return None

    def _check_call_no_mvv(
        self,
        node: ast.AST | None,
        callee_wrapped: Value,
        args: Iterable[Composite],
        keywords: Iterable[tuple[str | None, Composite]] = (),
        *,
        allow_call: bool = False,
    ) -> Value:
        args = tuple(args)
        keywords = tuple(keywords)
        if isinstance(callee_wrapped, KnownValue) and any(
            callee_wrapped.val is ignored
            for ignored in self.options.get_value_for(IgnoredCallees)
        ):
            self.log(logging.INFO, "Ignoring callee", callee_wrapped)
            return AnyValue(AnySource.error)

        self._check_invalid_typevar_bound(callee_wrapped, keywords, node=node)
        self._check_assignment_target_name_match(node, callee_wrapped, args, keywords)

        protocol_class_name = self._get_instantiable_protocol_class_name(callee_wrapped)
        if protocol_class_name is not None:
            if node is not None:
                self._show_error_if_checking(
                    node,
                    f"Cannot instantiate protocol class {protocol_class_name}",
                    error_code=ErrorCode.incompatible_call,
                )
            return AnyValue(AnySource.error)

        if (
            isinstance(callee_wrapped, SyntheticClassObjectValue)
            and isinstance(callee_wrapped.class_type, TypedValue)
            and isinstance(callee_wrapped.class_type.typ, str)
            and self._synthetic_abstract_methods.get(callee_wrapped.class_type.typ)
        ):
            if node is not None:
                self._show_error_if_checking(
                    node,
                    f"Cannot instantiate abstract class {callee_wrapped.name}",
                    error_code=ErrorCode.incompatible_call,
                )
            return AnyValue(AnySource.error)

        extended_argspec = self.signature_from_value(callee_wrapped, node)
        if extended_argspec is ANY_SIGNATURE:
            # don't bother calling it
            extended_argspec = None
            return_value = AnyValue(AnySource.from_another)

        elif extended_argspec is None:
            if node is not None:
                self._show_error_if_checking(
                    node,
                    f"{callee_wrapped} is not callable",
                    error_code=ErrorCode.not_callable,
                )
            return_value = AnyValue(AnySource.error)

        else:
            arguments = [
                (
                    (Composite(arg.value.root, arg.varname, arg.node), ARGS)
                    if (
                        isinstance(arg.value, PartialValue)
                        and arg.value.operation is PartialValueOperation.UNPACK
                    )
                    else (arg, None)
                )
                for arg in args
            ] + [
                (value, KWARGS) if keyword is None else (value, keyword)
                for keyword, value in keywords
            ]
            if self._is_checking():
                return_value = extended_argspec.check_call(arguments, self, node)
            else:
                with self.catch_errors():
                    return_value = extended_argspec.check_call(arguments, self, node)

        if extended_argspec is not None and not extended_argspec.has_return_value():
            local = self.get_local_return_value(extended_argspec)
            if local is not None:
                return_value = local

        synthesized_typevar = self._maybe_build_typevar_call_value(
            callee_wrapped, args, keywords
        )
        if synthesized_typevar is not None:
            self._check_typevar_default_constraints(synthesized_typevar, node)
        if synthesized_typevar is not None and (
            isinstance(return_value, AnyValue)
            or (
                isinstance(return_value, KnownValue)
                and is_typing_name(return_value.val, "TypeVar")
            )
            or (
                isinstance(return_value, TypedValue)
                and is_typing_name(return_value.typ, "TypeVar")
            )
        ):
            return_value = synthesized_typevar

        if (
            allow_call
            and isinstance(callee_wrapped, KnownValue)
            and self._should_perform_runtime_call(callee_wrapped)
        ):
            arg_values = [arg.value for arg in args]
            kw_values = [(kw, composite.value) for kw, composite in keywords]
            if self._can_perform_call(arg_values, kw_values):
                try:
                    result = callee_wrapped.val(
                        *[arg.val for arg in arg_values],
                        **{key: value.val for key, value in kw_values},
                    )
                except Exception as e:
                    self.log(logging.INFO, "exception calling", (callee_wrapped, e))
                else:
                    if result is NotImplemented:
                        self.show_error(
                            node,
                            f"Call to {callee_wrapped.val} is not supported",
                            error_code=ErrorCode.incompatible_call,
                        )
                    return_value = KnownValue(result)

        return_value = self._specialize_generic_alias_call_return(
            callee_wrapped, return_value, node
        )
        return_value = self._maybe_convert_local_namedtuple_class(
            callee_wrapped, return_value
        )

        if return_value is NO_RETURN_VALUE and node is not None:
            self._set_name_in_scope(LEAVES_SCOPE, node, AnyValue(AnySource.marker))

        # for .asynq functions, we use the argspec for the underlying function, but that means
        # that the return value is not wrapped in AsyncTask, so we do that manually here
        if isinstance(callee_wrapped, KnownValue) and is_dot_asynq_function(
            callee_wrapped.val
        ):
            async_fn = callee_wrapped.val.__self__
            return AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value)
        elif isinstance(
            callee_wrapped, UnboundMethodValue
        ) and callee_wrapped.secondary_attr_name in ("async", "asynq"):
            async_fn = callee_wrapped.get_method()
            return AsyncTaskIncompleteValue(_get_task_cls(async_fn), return_value)
        elif (
            asynq is not None
            and isinstance(callee_wrapped, UnboundMethodValue)
            and asynq.is_pure_async_fn(callee_wrapped.get_method())
        ):
            return return_value
        else:
            if (
                asynq is not None
                and isinstance(return_value, AnyValue)
                and isinstance(callee_wrapped, KnownValue)
                and asynq.is_pure_async_fn(callee_wrapped.val)
            ):
                task_cls = _get_task_cls(callee_wrapped.val)
                if isinstance(task_cls, type):
                    return TypedValue(task_cls)
            return return_value

    def _should_perform_runtime_call(self, callee: KnownValue) -> bool:
        callee_obj = callee.val
        return not (
            isinstance(callee_obj, type)
            and is_namedtuple_class(callee_obj)
            and should_disable_runtime_call_for_namedtuple_class(callee_obj)
            and not ClassesSafeToInstantiate.contains(callee_obj, self.options)
        )

    def _specialize_generic_alias_call_return(
        self, callee: Value, return_value: Value, node: ast.AST | None
    ) -> Value:
        if isinstance(callee, KnownValue):
            origin = get_origin(callee.val)
            if not isinstance(origin, type):
                return return_value
            runtime_args = get_args(callee.val)
            if not runtime_args:
                return return_value
            type_args = [
                type_from_value(KnownValue(arg), self, node, suppress_errors=True)
                for arg in runtime_args
            ]
        elif (
            isinstance(callee, PartialValue)
            and callee.operation is PartialValueOperation.SUBSCRIPT
            and isinstance(callee.root, KnownValue)
            and isinstance(callee.root.val, type)
        ):
            origin = callee.root.val
            if not callee.members:
                return return_value
            type_args = [
                type_from_value(member, self, node, suppress_errors=True)
                for member in callee.members
            ]
        else:
            return return_value
        if isinstance(return_value, KnownValue):
            if not isinstance(return_value.val, origin):
                return return_value
        elif isinstance(return_value, TypedValue):
            if return_value.typ is not origin:
                return return_value
        else:
            return return_value
        return GenericValue(origin, type_args)

    def _is_namedtuple_factory(self, value: Value) -> bool:
        return isinstance(value, KnownValue) and (
            value.val is collections.namedtuple
            or is_typing_name(value.val, "NamedTuple")
        )

    def _maybe_convert_local_namedtuple_class(
        self, callee: Value, return_value: Value
    ) -> Value:
        if self.scopes.scope_type() == ScopeType.module_scope:
            return return_value
        if not self._is_namedtuple_factory(callee):
            return return_value
        if not isinstance(return_value, KnownValue):
            return return_value
        runtime_class = return_value.val
        if not is_namedtuple_class(runtime_class):
            return return_value
        synthetic = SyntheticClassObjectValue(
            runtime_class.__name__,
            TypedValue(
                self._get_synthetic_class_fq_name_from_name(runtime_class.__name__)
            ),
            base_classes=(TypedValue(tuple),),
        )
        synthetic.class_attributes["%runtime_class"] = return_value
        synthetic.method_attributes.update(
            name
            for name, attr in runtime_class.__dict__.items()
            if callable(attr) or isinstance(attr, (staticmethod, classmethod, property))
        )
        for name, attr in runtime_class.__dict__.items():
            synthetic.class_attributes[name] = KnownValue(attr)
        self.checker.register_synthetic_class(synthetic)
        return synthetic

    def signature_from_value(
        self, value: Value, node: ast.AST | None = None
    ) -> MaybeSignature:
        def get_call_attribute(value: Value) -> Value:
            return self.get_attribute(
                Composite(value),
                "__call__",
                node,
                ignore_none=self.options.get_value_for(IgnoreNoneAttributes),
            )

        return self.checker.signature_from_value(
            value,
            get_return_override=self.get_local_return_value,
            get_call_attribute=get_call_attribute,
        )

    # Match statements

    def visit_Match(self, node: ast.Match) -> None:
        subject = self.composite_from_node(node.subject)
        patma_visitor = PatmaVisitor(self)
        with override(self, "match_subject", subject):
            constraints_to_apply = []
            subscopes = []
            for case in node.cases:
                with self.scopes.subscope() as case_scope:
                    for constraint in constraints_to_apply:
                        self.add_constraint(case, constraint)
                    self.match_subject = self.match_subject._replace(
                        value=constrain_value(
                            self.match_subject.value,
                            AndConstraint.make(constraints_to_apply),
                            ctx=self,
                        )
                    )

                    pattern_constraint = patma_visitor.visit(case.pattern)
                    constraints = [pattern_constraint]
                    self.add_constraint(case.pattern, pattern_constraint)
                    if case.guard is not None:
                        _, guard_constraint = self.constraint_from_condition(case.guard)
                        self.add_constraint(case.guard, guard_constraint)
                        constraints.append(guard_constraint)

                    constraints_to_apply.append(
                        AndConstraint.make(constraints).invert()
                    )
                    self._generic_visit_list(case.body)
                    subscopes.append(case_scope)

                self.yield_checker.reset_yield_checks()

            self.match_subject = self.match_subject._replace(
                value=constrain_value(
                    self.match_subject.value,
                    AndConstraint.make(constraints_to_apply),
                    ctx=self,
                )
            )

            if self.match_subject.value is NO_RETURN_VALUE:
                self._set_name_in_scope(LEAVES_SCOPE, node, NO_RETURN_VALUE)
            else:
                with self.scopes.subscope() as else_scope:
                    for constraint in constraints_to_apply:
                        self.add_constraint(node, constraint)
                    subscopes.append(else_scope)
            self.scopes.combine_subscopes(subscopes)

    # Attribute checking

    def _record_class_examined(self, cls: type) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_class_examined(cls)

    def _record_type_has_dynamic_attrs(self, typ: type) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_type_has_dynamic_attrs(typ)

    def _record_type_attr_set(
        self, typ: type, attr_name: str, node: ast.AST, value: Value
    ) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_set(typ, attr_name, node, value)

    def _record_synthetic_attr_set(
        self, node: ast.Attribute, root_value: Value
    ) -> None:
        if not self._is_checking() or self.being_assigned is None:
            return
        if not self._is_current_method_receiver_node(node.value):
            return
        class_key = self._class_key_for_attribute_target(node, root_value)
        if class_key is None or class_key != self.current_class_key:
            return
        if self._is_class_object_attribute_root(root_value) is False:
            class_type = self.checker.make_type_object(class_key)
            if class_type.is_protocol:
                if node.attr not in class_type.protocol_members:
                    self._show_error_if_checking(
                        node,
                        "Protocol members cannot be defined via assignment to self",
                        error_code=ErrorCode.invalid_annotation,
                    )
                # Protocol members must come from the class body (PEP 544), so don't
                # synthesize new members from method-body self assignments.
                return
        synthetic_class = self.checker.get_synthetic_class(class_key)
        if synthetic_class is None:
            return
        synthetic_class.class_attributes[node.attr] = self.being_assigned

    def _record_type_attr_read(self, typ: type, attr_name: str, node: ast.AST) -> None:
        if self.attribute_checker is not None:
            self.attribute_checker.record_attribute_read(typ, attr_name, node, self)

    def _maybe_get_attr_value(self, typ: type, attr_name: str) -> Value:
        if self.attribute_checker is not None:
            return self.attribute_checker.get_attribute_value(typ, attr_name)
        else:
            return AnyValue(AnySource.inference)

    def _is_dataclass_initvar_attribute(self, typ: type, attr_name: str) -> bool:
        synthetic = self.checker.get_synthetic_class(typ)
        if synthetic is not None:
            initvar_names = synthetic.class_attributes.get("%dataclass_initvar_fields")
            if isinstance(initvar_names, KnownValue) and isinstance(
                initvar_names.val, (set, frozenset, tuple, list)
            ):
                if attr_name in initvar_names.val:
                    return True
        try:
            annotations = typ.__annotations__
        except Exception:
            return False
        annotation = annotations.get(attr_name)
        if annotation is None:
            return False
        if annotation is dataclasses.InitVar:
            return True
        origin = get_origin(annotation)
        if origin is dataclasses.InitVar:
            return True
        if isinstance(annotation, str):
            return "InitVar" in annotation
        return False

    # Finding unused objects

    def _maybe_record_usage(
        self, module_or_class: object, attribute: str, value: Value
    ) -> None:
        if self.unused_finder is None:
            return

        # in this case class isn't available
        if self.scopes.scope_type() == ScopeType.function_scope and self._is_checking():
            return

        if isinstance(value, KnownValue) and self.current_class is not None:
            # exclude calls within a class (probably in super calls)
            if value.val is self.current_class:
                return

            inner = UnwrapClass.unwrap(value.val, self.options)
            if inner is self.current_class:
                return

        if self.module is not None and isinstance(module_or_class, types.ModuleType):
            self.unused_finder.record(module_or_class, attribute, self.module.__name__)

    @classmethod
    def _get_argument_parser(cls) -> ArgumentParser:
        parser = super()._get_argument_parser()
        parser.add_argument(
            "--find-unused",
            action="store_true",
            default=False,
            help="Find unused functions and classes",
        )
        parser.add_argument(
            "--find-unused-attributes",
            action="store_true",
            default=False,
            help="Find unused class attributes",
        )
        parser.add_argument(
            "--config-file",
            type=Path,
            help="Path to a pyproject.toml configuration file",
        )
        parser.add_argument(
            "--display-options",
            action="store_true",
            default=False,
            help="Display the options used for this check, then exit",
        )
        add_arguments(parser)
        return parser

    @classmethod
    def get_description_for_error_code(cls, error_code: Error) -> str:
        return error_code.description

    @classmethod
    def get_default_directories(
        cls, checker: Checker, **kwargs: Any
    ) -> tuple[str, ...]:
        paths = checker.options.get_value_for(Paths)
        return tuple(str(path) for path in paths)

    @classmethod
    def _get_default_settings(cls) -> dict[node_visitor.ErrorCodeInstance, bool] | None:
        return {}

    @classmethod
    def prepare_constructor_kwargs(
        cls, kwargs: Mapping[str, Any], extra_options: Sequence[ConfigOption[Any]] = ()
    ) -> Mapping[str, Any]:
        kwargs = dict(kwargs)
        instances = [*extra_options]
        if "settings" in kwargs:
            for error_code, value in kwargs["settings"].items():
                option_cls = ConfigOption.registry[error_code.name]
                instances.append(option_cls(value, from_command_line=True))
        files = kwargs.pop("files", [])
        if files:
            instances.append(Paths(files, from_command_line=True))
        for name, option_cls in ConfigOption.registry.items():
            if not option_cls.should_create_command_line_option:
                continue
            if name not in kwargs:
                continue
            value = kwargs.pop(name)
            instances.append(option_cls(value, from_command_line=True))
        config_file = kwargs.pop("config_file", None)
        if config_file is None:
            config_filename = cls.config_filename
            if config_filename is not None:
                module_path = Path(sys.modules[cls.__module__].__file__).parent
                config_file = module_path / config_filename
        options = Options.from_option_list(instances, config_file_path=config_file)
        if kwargs.pop("display_options", False):
            options.display()
            sys.exit(0)
        kwargs.setdefault("checker", Checker(raw_options=options))
        patch_typing_overload()
        return kwargs

    def is_enabled(self, error_code: node_visitor.ErrorCodeInstance) -> bool:
        if not isinstance(error_code, Error):
            return False
        return self.options.is_error_code_enabled(error_code)

    @classmethod
    def perform_final_checks(
        cls, kwargs: Mapping[str, Any]
    ) -> list[node_visitor.Failure]:
        return kwargs["checker"].perform_final_checks()

    @classmethod
    def _run_on_files(
        cls,
        files: list[str],
        *,
        checker: Checker,
        find_unused: bool = False,
        find_unused_attributes: bool = False,
        attribute_checker: ClassAttributeChecker | None = None,
        unused_finder: UnusedObjectFinder | None = None,
        **kwargs: Any,
    ) -> list[node_visitor.Failure]:
        attribute_checker_enabled = checker.options.is_error_code_enabled_anywhere(
            ErrorCode.attribute_is_never_set
        )
        if attribute_checker is None:
            inner_attribute_checker_obj = attribute_checker = ClassAttributeChecker(
                enabled=attribute_checker_enabled,
                should_check_unused_attributes=find_unused_attributes,
                should_serialize=kwargs.get("parallel", False),
                options=checker.options,
                ts_finder=checker.ts_finder,
            )
        else:
            inner_attribute_checker_obj = contextlib.nullcontext()
        if unused_finder is None:
            unused_finder = UnusedObjectFinder(
                checker.options,
                enabled=find_unused or checker.options.get_value_for(EnforceNoUnused),
                print_output=False,
            )
        with inner_attribute_checker_obj as inner_attribute_checker:
            with unused_finder as inner_unused_finder:
                all_failures = super()._run_on_files(
                    files,
                    attribute_checker=(
                        attribute_checker
                        if attribute_checker is not None
                        else inner_attribute_checker
                    ),
                    unused_finder=inner_unused_finder,
                    checker=checker,
                    **kwargs,
                )
        if unused_finder is not None:
            for unused_object in unused_finder.get_unused_objects():
                # Maybe we should switch to a shared structured format for errors
                # so we can share code with normal errors better.
                failure = str(unused_object)
                print(unused_object)
                all_failures.append(
                    {
                        "filename": node_visitor.UNUSED_OBJECT_FILENAME,
                        "absolute_filename": node_visitor.UNUSED_OBJECT_FILENAME,
                        "message": failure + "\n",
                        "description": failure,
                    }
                )
        if attribute_checker is not None:
            all_failures += attribute_checker.all_failures
        return all_failures

    @classmethod
    def check_file_in_worker(
        cls,
        filename: str,
        attribute_checker: ClassAttributeChecker | None = None,
        **kwargs: Any,
    ) -> tuple[list[node_visitor.Failure], Any]:
        failures = cls.check_file(
            filename, attribute_checker=attribute_checker, **kwargs
        )
        return failures, attribute_checker

    @classmethod
    def merge_extra_data(
        cls,
        extra_data: Any,
        attribute_checker: ClassAttributeChecker | None = None,
        **kwargs: Any,
    ) -> None:
        if attribute_checker is None:
            return
        for checker in extra_data:
            if checker is None:
                continue
            for serialized, attrs in checker.attributes_read.items():
                attribute_checker.attributes_read[serialized] += attrs
            for serialized, attrs in checker.attributes_set.items():
                attribute_checker.attributes_set[serialized] |= attrs
            for serialized, attrs in checker.attribute_values.items():
                for attr_name, value in attrs.items():
                    attribute_checker.merge_attribute_value(
                        serialized, attr_name, value
                    )
            attribute_checker.modules_examined |= checker.modules_examined
            attribute_checker.classes_examined |= checker.modules_examined
            attribute_checker.types_with_dynamic_attrs |= (
                checker.types_with_dynamic_attrs
            )
            attribute_checker.filename_to_visitor.update(checker.filename_to_visitor)

    # Protocol compliance
    def visit_expression(self, node: ast.AST) -> Value:
        return self.visit(node)


def _maybe_normalize_filename(filename: str) -> str:
    try:
        return str(Path(filename).resolve())
    except OSError:
        return os.path.abspath(filename)


def _classvar_names_from_mapping(attributes: Mapping[str, Value]) -> set[str]:
    classvars = attributes.get("%classvars")
    if isinstance(classvars, KnownValue) and isinstance(
        classvars.val, (set, frozenset, tuple, list)
    ):
        return {item for item in classvars.val if isinstance(item, str)}
    return set()


def _instance_only_names_from_mapping(attributes: Mapping[str, Value]) -> set[str]:
    raw = attributes.get("%instance_only_annotations")
    if isinstance(raw, KnownValue) and isinstance(
        raw.val, (set, frozenset, tuple, list)
    ):
        return {item for item in raw.val if isinstance(item, str)}
    return set()


def _is_newtype_base_value(base_value: Value) -> bool:
    for subval in flatten_values(replace_fallback(base_value)):
        if not isinstance(subval, KnownValue):
            continue
        if isinstance(subval.val, type):
            continue
        if safe_hasattr(subval.val, "__supertype__"):
            return True
    return False


def _is_type_alias_base_value(base_value: Value) -> bool:
    for subval in flatten_values(base_value, unwrap_annotated=True):
        if isinstance(subval, TypeAliasValue):
            return True
        if isinstance(subval, KnownValue) and (
            is_typing_name(subval.val, "TypeAliasType")
            or is_instance_of_typing_name(subval.val, "TypeAliasType")
        ):
            return True
    return False


def _is_typeddict_marker_base(base_value: Value) -> bool:
    return isinstance(base_value, KnownValue) and is_typing_name(
        base_value.val, "TypedDict"
    )


def _is_namedtuple_marker_base(base_value: Value) -> bool:
    return isinstance(base_value, KnownValue) and is_typing_name(
        base_value.val, "NamedTuple"
    )


def _populate_fallback_runtime_class(
    runtime_class: type, class_scope_values: Mapping[str, Value]
) -> None:
    for name, value in class_scope_values.items():
        if name.startswith("%") or name in runtime_class.__dict__:
            continue
        if not isinstance(value, KnownValue):
            continue
        try:
            setattr(runtime_class, name, value.val)
        except Exception:
            continue


def _mangle_private_enum_name(class_name: str, attr_name: str) -> str | None:
    if not attr_name.startswith("__") or attr_name.endswith("__"):
        return None
    return f"_{class_name}{attr_name}"


def _enum_ignore_names(value: Value | None) -> set[str]:
    if value is None:
        return set()
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        member_names = [_enum_ignore_names(subval) for subval in value.vals]
        if not member_names:
            return set()
        output = member_names[0].copy()
        for names in member_names[1:]:
            output &= names
        return output
    if isinstance(value, IntersectionValue):
        output: set[str] = set()
        for subval in value.vals:
            output |= _enum_ignore_names(subval)
        return output
    if isinstance(value, KnownValue):
        if isinstance(value.val, str):
            return set(value.val.split())
        if isinstance(value.val, (list, tuple, set)):
            return {elt for elt in value.val if isinstance(elt, str)}
    if isinstance(value, SequenceValue):
        members = value.get_member_sequence()
        if members is None:
            return set()
        output: set[str] = set()
        for member in members:
            member = replace_fallback(member)
            if isinstance(member, KnownValue) and isinstance(member.val, str):
                output.add(member.val)
            else:
                return set()
        return output
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return set()
    assert_never(value)
    return set()


def _iter_enum_assignment_candidates(
    node: ast.ClassDef,
) -> Iterable[tuple[str, ast.AST]]:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    yield target.id, statement
        elif isinstance(statement, ast.AnnAssign):
            if isinstance(statement.target, ast.Name):
                yield statement.target.id, statement
        elif isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            yield statement.name, statement


def _enum_statement_member_decorators(statement: ast.AST) -> tuple[bool, bool]:
    if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return False, False

    forced_member = False
    forced_nonmember = False
    for decorator in statement.decorator_list:
        target = decorator
        if isinstance(target, ast.Call):
            target = target.func
        if isinstance(target, ast.Name):
            if target.id == "member":
                forced_member = True
            elif target.id == "nonmember":
                forced_nonmember = True
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "enum"
        ):
            if target.attr == "member":
                forced_member = True
            elif target.attr == "nonmember":
                forced_nonmember = True
    return forced_member, forced_nonmember


def _unwrap_enum_member_wrapper(value: Value) -> tuple[Value, bool, bool]:
    value = replace_fallback(value)
    member_cls = getattr(enum, "member", None)
    nonmember_cls = getattr(enum, "nonmember", None)

    if (
        member_cls is not None
        and isinstance(value, GenericValue)
        and value.typ is member_cls
    ):
        if value.args:
            return value.args[0], True, False
        return AnyValue(AnySource.inference), True, False
    if (
        nonmember_cls is not None
        and isinstance(value, GenericValue)
        and value.typ is nonmember_cls
    ):
        if value.args:
            return value.args[0], False, True
        return AnyValue(AnySource.inference), False, True

    if (
        member_cls is not None
        and isinstance(value, KnownValue)
        and isinstance(value.val, member_cls)
    ):
        return KnownValue(value.val.value), True, False
    if (
        nonmember_cls is not None
        and isinstance(value, KnownValue)
        and isinstance(value.val, nonmember_cls)
    ):
        return KnownValue(value.val.value), False, True
    return value, False, False


def _is_nonmember_enum_assignment_value(
    value: Value, checker: "NameCheckVisitor"
) -> bool:
    """Whether an Enum class-body assignment should stay a non-member.

    Enum ignores callables and descriptor-like objects (e.g. ``property``),
    so we check assignability to both callable and descriptor protocols.
    """
    if value in (VOID, UNINITIALIZED_VALUE) or isinstance(value, ReferencingValue):
        return False
    value = replace_fallback(value)
    callable_or_descriptor = TypedValue(collections.abc.Callable) | TypedValue(
        _SupportsDescriptorGet
    )
    return is_assignable(callable_or_descriptor, value, checker)


def _runtime_object_for_enum_member(value: Value) -> object:
    if isinstance(value, InputSigValue):
        return object()
    if value in (VOID, UNINITIALIZED_VALUE) or isinstance(value, ReferencingValue):
        return object()
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        member_values = [
            _runtime_object_for_enum_member(subval) for subval in value.vals
        ]
        first_value = member_values[0]
        if all(member_value is first_value for member_value in member_values):
            return first_value
        return object()
    if isinstance(value, IntersectionValue):
        member_values = [
            _runtime_object_for_enum_member(subval) for subval in value.vals
        ]
        first_value = member_values[0]
        if all(member_value is first_value for member_value in member_values):
            return first_value
        return object()
    if isinstance(value, KnownValue):
        return value.val
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return object()
    assert_never(value)


def _extract_keywords(keywords: Sequence[ast.keyword]) -> dict[str, ast.expr]:
    keywords_dict = {}
    for keyword in keywords:
        if keyword.arg is None:
            continue
        keywords_dict[keyword.arg] = keyword.value
    return keywords_dict


def _extract_bool(expr: ast.expr | None, default: bool) -> bool:
    if expr is None:
        return default
    value = _get_bool_literal(expr)
    if value is None:
        return default
    return value


def _get_bool_literal(node: ast.AST) -> bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return None


def _is_known_decorator(value: Value, decorator: object) -> bool:
    if value is UNINITIALIZED_VALUE:
        return False
    if isinstance(value, (ReferencingValue, InputSigValue)):
        return False
    value = replace_fallback(value)
    return isinstance(value, KnownValue) and value.val is decorator


def _is_dataclass_decorator_value(value: Value) -> bool:
    return _is_known_decorator(value, dataclasses.dataclass)


def _merge_dataclass_transform_infos(
    infos: Sequence[DataclassTransformInfo],
) -> DataclassTransformInfo:

    def _merge_bool(values: Sequence[bool | None]) -> bool:
        bool_values = {value for value in values if isinstance(value, bool)}
        return next(iter(bool_values))

    field_specifiers: list[Value] = []
    for info in infos:
        for field_specifier in info.field_specifiers:
            if field_specifier not in field_specifiers:
                field_specifiers.append(field_specifier)
    return DataclassTransformInfo(
        eq_default=_merge_bool([info.eq_default for info in infos]),
        frozen_default=_merge_bool([info.frozen_default for info in infos]),
        kw_only_default=_merge_bool([info.kw_only_default for info in infos]),
        order_default=_merge_bool([info.order_default for info in infos]),
        field_specifiers=tuple(field_specifiers),
    )


def _synthesize_dataclass_hash_attribute(
    semantics: DataclassInfo | None,
) -> Value | None:
    if semantics is None:
        return None
    if semantics.unsafe_hash is True:
        return AnyValue(AnySource.inference)
    if semantics.eq is False:
        return AnyValue(AnySource.inference)
    if (
        semantics.eq is True
        and semantics.frozen is False
        and semantics.unsafe_hash is False
    ):
        return KnownValue(None)
    if semantics.eq is True and semantics.frozen is True:
        return AnyValue(AnySource.inference)
    return None


def _class_body_defines_slots(node: ast.ClassDef) -> bool:
    for statement in node.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "__slots__":
                    return True
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            if statement.target.id == "__slots__":
                return True
    return False


def _known_string_sequence_values(value: Value | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        members = [_known_string_sequence_values(subval) for subval in value.vals]
        if not members or any(member is None for member in members):
            return None
        first = members[0]
        assert first is not None
        if all(member == first for member in members[1:]):
            return first
        return None
    if isinstance(value, IntersectionValue):
        members = [
            member
            for subval in value.vals
            if (member := _known_string_sequence_values(subval)) is not None
        ]
        if not members:
            return None
        first = members[0]
        if all(member == first for member in members[1:]):
            return first
        return None
    if isinstance(value, KnownValue):
        raw = value.val
        if isinstance(raw, str):
            return (raw,)
        if isinstance(raw, (tuple, list, set, frozenset)):
            return tuple(item for item in raw if isinstance(item, str))
        return None
    if isinstance(value, SequenceValue):
        members = value.get_member_sequence()
        if members is None:
            return None
        output: list[str] = []
        for member in members:
            member = replace_fallback(member)
            if isinstance(member, KnownValue) and isinstance(member.val, str):
                output.append(member.val)
            else:
                return None
        return tuple(output)
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return None
    assert_never(value)
    return None


def _normalize_slot_names(raw_names: Iterable[str]) -> tuple[tuple[str, ...], bool]:
    names: list[str] = []
    has_dict = False
    for name in raw_names:
        if name == "__dict__":
            has_dict = True
            continue
        if name == "__weakref__":
            continue
        names.append(name)
    return tuple(names), has_dict


def _record_dataclass_slots_flag(
    synthetic_class: SyntheticClassObjectValue, semantics: DataclassInfo | None
) -> None:
    if semantics is None:
        synthetic_class.class_attributes.pop("%dataclass_slots", None)
        synthetic_class.class_attributes.pop("%dataclass_init", None)
        synthetic_class.class_attributes.pop("%dataclass_match_args", None)
        return
    synthetic_class.class_attributes["%dataclass_slots"] = KnownValue(semantics.slots)
    synthetic_class.class_attributes["%dataclass_init"] = KnownValue(semantics.init)
    synthetic_class.class_attributes["%dataclass_match_args"] = KnownValue(
        semantics.match_args
    )


def _is_dataclass_kw_only_marker_value(value: Value) -> bool:
    marker = getattr(dataclasses, "KW_ONLY", None)
    if marker is None:
        return False
    value = replace_fallback(value)
    if isinstance(value, AnnotatedValue):
        return _is_dataclass_kw_only_marker_value(value.value)
    if isinstance(value, MultiValuedValue):
        return any(_is_dataclass_kw_only_marker_value(subval) for subval in value.vals)
    return isinstance(value, KnownValue) and value.val is marker


def _value_matches_dataclass_field_specifier(
    value: Value, field_specifier: Value
) -> bool:
    value = replace_fallback(value)
    field_specifier = replace_fallback(field_specifier)
    if isinstance(value, AnnotatedValue):
        return _value_matches_dataclass_field_specifier(value.value, field_specifier)
    if isinstance(field_specifier, AnnotatedValue):
        return _value_matches_dataclass_field_specifier(value, field_specifier.value)
    if isinstance(value, MultiValuedValue):
        return any(
            _value_matches_dataclass_field_specifier(subval, field_specifier)
            for subval in value.vals
        )
    if isinstance(field_specifier, MultiValuedValue):
        return any(
            _value_matches_dataclass_field_specifier(value, subval)
            for subval in field_specifier.vals
        )
    if isinstance(value, KnownValue) and isinstance(field_specifier, KnownValue):
        return value.val is field_specifier.val
    if isinstance(value, SyntheticClassObjectValue) and isinstance(
        field_specifier, SyntheticClassObjectValue
    ):
        return value.class_type == field_specifier.class_type
    return value == field_specifier


def _arguments_from_call_composites(
    args: Sequence[Composite], keywords: Sequence[tuple[str | None, Composite]]
) -> list[Argument]:
    return [
        (
            (Composite(arg.value.root, arg.varname, arg.node), ARGS)
            if (
                isinstance(arg.value, PartialValue)
                and arg.value.operation is PartialValueOperation.UNPACK
            )
            else (arg, None)
        )
        for arg in args
    ] + [
        (value, KWARGS) if keyword is None else (value, keyword)
        for keyword, value in keywords
    ]


def _dataclass_field_bound_arg(
    bound_args: BoundArgs, name: str, *, include_default: bool = True
) -> Value | None:
    if name not in bound_args:
        return None
    if not include_default and bound_args[name][0] is type_evaluation.DEFAULT:
        return None
    return bound_args[name][1].value


def _dataclass_field_bound_bool_arg(
    bound_args: BoundArgs, name: str, *, include_default: bool = True
) -> bool | None:
    value = _dataclass_field_bound_arg(
        bound_args, name, include_default=include_default
    )
    if isinstance(value, KnownValue) and isinstance(value.val, bool):
        return value.val
    return None


def _dataclass_field_bound_str_arg(
    bound_args: BoundArgs, name: str, *, include_default: bool = True
) -> str | None:
    value = _dataclass_field_bound_arg(
        bound_args, name, include_default=include_default
    )
    if isinstance(value, KnownValue) and isinstance(value.val, str):
        return value.val
    return None


def _is_absent_dataclass_default_value(value: Value) -> bool:
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        return all(_is_absent_dataclass_default_value(subval) for subval in value.vals)
    if isinstance(value, IntersectionValue):
        return any(_is_absent_dataclass_default_value(subval) for subval in value.vals)
    if isinstance(value, KnownValue):
        return value.val is Ellipsis or value.val is dataclasses.MISSING
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return False
    assert_never(value)
    return False


def _callable_return_type_from_signature(
    signature: MaybeSignature, *, checker: "NameCheckVisitor"
) -> Value | None:
    if isinstance(signature, BoundMethodSignature):
        signature = signature.get_signature(ctx=checker)
    if isinstance(signature, Signature):
        return signature.return_value
    if isinstance(signature, OverloadedSignature):
        if not signature.signatures:
            return None
        return unite_values(*(sig.return_value for sig in signature.signatures))
    return None


def _slot_names_from_runtime_slots(value: object) -> tuple[str, ...] | None:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(item for item in value if isinstance(item, str))
    return None


def _get_dataclass_post_init_node(
    node: ast.ClassDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for stmt in node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name == "__post_init__":
                return stmt
    return None


def _compose_variance_polarity(polarity: int, variance: Variance) -> int:
    if polarity == 0 or variance is Variance.INVARIANT:
        return 0
    if variance is Variance.COVARIANT:
        return polarity
    return -polarity


def _record_variance_polarity(used_polarities: set[int], polarity: int) -> None:
    if polarity == 0:
        used_polarities.update({-1, 1})
    else:
        used_polarities.add(polarity)


def _variance_is_compatible_with_usage(
    variance: Variance, used_polarities: set[int]
) -> bool:
    if not used_polarities or variance is Variance.INVARIANT:
        return True
    if variance is Variance.COVARIANT:
        return -1 not in used_polarities
    return 1 not in used_polarities


def _is_variance_declaration_arg(arg: Value) -> bool:
    if isinstance(arg, TypeVarValue):
        return True
    if isinstance(arg, InputSigValue):
        return False
    if arg in (VOID, UNINITIALIZED_VALUE) or isinstance(arg, ReferencingValue):
        return False
    arg = replace_fallback(arg)
    if isinstance(arg, MultiValuedValue):
        return all(_is_variance_declaration_arg(subval) for subval in arg.vals)
    if isinstance(arg, IntersectionValue):
        return any(_is_variance_declaration_arg(subval) for subval in arg.vals)
    if isinstance(arg, KnownValue):
        type_param = arg.val
        return (
            is_instance_of_typing_name(type_param, "TypeVar")
            or is_instance_of_typing_name(type_param, "TypeVarTuple")
            or is_instance_of_typing_name(type_param, "ParamSpec")
        )
    if isinstance(
        arg,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return False
    assert_never(arg)


def _is_variance_declaration_base(base_value: Value) -> bool:
    subvals = list(flatten_values(replace_fallback(base_value)))
    if not subvals:
        return False
    for subval in subvals:
        if isinstance(subval, SyntheticClassObjectValue):
            subval = subval.class_type
        if isinstance(subval, GenericValue):
            typ: object = subval.typ
            type_args: Sequence[Value] = subval.args
        elif isinstance(subval, TypedValue):
            typ = subval.typ
            type_args = ()
        elif isinstance(subval, KnownValue):
            typ = subval.val
            type_args = ()
        else:
            return False
        if not (is_typing_name(typ, "Generic") or is_typing_name(typ, "Protocol")):
            return False
        if type_args and not all(
            _is_variance_declaration_arg(arg) for arg in type_args
        ):
            return False
    return True


def _type_param_value_from_value(value: Value) -> TypeVarValue | None:
    if isinstance(value, InputSigValue) and isinstance(value.input_sig, ParamSpecSig):
        return TypeVarValue(value.input_sig.param_spec)
    if isinstance(value, InputSigValue):
        return None
    if value in (VOID, UNINITIALIZED_VALUE) or isinstance(value, ReferencingValue):
        return None
    if isinstance(value, TypeVarValue):
        return value
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        type_params = [
            type_param
            for subval in value.vals
            if (type_param := _type_param_value_from_value(subval)) is not None
        ]
        if not type_params:
            return None
        first = type_params[0]
        if all(type_param.typevar is first.typevar for type_param in type_params):
            return first
        return None
    if isinstance(value, IntersectionValue):
        result: TypeVarValue | None = None
        for subval in value.vals:
            type_param = _type_param_value_from_value(subval)
            if type_param is None:
                continue
            if result is None:
                result = type_param
            elif result.typevar is not type_param.typevar:
                return None
        return result
    if isinstance(value, KnownValue):
        is_typevartuple = is_instance_of_typing_name(value.val, "TypeVarTuple")
        if (
            is_instance_of_typing_name(value.val, "TypeVar")
            or is_typevartuple
            or is_instance_of_typing_name(value.val, "ParamSpec")
        ):
            return TypeVarValue(
                value.val,
                variance=get_typevar_variance(value.val),
                is_typevartuple=is_typevartuple,
            )
        return None
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return None
    assert_never(value)


def _count_starred_type_param_args(slice_node: ast.AST) -> int:
    if isinstance(slice_node, ast.Tuple):
        return sum(isinstance(elt, ast.Starred) for elt in slice_node.elts)
    return int(isinstance(slice_node, ast.Starred))


def _is_protocol_base(base_value: Value) -> bool:
    if isinstance(base_value, InputSigValue):
        return False
    if base_value in (VOID, UNINITIALIZED_VALUE) or isinstance(
        base_value, ReferencingValue
    ):
        return False
    base_value = replace_fallback(base_value)
    if isinstance(base_value, MultiValuedValue):
        return any(_is_protocol_base(subval) for subval in base_value.vals)
    if isinstance(base_value, IntersectionValue):
        return any(_is_protocol_base(subval) for subval in base_value.vals)
    if isinstance(base_value, SyntheticClassObjectValue):
        return _is_protocol_base(base_value.class_type)
    if isinstance(base_value, KnownValue):
        return is_typing_name(base_value.val, "Protocol")
    if isinstance(base_value, TypedValue):
        return is_typing_name(base_value.typ, "Protocol")
    if isinstance(
        base_value,
        (
            AnyValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return False
    assert_never(base_value)


def _mangle_class_attribute_name(class_name: str, attribute_name: str) -> str:
    if attribute_name.startswith("__") and not attribute_name.endswith("__"):
        return f"_{class_name}{attribute_name}"
    return attribute_name


def _return_annotation_contains_self(return_annotation: Value | None) -> bool:
    if return_annotation is None:
        return False
    for subvalue in return_annotation.walk_values():
        if isinstance(subvalue, TypeVarValue) and subvalue.typevar is SelfT:
            return True
    return False


def _is_dataclass_classvar_final(expr: AnnotationExpr) -> bool:
    qualifier_order = [
        qualifier
        for qualifier, _ in expr.qualifiers
        if qualifier in {Qualifier.Final, Qualifier.ClassVar}
    ]
    return qualifier_order == [Qualifier.Final, Qualifier.ClassVar]


def _method_decorator_kind(*, decorator_kinds: Container[FunctionDecorator]) -> str:
    return FunctionDecorator.method_kind_for(decorator_kinds)


def _is_typealiastype_value(value: Value) -> bool:
    for subval in flatten_values(value, unwrap_annotated=True):
        if isinstance(subval, KnownValue) and (
            is_typing_name(subval.val, "TypeAliasType")
            or is_instance_of_typing_name(subval.val, "TypeAliasType")
        ):
            return True
    return False


def _get_runtime_type_alias_value_node(node: ast.Call) -> ast.AST | None:
    if len(node.args) >= 2:
        return node.args[1]
    for keyword in node.keywords:
        if keyword.arg == "value":
            return keyword.value
    return None


def _runtime_type_alias_self_reference(value_node: ast.AST, name: str) -> bool:
    return any(
        isinstance(subnode, ast.Name)
        and isinstance(subnode.ctx, ast.Load)
        and subnode.id == name
        for subnode in ast.walk(value_node)
    )


def _runtime_type_alias_declared_type_params(
    params: Sequence[TypeVarLike | TypeVarValue],
) -> Sequence[TypeVarValue]:
    declared: list[TypeVarValue] = []
    for param in params:
        if isinstance(param, TypeVarValue):
            declared.append(param)
        elif is_instance_of_typing_name(param, "TypeVarTuple") or is_typing_name(
            type(param), "TypeVarTuple"
        ):
            declared.append(TypeVarValue(param, is_typevartuple=True))
        else:
            declared.append(TypeVarValue(param))
    return declared


def build_stacked_scopes(
    module: types.ModuleType | None,
    simplification_limit: int | None = None,
    *,
    options: Options,
) -> StackedScopes:
    # Build a StackedScopes object.
    # Not part of stacked_scopes.py to avoid a circular dependency.
    if module is None:
        module_vars = {"__name__": TypedValue(str), "__file__": TypedValue(str)}
    else:
        module_vars = {}
        try:
            annotations = module.__annotations__
        except Exception:
            annotations = {}
        for key, value in module.__dict__.items():
            expr = annotation_expr_from_annotations(
                annotations, key, globals=module.__dict__
            )
            if expr is not None:
                val, _ = expr.maybe_unqualify({Qualifier.TypeAlias, Qualifier.Final})
            else:
                val = None
            if val is None:
                for transformer in options.get_value_for(TransformGlobals):
                    maybe_val = transformer(value)
                    if maybe_val is not None:
                        val = maybe_val
                if val is None:
                    val = KnownValue(value)
            module_vars[key] = val
    return StackedScopes(module_vars, module, simplification_limit=simplification_limit)


def _collect_unguarded_type_alias_refs(value_node: ast.AST) -> set[str]:
    refs: set[str] = set()

    def walk(node: ast.AST, guarded: bool, allow_string_parse: bool = True) -> None:
        if isinstance(node, ast.Subscript):
            walk(node.value, guarded, allow_string_parse)
            walk(node.slice, True, allow_string_parse)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if not guarded:
                refs.add(node.id)
            return
        if (
            allow_string_parse
            and isinstance(node, ast.Constant)
            and isinstance(node.value, str)
        ):
            parse_source = node.value
            if "\n" in parse_source or "\r" in parse_source:
                parse_source = f"({parse_source})"
            try:
                parsed = ast.parse(parse_source, mode="eval")
            except SyntaxError:
                return
            walk(parsed.body, guarded, False)
            return
        for child in ast.iter_child_nodes(node):
            walk(child, guarded, allow_string_parse)

    walk(value_node, False)
    return refs


def _type_param_identity(value: Value) -> object | None:
    if isinstance(value, TypeVarValue):
        return value.typevar
    if isinstance(value, InputSigValue) and isinstance(value.input_sig, ParamSpecSig):
        return value.input_sig.param_spec
    if isinstance(value, InputSigValue):
        return None
    if value in (VOID, UNINITIALIZED_VALUE) or isinstance(value, ReferencingValue):
        return None
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        identities = [
            identity
            for subval in value.vals
            if (identity := _type_param_identity(subval)) is not None
        ]
        if not identities:
            return None
        first = identities[0]
        if all(identity is first for identity in identities[1:]):
            return first
        return None
    if isinstance(value, IntersectionValue):
        identity: object | None = None
        for subval in value.vals:
            sub_identity = _type_param_identity(subval)
            if sub_identity is None:
                continue
            if identity is None:
                identity = sub_identity
            elif sub_identity is not identity:
                return None
        return identity
    if isinstance(value, KnownValue) and (
        is_instance_of_typing_name(value.val, "TypeVar")
        or is_instance_of_typing_name(value.val, "TypeVarTuple")
        or is_instance_of_typing_name(value.val, "ParamSpec")
    ):
        return value.val
    if isinstance(value, KnownValue):
        return None
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return None
    assert_never(value)
    return None


def _is_runtime_literal_index(value: Value) -> bool:
    if isinstance(value, InputSigValue):
        return False
    if value in (VOID, UNINITIALIZED_VALUE) or isinstance(value, ReferencingValue):
        return False
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        return all(_is_runtime_literal_index(subval) for subval in value.vals)
    if isinstance(value, IntersectionValue):
        return any(_is_runtime_literal_index(subval) for subval in value.vals)
    if isinstance(value, KnownValue):
        return True
    if isinstance(value, SequenceValue) and value.typ is tuple:
        members = value.get_member_sequence()
        return members is not None and all(
            isinstance(member, KnownValue) for member in members
        )
    if isinstance(
        value,
        (
            AnyValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypedValue,
            SubclassValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return False
    assert_never(value)
    return False


def _should_use_static_annotation_subscript(value: Value) -> bool:
    if not isinstance(value, KnownValue):
        return False
    root = value.val
    return (
        root is typing.Callable
        or root is collections.abc.Callable
        or is_typing_name(root, "Callable")
        or root is AsynqCallable
    )


def _should_use_static_annotation_subscript_on_import_failure(value: Value) -> bool:
    if not isinstance(value, KnownValue):
        return False
    root = value.val
    return (
        root is typing.Iterable
        or root is collections.abc.Iterable
        or is_typing_name(root, "Iterable")
    )


def _get_task_cls(fn: object) -> type[Any]:
    """Returns the task class for an async function."""
    assert asynq is not None

    if hasattr(fn, "task_cls"):
        cls = fn.task_cls
    elif hasattr(fn, "decorator") and hasattr(fn.decorator, "task_cls"):
        cls = fn.decorator.task_cls
    else:
        cls = asynq.AsyncTask

    if cls is None:  # @async_proxy()
        return asynq.FutureBase
    else:
        return cls


def _all_names_unused(
    elts: Iterable[ast.AST], unused_name_nodes: Container[ast.AST]
) -> bool:
    """Given the left-hand side of an assignment, returns whether all names assigned to are unused.

    elts is a list of assignment nodes, which may contain nested lists or tuples. unused_name_nodes
    is a list of Name nodes corresponding to unused variables.

    """
    for elt in elts:
        if isinstance(elt, (ast.List, ast.Tuple)):
            if not _all_names_unused(elt.elts, unused_name_nodes):
                return False
        if elt not in unused_name_nodes:
            return False
    return True


def _contains_node(elts: Iterable[ast.AST], node: ast.AST) -> bool:
    """Given a list of assignment targets (elts), return whether it contains the given Name node."""
    for elt in elts:
        if isinstance(elt, (ast.List, ast.Tuple)):
            if _contains_node(elt.elts, node):
                return True
        if elt is node:
            return True
    return False


def _static_hasattr(value: object, attr: str) -> bool:
    """Returns whether this value has the given attribute, ignoring __getattr__ overrides."""
    try:
        object.__getattribute__(value, attr)
    except AttributeError:
        return False
    else:
        return True


def _has_annotation_for_attr(typ: type, attr: str) -> bool:
    try:
        return attr in typ.__annotations__
    except Exception:
        # __annotations__ doesn't exist or isn't a dict
        return False


def _is_runtime_classvar_annotation(annotation: object) -> bool:
    origin = get_origin(annotation)
    if is_typing_name(annotation, "ClassVar") or is_typing_name(origin, "ClassVar"):
        return True
    return isinstance(annotation, str) and "ClassVar" in annotation


def _is_typing_alias_value(value: object) -> bool:
    typ = type(value)
    if safe_getattr(typ, "__module__", None) != "typing":
        return False
    name = safe_getattr(typ, "__name__", None)
    return isinstance(name, str) and name in {
        "_GenericAlias",
        "_SpecialGenericAlias",
        "_SpecialForm",
        "_AnnotatedAlias",
        "TypeAliasType",
    }


def _is_asynq_future(value: Value) -> bool:
    if asynq is None:
        return False
    return value.is_type(asynq.FutureBase) or value.is_type(asynq.AsyncTask)


def _extract_definite_value(val: Value) -> bool | None:
    if isinstance(val, AnnotatedValue):
        dv_exts = val.get_metadata_of_type(DefiniteValueExtension)
        for dv_ext in dv_exts:
            return dv_ext.value
    return None


try:
    from pydantic import BaseModel  # static analysis: ignore[import_failed]
except ImportError:

    def _is_safe_pydantic_class(typ: type) -> bool:
        return False

else:

    def _is_safe_pydantic_class(typ: type) -> bool:
        if not issubclass(typ, BaseModel):
            return False
        # Pydantic 1 is unsupported but we shouldn't crash
        if not safe_hasattr(typ, "model_config"):
            return False
        # Pydantic classes have a __getattr__ that looks at two fields,
        # __private_attributes__ and __pydantic_extra__. The latter is
        # used only if this config is set. The former doesn't seem to have
        # a way to detect from the class whether it will be used.
        return typ.model_config.get("extra") != "allow"


def _has_only_known_attributes(ts_finder: TypeshedFinder | None, typ: object) -> bool:
    if not isinstance(typ, type):
        return False
    if _is_safe_pydantic_class(typ) or issubclass(typ, enum.Enum):
        return True
    # Classes that override __getattr__ may have dynamic attributes.
    # We don't check this for pydantic classes because they always have
    # __getattr__, and we don't check it for enums because before 3.11,
    # there was an EnumMeta.__getattr__.
    if hasattr(typ, "__getattr__"):
        return False
    # for namedtuples
    if is_dataclass_type(typ) or issubclass(typ, tuple):
        return True
    if (
        ts_finder is not None
        and ts_finder.has_stubs(typ)
        and not ts_finder.has_attribute(typ, "__getattr__")
        and not ts_finder.has_attribute(typ, "__getattribute__")
        and not attributes.may_have_dynamic_attributes(typ)
        and not hasattr(typ, "__getattr__")
    ):
        return True
    return False
