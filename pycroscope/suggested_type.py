"""

Suggest types for untyped code.

"""

import ast
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

from typing_extensions import assert_never

from .error_code import ErrorCode
from .node_visitor import ErrorContext, Failure
from .safe import safe_getattr, safe_isinstance
from .signature import Signature
from .stacked_scopes import StackedScopes, VisitorState
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    PredicateValue,
    SequenceValue,
    SimpleType,
    SubclassValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    UnboundMethodValue,
    Value,
    VariableNameValue,
    replace_fallback,
    replace_known_sequence_value,
    stringify_object,
    unite_values,
)

CallArgs = Mapping[str, Value]
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class CallableData:
    node: FunctionNode
    ctx: ErrorContext
    sig: Signature
    scopes: StackedScopes
    calls: list[CallArgs] = field(default_factory=list)

    def check(self, ctx: CanAssignContext) -> Iterator[Failure]:
        if not self.calls:
            return
        for param in _extract_params(self.node):
            if param.annotation is not None:
                continue
            sig_param = self.sig.parameters.get(param.arg)
            if sig_param is None or not isinstance(sig_param.annotation, AnyValue):
                continue  # e.g. inferred type for self
            all_values = [call[param.arg] for call in self.calls]
            if sig_param.default is not None:
                all_values.append(sig_param.default)
            all_values = [prepare_type(v, ctx) for v in all_values]
            all_values = [v for v in all_values if not isinstance(v, AnyValue)]
            if not all_values:
                continue
            suggested = unite_values(*all_values)
            if not should_suggest_type(suggested):
                continue
            detail, metadata = display_suggested_type(suggested, self.scopes, ctx)
            failure = self.ctx.show_error(
                param,
                f"Suggested type for parameter {param.arg}",
                ErrorCode.suggested_parameter_type,
                detail=detail,
                # Otherwise we record it twice in tests. We should ultimately
                # refactor error tracking to make it less hacky for things that
                # show errors outside of files.
                save=False,
                extra_metadata=metadata,
            )
            if failure is not None:
                yield failure


@dataclass
class CallableTracker:
    callable_to_data: dict[object, CallableData] = field(default_factory=dict)
    callable_to_calls: dict[object, list[CallArgs]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record_callable(
        self,
        node: FunctionNode,
        callable: object,
        sig: Signature,
        scopes: StackedScopes,
        ctx: ErrorContext,
    ) -> None:
        """Record when we encounter a callable."""
        self.callable_to_data[callable] = CallableData(node, ctx, sig, scopes)

    def record_call(self, callable: object, arguments: Mapping[str, Value]) -> None:
        """Record the actual arguments passed in in a call."""
        self.callable_to_calls[callable].append(arguments)

    def check(self, ctx: CanAssignContext) -> list[Failure]:
        failures = []
        for callable, calls in self.callable_to_calls.items():
            if callable in self.callable_to_data:
                data = self.callable_to_data[callable]
                data.calls += calls
                failures += data.check(ctx)
        return failures


def display_suggested_type(
    value: Value, scopes: StackedScopes, ctx: CanAssignContext
) -> tuple[str, dict[str, Any] | None]:
    value = prepare_type(value, ctx)
    if isinstance(value, MultiValuedValue) and value.vals:
        cae = CanAssignError("Union", [CanAssignError(str(val)) for val in value.vals])
    else:
        cae = CanAssignError(str(value))
    # If the type is simple enough, add extra_metadata for autotyping to apply.
    if isinstance(value, TypedValue) and type(value) is TypedValue:
        # For now, only for exactly TypedValue
        if value.typ is FunctionType:
            # It will end up suggesting builtins.function, which doesn't
            # exist, and we should be using a Callable type instead anyway.
            metadata = None
        else:
            typ_str = stringify_object(value.typ)
            typ_name = typ_str.split(".")[-1]
            scope_value = scopes.get(
                typ_name, None, VisitorState.check_names, can_assign_ctx=ctx
            )
            if isinstance(scope_value, KnownValue) and scope_value.val is value.typ:
                metadata = {"suggested_type": typ_name, "imports": []}
            else:
                imports = []
                if isinstance(value.typ, str):
                    if "." in value.typ:
                        imports.append(value.typ)
                elif safe_getattr(value.typ, "__module__", None) != "builtins":
                    imports.append(typ_str.split(".")[0])
                metadata = {"suggested_type": typ_str, "imports": imports}
    else:
        metadata = None
    return str(cae), metadata


def should_suggest_type(value: Value) -> bool:
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        if not value.vals:
            return False
        if len(value.vals) > 5:
            # Big unions probably aren't useful
            return False
        return all(should_suggest_type(member) for member in value.vals)
    if isinstance(value, IntersectionValue):
        return all(should_suggest_type(member) for member in value.vals)
    return _should_suggest_simple_type(value)


def _should_suggest_simple_type(value: SimpleType) -> bool:
    # Literal[<some function>] isn't useful. In the future we should suggest a
    # Callable type.
    if isinstance(value, KnownValue) and isinstance(value.val, FunctionType):
        return False
    # These generally aren't useful.
    elif isinstance(value, TypedValue) and value.typ in (FunctionType, type):
        return False
    elif isinstance(value, AnyValue):
        return False
    elif isinstance(
        value,
        (
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            PredicateValue,
        ),
    ):
        return False
    elif isinstance(value, (TypedValue, SubclassValue, TypeFormValue, KnownValue)):
        return True
    else:
        assert_never(value)


def prepare_type(value: Value, ctx: CanAssignContext | None = None) -> Value:
    """Simplify a type to turn it into a suggestion."""
    from .input_sig import InputSigValue

    if isinstance(value, InputSigValue):
        return AnyValue(AnySource.inference)
    value = replace_known_sequence_value(value)
    value = replace_fallback(value)
    if isinstance(value, MultiValuedValue):
        vals = [prepare_type(subval, ctx) for subval in value.vals]
        # Throw out Anys
        vals = [val for val in vals if not isinstance(val, AnyValue)]
        type_literals: list[tuple[Value, type]] = []
        rest: list[Value] = []
        for subval in vals:
            if (
                isinstance(subval, SubclassValue)
                and isinstance(subval.typ, TypedValue)
                and safe_isinstance(subval.typ.typ, type)
            ):
                type_literals.append((subval, subval.typ.typ))
            else:
                rest.append(subval)
        if type_literals:
            shared_type = get_shared_type([typ for _, typ in type_literals])
            if shared_type is object:
                type_val = TypedValue(type)
            else:
                type_val = SubclassValue(TypedValue(shared_type))
            return unite_values(type_val, *rest)
        return unite_values(*[v for v, _ in type_literals], *rest)
    if isinstance(value, IntersectionValue):
        prepared_members = [prepare_type(member, ctx) for member in value.vals]
        if ctx is None:
            return IntersectionValue(tuple(prepared_members))
        # Avoid module import cycles at import time.
        from .relations import intersect_multi

        return intersect_multi(prepared_members, ctx)
    return _prepare_simple_type(value, ctx)


def _prepare_simple_type(value: SimpleType, ctx: CanAssignContext | None) -> Value:
    if isinstance(value, SequenceValue):
        if value.typ is tuple:
            members = value.get_member_sequence()
            if members is not None:
                return SequenceValue(
                    tuple, [(False, prepare_type(elt, ctx)) for elt in members]
                )
        return GenericValue(value.typ, [prepare_type(arg, ctx) for arg in value.args])
    elif isinstance(value, (TypedDictValue, CallableValue)):
        return value
    elif isinstance(value, GenericValue):
        # TODO maybe turn DictIncompleteValue into TypedDictValue?
        return GenericValue(value.typ, [prepare_type(arg, ctx) for arg in value.args])
    elif isinstance(value, VariableNameValue):
        return AnyValue(AnySource.unannotated)
    elif isinstance(value, KnownValue):
        if value.val is None:
            return value
        if safe_isinstance(value.val, type):
            return SubclassValue(TypedValue(value.val))
        if callable(value.val):
            return value  # TODO get the signature instead and return a CallableValue?
        return TypedValue(type(value.val))
    elif isinstance(
        value,
        (
            AnyValue,
            TypedValue,
            SubclassValue,
            SyntheticClassObjectValue,
            SyntheticModuleValue,
            UnboundMethodValue,
            TypeFormValue,
            PredicateValue,
        ),
    ):
        return value
    else:
        assert_never(value)


def get_shared_type(types: Sequence[type]) -> type:
    mros = [t.mro() for t in types]
    first, *rest = mros
    rest_sets = [set(mro) for mro in rest]
    for candidate in first:
        if all(candidate in mro for mro in rest_sets):
            return candidate
    assert False, "should at least have found object"


# We exclude *args and **kwargs by default because it's not currently possible
# to give useful types for them.
def _extract_params(
    node: FunctionNode, *, include_var: bool = False
) -> Iterator[ast.arg]:
    yield from node.args.args
    if include_var and node.args.vararg is not None:
        yield node.args.vararg
    yield from node.args.kwonlyargs
    if include_var and node.args.kwarg is not None:
        yield node.args.kwarg
