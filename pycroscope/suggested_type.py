"""

Suggest types for untyped code.

"""

import ast
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import FunctionType
from typing import Any

from typing_extensions import assert_never

from .analysis_lib import Sentinel
from .error_code import ErrorCode
from .node_visitor import ErrorContext, Failure
from .relations import is_assignable
from .safe import safe_getattr, safe_isinstance
from .signature import Signature
from .stacked_scopes import StackedScopes, VisitorState
from .type_evaluation import ARGS, DEFAULT, KWARGS, UNKNOWN
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

BOUND_RECEIVER = Sentinel("bound receiver")
CallSource = object


@dataclass(frozen=True)
class CallArg:
    value: Value
    source: CallSource


CallArgs = Mapping[str, CallArg]
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass
class CallableData:
    node: FunctionNode
    ctx: ErrorContext
    sig: Signature
    scopes: StackedScopes
    receiver_param_name: str | None = None
    owner_type: type | None = None
    method_name: str | None = None
    calls: list[CallArgs] = field(default_factory=list)

    def check(
        self,
        ctx: CanAssignContext,
        *,
        should_check_unused_call_patterns: bool,
        inherited_calls: Sequence[CallArgs] = (),
    ) -> Iterator[Failure]:
        if not self.calls and not inherited_calls:
            return
        if should_check_unused_call_patterns:
            yield from self.check_unused_call_patterns(ctx, inherited_calls)
        yield from self.check_suggested_parameter_types(ctx)

    def check_suggested_parameter_types(
        self, ctx: CanAssignContext
    ) -> Iterator[Failure]:
        for param in _extract_params(self.node):
            if param.annotation is not None:
                continue
            sig_param = self.sig.parameters.get(param.arg)
            if sig_param is None or not isinstance(sig_param.annotation, AnyValue):
                continue  # e.g. inferred type for self
            all_values = [
                call[param.arg].value for call in self.calls if param.arg in call
            ]
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

    def check_unused_call_patterns(
        self, ctx: CanAssignContext, inherited_calls: Sequence[CallArgs]
    ) -> Iterator[Failure]:
        all_calls = [*self.calls, *inherited_calls]
        for param in _extract_params(self.node):
            if param.arg == self.receiver_param_name:
                continue
            sig_param = self.sig.parameters.get(param.arg)
            if sig_param is None:
                continue
            calls = [call[param.arg] for call in all_calls if param.arg in call]
            if not calls:
                continue
            if sig_param.default is not None:
                issue = self._check_optional_parameter(param.arg, calls)
                if issue is not None:
                    yield _make_failure(
                        self.ctx, param, self._make_message(param.arg, issue)
                    )
            if any(call.source in {ARGS, KWARGS, UNKNOWN} for call in calls):
                continue
            value_calls = [call for call in calls if call.source is not BOUND_RECEIVER]
            if not value_calls:
                continue
            annotation = replace_fallback(sig_param.annotation)
            bool_issue = self._check_bool_parameter(annotation, value_calls)
            if bool_issue is not None:
                yield _make_failure(
                    self.ctx, param, self._make_message(param.arg, bool_issue)
                )
                continue
            union_issue = self._check_union_parameter(annotation, value_calls, ctx)
            if union_issue is not None:
                yield _make_failure(
                    self.ctx, param, self._make_message(param.arg, union_issue)
                )

    def _make_message(self, param_name: str, issue: str) -> str:
        return (
            f"Unused call pattern: {stringify_object(self.sig.callable)} "
            f"parameter '{param_name}' {issue}"
        )

    def _check_optional_parameter(
        self, param_name: str, calls: Sequence[CallArg]
    ) -> str | None:
        if any(
            call.source in {ARGS, KWARGS, UNKNOWN, BOUND_RECEIVER} for call in calls
        ):
            return None
        if all(call.source is DEFAULT for call in calls):
            return "is always omitted"
        return None

    def _check_bool_parameter(
        self, annotation: Value, calls: Sequence[CallArg]
    ) -> str | None:
        if not _is_bool_annotation(annotation):
            return None
        seen = set()
        for call in calls:
            if isinstance(call.value, KnownValue) and isinstance(call.value.val, bool):
                seen.add(call.value.val)
            else:
                return None
        if seen == {True}:
            return "is only called with literal True"
        if seen == {False}:
            return "is only called with literal False"
        return None

    def _check_union_parameter(
        self, annotation: Value, calls: Sequence[CallArg], ctx: CanAssignContext
    ) -> str | None:
        if not isinstance(annotation, MultiValuedValue) or len(annotation.vals) < 2:
            return None
        used_members: set[int] = set()
        for call in calls:
            matching = {
                i
                for i, member in enumerate(annotation.vals)
                if is_assignable(member, call.value, ctx)
                or is_assignable(call.value, member, ctx)
            }
            if not matching:
                return None
            used_members |= matching
        if len(used_members) == len(annotation.vals):
            return None
        unused = [
            _describe_call_pattern_value(member)
            for i, member in enumerate(annotation.vals)
            if i not in used_members
        ]
        used = [
            _describe_call_pattern_value(member)
            for i, member in enumerate(annotation.vals)
            if i in used_members
        ]
        if not unused or not used:
            return None
        return f"never receives {' | '.join(unused)} (observed {' | '.join(used)})"


@dataclass
class CallableTracker:
    callable_to_data: dict[object, CallableData] = field(default_factory=dict)
    callable_to_calls: dict[object, list[CallArgs]] = field(
        default_factory=lambda: defaultdict(list)
    )
    method_slot_to_callable: dict[tuple[type, str], object] = field(
        default_factory=dict
    )

    def record_callable(
        self,
        node: FunctionNode,
        callable: object,
        sig: Signature,
        scopes: StackedScopes,
        ctx: ErrorContext,
        receiver_param_name: str | None = None,
        owner_type: type | None = None,
        method_name: str | None = None,
    ) -> None:
        """Record when we encounter a callable."""
        self.callable_to_data[callable] = CallableData(
            node,
            ctx,
            sig,
            scopes,
            receiver_param_name=receiver_param_name,
            owner_type=owner_type,
            method_name=method_name,
        )
        if owner_type is not None and method_name is not None:
            self.method_slot_to_callable[(owner_type, method_name)] = callable

    def record_call(self, callable: object, arguments: CallArgs) -> None:
        """Record the actual arguments passed in in a call."""
        self.callable_to_calls[callable].append(arguments)

    def check(
        self, ctx: CanAssignContext, *, should_check_unused_call_patterns: bool
    ) -> list[Failure]:
        failures = []
        for callable, data in self.callable_to_data.items():
            own_calls = self.callable_to_calls.get(callable, ())
            data.calls += own_calls
            inherited_calls = self._get_inherited_calls(data)
            failures += list(
                data.check(
                    ctx,
                    should_check_unused_call_patterns=should_check_unused_call_patterns,
                    inherited_calls=inherited_calls,
                )
            )
        return failures

    def _get_inherited_calls(self, data: CallableData) -> list[CallArgs]:
        if data.owner_type is None or data.method_name is None:
            return []
        inherited_calls: list[CallArgs] = []
        seen: set[object] = set()
        for base in data.owner_type.__mro__[1:]:
            base_callable = self.method_slot_to_callable.get((base, data.method_name))
            if base_callable is None or base_callable in seen:
                continue
            seen.add(base_callable)
            inherited_calls.extend(self.callable_to_calls.get(base_callable, ()))
        return inherited_calls


def _is_bool_annotation(annotation: Value) -> bool:
    return isinstance(annotation, TypedValue) and annotation.typ is bool


def _describe_call_pattern_value(value: Value) -> str:
    return str(prepare_type(value))


def _make_failure(ctx: ErrorContext, node: ast.AST, description: str) -> Failure:
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    failure: Failure = {
        "description": description,
        "filename": ctx.filename,
        "absolute_filename": os.path.abspath(ctx.filename),
        "message": description + "\n",
    }
    concise_message = ctx.filename
    if lineno is not None:
        failure["lineno"] = lineno
        concise_message += f":{lineno}"
    if col_offset is not None:
        failure["col_offset"] = col_offset
        concise_message += f":{col_offset}"
    failure["concise_message"] = concise_message + f": {description}\n"
    return failure


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
        return GenericValue(
            value.typ, [prepare_type(arg, ctx) for arg in value.args], weak=value.weak
        )
    elif isinstance(value, (TypedDictValue, CallableValue)):
        return value
    elif isinstance(value, GenericValue):
        # TODO maybe turn DictIncompleteValue into TypedDictValue?
        return GenericValue(
            value.typ, [prepare_type(arg, ctx) for arg in value.args], weak=value.weak
        )
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
    yield from node.args.posonlyargs
    yield from node.args.args
    if include_var and node.args.vararg is not None:
        yield node.args.vararg
    yield from node.args.kwonlyargs
    if include_var and node.args.kwarg is not None:
        yield node.args.kwarg
