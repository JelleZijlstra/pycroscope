import typing
from collections.abc import Container, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import typing_extensions
from typing_extensions import Self, assert_never

import pycroscope
from pycroscope.relations import Relation
from pycroscope.safe import is_instance_of_typing_name
from pycroscope.stacked_scopes import Composite
from pycroscope.value import (
    AnyValue,
    Bound,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    LowerBound,
    TypeVarLike,
    TypeVarMap,
    TypeVarValue,
    UpperBound,
    Value,
)

ParamSpecLike = typing_extensions.ParamSpec | typing.ParamSpec


@dataclass(frozen=True)
class ParamSpecSig:
    param_spec: ParamSpecLike
    default: Value | None = None  # unsupported

    def substitute_typevars(self, typevars: TypeVarMap) -> "InputSig":
        if self.param_spec in typevars:
            return assert_input_sig(typevars[self.param_spec])
        return self

    def walk_values(self) -> Iterable[Value]:
        if self.default is not None:
            yield from self.default.walk_values()


@dataclass(frozen=True)
class AnySig:
    def substitute_typevars(self, typevars: TypeVarMap) -> Self:
        return self

    def walk_values(self) -> Iterable[Value]:
        return []


ELLIPSIS = AnySig()


@dataclass
class ActualArguments:
    """Represents the actual arguments to a call.

    Before creating this class, we decompose ``*args`` and ``**kwargs`` arguments
    of known composition into additional positional and keyword arguments, and we
    merge multiple ``*args`` or ``**kwargs``.

    Creating the ``ActualArguments`` for a call is independent of the signature
    of the callee.

    """

    positionals: list[tuple[bool, Composite]]
    star_args: Value | None  # represents the type of the elements of *args
    keywords: dict[str, tuple[bool, Composite]]
    star_kwargs: Value | None  # represents the type of the elements of **kwargs
    kwargs_required: bool
    pos_or_keyword_params: Container[int | str]
    ellipsis: bool = False
    param_spec: ParamSpecSig | None = None

    def substitute_typevars(self, typevars: TypeVarMap) -> Self:
        return self

    def __hash__(self) -> int:
        return id(self)

    def walk_values(self) -> Iterable[Value]:
        for _, composite in self.positionals:
            yield from composite.value.walk_values()
        if self.star_args is not None:
            yield from self.star_args.walk_values()
        for _, composite in self.keywords.values():
            yield from composite.value.walk_values()
        if self.star_kwargs is not None:
            yield from self.star_kwargs.walk_values()


@dataclass(frozen=True)
class FullSignature:
    sig: "pycroscope.signature.Signature"

    def substitute_typevars(self, typevars: TypeVarMap) -> "FullSignature":
        return FullSignature(sig=self.sig.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield from self.sig.walk_values()

    def __str__(self) -> str:
        return str(self.sig)


InputSig = ActualArguments | ParamSpecSig | AnySig | FullSignature


@dataclass(frozen=True)
class InputSigValue(Value):
    """Dummy value wrapping an InputSig."""

    input_sig: InputSig

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return InputSigValue(self.input_sig.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.input_sig.walk_values()

    def __str__(self) -> str:
        return str(self.input_sig)


def assert_input_sig(value: Value) -> InputSig:
    """Assert that the value is an InputSig."""
    if isinstance(value, InputSigValue):
        return value.input_sig
    elif isinstance(value, AnyValue):
        return ELLIPSIS
    raise TypeError(f"Expected InputSig, got {value!r}")


def input_sigs_have_relation(
    left: InputSig,
    right: InputSig,
    relation: Literal[Relation.ASSIGNABLE, Relation.SUBTYPE],
    ctx: CanAssignContext,
) -> CanAssign:
    if isinstance(left, AnySig):
        if relation is Relation.SUBTYPE:
            return CanAssignError("Cannot be assigned to")
        return {}
    elif isinstance(left, ParamSpecSig):
        return {left.param_spec: [LowerBound(left.param_spec, InputSigValue(right))]}
    elif isinstance(left, ActualArguments):
        if left == right:
            return {}
        return CanAssignError("Cannot be assigned to")
    elif isinstance(left, FullSignature):
        if isinstance(right, AnySig):
            if relation is Relation.SUBTYPE:
                return CanAssignError("Cannot be assigned")
            return {}
        elif isinstance(right, ParamSpecSig):
            return {
                right.param_spec: [UpperBound(right.param_spec, InputSigValue(left))]
            }
        elif isinstance(right, ActualArguments):
            return pycroscope.signature.check_call_preprocessed(left.sig, right, ctx)
        elif isinstance(right, FullSignature):
            return pycroscope.signature.signatures_have_relation(
                left.sig, right.sig, relation, ctx
            )
        else:
            assert_never(right)
    else:
        assert_never(left)


def solve_paramspec(
    bounds: Sequence[Bound], ctx: CanAssignContext
) -> Value | CanAssignError:
    if not bounds:
        return CanAssignError("Unsupported ParamSpec")
    bound = bounds[0]
    if not isinstance(bound, LowerBound):
        return CanAssignError("Unsupported ParamSpec")
    solution = assert_input_sig(bound.value)
    for i, bound in enumerate(bounds):
        if i == 0:
            continue
        if isinstance(bound, LowerBound):
            value = assert_input_sig(bound.value)
            can_assign = input_sigs_have_relation(
                solution, value, Relation.ASSIGNABLE, ctx
            )
            if isinstance(can_assign, CanAssignError):
                return can_assign
        elif isinstance(bound, UpperBound):
            value = assert_input_sig(bound.value)
            can_assign = input_sigs_have_relation(
                value, solution, Relation.ASSIGNABLE, ctx
            )
            if isinstance(can_assign, CanAssignError):
                return can_assign
        else:
            return CanAssignError("Unsupported ParamSpec bound")
    return InputSigValue(solution)


def extract_type_params(value: Value) -> Iterable[TypeVarLike]:
    for val in value.walk_values():
        if isinstance(val, TypeVarValue):
            yield val.typevar
        elif isinstance(val, InputSigValue):
            input_sig = val.input_sig
            if isinstance(input_sig, ParamSpecSig):
                yield input_sig.param_spec


def wrap_type_param(type_param: TypeVarLike) -> Value:
    """Wrap a type parameter in an InputSigValue."""
    if is_instance_of_typing_name(type_param, "ParamSpec"):
        # static analysis: ignore[incompatible_argument]
        return InputSigValue(ParamSpecSig(type_param))
    elif is_instance_of_typing_name(type_param, "TypeVar"):
        return TypeVarValue(type_param)
    else:
        raise TypeError(f"Unsupported type parameter: {type_param!r}")
