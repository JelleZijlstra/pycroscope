from collections.abc import Container
from dataclasses import dataclass
from typing import Literal, Optional, Union

from typing_extensions import Self, assert_never

import pycroscope
from pycroscope.extensions import ExternalType
from pycroscope.relations import Relation
from pycroscope.stacked_scopes import Composite
from pycroscope.value import (
    AnyValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    LowerBound,
    TypeVarMap,
    UpperBound,
    Value,
)

ParamSpecLike = Union[
    ExternalType["typing.ParamSpec"], ExternalType["typing_extensions.ParamSpec"]
]


@dataclass(frozen=True)
class ParamSpecSig:
    param_spec: ParamSpecLike
    default: Optional[Value] = None  # unsupported

    def substitute_typevars(self, typevars: TypeVarMap) -> "InputSig":
        if self.param_spec in typevars:
            return assert_input_sig(typevars[self.param_spec])
        return self


@dataclass(frozen=True)
class AnySig:
    def substitute_typevars(self, typevars: TypeVarMap) -> Self:
        return self


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
    star_args: Optional[Value]  # represents the type of the elements of *args
    keywords: dict[str, tuple[bool, Composite]]
    star_kwargs: Optional[Value]  # represents the type of the elements of **kwargs
    kwargs_required: bool
    pos_or_keyword_params: Container[Union[int, str]]
    ellipsis: bool = False
    param_spec: Optional[ParamSpecSig] = None

    def substitute_typevars(self, typevars: TypeVarMap) -> Self:
        return self

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True)
class FullSignature:
    sig: "pycroscope.signature.Signature"

    def substitute_typevars(self, typevars: TypeVarMap) -> "FullSignature":
        return FullSignature(sig=self.sig.substitute_typevars(typevars))

    def __str__(self) -> str:
        return str(self.sig)


InputSig = Union[ActualArguments, ParamSpecSig, AnySig, FullSignature]


@dataclass(frozen=True)
class InputSigValue(Value):
    """Dummy value wrapping an InputSig."""

    input_sig: InputSig

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return InputSigValue(self.input_sig.substitute_typevars(typevars))

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
