"""

Value classes represent the value of an expression in a Python program. Values
are the key data type used in pycroscope's internals.

Values are instances of a subclass of :class:`Value`. This module defines
these subclasses and some related utilities.

:func:`dump_value` can be used to show inferred values during type checking. Examples::

    from typing import Any
    from pycroscope import dump_value

    def function(x: int, y: list[int], z: Any):
        dump_value(1)  # Literal[1]
        dump_value(x)  # int
        dump_value(y)  # list[int]
        dump_value(z)  # Any

"""

import ast
import builtins
import collections.abc
import contextlib
import enum
import inspect
import sys
import textwrap
import types
import typing
from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field
from itertools import chain
from types import FunctionType, ModuleType
from typing import Any, Optional, TypeGuard, TypeVar, Union

import typing_extensions
from typing_extensions import NoDefault, Protocol, assert_never

import pycroscope
from pycroscope.error_code import Error, ErrorCode
from pycroscope.extensions import CustomCheck, ExternalType
from pycroscope.safe import (
    is_instance_of_typing_name,
    safe_equals,
    safe_getattr,
    safe_isinstance,
    safe_issubclass,
)

T = TypeVar("T")
# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__
KNOWN_MUTABLE_TYPES = (list, set, dict, deque)
ITERATION_LIMIT = 1000

TypeVarType = ExternalType["typing.TypeVar"] | ExternalType["typing_extensions.TypeVar"]
ParamSpecLike = (
    ExternalType["typing.ParamSpec"] | ExternalType["typing_extensions.ParamSpec"]
)
if sys.version_info >= (3, 11):
    TypeVarTupleLike = typing.TypeVarTuple | typing_extensions.TypeVarTuple
else:
    TypeVarTupleLike = typing_extensions.TypeVarTuple
TypeVarLike = TypeVarType | ParamSpecLike | TypeVarTupleLike
SequenceMember = tuple[bool, "Value"]
SequenceMembers = tuple[SequenceMember, ...]


def _is_paramspec_substitution(
    value: object,
) -> TypeGuard["pycroscope.input_sig.InputSig"]:
    from pycroscope.input_sig import AnySig, FullSignature, ParamSpecParam
    from pycroscope.signature import ActualArguments

    return isinstance(value, (ActualArguments, ParamSpecParam, AnySig, FullSignature))


def _is_sequence_members(value: object) -> TypeGuard[SequenceMembers]:
    return isinstance(value, tuple) and all(
        isinstance(member, tuple)
        and len(member) == 2
        and isinstance(member[0], bool)
        and isinstance(member[1], Value)
        for member in value
    )


def _paramspec_value_to_input_sig(value: object) -> "pycroscope.input_sig.InputSig":
    from pycroscope.input_sig import (
        InputSigValue,
        assert_input_sig,
        coerce_paramspec_specialization_to_input_sig,
    )

    if _is_paramspec_substitution(value):
        return value
    if isinstance(value, InputSigValue):
        return value.input_sig
    if isinstance(value, Value):
        return assert_input_sig(coerce_paramspec_specialization_to_input_sig(value))
    raise TypeError(f"Expected ParamSpec substitution, got {value!r}")


@dataclass(frozen=True, init=False)
class TypeVarMap:
    """Immutable typed storage for type parameter substitutions."""

    _typevars: dict[TypeVarType, "Value"]
    _paramspecs: dict[ParamSpecLike, "pycroscope.input_sig.InputSig"]
    _typevartuples: dict[TypeVarTupleLike, SequenceMembers]

    def __init__(
        self,
        *,
        typevars: Mapping[TypeVarType, "Value"] | None = None,
        paramspecs: (
            Mapping[ParamSpecLike, "pycroscope.input_sig.InputSig"] | None
        ) = None,
        typevartuples: Mapping[TypeVarTupleLike, SequenceMembers] | None = None,
    ) -> None:
        typevars_dict = dict({} if typevars is None else typevars)
        if not all(isinstance(value, Value) for value in typevars_dict.values()):
            raise TypeError("TypeVar substitutions must be Values")
        paramspecs_dict = dict({} if paramspecs is None else paramspecs)
        if not all(
            _is_paramspec_substitution(value) for value in paramspecs_dict.values()
        ):
            raise TypeError("ParamSpec substitutions must be InputSig values")
        typevartuples_dict = dict({} if typevartuples is None else typevartuples)
        if not all(
            _is_sequence_members(value) for value in typevartuples_dict.values()
        ):
            raise TypeError("TypeVarTuple substitutions must be SequenceMembers")
        object.__setattr__(self, "_typevars", typevars_dict)
        object.__setattr__(self, "_paramspecs", paramspecs_dict)
        object.__setattr__(self, "_typevartuples", typevartuples_dict)

    def __bool__(self) -> bool:
        return bool(self._typevars or self._paramspecs or self._typevartuples)

    @typing_extensions.overload
    def get_typevar(self, type_param: "TypeVarParam") -> "Value | None": ...

    @typing_extensions.overload
    def get_typevar(self, type_param: "TypeVarParam", default: T) -> "Value | T": ...

    def get_typevar(
        self, type_param: "TypeVarParam", default: T | None = None
    ) -> "Value | T | None":
        return self._typevars.get(type_param.typevar, default)

    def has_typevar(self, type_param: "TypeVarParam") -> bool:
        return type_param.typevar in self._typevars

    @typing_extensions.overload
    def get_paramspec(
        self, type_param: "ParamSpecParam"
    ) -> "pycroscope.input_sig.InputSig | None": ...

    @typing_extensions.overload
    def get_paramspec(
        self, type_param: "ParamSpecParam", default: T
    ) -> "pycroscope.input_sig.InputSig | T": ...

    def get_paramspec(
        self, type_param: "ParamSpecParam", default: T | None = None
    ) -> "pycroscope.input_sig.InputSig | T | None":
        return self._paramspecs.get(type_param.param_spec, default)

    def has_paramspec(self, type_param: "ParamSpecParam") -> bool:
        return type_param.param_spec in self._paramspecs

    @typing_extensions.overload
    def get_typevartuple(
        self, type_param: "TypeVarTupleParam"
    ) -> "SequenceMembers | None": ...

    @typing_extensions.overload
    def get_typevartuple(
        self, type_param: "TypeVarTupleParam", default: T
    ) -> "SequenceMembers | T": ...

    def get_typevartuple(
        self, type_param: "TypeVarTupleParam", default: T | None = None
    ) -> "SequenceMembers | T | None":
        return self._typevartuples.get(type_param.typevar, default)

    def has_typevartuple(self, type_param: "TypeVarTupleParam") -> bool:
        return type_param.typevar in self._typevartuples

    @typing_extensions.overload
    def get_value(self, type_param: "TypeParam") -> "Value | None": ...

    @typing_extensions.overload
    def get_value(self, type_param: "TypeParam", default: T) -> "Value | T": ...

    def get_value(
        self, type_param: "TypeParam", default: T | None = None
    ) -> "Value | T | None":
        if isinstance(type_param, TypeVarParam):
            return self._typevars.get(type_param.typevar, default)
        if isinstance(type_param, ParamSpecParam):
            paramspec = self._paramspecs.get(type_param.param_spec, default)
            if paramspec is default:
                return default
            assert _is_paramspec_substitution(paramspec), paramspec
            return self._paramspec_to_value(paramspec)
        typevartuple = self._typevartuples.get(type_param.typevar, default)
        if typevartuple is default:
            return default
        assert _is_sequence_members(typevartuple), typevartuple
        return typevartuple_binding_to_value(typevartuple)

    def with_typevar(self, type_param: "TypeVarParam", value: "Value") -> "TypeVarMap":
        return self.merge(TypeVarMap(typevars={type_param.typevar: value}))

    def with_paramspec(
        self, type_param: "ParamSpecParam", value: "pycroscope.input_sig.InputSig"
    ) -> "TypeVarMap":
        return self.merge(TypeVarMap(paramspecs={type_param.param_spec: value}))

    def with_typevartuple(
        self, type_param: "TypeVarTupleParam", value: SequenceMembers
    ) -> "TypeVarMap":
        return self.merge(TypeVarMap(typevartuples={type_param.typevar: value}))

    def with_value(self, type_param: "TypeParam", value: object) -> "TypeVarMap":
        if isinstance(type_param, TypeVarParam):
            assert isinstance(value, Value), value
            return self.with_typevar(type_param, value)
        if isinstance(type_param, ParamSpecParam):
            return self.with_paramspec(type_param, _paramspec_value_to_input_sig(value))
        if _is_sequence_members(value):
            return self.with_typevartuple(type_param, value)
        assert isinstance(value, Value), value
        return self.with_typevartuple(type_param, typevartuple_value_to_members(value))

    def merge(self, *others: "TypeVarMap") -> "TypeVarMap":
        if not others:
            return self
        typevars = dict(self._typevars)
        paramspecs = dict(self._paramspecs)
        typevartuples = dict(self._typevartuples)
        for other in others:
            typevars.update(other._typevars)
            paramspecs.update(other._paramspecs)
            typevartuples.update(other._typevartuples)
        return TypeVarMap(
            typevars=typevars, paramspecs=paramspecs, typevartuples=typevartuples
        )

    def iter_typevars(self) -> Iterator[tuple[TypeVarType, "Value"]]:
        return iter(self._typevars.items())

    def iter_paramspecs(
        self,
    ) -> Iterator[tuple[ParamSpecLike, "pycroscope.input_sig.InputSig"]]:
        return iter(self._paramspecs.items())

    def iter_typevartuples(
        self,
    ) -> "Iterator[tuple[TypeVarTupleLike, SequenceMembers]]":
        return iter(self._typevartuples.items())

    def __str__(self) -> str:
        parts = []
        if self._typevars:
            parts.append(
                "typevars={"
                + ", ".join(
                    f"{tv.__name__}={value}" for tv, value in self._typevars.items()
                )
                + "}"
            )
        if self._paramspecs:
            parts.append(
                "paramspecs={"
                + ", ".join(
                    f"{tv.__name__}={value}" for tv, value in self._paramspecs.items()
                )
                + "}"
            )
        if self._typevartuples:
            parts.append(
                "typevartuples={"
                + ", ".join(
                    f"{tv.__name__}={typevartuple_binding_to_value(value)}"
                    for tv, value in self._typevartuples.items()
                )
                + "}"
            )
        return f"TypeVarMap({', '.join(parts)})"

    def substitute_typevars(self, tv_map: "TypeVarMap") -> "TypeVarMap":
        if not self or not tv_map:
            return self
        substituted_typevars = {
            tv: value.substitute_typevars(tv_map)
            for tv, value in self._typevars.items()
        }
        substituted_paramspecs = {
            ps: ps_value.substitute_typevars(tv_map)
            for ps, ps_value in self._paramspecs.items()
        }
        substituted_typevartuples = {
            tv: substitute_typevartuple_binding(binding, tv_map)
            for tv, binding in self._typevartuples.items()
        }
        return TypeVarMap(
            typevars=substituted_typevars,
            paramspecs=substituted_paramspecs,
            typevartuples=substituted_typevartuples,
        )

    @staticmethod
    def _paramspec_to_value(paramspec: "pycroscope.input_sig.InputSig") -> "Value":
        from pycroscope.input_sig import InputSigValue

        return InputSigValue(paramspec)


def pack_typevartuple_binding(values: Iterable["Value"]) -> SequenceMembers:
    members: list[SequenceMember] = []
    for value in values:
        if isinstance(value, TypeVarTupleValue):
            members.append((True, value))
        elif isinstance(value, TypeVarTupleBindingValue):
            members.extend(value.binding)
        else:
            normalized = (
                replace_known_sequence_value(value)
                if isinstance(value, KnownValue)
                else value
            )
            if (
                isinstance(normalized, SequenceValue)
                and normalized.typ is tuple
                and not normalized.members
            ):
                continue
            members.append((False, value))
    return tuple(members)


def typevartuple_value_to_members(value: "Value") -> SequenceMembers:
    if isinstance(value, TypeVarTupleValue):
        return ((True, value),)
    if isinstance(value, TypeVarTupleBindingValue):
        return value.binding
    if isinstance(value, AnyValue):
        return ((True, value),)
    normalized = replace_known_sequence_value(value)
    if isinstance(normalized, SequenceValue) and normalized.typ is tuple:
        return normalized.members
    if (
        isinstance(normalized, GenericValue)
        and normalized.typ is tuple
        and len(normalized.args) == 1
    ):
        return ((True, normalized.args[0]),)
    raise TypeError("TypeVarTuple substitutions must be tuple-like values")


def typevartuple_binding_to_value(binding: SequenceMembers) -> "Value":
    return TypeVarTupleBindingValue(binding)


def typevartuple_binding_to_tuple_value(binding: SequenceMembers) -> "SequenceValue":
    return SequenceValue(tuple, binding)


def typevartuple_binding_to_generic_args(binding: SequenceMembers) -> list["Value"]:
    args: list[Value] = []
    for is_many, value in binding:
        if is_many and isinstance(value, TypeVarTupleValue):
            args.append(value)
        elif is_many:
            args.append(TypeVarTupleBindingValue(((True, value),)))
        else:
            args.append(value)
    return args


def substitute_typevartuple_binding(
    binding: SequenceMembers, typevars: TypeVarMap
) -> SequenceMembers:
    substituted_members: list[SequenceMember] = []
    for is_many, value in binding:
        substituted = value.substitute_typevars(typevars)
        if is_many and isinstance(value, TypeVarTupleValue):
            if isinstance(substituted, TypeVarTupleBindingValue):
                substituted_members.extend(substituted.binding)
                continue
            if isinstance(substituted, TypeVarTupleValue):
                substituted_members.append((True, substituted))
                continue
        substituted_members.append((is_many, substituted))
    return tuple(substituted_members)


def _typevar_map_from_varlike_pairs(
    pairs: Iterable[tuple[TypeVarLike, object]],
) -> TypeVarMap:
    typevars: dict[TypeVarType, Value] = {}
    paramspecs: dict[ParamSpecLike, pycroscope.input_sig.InputSig] = {}
    typevartuples: dict[TypeVarTupleLike, SequenceMembers] = {}
    for typevar, value in pairs:
        if is_instance_of_typing_name(typevar, "TypeVar"):
            assert isinstance(value, Value), value
            typevars[typevar] = value
        elif is_instance_of_typing_name(typevar, "ParamSpec"):
            paramspecs[typevar] = _paramspec_value_to_input_sig(value)
        elif is_instance_of_typing_name(typevar, "TypeVarTuple"):
            if isinstance(value, Value):
                typevartuples[typevar] = typevartuple_value_to_members(value)
            elif _is_sequence_members(value):
                typevartuples[typevar] = value
            else:
                raise TypeError(
                    f"Expected SequenceMembers for {typevar!r}, got {value!r}"
                )
        else:
            raise TypeError(f"Unrecognized type parameter {typevar!r}")
    return TypeVarMap(
        typevars=typevars, paramspecs=paramspecs, typevartuples=typevartuples
    )


def _iter_typevar_map_items(
    typevars: TypeVarMap,
) -> Iterator[tuple[TypeVarLike, "Value"]]:
    # TODO: This kind-erased compatibility layer lets callers reconstruct the old
    # TypeVarLike -> Value map shape and bypass the typed TypeVarMap API. That makes
    # it easier to accidentally treat TypeVar, ParamSpec, and TypeVarTuple uniformly
    # again. Migrate call sites to iter_typevars()/iter_paramspecs()/iter_typevartuples()
    # and delete these mixed-key helpers once the remaining users are gone.
    yield from typevars.iter_typevars()
    for paramspec, input_sig in typevars.iter_paramspecs():
        yield paramspec, TypeVarMap._paramspec_to_value(input_sig)
    for typevartuple, binding in typevars.iter_typevartuples():
        yield typevartuple, typevartuple_binding_to_value(binding)


@typing_extensions.overload
def _get_typevar_map_value(
    typevars: TypeVarMap, typevar: TypeVarLike
) -> "Value | None": ...


@typing_extensions.overload
def _get_typevar_map_value(
    typevars: TypeVarMap, typevar: TypeVarLike, default: T
) -> "Value | T": ...


def _get_typevar_map_value(
    typevars: TypeVarMap, typevar: TypeVarLike, default: T | None = None
) -> "Value | T | None":
    if is_instance_of_typing_name(typevar, "TypeVar"):
        return typevars._typevars.get(typevar, default)
    if is_instance_of_typing_name(typevar, "ParamSpec"):
        paramspec = typevars._paramspecs.get(typevar, default)
        if paramspec is default:
            return default
        assert _is_paramspec_substitution(paramspec), paramspec
        return TypeVarMap._paramspec_to_value(paramspec)
    if is_instance_of_typing_name(typevar, "TypeVarTuple"):
        binding = typevars._typevartuples.get(typevar, default)
        if binding is default:
            return default
        assert _is_sequence_members(binding), binding
        return typevartuple_binding_to_value(binding)
    raise TypeError(f"Unrecognized type parameter {typevar!r}")


def _has_typevar_map_value(typevars: TypeVarMap, typevar: TypeVarLike) -> bool:
    if is_instance_of_typing_name(typevar, "TypeVar"):
        return typevars.has_typevar(TypeVarParam(typevar))
    if is_instance_of_typing_name(typevar, "ParamSpec"):
        return typevars.has_paramspec(ParamSpecParam(typevar))
    if is_instance_of_typing_name(typevar, "TypeVarTuple"):
        return typevars.has_typevartuple(TypeVarTupleParam(typevar))
    raise TypeError(f"Unrecognized type parameter {typevar!r}")


def _with_typevar_map_value(
    typevars: TypeVarMap, typevar: TypeVarLike, value: object
) -> TypeVarMap:
    if is_instance_of_typing_name(typevar, "TypeVar"):
        assert isinstance(value, Value), value
        return typevars.with_typevar(TypeVarParam(typevar), value)
    if is_instance_of_typing_name(typevar, "ParamSpec"):
        return typevars.with_paramspec(
            ParamSpecParam(typevar), _paramspec_value_to_input_sig(value)
        )
    if is_instance_of_typing_name(typevar, "TypeVarTuple"):
        if isinstance(value, Value):
            return typevars.with_typevartuple(
                TypeVarTupleParam(typevar), typevartuple_value_to_members(value)
            )
        assert _is_sequence_members(value), value
        return typevars.with_typevartuple(TypeVarTupleParam(typevar), value)
    raise TypeError(f"Unrecognized type parameter {typevar!r}")


BoundsMap = Mapping[TypeVarLike, Sequence[ExternalType["pycroscope.value.Bound"]]]
GenericBases = Mapping[type | str, TypeVarMap]


class OverlapMode(enum.Enum):
    IS = 1
    MATCH = 2
    EQ = 3


class Variance(enum.Enum):
    COVARIANT = 1
    CONTRAVARIANT = 2
    INVARIANT = 3
    INFERRED = 4

    def display_name(self) -> str:
        return self.name.lower()


def get_typevar_variance(typevar: TypeVarLike) -> Variance:
    if not is_instance_of_typing_name(typevar, "TypeVar"):
        return Variance.INVARIANT
    is_covariant = bool(getattr(typevar, "__covariant__", False))
    is_contravariant = bool(getattr(typevar, "__contravariant__", False))
    if is_covariant and not is_contravariant:
        return Variance.COVARIANT
    if is_contravariant and not is_covariant:
        return Variance.CONTRAVARIANT
    return Variance.INVARIANT


class Value:
    """Base class for all values."""

    __slots__ = ()

    def can_assign(self, other: "Value", ctx: "CanAssignContext") -> "CanAssign":
        """Whether other can be assigned to self.

        If yes, return a (possibly empty) map with the TypeVar values dictated by the
        assignment. If not, return a :class:`CanAssignError` explaining why the types
        are not compatible.

        For example, calling ``a.can_assign(b, ctx)`` where `a` is ``Iterable[T]``
        and `b` is ``List[int]`` will return ``{T: TypedValue(int)}``.

        This is the primary mechanism used for checking type compatibility.

        """
        return pycroscope.relations.has_relation(
            self, other, pycroscope.relations.Relation.ASSIGNABLE, ctx
        )

    def can_overlap(
        self, other: "Value", ctx: "CanAssignContext", mode: OverlapMode
    ) -> Optional["CanAssignError"]:
        """Returns whether self and other can overlap.

        Return None if they can overlap, otherwise a CanAssignError explaining
        why they cannot.

        """
        if isinstance(other, (AnyValue, VariableNameValue)):
            return None
        if isinstance(other, MultiValuedValue):
            # allow overlap with Never
            if other is NO_RETURN_VALUE:
                return None
            errors: list[CanAssignError] = []
            for val in other.vals:
                maybe_error = self.can_overlap(val, ctx, mode)
                if maybe_error is None:
                    return None
                errors.append(maybe_error)
            return CanAssignError("cannot overlap with union", errors)
        if isinstance(other, AnnotatedValue):
            return self.can_overlap(other.value, ctx, mode)
        if isinstance(other, TypeVarValue):
            return self.can_overlap(other.get_fallback_value(), ctx, mode)
        if isinstance(other, TypeAliasValue):
            return self.can_overlap(other.get_value(), ctx, mode)
        return CanAssignError(f"{self} and {other} cannot overlap")

    def is_assignable(self, other: "Value", ctx: "CanAssignContext") -> bool:
        """Similar to :meth:`can_assign` but returns a bool for simplicity."""
        return isinstance(self.can_assign(other, ctx), dict)

    def walk_values(self) -> Iterable["Value"]:
        """Iterator that yields all sub-values contained in this value."""
        yield self

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        """Substitute the typevars in the map to produce a new Value.

        This is used to specialize a generic. For example, substituting
        ``{T: int}`` on ``List[T]`` will produce ``List[int]``.

        """
        return self

    def is_type(self, typ: type) -> bool:
        """Returns whether this value is an instance of the given type.

        This method should be avoided. Use :meth:`can_assign` instead for
        checking compatibility.

        """
        return False

    def get_type(self) -> type | None:
        """Returns the type of this value, or None if it is not known.

        This method should be avoided.

        """
        return None

    def get_fallback_value(self) -> Optional["Value"]:
        """Returns a fallback value for this value, or None if it is not known.

        Implement this on Value subclasses for which most processing can be done
        on a different type.

        """
        return None

    def get_type_value(self) -> "Value":
        """Return the type of this object as used for dunder lookups."""
        return self

    def simplify(self) -> "Value":
        """Simplify this Value to reduce excessive detail."""
        return self

    def decompose(self) -> Iterable["Value"] | None:
        """Optionally, decompose this value into smaller values. The union of these
        values should be equivalent to this value."""
        return None

    def __or__(self, other: "Value") -> "Value":
        """Shortcut for defining a MultiValuedValue."""
        return unite_values(self, other)

    def __ror__(self, other: "Value") -> "Value":
        return unite_values(other, self)

    def __and__(self, other: "Value") -> "Value":
        return IntersectionValue((self, other))

    def __rand__(self, other: "Value") -> "Value":
        return IntersectionValue((other, self))


class CanAssignContext(Protocol):
    """A context passed to the :meth:`Value.can_assign` method.

    Provides access to various functionality used for type checking.

    """

    def make_type_object(self, typ: type | str) -> "pycroscope.type_object.TypeObject":
        """Return a ``pycroscope.type_object.TypeObject`` for this concrete type."""
        raise NotImplementedError

    def get_call_result(
        self,
        callee: "Value",
        args: Iterable["Value"] = (),
        kwargs: Iterable[tuple[str | None, "Value"]] = (),
        node: ast.AST | None = None,
    ) -> "Value":
        """Return the result of calling callee with the given arguments."""
        return AnyValue(AnySource.inference)

    def get_generic_bases(
        self, typ: type | str, generic_args: Sequence["Value"] = ()
    ) -> GenericBases:
        """Return the base classes for `typ` with their generic arguments.

        For example, calling
        ``ctx.get_generic_bases(dict, [TypedValue(int), TypedValue(str)])``
        may produce a map containing the following::

            {
                dict: [TypedValue(int), TypedValue(str)],
                Mapping: [TypedValue(int), TypedValue(str)],
                Iterable: [TypedValue(int)],
                Sized: [],
            }

        """
        return {}

    def get_type_parameters(self, typ: type | str) -> Sequence["TypeParam"]:
        """Return declared generic parameters for `typ`, if available."""
        return ()

    def get_signature(
        self, obj: object
    ) -> Optional["pycroscope.signature.ConcreteSignature"]:
        """Return a :class:`pycroscope.signature.Signature` for this object.

        Return None if the object is not callable.

        """
        return None

    def signature_from_value(
        self, value: "Value"
    ) -> "pycroscope.signature.MaybeSignature":
        """Return a :class:`pycroscope.signature.Signature` for a :class:`Value`.

        Return None if the object is not callable.

        """
        return None

    def get_attribute_from_value(
        self, root_value: "Value", attribute: str, *, prefer_typeshed: bool = False
    ) -> "Value":
        return UNINITIALIZED_VALUE

    def resolve_name(
        self,
        node: ast.Name,
        error_node: ast.AST | None = None,
        suppress_errors: bool = False,
    ) -> tuple["Value", object]:
        """Resolve a name for annotation evaluation."""
        return AnyValue(AnySource.inference), node.id

    def can_assume_compatibility(
        self,
        left: "pycroscope.type_object.TypeObject",
        right: "pycroscope.type_object.TypeObject",
    ) -> bool:
        return False

    def assume_compatibility(
        self,
        left: "pycroscope.type_object.TypeObject",
        right: "pycroscope.type_object.TypeObject",
    ) -> AbstractContextManager[None]:
        return contextlib.nullcontext()

    def can_aliases_assume_compatibility(
        self, left: "TypeAliasValue", right: "TypeAliasValue"
    ) -> bool:
        return False

    def aliases_assume_compatibility(
        self, left: "TypeAliasValue", right: "TypeAliasValue"
    ) -> AbstractContextManager[None]:
        return contextlib.nullcontext()

    def get_relation_cache(self) -> MutableMapping[object, object] | None:
        """Return storage for relation memoization, if supported by this context."""
        return None

    def has_active_relation_assumptions(self) -> bool:
        """Whether relation memoization should be disabled for this context."""
        return False

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""

    def record_protocol_implementation(
        self, protocol: type[object], implementing_class: type[object]
    ) -> None:
        """Record that implementing_class was shown assignable to protocol."""

    def display_value(self, value: Value) -> str:
        """Provide a pretty, user-readable display of this value."""
        return str(value)


@dataclass(frozen=True)
class CanAssignError:
    """A type checking error message with nested details.

    This exists in order to produce more useful error messages
    when there is a mismatch between complex types.

    """

    message: str = ""
    children: list["CanAssignError"] = field(default_factory=list)
    error_code: Error | None = None

    def display(self, depth: int = 2) -> str:
        """Display all errors in a human-readable format."""
        child_result = "".join(
            child.display(depth=depth + 2) for child in self.children
        )
        if self.message:
            message = textwrap.indent(self.message, " " * depth)
            return f"{message}\n{child_result}"
        else:
            return child_result

    def get_error_code(self) -> Error | None:
        errors = {child.get_error_code() for child in self.children}
        if self.error_code:
            errors.add(self.error_code)
        if len(errors) == 1:
            return next(iter(errors))
        return None

    def __str__(self) -> str:
        return self.display()


# Return value of CanAssign
CanAssign = BoundsMap | CanAssignError


def assert_is_value(obj: object, value: Value, *, skip_annotated: bool = False) -> None:
    """Used to test pycroscope's value inference.

    Takes two arguments: a Python object and a :class:`Value` object. At runtime
    this does nothing, but pycroscope throws an error if the object is not
    inferred to be the same as the :class:`Value`.

    Example usage::

        assert_is_value([], KnownValue([]))  # passes
        assert_is_value([], TypedValue(list))  # shows an error

    If skip_annotated is True, unwraps any :class:`AnnotatedValue` in the input.

    """
    pass


def dump_value(value: T) -> T:
    """Print out the :class:`Value` representation of its argument.

    Calling it will make pycroscope print out an internal
    representation of the argument's inferred value. Use
    :func:`pycroscope.extensions.reveal_type` for a
    more user-friendly representation.

    At runtime this returns the argument unchanged.

    """
    return value


def get_mro(typ: object, *, include_virtual: bool = False) -> tuple[type, ...]:
    """Return the runtime MRO for a class-like object.

    During static analysis, pycroscope replaces this with a value-level MRO that
    preserves generic specialization for synthetic and runtime classes.

    """
    if isinstance(typ, type):
        return inspect.getmro(typ)
    return ()


class AnySource(enum.Enum):
    """Sources of Any values."""

    default = 1
    """Any that has not been categorized."""
    explicit = 2
    """The user wrote 'Any' in an annotation."""
    error = 3
    """An error occurred."""
    unreachable = 4
    """Value that is inferred to be unreachable."""
    inference = 5
    """Insufficiently powerful type inference."""
    unannotated = 6
    """Unannotated code."""
    variable_name = 7
    """A :class:`VariableNameValue`."""
    from_another = 8
    """An Any derived from another Any, for example as an attribute."""
    generic_argument = 9
    """Missing type argument to a generic class."""
    marker = 10
    """Marker object used internally."""
    incomplete_annotation = 11
    """A special form like ClassVar without a type argument."""
    multiple_overload_matches = 12
    """Multiple matching overloads."""
    ellipsis_callable = 13
    """Callable using an ellipsis."""
    unresolved_import = 14
    """An unresolved import."""


@dataclass(frozen=True)
class AnyValue(Value):
    """An unknown value, equivalent to ``typing.Any``."""

    source: AnySource
    """The source of this value, such as a user-defined annotation
    or a previous error."""

    def __str__(self) -> str:
        if self.source is AnySource.default:
            return "Any"
        return f"Any[{self.source.name}]"

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return None  # always overlaps

    def substitute_typevars(self, typevars: TypeVarMap) -> "AnyValue":
        return self


UNRESOLVED_VALUE = AnyValue(AnySource.default)
"""The default instance of :class:`AnyValue`.

In the future, this should be replaced with instances of
`AnyValue` with a specific source.

"""


class PartialValueOperation(enum.Enum):
    """Kinds of partially evaluated operations represented by :class:`PartialValue`."""

    SUBSCRIPT = 1
    UNPACK = 2
    BITOR = 3


@dataclass(frozen=True)
class PartialValue(Value):
    """Represents a partially evaluated expression."""

    operation: PartialValueOperation
    root: Value
    node: ast.AST = field(compare=False, hash=False)
    members: tuple[Value, ...]
    runtime_value: Value

    def __str__(self) -> str:
        match self.operation:
            case PartialValueOperation.SUBSCRIPT:
                members = ", ".join(str(member) for member in self.members)
                return f"{self.runtime_value} (partial from {self.root}[{members}])"
            case PartialValueOperation.UNPACK:
                return f"{self.runtime_value} (partial from *{self.root})"
            case PartialValueOperation.BITOR:
                members = " | ".join(str(member) for member in self.members)
                return f"{self.runtime_value} (partial from {self.root} | {members})"
            case _:
                assert_never(self.operation)

    def get_fallback_value(self) -> Value:
        return self.runtime_value

    def get_type_value(self) -> Value:
        return self.runtime_value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> "PartialValue":
        return PartialValue(
            operation=self.operation,
            root=self.root.substitute_typevars(typevars),
            node=self.node,
            members=tuple(
                member.substitute_typevars(typevars) for member in self.members
            ),
            runtime_value=self.runtime_value.substitute_typevars(typevars),
        )

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return self.runtime_value.can_overlap(other, ctx, mode)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.root.walk_values()
        for member in self.members:
            yield from member.walk_values()
        yield from self.runtime_value.walk_values()


@dataclass(frozen=True)
class PartialCallValue(Value):
    """Represents a partially evaluated call expression, where the function is known but
    the arguments are not."""

    callee: Value
    arguments: dict[str, Value]
    runtime_value: Value
    node: ast.AST = field(compare=False, hash=False)

    def __str__(self) -> str:
        return f"{self.runtime_value} (call with arguments {self.arguments})"

    def get_fallback_value(self) -> Value:
        return self.runtime_value

    def get_type_value(self) -> Value:
        return self.runtime_value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> "PartialCallValue":
        return PartialCallValue(
            # We don't substitute typevars on the callee or arguments, they record
            # what happened at call time.
            callee=self.callee,
            arguments=self.arguments,
            runtime_value=self.runtime_value.substitute_typevars(typevars),
            node=self.node,
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        for argument in self.arguments.values():
            yield from argument.walk_values()
        yield from self.runtime_value.walk_values()


@dataclass(frozen=True)
class SuperValue(Value):
    """Value representing a super() call.

    super(typ, self) => SuperValue(thisclass=typ, selfobj=self)
    """

    thisclass: Value
    selfobj: Value | None = None

    def __str__(self) -> str:
        if self.selfobj is not None:
            return f"super({self.thisclass}, {self.selfobj})"
        else:
            return f"super({self.thisclass})"

    def get_fallback_value(self) -> Value:
        return TypedValue(super)

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> "SuperValue":
        return SuperValue(
            thisclass=self.thisclass.substitute_typevars(typevars),
            selfobj=(
                self.selfobj.substitute_typevars(typevars)
                if self.selfobj is not None
                else None
            ),
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.thisclass.walk_values()
        if self.selfobj is not None:
            yield from self.selfobj.walk_values()


@dataclass(frozen=True)
class VoidValue(Value):
    """Dummy Value used as the inferred type of AST nodes that
    do not represent expressions.

    This is useful so that we can infer a Value for every AST node,
    but notice if we unexpectedly use it like an actual value.

    """

    def __str__(self) -> str:
        return "(void)"


VOID = VoidValue()


def _type_param_to_string(
    typevar: TypeVarLike, bound: Value | None, constraints: Sequence[Value]
) -> str:
    if bound is not None:
        return f"{typevar}: {bound}"
    if constraints:
        constraint_list = ", ".join(map(str, constraints))
        return f"{typevar}: ({constraint_list})"
    return str(typevar)


_NO_DEFAULT = object()


@dataclass(frozen=True)
class TypeVarParam:
    typevar: TypeVarType
    bound: Value | None = None
    default: Value | None = None
    constraints: Sequence[Value] = ()
    variance: Variance = Variance.INVARIANT

    def __post_init__(self) -> None:
        if self.bound is None:
            runtime_bound = safe_getattr(self.typevar, "__bound__", None)
            if runtime_bound is not None:
                object.__setattr__(
                    self,
                    "bound",
                    _value_from_runtime_type_param_component(runtime_bound),
                )
        if not self.constraints:
            runtime_constraints = safe_getattr(self.typevar, "__constraints__", ())
            if runtime_constraints:
                object.__setattr__(
                    self,
                    "constraints",
                    tuple(
                        _value_from_runtime_type_param_component(constraint)
                        for constraint in runtime_constraints
                    ),
                )
        if self.default is None:
            runtime_default = safe_getattr(self.typevar, "__default__", _NO_DEFAULT)
            if runtime_default is not _NO_DEFAULT and runtime_default is not NoDefault:
                object.__setattr__(
                    self,
                    "default",
                    _value_from_runtime_type_param_component(runtime_default),
                )
        if self.variance is Variance.INVARIANT:
            object.__setattr__(self, "variance", get_typevar_variance(self.typevar))

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        substituted = typevars.get_typevar(self)
        if substituted is not None:
            return substituted
        return type_param_to_value(self)

    def walk_values(self) -> Iterable[Value]:
        if self.bound is not None:
            yield from self.bound.walk_values()
        if self.default is not None:
            yield from self.default.walk_values()
        for constraint in self.constraints:
            yield from constraint.walk_values()

    def get_fallback_value(self) -> None:
        return None

    def __str__(self) -> str:
        return _type_param_to_string(self.typevar, self.bound, self.constraints)


@dataclass(frozen=True)
class ParamSpecParam:
    param_spec: ParamSpecLike
    default: Value | None = None
    variance: Variance = Variance.INVARIANT

    @property
    def typevar(self) -> ParamSpecLike:
        return self.param_spec

    def substitute_typevars(
        self, typevars: TypeVarMap
    ) -> "pycroscope.input_sig.InputSig":
        substituted = typevars.get_paramspec(self)
        if substituted is not None:
            return substituted
        return self

    def walk_values(self) -> Iterable[Value]:
        if self.default is not None:
            yield from self.default.walk_values()

    def __str__(self) -> str:
        return str(self.param_spec)


@dataclass(frozen=True)
class TypeVarTupleParam:
    typevar_tuple: TypeVarTupleLike
    default: Value | None = None
    variance: Variance = Variance.INVARIANT

    @property
    def typevar(self) -> TypeVarTupleLike:
        return self.typevar_tuple

    def __str__(self) -> str:
        return str(self.typevar)


TypeParam = TypeVarParam | ParamSpecParam | TypeVarTupleParam


def _value_from_runtime_type_param_component(component: object) -> Value:
    if isinstance(component, Value):
        return component
    if is_instance_of_typing_name(component, "TypeVar"):
        return TypeVarValue(TypeVarParam(component))
    if is_instance_of_typing_name(component, "TypeVarTuple"):
        return TypeVarTupleValue(component)
    if is_instance_of_typing_name(component, "ParamSpec"):
        from pycroscope.input_sig import InputSigValue

        return InputSigValue(ParamSpecParam(component))
    if isinstance(component, tuple):
        return SequenceValue(
            tuple,
            [
                (False, _value_from_runtime_type_param_component(member))
                for member in component
            ],
        )
    if isinstance(component, list):
        return SequenceValue(
            list,
            [
                (False, _value_from_runtime_type_param_component(member))
                for member in component
            ],
        )
    if isinstance(component, type):
        return TypedValue(component)
    return KnownValue(component)


def type_param_to_value(type_param: TypeParam) -> Value:
    if isinstance(type_param, TypeVarParam):
        return TypeVarValue(type_param)
    if isinstance(type_param, TypeVarTupleParam):
        return TypeVarTupleValue(type_param)
    from pycroscope.input_sig import InputSigValue

    return InputSigValue(type_param)


def get_single_typevartuple_param(value: Value) -> TypeVarTupleParam | None:
    if isinstance(value, TypeVarTupleValue):
        return value.typevar_tuple_param
    if (
        isinstance(value, TypeVarTupleBindingValue)
        and len(value.binding) == 1
        and value.binding[0][0]
        and isinstance(value.binding[0][1], TypeVarTupleValue)
    ):
        return value.binding[0][1].typevar_tuple_param
    return None


def iter_type_params_in_value(value: Value) -> Iterator[TypeParam]:
    for subval in value.walk_values():
        if isinstance(subval, TypeVarValue):
            yield subval.typevar_param
        else:
            typevartuple_param = get_single_typevartuple_param(subval)
            if typevartuple_param is not None:
                yield typevartuple_param
                continue
            from pycroscope.input_sig import InputSigValue

            if isinstance(subval, InputSigValue) and isinstance(
                subval.input_sig, ParamSpecParam
            ):
                yield subval.input_sig


def get_type_params_by_typevar(value: Value) -> dict[TypeVarLike, TypeParam]:
    return {
        type_param.typevar: type_param
        for type_param in iter_type_params_in_value(value)
    }


def make_inference_typevar_map(values: Iterable[Value]) -> TypeVarMap:
    inference_typevars: dict[TypeVarType, Value] = {}
    for value in values:
        for type_param in iter_type_params_in_value(value):
            if isinstance(type_param, TypeVarParam):
                inference_typevars.setdefault(
                    type_param.typevar, InferenceVarValue(type_param)
                )
    return TypeVarMap(typevars=inference_typevars)


def freshen_typevars_for_inference(value: Value) -> Value:
    inference_map = make_inference_typevar_map([value])
    if not inference_map:
        return value
    return value.substitute_typevars(inference_map)


@dataclass
class TypeAlias:
    evaluator: Callable[[], Value]
    """Callable that evaluates the value."""
    evaluate_type_params: Callable[[], Sequence["TypeParam"]]
    """Callable that evaluates the type parameters."""
    evaluated_value: Value | None = None
    """Value that the type alias evaluates to."""
    type_params: Sequence["TypeParam"] | None = None
    """Type parameters of the type alias."""
    is_evaluating: bool = False
    """Whether this type alias is currently being evaluated."""

    def get_value(self) -> Value:
        if self.evaluated_value is None:
            if self.is_evaluating:
                return AnyValue(AnySource.inference)
            self.is_evaluating = True
            try:
                self.evaluated_value = self.evaluator()
            finally:
                self.is_evaluating = False
        return self.evaluated_value

    def get_type_params(self) -> Sequence["TypeParam"]:
        if self.type_params is None:
            self.type_params = tuple(self.evaluate_type_params())
        return self.type_params


def default_value_for_type_param(type_param: "TypeParam") -> Value:
    if type_param.default is not None:
        return type_param.default
    return AnyValue(AnySource.generic_argument)


def _split_variadic_type_arguments(
    type_params: Sequence["TypeParam"],
    type_arguments: Sequence[Value],
    variadic_index: int,
) -> tuple[int, int] | None:
    suffix_params = type_params[variadic_index + 1 :]
    prefix_explicit_count = min(variadic_index, len(type_arguments))
    while prefix_explicit_count >= 0:
        if any(
            type_param.default is None
            for type_param in type_params[prefix_explicit_count:variadic_index]
        ):
            prefix_explicit_count -= 1
            continue
        suffix_explicit_count = min(
            len(suffix_params), len(type_arguments) - prefix_explicit_count
        )
        omitted_suffix_count = len(suffix_params) - suffix_explicit_count
        if any(
            type_param.default is None
            for type_param in suffix_params[:omitted_suffix_count]
        ):
            prefix_explicit_count -= 1
            continue
        return prefix_explicit_count, suffix_explicit_count
    return None


def match_typevar_arguments(
    type_params: Sequence["TypeParam"],
    type_arguments: Sequence[Value],
    *,
    type_arguments_are_packed: bool = False,
) -> Sequence[tuple[TypeVarLike, Value]] | None:
    if type_arguments_are_packed:
        if len(type_params) != len(type_arguments):
            return None
        return [(param.typevar, arg) for param, arg in zip(type_params, type_arguments)]

    variadic_indexes = [
        i
        for i, type_param in enumerate(type_params)
        if isinstance(type_param, TypeVarTupleParam)
    ]
    if len(variadic_indexes) > 1:
        return None

    substitutions = TypeVarMap()
    matched: list[tuple[TypeVarLike, Value]] = []

    def _record(type_param: TypeParam, argument: Value) -> None:
        matched.append((type_param.typevar, argument))
        nonlocal substitutions
        if isinstance(type_param, TypeVarParam):
            substitutions = substitutions.with_typevar(type_param, argument)
        elif isinstance(type_param, ParamSpecParam):
            from pycroscope.input_sig import AnySig

            try:
                substitutions = substitutions.with_paramspec(
                    type_param, _paramspec_value_to_input_sig(argument)
                )
            except TypeError:
                substitutions = substitutions.with_paramspec(type_param, AnySig())
        else:
            substitutions = substitutions.with_typevartuple(
                type_param, typevartuple_value_to_members(argument)
            )

    def _default_argument(type_param: TypeParam) -> Value:
        return default_value_for_type_param(type_param).substitute_typevars(
            substitutions
        )

    if not variadic_indexes:
        if len(type_arguments) > len(type_params):
            return None
        minimum_required = sum(
            1 for type_param in type_params if type_param.default is None
        )
        if len(type_arguments) < minimum_required:
            return None
        for i, type_param in enumerate(type_params):
            argument = (
                type_arguments[i]
                if i < len(type_arguments)
                else _default_argument(type_param)
            )
            _record(type_param, argument)
        return matched

    variadic_index = variadic_indexes[0]
    minimum_required = sum(
        1
        for i, type_param in enumerate(type_params)
        if i != variadic_index and type_param.default is None
    )
    if len(type_arguments) < minimum_required:
        return None
    split = _split_variadic_type_arguments(type_params, type_arguments, variadic_index)
    if split is None:
        return None
    prefix_explicit_count, suffix_explicit_count = split
    suffix_count = len(type_params) - variadic_index - 1
    omitted_suffix_count = suffix_count - suffix_explicit_count
    variadic_start = prefix_explicit_count
    variadic_end = len(type_arguments) - suffix_explicit_count

    for i, type_param in enumerate(type_params):
        if i < variadic_index:
            argument = (
                type_arguments[i]
                if i < prefix_explicit_count
                else _default_argument(type_param)
            )
        elif i == variadic_index:
            argument = TypeVarTupleBindingValue(
                pack_typevartuple_binding(type_arguments[variadic_start:variadic_end])
            )
        else:
            suffix_index = i - variadic_index - 1
            argument = (
                _default_argument(type_param)
                if suffix_index < omitted_suffix_count
                else type_arguments[variadic_end + suffix_index - omitted_suffix_count]
            )
        _record(type_param, argument)
    return matched


def _match_type_alias_type_arguments(
    type_params: Sequence["TypeParam"],
    type_arguments: Sequence[Value],
    *,
    type_arguments_are_packed: bool = False,
) -> Sequence[tuple[TypeVarLike, Value]] | None:
    return match_typevar_arguments(
        type_params, type_arguments, type_arguments_are_packed=type_arguments_are_packed
    )


@dataclass(frozen=True)
class TypeAliasValue(Value):
    """Value representing a type alias."""

    name: str
    """Name of the type alias."""
    module: str
    """Module where the type alias is defined."""
    alias: TypeAlias = field(compare=False, hash=False)
    type_arguments: Sequence[Value] = ()
    runtime_allows_value_call: bool = False
    uses_type_alias_object_semantics: bool = False
    """Whether symbol access should behave like a runtime TypeAliasType object."""
    is_specialized: bool = False
    type_arguments_are_packed: bool = False

    def get_value(self) -> Value:
        val = self.alias.get_value()
        type_params = self.alias.get_type_params()
        if self.type_arguments or self.is_specialized:
            matched_type_arguments = _match_type_alias_type_arguments(
                type_params,
                self.type_arguments,
                type_arguments_are_packed=self.type_arguments_are_packed,
            )
            if matched_type_arguments is None:
                # TODO this should be an error
                return AnyValue(AnySource.inference)
            typevars = _typevar_map_from_varlike_pairs(matched_type_arguments)
            val = val.substitute_typevars(typevars)
        elif type_params:
            # Unsubscripted aliases default each unspecialized parameter.
            substitutions = TypeVarMap()
            for type_param in type_params:
                default_value = default_value_for_type_param(
                    type_param
                ).substitute_typevars(substitutions)
                substitutions = substitutions.with_value(type_param, default_value)
            val = val.substitute_typevars(substitutions)
        return val

    def get_fallback_value(self) -> Value:
        return self.get_value()

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for arg in self.type_arguments:
            yield from arg.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> "TypeAliasValue":
        if not self.type_arguments:
            return self
        substituted_type_arguments = tuple(
            arg.substitute_typevars(typevars) for arg in self.type_arguments
        )
        if all(
            safe_equals(existing, substituted)
            for existing, substituted in zip(
                self.type_arguments, substituted_type_arguments
            )
        ):
            return self
        return TypeAliasValue(
            self.name,
            self.module,
            self.alias,
            substituted_type_arguments,
            runtime_allows_value_call=self.runtime_allows_value_call,
            uses_type_alias_object_semantics=self.uses_type_alias_object_semantics,
            is_specialized=self.is_specialized,
            type_arguments_are_packed=self.type_arguments_are_packed,
        )

    def is_type(self, typ: type) -> bool:
        return self.get_value().is_type(typ)

    def get_type(self) -> type | None:
        return self.get_value().get_type()

    def get_type_value(self) -> Value:
        return self.get_value().get_type_value()

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if isinstance(other, TypeAliasValue) and self.alias is other.alias:
            return None
        return self.get_value().can_overlap(other, ctx, mode)

    def __str__(self) -> str:
        is_in_set = self in _being_printed
        _being_printed.add(self)
        try:
            text = f"{self.module}.{self.name}"
            if self.type_arguments:
                text += f"[{', '.join(map(str, self.type_arguments))}]"
            if self.alias.evaluated_value is not None and not is_in_set:
                text += f" = {self.alias.evaluated_value}"
            return text
        finally:
            if not is_in_set:
                _being_printed.remove(self)


_being_printed: set[TypeAliasValue] = set()


@dataclass(frozen=True)
class UninitializedValue(Value):
    """Value for variables that have not been initialized.

    Usage of variables with this value should be an error.

    """

    def __str__(self) -> str:
        return "<uninitialized>"


UNINITIALIZED_VALUE = UninitializedValue()
"""The only instance of :class:`UninitializedValue`."""


@dataclass(frozen=True)
class KnownValue(Value):
    """Equivalent to ``typing.Literal``. Represents a specific value.

    This is inferred for constants and for references to objects
    like modules, classes, and functions.

    """

    val: Any
    """The Python object that this ``KnownValue`` represents."""

    def is_type(self, typ: type) -> bool:
        return safe_isinstance(self.val, typ)

    def get_type(self) -> type:
        return type(self.val)

    def get_type_object(
        self, ctx: CanAssignContext
    ) -> "pycroscope.type_object.TypeObject":
        return ctx.make_type_object(type(self.val))

    def get_type_value(self) -> Value:
        return KnownValue(type(self.val))

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if isinstance(other, (SubclassValue, TypedValue)):
            return other.can_overlap(self, ctx, mode)
        elif isinstance(other, KnownValue):
            if self.val is other.val:
                return None
            if mode is OverlapMode.IS:
                # Allow different literals of the same type, otherwise
                # we get lots of false positives.
                if type(self.val) is type(other.val):
                    return None
                return CanAssignError(f"{self} and {other} cannot overlap")
            elif mode is OverlapMode.MATCH:
                if safe_equals(self.val, other.val):
                    return None
                return CanAssignError(f"{self} and {other} cannot overlap")
            elif mode is OverlapMode.EQ:
                # For EQ mode we're more permissive and allow overlapping types
                return TypedValue(type(self.val)).can_overlap(
                    TypedValue(type(other.val)), ctx, mode
                )
            else:
                assert_never(mode)
        return super().can_overlap(other, ctx, mode)

    def __eq__(self, other: Value) -> bool:
        return (
            isinstance(other, KnownValue)
            and type(self.val) is type(other.val)
            and safe_equals(self.val, other.val)
        )

    def __ne__(self, other: Value) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        # Make sure e.g. 1 and True are handled differently.
        try:
            return hash((type(self.val), self.val))
        except TypeError:
            # If the value is not directly hashable, hash it by identity instead. This breaks
            # the rule that x == y should imply hash(x) == hash(y), but hopefully that will
            # be fine.
            return hash((type(self.val), id(self.val)))

    def __str__(self) -> str:
        if self.val is None:
            return "None"
        elif isinstance(self.val, ModuleType):
            return f"module {self.val.__name__!r}"
        elif isinstance(self.val, FunctionType):
            return f"function {get_fully_qualified_name(self.val)!r}"
        elif isinstance(self.val, type):
            return f"type {get_fully_qualified_name(self.val)!r}"
        else:
            return f"Literal[{self.val!r}]"

    def substitute_typevars(self, typevars: TypeVarMap) -> "KnownValue":
        if not typevars or not callable(self.val):
            return self
        return KnownValueWithTypeVars(self.val, typevars)

    def simplify(self) -> Value:
        val = replace_known_sequence_value(self)
        if isinstance(val, KnownValue):
            # don't simplify None
            if val.val is None:
                return self
            return TypedValue(type(val.val))
        return val.simplify()


def get_fully_qualified_name(obj: FunctionType | type) -> str:
    mod = getattr(obj, "__module__", None)
    if mod == "builtins":
        mod = None
    name = getattr(obj, "__qualname__", None)
    if name is None:
        return repr(obj)
    if mod:
        return f"{mod}.{name}"
    return name


@dataclass(frozen=True)
class KnownValueWithTypeVars(KnownValue):
    """Subclass of KnownValue that records a TypeVar substitution."""

    typevars: TypeVarMap = field(compare=False)
    """TypeVars substituted on this value."""

    def substitute_typevars(self, typevars: TypeVarMap) -> "KnownValueWithTypeVars":
        return KnownValueWithTypeVars(
            self.val, self.typevars.substitute_typevars(typevars).merge(typevars)
        )

    def __post_init__(self) -> None:
        if not isinstance(self.typevars, TypeVarMap):
            raise TypeError(
                f"KnownValueWithTypeVars.typevars must be TypeVarMap, got {self.typevars!r}"
            )

    def __str__(self) -> str:
        return super().__str__() + f" with typevars {self.typevars}"


@dataclass(frozen=True)
class SyntheticModuleValue(Value):
    """Represents a module that exists only in stub files."""

    module_path: Sequence[str]


@dataclass(frozen=True)
class UnboundMethodValue(Value):
    """Value that represents a method on an underlying :class:`Value`.

    Despite the name this really represents a method bound to a value. For
    example, given ``s: str``, ``s.strip`` will be inferred as
    ``UnboundMethodValue("strip", Composite(TypedValue(str), "s"))``.

    """

    attr_name: str
    """Name of the method."""
    composite: "pycroscope.stacked_scopes.Composite"
    """Value the method is bound to."""
    secondary_attr_name: str | None = None
    """Used when an attribute is accessed on an existing ``UnboundMethodValue``.

    This is mostly useful in conjunction with asynq, where we might use
    ``object.method.asynq``. In that case, we would infer an ``UnboundMethodValue``
    with `secondary_attr_name` set to ``"asynq"``.

    """
    typevars: TypeVarMap | None = field(default=None, compare=False)
    """Extra TypeVars applied to this method."""

    def __post_init__(self) -> None:
        if self.typevars is not None and not isinstance(self.typevars, TypeVarMap):
            raise TypeError(
                f"UnboundMethodValue.typevars must be TypeVarMap or None, got {self.typevars!r}"
            )

    def get_method(self) -> Any | None:
        """Return the runtime callable for this ``UnboundMethodValue``, or
        None if it cannot be found."""
        root = replace_fallback(self.composite.value)
        target: object
        if isinstance(root, KnownValue):
            target = root.val
        else:
            target = root.get_type()
        try:
            method = getattr(target, self.attr_name)
            if self.secondary_attr_name is not None:
                try:
                    method = getattr(method, self.secondary_attr_name)
                except AttributeError:
                    if not isinstance(target, type):
                        raise
                    bound_target = _bind_runtime_descriptor_for_secondary_attribute(
                        target, self.attr_name
                    )
                    if bound_target is None:
                        raise
                    method = getattr(bound_target, self.secondary_attr_name)
        except AttributeError:
            return None
        return method

    def is_type(self, typ: type) -> bool:
        return isinstance(self.get_method(), typ)

    def get_type(self) -> type:
        return type(self.get_method())

    def get_type_value(self) -> Value:
        return KnownValue(type(self.get_method()))

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        signature = self.get_signature(ctx)
        if signature is None:
            return None
        return CallableValue(signature).can_overlap(other, ctx, mode)

    def get_signature(
        self, ctx: CanAssignContext
    ) -> Optional["pycroscope.signature.ConcreteSignature"]:
        signature = ctx.signature_from_value(self)
        if signature is None:
            return None
        if isinstance(signature, pycroscope.signature.BoundMethodSignature):
            signature = signature.get_signature(ctx=ctx)
        return signature

    def substitute_typevars(self, typevars: TypeVarMap) -> "UnboundMethodValue":
        merged_typevars = typevars
        if self.typevars is not None:
            merged_typevars = self.typevars.merge(typevars)
        return UnboundMethodValue(
            self.attr_name,
            self.composite.substitute_typevars(typevars),
            self.secondary_attr_name,
            typevars=merged_typevars,
        )

    def __str__(self) -> str:
        return "<method {}{} on {}>".format(
            self.attr_name,
            f".{self.secondary_attr_name}" if self.secondary_attr_name else "",
            self.composite.value,
        )


def _bind_runtime_descriptor_for_secondary_attribute(
    owner: type, attr_name: str
) -> object | None:
    try:
        descriptor = inspect.getattr_static(owner, attr_name)
    except AttributeError:
        return None
    if not safe_getattr(descriptor, "__get__", None):
        return None
    try:
        instance = object.__new__(owner)
    except Exception:
        return None
    try:
        return descriptor.__get__(instance, owner)
    except Exception:
        return None


@dataclass(unsafe_hash=True)
class TypedValue(Value):
    """Value for which we know the type. This is equivalent to simple type
    annotations: an annotation of ``int`` will yield ``TypedValue(int)`` during
    type inference.

    """

    typ: type | str
    """The underlying type, or a fully qualified reference to one."""
    literal_only: bool = False
    """True if this is LiteralString (PEP 675)."""

    def get_type_object(
        self, ctx: CanAssignContext
    ) -> "pycroscope.type_object.TypeObject":
        # `TypeObject` instances are checker-specific (they include checker-
        # specific synthetic bases/protocol members and protocol caches), so do
        # not cache them on Value instances that can outlive a single checker.
        return ctx.make_type_object(self.typ)

    def can_assign_thrift_enum(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, AnyValue):
            ctx.record_any_used()
            return {}
        elif isinstance(other, TypeVarValue):
            return super().can_assign(other, ctx)
        elif isinstance(other, KnownValue):
            if not isinstance(other.val, int):
                return CanAssignError(f"{other} is not an int")
            assert hasattr(self.typ, "_VALUES_TO_NAMES"), f"{self} is not a Thrift enum"
            if other.val in self.typ._VALUES_TO_NAMES:
                return {}
        elif isinstance(other, TypedValue):
            tobj = other.get_type_object(ctx)
            if tobj.is_assignable_to_type(int):
                return {}
            return self.get_type_object(ctx).can_assign(self, other, ctx)
        elif isinstance(other, MultiValuedValue):
            bounds_maps = []
            for val in other.vals:
                can_assign = self.can_assign(val, ctx)
                if isinstance(can_assign, CanAssignError):
                    # Adding an additional layer here isn't helpful
                    return can_assign
                bounds_maps.append(can_assign)
            if not bounds_maps:
                return CanAssignError(f"Cannot assign {other} to Thrift enum {self}")
            return unify_bounds_maps(bounds_maps)
        elif isinstance(other, AnnotatedValue):
            return self.can_assign_thrift_enum(other.value, ctx)
        return CanAssignError(f"Cannot assign {other} to Thrift enum {self}")

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        self_tobj = self.get_type_object(ctx)
        if self_tobj.is_thrift_enum():
            if isinstance(other, (KnownValue, TypedValue)):
                can_assign = self.can_assign_thrift_enum(other, ctx)
                if isinstance(can_assign, CanAssignError):
                    return can_assign
                return None
            else:
                return super().can_overlap(other, ctx, mode)
        elif isinstance(other, KnownValue):
            if mode is OverlapMode.IS:
                if (
                    isinstance(self.typ, type)
                    and safe_issubclass(self.typ, enum.Enum)
                    and safe_isinstance(other.val, enum.Enum)
                ):
                    return None
                if self_tobj.is_instance(other.val):
                    return None
                return CanAssignError(f"{self} and {other} cannot overlap")
            return self.can_overlap(TypedValue(type(other.val)), ctx, mode)
        elif isinstance(other, TypedValue):
            left_errors = self_tobj.can_assign(self, other, ctx)
            if not isinstance(left_errors, CanAssignError):
                return None
            other_tobj = other.get_type_object(ctx)
            right_errors = other_tobj.can_assign(other, self, ctx)
            if not isinstance(right_errors, CanAssignError):
                return None
            if mode in (OverlapMode.EQ, OverlapMode.MATCH):
                if self_tobj.overrides_eq(self, ctx) or other_tobj.overrides_eq(
                    other, ctx
                ):
                    return None
            return CanAssignError(
                f"{self} and {other} cannot overlap", [left_errors, right_errors]
            )
        elif isinstance(other, SubclassValue):
            if isinstance(other.typ, TypedValue):
                error = self_tobj.can_assign(self, other, ctx)
                if isinstance(error, CanAssignError):
                    return error
                return None
            else:
                return None
        else:
            return super().can_overlap(other, ctx, mode)

    def get_generic_args_for_type(
        self, typ: type | str, ctx: CanAssignContext
    ) -> list[Value] | None:
        if isinstance(self, GenericValue):
            args = self.args
        else:
            args = ()
        generic_bases = ctx.get_generic_bases(self.typ, args)
        params_key: type | str = typ
        if params_key in generic_bases:
            raw_args: list[Value] = []
            for type_param in ctx.get_type_parameters(params_key):
                raw_arg = generic_bases[params_key].get_value(type_param)
                assert isinstance(raw_arg, Value), raw_arg
                raw_args.append(raw_arg)
            if (
                not raw_args
                and isinstance(self, GenericValue)
                and params_key == self.typ
                and self.args
            ):
                # Synthetic generic metadata can occasionally lose self-mapping
                # while we still have explicit specialization arguments.
                return list(self.args)
            declared_params = ctx.get_type_parameters(params_key)
            if declared_params and len(declared_params) == len(raw_args):
                expanded_args: list[Value] = []
                for declared_param, raw_arg in zip(declared_params, raw_args):
                    if isinstance(declared_param, TypeVarTupleParam):
                        normalized_arg = replace_known_sequence_value(raw_arg)
                        if (
                            params_key is not tuple
                            and isinstance(normalized_arg, SequenceValue)
                            and normalized_arg.typ is tuple
                            and all(
                                not is_many for is_many, _ in normalized_arg.members
                            )
                        ):
                            expanded_args.extend(
                                member for _, member in normalized_arg.members
                            )
                            continue
                    expanded_args.append(raw_arg)
                return expanded_args
            return raw_args
        if isinstance(self, GenericValue) and typ == self.typ:
            # Preserve explicitly provided arguments when base expansion has
            # no self-entry for this synthetic specialization.
            return list(self.args)
        return None

    def get_generic_arg_for_type(
        self, typ: type | str, ctx: CanAssignContext, index: int
    ) -> Value:
        args = self.get_generic_args_for_type(typ, ctx)
        if args and index < len(args):
            return args[index]
        return AnyValue(AnySource.generic_argument)

    def is_type(self, typ: type) -> bool:
        return isinstance(self.typ, type) and safe_issubclass(self.typ, typ)

    def get_type(self) -> type | None:
        if isinstance(self.typ, str):
            return None
        return self.typ

    def get_type_value(self) -> Value:
        if isinstance(self.typ, str):
            return AnyValue(AnySource.inference)
        return KnownValue(self.typ)

    def decompose(self) -> Iterable[Value] | None:
        if self.typ is bool:
            return [KnownValue(True), KnownValue(False)]
        if (
            isinstance(self.typ, type)
            and safe_issubclass(self.typ, enum.Enum)
            and not safe_issubclass(self.typ, enum.Flag)
        ):
            # Decompose enum into its members
            assert issubclass(self.typ, enum.Enum)
            return (KnownValue(member) for member in self.typ)
        else:
            return None

    def substitute_typevars(self, typevars: TypeVarMap) -> "TypedValue":
        return self

    def __str__(self) -> str:
        if self.literal_only:
            if self.typ is str:
                return "LiteralString"
            suffix = " (literal only)"
        else:
            suffix = ""
        return stringify_object(self.typ) + suffix


@dataclass(unsafe_hash=True)
class NewTypeValue(Value):
    """A wrapper around an underlying type.

    Corresponds to ``typing.NewType``.

    This is a subclass of :class:`TypedValue`. Currently only NewTypes over simple,
    non-generic types are supported.

    """

    name: str
    """Name of the ``NewType``."""
    value: Value
    """The underlying value of the ``NewType``."""
    newtype: Any
    """Underlying ``NewType`` object."""

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if isinstance(other, NewTypeValue):
            if self.newtype is other.newtype:
                return None
            return CanAssignError(f"NewTypes {self} and {other} cannot overlap")
        return super().can_overlap(other, ctx, mode)

    def get_type_value(self) -> Value:
        return self.value.get_type_value()

    def get_fallback_value(self) -> Value:
        return self.value

    def __str__(self) -> str:
        return f"NewType({self.name!r}, {self.value})"


@dataclass(unsafe_hash=True, init=False)
class GenericValue(TypedValue):
    """Subclass of :class:`TypedValue` that can represent generics.

    For example, ``List[int]`` is represented as ``GenericValue(list, [TypedValue(int)])``.

    """

    args: tuple[Value, ...]
    """The generic arguments to the type."""

    def __init__(self, typ: type | str, args: Iterable[Value]) -> None:
        super().__init__(typ)
        args = tuple(args)
        assert all(isinstance(arg, Value) for arg in args), args
        self.args = args

    def __str__(self) -> str:
        if self.typ is tuple:
            args = [*self.args, "..."]
        else:
            args = self.args
        args_str = ", ".join(str(arg) for arg in args)
        return f"{stringify_object(self.typ)}[{args_str}]"

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if isinstance(other, GenericValue) and self.typ is other.typ:
            if len(self.args) != len(other.args):
                return CanAssignError(f"Cannot overlap {self} and {other}")
            for i, (my_arg, their_arg) in enumerate(zip(self.args, other.args)):
                maybe_error = my_arg.can_overlap(their_arg, ctx, mode)
                if maybe_error is not None:
                    return CanAssignError(
                        f"In generic argument {i} to {self}", [maybe_error]
                    )
            return None
        return super().can_overlap(other, ctx, mode)

    def get_arg(self, index: int) -> Value:
        try:
            return self.args[index]
        except IndexError:
            return AnyValue(AnySource.generic_argument)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for arg in self.args:
            yield from arg.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> "GenericValue":
        new_args: list[Value] = []
        for arg in self.args:
            substituted = arg.substitute_typevars(typevars)
            if isinstance(arg, TypeVarTupleBindingValue):
                assert isinstance(substituted, TypeVarTupleBindingValue), substituted
                new_args.extend(
                    typevartuple_binding_to_generic_args(substituted.binding)
                )
                continue
            if isinstance(arg, TypeVarTupleValue) and substituted is not arg:
                if isinstance(substituted, TypeVarTupleBindingValue):
                    new_args.extend(
                        typevartuple_binding_to_generic_args(substituted.binding)
                    )
                    continue
            new_args.append(substituted)
        return GenericValue(self.typ, new_args)

    def simplify(self) -> Value:
        return GenericValue(self.typ, [arg.simplify() for arg in self.args])

    def decompose(self) -> Iterable[Value] | None:
        if self.typ is tuple and len(self.args) == 1:
            # either it's empty, or it has at least one element
            arg = self.args[0]
            return [KnownValue(()), SequenceValue(tuple, [(False, arg), (True, arg)])]
        else:
            return None


@dataclass(unsafe_hash=True, init=False)
class SequenceValue(GenericValue):
    """A :class:`TypedValue` subclass representing a sequence of known type.

    This is represented as a sequence, but each entry in the sequence may
    consist of multiple values.
    For example, the expression ``[int(self.foo)]`` may be typed as
    ``SequenceValue(list, [(False, TypedValue(int))])``. The expression
    ``["x", *some_str.split()]`` would be represented as
    ``SequenceValue(list, [(False, KnownValue("x")), (True, TypedValue(str))])``.

    This is only used for ``set``, ``list``, and ``tuple``.

    """

    members: SequenceMembers
    """The elements of the sequence."""

    def __init__(self, typ: type | str, members: Sequence[SequenceMember]) -> None:
        if members:
            args = (unite_values(*[typ for _, typ in members]),)
        elif typ is tuple:
            args = (NO_RETURN_VALUE,)
        else:
            # Using Never for mutable types leads to issues
            args = (AnyValue(AnySource.unreachable),)
        super().__init__(typ, args)
        self.members = tuple(members)

    def get_member_sequence(self) -> Sequence[Value] | None:
        """Return the :class:`Value` objects in this sequence. Return
        None if there are any unpacked values in the sequence."""
        members = []
        for is_many, member in self.members:
            if is_many:
                return None
            members.append(member)
        return members

    def make_known_value(self) -> Value:
        """Turn this value into a KnownValue if possible."""
        if isinstance(self.typ, str):
            return self
        return self.make_or_known(self.typ, self.members)

    @classmethod
    def make_or_known(
        cls, typ: type, members: Sequence[SequenceMember]
    ) -> Union[KnownValue, "SequenceValue"]:
        known_members = []
        for is_many, member in members:
            if is_many or not isinstance(member, KnownValue):
                return SequenceValue(typ, members)
            known_members.append(member.val)
        try:
            return KnownValue(typ(known_members))
        except TypeError:
            # Probably an unhashable object in a set.
            return SequenceValue(typ, members)

    def substitute_typevars(self, typevars: TypeVarMap) -> "SequenceValue":
        new_members: list[SequenceMember] = []
        for is_many, member in self.members:
            substituted = member.substitute_typevars(typevars)
            if (
                is_many
                and isinstance(member, TypeVarTupleValue)
                and self.typ is tuple
                and substituted is not member
            ):
                if isinstance(substituted, TypeVarTupleBindingValue):
                    new_members.extend(substituted.binding)
                    continue
                if isinstance(substituted, TypeVarTupleValue):
                    new_members.append((True, substituted))
                    continue
                substituted = replace_known_sequence_value(substituted)
                if isinstance(substituted, SequenceValue) and substituted.typ is tuple:
                    new_members.extend(substituted.members)
                    continue
            new_members.append((is_many, substituted))
        return SequenceValue(self.typ, new_members)

    def __str__(self) -> str:
        members = ", ".join(
            (f"*tuple[{m}, ...]" if is_many else str(m)) for is_many, m in self.members
        )
        if self.typ is tuple:
            if not members:
                return "tuple[()]"
            return f"tuple[{members}]"
        return f"<{stringify_object(self.typ)} containing [{members}]>"

    def walk_values(self) -> Iterable[Value]:
        yield self
        for _, member in self.members:
            yield from member.walk_values()

    def simplify(self) -> GenericValue:
        if self.typ is tuple:
            return SequenceValue(
                tuple,
                [(is_many, member.simplify()) for is_many, member in self.members],
            )
        members = [member.simplify() for _, member in self.members]
        arg = unite_values(*members)
        if arg is NO_RETURN_VALUE:
            arg = AnyValue(AnySource.unreachable)
        return GenericValue(self.typ, [arg])

    def decompose(self) -> Iterable[Value] | None:
        if not self.members:
            return None
        if self.members[0][0]:
            return [
                # treat it as empty
                SequenceValue(self.typ, self.members[1:]),
                # it has at least one member
                SequenceValue(self.typ, [(False, self.members[0][1]), *self.members]),
            ]
        elif self.members[-1][0]:
            return [
                # treat it as empty
                SequenceValue(self.typ, self.members[:-1]),
                # it has at least one member
                SequenceValue(self.typ, [*self.members, (False, self.members[-1][1])]),
            ]
        else:
            # For simplicity, decompose a single fixed-position member.
            for index, (_, member) in enumerate(self.members):
                decomposed = member.decompose()
                if decomposed is None:
                    continue
                prefix = list(self.members[:index])
                suffix = list(self.members[index + 1 :])
                return [
                    SequenceValue(self.typ, [*prefix, (False, val), *suffix])
                    for val in decomposed
                ]
            return None


@dataclass(frozen=True)
class KVPair:
    """Represents a single entry in a :class:`DictIncompleteValue`."""

    key: Value
    """Represents the key."""
    value: Value
    """Represents the value."""
    is_many: bool = False
    """Whether this key-value pair represents possibly multiple keys."""
    is_required: bool = True
    """Whether this key-value pair is definitely present."""

    def substitute_typevars(self, typevars: TypeVarMap) -> "KVPair":
        return KVPair(
            self.key.substitute_typevars(typevars),
            self.value.substitute_typevars(typevars),
            self.is_many,
            self.is_required,
        )

    def __str__(self) -> str:
        query = "" if self.is_required else "?"
        text = f"{self.key}{query}: {self.value}"
        if self.is_many:
            return f"**{{{text}}}"
        else:
            return text


@dataclass(unsafe_hash=True, init=False)
class DictIncompleteValue(GenericValue):
    """A :class:`TypedValue` representing a dictionary of known size.

    For example, the expression ``{'foo': int(self.bar)}`` may be typed as
    ``DictIncompleteValue(dict, [KVPair(KnownValue('foo'), TypedValue(int))])``.

    """

    kv_pairs: tuple[KVPair, ...]
    """Sequence of :class:`KVPair` objects representing the keys and values of the dict."""

    def __init__(self, typ: type | str, kv_pairs: Sequence[KVPair]) -> None:
        if kv_pairs:
            key_type = unite_values(*[pair.key for pair in kv_pairs])
            value_type = unite_values(*[pair.value for pair in kv_pairs])
        else:
            key_type = value_type = AnyValue(AnySource.unreachable)
        super().__init__(typ, (key_type, value_type))
        self.kv_pairs = tuple(kv_pairs)

    def __str__(self) -> str:
        items = ", ".join(map(str, self.kv_pairs))
        return f"<{stringify_object(self.typ)} containing {{{items}}}>"

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for pair in self.kv_pairs:
            yield from pair.key.walk_values()
            yield from pair.value.walk_values()

    def substitute_typevars(self, typevars: TypeVarMap) -> "DictIncompleteValue":
        return DictIncompleteValue(
            self.typ, [pair.substitute_typevars(typevars) for pair in self.kv_pairs]
        )

    def simplify(self) -> GenericValue:
        keys = [pair.key.simplify() for pair in self.kv_pairs]
        values = [pair.value.simplify() for pair in self.kv_pairs]
        key = unite_values(*keys)
        value = unite_values(*values)
        if key is NO_RETURN_VALUE:
            key = AnyValue(AnySource.unreachable)
        if value is NO_RETURN_VALUE:
            value = AnyValue(AnySource.unreachable)
        return GenericValue(self.typ, [key, value])

    @property
    def items(self) -> Sequence[tuple[Value, Value]]:
        """Sequence of pairs representing the keys and values of the dict."""
        return [(pair.key, pair.value) for pair in self.kv_pairs]

    def get_value(self, key: Value, ctx: CanAssignContext) -> Value:
        """Return the :class:`Value` for a specific key."""
        possible_values = []
        covered_keys: set[Value] = set()
        for pair in reversed(self.kv_pairs):
            if not pair.is_many:
                my_key = replace_fallback(pair.key)
                if isinstance(my_key, KnownValue):
                    if my_key == key and pair.is_required:
                        return unite_values(*possible_values, pair.value)
                    elif my_key in covered_keys:
                        continue
                    elif pair.is_required:
                        covered_keys.add(my_key)
            if key.is_assignable(pair.key, ctx) or pair.key.is_assignable(key, ctx):
                possible_values.append(pair.value)
        if not possible_values:
            return UNINITIALIZED_VALUE
        return unite_values(*possible_values)


@dataclass(frozen=True)
class TypedDictEntry:
    typ: Value
    required: bool = True
    readonly: bool = False

    def __str__(self) -> str:
        val = str(self.typ)
        if self.readonly:
            val = f"Readonly[{val}]"
        if not self.required:
            val = f"NotRequired[{val}]"
        return val


@dataclass(init=False)
class TypedDictValue(GenericValue):
    """Equivalent to ``typing.TypedDict``; a dictionary with a known set of string keys."""

    items: dict[str, TypedDictEntry]
    """The items of the ``TypedDict``. Required items are represented as (True, value) and optional
    ones as (False, value)."""
    extra_keys: Value | None = None
    """The type of unknown keys, if any."""
    extra_keys_readonly: bool = False
    """Whether the extra keys are readonly."""

    def __init__(
        self,
        items: dict[str, TypedDictEntry],
        extra_keys: Value | None = None,
        extra_keys_readonly: bool = False,
    ) -> None:
        # Compatibility with old format, where values were (required, type) tuples.
        items = {
            key: (
                value
                if isinstance(value, TypedDictEntry)
                else TypedDictEntry(value[1], required=value[0])
            )
            for key, value in items.items()
        }
        value_types = []
        if items:
            value_types += [val.typ for val in items.values()]
        if extra_keys is not None:
            value_types.append(extra_keys)
        value_type = (
            unite_values(*value_types)
            if value_types
            else AnyValue(AnySource.unreachable)
        )
        # The key type must be str so dict[str, Any] is compatible with a TypedDict
        key_type = TypedValue(str)
        super().__init__(dict, (key_type, value_type))
        self.items = items
        self.extra_keys = extra_keys
        self.extra_keys_readonly = extra_keys_readonly

    def num_required_keys(self) -> int:
        return sum(1 for entry in self.items.values() if entry.required)

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        other = replace_known_sequence_value(other)
        if isinstance(other, TypedDictValue):
            for key, entry in self.items.items():
                if key not in other.items:
                    if entry.required:
                        if other.extra_keys is None:
                            return CanAssignError(f"Key {key} is missing in {other}")
                        else:
                            maybe_error = entry.typ.can_overlap(
                                other.extra_keys, ctx, mode
                            )
                            if maybe_error is not None:
                                return CanAssignError(
                                    f"Type for key {key} is incompatible with extra keys"
                                    f" type {other.extra_keys}",
                                    children=[maybe_error],
                                )
                else:
                    their_entry = other.items[key]
                    maybe_error = entry.typ.can_overlap(their_entry.typ, ctx, mode)
                    if maybe_error is not None:
                        return CanAssignError(
                            f"Types for key {key} cannot overlap",
                            children=[maybe_error],
                        )
            for key, entry in other.items.items():
                if key not in self.items and entry.required:
                    if self.extra_keys is None:
                        return CanAssignError(f"Key {key} is missing in {self}")
                    else:
                        maybe_error = entry.typ.can_overlap(self.extra_keys, ctx, mode)
                        if maybe_error is not None:
                            return CanAssignError(
                                f"Type for key {key} is incompatible with extra keys"
                                f" type {self.extra_keys}",
                                children=[maybe_error],
                            )
        return super().can_overlap(other, ctx, mode)

    def substitute_typevars(self, typevars: TypeVarMap) -> "TypedDictValue":
        return TypedDictValue(
            {
                key: TypedDictEntry(
                    entry.typ.substitute_typevars(typevars),
                    required=entry.required,
                    readonly=entry.readonly,
                )
                for key, entry in self.items.items()
            },
            extra_keys=(
                self.extra_keys.substitute_typevars(typevars)
                if self.extra_keys is not None
                else None
            ),
            extra_keys_readonly=self.extra_keys_readonly,
        )

    def __str__(self) -> str:
        items = [f'"{key}": {entry}' for key, entry in self.items.items()]
        if self.extra_keys is not None and self.extra_keys is not NO_RETURN_VALUE:
            extra_typ = str(self.extra_keys)
            if self.extra_keys_readonly:
                extra_typ = f"ReadOnly[{extra_typ}]"
            items.append(f'"__extra_items__": {extra_typ}')
        closed = ", closed=True" if self.extra_keys is not None else ""
        return f"TypedDict({{{', '.join(items)}}}{closed})"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items)))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for entry in self.items.values():
            yield from entry.typ.walk_values()


@dataclass(unsafe_hash=True)
class SyntheticClassObjectValue(Value):
    """Represents a singleton class object that exists but has no runtime object."""

    name: str
    class_type: TypedValue | TypedDictValue

    def substitute_typevars(self, typevars: TypeVarMap) -> "SyntheticClassObjectValue":
        substituted = self.class_type.substitute_typevars(typevars)
        return SyntheticClassObjectValue(self.name, substituted)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.class_type.walk_values()

    def get_type_value(self) -> Value:
        if isinstance(self.class_type, TypedValue) and isinstance(
            self.class_type.typ, type
        ):
            return KnownValue(type(self.class_type.typ))
        return TypedValue(type)

    def get_type_object(
        self, ctx: CanAssignContext
    ) -> "pycroscope.type_object.TypeObject":
        return ctx.make_type_object(self.class_type.typ)

    def __str__(self) -> str:
        return f"<class {self.name!r}>"


@dataclass(unsafe_hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A :class:`GenericValue` representing an async task.

    This should probably just be replaced with ``GenericValue``.

    """

    value: Value
    """The value returned by the task on completion."""

    def __init__(self, typ: type | str, value: Value) -> None:
        super().__init__(typ, (value,))
        self.value = value

    def substitute_typevars(self, typevars: TypeVarMap) -> "AsyncTaskIncompleteValue":
        return AsyncTaskIncompleteValue(
            self.typ, self.value.substitute_typevars(typevars)
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.value.walk_values()


@dataclass(unsafe_hash=True, init=False)
class CallableValue(TypedValue):
    """Equivalent to the ``Callable`` type.

    This is a thin wrapper around :class:`pycroscope.signature.Signature`.

    """

    signature: "pycroscope.signature.ConcreteSignature"

    def __init__(
        self,
        signature: "pycroscope.signature.ConcreteSignature",
        fallback: type | str = collections.abc.Callable,
    ) -> None:
        super().__init__(fallback)
        self.signature = signature

    def substitute_typevars(self, typevars: TypeVarMap) -> "CallableValue":
        from .signature import keep_inferable_typevars_from_params

        return CallableValue(
            keep_inferable_typevars_from_params(
                self.signature.substitute_typevars(
                    typevars, infer_substituted_typevars=True
                )
            ),
            self.typ,
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.signature.walk_values()

    def get_asynq_value(self) -> Value:
        """Return the CallableValue for the .asynq attribute of an AsynqCallable."""
        sig = self.signature.get_asynq_value()
        return CallableValue(sig, self.typ)

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if not isinstance(other, (MultiValuedValue, AnyValue, AnnotatedValue)):
            signature = ctx.signature_from_value(other)
            return _signatures_overlap(self.signature, signature, ctx)
        return super().can_overlap(other, ctx, mode)

    def __str__(self) -> str:
        return str(self.signature)


def _signatures_overlap(
    left: "pycroscope.signature.MaybeSignature",
    right: "pycroscope.signature.MaybeSignature",
    ctx: CanAssignContext,
) -> CanAssignError | None:
    if left is None or right is None:
        return CanAssignError("Not a callable type")
    if isinstance(left, pycroscope.signature.BoundMethodSignature):
        left = left.get_signature(ctx=ctx)
    if isinstance(right, pycroscope.signature.BoundMethodSignature):
        right = right.get_signature(ctx=ctx)
    if isinstance(left, pycroscope.signature.Signature) and isinstance(
        right, pycroscope.signature.Signature
    ):
        left_errors = left.can_assign(right, ctx)
        if not isinstance(left_errors, CanAssignError):
            return None
        right_errors = right.can_assign(left, ctx)
        if not isinstance(right_errors, CanAssignError):
            return None
        return CanAssignError(
            f"Signatures {left} and {right} cannot overlap",
            children=[left_errors, right_errors],
        )
    return None


@dataclass(frozen=True)
class SubclassValue(Value):
    """Equivalent of ``Type[]``.

    The `typ` attribute can be either a :class:`TypedValue` or a
    :class:`TypeVarValue`. The former is equivalent to ``Type[int]``
    and represents the ``int`` class or a subclass. The latter is
    equivalent to ``Type[T]`` where ``T`` is a type variable.
    The third legal argument to ``Type[]`` is ``Any``, but
    ``Type[Any]`` is represented as ``TypedValue(type)``.

    """

    typ: Union[TypedValue, "TypeVarValue"]
    """The underlying type."""

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return self.make(self.typ.substitute_typevars(typevars))

    def get_type_object(
        self, ctx: CanAssignContext
    ) -> "pycroscope.type_object.TypeObject":
        if isinstance(self.typ, TypedValue) and safe_isinstance(self.typ.typ, type):
            return ctx.make_type_object(type(self.typ.typ))
        # TODO synthetic types
        return ctx.make_type_object(object)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.typ.walk_values()

    def is_type(self, typ: type) -> bool:
        if isinstance(self.typ, TypedValue) and isinstance(self.typ.typ, type):
            return safe_issubclass(self.typ.typ, typ)
        return False

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if isinstance(other, (KnownValue, TypedValue)):
            can_assign = self.can_assign(other, ctx)
            if not isinstance(can_assign, CanAssignError):
                return None
            return can_assign
        elif isinstance(other, SubclassValue):
            left_errors = self.can_assign(other, ctx)
            if not isinstance(left_errors, CanAssignError):
                return None
            right_errors = other.can_assign(self, ctx)
            if not isinstance(right_errors, CanAssignError):
                return None
            return CanAssignError(
                f"Types {self} and {other} cannot overlap",
                children=[left_errors, right_errors],
            )
        return super().can_overlap(other, ctx, mode)

    def get_type(self) -> type | None:
        if isinstance(self.typ, TypedValue):
            return type(self.typ.typ)
        else:
            return None

    def get_type_value(self) -> Value:
        typ = self.get_type()
        if typ is not None:
            return KnownValue(typ)
        else:
            return AnyValue(AnySource.inference)

    def __str__(self) -> str:
        return f"type[{self.typ}]"

    @classmethod
    def make(cls, origin: Value) -> Value:
        if isinstance(origin, MultiValuedValue):
            return unite_values(*[cls.make(val) for val in origin.vals])
        elif isinstance(origin, AnyValue):
            # Type[Any] is equivalent to plain type
            return TypedValue(type)
        elif isinstance(origin, KnownValue):
            if origin.val is None:
                return cls(TypedValue(type(None)))
            elif isinstance(origin.val, type):
                return cls(TypedValue(origin.val))
            return AnyValue(AnySource.error)
        elif isinstance(origin, (TypeVarValue, TypedValue)):
            return cls(origin)
        else:
            return AnyValue(AnySource.inference)


@dataclass(frozen=True, order=False)
class IntersectionValue(Value):
    """Represents the intersection of multiple values."""

    vals: tuple[Value, ...]

    def __post_init__(self) -> None:
        assert self.vals, "IntersectionValue must have at least one value"
        for val in self.vals:
            assert not isinstance(
                val, IntersectionValue
            ), "Nested IntersectionValues are not allowed"
            assert not isinstance(
                val, MultiValuedValue
            ), "IntersectionValues cannot contain MultiValuedValues"

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return IntersectionValue(
            tuple(val.substitute_typevars(typevars) for val in self.vals)
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        for val in self.vals:
            yield from val.walk_values()

    def get_type_value(self) -> Value:
        return IntersectionValue(tuple(val.get_type_value() for val in self.vals))

    def __str__(self) -> str:
        return " & ".join(str(val) for val in self.vals)


@dataclass(frozen=True, order=False)
class OverlappingValue(Value):
    """Represents ``Overlapping[T]``."""

    type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return OverlappingValue(self.type.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.type.walk_values()

    def get_fallback_value(self) -> Value:
        return TypedValue(object)

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def __str__(self) -> str:
        return f"Overlapping[{self.type}]"


@dataclass(frozen=True, order=False)
class MultiValuedValue(Value):
    """Equivalent of ``typing.Union``. Represents the union of multiple values."""

    raw_vals: InitVar[Iterable[Value]]
    vals: tuple[Value, ...] = field(init=False)
    """The underlying values of the union."""

    def __post_init__(self, raw_vals: Iterable[Value]) -> None:
        object.__setattr__(
            self,
            "vals",
            tuple(chain.from_iterable(flatten_values(val) for val in raw_vals)),
        )

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        if not self.vals or not typevars:
            return self
        return MultiValuedValue(
            [val.substitute_typevars(typevars) for val in self.vals]
        )

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        if not self.vals:
            return None
        errors: list[CanAssignError] = []
        for val in self.vals:
            error = val.can_overlap(other, ctx, mode)
            if error is None:
                return None
            errors.append(error)
        return CanAssignError("Cannot overlap with Union", errors)

    def get_type_value(self) -> Value:
        if not self.vals:
            return self
        return MultiValuedValue([val.get_type_value() for val in self.vals])

    def decompose(self) -> Iterable[Value]:
        return self.vals

    def __eq__(self, other: Value) -> bool:
        if not isinstance(other, MultiValuedValue):
            return NotImplemented
        if self.vals == other.vals:
            return True
        # try to put the values in a set so different objects that happen to have different order
        # compare equal, but don't worry if some aren't hashable
        try:
            left_vals = set(self.vals)
            right_vals = set(other.vals)
        except Exception:
            return False
        return left_vals == right_vals

    def __ne__(self, other: Value) -> bool:
        return not (self == other)

    def __str__(self) -> str:
        if not self.vals:
            return "Never"
        literals: list[KnownValue] = []
        has_none = False
        others: list[Value] = []
        for val in self.vals:
            if val == KnownValue(None):
                has_none = True
            elif isinstance(val, KnownValue):
                literals.append(val)
            else:
                others.append(val)
        if not others:
            if has_none:
                literals.append(KnownValue(None))
            body = ", ".join(repr(val.val) for val in literals)
            return f"Literal[{body}]"
        else:
            elements = [str(val) for val in others]
            if literals:
                body = ", ".join(repr(val.val) for val in literals)
                elements.append(f"Literal[{body}]")
            if has_none:
                elements.append("None")
            return " | ".join(elements)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for val in self.vals:
            yield from val.walk_values()

    def simplify(self) -> Value:
        return unite_values(*[val.simplify() for val in self.vals])


NO_RETURN_VALUE = MultiValuedValue([])
"""The empty union, equivalent to ``typing.Never``."""


@dataclass(frozen=True)
class ReferencingValue(Value):
    """Value that is a reference to another value (used to implement globals)."""

    scope: Any
    name: str

    def __str__(self) -> str:
        return f"<reference to {self.name}>"


# Special TypeVar used to implement PEP 673 Self.
SelfT = TypeVar("SelfT")


@dataclass(frozen=True)
class Bound:
    pass


@dataclass(frozen=True)
class LowerBound(Bound):
    """LowerBound(T, V) means V must be assignable to the value of T."""

    type_param: TypeParam
    value: Value

    def __str__(self) -> str:
        return f"{self.type_param} >= {self.value}"


@dataclass(frozen=True)
class UpperBound(Bound):
    """UpperBound(T, V) means the value of T must be assignable to V."""

    type_param: TypeParam
    value: Value

    def __str__(self) -> str:
        return f"{self.type_param} <= {self.value}"


@dataclass(frozen=True)
class OrBound(Bound):
    """At least one of the specified bounds must be true."""

    bounds: Sequence[Sequence[Bound]]


@dataclass(frozen=True)
class IsOneOf(Bound):
    type_param: TypeParam
    constraints: Sequence[Value]


@dataclass(frozen=True)
class TypeVarValue(Value):
    """Value representing a ``typing.TypeVar``."""

    typevar_param: TypeVarParam

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get_value(self.typevar_param, self)

    def get_inherent_bounds(self) -> Iterator[Bound]:
        if self.typevar_param.bound is not None:
            yield UpperBound(self.typevar_param, self.typevar_param.bound)
        elif self.typevar_param.constraints:
            yield IsOneOf(self.typevar_param, self.typevar_param.constraints)
        # TODO: Consider adding this, but it leads to worse type inference
        # in some cases (inferring object where we should infer Any). Examples
        # in the taxonomy repo.
        # else:
        #     yield UpperBound(self.typevar, TypedValue(object))

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return self.get_fallback_value().can_overlap(other, ctx, mode)

    def get_upper_bound_value(self) -> Value:
        if self.typevar_param.bound is not None:
            return self.typevar_param.bound
        elif self.typevar_param.constraints:
            return unite_values(*self.typevar_param.constraints)
        return TypedValue(object)

    def get_fallback_value(self) -> Value:
        if self.typevar_param.bound is not None:
            return self.typevar_param.bound
        elif self.typevar_param.constraints:
            return unite_values(*self.typevar_param.constraints)
        return AnyValue(AnySource.inference)  # TODO: should be object

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def __str__(self) -> str:
        return str(self.typevar_param)


@dataclass(frozen=True)
class InferenceVarValue(TypeVarValue):
    """A fresh inference variable created from a declared TypeVar."""

    def make_bounds_map(
        self, bounds: Sequence[Bound], other: Value, ctx: CanAssignContext
    ) -> CanAssign:
        bounds_map = {self.typevar_param.typevar: bounds}
        if self.typevar_param.bound is None and not self.typevar_param.constraints:
            return bounds_map
        _, errors = pycroscope.typevar.resolve_bounds_map(bounds_map, ctx)
        if errors:
            return CanAssignError(f"Value of {self} cannot be {other}", list(errors))
        return bounds_map


@dataclass(frozen=True)
class TypeVarTupleValue(Value):
    typevar_tuple_param: TypeVarTupleParam

    @property
    def typevar(self) -> TypeVarTupleLike:
        return self.typevar_tuple_param.typevar

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get_value(self.typevar_tuple_param, self)

    def get_inherent_bounds(self) -> Iterator[Bound]:
        return iter(())

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return self.get_fallback_value().can_overlap(other, ctx, mode)

    def get_fallback_value(self) -> Value:
        return AnyValue(AnySource.inference)

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def __str__(self) -> str:
        return str(self.typevar)


@dataclass(frozen=True)
class TypeVarTupleBindingValue(Value):
    binding: SequenceMembers

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return TypeVarTupleBindingValue(
            substitute_typevartuple_binding(self.binding, typevars)
        )

    def get_inherent_bounds(self) -> Iterator[Bound]:
        return iter(())

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return typevartuple_binding_to_tuple_value(self.binding).can_overlap(
            other, ctx, mode
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        for _, value in self.binding:
            yield from value.walk_values()

    def __str__(self) -> str:
        parts = []
        for is_many, value in self.binding:
            if is_many and isinstance(value, TypeVarTupleValue):
                parts.append(str(value))
            elif is_many:
                parts.append(f"*tuple[{value}, ...]")
            else:
                parts.append(str(value))
        return ", ".join(parts) if parts else "tuple[()]"


SelfTVV = TypeVarValue(TypeVarParam(SelfT))
_NestedSelfT = TypeVar("_NestedSelfT")


def is_self_typevar(
    typevar: TypeVarType, *, include_nested_placeholders: bool = False
) -> bool:
    return typevar is SelfT or (include_nested_placeholders and typevar is _NestedSelfT)


def is_self_typevar_value(
    value: Value, *, include_nested_placeholders: bool = False
) -> bool:
    return isinstance(value, TypeVarValue) and is_self_typevar(
        value.typevar_param.typevar,
        include_nested_placeholders=include_nested_placeholders,
    )


def bound_self_type_from_class_key(
    current_class_key: Value | type | str,
) -> TypeVarValue:
    bound = (
        current_class_key
        if isinstance(current_class_key, Value)
        else TypedValue(current_class_key)
    )
    return TypeVarValue(TypeVarParam(SelfT, bound=bound))


def shield_nested_self_typevars(value: Value) -> tuple[Value, TypeVarMap]:
    """Protect nested ``Self`` values while specializing an outer receiver."""
    first_self_typevar = next(
        (
            subval
            for subval in value.walk_values()
            if isinstance(subval, TypeVarValue)
            and subval.typevar_param.typevar is SelfT
        ),
        None,
    )
    if first_self_typevar is None:
        return value, TypeVarMap()
    placeholder = TypeVarValue(
        TypeVarParam(
            _NestedSelfT,
            bound=first_self_typevar.typevar_param.bound,
            default=first_self_typevar.typevar_param.default,
            constraints=first_self_typevar.typevar_param.constraints,
            variance=first_self_typevar.typevar_param.variance,
        )
    )
    return value.substitute_typevars(
        TypeVarMap(typevars={SelfT: placeholder})
    ), TypeVarMap(typevars={_NestedSelfT: first_self_typevar})


def receiver_to_self_type(
    self_value: Value, ctx: CanAssignContext | None = None
) -> Value:
    if isinstance(
        self_value, KnownValueWithTypeVars
    ) and self_value.typevars.has_typevar(TypeVarParam(SelfT)):
        self_substitution = self_value.typevars.get_typevar(TypeVarParam(SelfT))
        assert self_substitution is not None
        return receiver_to_self_type(self_substitution, ctx)
    if (
        ctx is not None
        and isinstance(self_value, KnownValueWithTypeVars)
        and not isinstance(self_value.val, type)
    ):
        runtime_type = type(self_value.val)
        type_params = ctx.get_type_parameters(runtime_type)
        if type_params:
            return GenericValue(
                runtime_type,
                [
                    self_value.typevars.get_value(
                        type_param, default_value_for_type_param(type_param)
                    )
                    for type_param in type_params
                ],
            )
    if isinstance(self_value, SequenceValue):
        if self_value.typ in (list, set):
            return self_value.simplify()
        return self_value
    if isinstance(self_value, DictIncompleteValue):
        return self_value.simplify()
    if isinstance(self_value, KnownValue):
        replaced = replace_known_sequence_value(self_value)
        if isinstance(replaced, SequenceValue) and replaced.typ in (list, set):
            return replaced.simplify()
        if isinstance(replaced, DictIncompleteValue):
            return replaced.simplify()
        if not isinstance(replaced, KnownValue):
            return replaced
        return TypedValue(
            replaced.val if isinstance(replaced.val, type) else type(replaced.val)
        )
    if isinstance(self_value, SubclassValue):
        return self_value.typ
    return self_value


def _has_nested_self_typevar(value: Value) -> bool:
    return not (
        isinstance(value, TypeVarValue) and value.typevar_param.typevar is SelfT
    ) and any(
        isinstance(subval, TypeVarValue) and subval.typevar_param.typevar is SelfT
        for subval in value.walk_values()
    )


def set_self(value: Value, self_value: Value) -> Value:
    self_type = receiver_to_self_type(self_value)
    if _has_nested_self_typevar(self_type):
        return value
    self_type, restore_typevars = shield_nested_self_typevars(self_type)
    if isinstance(value, KnownValueWithTypeVars):
        merged_typevars = value.typevars.with_typevar(TypeVarParam(SelfT), self_type)
        result: Value = KnownValueWithTypeVars(value.val, merged_typevars)
    else:
        result = value.substitute_typevars(TypeVarMap(typevars={SelfT: self_type}))
    if restore_typevars:
        result = result.substitute_typevars(restore_typevars)
    return result


@dataclass(frozen=True)
class ParamSpecArgsValue(Value):
    param_spec: ParamSpecLike

    def __str__(self) -> str:
        return f"{self.param_spec}.args"

    def get_fallback_value(self) -> Value:
        return GenericValue(tuple, [TypedValue(object)])


@dataclass(frozen=True)
class ParamSpecKwargsValue(Value):
    param_spec: ParamSpecLike

    def __str__(self) -> str:
        return f"{self.param_spec}.kwargs"

    def get_fallback_value(self) -> Value:
        return GenericValue(dict, [TypedValue(str), TypedValue(object)])


class Extension:
    """An extra piece of information about a type that can be stored in
    an :class:`AnnotatedValue`."""

    __slots__ = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return self

    def walk_values(self) -> Iterable[Value]:
        return []

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        return {}

    def can_be_assigned(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        return {}


@dataclass(frozen=True)
class CustomCheckExtension(Extension):
    custom_check: CustomCheck

    def __str__(self) -> str:
        # This extra wrapper class just adds noise
        return str(self.custom_check)

    def substitute_typevars(self, typevars: TypeVarMap) -> "Extension":
        return CustomCheckExtension(self.custom_check.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield from self.custom_check.walk_values()

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        return self.custom_check.can_assign(value, ctx)

    def can_be_assigned(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        return self.custom_check.can_be_assigned(value, ctx)


@dataclass(frozen=True)
class ParameterTypeGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the parameter named `varname` is of type `guarded_type`.

    Corresponds to :class:`pycroscope.extensions.ParameterTypeGuard`.

    """

    varname: str
    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return ParameterTypeGuardExtension(self.varname, guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass(frozen=True)
class NoReturnGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that unless the parameter named `varname` is of type `guarded_type`,
    the function does not return.

    Corresponds to :class:`pycroscope.extensions.NoReturnGuard`.

    """

    varname: str
    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return NoReturnGuardExtension(self.varname, guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()


@dataclass(frozen=True)
class TypeGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the first function argument is of type `guarded_type`.

    Corresponds to :class:`pycroscope.extensions.TypeGuard`, or ``typing.TypeGuard``.

    """

    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return TypeGuardExtension(guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        can_assign_maps = []
        if isinstance(value, AnnotatedValue):
            for ext in value.get_metadata_of_type(Extension):
                if isinstance(ext, TypeIsExtension):
                    return CanAssignError("TypeGuard is not compatible with TypeIs")
                elif isinstance(ext, TypeGuardExtension):
                    # TypeGuard is covariant
                    left_can_assign = self.guarded_type.can_assign(
                        ext.guarded_type, ctx
                    )
                    if isinstance(left_can_assign, CanAssignError):
                        return CanAssignError(
                            "Incompatible types in TypeGuard",
                            children=[left_can_assign],
                        )
                    can_assign_maps.append(left_can_assign)
        if not can_assign_maps:
            return CanAssignError(f"{value} is not a TypeGuard")
        return unify_bounds_maps(can_assign_maps)


@dataclass(frozen=True)
class TypeIsExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the first function argument may be narrowed to type `guarded_type`.

    Corresponds to ``typing_extensions.TypeIs`` (see PEP 742).

    """

    guarded_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        guarded_type = self.guarded_type.substitute_typevars(typevars)
        return TypeIsExtension(guarded_type)

    def walk_values(self) -> Iterable[Value]:
        yield from self.guarded_type.walk_values()

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        can_assign_maps = []
        if isinstance(value, AnnotatedValue):
            for ext in value.get_metadata_of_type(Extension):
                if isinstance(ext, TypeGuardExtension):
                    return CanAssignError("TypeGuard is not compatible with TypeIs")
                elif isinstance(ext, TypeIsExtension):
                    # TypeIs is invariant
                    left_can_assign = self.guarded_type.can_assign(
                        ext.guarded_type, ctx
                    )
                    if isinstance(left_can_assign, CanAssignError):
                        return CanAssignError(
                            "Incompatible types in TypeIs", children=[left_can_assign]
                        )
                    right_can_assign = ext.guarded_type.can_assign(
                        self.guarded_type, ctx
                    )
                    if isinstance(right_can_assign, CanAssignError):
                        return CanAssignError(
                            "Incompatible types in TypeIs", children=[right_can_assign]
                        )
                    can_assign_maps += [left_can_assign, right_can_assign]
        if not can_assign_maps:
            return CanAssignError(f"{value} is not a TypeIs")
        return unify_bounds_maps(can_assign_maps)


@dataclass(frozen=True)
class TypeFormValue(Value):
    """Represents a ``typing.TypeForm`` value."""

    inner_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        inner_type = self.inner_type.substitute_typevars(typevars)
        return TypeFormValue(inner_type)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.inner_type.walk_values()

    def get_fallback_value(self) -> Value:
        # TypeForm is a subtype of object.
        return TypedValue(object)

    def __str__(self) -> str:
        return f"TypeForm[{self.inner_type}]"


@dataclass(frozen=True)
class AddPredicateExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the function argument named `varname` should receive
    the predicate `predicate`.

    """

    varname: str
    predicate: "Predicate"

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return AddPredicateExtension(
            self.varname, self.predicate.substitute_typevars(typevars)
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.predicate.walk_values()


@dataclass(frozen=True, eq=False)
class ConstraintExtension(Extension):
    """Encapsulates a Constraint. If the value is evaluated and is truthy, the
    constraint must be True."""

    constraint: "pycroscope.stacked_scopes.AbstractConstraint"

    # Comparing them can get too expensive
    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        return str(self.constraint)


@dataclass(frozen=True, eq=False)
class NoReturnConstraintExtension(Extension):
    """Encapsulates a Constraint. If the value is evaluated and completes, the
    constraint must be True."""

    constraint: "pycroscope.stacked_scopes.AbstractConstraint"

    # Comparing them can get too expensive
    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True)
class AlwaysPresentExtension(Extension):
    """Extension that indicates that an iterable value is nonempty.

    Currently cannot be used from user code.

    """


@dataclass(frozen=True)
class AssertErrorExtension(Extension):
    """Used for the implementation of :func:`pycroscope.extensions.assert_error`."""


@dataclass(frozen=True)
class SkipDeprecatedExtension(Extension):
    """Indicates that use of this value should not trigger deprecation errors."""


@dataclass(frozen=True)
class DeprecatedExtension(Extension):
    """Indicates that use of this value should trigger a deprecation error."""

    deprecation_message: str


@dataclass(frozen=True)
class SysPlatformExtension(Extension):
    """Used for sys.platform."""


SYS_PLATFORM_EXTENSION = SysPlatformExtension()


@dataclass(frozen=True)
class SysVersionInfoExtension(Extension):
    """Used for sys.version_info."""


SYS_VERSION_INFO_EXTENSION = SysVersionInfoExtension()


@dataclass(frozen=True)
class DefiniteValueExtension(Extension):
    """Used if a comparison has a definite value that should be used
    to skip type checking."""

    value: bool


@dataclass(frozen=True)
class DataclassTransformInfo:
    eq_default: bool = True
    frozen_default: bool = False
    kw_only_default: bool = False
    order_default: bool = False
    field_specifiers: tuple[Value, ...] = ()

    def substitute_typevars(self, typevars: TypeVarMap) -> "DataclassTransformInfo":
        return DataclassTransformInfo(
            eq_default=self.eq_default,
            frozen_default=self.frozen_default,
            kw_only_default=self.kw_only_default,
            order_default=self.order_default,
            field_specifiers=tuple(
                value.substitute_typevars(typevars) for value in self.field_specifiers
            ),
        )

    def walk_values(self) -> Iterable[Value]:
        for field_specifier in self.field_specifiers:
            yield from field_specifier.walk_values()


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


@dataclass(frozen=True)
class DataclassFieldInfo:
    has_default: bool = False
    init: bool = True
    kw_only: bool = False
    alias: str | None = None
    converter_input_type: Value | None = None

    def substitute_typevars(self, typevars: TypeVarMap) -> "DataclassFieldInfo":
        return DataclassFieldInfo(
            has_default=self.has_default,
            init=self.init,
            kw_only=self.kw_only,
            alias=self.alias,
            converter_input_type=(
                self.converter_input_type.substitute_typevars(typevars)
                if self.converter_input_type is not None
                else None
            ),
        )

    def walk_values(self) -> Iterable[Value]:
        if self.converter_input_type is not None:
            yield from self.converter_input_type.walk_values()


@dataclass(frozen=True, kw_only=True)
class PropertyInfo:
    # Can be None if a property is manually constructed (property(fset=...))
    fget: "ClassSymbol | None"
    fset: "ClassSymbol | None" = None
    fdel: "ClassSymbol | None" = None

    def substitute_typevars(self, typevars: TypeVarMap) -> "PropertyInfo":
        return PropertyInfo(
            fget=(
                self.fget.substitute_typevars(typevars)
                if self.fget is not None
                else None
            ),
            fset=(
                self.fset.substitute_typevars(typevars)
                if self.fset is not None
                else None
            ),
            fdel=(
                self.fdel.substitute_typevars(typevars)
                if self.fdel is not None
                else None
            ),
        )


@dataclass(frozen=True)
class DataclassTransformExtension(Extension):
    info: DataclassTransformInfo

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return DataclassTransformExtension(self.info.substitute_typevars(typevars))

    def walk_values(self) -> Iterable[Value]:
        yield from self.info.walk_values()


@dataclass(frozen=True)
class DataclassTransformDecoratorExtension(Extension):
    info: DataclassTransformInfo

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return DataclassTransformDecoratorExtension(
            self.info.substitute_typevars(typevars)
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.info.walk_values()


@dataclass(frozen=True)
class AnnotatedValue(Value):
    """Value representing a `PEP 593 <https://www.python.org/dev/peps/pep-0593/>`_ Annotated object.

    Pycroscope uses ``Annotated`` types to represent types with some extra
    information added to them in the form of :class:`Extension` objects.

    """

    value: Value
    """The underlying value."""
    metadata: tuple[Extension, ...]
    """The extensions associated with this value."""

    def __init__(self, value: Value, metadata: Sequence[Extension]) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", tuple(metadata))

    def is_type(self, typ: type) -> bool:
        return self.value.is_type(typ)

    def get_type(self) -> type | None:
        return self.value.get_type()

    def get_type_value(self) -> Value:
        return self.value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        metadata = tuple(val.substitute_typevars(typevars) for val in self.metadata)
        return AnnotatedValue(self.value.substitute_typevars(typevars), metadata)

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return self.value.can_overlap(other, ctx, mode)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.value.walk_values()
        for val in self.metadata:
            yield from val.walk_values()

    def get_metadata_of_type(self, typ: type[T]) -> Iterable[T]:
        """Return any metadata of the given type."""
        for data in self.metadata:
            if isinstance(data, typ):
                yield data

    def get_custom_check_of_type(self, typ: type[T]) -> Iterable[T]:
        """Return any CustomChecks of the given type in the metadata."""
        for custom_check in self.get_metadata_of_type(CustomCheckExtension):
            if isinstance(custom_check.custom_check, typ):
                yield custom_check.custom_check

    def has_metadata_of_type(self, typ: type[Extension]) -> bool:
        """Return whether there is metadat of the given type."""
        return any(isinstance(data, typ) for data in self.metadata)

    def __str__(self) -> str:
        return f"Annotated[{self.value}, {', '.join(map(str, self.metadata))}]"

    def simplify(self) -> Value:
        return AnnotatedValue(self.value.simplify(), self.metadata)

    def get_fallback_value(self) -> Value:
        return self.value


@dataclass(frozen=True)
class VariableNameValue(AnyValue):
    """Value that is stored in a variable associated with a particular kind of value.

    For example, any variable named `uid` will get resolved into a ``VariableNameValue``
    of type `uid`,
    and if it gets passed into a function that takes an argument called `aid`,
    the call will be rejected.

    This was created for a legacy codebase without type annotations. If possible, prefer
    using NewTypes or other more explicit types.

    There should only be a limited set of ``VariableNameValue`` objects,
    created through the pycroscope configuration.

    """

    def __init__(self, varnames: Iterable[str]) -> None:
        super().__init__(AnySource.variable_name)
        object.__setattr__(self, "varnames", tuple(varnames))

    varnames: tuple[str, ...]

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> CanAssignError | None:
        return None

    def __str__(self) -> str:
        return "<variable name: {}>".format(", ".join(self.varnames))

    @classmethod
    def from_varname(
        cls, varname: str, varname_map: dict[str, "VariableNameValue"]
    ) -> Optional["VariableNameValue"]:
        """Returns the VariableNameValue corresponding to a variable name.

        If there is no VariableNameValue that corresponds to the variable name, returns None.

        """
        if varname in varname_map:
            return varname_map[varname]
        if "_" in varname:
            parts = varname.split("_")
            if parts[-1] == "id":
                shortened_varname = "_".join(parts[-2:])
            else:
                shortened_varname = parts[-1]
            return varname_map.get(shortened_varname)
        return None


class Predicate:
    """Represents a predicate on a value, such as "has an attribute named 'x' of type int"."""

    def has_relation(
        self,
        other: "GradualType",
        relation: "pycroscope.relations.Relation",
        ctx: CanAssignContext,
    ) -> bool:
        """Whether this predicate has the given relation to another GradualType.

        For example, whether other is a subtype of this predicate, meaning that all values
        included in the type are guaranteed to satisfy the predicate.
        """
        other = replace_fallback(other)
        if isinstance(other, (MultiValuedValue, IntersectionValue)):
            return False
        return self.has_relation_simple_type(other, relation, ctx)

    def has_relation_simple_type(
        self,
        other: "SimpleType",
        relation: "pycroscope.relations.Relation",
        ctx: CanAssignContext,
    ) -> bool:
        return False

    def intersect_with(
        self, other: "GradualType", ctx: CanAssignContext
    ) -> Value | None:
        """Return a Value representing the intersection of this predicate with another
        GradualType, or None if the intersection is irreducible."""
        other = replace_fallback(other)
        if isinstance(other, (MultiValuedValue, IntersectionValue)):
            return None
        return self.intersect_with_simple_type(other, ctx)

    def intersect_with_simple_type(
        self, other: "SimpleType", ctx: CanAssignContext
    ) -> Value | None:
        return None

    def substitute_typevars(self, typevars: TypeVarMap) -> "Predicate":
        return self

    def walk_values(self) -> Iterable[Value]:
        return ()


@dataclass(frozen=True)
class PredicateValue(Value):
    """Represents a type-level predicate constraint."""

    predicate: Predicate

    def __str__(self) -> str:
        return f"Predicate[{self.predicate}]"

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return PredicateValue(self.predicate.substitute_typevars(typevars))

    def get_type_value(self) -> Value:
        return KnownValue(object)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.predicate.walk_values()


SimpleType: typing_extensions.TypeAlias = (
    AnyValue
    | KnownValue
    | SyntheticClassObjectValue
    | SyntheticModuleValue
    | UnboundMethodValue
    | TypedValue
    | SubclassValue
    | TypeFormValue
    | PredicateValue
)

BasicType: typing_extensions.TypeAlias = (
    SimpleType | MultiValuedValue | IntersectionValue
)

# Subclasses of Value that represent real types in the type system.
# There are a few other subclasses of Value that represent temporary
# objects in some contexts; this alias exists to make it easier to refer
# to those Values that are actually part of the type system.
GradualType: typing_extensions.TypeAlias = (
    BasicType
    | OverlappingValue
    | TypeAliasValue
    | NewTypeValue
    | TypeVarValue
    | TypeVarTupleBindingValue
    | TypeVarTupleValue
    | ParamSpecArgsValue
    | ParamSpecKwargsValue
    | AnnotatedValue
    | PartialValue
    | PartialCallValue
    | SuperValue
)

GRADUAL_TYPE = GradualType.__args__
BASIC_TYPE = BasicType.__args__


class NotAGradualType(Exception):
    """Raised when a value is not a gradual type."""


def gradualize(value: Value) -> GradualType:
    if not isinstance(value, GRADUAL_TYPE):
        raise NotAGradualType(f"Encountered non-type {value!r}")
    return value


def replace_fallback(val: Value) -> BasicType:
    while True:
        fallback = val.get_fallback_value()
        if fallback is None:
            break
        val = fallback
    if not isinstance(val, BASIC_TYPE):
        raise NotAGradualType(f"Encountered non-basic type {val!r}")
    return val


def is_union(val: Value) -> bool:
    return isinstance(val, MultiValuedValue) or (
        isinstance(val, AnnotatedValue) and isinstance(val.value, MultiValuedValue)
    )


def flatten_values(val: Value, *, unwrap_annotated: bool = False) -> Iterable[Value]:
    """Flatten a :class:`MultiValuedValue` into its constituent values.

    We don't need to do this recursively because the
    :class:`MultiValuedValue` constructor applies this to its arguments.

    if `unwrap_annotated` is true, produces the underlying values for
    :class:`AnnotatedValue` objects.

    """
    if isinstance(val, MultiValuedValue):
        yield from val.vals
    elif isinstance(val, AnnotatedValue) and isinstance(val.value, MultiValuedValue):
        if unwrap_annotated:
            yield from val.value.vals
        else:
            subvals = [
                annotate_value(subval, val.metadata) for subval in val.value.vals
            ]
            yield from subvals
    elif unwrap_annotated and isinstance(val, AnnotatedValue):
        yield val.value
    else:
        yield val


def get_tv_map(
    left: Value, right: Value, ctx: CanAssignContext
) -> TypeVarMap | CanAssignError:
    from .relations import Relation
    from .relations import get_tv_map as relation_get_tv_map

    return relation_get_tv_map(left, right, Relation.ASSIGNABLE, ctx)


def unify_bounds_maps(bounds_maps: Sequence[BoundsMap]) -> BoundsMap:
    result = {}
    for bounds_map in bounds_maps:
        for tv, bounds in bounds_map.items():
            result.setdefault(tv, []).extend(bounds)
    return result


def intersect_bounds_maps(bounds_maps: Sequence[BoundsMap]) -> BoundsMap:
    intermediate: dict[TypeVarLike, set[tuple[Bound, ...]]] = {}
    for bounds_map in bounds_maps:
        for tv, bounds in bounds_map.items():
            intermediate.setdefault(tv, set()).add(tuple(bounds))
    return {
        tv: (
            [OrBound(tuple(bound_lists))]
            if len(bound_lists) > 1
            else next(iter(bound_lists))
        )
        for tv, bound_lists in intermediate.items()
        if all(tv in bounds_map for bounds_map in bounds_maps)
    }


def annotate_value(origin: Value, metadata: Sequence[Extension]) -> Value:
    if not metadata:
        return origin
    if isinstance(origin, AnnotatedValue):
        # Flatten it
        metadata = (*origin.metadata, *metadata)
        origin = origin.value
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = {}
    unhashable_vals = []
    for item in metadata:
        try:
            # Don't readd it to preserve original ordering.
            if item not in hashable_vals:
                hashable_vals[item] = None
        except Exception:
            unhashable_vals.append(item)
    metadata = (*hashable_vals, *unhashable_vals)
    return AnnotatedValue(origin, metadata)


ExtensionT = TypeVar("ExtensionT", bound=Extension)


def unannotate_value(
    origin: Value, extension: type[ExtensionT]
) -> tuple[Value, Sequence[ExtensionT]]:
    if not isinstance(origin, AnnotatedValue):
        return origin, []
    matches = [
        metadata for metadata in origin.metadata if isinstance(metadata, extension)
    ]
    if matches:
        remaining = [
            metadata
            for metadata in origin.metadata
            if not isinstance(metadata, extension)
        ]
        return annotate_value(origin.value, remaining), matches
    return origin, []


def unannotate(value: Value) -> Value:
    if isinstance(value, AnnotatedValue):
        return value.value
    return value


def unite_and_simplify(*values: Value, limit: int) -> Value:
    united = unite_values(*values)
    if not isinstance(united, MultiValuedValue) or len(united.vals) < limit:
        return united
    simplified = [val.simplify() for val in united.vals]
    return unite_values(*simplified)


def _is_unreachable(value: Value) -> bool:
    if isinstance(value, AnnotatedValue):
        return _is_unreachable(value.value)
    return isinstance(value, AnyValue) and value.source is AnySource.unreachable


def unite_values(*values: Value) -> Value:
    """Unite multiple values into a single :class:`Value`.

    This collapses equal values and returns a :class:`MultiValuedValue`
    if multiple remain.

    """
    if not values:
        return NO_RETURN_VALUE
    # Make sure order is consistent; conceptually this is a set but
    # sets have unpredictable iteration order.
    hashable_vals = {}
    unhashable_vals = []
    for value in values:
        assert isinstance(value, Value), repr(value)
        if isinstance(value, MultiValuedValue):
            subvals = value.vals
        elif isinstance(value, AnnotatedValue) and isinstance(
            value.value, MultiValuedValue
        ):
            subvals = [
                annotate_value(subval, value.metadata) for subval in value.value.vals
            ]
        else:
            subvals = [value]
        for subval in subvals:
            try:
                # Don't readd it to preserve original ordering.
                if subval not in hashable_vals:
                    hashable_vals[subval] = None
            except Exception:
                unhashable_vals.append(subval)
    existing = list(hashable_vals) + unhashable_vals
    reachabilities = [_is_unreachable(val) for val in existing]
    num_unreachable = sum(reachabilities)
    num = len(existing) - num_unreachable
    if num == 0:
        if num_unreachable:
            return AnyValue(AnySource.unreachable)
        return NO_RETURN_VALUE
    if num_unreachable:
        existing = [val for i, val in enumerate(existing) if not reachabilities[i]]
    if num == 1:
        return existing[0]
    else:
        return MultiValuedValue(existing)


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(TypeVarParam(T))])
AsyncIterableValue = GenericValue(
    collections.abc.AsyncIterable, [TypeVarValue(TypeVarParam(T))]
)


class GetItemProto(Protocol[T_co]):
    def __getitem__(self, __i: int) -> T_co:
        raise NotImplementedError


GetItemProtoValue = GenericValue(GetItemProto, [TypeVarValue(TypeVarParam(T_co))])
TypingGenericAlias = type(list[int])


def len_of_value(val: Value) -> Value:
    if isinstance(val, AnnotatedValue):
        return len_of_value(val.value)
    if (
        isinstance(val, SequenceValue)
        and isinstance(val.typ, type)
        and not issubclass(val.typ, KNOWN_MUTABLE_TYPES)
    ):
        members = val.get_member_sequence()
        if members is not None:
            return KnownValue(len(members))
    if isinstance(val, KnownValue):
        try:
            if not isinstance(val.val, KNOWN_MUTABLE_TYPES):
                return KnownValue(len(val.val))
        except Exception:
            return TypedValue(int)
    return TypedValue(int)


def concrete_values_from_iterable(
    value: Value, ctx: CanAssignContext
) -> CanAssignError | Value | Sequence[Value]:
    """Return the exact values that can be extracted from an iterable.

    Three possible return types:

    - :class:`CanAssignError` if the argument is not iterable
    - A sequence of :class:`Value` if we know the exact types in the iterable
    - A single :class:`Value` if we just know that the iterable contains this
      value, but not the precise number of them.

    Examples:

    - ``int`` -> ``CanAssignError``
    - ``tuple[int, str]`` -> ``(int, str)``
    - ``tuple[int, ...]`` -> ``int``

    """
    value = replace_known_sequence_value(value)
    is_nonempty = False
    if value is NO_RETURN_VALUE:
        return NO_RETURN_VALUE
    elif isinstance(value, MultiValuedValue):
        subvals = [concrete_values_from_iterable(val, ctx) for val in value.vals]
        errors = [subval for subval in subvals if isinstance(subval, CanAssignError)]
        if errors:
            return CanAssignError(
                "At least one member of Union is not iterable", errors
            )
        value_subvals = [subval for subval in subvals if isinstance(subval, Value)]
        seq_subvals = [
            subval
            for subval in subvals
            if not isinstance(subval, (Value, CanAssignError))
        ]
        if not value_subvals and len(set(map(len, seq_subvals))) == 1:
            return [unite_values(*vals) for vals in zip(*seq_subvals)]
        return unite_values(*value_subvals, *chain.from_iterable(seq_subvals))
    if isinstance(value, SequenceValue):
        members = value.get_member_sequence()
        if members is None:
            return value.args[0]
        return members
    elif isinstance(value, TypedDictValue):
        if value.extra_keys is NO_RETURN_VALUE and all(
            entry.required for entry in value.items.values()
        ):
            return [KnownValue(key) for key in value.items]
        possibilities = [KnownValue(key) for key in value.items]
        if value.extra_keys is not NO_RETURN_VALUE:
            possibilities.append(TypedValue(str))
        return MultiValuedValue(possibilities)
    elif isinstance(value, DictIncompleteValue):
        if all(pair.is_required and not pair.is_many for pair in value.kv_pairs):
            return [pair.key for pair in value.kv_pairs]
    elif isinstance(value, KnownValue):
        if isinstance(value.val, (str, bytes, range)):
            if len(value.val) < ITERATION_LIMIT:
                return [KnownValue(c) for c in value.val]
            is_nonempty = True
        if (
            sys.version_info >= (3, 11)
            and isinstance(value.val, (types.GenericAlias, TypingGenericAlias))
        ) or is_instance_of_typing_name(value.val, "TypeVarTuple"):
            return [KnownValue(c) for c in value.val]
        if isinstance(value.val, type) and safe_issubclass(value.val, enum.Enum):
            return TypedValue(value.val)
    iterable_type = is_iterable(value, ctx)
    if isinstance(iterable_type, Value):
        val = iterable_type
    else:
        getitem_tv_map = get_tv_map(GetItemProtoValue, value, ctx)
        if not isinstance(getitem_tv_map, CanAssignError):
            val = getitem_tv_map.get_typevar(
                TypeVarParam(T_co), AnyValue(AnySource.generic_argument)
            )
        # Hack to support iteration over StrEnum. A better solution would have to
        # handle descriptors better in attribute assignment and Protocol compatibility.
        elif (
            isinstance(value, SubclassValue)
            and isinstance(value.typ, TypedValue)
            and isinstance(value.typ.typ, type)
            and safe_issubclass(value.typ.typ, enum.Enum)
        ):
            return value.typ
        else:
            # We return the error from the __iter__ check because the __getitem__
            # check is more arcane.
            return iterable_type
    if is_nonempty:
        return annotate_value(val, [AlwaysPresentExtension()])
    return val


K = TypeVar("K")
V = TypeVar("V")
V_co = TypeVar("V_co", covariant=True)

EMPTY_DICTS = (KnownValue({}), DictIncompleteValue(dict, []))


# This is all the runtime requires in places like {**k}
class CustomMapping(Protocol[K, V_co]):
    def keys(self) -> Iterable[K]:
        raise NotImplementedError

    def __getitem__(self, __key: K) -> V_co:
        raise NotImplementedError


NominalMappingValue = GenericValue(
    collections.abc.Mapping,
    [TypeVarValue(TypeVarParam(K)), TypeVarValue(TypeVarParam(V))],
)
ProtocolMappingValue = GenericValue(
    CustomMapping, [TypeVarValue(TypeVarParam(K)), TypeVarValue(TypeVarParam(V_co))]
)


def kv_pairs_from_mapping(
    value_val: Value, ctx: CanAssignContext
) -> Sequence[KVPair] | CanAssignError:
    """Return the :class:`KVPair` objects that can be extracted from this value,
    or a :class:`CanAssignError` on error."""
    value_val = replace_known_sequence_value(value_val)
    # Special case: if we have a Union including an empty dict, just get the
    # pairs from the rest of the union and make them all non-required.
    if isinstance(value_val, MultiValuedValue):
        subvals = [replace_known_sequence_value(subval) for subval in value_val.vals]
        if any(subval in EMPTY_DICTS for subval in subvals):
            other_val = unite_values(
                *[subval for subval in subvals if subval not in EMPTY_DICTS]
            )
            pairs = kv_pairs_from_mapping(other_val, ctx)
            if isinstance(pairs, CanAssignError):
                return pairs
            return [
                KVPair(pair.key, pair.value, pair.is_many, is_required=False)
                for pair in pairs
            ]
    if isinstance(value_val, DictIncompleteValue):
        return value_val.kv_pairs
    elif isinstance(value_val, TypedDictValue):
        pairs = []
        for key, entry in value_val.items.items():
            # Optional Never keys are uninhabitable and cannot be present.
            if not entry.required and entry.typ is NO_RETURN_VALUE:
                continue
            pairs.append(KVPair(KnownValue(key), entry.typ, is_required=entry.required))
        if (
            value_val.extra_keys is not None
            and value_val.extra_keys is not NO_RETURN_VALUE
        ):
            pairs.append(
                KVPair(
                    TypedValue(str),
                    value_val.extra_keys,
                    is_many=True,
                    is_required=False,
                )
            )
        return pairs
    else:
        # Ideally we should only need to check ProtocolMappingValue, but if
        # we do that we can't infer the right types for dict, so try the
        # nominal Mapping first.
        can_assign = get_tv_map(NominalMappingValue, value_val, ctx)
        if isinstance(can_assign, CanAssignError):
            can_assign = get_tv_map(ProtocolMappingValue, value_val, ctx)
            if isinstance(can_assign, CanAssignError):
                return can_assign
        key_type = can_assign.get_typevar(
            TypeVarParam(K), AnyValue(AnySource.generic_argument)
        )
        value_type = can_assign.get_typevar(
            TypeVarParam(V),
            can_assign.get_typevar(
                TypeVarParam(V_co), AnyValue(AnySource.generic_argument)
            ),
        )
        return [KVPair(key_type, value_type, is_many=True)]


def unpack_values(
    value: Value,
    ctx: CanAssignContext,
    target_length: int,
    post_starred_length: int | None = None,
) -> Sequence[Value] | CanAssignError:
    """Implement iterable unpacking.

    If `post_starred_length` is None, return a list of `target_length`
    values, or :class:`CanAssignError` if value is not an iterable of
    the expected length. If `post_starred_length` is not None,
    return a list of `target_length + 1 + post_starred_length` values. This implements
    unpacking like ``a, b, *c, d = ...``.

    """
    if isinstance(value, TypeAliasValue):
        value = value.get_value()
    if isinstance(value, MultiValuedValue):
        subvals = [
            unpack_values(val, ctx, target_length, post_starred_length)
            for val in value.vals
        ]
        good_subvals = []
        for subval in subvals:
            if isinstance(subval, CanAssignError):
                return CanAssignError(f"Cannot unpack {value}", [subval])
            good_subvals.append(subval)
        if not good_subvals:
            return _create_unpacked_list(
                (
                    AnyValue(AnySource.error)
                    if subvals
                    else AnyValue(AnySource.unreachable)
                ),
                target_length,
                post_starred_length,
            )
        return [unite_values(*vals) for vals in zip(*good_subvals)]
    value = replace_known_sequence_value(value)
    if (tuple_members := tuple_members_from_value(value, ctx)) is not None:
        value = SequenceValue(tuple, tuple_members)

    # We treat the different sequence types differently here.
    # - Tuples are  immutable so we can always unpack and show
    #   an error if the length doesn't match.
    # - Sets have randomized order so unpacking into specific values
    #   doesn't make sense. We just fallback to the behavior for
    #   general iterables.
    # - Dicts do have deterministic order but unpacking them doesn't
    #   seem like a common use case. They're also mutable, so if we
    #   did decide to unpack, we'd have to do something similar to
    #   what we do for lists.
    # - Lists can be sensibly unpacked but they are also mutable. Therefore,
    #   we try first to unpack into specific values, and if that doesn't
    #   work due to a length mismatch we fall back to the generic
    #   iterable approach. We experimented both with treating lists
    #   like tuples and with always falling back, and both approaches
    #   led to false positives.
    if isinstance(value, SequenceValue):
        if value.typ is tuple:
            return _unpack_sequence_value(value, target_length, post_starred_length)
        elif value.typ is list:
            vals = _unpack_sequence_value(value, target_length, post_starred_length)
            if not isinstance(vals, CanAssignError):
                return vals

    iterable_type = is_iterable(value, ctx)
    if isinstance(iterable_type, CanAssignError):
        return iterable_type
    return _create_unpacked_list(iterable_type, target_length, post_starred_length)


def is_iterable(value: Value, ctx: CanAssignContext) -> CanAssignError | Value:
    """Check whether a value is iterable."""
    tv_map = get_tv_map(IterableValue, value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get_typevar(TypeVarParam(T), AnyValue(AnySource.generic_argument))


def get_namedtuple_field_annotation(namedtuple_type: type, field_name: str) -> object:
    for base in namedtuple_type.__mro__:
        annotations = safe_getattr(base, "__annotations__", None)
        if isinstance(annotations, Mapping) and field_name in annotations:
            return annotations[field_name]
    return typing.Any


def tuple_members_from_value(
    value: Value, ctx: CanAssignContext
) -> SequenceMembers | None:
    if isinstance(value, TypeVarTupleBindingValue):
        return value.binding
    value = replace_known_sequence_value(value)
    if isinstance(value, SequenceValue) and value.typ is tuple:
        return value.members
    elif isinstance(value, TypedValue):
        tobj = value.get_type_object(ctx)
        mro = tobj.get_mro()
        tuple_entries = [
            entry
            for entry in mro
            if entry.tobj is not None
            and entry.tobj.typ is tuple
            and isinstance(entry.value, SequenceValue)
        ]
        maybe_iter = tobj.get_declared_symbol_with_owner("__iter__", ctx)
        if maybe_iter is None:
            return None
        owner_tobj, _ = maybe_iter
        if owner_tobj.typ is not tuple:
            return None  # overrides __iter__
        if tuple_entries:
            seq = tuple_entries[0].value
            assert isinstance(seq, SequenceValue)
            if isinstance(value, GenericValue):
                substitutions = tobj.get_substitutions(value.args)
                seq = seq.substitute_typevars(substitutions)
            return seq.members
    return None


def is_async_iterable(value: Value, ctx: CanAssignContext) -> CanAssignError | Value:
    """Check whether a value is an async iterable."""
    tv_map = get_tv_map(AsyncIterableValue, value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get_typevar(TypeVarParam(T), AnyValue(AnySource.generic_argument))


def _create_unpacked_list(
    iterable_type: Value, target_length: int, post_starred_length: int | None
) -> list[Value]:
    if post_starred_length is not None:
        return [
            *([iterable_type] * target_length),
            GenericValue(list, [iterable_type]),
            *([iterable_type] * post_starred_length),
        ]
    else:
        return [iterable_type] * target_length


def _unpack_sequence_value(
    value: SequenceValue, target_length: int, post_starred_length: int | None
) -> Sequence[Value] | CanAssignError:
    head = []
    tail = []
    while len(head) < target_length:
        if len(head) >= len(value.members):
            return CanAssignError(
                f"{value} must have at least {target_length} elements"
            )
        is_many, val = value.members[len(head)]
        if is_many:
            break
        head.append(val)
    remaining_target_length = target_length - len(head)
    if post_starred_length is None:
        if remaining_target_length == 0:
            if all(is_many for is_many, _ in value.members[target_length:]):
                return head
            return CanAssignError(f"{value} must have exactly {target_length} elements")

        tail = []
        while len(tail) < remaining_target_length:
            if len(tail) + len(head) >= len(value.members):
                return CanAssignError(
                    f"{value} must have at least {target_length} elements"
                )
            is_many, val = value.members[-len(tail) - 1]
            if is_many:
                break
            tail.append(val)

        if tail:
            remaining_members = value.members[len(head) : -len(tail)]
        else:
            remaining_members = value.members[len(head) :]
        if not remaining_members:
            return CanAssignError(f"{value} must have exactly {target_length} elements")
        middle_length = remaining_target_length - len(tail)
        fallback_value = unite_values(*[val for _, val in remaining_members])
        return [*head, *[fallback_value for _ in range(middle_length)], *reversed(tail)]
    else:
        while len(tail) < post_starred_length:
            if len(tail) >= len(value.members) - len(head):
                return CanAssignError(
                    f"{value} must have at least"
                    f" {target_length + post_starred_length} elements"
                )
            is_many, val = value.members[-len(tail) - 1]
            if is_many:
                break
            tail.append(val)
        remaining_post_starred_length = post_starred_length - len(tail)

        if tail:
            remaining_members = value.members[len(head) : -len(tail)]
        else:
            remaining_members = value.members[len(head) :]
        if remaining_target_length != 0 or remaining_post_starred_length != 0:
            if not remaining_members:
                return CanAssignError(
                    f"{value} must have at least"
                    f" {target_length + post_starred_length} elements"
                )
            else:
                fallback_value = unite_values(*[val for _, val in remaining_members])
                return [
                    *head,
                    *[fallback_value for _ in range(remaining_target_length)],
                    GenericValue(list, [fallback_value]),
                    *[fallback_value for _ in range(remaining_post_starred_length)],
                    *reversed(tail),
                ]
        else:
            if len(remaining_members) == 1 and remaining_members[0][0]:
                middle = GenericValue(list, [remaining_members[0][1]])
            else:
                middle = SequenceValue(list, remaining_members)
            return [*head, middle, *reversed(tail)]


def replace_known_sequence_value(
    value: Value, ctx: CanAssignContext | None = None
) -> BasicType:
    """Simplify a Value in a way that is easier to handle for most typechecking use cases.

    Does the following:

    - Replace AnnotatedValue with its inner type
    - Replace TypeVarValue with its fallback type
    - Replace KnownValues representing list, tuples, sets, or dicts with
      SequenceValue or DictIncompleteValue.

    """
    value = replace_fallback(value)
    if isinstance(value, KnownValue):
        return typify_literal(value, ctx)
    return value


def typify_literal(
    value: KnownValue, ctx: CanAssignContext | None = None
) -> KnownValue | TypedValue:
    if isinstance(value.val, tuple):
        if type(value.val) is tuple:
            return SequenceValue(tuple, [(False, KnownValue(elt)) for elt in value.val])
        if ctx is not None:
            return TypedValue(type(value.val))
    if isinstance(value.val, (list, set)):
        return SequenceValue(
            type(value.val), [(False, KnownValue(elt)) for elt in value.val]
        )
    elif isinstance(value.val, dict):
        return DictIncompleteValue(
            type(value.val),
            [KVPair(KnownValue(k), KnownValue(v)) for k, v in value.val.items()],
        )
    else:
        return value


def stringify_object(obj: Any) -> str:
    # Stringify arbitrary Python objects such as methods and types.
    if isinstance(obj, str):
        return obj
    try:
        objclass = getattr(obj, "__objclass__", None)
        if objclass is not None:
            return f"{stringify_object(objclass)}.{obj.__name__}"
        if obj.__module__ == BUILTIN_MODULE:
            return obj.__name__
        elif hasattr(obj, "__qualname__"):
            return f"{obj.__module__}.{obj.__qualname__}"
        else:
            return f"{obj.__module__}.{obj.__name__}"
    except Exception:
        return repr(obj)


def _deliteral(value: Value) -> Value:
    value = unannotate(value)
    if isinstance(value, KnownValue):
        value = TypedValue(type(value.val))
    if isinstance(value, SequenceValue):
        value = TypedValue(value.typ)
    return value


def is_overlapping(left: Value, right: Value, ctx: CanAssignContext) -> bool:
    # Fairly permissive checks for now; possibly this can be tightened up later.
    left = _deliteral(left)
    right = _deliteral(right)
    # TODO: we should always do this but that leads to some silly behavior
    # (it starts thinking about dictionary subclasses that are also sequences)
    if isinstance(left, PredicateValue) or isinstance(right, PredicateValue):
        inters = pycroscope.relations.intersect_values(left, right, ctx)
        return inters is not NO_RETURN_VALUE
    if isinstance(left, (MultiValuedValue, IntersectionValue)) and left.vals:
        # Swap the operands so we decompose and de-Literal the other union too
        return any(is_overlapping(right, val, ctx) for val in left.vals)
    return left.is_assignable(right, ctx) or right.is_assignable(left, ctx)


def make_coro_type(return_type: Value) -> GenericValue:
    return GenericValue(
        collections.abc.Coroutine,
        [AnyValue(AnySource.inference), AnyValue(AnySource.inference), return_type],
    )


class Qualifier(enum.Enum):
    ClassVar = "ClassVar"
    Final = "Final"
    Unpack = "Unpack"
    ReadOnly = "ReadOnly"
    Required = "Required"
    NotRequired = "NotRequired"
    InitVar = "InitVar"
    TypeAlias = "TypeAlias"


class FunctionDecorator(enum.Enum):
    classmethod = enum.auto()
    staticmethod = enum.auto()
    decorated_coroutine = enum.auto()  # @asyncio.coroutine
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


@dataclass(frozen=True, kw_only=True)
class ClassSymbol:
    # Declared annotation for the member, if any. This is present for annotated
    # attributes and absent for unannotated synthetic members and methods.
    annotation: Value | None = None
    # Stored value information for the member. For annotated attributes this is
    # the assigned/default value when known; for methods it is the callable
    # value; for unannotated synthetic members it is the best available value.
    initializer: Value | None = None
    qualifiers: frozenset[Qualifier] = frozenset()
    function_decorators: frozenset[FunctionDecorator] = frozenset()
    deprecation_message: str | None = None
    # TODO: How do we determine this? Does it add information over initializer/annotation?
    is_instance_only: bool = False
    is_method: bool = False
    # TODO: not sure why this exists or why we need it
    returns_self_on_class_access: bool = False
    property_info: PropertyInfo | None = None
    dataclass_field: DataclassFieldInfo | None = None

    def __post_init__(self) -> None:
        if self.returns_self_on_class_access:
            assert self.is_method, self
        if self.property_info is not None:
            assert not self.is_method, self
            assert not self.is_classmethod, self
            assert not self.is_staticmethod, self
            assert not self.returns_self_on_class_access, self
            assert self.initializer is None or _is_property_initializer(
                self.initializer
            ), self
        if self.is_method:
            assert self.initializer is not None, self
            assert self.property_info is None, self
            assert self.annotation is None, self

    @property
    def is_classmethod(self) -> bool:
        return FunctionDecorator.classmethod in self.function_decorators

    @property
    def is_staticmethod(self) -> bool:
        return FunctionDecorator.staticmethod in self.function_decorators

    @property
    def is_classvar(self) -> bool:
        return Qualifier.ClassVar in self.qualifiers

    @property
    def is_readonly(self) -> bool:
        return Qualifier.ReadOnly in self.qualifiers

    @property
    def is_initvar(self) -> bool:
        return Qualifier.InitVar in self.qualifiers

    @property
    def is_final(self) -> bool:
        return (
            Qualifier.Final in self.qualifiers
            or FunctionDecorator.final in self.function_decorators
        )

    @property
    def is_property(self) -> bool:
        return self.property_info is not None

    def substitute_typevars(self, substitutions: TypeVarMap) -> "ClassSymbol":
        return ClassSymbol(
            annotation=(
                self.annotation.substitute_typevars(substitutions)
                if self.annotation is not None
                else None
            ),
            initializer=(
                self.initializer.substitute_typevars(substitutions)
                if self.initializer is not None
                else None
            ),
            qualifiers=self.qualifiers,
            function_decorators=self.function_decorators,
            deprecation_message=self.deprecation_message,
            is_instance_only=self.is_instance_only,
            is_method=self.is_method,
            returns_self_on_class_access=self.returns_self_on_class_access,
            property_info=(
                self.property_info.substitute_typevars(substitutions)
                if self.property_info is not None
                else None
            ),
            dataclass_field=(
                self.dataclass_field.substitute_typevars(substitutions)
                if self.dataclass_field is not None
                else None
            ),
        )

    # TODO: I don't think these two methods should exist, they are confusing
    def get_declared_type(self) -> Value | None:
        if self.annotation is not None:
            return self.annotation
        return self.initializer

    def get_effective_type(self) -> Value:
        declared_type = self.get_declared_type()
        if declared_type is not None:
            return declared_type
        return AnyValue(AnySource.inference)


def _is_property_initializer(value: Value) -> bool:
    value = replace_fallback(value)
    return (
        isinstance(value, KnownValue)
        and isinstance(value.val, (property, types.GetSetDescriptorType))
        or isinstance(value, TypedValue)
        and (value.typ is property or value.typ is types.GetSetDescriptorType)
    )


@dataclass(frozen=True)
class AnnotationExpr:
    ctx: "pycroscope.annotations.Context"
    _value: Value | None
    qualifiers: Sequence[tuple[Qualifier, ast.AST | None]] = field(default_factory=list)
    metadata: Sequence[Extension] = field(default_factory=list)

    def add_qualifier(
        self, qualifier: Qualifier, node: ast.AST | None
    ) -> "AnnotationExpr":
        return AnnotationExpr(
            self.ctx,
            self._value,
            qualifiers=[*self.qualifiers, (qualifier, node)],
            metadata=self.metadata,
        )

    def add_metadata(
        self, metadata: tuple[Sequence[Value], Sequence[Extension]]
    ) -> "AnnotationExpr":
        new_intersects, new_extensions = metadata
        if new_intersects:
            if self._value:
                new_value = IntersectionValue((self._value, *new_intersects))
            else:
                new_value = IntersectionValue(tuple(new_intersects))
        else:
            new_value = self._value
        return AnnotationExpr(
            self.ctx,
            new_value,
            qualifiers=self.qualifiers,
            metadata=[*self.metadata, *new_extensions],
        )

    def to_value(
        self,
        *,
        allow_qualifiers: bool = False,
        allow_empty: bool = False,
        qualifier_error_code: Error = ErrorCode.invalid_qualifier,
    ) -> Value:
        if self._value is None:
            if allow_empty:
                return AnyValue(AnySource.incomplete_annotation)
            else:
                innermost, node = self.qualifiers[-1]
                self.ctx.show_error(
                    f"Invalid bare {innermost.name} annotation",
                    node=node,
                    error_code=qualifier_error_code,
                )
                return AnyValue(AnySource.error)
        if not allow_qualifiers:
            for qualifier, node in self.qualifiers:
                self.ctx.show_error(
                    f"Unexpected {qualifier.name} annotation",
                    node=node,
                    error_code=qualifier_error_code,
                )
        if self.metadata:
            return annotate_value(self._value, self.metadata)
        return self._value

    def unqualify(
        self,
        allowed_qualifiers: Container[Qualifier] = frozenset(),
        *,
        mutually_exclusive_qualifiers: Collection[Collection[Qualifier]] = (),
        qualifier_error_code: Error = ErrorCode.invalid_qualifier,
    ) -> tuple[Value, set[Qualifier]]:
        value, qualifiers = self.maybe_unqualify(
            allowed_qualifiers,
            mutually_exclusive_qualifiers=mutually_exclusive_qualifiers,
            qualifier_error_code=qualifier_error_code,
        )
        if value is None:
            innermost, node = self.qualifiers[-1]
            self.ctx.show_error(
                f"Invalid bare {innermost.name} annotation",
                node=node,
                error_code=qualifier_error_code,
            )
            return AnyValue(AnySource.error), set()
        return value, qualifiers

    def maybe_unqualify(
        self,
        allowed_qualifiers: Container[Qualifier] = frozenset(),
        *,
        mutually_exclusive_qualifiers: Collection[Collection[Qualifier]] = (),
        qualifier_error_code: Error = ErrorCode.invalid_qualifier,
    ) -> tuple[Value | None, set[Qualifier]]:
        qualifiers = set()
        qualifier_counts: dict[Qualifier, int] = {}
        qualifier_nodes: dict[Qualifier, ast.AST | None] = {}
        for qualifier, node in self.qualifiers:
            qualifier_counts[qualifier] = qualifier_counts.get(qualifier, 0) + 1
            qualifier_nodes.setdefault(qualifier, node)
            if qualifier in allowed_qualifiers:
                qualifiers.add(qualifier)
            else:
                self.ctx.show_error(
                    f"Unexpected {qualifier.name} annotation",
                    node=node,
                    error_code=qualifier_error_code,
                )
        for qualifier, count in qualifier_counts.items():
            if count > 1:
                self.ctx.show_error(
                    f"{qualifier.name}[] cannot be nested",
                    node=qualifier_nodes.get(qualifier),
                    error_code=qualifier_error_code,
                )
        for qualifier_group in mutually_exclusive_qualifiers:
            present = [
                qualifier
                for qualifier in qualifier_group
                if qualifier_counts.get(qualifier, 0) > 0
            ]
            if len(present) > 1:
                if len(present) == 2:
                    message = (
                        f"{present[0].name}[] and {present[1].name}[] cannot be nested"
                    )
                else:
                    members = ", ".join(f"{qualifier.name}[]" for qualifier in present)
                    message = f"{members} cannot be nested together"
                self.ctx.show_error(
                    message,
                    node=qualifier_nodes[present[0]],
                    error_code=qualifier_error_code,
                )
        if self._value is None:
            return None, qualifiers
        if self.metadata:
            value = annotate_value(self._value, self.metadata)
        else:
            value = self._value
        return value, qualifiers
