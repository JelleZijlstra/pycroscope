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
import collections.abc
import contextlib
import enum
import sys
import textwrap
from collections import deque
from collections.abc import Container, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import InitVar, dataclass, field
from itertools import chain
from types import FunctionType, ModuleType
from typing import Any, Callable, Optional, TypeVar, Union

import typing_extensions
from typing_extensions import ParamSpec, Protocol, assert_never

import pycroscope
from pycroscope.error_code import Error
from pycroscope.extensions import CustomCheck, ExternalType
from pycroscope.safe import all_of_type, safe_equals, safe_isinstance, safe_issubclass

T = TypeVar("T")
# __builtin__ in Python 2 and builtins in Python 3
BUILTIN_MODULE = str.__module__
KNOWN_MUTABLE_TYPES = (list, set, dict, deque)
ITERATION_LIMIT = 1000

if sys.version_info >= (3, 11):
    TypeVarLike = Union[
        ExternalType["typing.TypeVar"],
        ExternalType["typing_extensions.TypeVar"],
        ExternalType["typing.ParamSpec"],
        ExternalType["typing_extensions.ParamSpec"],
        ExternalType["typing.TypeVarTuple"],
        ExternalType["typing_extensions.TypeVarTuple"],
    ]
elif sys.version_info >= (3, 10):
    TypeVarLike = Union[
        ExternalType["typing.TypeVar"],
        ExternalType["typing_extensions.TypeVar"],
        ExternalType["typing.ParamSpec"],
        ExternalType["typing_extensions.ParamSpec"],
        ExternalType["typing_extensions.TypeVarTuple"],
    ]
else:
    TypeVarLike = Union[
        ExternalType["typing.TypeVar"],
        ExternalType["typing_extensions.TypeVar"],
        ExternalType["typing_extensions.ParamSpec"],
        ExternalType["typing_extensions.TypeVarTuple"],
    ]

TypeVarMap = Mapping[TypeVarLike, ExternalType["pycroscope.value.Value"]]
BoundsMap = Mapping[TypeVarLike, Sequence[ExternalType["pycroscope.value.Bound"]]]
GenericBases = Mapping[Union[type, str], TypeVarMap]


class OverlapMode(enum.Enum):
    IS = 1
    MATCH = 2
    EQ = 3


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

    def get_type(self) -> Optional[type]:
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

    def decompose(self) -> Optional[Iterable["Value"]]:
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

    def make_type_object(
        self, typ: Union[type, super, str]
    ) -> "pycroscope.type_object.TypeObject":
        """Return a :class:`pycroscope.type_object.TypeObject` for this concrete type."""
        raise NotImplementedError

    def get_generic_bases(
        self, typ: Union[type, str], generic_args: Sequence["Value"] = ()
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

    def record_any_used(self) -> None:
        """Record that Any was used to secure a match."""

    def set_exclude_any(self) -> AbstractContextManager[None]:
        """Within this context, `Any` is compatible only with itself."""
        return contextlib.nullcontext()

    def should_exclude_any(self) -> bool:
        """Whether Any should be compatible only with itself."""
        return False

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
    error_code: Optional[Error] = None

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

    def get_error_code(self) -> Optional[Error]:
        errors = {child.get_error_code() for child in self.children}
        if self.error_code:
            errors.add(self.error_code)
        if len(errors) == 1:
            return next(iter(errors))
        return None

    def __str__(self) -> str:
        return self.display()


# Return value of CanAssign
CanAssign = Union[BoundsMap, CanAssignError]


def assert_is_value(obj: object, value: Value, *, skip_annotated: bool = False) -> None:
    """Used to test pycroscope's value inference.

    Takes two arguments: a Python object and a :class:`Value` object. At runtime
    this does nothing, but pycroscope throws an error if the object is not
    inferred to be the same as the :class:`Value`.

    Example usage::

        assert_is_value(1, KnownValue(1))  # passes
        assert_is_value(1, TypedValue(int))  # shows an error

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
    ) -> Optional[CanAssignError]:
        return None  # always overlaps


UNRESOLVED_VALUE = AnyValue(AnySource.default)
"""The default instance of :class:`AnyValue`.

In the future, this should be replaced with instances of
`AnyValue` with a specific source.

"""


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


@dataclass
class TypeAlias:
    evaluator: Callable[[], Value]
    """Callable that evaluates the value."""
    evaluate_type_params: Callable[[], Sequence[TypeVarLike]]
    """Callable that evaluates the type parameters."""
    evaluated_value: Optional[Value] = None
    """Value that the type alias evaluates to."""
    type_params: Optional[Sequence[TypeVarLike]] = None
    """Type parameters of the type alias."""

    def get_value(self) -> Value:
        if self.evaluated_value is None:
            self.evaluated_value = self.evaluator()
        return self.evaluated_value

    def get_type_params(self) -> Sequence[TypeVarLike]:
        if self.type_params is None:
            self.type_params = self.evaluate_type_params()
        return self.type_params

    def get_fallback_value(self) -> Value:
        return self.get_value()


@dataclass(frozen=True)
class TypeAliasValue(Value):
    """Value representing a type alias."""

    name: str
    """Name of the type alias."""
    module: str
    """Module where the type alias is defined."""
    alias: TypeAlias = field(compare=False, hash=False)
    type_arguments: Sequence[Value] = ()

    def get_value(self) -> Value:
        val = self.alias.get_value()
        if self.type_arguments:
            type_params = self.alias.get_type_params()
            if len(type_params) != len(self.type_arguments):
                # TODO this should be an error
                return AnyValue(AnySource.inference)
            typevars = {
                type_param: arg
                for type_param, arg in zip(type_params, self.type_arguments)
            }
            val = val.substitute_typevars(typevars)
        return val

    def get_fallback_value(self) -> Value:
        return self.get_value()

    def is_type(self, typ: type) -> bool:
        return self.get_value().is_type(typ)

    def get_type(self) -> Optional[type]:
        return self.get_value().get_type()

    def get_type_value(self) -> Value:
        return self.get_value().get_type_value()

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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
        return self.get_type_object().is_assignable_to_type(typ)

    def get_type(self) -> type:
        return type(self.val)

    def get_type_object(
        self, ctx: Optional[CanAssignContext] = None
    ) -> "pycroscope.type_object.TypeObject":
        if ctx is not None:
            return ctx.make_type_object(type(self.val))
        return pycroscope.type_object.TypeObject(type(self.val))

    def get_type_value(self) -> Value:
        return KnownValue(type(self.val))

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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


def get_fully_qualified_name(obj: Union[FunctionType, type]) -> str:
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
    secondary_attr_name: Optional[str] = None
    """Used when an attribute is accessed on an existing ``UnboundMethodValue``.

    This is mostly useful in conjunction with asynq, where we might use
    ``object.method.asynq``. In that case, we would infer an ``UnboundMethodValue``
    with `secondary_attr_name` set to ``"asynq"``.

    """
    typevars: Optional[TypeVarMap] = field(default=None, compare=False)
    """Extra TypeVars applied to this method."""

    def get_method(self) -> Optional[Any]:
        """Return the runtime callable for this ``UnboundMethodValue``, or
        None if it cannot be found."""
        root = replace_fallback(self.composite.value)
        if isinstance(root, KnownValue):
            typ = root.val
        else:
            typ = root.get_type()
        try:
            method = getattr(typ, self.attr_name)
            if self.secondary_attr_name is not None:
                method = getattr(method, self.secondary_attr_name)
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
    ) -> Optional[CanAssignError]:
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

    def substitute_typevars(self, typevars: TypeVarMap) -> "Value":
        return UnboundMethodValue(
            self.attr_name,
            self.composite.substitute_typevars(typevars),
            self.secondary_attr_name,
            typevars=(
                typevars if self.typevars is None else {**self.typevars, **typevars}
            ),
        )

    def __str__(self) -> str:
        return "<method {}{} on {}>".format(
            self.attr_name,
            f".{self.secondary_attr_name}" if self.secondary_attr_name else "",
            self.composite.value,
        )


@dataclass(unsafe_hash=True)
class TypedValue(Value):
    """Value for which we know the type. This is equivalent to simple type
    annotations: an annotation of ``int`` will yield ``TypedValue(int)`` during
    type inference.

    """

    typ: Union[type, str]
    """The underlying type, or a fully qualified reference to one."""
    literal_only: bool = False
    """True if this is LiteralString (PEP 675)."""
    _type_object: Optional["pycroscope.type_object.TypeObject"] = field(
        init=False, repr=False, hash=False, compare=False, default=None
    )

    def get_type_object(
        self, ctx: Optional[CanAssignContext] = None
    ) -> "pycroscope.type_object.TypeObject":
        if self._type_object is None:
            if ctx is None:
                # TODO: remoove this behavior and make ctx required
                return pycroscope.type_object.TypeObject(self.typ)
            self._type_object = ctx.make_type_object(self.typ)
        return self._type_object

    def can_assign_thrift_enum(self, other: Value, ctx: CanAssignContext) -> CanAssign:
        if isinstance(other, AnyValue) and not ctx.should_exclude_any():
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
    ) -> Optional[CanAssignError]:
        self_tobj = self.get_type_object(ctx)
        if self_tobj.is_thrift_enum:
            if isinstance(other, (KnownValue, TypedValue)):
                can_assign = self.can_assign_thrift_enum(other, ctx)
                if isinstance(can_assign, CanAssignError):
                    return can_assign
                return None
            else:
                return super().can_overlap(other, ctx, mode)
        elif isinstance(other, KnownValue):
            if mode is OverlapMode.IS:
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
        self, typ: Union[type, super, str], ctx: CanAssignContext
    ) -> Optional[list[Value]]:
        if isinstance(self, GenericValue):
            args = self.args
        else:
            args = ()
        if isinstance(self.typ, super):
            generic_bases = ctx.get_generic_bases(self.typ.__self_class__, args)
        else:
            generic_bases = ctx.get_generic_bases(self.typ, args)
        if typ in generic_bases:
            return list(generic_bases[typ].values())
        return None

    def get_generic_arg_for_type(
        self, typ: Union[type, super], ctx: CanAssignContext, index: int
    ) -> Value:
        args = self.get_generic_args_for_type(typ, ctx)
        if args and index < len(args):
            return args[index]
        return AnyValue(AnySource.generic_argument)

    def is_type(self, typ: type) -> bool:
        return self.get_type_object().is_assignable_to_type(typ)

    def get_type(self) -> Optional[type]:
        if isinstance(self.typ, str):
            return None
        return self.typ

    def get_type_value(self) -> Value:
        if isinstance(self.typ, str):
            return AnyValue(AnySource.inference)
        return KnownValue(self.typ)

    def decompose(self) -> Optional[Iterable[Value]]:
        if self.typ is bool:
            return [KnownValue(True), KnownValue(False)]
        type_object = self.get_type_object()
        if (
            isinstance(self.typ, type)
            and type_object.is_assignable_to_type(enum.Enum)
            and not type_object.is_assignable_to_type(enum.Flag)
        ):
            # Decompose enum into its members
            assert issubclass(self.typ, enum.Enum)
            return (KnownValue(member) for member in self.typ)
        else:
            return None

    def __str__(self) -> str:
        if self.literal_only:
            if self.typ is str:
                return "LiteralString"
            suffix = " (literal only)"
        else:
            suffix = ""
        if self._type_object is not None:
            return f"{self._type_object}{suffix}"
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
    ) -> Optional[CanAssignError]:
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

    def __init__(self, typ: Union[type, str], args: Iterable[Value]) -> None:
        super().__init__(typ)
        self.args = tuple(args)

    def __str__(self) -> str:
        if self.typ is tuple:
            args = [*self.args, "..."]
        else:
            args = self.args
        args_str = ", ".join(str(arg) for arg in args)
        return f"{stringify_object(self.typ)}[{args_str}]"

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return GenericValue(
            self.typ, [arg.substitute_typevars(typevars) for arg in self.args]
        )

    def simplify(self) -> Value:
        return GenericValue(self.typ, [arg.simplify() for arg in self.args])

    def decompose(self) -> Optional[Iterable[Value]]:
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

    members: tuple[tuple[bool, Value], ...]
    """The elements of the sequence."""

    def __init__(
        self, typ: Union[type, str], members: Sequence[tuple[bool, Value]]
    ) -> None:
        if members:
            args = (unite_values(*[typ for _, typ in members]),)
        elif typ is tuple:
            args = (NO_RETURN_VALUE,)
        else:
            # Using Never for mutable types leads to issues
            args = (AnyValue(AnySource.unreachable),)
        super().__init__(typ, args)
        self.members = tuple(members)

    def get_member_sequence(self) -> Optional[Sequence[Value]]:
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
        cls, typ: type, members: Sequence[tuple[bool, Value]]
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

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return SequenceValue(
            self.typ,
            [
                (is_many, member.substitute_typevars(typevars))
                for is_many, member in self.members
            ],
        )

    def __str__(self) -> str:
        members = ", ".join(
            (f"*tuple[{m}, ...]" if is_many else str(m)) for is_many, m in self.members
        )
        if self.typ is tuple:
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

    def decompose(self) -> Optional[Iterable[Value]]:
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
            # For simplicity, only decompose the first member
            first_decomposed = self.members[0][1].decompose()
            if first_decomposed is not None:
                return [
                    SequenceValue(self.typ, [(False, val), *self.members[1:]])
                    for val in first_decomposed
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

    def __init__(self, typ: Union[type, str], kv_pairs: Sequence[KVPair]) -> None:
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

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
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
    extra_keys: Optional[Value] = None
    """The type of unknown keys, if any."""
    extra_keys_readonly: bool = False
    """Whether the extra keys are readonly."""

    def __init__(
        self,
        items: dict[str, TypedDictEntry],
        extra_keys: Optional[Value] = None,
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

    def all_keys_required(self) -> bool:
        return all(entry.required for entry in self.items.values())

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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
        entries: list[tuple[str, object]] = list(self.items.items())
        if self.extra_keys is not None and self.extra_keys is not NO_RETURN_VALUE:
            extra_typ = str(self.extra_keys)
            if self.extra_keys_readonly:
                extra_typ = f"ReadOnly[{extra_typ}]"
            entries.append(("__extra_items__", extra_typ))
        items = [f'"{key}": {entry}' for key, entry in entries]
        closed = ", closed=True" if self.extra_keys is not None else ""
        return f"TypedDict({{{', '.join(items)}}}{closed})"

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items)))

    def walk_values(self) -> Iterable["Value"]:
        yield self
        for entry in self.items.values():
            yield from entry.typ.walk_values()


@dataclass(unsafe_hash=True, init=False)
class AsyncTaskIncompleteValue(GenericValue):
    """A :class:`GenericValue` representing an async task.

    This should probably just be replaced with ``GenericValue``.

    """

    value: Value
    """The value returned by the task on completion."""

    def __init__(self, typ: Union[type, str], value: Value) -> None:
        super().__init__(typ, (value,))
        self.value = value

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
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
        fallback: Union[type, str] = collections.abc.Callable,
    ) -> None:
        super().__init__(fallback)
        self.signature = signature

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return CallableValue(self.signature.substitute_typevars(typevars), self.typ)

    def walk_values(self) -> Iterable[Value]:
        yield self
        yield from self.signature.walk_values()

    def get_asynq_value(self) -> Value:
        """Return the CallableValue for the .asynq attribute of an AsynqCallable."""
        sig = self.signature.get_asynq_value()
        return CallableValue(sig, self.typ)

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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
) -> Optional[CanAssignError]:
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
    exactly: bool = False
    """If True, represents exactly this class and not a subclass."""

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return self.make(self.typ.substitute_typevars(typevars), exactly=self.exactly)

    def get_type_object(
        self, ctx: CanAssignContext
    ) -> "pycroscope.type_object.TypeObject":
        if isinstance(self.typ, TypedValue) and safe_isinstance(self.typ.typ, type):
            return ctx.make_type_object(type(self.typ.typ))
        # TODO synthetic types
        return pycroscope.type_object.TypeObject(object)

    def walk_values(self) -> Iterable["Value"]:
        yield self
        yield from self.typ.walk_values()

    def is_type(self, typ: type) -> bool:
        if isinstance(self.typ, TypedValue) and isinstance(self.typ.typ, type):
            return safe_issubclass(self.typ.typ, typ)
        return False

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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

    def get_type(self) -> Optional[type]:
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
    def make(cls, origin: Value, *, exactly: bool = False) -> Value:
        if isinstance(origin, MultiValuedValue):
            return unite_values(
                *[cls.make(val, exactly=exactly) for val in origin.vals]
            )
        elif isinstance(origin, AnyValue):
            # Type[Any] is equivalent to plain type
            return TypedValue(type)
        elif isinstance(origin, (TypeVarValue, TypedValue)):
            return cls(origin, exactly=exactly)
        else:
            return AnyValue(AnySource.inference)


@dataclass(frozen=True, order=False)
class IntersectionValue(Value):
    """Represents the intersection of multiple values."""

    vals: tuple[Value, ...]

    def __post_init__(self) -> None:
        assert self.vals, "IntersectionValue must have at least one value"

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return IntersectionValue(
            tuple(val.substitute_typevars(typevars) for val in self.vals)
        )

    def walk_values(self) -> Iterable[Value]:
        yield self
        for val in self.vals:
            yield from val.walk_values()

    def __str__(self) -> str:
        return " & ".join(str(val) for val in self.vals)


@dataclass(frozen=True, order=False)
class MultiValuedValue(Value):
    """Equivalent of ``typing.Union``. Represents the union of multiple values."""

    raw_vals: InitVar[Iterable[Value]]
    vals: tuple[Value, ...] = field(init=False)
    """The underlying values of the union."""
    _known_subvals: Optional[tuple[set[tuple[object, type]], Sequence[Value]]] = field(
        init=False, repr=False, hash=False, compare=False
    )

    def __post_init__(self, raw_vals: Iterable[Value]) -> None:
        object.__setattr__(
            self,
            "vals",
            tuple(chain.from_iterable(flatten_values(val) for val in raw_vals)),
        )
        object.__setattr__(self, "_known_subvals", self._get_known_subvals())

    def _get_known_subvals(
        self,
    ) -> Optional[tuple[set[tuple[object, type]], Sequence[Value]]]:
        # Not worth it for small unions
        if len(self.vals) < 10:
            return None
        # Optimization for comparing Unions containing large unions of literals.
        try:
            # Include the type to avoid e.g. 1 and True matching
            known_values = {
                (subval.val, type(subval.val))
                for subval in self.vals
                if isinstance(subval, KnownValue)
            }
        except TypeError:
            return None  # not hashable
        else:
            # Make remaining check not consider the KnownValues again
            remaining_vals = [
                subval for subval in self.vals if not isinstance(subval, KnownValue)
            ]
            return known_values, remaining_vals

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        if not self.vals or not typevars:
            return self
        return MultiValuedValue(
            [val.substitute_typevars(typevars) for val in self.vals]
        )

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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

    typevar: TypeVarLike
    value: Value

    def __str__(self) -> str:
        return f"{self.typevar} >= {self.value}"


@dataclass(frozen=True)
class UpperBound(Bound):
    """UpperBound(T, V) means the value of T must be assignable to V."""

    typevar: TypeVarLike
    value: Value

    def __str__(self) -> str:
        return f"{self.typevar} <= {self.value}"


@dataclass(frozen=True)
class OrBound(Bound):
    """At least one of the specified bounds must be true."""

    bounds: Sequence[Sequence[Bound]]


@dataclass(frozen=True)
class IsOneOf(Bound):
    typevar: TypeVarLike
    constraints: Sequence[Value]


@dataclass(frozen=True)
class TypeVarValue(Value):
    """Value representing a ``typing.TypeVar``.

    Currently, variance is ignored.

    """

    typevar: TypeVarLike
    bound: Optional[Value] = None
    default: Optional[Value] = None  # unsupported
    constraints: Sequence[Value] = ()
    is_typevartuple: bool = False  # unsupported

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        return typevars.get(self.typevar, self)

    def get_inherent_bounds(self) -> Iterator[Bound]:
        if self.bound is not None:
            yield UpperBound(self.typevar, self.bound)
        elif self.constraints:
            yield IsOneOf(self.typevar, self.constraints)
        # TODO: Consider adding this, but it leads to worse type inference
        # in some cases (inferring object where we should infer Any). Examples
        # in the taxonomy repo.
        # else:
        #     yield UpperBound(self.typevar, TypedValue(object))

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
        return self.get_fallback_value().can_overlap(other, ctx, mode)

    def make_bounds_map(
        self, bounds: Sequence[Bound], other: Value, ctx: CanAssignContext
    ) -> CanAssign:
        bounds_map = {self.typevar: bounds}
        _, errors = pycroscope.typevar.resolve_bounds_map(bounds_map, ctx)
        if errors:
            return CanAssignError(f"Value of {self} cannot be {other}", list(errors))
        return bounds_map

    def get_fallback_value(self) -> Value:
        if self.bound is not None:
            return self.bound
        elif self.constraints:
            return unite_values(*self.constraints)
        return AnyValue(AnySource.inference)  # TODO: should be object

    def get_type_value(self) -> Value:
        return self.get_fallback_value().get_type_value()

    def __str__(self) -> str:
        if self.bound is not None:
            return f"{self.typevar}: {self.bound}"
        elif self.constraints:
            constraints = ", ".join(map(str, self.constraints))
            return f"{self.typevar}: ({constraints})"
        return str(self.typevar)


SelfTVV = TypeVarValue(SelfT)


def set_self(value: Value, self_value: Value) -> Value:
    return value.substitute_typevars({SelfT: self_value})


@dataclass(frozen=True)
class ParamSpecArgsValue(Value):
    param_spec: ParamSpec

    def __str__(self) -> str:
        return f"{self.param_spec}.args"

    def get_fallback_value(self) -> Value:
        return GenericValue(tuple, [TypedValue(object)])


@dataclass(frozen=True)
class ParamSpecKwargsValue(Value):
    param_spec: ParamSpec

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
                            "Incompatible types in TypeIs", children=[left_can_assign]
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
class HasAttrGuardExtension(Extension):
    """An :class:`Extension` used in a function return type. Used to
    indicate that the function argument named `varname` has an attribute
    named `attribute_name` of type `attribute_type`.

    Corresponds to :class:`pycroscope.extensions.HasAttrGuard`.

    """

    varname: str
    attribute_name: Value
    attribute_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return HasAttrGuardExtension(
            self.varname,
            self.attribute_name.substitute_typevars(typevars),
            self.attribute_type.substitute_typevars(typevars),
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.attribute_name.walk_values()
        yield from self.attribute_type.walk_values()


@dataclass(frozen=True)
class HasAttrExtension(Extension):
    """Attached to an object to indicate that it has the given attribute.

    These cannot be created directly from user code, only through the
    :class:`pycroscope.extension.HasAttrGuard` mechanism. This is
    because of potential code like this::

        def f(x: Annotated[object, HasAttr["y", int]]) -> None:
            return x.y

    Here, we would correctly type check the function body, but we currently
    have no way to enforce that the function is only called with arguments that
    obey the constraint.

    """

    attribute_name: Value
    attribute_type: Value

    def substitute_typevars(self, typevars: TypeVarMap) -> Extension:
        return HasAttrExtension(
            self.attribute_name.substitute_typevars(typevars),
            self.attribute_type.substitute_typevars(typevars),
        )

    def walk_values(self) -> Iterable[Value]:
        yield from self.attribute_name.walk_values()
        yield from self.attribute_type.walk_values()


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
class AnnotatedValue(Value):
    """Value representing a `PEP 593 <https://www.python.org/dev/peps/pep-0593/>`_ Annotated object.

    Pycroscope uses ``Annotated`` types to represent types with some extra
    information added to them in the form of :class:`Extension` objects.

    """

    value: Value
    """The underlying value."""
    metadata: tuple[Union[Value, Extension], ...]
    """The extensions associated with this value."""

    def __init__(
        self, value: Value, metadata: Sequence[Union[Value, Extension]]
    ) -> None:
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", tuple(metadata))

    def is_type(self, typ: type) -> bool:
        return self.value.is_type(typ)

    def get_type(self) -> Optional[type]:
        return self.value.get_type()

    def get_type_value(self) -> Value:
        return self.value.get_type_value()

    def substitute_typevars(self, typevars: TypeVarMap) -> Value:
        metadata = tuple(val.substitute_typevars(typevars) for val in self.metadata)
        return AnnotatedValue(self.value.substitute_typevars(typevars), metadata)

    def can_overlap(
        self, other: Value, ctx: CanAssignContext, mode: OverlapMode
    ) -> Optional[CanAssignError]:
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
    ) -> Optional[CanAssignError]:
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


SimpleType: typing_extensions.TypeAlias = Union[
    AnyValue,
    KnownValue,
    SyntheticModuleValue,
    UnboundMethodValue,
    TypedValue,
    SubclassValue,
]

BasicType: typing_extensions.TypeAlias = Union[
    SimpleType, MultiValuedValue, IntersectionValue
]

# Subclasses of Value that represent real types in the type system.
# There are a few other subclasses of Value that represent temporary
# objects in some contexts; this alias exists to make it easier to refer
# to those Values that are actually part of the type system.
GradualType: typing_extensions.TypeAlias = Union[
    BasicType,
    # Invariant: all non-basic types support get_fallback_value()
    TypeAliasValue,
    NewTypeValue,
    TypeVarValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    AnnotatedValue,
]

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
) -> Union[TypeVarMap, CanAssignError]:
    bounds_map = left.can_assign(right, ctx)
    if isinstance(bounds_map, CanAssignError):
        return bounds_map
    tv_map, errors = pycroscope.typevar.resolve_bounds_map(bounds_map, ctx)
    if errors:
        return CanAssignError(children=list(errors))
    return tv_map


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


def annotate_value(origin: Value, metadata: Sequence[Union[Value, Extension]]) -> Value:
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
    # the all_of_type call is redundant but necessary for pycroscope's narrower for now
    # TODO remove it
    if matches and all_of_type(matches, Extension):
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
IterableValue = GenericValue(collections.abc.Iterable, [TypeVarValue(T)])
AsyncIterableValue = GenericValue(collections.abc.AsyncIterable, [TypeVarValue(T)])


class GetItemProto(Protocol[T]):
    def __getitem__(self, __i: int) -> T:
        raise NotImplementedError


GetItemProtoValue = GenericValue(GetItemProto, [TypeVarValue(T)])


def concrete_values_from_iterable(
    value: Value, ctx: CanAssignContext
) -> Union[CanAssignError, Value, Sequence[Value]]:
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
    iterable_type = is_iterable(value, ctx)
    if isinstance(iterable_type, Value):
        val = iterable_type
    else:
        getitem_tv_map = get_tv_map(GetItemProtoValue, value, ctx)
        if not isinstance(getitem_tv_map, CanAssignError):
            val = getitem_tv_map.get(T, AnyValue(AnySource.generic_argument))
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

EMPTY_DICTS = (KnownValue({}), DictIncompleteValue(dict, []))


# This is all the runtime requires in places like {**k}
class CustomMapping(Protocol[K, V]):
    def keys(self) -> Iterable[K]:
        raise NotImplementedError

    def __getitem__(self, __key: K) -> V:
        raise NotImplementedError


NominalMappingValue = GenericValue(
    collections.abc.Mapping, [TypeVarValue(K), TypeVarValue(V)]
)
ProtocolMappingValue = GenericValue(CustomMapping, [TypeVarValue(K), TypeVarValue(V)])


def kv_pairs_from_mapping(
    value_val: Value, ctx: CanAssignContext
) -> Union[Sequence[KVPair], CanAssignError]:
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
        pairs = [
            KVPair(KnownValue(key), entry.typ, is_required=entry.required)
            for key, entry in value_val.items.items()
        ]
        if value_val.extra_keys is not None:
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
        key_type = can_assign.get(K, AnyValue(AnySource.generic_argument))
        value_type = can_assign.get(V, AnyValue(AnySource.generic_argument))
        return [KVPair(key_type, value_type, is_many=True)]


def unpack_values(
    value: Value,
    ctx: CanAssignContext,
    target_length: int,
    post_starred_length: Optional[int] = None,
) -> Union[Sequence[Value], CanAssignError]:
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


def is_iterable(value: Value, ctx: CanAssignContext) -> Union[CanAssignError, Value]:
    """Check whether a value is iterable."""
    tv_map = get_tv_map(IterableValue, value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get(T, AnyValue(AnySource.generic_argument))


def is_async_iterable(
    value: Value, ctx: CanAssignContext
) -> Union[CanAssignError, Value]:
    """Check whether a value is an async iterable."""
    tv_map = get_tv_map(AsyncIterableValue, value, ctx)
    if isinstance(tv_map, CanAssignError):
        return tv_map
    return tv_map.get(T, AnyValue(AnySource.generic_argument))


def _create_unpacked_list(
    iterable_type: Value, target_length: int, post_starred_length: Optional[int]
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
    value: SequenceValue, target_length: int, post_starred_length: Optional[int]
) -> Union[Sequence[Value], CanAssignError]:
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
            return [*head, SequenceValue(list, remaining_members), *reversed(tail)]


def replace_known_sequence_value(value: Value) -> BasicType:
    """Simplify a Value in a way that is easier to handle for most typechecking use cases.

    Does the following:

    - Replace AnnotatedValue with its inner type
    - Replace TypeVarValue with its fallback type
    - Replace KnownValues representing list, tuples, sets, or dicts with
      SequenceValue or DictIncompleteValue.

    """
    value = replace_fallback(value)
    if isinstance(value, KnownValue):
        return typify_literal(value)
    return value


def typify_literal(value: KnownValue) -> Union[KnownValue, TypedValue]:
    if isinstance(value.val, (list, tuple, set)):
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
    if isinstance(left, MultiValuedValue) and left.vals:
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


@dataclass(frozen=True)
class AnnotationExpr:
    ctx: "pycroscope.annotations.Context"
    _value: Optional[Value]
    qualifiers: Sequence[tuple[Qualifier, Optional[ast.AST]]] = field(
        default_factory=list
    )
    metadata: Sequence[Union[Value, Extension]] = field(default_factory=list)

    def add_qualifier(
        self, qualifier: Qualifier, node: Optional[ast.AST]
    ) -> "AnnotationExpr":
        return AnnotationExpr(
            self.ctx,
            self._value,
            qualifiers=[*self.qualifiers, (qualifier, node)],
            metadata=self.metadata,
        )

    def add_metadata(
        self, metadata: Sequence[Union[Value, Extension]]
    ) -> "AnnotationExpr":
        return AnnotationExpr(
            self.ctx,
            self._value,
            qualifiers=self.qualifiers,
            metadata=[*self.metadata, *metadata],
        )

    def to_value(
        self, *, allow_qualifiers: bool = False, allow_empty: bool = False
    ) -> Value:
        if self._value is None:
            if allow_empty:
                return AnyValue(AnySource.incomplete_annotation)
            else:
                innermost, node = self.qualifiers[-1]
                self.ctx.show_error(
                    f"Invalid bare {innermost.name} annotation", node=node
                )
                return AnyValue(AnySource.error)
        if not allow_qualifiers:
            for qualifier, node in self.qualifiers:
                self.ctx.show_error(
                    f"Unexpected {qualifier.name} annotation", node=node
                )
        if self.metadata:
            return annotate_value(self._value, self.metadata)
        return self._value

    def unqualify(
        self, allowed_qualifiers: Container[Qualifier] = frozenset()
    ) -> tuple[Value, set[Qualifier]]:
        value, qualifiers = self.maybe_unqualify(allowed_qualifiers)
        if value is None:
            innermost, node = self.qualifiers[-1]
            self.ctx.show_error(f"Invalid bare {innermost.name} annotation", node=node)
            return AnyValue(AnySource.error), set()
        return value, qualifiers

    def maybe_unqualify(
        self, allowed_qualifiers: Container[Qualifier] = frozenset()
    ) -> tuple[Optional[Value], set[Qualifier]]:
        qualifiers = set()
        for qualifier, node in self.qualifiers:
            if qualifier in allowed_qualifiers:
                qualifiers.add(qualifier)
            else:
                self.ctx.show_error(
                    f"Unexpected {qualifier.name} annotation", node=node
                )
        if self._value is None:
            return None, qualifiers
        if self.metadata:
            value = annotate_value(self._value, self.metadata)
        else:
            value = self._value
        return value, qualifiers
