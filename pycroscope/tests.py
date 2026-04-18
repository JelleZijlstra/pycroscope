# static analysis: ignore
"""

Functions to be used in test_scope unit tests.

"""

from collections.abc import Sequence
from typing import ClassVar, Generic, NoReturn, TypeVar

from typing_extensions import Self, final

from .extensions import overload
from .value import SequenceValue, Value, VariableNameValue

ASYNQ_METHOD_NAME = "asynq"
ASYNQ_METHOD_NAMES = ("asynq",)

uid_vnv = VariableNameValue(["uid"])
qid_vnv = VariableNameValue(["qid"])


class Wrapper:
    base: ClassVar[type]


def wrap(cls):
    """Decorator that wraps a class."""

    class NewWrapper(Wrapper):
        base = cls

    NewWrapper.__module__ = cls.__module__
    return NewWrapper


def takes_kwonly_argument(a, **kwargs):
    assert set(kwargs) == {"kwonly_arg"}


class PropertyObject:
    def __init__(self, poid):
        self.poid = poid

    def non_async_method(self):
        pass

    @final
    def decorated_method(self):
        pass

    @property
    def string_property(self) -> str:
        return str(self.poid)

    @property
    def prop(self):
        return 42

    prop_with_get = prop
    prop_with_is = prop

    def _private_method(self):
        pass

    @classmethod
    def no_args_classmethod(cls):
        pass


class KeywordOnlyArguments:
    def __init__(self, *args, **kwargs):
        assert set(kwargs) <= {"kwonly_arg"}


class WhatIsMyName:
    def __init__(self):
        pass


WhatIsMyName.__name__ = "Capybara"
WhatIsMyName.__init__.__name__ = "capybara"


class FailingImpl:
    def __init__(self) -> None:
        pass


def custom_code() -> None:
    pass


@overload
def overloaded() -> int: ...


@overload
def overloaded(x: str) -> str: ...


def overloaded(*args: str) -> int | str:
    if len(args) == 0:
        return len(args)
    elif len(args) == 1:
        return args[0]
    else:
        raise TypeError("too many arguments")


def assert_never(arg: NoReturn) -> NoReturn:
    raise RuntimeError("no way")


def make_simple_sequence(typ: type, vals: Sequence[Value]) -> SequenceValue:
    return SequenceValue(typ, [(False, val) for val in vals])


def make_union_in_annotated() -> object:
    return 42


T = TypeVar("T")


class ImportedCondition:
    pass


class ImportedField(Generic[T]):
    @overload
    def __get__(self, obj: None, owner: object) -> Self: ...

    @overload
    def __get__(self, obj: object, owner: object) -> T: ...

    def __get__(self, obj: object | None, owner: object) -> Self | T:
        if obj is None:
            return self
        return obj  # type: ignore[return-value]

    def contains(self, value: str) -> ImportedCondition:
        return ImportedCondition()

    def is_in(self, value: list[object]) -> ImportedCondition:
        return ImportedCondition()


class ReexportedFieldModel:
    tags = ImportedField[tuple[str, ...]]()
    kind = ImportedField[object]()
