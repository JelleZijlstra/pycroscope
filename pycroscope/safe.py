"""

"Safe" operations that call into user code and catch any exceptions.

"""

import inspect
import sys
import types
import typing
from collections.abc import Container, Sequence
from typing import Any, NewType, TypeVar

import typing_extensions

try:
    import mypy_extensions
except ImportError:
    mypy_extensions = None

from .maybe_asynq import asynq

T = TypeVar("T")


def hasattr_static(object: object, name: str) -> bool:
    """Similar to ``inspect.getattr_static()``."""
    try:
        inspect.getattr_static(object, name)
    except AttributeError:
        return False
    else:
        return True


def safe_hasattr(item: object, member: str) -> bool:
    """Safe version of ``hasattr()``."""
    try:
        # some sketchy implementation (like paste.registry) of
        # __getattr__ cause hasattr() to throw an error.
        return hasattr(item, member)
    except Exception:
        return False


def safe_getattr(value: object, attr: str, default: object) -> Any:
    """Return ``getattr(value, attr)`` or ``default`` if access raises an exception."""
    try:
        return getattr(value, attr)
    except Exception:
        return default


def safe_equals(left: object, right: object) -> bool:
    """Safely check whether two objects are equal."""
    try:
        return bool(left == right)
    except Exception:
        return False


def safe_issubclass(cls: type, class_or_tuple: type | tuple[type, ...]) -> bool:
    """Safe version of ``issubclass()``.

    Apart from incorrect arguments, ``issubclass(a, b)`` can throw an error
    only if `b` has a ``__subclasscheck__`` method that throws an error.
    Therefore, it is not necessary to use ``safe_issubclass()`` if the class
    is known to not override ``__subclasscheck__``.

    Defaults to False if ``issubclass()`` throws.

    """
    try:
        return issubclass(cls, class_or_tuple)
    except Exception:
        return False


def safe_isinstance(
    obj: object, class_or_tuple: type[T] | tuple[type[T], ...]
) -> typing_extensions.TypeIs[T]:
    """Safe version of ``isinstance()``.

    ``isinstance(a, b)`` can throw an error in the following circumstances:

    - ``b`` is not a class
    - ``b`` has an ``__instancecheck__`` method that throws an error
    - ``a`` has a ``__class__`` property that throws an error

    Therefore, ``safe_isinstance()`` must be used when doing ``isinstance`` checks
    on arbitrary objects that come from user code.

    Defaults to False if ``isinstance()`` throws.

    """
    try:
        return isinstance(obj, class_or_tuple)
    except Exception:
        return False


def safe_in(item: T, collection: Container[T]) -> bool:
    """Safely check whether item is in collection. Defaults to returning false."""
    # Workaround against mock objects sometimes throwing ValueError if you compare them,
    # and against objects throwing other kinds of errors if you use in.
    try:
        return item in collection
    except Exception:
        return False


def is_hashable(obj: object) -> bool:
    """Return whether an object is hashable."""
    try:
        hash(obj)
    except Exception:
        return False
    else:
        return True


def all_of_type(
    elts: Sequence[object], typ: type[T]
) -> typing_extensions.TypeGuard[Sequence[T]]:
    """Returns whether all elements of elts are instances of typ."""
    return all(isinstance(elt, typ) for elt in elts)


if sys.version_info >= (3, 10):

    def is_newtype(obj: object) -> bool:
        return isinstance(obj, NewType)

else:

    def is_newtype(obj: object) -> bool:
        return (
            inspect.isfunction(obj)
            and hasattr(obj, "__supertype__")
            and isinstance(obj.__supertype__, type)
        )


def is_typing_name(obj: object, name: str) -> bool:
    objs, names = _fill_typing_name_cache(name)
    for typing_obj in objs:
        if obj is typing_obj:
            return True
    return safe_in(obj, names)


try:
    from types import UnionType
except ImportError:
    UnionType = None


def is_union(obj: object) -> bool:
    return is_typing_name(obj, "Union") or (UnionType is not None and obj is UnionType)


def is_instance_of_typing_name(obj: object, name: str) -> bool:
    objs, _ = _fill_typing_name_cache(name)
    return isinstance(obj, objs)


_typing_name_cache: dict[str, tuple[tuple[Any, ...], tuple[str, ...]]] = {}


def _fill_typing_name_cache(name: str) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    try:
        return _typing_name_cache[name]
    except KeyError:
        objs = []
        names = []
        for mod in (typing, typing_extensions, mypy_extensions):
            if mod is None:
                continue
            try:
                objs.append(getattr(mod, name))
                names.append(f"{mod}.{name}")
            except AttributeError:
                pass
        result = tuple(objs), tuple(names)
        _typing_name_cache[name] = result
        return result


def get_fully_qualified_name(obj: object) -> str | None:
    if safe_hasattr(obj, "__module__") and safe_hasattr(obj, "__qualname__"):
        return f"{obj.__module__}.{obj.__qualname__}"
    return None


def is_dataclass_type(cls: type) -> bool:
    """Like dataclasses.is_dataclass(), but works correctly for a
    non-dataclass subclass of a dataclass."""
    try:
        return "__dataclass_fields__" in cls.__dict__
    except Exception:
        return False


def is_bound_classmethod(obj: object) -> bool:
    """Returns whether the object is a bound classmethod."""
    return safe_isinstance(obj, types.MethodType) and safe_isinstance(
        obj.__self__, type
    )


def safe_str(obj: object) -> str:
    """Like str(), but catches exceptions."""
    try:
        return str(obj)
    except Exception as e:
        try:
            return f"<error in str(): {e!r}>"
        except Exception:
            return "<error in str() and in error message>"


def is_async_fn(obj: object) -> bool:
    if asynq is None:
        return False
    try:
        return asynq.is_async_fn(obj)
    except Exception:
        # This does certain attribute accesses, which may raise if the object has
        # an unusual __getattr__.
        return False
