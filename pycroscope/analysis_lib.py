"""

Commonly useful components for static analysis tools.

"""

import ast
import importlib
import linecache
import os
import secrets
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, ParamSpec, TypeVar, Union

from pycroscope.find_unused import used

T = TypeVar("T")
P = ParamSpec("P")


def _all_files(
    root: Union[str, Path], filter_function: Optional[Callable[[str], bool]] = None
) -> set[str]:
    """Returns the set of all files at the given root.

    Filtered optionally by the filter_function.

    """
    all_files = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filter_function is not None and not filter_function(filename):
                continue
            all_files.add(os.path.join(dirpath, filename))
    return all_files


def files_with_extension_from_directory(
    extension: str, dirname: Union[str, Path]
) -> set[str]:
    """Finds all files in a given directory with this extension."""
    return _all_files(dirname, filter_function=lambda fn: fn.endswith("." + extension))


def get_indentation(line: str) -> int:
    """Returns the indentation of a line of code."""
    if len(line.lstrip()) == 0:
        # if it is a newline or a line with just spaces
        return 0
    return len(line) - len(line.lstrip())


def get_line_range_for_node(
    node: Union[ast.stmt, ast.expr], lines: list[str]
) -> list[int]:
    """Returns the lines taken up by a Python ast node.

    lines is a list of code lines for the file the node came from.

    """
    first_lineno = node.lineno
    # iterate through all childnodes and find the max lineno
    last_lineno = first_lineno + 1
    for childnode in ast.walk(node):
        end_lineno = getattr(childnode, "end_lineno", None)
        if end_lineno is not None:
            last_lineno = max(last_lineno, end_lineno)
        elif hasattr(childnode, "lineno"):
            last_lineno = max(last_lineno, childnode.lineno)

    def is_part_of_same_node(first_line: str, line: str) -> bool:
        current_indent = get_indentation(line)
        first_indent = get_indentation(first_line)
        if current_indent > first_indent:
            return True
        # because closing parenthesis are at the same indentation
        # as the expression
        line = line.lstrip()
        if len(line) == 0:
            # if it is just a newline then the node has likely ended
            return False
        if current_indent == first_indent and line.lstrip()[0] in [")", "]", "}"]:
            return True
        # probably part of the same multiline string
        for multiline_delim in ('"""', "'''"):
            if multiline_delim in first_line and line.strip() == multiline_delim:
                return True
        return False

    first_line = lines[first_lineno - 1]

    while last_lineno - 1 < len(lines) and is_part_of_same_node(
        first_line, lines[last_lineno - 1]
    ):
        last_lineno += 1
    return list(range(first_lineno, last_lineno))


@dataclass
class _FakeLoader:
    source: str

    def get_source(self, name: object) -> str:
        return self.source


def make_module(
    code_str: str, extra_scope: Mapping[str, object] = {}
) -> types.ModuleType:
    """Creates a Python module with the given code."""
    # Make the name unique to avoid clobbering the overloads dict
    # from pycroscope.extensions.overload.
    token = secrets.token_hex()
    module_name = f"<test input {secrets.token_hex()}>"
    filename = f"{token}.py"
    mod = types.ModuleType(module_name)
    scope = mod.__dict__
    scope["__name__"] = module_name
    scope["__file__"] = filename
    scope["__loader__"] = _FakeLoader(code_str)

    # This allows linecache later to retrieve source code
    # from this module, which helps the type evaluator.
    linecache.lazycache(filename, scope)
    scope.update(extra_scope)
    code = compile(code_str, filename, "exec")
    exec(code, scope)
    sys.modules[module_name] = mod
    return mod


def is_positional_only_arg_name(name: str, class_name: Optional[str] = None) -> bool:
    # https://www.python.org/dev/peps/pep-0484/#positional-only-arguments
    # Work around Python's name mangling
    if class_name is not None:
        prefix = f"_{class_name}"
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.startswith("__") and not name.endswith("__")


def get_attribute_path(node: ast.AST) -> Optional[list[str]]:
    """Gets the full path of an attribute lookup.

    For example, the code string "a.model.question.Question" will resolve to the path
    ['a', 'model', 'question', 'Question']. This is used for comparing such paths to
    lists of functions that we treat specially.

    """
    if isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Attribute):
        root_value = get_attribute_path(node.value)
        if root_value is None:
            return None
        root_value.append(node.attr)
        return root_value
    else:
        return None


class override:
    """Temporarily overrides an attribute of an object."""

    def __init__(self, obj: Any, attr: str, value: Any) -> None:
        self.obj = obj
        self.attr = attr
        self.value = value

    def __enter__(self) -> None:
        self.old_value = getattr(self.obj, self.attr)
        setattr(self.obj, self.attr, self.value)

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        setattr(self.obj, self.attr, self.old_value)


def object_from_string(object_reference: str) -> object:
    if ":" in object_reference:
        module_name, object_name = object_reference.split(":")
        mod = importlib.import_module(module_name)
        obj = mod
        for part in object_name.split("."):
            obj = getattr(obj, part)
        return obj
    else:
        parts = object_reference.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_path = parts[:i]
            object_name = parts[i:]
            try:
                mod = importlib.import_module(".".join(module_path))
            except ImportError:
                if i == 1:
                    raise
            else:
                obj = mod
                try:
                    for part in object_name:
                        obj = getattr(obj, part)
                except AttributeError:
                    if i == 1:
                        raise
                    else:
                        continue
                return obj
        raise ValueError(f"Could not find object {object_reference}")


def get_subclasses_recursively(cls: type[T]) -> set[type[T]]:
    """Returns all subclasses of a class recursively."""
    all_subclasses = set()
    for subcls in type.__subclasses__(cls):
        try:
            all_subclasses.add(subcls)
        except TypeError:
            pass  # Ignore unhashable classes
        all_subclasses.update(get_subclasses_recursively(subcls))
    return all_subclasses


def is_cython_class(cls: type[object]) -> bool:
    """Returns whether a class is a Cython extension class."""
    return "__pyx_vtable__" in cls.__dict__


@dataclass(frozen=True)
class Sentinel:
    name: str


@used
def trace(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to trace function calls."""

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result = func(*args, **kwargs)
        pieces = [func.__name__, "("]
        for i, arg in enumerate(args):
            if i > 0:
                pieces.append(", ")
            pieces.append(str(arg))
        for i, (k, v) in enumerate(kwargs.items()):
            if i > 0 or args:
                pieces.append(", ")
            pieces.append(f"{k}={v}")
        pieces.append(")")
        pieces.append(" -> ")
        pieces.append(str(result))
        print("".join(pieces))
        return result

    return wrapper
