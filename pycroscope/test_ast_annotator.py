import ast
from typing import Callable, Type

from .ast_annotator import annotate_code
from .value import KnownValue, Value, unannotate


def _check_inferred_value(
    tree: ast.Module,
    node_type: Type[ast.AST],
    value: Value,
    predicate: Callable[[ast.AST], bool] = lambda node: True,
) -> None:
    for node in ast.walk(tree):
        if isinstance(node, node_type) and predicate(node):
            assert hasattr(node, "inferred_value"), repr(node)
            assert value == unannotate(node.inferred_value), ast.dump(node)


def test_annotate_code() -> None:
    tree = annotate_code("a = 1")
    _check_inferred_value(tree, ast.Constant, KnownValue(1))
    _check_inferred_value(tree, ast.Name, KnownValue(1))

    tree = annotate_code(
        """
        class X:
            def __init__(self):
                self.a = 1
        """
    )
    _check_inferred_value(tree, ast.Attribute, KnownValue(1))
    tree = annotate_code(
        """
        class X:
            def __init__(self):
                self.a = 1

        x = X()
        x.a + 1
        """
    )
    _check_inferred_value(tree, ast.BinOp, KnownValue(2))

    tree = annotate_code(
        """
        class A:
            def __init__(self):
                self.a = 1

            def bla(self):
                return self.a


        a = A()
        b = a.bla()
        """
    )
    _check_inferred_value(tree, ast.Name, KnownValue(1), lambda node: node.id == "b")
