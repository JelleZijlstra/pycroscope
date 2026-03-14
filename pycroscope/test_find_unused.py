# static analysis: ignore
import ast
import textwrap

from .error_code import DISABLED_IN_TESTS, ErrorCode
from .name_check_visitor import (
    IgnoreUnusedAttributePredicates,
    _ignore_unused_ast_visit_methods,
    _ignore_unused_test_helper_attributes,
)
from .test_name_check_visitor import (
    ClassAttributeChecker,
    TestNameCheckVisitorBase,
    _make_module,
)


class TestFindUnused(TestNameCheckVisitorBase):
    def assert_unused_attributes(self, code_str, expected_unused, extra_options=()):
        code_str = textwrap.dedent(code_str)
        tree = ast.parse(code_str, "<test input>")
        settings = {code: code not in DISABLED_IN_TESTS for code in ErrorCode}
        kwargs = self.visitor_cls.prepare_constructor_kwargs(
            {"settings": settings}, extra_options=extra_options
        )
        with ClassAttributeChecker(
            enabled=True,
            should_check_unused_attributes=True,
            options=kwargs["checker"].options,
        ) as attribute_checker:
            visitor = self.visitor_cls(
                "<test input>",
                code_str,
                tree,
                module=_make_module(code_str),
                attribute_checker=attribute_checker,
                **kwargs,
            )
            result = visitor.check_for_test()
            result += visitor.perform_final_checks(kwargs)
        assert not result
        actual_unused = {
            (item.typ.__name__, item.attr_name, item.is_method)
            for item in attribute_checker.unused_attributes
        }
        assert actual_unused == set(expected_unused)

    def test_unused_attributes_follow_overrides(self):
        self.assert_unused_attributes(
            """
            class Base:
                value: int = 1

                def method(self) -> int:
                    return 1

            class Child(Base):
                value: int = 2

                def method(self) -> int:
                    return 2

            def use(child: Child) -> int:
                return child.method() + child.value
            """,
            {("Base", "method", True), ("Base", "value", False)},
        )

    def test_unused_attributes_consider_polymorphism(self):
        self.assert_unused_attributes(
            """
            class Base:
                def method(self) -> int:
                    return 1

            class Child(Base):
                def method(self) -> int:
                    return 2

            def use(obj: Base) -> int:
                return obj.method()
            """,
            set(),
        )

    def test_inherited_reads_mark_base_attribute_used(self):
        self.assert_unused_attributes(
            """
            class Base:
                value = 1

            class Child(Base):
                pass

            def use(child: Child) -> int:
                return child.value
            """,
            set(),
        )

    def test_protocol_usage_marks_implementation_members_used(self):
        self.assert_unused_attributes(
            """
            from typing import Protocol

            class Context(Protocol):
                @property
                def visitor(self) -> int: ...

                def on_error(self, message: str) -> None: ...

            class ConcreteContext:
                @property
                def visitor(self) -> int:
                    return 1

                def on_error(self, message: str) -> None:
                    pass

            def use(ctx: Context) -> int:
                ctx.on_error("oops")
                return ctx.visitor

            def run() -> int:
                return use(ConcreteContext())
            """,
            set(),
        )

    def test_unused_nonmethod_attribute(self):
        self.assert_unused_attributes(
            """
            class Config:
                enabled = True

                def __init__(self) -> None:
                    self.cache = 0

                def is_enabled(self) -> bool:
                    return self.enabled

            def use(config: Config) -> bool:
                return config.is_enabled()
            """,
            {("Config", "cache", False)},
        )

    def test_unused_test_methods_are_ignored(self):
        self.assert_unused_attributes(
            """
            class TestThing:
                def test_case(self) -> None:
                    pass

                def helper(self) -> None:
                    pass
            """,
            {("TestThing", "helper", True)},
        )

    def test_unused_enum_internals_are_ignored(self):
        self.assert_unused_attributes(
            """
            import enum

            class Rodent(enum.Enum):
                capybara = 1
                guinea_pig = 2
            """,
            {("Rodent", "capybara", False), ("Rodent", "guinea_pig", False)},
        )

    def test_used_enum_member_is_not_reported_unused(self):
        self.assert_unused_attributes(
            """
            import enum

            class Rodent(enum.Enum):
                capybara = 1
                guinea_pig = 2

            def use() -> Rodent:
                return Rodent.capybara
            """,
            {("Rodent", "guinea_pig", False)},
        )

    def test_unused_namedtuple_framework_attributes_are_ignored(self):
        self.assert_unused_attributes(
            """
            from typing import NamedTuple

            class Pair(NamedTuple):
                left: int
                right: int

            def use(pair: Pair) -> int:
                return pair.left + pair.right
            """,
            set(),
        )

    def test_unused_typed_dict_fields_are_ignored(self):
        self.assert_unused_attributes(
            """
            from typing import TypedDict

            class Failure(TypedDict):
                filename: str
                message: str
            """,
            set(),
        )

    def test_unused_initvar_fields_are_ignored(self):
        self.assert_unused_attributes(
            """
            from dataclasses import InitVar, dataclass

            @dataclass
            class Config:
                raw_vals: InitVar[list[int]]
            """,
            set(),
        )

    def test_visit_methods_can_be_ignored_by_predicate(self):
        self.assert_unused_attributes(
            """
            import ast

            class Visitor(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name) -> None:
                    pass
            """,
            set(),
            extra_options=(
                IgnoreUnusedAttributePredicates(
                    [_ignore_unused_ast_visit_methods], from_command_line=True
                ),
            ),
        )

    def test_generic_visit_can_be_ignored_by_predicate(self):
        self.assert_unused_attributes(
            """
            import ast

            class Visitor(ast.NodeVisitor):
                def generic_visit(self, node: ast.AST) -> None:
                    pass
            """,
            set(),
            extra_options=(
                IgnoreUnusedAttributePredicates(
                    [_ignore_unused_ast_visit_methods], from_command_line=True
                ),
            ),
        )

    def test_test_helper_modules_can_be_ignored_by_predicate(self):
        helper = type("Helper", (), {})
        helper.__module__ = "pycroscope.tests"
        assert _ignore_unused_test_helper_attributes(helper, "anything")

        production = type("Production", (), {})
        production.__module__ = "pycroscope.name_check_visitor"
        assert not _ignore_unused_test_helper_attributes(production, "anything")
