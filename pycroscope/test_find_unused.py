# static analysis: ignore
import ast
import sys
import textwrap
import types

from .error_code import DISABLED_IN_TESTS, ErrorCode
from .name_check_visitor import (
    IgnoredUnusedAttributePaths,
    IgnoredUnusedClassAttributes,
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
    def assert_unused_attributes(
        self,
        code_str,
        expected_unused,
        extra_options=(),
        module=None,
        *,
        should_serialize=False,
    ):
        code_str = textwrap.dedent(code_str)
        tree = ast.parse(code_str, "<test input>")
        settings = {code: code not in DISABLED_IN_TESTS for code in ErrorCode}
        kwargs = self.visitor_cls.prepare_constructor_kwargs(
            {"settings": settings}, extra_options=extra_options
        )
        module = _make_module(code_str) if module is None else module
        with ClassAttributeChecker(
            enabled=True,
            should_check_unused_attributes=True,
            should_serialize=should_serialize,
            options=kwargs["checker"].options,
        ) as attribute_checker:
            visitor = self.visitor_cls(
                "<test input>",
                code_str,
                tree,
                module=module,
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

    @staticmethod
    def _make_named_module(code_str: str, module_name: str) -> types.ModuleType:
        module = types.ModuleType(module_name)
        module.__file__ = f"{module_name}.py"
        sys.modules[module_name] = module
        exec(compile(code_str, module.__file__, "exec"), module.__dict__)
        return module

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

    def test_unused_attribute_can_be_ignored_by_full_path(self):
        code = """
            class Config:
                enabled = True

                def __init__(self) -> None:
                    self.cache = 0
            """
        module = _make_module(textwrap.dedent(code))
        self.assert_unused_attributes(
            code,
            {("Config", "enabled", False)},
            extra_options=(
                IgnoredUnusedAttributePaths(
                    [f"{module.__name__}.Config.cache"], from_command_line=True
                ),
            ),
            module=module,
        )

    def test_unused_attributes_can_be_ignored_for_related_classes_in_serialized_mode(
        self,
    ):
        code = """
            class Base:
                helper = 1

            class Child(Base):
                pass
            """
        module = _make_module(textwrap.dedent(code))
        self.assert_unused_attributes(
            code,
            set(),
            extra_options=(
                IgnoredUnusedClassAttributes(
                    [(module.Base, {"helper"})], from_command_line=True
                ),
            ),
            module=module,
            should_serialize=True,
        )

    def test_setup_method_is_ignored_as_unused_test_helper(self):
        self.assert_unused_attributes(
            """
            class Helper:
                def setup_method(self) -> None:
                    pass
            """,
            set(),
        )

    def test_test_methods_can_be_ignored_based_on_module_name(self):
        code = """
            class Helper:
                def test_case(self) -> None:
                    pass
            """
        module = self._make_named_module(textwrap.dedent(code), "test_helpers")
        self.assert_unused_attributes(code, set(), module=module)

    def test_server_call_attributes_are_ignored(self):
        self.assert_unused_attributes(
            """
            class ServerCall:
                server_call = True

                def __get__(self, obj: object, owner: object) -> "ServerCall":
                    return self

            class Api:
                fetch = ServerCall()
            """,
            {("ServerCall", "server_call", False)},
        )

    def test_hasattr_inferred_attributes_are_not_reported_unused(self):
        self.assert_unused_attributes(
            """
            def use(obj: object) -> None:
                if hasattr(obj, "task_cls"):
                    print(obj.task_cls)
                if hasattr(obj, "decorator") and hasattr(obj.decorator, "task_cls"):
                    print(obj.decorator.task_cls)
            """,
            set(),
        )

    def test_test_helper_modules_can_be_ignored_by_predicate(self):
        helper = type("Helper", (), {})
        helper.__module__ = "pycroscope.tests"
        assert _ignore_unused_test_helper_attributes(helper, "anything")

        production = type("Production", (), {})
        production.__module__ = "pycroscope.name_check_visitor"
        assert not _ignore_unused_test_helper_attributes(production, "anything")
