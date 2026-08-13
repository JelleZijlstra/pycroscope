# static analysis: ignore
from .error_code import ErrorCode
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_fails, assert_passes


class TestOverride(TestNameCheckVisitorBase):
    @assert_fails(ErrorCode.invalid_override_decorator)
    def test_invalid_usage(self):
        from typing_extensions import override

        @override
        def not_a_method():
            pass

    @assert_passes()
    def test_valid_method(self):
        from typing_extensions import override

        class Base:
            def method(self):
                pass

        class Capybara(Base):
            @override
            def method(self):
                pass

    @assert_passes()
    def test_constructor_compatibility_is_checked_for_explicit_override(self):
        from typing_extensions import override

        class Parent:
            def __init__(self, x: int) -> None: ...

            def __new__(cls, x: int) -> "Parent":
                raise NotImplementedError

        class GoodChild(Parent):
            @override
            def __init__(self, x: int) -> None: ...

            @override
            def __new__(cls, x: int) -> "GoodChild":
                raise NotImplementedError

        class BadChild(Parent):
            @override
            def __init__(self, x: str) -> None:  # E: incompatible_override
                pass

            @override
            def __new__(cls, x: str) -> "BadChild":  # E: incompatible_override
                raise NotImplementedError

        class UndecoratedChild(Parent):
            def __init__(self, x: str) -> None: ...

            def __new__(cls, x: str) -> "UndecoratedChild":
                raise NotImplementedError

    @assert_fails(ErrorCode.override_does_not_override)
    def test_invalid_method(self):
        from typing_extensions import override

        class Base:
            def method(self):
                pass

        class Capybara(Base):
            @override
            def no_base_method(self):  # E: override_does_not_override
                pass

    @assert_passes()
    def test_any_derived_base(self):
        from typing_extensions import Any, override

        class Base(Any):
            pass

        class Capybara(Base):
            @override
            def method(self):
                pass
