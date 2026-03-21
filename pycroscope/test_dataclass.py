# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestDataclass(TestNameCheckVisitorBase):
    @assert_passes()
    def test_only_known_attributes(self):
        from dataclasses import dataclass
        from typing import NamedTuple

        @dataclass
        class DC:
            a: int

        class NT(NamedTuple):
            a: int

        def capybara(dc: DC, nt: NT) -> None:
            assert_type(dc.a, int)
            assert_type(nt.a, int)

            dc.b  # E: undefined_attribute
            nt.b  # E: undefined_attribute

    @assert_passes()
    def test_union(self):
        from dataclasses import dataclass
        from typing import Union

        @dataclass
        class Capybara:
            attr: int

        @dataclass
        class Paca:
            attr: str

        def test(x: Union[Capybara, Paca]) -> None:
            assert_is_value(
                x.attr, MultiValuedValue([TypedValue(int), TypedValue(str)])
            )

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_comparison_after_import_failure(self):
        from dataclasses import dataclass

        @dataclass(order=True)
        class DC1:
            a: str
            b: int

        @dataclass(order=True)
        class DC2:
            a: str
            b: int

        dc1_1 = DC1("", 0)
        dc2_1 = DC2("hi", 2)

        def capybara() -> None:
            if dc1_1 < dc2_1:  # E: unsupported_operation
                pass

            if dc1_1 != dc2_1:
                pass

    @assert_passes()
    def test_frozen_dataclass_disallows_instance_attribute_assignment(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Frozen:
            value: int

        def mutate() -> None:
            frozen = Frozen(1)
            frozen.value = 2  # E: incompatible_assignment

    @assert_passes()
    def test_dataclass_rejects_classvar_instance_override_mismatch(self):
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass
        class Base:
            x: int
            y: ClassVar[int] = 1

        @dataclass
        class Child(Base):
            x: ClassVar[int]  # E: incompatible_override
            y: int  # E: incompatible_override

    @assert_passes()
    def test_frozen_dataclass_disallows_intersection_attribute_assignment(self):
        from dataclasses import dataclass

        from pycroscope.extensions import Intersection

        @dataclass(frozen=True)
        class FrozenA:
            value: int

        @dataclass(frozen=True)
        class FrozenB:
            value: int

        def mutate(value: Intersection[FrozenA, FrozenB]) -> None:
            value.value = 2  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_frozen_dataclass_checks_after_import_failure(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Frozen:
            value: int

        @dataclass
        class Mutable:
            value: int

        def capybara() -> None:
            frozen = Frozen(1)
            frozen.value = 2  # E: incompatible_assignment

            @dataclass
            class NonFrozenChild(Frozen):  # E: invalid_base
                pass

            @dataclass(frozen=True)
            class FrozenChild(Mutable):  # E: invalid_base
                pass

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_classvar_instance_override_mismatch_after_import_failure(self):
        from dataclasses import dataclass
        from typing import ClassVar

        @dataclass
        class Base:
            x: int
            y: ClassVar[int] = 1

        @dataclass
        class Child(Base):
            x: ClassVar[int]  # E: incompatible_override
            y: int  # E: incompatible_override

    @assert_passes()
    def test_dataclass_hashability(self):
        from dataclasses import dataclass
        from typing import Hashable

        @dataclass
        class Unhashable:
            value: int

        @dataclass(frozen=True)
        class Frozen:
            value: int

        @dataclass(eq=False)
        class NoEq:
            value: int

        @dataclass(unsafe_hash=True)
        class UnsafeHash:
            value: int

        @dataclass
        class ExplicitHash:
            value: int

            def __hash__(self) -> int:
                return 0

        @dataclass(eq=False)
        class ExplicitlyUnhashable:
            value: int
            __hash__ = None

        bad_unhashable: Hashable = Unhashable(1)  # E: incompatible_assignment
        ok_frozen: Hashable = Frozen(1)
        ok_no_eq: Hashable = NoEq(1)
        ok_unsafe_hash: Hashable = UnsafeHash(1)
        ok_explicit_hash: Hashable = ExplicitHash(1)
        # E: incompatible_assignment
        bad_explicit_none: Hashable = ExplicitlyUnhashable(1)

    @assert_passes()
    def test_dataclass_kw_only_marker_is_allowed(self):
        from dataclasses import KW_ONLY, dataclass

        @dataclass
        class DC:
            a: str
            _: KW_ONLY
            b: int = 0

        DC("hi")
        DC("hi", b=1)

    @assert_passes()
    def test_dataclass_slots_semantics(self):
        from dataclasses import dataclass

        @dataclass(slots=True)
        class Slotted:
            x: int

            def set_bad(self) -> None:
                self.y = 3  # E: incompatible_assignment

        Slotted.__slots__
        Slotted(1).__slots__

        @dataclass
        class NotSlotted:
            x: int

        @dataclass(slots=False)
        class ExplicitSlots:
            x: int
            __slots__ = ("x",)

            def set_bad(self) -> None:
                self.y = 3  # E: incompatible_assignment

        def check_errors() -> None:
            NotSlotted.__slots__  # E: undefined_attribute
            NotSlotted(1).__slots__  # E: undefined_attribute

            @dataclass(slots=True)
            class DataclassWithSlotsAttribute:  # E: invalid_annotation
                x: int
                __slots__ = ()

    @assert_passes()
    def test_dataclass_slots_semantics_for_intersection_instances(self):
        from pycroscope.extensions import Intersection

        class SlottedA:
            __slots__ = ("x",)

        class SlottedB:
            __slots__ = ("x",)

        def mutate(value: Intersection[SlottedA, SlottedB]) -> None:
            value.y = 3  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_init_and_match_args_after_import_failure(self):
        from dataclasses import dataclass

        @dataclass(init=False)
        class InitDisabled:
            x: int
            y: int

        def match_init_disabled(value: InitDisabled) -> None:
            match value:
                case InitDisabled(1, 2):
                    pass

        @dataclass(match_args=False)
        class NoMatchArgs:
            x: int

        def reject_positional_patterns(value: NoMatchArgs) -> None:
            match value:
                case NoMatchArgs(1):  # E: bad_match
                    pass

        def capybara() -> None:
            InitDisabled()
            InitDisabled(1, 2)  # E: incompatible_call

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_match_args_false_has_no_match_args(self):
        from dataclasses import dataclass

        @dataclass(match_args=False)
        class NoMatchArgs:
            x: int

        def capybara() -> None:
            NoMatchArgs.__match_args__  # E: undefined_attribute

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_kw_only_checks_after_import_failure(self):
        from dataclasses import KW_ONLY, dataclass, field
        from typing import Literal

        from typing_extensions import assert_type

        @dataclass
        class DC1:
            a: str
            _: KW_ONLY
            b: int = 0

        @dataclass
        class DC2:
            b: int = field(kw_only=True, default=3)
            a: str

        @dataclass(kw_only=True)
        class DC3:
            a: str = field(kw_only=False)
            b: int = 0

        @dataclass
        class DC4(DC3):
            c: float

        def capybara() -> None:
            assert_type(DC1.__match_args__, tuple[Literal["a"]])
            assert_type(DC2.__match_args__, tuple[Literal["a"]])
            assert_type(DC3.__match_args__, tuple[Literal["a"]])
            assert_type(DC4.__match_args__, tuple[Literal["a"], Literal["c"]])

            DC1("hi")
            DC1(a="hi")
            DC1(a="hi", b=1)
            DC1("hi", b=1)
            DC1("hi", 1)  # E: incompatible_call

            DC2("hi")
            DC2(a="hi")
            DC2(a="hi", b=1)
            DC2("hi", b=1)
            DC2("hi", 1)  # E: incompatible_call

            DC3("hi")
            DC3(a="hi")
            DC3(a="hi", b=1)
            DC3("hi", b=1)
            DC3("hi", 1)  # E: incompatible_call

            DC4("", 0.2, b=3)
            DC4(a="", b=3, c=0.2)

    def test_dataclass_kw_only_marker_cannot_have_default(self):
        self.assert_passes(
            """
            from dataclasses import KW_ONLY, dataclass

            @dataclass
            class BadMarker:
                x: int
                _: KW_ONLY = 0  # E: invalid_annotation
                y: int
            """,
            run_in_both_module_modes=True,
        )

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_constructor_field_metadata_after_import_failure(self):
        from dataclasses import dataclass, field

        @dataclass
        class InventoryItem:
            x = 0
            name: str
            unit_price: float
            quantity_on_hand: int = 0

        @dataclass
        class WithInitFalse:
            a: int = field(init=False)
            b: int

        def capybara() -> None:
            InventoryItem("soap", 2.3)
            InventoryItem("name")  # E: incompatible_call

            WithInitFalse(1)
            WithInitFalse(a=1, b=2)  # E: incompatible_call

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_usage_features_after_import_failure(self):
        from dataclasses import dataclass, field
        from typing import Any, Callable, ClassVar, Generic, Mapping, Protocol, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        @dataclass(order=True)
        class InventoryItem:
            name: str
            unit_price: float
            quantity_on_hand: int = 0

        class InventoryItemInitProto(Protocol):
            def __call__(
                self, name: str, unit_price: float, quantity_on_hand: int = ...
            ) -> None: ...

        item = InventoryItem("soap", 2.3)
        InventoryItem.__init__(item, "soap", 2.3)
        init_proto: InventoryItemInitProto = item.__init__
        item.__repr__
        item.__eq__
        item.__ne__
        item.__lt__
        item.__le__
        item.__gt__
        item.__ge__

        def parser(s: str) -> int:
            return int(s)

        @dataclass
        class WithCallableDefault:
            c: Callable[[str], int] = parser

        with_callable_default = WithCallableDefault()
        assert_type(with_callable_default.c, Callable[[str], int])

        @dataclass
        class WithBadFactory:
            a: int = field(default_factory=str)  # E: incompatible_assignment

        @dataclass
        class WithDictFactory:
            a: dict[tuple[str, int], str] = field(default_factory=dict)

        with_dict_factory = WithDictFactory()
        assert_type(with_dict_factory.a, dict[tuple[str, int], str])

        class DataclassProto(Protocol):
            __dataclass_fields__: ClassVar[dict[str, Any]]

        item_fields: Mapping[str, Any] = item.__dataclass_fields__
        cls_fields: Mapping[str, Any] = InventoryItem.__dataclass_fields__
        proto: DataclassProto = item

        @dataclass
        class Box(Generic[T]):
            value: T

        class StrBox(Box[str]):
            pass

        StrBox("")

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_default_order_validation(self):
        from dataclasses import InitVar, dataclass, field

        @dataclass
        class DC4:
            a: int = field(repr=False)
            b: int

        @dataclass
        class DC5:
            a: int = 0
            b: int = field(init=False)

            def method(self) -> None:
                local: int = 0
                _ = local

        def capybara() -> None:
            @dataclass  # E: invalid_annotation
            class DC1:
                a: int = 0
                b: int

            @dataclass  # E: invalid_annotation
            class DC2:
                a: int = field(default=1)
                b: int

            @dataclass  # E: invalid_annotation
            class DC3:
                a: InitVar[int] = 0
                b: int

            DC4(1, 2)

    @assert_passes()
    def test_dataclass_post_init_initvar_semantics(self):
        from dataclasses import InitVar, dataclass, field

        @dataclass
        class DC1:
            a: int
            b: int
            x: InitVar[int]
            c: int
            y: InitVar[str]

            def __post_init__(self, x: int, y: int) -> None:  # E: incompatible_override
                pass

        def f(dc1: DC1) -> None:
            dc1.x  # E: undefined_attribute
            dc1.y  # E: undefined_attribute

        @dataclass
        class DC2:
            x: InitVar[int]
            y: InitVar[str]

            def __post_init__(self, x: int) -> None:  # E: incompatible_override
                pass

        @dataclass
        class DC3:
            _name: InitVar[str] = field()
            name: str = field(init=False)

            def __post_init__(self, _name: str): ...

        @dataclass
        class DC4(DC3):
            _age: InitVar[int] = field()
            age: int = field(init=False)

            def __post_init__(self, _name: str, _age: int): ...

    @assert_passes(run_in_both_module_modes=True)
    def test_dataclass_post_init_initvar_semantics_after_import_failure(self):
        from dataclasses import InitVar, dataclass

        @dataclass
        class DC:
            x: InitVar[int]
            y: InitVar[str]

            def __post_init__(self, x: int) -> None:  # E: incompatible_override
                pass

        def f(dc: DC) -> None:
            dc.x  # E: undefined_attribute
            dc.y  # E: undefined_attribute

    @assert_passes()
    def test_attribute_checker_respects_isinstance_narrowing_for_attributes(self):
        from dataclasses import dataclass

        class Value:
            pass

        @dataclass
        class CombinedReturn:
            children: list[int]

        def f(x: Value | CombinedReturn | None) -> list[int]:
            if isinstance(x, CombinedReturn):
                return x.children
            return []

    @assert_passes(allow_import_failures=True)
    def test_generic_constructor_inference_widens_literals_in_unimportable_module(self):
        from dataclasses import dataclass
        from typing import Generic, TypeVar

        import does_not_exist  # noqa: F401
        from typing_extensions import assert_type

        from pycroscope.test_name_check_visitor import BOX_FLOAT_OR_INT_IN_TEST_INPUT
        from pycroscope.value import assert_is_value

        T = TypeVar("T")

        class Box(Generic[T]):
            def __init__(self, value: T) -> None:
                self.value = value

        assert_type(Box(1), Box[int])
        assert_is_value(Box(1.0), BOX_FLOAT_OR_INT_IN_TEST_INPUT)
        assert_type(Box(""), Box[str])
        assert_type(Box[float](1), Box[float | int])

        @dataclass
        class Data(Generic[T]):
            value: T

        assert_type(Data(1), Data[int])
