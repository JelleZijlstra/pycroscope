# static analysis: ignore
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes


class TestProtocol(TestNameCheckVisitorBase):
    @assert_passes()
    def test_generic_constructor_accepts_known_protocol_value(self):
        import logging
        import sys

        logging.StreamHandler(sys.stderr)

    @assert_passes(allow_import_failures=True)
    def test_callable_protocol_after_import_failure(self):
        from typing import Any, Callable, Protocol

        _Bad: Callable[int]  # E: invalid_annotation

        class Proto(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        def f(p: Proto) -> None:
            cb: Callable[..., None] = p
            cb()

    @assert_passes()
    def test_unknown_attribute_assignment_on_protocol_typed_callable(self):
        from typing import Callable, ParamSpec, Protocol, TypeVar, cast

        P = ParamSpec("P")
        R = TypeVar("R", covariant=True)

        class CallableWithAttr(Protocol[P, R]):
            other_attribute: int

            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

        def decorator(func: Callable[P, R]) -> CallableWithAttr[P, R]:
            converted = cast(CallableWithAttr[P, R], func)
            converted.other_attribute = 1
            converted.missing = 2  # E: undefined_attribute
            return converted

        @decorator
        def wrapped(x: int) -> str:
            return str(x)

    @assert_passes()
    def test_protocol_with_function_metadata_members_accepts_function(self):
        from typing import Any, Protocol

        from typing_extensions import assert_type

        class FunctionLike(Protocol):
            __name__: str
            __module__: str
            __qualname__: str
            __annotations__: dict[str, Any]

            def __call__(self) -> None: ...

        def f() -> None:
            pass

        wrapped: FunctionLike = f
        assert_type(f.__name__, str)
        assert_type(f.__module__, str)
        assert_type(f.__qualname__, str)
        assert_type(wrapped.__name__, str)

    @assert_passes(allow_import_failures=True)
    def test_protocol_merging_in_unimportable_module(self):
        from abc import abstractmethod
        from collections.abc import Sized
        from typing import Protocol

        class SizedAndClosable1(Sized, Protocol):
            def close(self) -> None: ...

        class SizedAndClosable2(Protocol):
            def __len__(self) -> int: ...

            def close(self) -> None: ...

        class SCConcrete1:
            def __len__(self) -> int:
                return 0

            def close(self) -> None:
                pass

        class SCConcrete2:
            def close(self) -> None:
                pass

        s1: SizedAndClosable1 = SCConcrete1()
        s2: SizedAndClosable2 = SCConcrete1()

        bad1: SizedAndClosable1 = SCConcrete2()  # E: incompatible_assignment
        bad2: SizedAndClosable2 = SCConcrete2()  # E: incompatible_assignment

        def accepts_both(p1: SizedAndClosable1, p2: SizedAndClosable2) -> None:
            merged1: SizedAndClosable2 = p1
            merged2: SizedAndClosable1 = p2
            print(merged1, merged2)

        class SizedClosableFlush(SizedAndClosable2, Protocol):
            def flush(self) -> None: ...

        class NotAProtocol(SizedAndClosable1):
            pass

        class BadProto(NotAProtocol, Protocol):  # E: invalid_base
            ...

        class FlushOnly:
            def flush(self) -> None:
                pass

        class WithFlush:
            def __len__(self) -> int:
                return 0

            def close(self) -> None:
                pass

            def flush(self) -> None:
                pass

        class AbstractSized(SizedAndClosable1):
            @abstractmethod
            def close(self) -> None:
                raise NotImplementedError

        x = AbstractSized()  # E: incompatible_call
        f1: SizedClosableFlush = WithFlush()
        f2: SizedClosableFlush = FlushOnly()  # E: incompatible_assignment

    @assert_passes(allow_import_failures=True)
    def test_explicit_protocol_abstract_instantiation_in_unimportable_module(self):
        from abc import ABC, abstractmethod
        from typing import ClassVar, Protocol

        class PColor(Protocol):
            @abstractmethod
            def draw(self) -> str: ...

        class BadColor(PColor):
            def draw(self) -> str:
                return super().draw()  # E: bad_super_call

        class RGB(Protocol):
            rgb: tuple[int, int, int]

            @abstractmethod
            def intensity(self) -> int:
                return 1

            def transparency(self) -> int: ...

        class Point(RGB):
            def __init__(self, blue: str) -> None:
                self.rgb = 0, 0, blue  # E: incompatible_assignment

        Point("")  # E: incompatible_call

        class Proto1(Protocol):
            cm1: ClassVar[int]

        class Concrete1(Proto1):
            pass

        Concrete1()  # E: incompatible_call

        class Proto5(Protocol):
            def method1(self) -> int: ...

        class Concrete5(Proto5):
            pass

        Concrete5()  # E: incompatible_call

        class Proto7(Protocol):
            @abstractmethod
            def method1(self) -> None: ...

        class Mixin7(Proto7, ABC):
            def method1(self) -> None:
                pass

        class Concrete7A(Proto7):
            pass

        class Concrete7B(Mixin7, Proto7):
            pass

        Concrete7A()  # E: incompatible_call
        Concrete7B()

    @assert_passes()
    def test_protocol_base_member_assignment_type(self):
        from typing import Protocol

        class RGB(Protocol):
            rgb: tuple[int, int, int]

        class Point(RGB):
            def __init__(self, blue: str) -> None:
                self.rgb = 0, 0, blue  # E: incompatible_assignment

    @assert_passes()
    def test_protocol_assignment_to_declared_self_attribute(self):
        from typing import Protocol

        class Proto(Protocol):
            value: int

            def __init__(self) -> None:
                self.value = 3

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_class_object_method_and_property_shapes(self):
        from typing import Any, Protocol

        class ProtoA1(Protocol):
            def method1(self, x: int) -> int: ...

        class ProtoA2(Protocol):
            def method1(_self, self: Any, x: int) -> int: ...  # E: method_first_arg

        class ConcreteA:
            def method1(self, x: int) -> int:
                return 0

        pa1: ProtoA1 = ConcreteA  # E: incompatible_assignment
        pa2: ProtoA2 = ConcreteA

        class ProtoB1(Protocol):
            @property
            def prop1(self) -> int: ...

        class ConcreteB:
            @property
            def prop1(self) -> int:
                return 0

        pb1: ProtoB1 = ConcreteB  # E: incompatible_assignment

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_class_object_classvar_members(self):
        from typing import ClassVar, Protocol

        class ProtoC1(Protocol):
            attr1: ClassVar[int]

        class ProtoC2(Protocol):
            attr1: int

        class ConcreteC1:
            attr1: ClassVar[int] = 1

        class ConcreteC2:
            attr1: int = 1

        class CMeta(type):
            pass

        class ConcreteC3(metaclass=CMeta):
            pass

        ConcreteC3.attr1 = 1

        pc1: ProtoC1 = ConcreteC1  # E: incompatible_assignment
        pc2: ProtoC2 = ConcreteC1
        pc3: ProtoC1 = ConcreteC2  # E: incompatible_assignment
        pc4: ProtoC2 = ConcreteC2  # E: incompatible_assignment
        pc5: ProtoC1 = ConcreteC3  # E: incompatible_assignment
        pc6: ProtoC2 = ConcreteC3  # E: incompatible_assignment

    @assert_passes()
    def test_protocol_class_object_call_member(self):
        from typing import Protocol

        class Concrete:
            def __init__(self, x: int) -> None:
                self.x = x

        class Factory(Protocol):
            def __call__(self, x: int) -> Concrete: ...

        factory: Factory = Concrete
        created: Concrete = factory(1)

    @assert_passes()
    def test_callable_assignment_from_protocol_instance(self):
        from collections.abc import Callable
        from typing import Protocol

        class GoodCallable(Protocol):
            def __call__(self, x: int) -> str: ...

        class BadCallable(Protocol):
            @property
            def __call__(self) -> int: ...

        def wants_callable(fn: Callable[[int], str]) -> None:
            pass

        def capybara(good: GoodCallable, bad: BadCallable) -> None:
            wants_callable(good)
            wants_callable(bad)  # E: incompatible_argument

    @assert_passes()
    def test_collections_callable_assignment_from_protocol_instance(self):
        from collections.abc import Callable
        from typing import Protocol

        class GoodCallable(Protocol):
            def __call__(self, x: int) -> str: ...

        class NarrowCallable(Protocol):
            def __call__(self, x: int, y: int) -> str: ...

        def capybara(good: GoodCallable, narrow: NarrowCallable) -> None:
            generic_cb: Callable = good
            specific_cb: Callable[[int], str] = good
            bad_cb: Callable[[int], str] = narrow  # E: incompatible_assignment
            print(generic_cb, specific_cb, bad_cb)

    @assert_passes()
    def test_protocol_class_object_metaclass_members(self):
        from typing import Protocol

        class Meta(type):
            answer: int = 1

        class WantsAnswer(Protocol):
            answer: int

        class Concrete(metaclass=Meta):
            pass

        good: WantsAnswer = Concrete
        print(good)

    @assert_passes()
    def test_type_protocol_constructor_call_allows_concrete_implementers(self):
        from typing import Protocol, cast

        class Proto(Protocol):
            def meth(self) -> int: ...

        class Concrete:
            def meth(self) -> int:
                return 1

        def call_it(cls: type[Proto]) -> int:
            return cls().meth()

        call_it(cast(type[Proto], Concrete))
        impl = cast(type[Proto], Concrete)
        impl().meth()

    @assert_passes(run_in_both_module_modes=True)
    def test_specialized_protocol_base_preserves_member_types(self):
        from typing import Protocol, TypeVar

        from typing_extensions import assert_type

        T_co = TypeVar("T_co", covariant=True)

        class Base(Protocol[T_co]):
            def meth(self) -> T_co: ...

        class Child(Base[int]):
            pass

        def capybara(value: Child) -> None:
            assert_type(value.meth(), int)

    @assert_passes()
    def test_protocol_override_keeps_compatible_self_type(self):
        from abc import abstractmethod
        from collections.abc import Sized
        from typing import Protocol

        class SizedAndClosable(Sized, Protocol):
            def close(self) -> None: ...

        class AbstractSized(SizedAndClosable):
            @abstractmethod
            def close(self) -> None:
                raise NotImplementedError

    @assert_passes()
    def test_protocol_receiver_assignment_with_nonstandard_receiver_name(self):
        from typing import Protocol

        class Proto(Protocol):
            allowed: int

            def assign(this) -> None:  # E: method_first_arg
                this.disallowed = 1  # E: invalid_protocol
                this.allowed = 1

    @assert_passes(run_in_both_module_modes=True)
    def test_protocol_staticmethod_with_receiver_param_is_incompatible(self):
        from typing import Protocol

        class Proto(Protocol):
            def method1(self, a: int, b: int) -> float: ...

        class Good:
            @staticmethod
            def method1(a: int, b: int) -> float:
                return 0

        class Bad:
            @staticmethod
            def method1(self, a: int, b: int) -> float:
                return 0

        ok: Proto = Good()
        bad: Proto = Bad()  # E: incompatible_assignment

    @assert_passes()
    def test_unannotated_protocol_classvar_override(self):
        from typing import ClassVar, Protocol

        class Proto(Protocol):
            z: ClassVar[int]

        class ProtoImpl(Proto):
            z = 0

    @assert_passes()
    def test_protocol_instantiation_is_rejected(self):
        from typing import Protocol

        class Proto(Protocol):
            def meth(self) -> int: ...

        def capybara() -> None:
            Proto()  # E: incompatible_call
