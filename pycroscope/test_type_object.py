# static analysis: ignore

from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .type_object import (
    TypeObject,
    _class_key_from_value,
    _iter_base_keys,
    _should_use_permissive_dunder_hash,
)
from .value import (
    AnySource,
    AnyValue,
    CallableValue,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    SubclassValue,
    SyntheticClassObjectValue,
    TypedValue,
    assert_is_value,
)


def test_protocol_member_str_order_is_deterministic() -> None:
    from typing_extensions import Protocol

    class HasMembers(Protocol):
        def f(self) -> int: ...

        def m(self) -> int: ...

    type_object = TypeObject(HasMembers, is_protocol=True, protocol_members={"m", "f"})
    assert str(type_object).endswith("(Protocol with members 'f', 'm')")

    synthetic_protocol = TypeObject(
        "synthetic.Protocol", is_protocol=True, protocol_members={"m", "f"}
    )
    assert (
        str(synthetic_protocol) == "synthetic.Protocol (Protocol with members 'f', 'm')"
    )


def test_class_key_from_subclass_generic_value() -> None:
    value = SubclassValue(GenericValue("mod.Base", [TypedValue(int)]))
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_union_with_consistent_key() -> None:
    value = MultiValuedValue(
        [
            SubclassValue(TypedValue("mod.Base")),
            GenericValue("mod.Base", [TypedValue(int)]),
        ]
    )
    assert _class_key_from_value(value) == "mod.Base"


def test_class_key_from_intersection_with_consistent_key() -> None:
    value = IntersectionValue(
        (TypedValue("mod.Base"), SubclassValue(TypedValue("mod.Base")))
    )
    assert _class_key_from_value(value) == "mod.Base"


def test_iter_base_keys_handles_subclass_synthetic_base() -> None:
    synthetic = SyntheticClassObjectValue(
        "Child",
        TypedValue("mod.Child"),
        base_classes=(SubclassValue(TypedValue("mod.Base")),),
    )

    class _Checker:
        def get_synthetic_class(self, typ):
            if typ == "mod.Child":
                return synthetic
            return None

        def get_generic_bases(self, typ):
            return {}

    class _Ctx:
        checker = _Checker()

    assert _iter_base_keys("mod.Child", _Ctx()) == ["mod.Base"]


def test_permissive_dunder_hash_class_object_detection() -> None:
    assert _should_use_permissive_dunder_hash(TypedValue(type))
    assert _should_use_permissive_dunder_hash(GenericValue(type, [TypedValue(int)]))
    assert not _should_use_permissive_dunder_hash(TypedValue(list))
    assert not _should_use_permissive_dunder_hash(
        MultiValuedValue([TypedValue(type), TypedValue(list)])
    )
    assert _should_use_permissive_dunder_hash(
        IntersectionValue((TypedValue(type), TypedValue(list)))
    )


class TestNumerics(TestNameCheckVisitorBase):
    @assert_passes()
    def test_float(self):
        from typing import NewType

        NT = NewType("NT", int)

        def take_float(x: float) -> None:
            pass

        class IntSubclass(int):
            pass

        def capybara(nt: NT, i: int, f: float) -> None:
            take_float(nt)
            take_float(i)
            take_float(f)
            take_float(3.0)
            take_float(3)
            take_float(1 + 1j)  # E: incompatible_argument
            take_float("string")  # E: incompatible_argument
            # bool is a subclass of int, which is treated as a subclass of float
            take_float(True)
            take_float(IntSubclass(3))

    @assert_passes()
    def test_complex(self):
        from typing import NewType

        NTI = NewType("NTI", int)
        NTF = NewType("NTF", float)

        def take_complex(c: complex) -> None:
            pass

        def capybara(nti: NTI, ntf: NTF, i: int, f: float, c: complex) -> None:
            take_complex(ntf)
            take_complex(nti)
            take_complex(i)
            take_complex(f)
            take_complex(c)
            take_complex(3.0)
            take_complex(3)
            take_complex(1 + 1j)
            take_complex("string")  # E: incompatible_argument
            take_complex(True)  # bool is an int, which is a float, which is a complex


class TestSyntheticType(TestNameCheckVisitorBase):
    @assert_passes()
    def test_overloaded_callable_protocols(self):
        from typing import Protocol

        from pycroscope.extensions import overload

        class OverloadedNarrow(Protocol):
            @overload
            def __call__(self, x: int) -> int: ...

            @overload
            def __call__(self, x: str) -> str: ...

        class FloatArg(Protocol):
            def __call__(self, x: float) -> float: ...

        class OverloadedWide(Protocol):
            @overload
            def __call__(self, x: int, y: str) -> float: ...

            @overload
            def __call__(self, x: str) -> complex: ...

        class IntStrArg(Protocol):
            def __call__(self, x: int | str, y: str = "") -> int: ...

        class StrArg(Protocol):
            def __call__(self, x: str) -> complex: ...

        def capybara(
            overloaded_narrow: OverloadedNarrow, int_str_arg: IntStrArg, str_arg: StrArg
        ) -> None:
            bad: FloatArg = overloaded_narrow  # E: incompatible_assignment
            ok: OverloadedWide = int_str_arg
            bad2: OverloadedWide = str_arg  # E: incompatible_assignment
            print(bad, ok, bad2)

    @assert_passes()
    def test_callable_protocol_instance_call_not_treated_as_instantiation(self):
        from typing import Protocol

        class IdentityFunction(Protocol):
            def __call__(self, x: int) -> int: ...

        def apply_identity(identity: IdentityFunction) -> int:
            return identity(1)

    @assert_passes()
    def test_paramspec_callable_protocol_equivalence(self):
        from typing import Callable, ParamSpec, Protocol, TypeAlias

        P = ParamSpec("P")

        class ProtocolWithP(Protocol[P]):
            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

        TypeAliasWithP: TypeAlias = Callable[P, None]

        def capybara(proto: ProtocolWithP[P], ta: TypeAliasWithP[P]) -> None:
            as_callable: TypeAliasWithP[P] = proto
            as_protocol: ProtocolWithP[P] = ta
            print(as_callable, as_protocol)

    @assert_passes(allow_import_failures=True)
    def test_callable_annotation_protocol_interop(self):
        from typing import Any, Callable, ParamSpec, Protocol, TypeVar

        T_contra = TypeVar("T_contra", contravariant=True)
        P = ParamSpec("P")

        class ProtoAnyTail(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class ProtoFixedTail(Protocol):
            def __call__(self, a: int, *args: Any, **kwargs: Any) -> None: ...

        class ProtoParamSpecAny(Protocol[P]):
            def __call__(self, a: int, *args: P.args, **kwargs: P.kwargs) -> None: ...

        class ProtoStrict(Protocol):
            def __call__(self, a: float, b: int, *, k: str, m: str) -> None: ...

        class ProtoZero(Protocol):
            def __call__(self, *args: Any, **kwargs: Any) -> None: ...

        class Proto5(Protocol[T_contra]):
            def __call__(self, *args: T_contra, **kwargs: T_contra) -> None: ...

        class Proto8(Protocol):
            def __call__(self) -> None: ...

        def capybara(
            p_any: ProtoAnyTail,
            p_ps: ProtoParamSpecAny[...],
            p5: Proto5[Any],
            p_strict: ProtoStrict,
            p8: Proto8,
            c1: Callable[..., None],
            c2: ProtoFixedTail,
        ) -> None:
            ok1: ProtoAnyTail = c2
            ok2: ProtoFixedTail = p_ps
            ok3: ProtoFixedTail = p_strict
            ok4: ProtoParamSpecAny[...] = p_any  # keep reverse direction covered
            ok5: Proto5[Any] = c1
            err1: Proto5[Any] = p8  # E: incompatible_assignment
            print(ok1, ok2, ok3, ok4, ok5, err1)

    @assert_passes()
    def test_functools(self):
        import functools
        import types

        from pycroscope.signature import ELLIPSIS_PARAM, Signature

        sig = Signature.make([ELLIPSIS_PARAM], return_annotation=TypedValue(int))

        def f() -> int:
            return 0

        def capybara():
            c = functools.singledispatch(f)
            assert_is_value(
                c, GenericValue("functools._SingleDispatchCallable", [TypedValue(int)])
            )
            assert_is_value(
                c.registry,
                GenericValue(
                    types.MappingProxyType,
                    [AnyValue(AnySource.explicit), CallableValue(sig)],
                ),
            )
            assert_is_value(c._clear_cache(), KnownValue(None))
            assert_type(c(), int)
            c.doesnt_exist  # E: undefined_attribute

    @assert_passes()
    def test_protocol(self):
        # Note that csv.writer expects this protocol:
        # class _Writer(Protocol):
        #    def write(self, s: str) -> Any: ...
        import csv
        import io

        class BadWrite:
            def write(self, s: int) -> object:
                return object()

        class GoodWrite:
            def write(self, s: str) -> object:
                return object()

        class BadArgKind:
            def write(self, *, s: str) -> object:
                return object()

        def capybara(s: str):
            writer = io.StringIO()
            # Ideally we'd test the return type but it's a private type
            # and the exact one changes between typeshed versions.
            csv.writer(writer)

            csv.writer(1)  # E: incompatible_argument
            csv.writer(s)  # E: incompatible_argument
            csv.writer(BadWrite())  # E: incompatible_argument
            csv.writer(GoodWrite())
            csv.writer(BadArgKind())  # E: incompatible_argument

    @assert_passes(allow_import_failures=True)
    def test_protocol_generic_base_after_import_failure(self):
        from typing import Hashable, Iterable, Protocol

        class P0(Protocol):
            pass

        P0()  # E: incompatible_call

        class HashableFloats(Iterable[float], Hashable, Protocol):
            pass

        def cached_func(args: HashableFloats) -> float:
            return 0.0

        cached_func((1, 2, 3))

    @assert_passes(allow_import_failures=True)
    def test_protocol_subtyping_after_import_failure(self):
        from typing import Protocol, Sequence, TypeVar

        class Proto1(Protocol):
            pass

        Proto1()  # E: incompatible_call

        class Proto2(Protocol):
            def method1(self) -> None: ...

        class Concrete2:
            def method1(self) -> None:
                pass

        def func1(p2: Proto2, c2: Concrete2) -> None:
            v1: Proto2 = c2
            v2: Concrete2 = p2  # E: incompatible_assignment
            print(v1, v2)

        class Proto3(Protocol):
            def method1(self) -> None: ...

            def method2(self) -> None: ...

        def func2(p2: Proto2, p3: Proto3) -> None:
            v1: Proto2 = p3
            v2: Proto3 = p2  # E: incompatible_assignment
            print(v1, v2)

        S = TypeVar("S")
        T = TypeVar("T")

        class Proto4(Protocol[S, T]):
            def method1(self, a: S, b: T) -> tuple[S, T]: ...

        class Proto5(Protocol[T]):
            def method1(self, a: T, b: T) -> tuple[T, T]: ...

        def func3(p4_int: Proto4[int, int], p5_int: Proto5[int]) -> None:
            v1: Proto4[int, int] = p5_int
            v2: Proto5[int] = p4_int
            v3: Proto4[int, float] = p5_int  # E: incompatible_assignment
            v4: Proto5[float] = p4_int  # E: incompatible_assignment
            print(v1, v2, v3, v4)

        S_co = TypeVar("S_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class Proto6(Protocol[S_co, T_contra]):
            def method1(self, a: T_contra) -> Sequence[S_co]: ...

        class Proto7(Protocol[S_co, T_contra]):
            def method1(self, a: T_contra) -> Sequence[S_co]: ...

        def func4(p6: Proto6[float, float]) -> None:
            v1: Proto7[object, int] = p6
            v2: Proto7[float, float] = p6
            v3: Proto7[int, float] = p6  # E: incompatible_assignment
            v4: Proto7[float, object] = p6  # E: incompatible_assignment
            print(v1, v2, v3, v4)

    @assert_passes()
    def test_protocol_shorthand_type_parameter_order(self):
        from typing import Iterator, Protocol, TypeVar

        S = TypeVar("S")
        T_co = TypeVar("T_co", covariant=True)

        class Iterable(Protocol[T_co]):
            def __iter__(self) -> Iterator[T_co]: ...

        class Proto1(Iterable[T_co], Protocol[S, T_co]):
            def method1(self, x: S) -> S: ...

        class Concrete1:
            def __iter__(self) -> Iterator[int]:
                return iter((1, 2, 3))

            def method1(self, x: str) -> str:
                return x

        ok: Proto1[str, int] = Concrete1()
        bad: Proto1[int, str] = Concrete1()  # E: incompatible_assignment
        print(ok, bad)

    @assert_passes()
    def test_protocol_member_constraints_must_be_satisfiable(self):
        from typing import Callable, Protocol, Self, TypeVar

        T = TypeVar("T")

        class HasPropertyProto(Protocol):
            @property
            def f(self: T) -> T: ...

            def m(self, item: T, callback: Callable[[T], str]) -> str: ...

        class ConcreteHasProperty1:
            @property
            def f(self: T) -> T:
                return self

            def m(self, item: T, callback: Callable[[T], str]) -> str:
                return ""

        class ConcreteHasProperty2:
            @property
            def f(self) -> Self:
                return self

            def m(self, item: int, callback: Callable[[int], str]) -> str:
                return ""

        class ConcreteHasProperty4:
            @property
            def f(self) -> Self:
                return self

            def m(self, item: str, callback: Callable[[int], str]) -> str:
                return ""

        hp1: HasPropertyProto = ConcreteHasProperty1()
        hp2: HasPropertyProto = ConcreteHasProperty2()  # E: incompatible_assignment
        hp4: HasPropertyProto = ConcreteHasProperty4()  # E: incompatible_assignment
        print(hp1, hp2, hp4)

    @assert_passes(allow_import_failures=True)
    def test_protocol_member_semantics_after_import_failure(self):
        from dataclasses import dataclass
        from typing import ClassVar, NamedTuple, Protocol, Sequence

        class WantsClassVar(Protocol):
            val: ClassVar[Sequence[int]]

        class HasInstanceVal:
            val: Sequence[int] = [0]

        bad_classvar: WantsClassVar = HasInstanceVal()  # E: incompatible_assignment

        class WantsDataMember(Protocol):
            val: Sequence[int]

        class HasClassVar:
            val: ClassVar[Sequence[int]] = [0]

        class HasReadOnlyProperty:
            @property
            def val(self) -> Sequence[int]:
                return [0]

        class HasMutableList:
            val: list[int] = [0]

        bad_data1: WantsDataMember = HasClassVar()  # E: incompatible_assignment
        bad_data2: WantsDataMember = HasReadOnlyProperty()  # E: incompatible_assignment
        bad_data3: WantsDataMember = HasMutableList()  # E: incompatible_assignment

        class WantsReadOnlyProperty(Protocol):
            @property
            def val(self) -> Sequence[float]: ...

        class PlainAttrForReadOnly:
            val: Sequence[float] = [0]

        ok_read_only: WantsReadOnlyProperty = PlainAttrForReadOnly()

        class WantsSettableProperty(Protocol):
            @property
            def val(self) -> Sequence[float]: ...

            @val.setter
            def val(self, value: Sequence[float]) -> None: ...

        class SettablePropertyImpl:
            @property
            def val(self) -> Sequence[float]:
                return [0]

            @val.setter
            def val(self, value: Sequence[float]) -> None:
                pass

        class PlainAttrImpl:
            val: Sequence[float] = [0]

        class ReadOnlyPropertyImpl:
            @property
            def val(self) -> Sequence[float]:
                return [0]

        class NamedTupleImpl(NamedTuple):
            val: Sequence[float] = [0]

        @dataclass(frozen=True)
        class FrozenDataclassImpl:
            val: Sequence[float] = [0]

        ok_settable1: WantsSettableProperty = SettablePropertyImpl()
        ok_settable2: WantsSettableProperty = PlainAttrImpl()
        bad_settable1: WantsSettableProperty = (  # E: incompatible_assignment
            ReadOnlyPropertyImpl()
        )
        bad_settable2: WantsSettableProperty = (  # E: incompatible_assignment
            NamedTupleImpl()
        )
        bad_settable3: WantsSettableProperty = (  # E: incompatible_assignment
            FrozenDataclassImpl()
        )
        print(
            bad_classvar,
            bad_data1,
            bad_data2,
            bad_data3,
            ok_read_only,
            ok_settable1,
            ok_settable2,
            bad_settable1,
            bad_settable2,
            bad_settable3,
        )

    @assert_passes()
    def test_custom_subclasscheck(self):
        class _ThriftEnumMeta(type):
            def __subclasscheck__(self, subclass):
                return hasattr(subclass, "_VALUES_TO_NAMES")

        class ThriftEnum(metaclass=_ThriftEnumMeta):
            pass

        class IsOne:
            _VALUES_TO_NAMES = {}

        class IsntOne:
            _NAMES_TO_VALUES = {}

        def want_enum(te: ThriftEnum) -> None:
            pass

        def capybara(good_instance: IsOne, bad_instance: IsntOne, te: ThriftEnum):
            want_enum(good_instance)
            want_enum(bad_instance)  # E: incompatible_argument
            want_enum(IsOne())
            want_enum(IsntOne())  # E: incompatible_argument
            want_enum(te)

    @assert_passes()
    def test_generic_stubonly(self):
        import pkgutil

        # pkgutil.read_code requires SupportsRead[bytes]

        class Good:
            def read(self, length: int = 0) -> bytes:
                return b""

        class Bad:
            def read(self, length: int = 0) -> str:
                return ""

        def capybara():
            pkgutil.read_code(1)  # E: incompatible_argument
            pkgutil.read_code(Good())
            pkgutil.read_code(Bad())  # E: incompatible_argument

    @assert_passes()
    def test_protocol_inheritance(self):
        import operator

        # operator.getitem requires SupportsGetItem[K, V]

        class Good:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: str) -> str:
                raise KeyError(k)

        class Bad:
            def __contains__(self, obj: object) -> bool:
                return False

            def __getitem__(self, k: bytes) -> str:
                raise KeyError(k)

        def capybara():
            operator.getitem(Good(), "hello")
            operator.getitem(Bad(), "hello")  # E: incompatible_call
            operator.getitem(1, "hello")  # E: incompatible_argument

    @assert_passes()
    def test_iterable(self):
        from typing import Iterable, Iterator

        class Bad:
            def __iter__(self, some, random, args):
                pass

        class Good:
            def __iter__(self) -> Iterator[int]:
                raise NotImplementedError

        class BadType:
            def __iter__(self) -> Iterator[str]:
                raise NotImplementedError

        def want_iter_int(f: Iterable[int]) -> None:
            pass

        def capybara():
            want_iter_int(Bad())  # E: incompatible_argument
            want_iter_int(Good())
            want_iter_int(BadType())  # E: incompatible_argument

    @assert_passes()
    def test_self_iterator(self):
        from typing import Iterator

        class MyIter:
            def __iter__(self) -> "MyIter":
                return self

            def __next__(self) -> int:
                return 42

        def want_iter(it: Iterator[int]):
            pass

        def capybara():
            want_iter(MyIter())

    @assert_passes()
    def test_container(self):
        from typing import Any, Container

        class Good:
            def __contains__(self, whatever: object) -> bool:
                return False

        class Bad:
            def __contains__(self, too, many, arguments) -> bool:
                return True

        def want_container(c: Container[Any]) -> None:
            pass

        def capybara() -> None:
            want_container(Bad())  # E: incompatible_argument
            want_container(Good())
            want_container([1])
            want_container(1)  # E: incompatible_argument

    @assert_passes()
    def test_runtime_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            a: int

            def b(self) -> int:
                raise NotImplementedError

        class Q(P, Protocol):
            c: str

        class NotAProtocol(P):
            c: str

        def want_p(x: P):
            print(x.a + x.b())

        def want_q(q: Q):
            pass

        def want_not_a_proto(nap: NotAProtocol):
            pass

        class GoodP:
            a: int

            def b(self) -> int:
                return 3

        class BadP:
            def a(self) -> int:
                return 5

            def b(self) -> int:
                return 4

        class GoodQ(GoodP):
            c: str

        class BadQ(GoodP):
            c: float

        def capybara():
            want_p(GoodP())
            want_p(BadP())  # E: incompatible_argument
            want_q(GoodQ())
            want_q(BadQ())  # E: incompatible_argument
            want_not_a_proto(GoodQ())  # E: incompatible_argument

    @assert_passes()
    def test_callable_protocol(self):
        from typing_extensions import Protocol

        class P(Protocol):
            def __call__(self, x: int) -> str:
                return str(x)

        def want_p(p: P) -> str:
            return p(1)

        def good(x: int) -> str:
            return "hello"

        def bad(x: str) -> str:
            return x

        def capybara():
            want_p(good)
            want_p(bad)  # E: incompatible_argument


class TestHashable(TestNameCheckVisitorBase):
    @assert_passes()
    def test_type(self):
        from typing import Hashable, Type

        from typing_extensions import Protocol

        class MyHashable(Protocol):
            def __hash__(self) -> int:
                raise NotImplementedError

        def want_hash(h: Hashable):
            pass

        def want_myhash(h: MyHashable):
            pass

        class A:
            pass

        class B:
            def __hash__(self) -> int:
                return 42

        def capybara(t1: Type[int], t2: type, x: list[int]):
            want_hash(t1)
            want_hash(t2)
            want_hash(int)
            want_hash(A)
            want_hash(B)

            {t1: 0}
            {t2: 0}
            {int: 0}
            {A: 0}

            want_hash(x)  # E: incompatible_argument
            want_hash([x])  # E: incompatible_argument
            want_hash([])  # E: incompatible_argument
            want_myhash(x)  # E: incompatible_argument
            want_myhash([x])  # E: incompatible_argument
            want_myhash([])  # E: incompatible_argument


class TestIO(TestNameCheckVisitorBase):
    @assert_passes()
    def test_text(self):
        import io
        from typing import TextIO

        def want_io(x: TextIO):
            x.write("hello")

        def capybara():
            with open("x") as f:
                assert_type(f, io.TextIOWrapper)
                want_io(f)

    @assert_passes()
    def test_binary(self):
        import io
        from typing import BinaryIO

        def want_io(x: BinaryIO):
            x.write(b"hello")

        def capybara():
            with open("x", "rb") as f:
                assert_type(f, io.BufferedReader)
                want_io(f)

        def pacarana():
            with open("x", "w+b") as f:
                assert_type(f, io.BufferedRandom)
                want_io(f)
