# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .tests import make_simple_sequence
from .value import (
    AnnotatedValue,
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    TypedValue,
)


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing import Generic, List, TypeVar

        T = TypeVar("T")

        def id(obj: T) -> T:
            return obj

        def get_one(obj: List[T]) -> T:
            for elt in obj:
                return elt
            assert False

        class GenCls(Generic[T]):
            def get_one(self: "GenCls[T]") -> T:
                raise NotImplementedError

            def get_another(self) -> T:
                raise NotImplementedError

        def capybara(x: str, xs: List[int], gen: GenCls[int]) -> None:
            assert_is_value(id(3), KnownValue(3))
            assert_is_value(id(x), TypedValue(str))
            assert_is_value(get_one(xs), TypedValue(int))
            assert_is_value(get_one([int(3)]), TypedValue(int))
            # This one doesn't work yet because we don't know how to go from
            # KnownValue([3]) to a GenericValue of some sort.
            # assert_is_value(get_one([3]), KnownValue(3))

            assert_is_value(gen.get_one(), TypedValue(int))
            assert_is_value(gen.get_another(), TypedValue(int))

    @assert_passes()
    def test_identity(self):
        from typing import TypeVar

        T = TypeVar("T")

        def id(obj: T) -> T:
            return obj

        def capybara(unannotated) -> None:
            assert_is_value(id(unannotated), AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_union_math(self):
        from typing import Optional, TypeVar

        T = TypeVar("T")

        def assert_not_none(arg: Optional[T]) -> T:
            assert arg is not None
            return arg

        def capybara(x: Optional[int]):
            assert_is_value(x, MultiValuedValue([KnownValue(None), TypedValue(int)]))
            assert_is_value(assert_not_none(x), TypedValue(int))

    @assert_passes()
    def test_only_T(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def add_one(self, obj: T) -> None:
                pass

        def capybara(x: Capybara[int]) -> None:
            x.add_one("x")  # E: incompatible_argument

    @assert_passes()
    def test_multi_typevar(self):
        from typing import Optional, TypeVar

        T = TypeVar("T")

        # inspired by tempfile.mktemp
        def mktemp(prefix: Optional[T] = None, suffix: Optional[T] = None) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_is_value(mktemp(), AnyValue(AnySource.generic_argument))
            assert_is_value(mktemp(prefix="p"), KnownValue("p"))
            assert_is_value(mktemp(suffix="s"), KnownValue("s"))
            assert_is_value(mktemp("p", "s"), KnownValue("p") | KnownValue("s"))

    @assert_passes()
    def test_generic_constructor(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            x: T

            def __init__(self, x: T) -> None:
                self.x = x

        def capybara(i: int) -> None:
            assert_is_value(Capybara(i).x, TypedValue(int))

    @assert_passes()
    def test_generic_base(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[int]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)

    @assert_passes()
    def test_wrong_generic_base(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Base(Generic[T]):
            pass

        class Derived(Base[int]):
            pass

        def take_base(b: Base[str]) -> None:
            pass

        def capybara(c: Derived):
            take_base(c)  # E: incompatible_argument

    @assert_passes()
    def test_variance_in_assignability(self):
        from typing import Generic, TypeVar

        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)
        T_inv = TypeVar("T_inv")

        class Co(Generic[T_co]):
            pass

        class Contra(Generic[T_contra]):
            pass

        class Inv(Generic[T_inv]):
            pass

        def capybara(
            co_int: Co[int],
            co_object: Co[object],
            contra_int: Contra[int],
            contra_object: Contra[object],
            inv_int: Inv[int],
            inv_object: Inv[object],
        ) -> None:
            co_assign_ok: Co[object] = co_int
            print(co_assign_ok, co_object)
            co_assign_bad: Co[int] = co_object  # E: incompatible_assignment
            print(co_assign_bad)

            contra_assign_ok: Contra[int] = contra_object
            print(contra_assign_ok, contra_int)
            contra_assign_bad: Contra[object] = contra_int  # E: incompatible_assignment
            print(contra_assign_bad)

            inv_assign_bad_1: Inv[object] = inv_int  # E: incompatible_assignment
            inv_assign_bad_2: Inv[int] = inv_object  # E: incompatible_assignment
            print(inv_assign_bad_1, inv_assign_bad_2)

    @assert_passes()
    def test_variance_in_class_bases_and_aliases(self):
        from typing import Generic, TypeAlias, TypeVar

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class ClassA(Generic[T]):
            pass

        class ClassB(Generic[T, T_co]):
            pass

        class DeclaresCovariant(Generic[T_co]):
            pass

        alias_a_1: TypeAlias = ClassA[T_co]
        alias_a_2: TypeAlias = alias_a_1[T_co]
        alias_b = ClassB[T_co, T_contra]

        class BadA1(ClassA[T_co]):  # E: invalid_annotation
            ...

        class BadA2(alias_a_1[T_co]):  # E: invalid_annotation
            ...

        class BadA3(alias_a_2[T_co]):  # E: invalid_annotation
            ...

        class BadB(alias_b[T_contra, T_co]):  # E: invalid_annotation
            ...

    @assert_passes()
    def test_protocol_variance_mismatch(self):
        from typing import Protocol, TypeVar

        T = TypeVar("T")

        class ReturnsT(Protocol[T]):  # E: invalid_annotation
            def get(self) -> T: ...

        class TakesT(Protocol[T]):  # E: invalid_annotation
            def put(self, value: T) -> None: ...

    @assert_passes()
    def test_protocol_explicit_variance_mismatch(self):
        from typing import Protocol, TypeVar

        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class BadCovariant(Protocol[T_co]):  # E: invalid_annotation
            def put(self, value: T_co) -> None: ...

        class BadContravariant(Protocol[T_contra]):  # E: invalid_annotation
            def get(self) -> T_contra: ...

    @assert_passes()
    def test_protocol_unused_typevar_is_covariant(self):
        from typing import Protocol, TypeVar

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)

        class InvariantByDefault(Protocol[T]):  # E: invalid_annotation
            def __init__(self, value: T) -> None: ...

        class CovariantIsOkay(Protocol[T_co]):
            def __init__(self, value: T_co) -> None: ...

    @assert_passes()
    def test_protocol_paramspec_is_ignored_for_variance_check(self):
        from typing import ParamSpec, Protocol, TypeVar

        P = ParamSpec("P")
        R = TypeVar("R", covariant=True)

        class Callback(Protocol[P, R]):
            def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

    @assert_passes()
    def test_protocol_output_only_covariant_typevar_is_valid(self):
        from typing import Protocol, TypeVar

        T_co = TypeVar("T_co", covariant=True)

        class Reader(Protocol[T_co]):
            def get(self) -> T_co: ...

    @assert_passes()
    def test_protocol_mapping_value_type_can_be_covariant(self):
        from typing import Iterable, Protocol, TypeVar

        K = TypeVar("K")
        V_co = TypeVar("V_co", covariant=True)

        class MappingLike(Protocol[K, V_co]):
            def keys(self) -> Iterable[K]: ...

            def __getitem__(self, key: K) -> V_co: ...

    @assert_passes()
    def test_protocol_staticmethod_alias_participates_in_variance(self):
        from typing import Protocol, TypeVar

        my_staticmethod = staticmethod
        T = TypeVar("T")
        T_contra = TypeVar("T_contra", contravariant=True)

        class BadAliasStaticMethod(Protocol[T]):  # E: invalid_annotation
            @my_staticmethod
            def put(value: T) -> None: ...

        class GoodAliasStaticMethod(Protocol[T_contra]):
            @my_staticmethod
            def put(value: T_contra) -> None: ...

    @assert_passes()
    def test_typeshed(self):
        from typing import List

        def capybara(lst: List[int]) -> None:
            lst.append("x")  # E: incompatible_argument

    @assert_passes()
    def test_generic_super(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class A(Generic[T]):
            def capybara(self) -> None:
                pass

        class B(A):
            def capybara(self) -> None:
                super().capybara()

    @assert_passes()
    def test_default(self):
        from typing import Dict, TypeVar, Union

        KT = TypeVar("KT")
        VT = TypeVar("VT")
        T = TypeVar("T")

        def dictget(d: Dict[KT, VT], key: KT, default: T = None) -> Union[VT, T]:
            try:
                return d[key]
            except KeyError:
                return default

        def capybara(d: Dict[str, str], key: str) -> None:
            assert_is_value(dictget(d, key), TypedValue(str) | KnownValue(None))
            assert_is_value(dictget(d, key, 1), TypedValue(str) | KnownValue(1))


class TestSolve(TestNameCheckVisitorBase):
    @assert_passes()
    def test_filter_like(self):
        from typing import Callable, TypeVar

        T = TypeVar("T")

        def callable(o: object) -> bool:
            return True

        def filterish(func: Callable[[T], bool], data: T) -> T:
            return data

        def capybara():
            assert_is_value(filterish(callable, 1), KnownValue(1))

    @assert_passes()
    def test_one_any(self):
        from typing import TypeVar

        T = TypeVar("T")

        def sub(x: T, y: T) -> T:
            return x

        def capybara(unannotated):
            assert_is_value(sub(1, unannotated), KnownValue(1))

    @assert_passes()
    def test_isinstance(self):
        from typing import TypeVar

        AnyStr = TypeVar("AnyStr", str, bytes)

        class StrSub(str):
            pass

        def want_str(s: StrSub) -> None:
            pass

        def take_tv(t: AnyStr) -> AnyStr:
            if isinstance(t, StrSub):
                assert_is_value(t, TypedValue(StrSub))
                return t
            else:
                return t

    @assert_passes()
    def test_tv_union(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", str, bytes)

        def take_seq(seq: AnyStr) -> AnyStr:
            return seq

        def take_union(seq: Union[bytes, str]) -> None:
            assert_is_value(take_seq(seq), TypedValue(bytes) | TypedValue(str))

    @assert_passes()
    def test_tv_union_list(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", str, bytes)

        def take_list(seq: list[AnyStr]) -> list[AnyStr]:
            return seq

        def take_union(seq: Union[list[bytes], list[str]]) -> None:
            assert_is_value(
                take_list(seq),
                GenericValue(list, [TypedValue(bytes)])
                | GenericValue(list, [TypedValue(str)]),
            )

    @assert_passes()
    def test_tv_multiple_params(self):
        from typing import TypeVar, Union

        AnyStr = TypeVar("AnyStr", str, bytes)

        def pick_first(x: AnyStr, y: AnyStr) -> AnyStr:
            return x

        def capybara(u: Union[str, bytes]) -> None:
            assert_is_value(pick_first("a", "b"), TypedValue(str))
            assert_is_value(pick_first(b"a", b"b"), TypedValue(bytes))
            pick_first(u, "b")  # E: incompatible_argument
            pick_first(u, b"b")  # E: incompatible_argument
            pick_first("a", b"b")  # E: incompatible_call
            pick_first(b"a", "b")  # E: incompatible_call

    @assert_passes()
    def test_tv_sequence(self):
        from typing import Sequence, TypeVar, Union

        AnyStr = TypeVar("AnyStr", bound=Union[str, bytes])

        def take_seq(seq: Sequence[AnyStr]) -> Sequence[AnyStr]:
            return seq

        def take_union(seq: Union[Sequence[bytes], Sequence[str]]) -> None:
            take_seq(seq)

    @assert_passes()
    def test_call_with_value_restriction(self):
        from typing import Callable, TypeVar, Union

        CallableT = TypeVar("CallableT", Callable[..., str], Callable[..., int])
        UnionT = TypeVar("UnionT", bound=Union[Callable[..., str], Callable[..., int]])

        def capybara(c: CallableT, u: UnionT) -> None:
            c(3)
            u(3)

    @assert_passes()
    def test_min_enum(self):
        import enum

        class E(enum.IntEnum):
            a = 1
            b = 2

        def capybara():
            m = min(E)
            assert_is_value(m, TypedValue(E))

    @assert_passes()
    def test_constraints(self):
        from typing import List, TypeVar

        LT = TypeVar("LT", List[int], List[str])

        def g(x: LT) -> LT:
            return x

        def pacarana() -> None:
            assert_is_value(g([]), AnyValue(AnySource.inference))
            assert_is_value(g([1]), GenericValue(list, [TypedValue(int)]))

    @assert_passes()
    def test_redundant_constraints(self):
        from typing import TypeVar

        from typing_extensions import SupportsIndex

        T = TypeVar("T", int, float, SupportsIndex)

        def f(x: T) -> T:
            return x

        def capybara(si: SupportsIndex):
            assert_is_value(f(1), TypedValue(int))
            assert_is_value(f(si), TypedValue(SupportsIndex))
            assert_is_value(f(1.0), TypedValue(float) | TypedValue(int))

    @assert_passes()
    def test_lots_of_constraints(self):
        from typing import TypeVar, Union

        from typing_extensions import SupportsIndex

        T = TypeVar(
            "T",
            Union[int, str],
            Union[int, float],
            Union[int, range],
            Union[int, bytes],
            SupportsIndex,
            Union[int, bytearray],
            Union[int, memoryview],
            Union[int, list],
            Union[int, tuple],
            Union[int, set],
            Union[int, frozenset],
            Union[int, dict],
        )

        def f(x: T) -> T:
            return x

        def capybara(si: SupportsIndex):
            assert_is_value(f(1), AnyValue(AnySource.inference))

    @assert_passes()
    def test_or_bounds(self):
        from typing import Dict, Mapping, Tuple, TypeVar, Union

        T = TypeVar("T")
        U = TypeVar("U")

        def capybara(d: Union[Dict[T, U], Mapping[U, T]]) -> Tuple[T, U]:
            raise NotImplementedError

        def caller():
            result = capybara({"x": 1})
            assert_is_value(
                result,
                make_simple_sequence(
                    tuple,
                    [
                        AnyValue(AnySource.generic_argument),
                        AnyValue(AnySource.generic_argument),
                    ],
                ),
            )


class TestAnnotated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_preserve(self):
        from typing import TypeVar

        from typing_extensions import Annotated

        T = TypeVar("T")

        def f(x: T) -> T:
            return x

        def caller(x: Annotated[int, 42]):
            assert_is_value(x, AnnotatedValue(TypedValue(int), [KnownValue(42)]))
            assert_is_value(f(x), AnnotatedValue(TypedValue(int), [KnownValue(42)]))


class TestDunder(TestNameCheckVisitorBase):
    @assert_passes()
    def test_sequence(self):
        from typing import Sequence

        from typing_extensions import assert_type

        def capybara(s: Sequence[int], t: str):
            assert_type(s[0], int)


class TestGenericClasses(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_generic(self):
        self.assert_passes("""
            from typing_extensions import assert_type

            class C[T]:
                x: T

                def __init__(self, x: T) -> None:
                    self.x = x

            def capybara(i: int):
                assert_type(C(i).x, int)
        """)

    @skip_before((3, 12))
    def test_generic_with_bound(self):
        self.assert_passes("""
            from typing_extensions import assert_type

            class C[T: int]:
                x: T

                def __init__(self, x: T) -> None:
                    self.x = x

            def capybara(i: int, s: str, b: bool):
                assert_type(C(i).x, int)
                assert_type(C(b).x, bool)
                C(s)  # E: incompatible_argument
        """)

    @assert_passes()
    def test_legacy_generic_alias_constructor_call_preserves_type_args(self):
        from typing import Generic, TypeVar

        from typing_extensions import assert_type

        T = TypeVar("T")

        class Node(Generic[T]):
            label: T

            def __init__(self, label: T | None = None) -> None:
                if label is not None:
                    self.label = label

        def capybara() -> None:
            n1 = Node[int]()
            n2 = Node[str]()
            assert_type(n1, Node[int])
            assert_type(n2, Node[str])
            assert_type(Node[int]().label, int)

            Node[int](0)
            Node[int]("")  # E: incompatible_argument
            Node[str]("")
            Node[str](0)  # E: incompatible_argument

    @skip_before((3, 12))
    def test_infer_variance_from_member_annotations(self):
        self.assert_passes("""
            from typing import Iterator

            class C[T]:
                def __iter__(self) -> Iterator[T]:
                    raise NotImplementedError

            x: C[float] = C[int]()
            y: C[int] = C[float]()  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_from_generic_base(self):
        self.assert_passes("""
            from typing import Generic, TypeVar

            T_co = TypeVar("T_co", covariant=True)
            T_contra = TypeVar("T_contra", contravariant=True)

            class ParentCo(Generic[T_co]):
                pass

            class ChildCo[T](ParentCo[T]):
                pass

            class ParentContra(Generic[T_contra]):
                pass

            class ChildContra[T](ParentContra[T]):
                pass

            a: ChildCo[int] = ChildCo[float]()  # E: incompatible_assignment
            b: ChildCo[float] = ChildCo[int]()
            c: ChildContra[int] = ChildContra[float]()
            d: ChildContra[float] = ChildContra[int]()  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_mixed_input_and_output_is_invariant(self):
        self.assert_passes("""
            class Box[T]:
                def get(self) -> T:
                    raise NotImplementedError

                def put(self, value: T) -> None:
                    raise NotImplementedError

            a: Box[int] = Box[float]()  # E: incompatible_assignment
            b: Box[float] = Box[int]()  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_frozen_and_mutable_dataclasses(self):
        self.assert_passes("""
            from dataclasses import dataclass

            @dataclass
            class MutableBox[T]:
                value: T

            @dataclass(frozen=True)
            class FrozenBox[T]:
                value: T

            m1: MutableBox[float] = MutableBox[int](1)  # E: incompatible_assignment
            m2: MutableBox[int] = MutableBox[float](1.0)  # E: incompatible_assignment

            f1: FrozenBox[float] = FrozenBox[int](1)
            f2: FrozenBox[int] = FrozenBox[float](1.0)  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_property_setter_makes_type_param_invariant(self):
        self.assert_passes("""
            class PropBox[T]:
                @property
                def value(self) -> T:
                    raise NotImplementedError

                @value.setter
                def value(self, new_value: T) -> None:
                    raise NotImplementedError

            a: PropBox[int] = PropBox[float]()  # E: incompatible_assignment
            b: PropBox[float] = PropBox[int]()  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_with_nested_callable_positions(self):
        self.assert_passes("""
            from typing import Callable

            class CallableArgBox[T]:
                def consume_callable(self, cb: Callable[[T], int]) -> None:
                    raise NotImplementedError

            class CallableReturnBox[T]:
                def produce_callable(self) -> Callable[[T], int]:
                    raise NotImplementedError

            c1: CallableArgBox[float] = CallableArgBox[int]()
            c2: CallableArgBox[int] = CallableArgBox[float]()  # E: incompatible_assignment

            r1: CallableReturnBox[int] = CallableReturnBox[float]()
            r2: CallableReturnBox[float] = CallableReturnBox[int]()  # E: incompatible_assignment
        """)

    @skip_before((3, 12))
    def test_infer_variance_ignores_bad_sequence_annotation_without_crashing(self):
        self.assert_passes("""
            class C[T]:
                def f(self, x: [T]) -> None:  # E: invalid_annotation
                    raise NotImplementedError
        """)

    @skip_before((3, 12))
    def test_reject_legacy_typevar_in_generic_class_bases(self):
        self.assert_passes(
            """
            from typing import TypeVar

            K = TypeVar("K")

            class ClassA[V](dict[K, V]):  # E: invalid_annotation
                pass
        """,
            allow_import_failures=True,
        )

    @skip_before((3, 12))
    def test_reject_legacy_typevar_in_generic_function_annotations(self):
        self.assert_passes("""
            from typing import TypeVar

            K = TypeVar("K")

            class ClassC[V]:
                def method1(self, a: V, b: K) -> V | K:
                    raise NotImplementedError

                def method2[M](self, a: M, b: K) -> M | K:  # E: invalid_annotation
                    raise NotImplementedError
        """)

    @skip_before((3, 12))
    def test_allow_outer_pep695_type_params_in_nested_generic_function(self):
        self.assert_passes("""
            class Box[T]:
                def wrap[U](self, x: T, y: U) -> tuple[T, U]:
                    return x, y
        """)


class TestIntegration(TestNameCheckVisitorBase):
    @assert_passes()
    def test_wraps(self):
        import functools
        from typing import Callable

        from typing_extensions import assert_type

        CachedCallable = Callable[[str], str]

        def cached() -> Callable[[CachedCallable], CachedCallable]:
            def decorator(func: CachedCallable) -> CachedCallable:
                @functools.wraps(func)
                def wrapper(key: str) -> str:
                    return func(key)

                return wrapper

            return decorator

        def capybara() -> None:
            @cached()
            def paca(x: str) -> str:
                return x

            assert_type(paca("hello"), str)
            paca(1)  # E: incompatible_argument

    @assert_passes()
    def test_wraps_inlined(self):
        from collections.abc import Callable
        from typing import Generic, TypeVar

        from typing_extensions import ParamSpec, assert_type

        _PWrapped = ParamSpec("_PWrapped")
        _RWrapped = TypeVar("_RWrapped")
        _PWrapper = ParamSpec("_PWrapper")
        _RWrapper = TypeVar("_RWrapper")

        class _Wrapped(Generic[_PWrapped, _RWrapped, _PWrapper, _RWrapper]):
            __wrapped__: Callable[_PWrapped, _RWrapped]

            def __call__(
                self, *args: _PWrapper.args, **kwargs: _PWrapper.kwargs
            ) -> _RWrapper:
                raise NotImplementedError

        class _Wrapper(Generic[_PWrapped, _RWrapped]):
            def __call__(
                self, f: Callable[_PWrapper, _RWrapper]
            ) -> _Wrapped[_PWrapped, _RWrapped, _PWrapper, _RWrapper]:
                raise NotImplementedError

        def wraps(
            wrapped: Callable[_PWrapped, _RWrapped],
        ) -> _Wrapper[_PWrapped, _RWrapped]:
            raise NotImplementedError

        def paca(x: str) -> str:
            return x

        def capybara() -> None:
            @wraps(paca)
            def wrapper(*args: str, **kwargs: str) -> str:
                return paca(*args, **kwargs)

            assert_type(wrapper("hello"), str)
            wrapper(1)  # TODO should fail

    @assert_passes()
    def test_independent_generics(self):
        from typing import Generic, TypeVar

        from typing_extensions import Literal, assert_type

        T = TypeVar("T")
        U = TypeVar("U")

        class B(Generic[T, U]):
            pass

        class A(Generic[T]):
            def meth(self, obj: U) -> B[T, U]:
                raise NotImplementedError

        def capybara(a: A[int]) -> None:
            assert_type(a.meth(1), B[int, Literal[1]])
