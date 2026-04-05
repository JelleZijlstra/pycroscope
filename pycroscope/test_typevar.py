# static analysis: ignore
from .checker import Checker
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_before
from .tests import make_simple_sequence
from .value import (
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    TypedValue,
    assert_is_value,
)


def test_runtime_constructor_instance_value_substitutes_earlier_defaults() -> None:
    from typing import Generic

    from typing_extensions import TypeVar

    checker = Checker()
    T = TypeVar("T", default=int)
    U = TypeVar("U", default=list[T])

    class Box(Generic[T, U]):
        pass

    assert checker._runtime_constructor_instance_value(Box) == GenericValue(
        Box, [TypedValue(int), GenericValue(list, [TypedValue(int)])]
    )


class TestTypeVar(TestNameCheckVisitorBase):
    @assert_passes()
    def test_simple(self):
        from typing import Generic, List, TypeVar

        from typing_extensions import Literal

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
            assert_type(id(3), Literal[3])
            assert_type(id(x), str)
            assert_type(get_one(xs), int)
            assert_type(get_one([int(3)]), int)
            # This one doesn't work yet because we don't know how to go from
            # KnownValue([3]) to a GenericValue of some sort, so we can't yet
            # prove a Literal return here.
            # assert_type(get_one([3]), Literal[3])

            assert_type(gen.get_one(), int)
            assert_type(gen.get_another(), int)

    @assert_passes()
    def test_identity(self):
        from typing import TypeVar

        T = TypeVar("T")

        def id(obj: T) -> T:
            return obj

        def capybara(unannotated) -> None:
            assert_is_value(id(unannotated), AnyValue(AnySource.unannotated))
            assert_is_value(id(1), KnownValue(1))

    @assert_passes()
    def test_union_math(self):
        from typing import Optional, TypeVar

        T = TypeVar("T")

        def assert_not_none(arg: Optional[T]) -> T:
            assert arg is not None
            return arg

        def capybara(x: Optional[int]):
            assert_type(x, int | None)
            assert_type(assert_not_none(x), int)

    @assert_passes()
    def test_only_T(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            def add_one(self, obj: T) -> None:
                pass

        def capybara(x: Capybara[int]) -> None:
            x.add_one("x")  # E: incompatible_argument

    def test_generic_too_few_args_in_annotation_without_future(self):
        self.assert_passes(
            """
            from typing_extensions import Generic, TypeVar

            T = TypeVar("T")
            U = TypeVar("U")

            class Capybara(Generic[T, U]):
                pass

            def outer() -> None:
                def capybara(
                    w: Capybara,  # E: missing_generic_parameters
                    x: Capybara[int],  # E: invalid_specialization
                    y: Capybara[int, int],
                ) -> None:
                    pass
            """,
            run_in_both_module_modes=True,
        )

    def test_generic_too_few_args_in_annotation_with_future(self):
        self.assert_passes("""
            from __future__ import annotations

            from typing_extensions import Generic, TypeVar

            T = TypeVar("T")
            U = TypeVar("U")

            class Capybara(Generic[T, U]):
                pass

            def capybara(
                w: Capybara,  # E: missing_generic_parameters
                x: Capybara[int],  # E: invalid_specialization
                y: Capybara[int, int],
            ) -> None:
                pass
            """)

    @assert_passes()
    def test_multi_typevar(self):
        from typing import Optional, TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        # inspired by tempfile.mktemp
        def mktemp(prefix: Optional[T] = None, suffix: Optional[T] = None) -> T:
            raise NotImplementedError

        def capybara() -> None:
            assert_is_value(mktemp(), AnyValue(AnySource.generic_argument))
            assert_type(mktemp(prefix="p"), Literal["p"])
            assert_type(mktemp(suffix="s"), Literal["s"])
            assert_type(mktemp("p", "s"), Literal["p", "s"])

    @assert_passes()
    def test_generic_constructor(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Capybara(Generic[T]):
            x: T

            def __init__(self, x: T) -> None:
                self.x = x

        def capybara(i: int) -> None:
            assert_type(Capybara(i).x, int)

    @assert_passes()
    def test_complex_constructor(self):
        from typing import Callable, TypeVar

        T = TypeVar("T")
        S = TypeVar("S")

        def model_field(*, converter: Callable[[S], T], default: S) -> T:
            raise NotImplementedError

        def capybara() -> None:
            model_field(converter=dict, default=())

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

        class BadA1(ClassA[T_co]):  # E: invalid_base
            ...

        class BadA2(alias_a_1[T_co]):  # E: invalid_base
            ...

        class BadA3(alias_a_2[T_co]):  # E: invalid_base
            ...

        class BadB(alias_b[T_contra, T_co]):  # E: invalid_base
            ...

    @assert_passes(run_in_both_module_modes=True)
    def test_variance_in_class_bases_after_import_failure(self):
        from typing import Generic, TypeVar

        try:
            TypeVar("X1", covariant=True, contravariant=True)  # E: incompatible_call
        except (TypeError, ValueError):
            pass

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class Co(Generic[T_co]): ...

        class Contra(Generic[T_contra]): ...

        class Inv(Generic[T]): ...

        class GoodCo(Co[T_co]): ...

        class GoodContra(Contra[T_contra]): ...

        class BadCo(Co[T_contra]):  # E: invalid_base
            ...

        class BadContra(Contra[T_co]):  # E: invalid_base
            ...

        class BadInv(Inv[T_co]):  # E: invalid_base
            ...

    @assert_passes(run_in_both_module_modes=True)
    def test_typevar_scoping_in_annotations(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")
        S = TypeVar("S")

        def func(x: T) -> list[T]:
            ok: list[T] = []
            _bad: list[S] = []  # E: invalid_annotation
            return ok

        class Box(Generic[T]):
            attr: list[S] = []  # E: invalid_annotation

        _global_value: T  # E: invalid_annotation
        global_list: list[T] = []  # E: invalid_annotation
        list[T]()  # E: invalid_annotation

    @assert_passes(run_in_both_module_modes=True)
    def test_nested_classes_do_not_capture_outer_type_params(self):
        from typing import Generic, Iterable, TypeVar

        T = TypeVar("T")
        S = TypeVar("S")

        def func(x: T) -> None:
            class Bad(Generic[T]):  # E: invalid_annotation
                ...

        class Outer(Generic[T]):
            class Bad(Iterable[T]):  # E: invalid_annotation
                ...

            class AlsoBad:
                attr: list[T]  # E: invalid_annotation

            class Inner(Iterable[S]): ...

            ok: Inner[T]

    @assert_passes(run_in_both_module_modes=True)
    def test_class_body_type_alias_cannot_capture_class_type_params(self):
        from typing import Generic, TypeAlias, TypeVar

        T = TypeVar("T")

        Alias: TypeAlias = list[T]

        class Outer(Generic[T]):
            BadAlias: TypeAlias = list[T]  # E: invalid_annotation

    @assert_passes(run_in_both_module_modes=True)
    def test_nested_alias_variance_after_import_failure(self):
        from typing import Generic, TypeVar

        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class Co(Generic[T_co]): ...

        class Contra(Generic[T_contra]): ...

        Co_TA = Co[T_co]
        Contra_TA = Contra[T_contra]

        class Good1(Contra_TA[Co_TA[T_contra]]): ...

        class Good2(Contra_TA[Contra_TA[T_co]]): ...

        class Bad1(Contra_TA[Co_TA[Contra_TA[T_contra]]]):  # E: invalid_base
            ...

        class Bad2(Contra_TA[Contra_TA[Contra_TA[T_co]]]):  # E: invalid_base
            ...

    @assert_passes()
    def test_protocol_variance_mismatch(self):
        from typing import Protocol, TypeVar

        T = TypeVar("T")

        class ReturnsT(Protocol[T]):  # E: invalid_protocol
            def get(self) -> T: ...

        class TakesT(Protocol[T]):  # E: invalid_protocol
            def put(self, value: T) -> None: ...

    @assert_passes()
    def test_protocol_explicit_variance_mismatch(self):
        from typing import Protocol, TypeVar

        T_co = TypeVar("T_co", covariant=True)
        T_contra = TypeVar("T_contra", contravariant=True)

        class BadCovariant(Protocol[T_co]):  # E: invalid_protocol
            def put(self, value: T_co) -> None: ...

        class BadContravariant(Protocol[T_contra]):  # E: invalid_protocol
            def get(self) -> T_contra: ...

    @assert_passes()
    def test_protocol_unused_typevar_is_covariant(self):
        from typing import Protocol, TypeVar

        T = TypeVar("T")
        T_co = TypeVar("T_co", covariant=True)

        class InvariantByDefault(Protocol[T]):  # E: invalid_protocol
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

        class BadAliasStaticMethod(Protocol[T]):  # E: invalid_protocol
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

        from typing_extensions import Literal

        KT = TypeVar("KT")
        VT = TypeVar("VT")
        T = TypeVar("T")

        def dictget(d: Dict[KT, VT], key: KT, default: T = None) -> Union[VT, T]:
            try:
                return d[key]
            except KeyError:
                return default

        def capybara(d: Dict[str, str], key: str) -> None:
            assert_type(dictget(d, key), str | None)
            assert_type(dictget(d, key, 1), str | Literal[1])

    @assert_passes()
    def test_constrained_typevar_assignability(self):
        from typing import TypeVar

        T = TypeVar("T", int, str)

        def f(x: int) -> None:
            pass

        def capybara(x: T) -> T:
            f(x)  # E: incompatible_argument
            return x

    @assert_passes()
    def test_typevar_default_must_match_bound_and_constraints(self):
        from typing_extensions import TypeVar

        TypeVar("GoodBound", bound=float, default=int)
        TypeVar("BadBound", bound=str, default=int)  # E: incompatible_call
        TypeVar("BadConstraint", float, str, default=int)  # E: incompatible_call
        Base = TypeVar("Base", int, str)
        TypeVar("GoodConstraint", int, str, bool, default=Base)
        TypeVar(
            "BadConstraintTypeVar", bool, complex, default=Base  # E: incompatible_call
        )

    @assert_passes()
    def test_typevar_default_is_not_used_as_fallback(self):
        from typing_extensions import TypeVar, assert_type

        T = TypeVar("T", default=int)

        def f(x: type[T]) -> None:
            assert_type(x(), int)  # E: inference_failure

    # Not sure why this fails on 3.10, TODO investigate
    @skip_before((3, 11))
    def test_type_parameter_defaults_on_generic_classes(self):
        self.assert_passes("""
            from typing import Generic

            from typing_extensions import ParamSpec, TypeVarTuple, Unpack, assert_type

            DefaultP = ParamSpec("DefaultP", default=[str, int])
            DefaultTs = TypeVarTuple("DefaultTs", default=Unpack[tuple[str, int]])

            class ClassParamSpec(Generic[DefaultP]): ...

            class ClassTypeVarTuple(Generic[*DefaultTs]): ...

            def capybara() -> None:
                assert_type(ClassParamSpec(), ClassParamSpec[str, int])
                assert_type(ClassTypeVarTuple(), ClassTypeVarTuple[str, int])
                assert_type(ClassTypeVarTuple[int, bool](), ClassTypeVarTuple[int, bool])
        """)

    @skip_before((3, 11))
    def test_typevartuple_unpack_assignability(self):
        self.assert_passes("""
            from typing import Any, Generic, NewType, TypeVarTuple

            Shape = TypeVarTuple("Shape")

            Height = NewType("Height", int)
            Width = NewType("Width", int)
            Batch = NewType("Batch", int)
            Channels = NewType("Channels", int)

            class Array(Generic[*Shape]): ...

            def process_batch_channels(x: Array[Batch, *tuple[Any, ...], Channels]) -> None:
                pass

            def expect_variadic_array(x: Array[Batch, *Shape]) -> None:
                pass

            def expect_precise_array(x: Array[Batch, Height, Width, Channels]) -> None:
                pass

            def capybara(
                x: Array[Batch, Height, Width, Channels],
                y: Array[Batch, Channels],
                z: Array[Batch],
                any_array: Array[*tuple[Any, ...]],
            ) -> None:
                process_batch_channels(x)
                process_batch_channels(y)
                process_batch_channels(z)  # E: incompatible_argument
                expect_variadic_array(any_array)
                expect_precise_array(any_array)
        """)

    @skip_before((3, 11))
    def test_typevartuple_conformance_repeated_positions(self):
        self.assert_passes("""
            from typing import Generic, NewType, TypeVarTuple

            Ts = TypeVarTuple("Ts")

            class Array(Generic[*Ts]): ...

            Height = NewType("Height", int)
            Width = NewType("Width", int)

            def func2(arg1: tuple[*Ts], arg2: tuple[*Ts]) -> tuple[*Ts]:
                raise NotImplementedError

            def multiply(x: Array[*Ts], y: Array[*Ts]) -> Array[*Ts]:
                raise NotImplementedError

            def capybara(
                x: Array[Height], y: Array[Width], z: Array[Height, Width]
            ) -> None:
                func2((0,), (1,))
                func2((0,), (0.0,))
                func2((0.0,), (0,))
                func2((0,), ("0",))
                func2((0, 0), (0,))  # E: incompatible_call

                multiply(x, x)
                multiply(x, y)  # E: incompatible_call
                multiply(x, z)  # E: incompatible_call
        """)

    @skip_before((3, 11))
    @assert_passes(allow_import_failures=True)
    def test_class_type_param_default_ordering_rules(self):
        from typing import Generic

        from typing_extensions import ParamSpec, TypeVar, TypeVarTuple, Unpack

        DefaultT = TypeVar("DefaultT", default=int)
        T = TypeVar("T")
        Ts = TypeVarTuple("Ts")
        DefaultAfterVariadic = TypeVar("DefaultAfterVariadic", default=bool)
        P = ParamSpec("P", default=[str])

        class BadOrder(Generic[DefaultT, T]): ...  # E: invalid_type_parameter

        class GoodAfterVariadic(Generic[Unpack[Ts], P]): ...

    @skip_before((3, 11))
    @assert_passes(allow_import_failures=True)
    def test_class_type_param_defaults_cannot_follow_variadic(self):
        from typing import Generic

        from typing_extensions import TypeVar, TypeVarTuple, Unpack

        Ts = TypeVarTuple("Ts")
        DefaultAfterVariadic = TypeVar("DefaultAfterVariadic", default=bool)

        class BadAfterVariadic(  # E: invalid_type_parameter
            Generic[Unpack[Ts], DefaultAfterVariadic]
        ): ...

    @skip_before((3, 11))
    @assert_passes(allow_import_failures=True)
    def test_class_type_param_defaults_cannot_reference_later_params(self):
        from typing import Generic

        from typing_extensions import TypeVar

        LaterT = TypeVar("LaterT")
        EarlierDefaultT = TypeVar("EarlierDefaultT", default=list[LaterT])

        class Bad(Generic[EarlierDefaultT, LaterT]): ...  # E: invalid_type_parameter

    @skip_before((3, 13))
    def test_pep696_class_type_param_defaults_cannot_reference_later_params(self):
        self.assert_passes(
            """
            class Good[T, U = tuple[T]]:
                ...

            class Bad[T = tuple[U], U]:  # E: undefined_name  # E: invalid_type_parameter
                ...
        """,
            allow_import_failures=True,
        )

    @skip_before((3, 11))
    @assert_passes(allow_import_failures=True)
    def test_typevartuple_unpack_singleton_tuple_in_generic_base(self):
        from typing import Generic

        from typing_extensions import TypeVarTuple, Unpack

        Ts = TypeVarTuple("Ts")

        class Array(Generic[Unpack[(Ts,)]]): ...

    @assert_passes(run_in_both_module_modes=True)
    def test_class_assignability_with_defaults(self):
        from typing import Generic

        from typing_extensions import TypeVar

        T1 = TypeVar("T1")
        DefaultStrT = TypeVar("DefaultStrT", default=str)

        class SubclassMe(Generic[T1, DefaultStrT]):
            x: DefaultStrT

        class Bar(SubclassMe[int, DefaultStrT]): ...

        x1: type[Bar[str]] = Bar  # ok

    @assert_passes(allow_import_failures=True)
    def test_generic_default_specialization_after_import_failure(self):
        from typing import Generic, TypeAlias

        from typing_extensions import TypeVar, assert_type

        T1 = TypeVar("T1")
        T2 = TypeVar("T2")
        DefaultIntT = TypeVar("DefaultIntT", default=int)
        DefaultStrT = TypeVar("DefaultStrT", default=str)

        class SomethingWithNoDefaults(Generic[T1, T2]): ...

        MyAlias: TypeAlias = SomethingWithNoDefaults[int, DefaultStrT]

        def func1(p1: MyAlias, p2: MyAlias[bool]) -> None:
            assert_type(p1, SomethingWithNoDefaults[int, str])
            assert_type(p2, SomethingWithNoDefaults[int, bool])

        MyAlias[bool, int]  # E: invalid_specialization

        class SubclassMe(Generic[T1, DefaultStrT]):
            x: DefaultStrT

        class Bar(SubclassMe[int, DefaultStrT]): ...

        x1: type[Bar[str]] = Bar  # ok
        x2: type[Bar[int]] = Bar  # E: incompatible_assignment
        assert_type(Bar(), Bar[str])
        assert_type(Bar[bool](), Bar[bool])

        class Foo(SubclassMe[float]): ...

        assert_type(Foo().x, str)
        Foo[str]  # E: invalid_specialization

        class Baz(Generic[DefaultIntT, DefaultStrT]): ...

        class Spam(Baz): ...

        value: Baz[int, str] = Spam()
        assert_type(value, Baz[int, str])


class TestSolve(TestNameCheckVisitorBase):
    @assert_passes()
    def test_filter_like(self):
        from typing import Callable, TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def callable(o: object) -> bool:
            return True

        def filterish(func: Callable[[T], bool], data: T) -> T:
            return data

        def capybara():
            assert_type(filterish(callable, 1), Literal[1])

    @assert_passes()
    def test_one_any(self):
        from typing import TypeVar

        from typing_extensions import Literal

        T = TypeVar("T")

        def sub(x: T, y: T) -> T:
            return x

        def capybara(unannotated):
            assert_type(sub(1, unannotated), Literal[1])

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
                assert_type(t, StrSub)
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
            assert_type(take_seq(seq), bytes | str)

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
            assert_type(pick_first("a", "b"), str)
            assert_type(pick_first(b"a", b"b"), bytes)
            pick_first(u, "b")  # E: incompatible_argument
            pick_first(u, b"b")  # E: incompatible_argument
            pick_first("a", b"b")  # E: incompatible_call
            pick_first(b"a", "b")  # E: incompatible_call

    @assert_passes()
    def test_rigid_constrained_typevar_is_not_assignable_to_constraint_member(self):
        from typing import TypeVar

        T = TypeVar("T", int, str)

        def takes_int(x: int) -> None:
            pass

        def capybara(x: T) -> T:
            takes_int(x)  # E: incompatible_argument
            return x

    @assert_passes()
    def test_rigid_unconstrained_typevar_is_not_assignable_to_concrete_parameter(self):
        from typing import TypeVar

        T = TypeVar("T")

        def takes_int(x: int) -> None:
            pass

        def capybara(x: T) -> T:
            takes_int(x)  # E: incompatible_argument
            return x

    @assert_passes()
    def test_rigid_bound_typevar_is_not_assignable_to_narrower_parameter(self):
        from typing import TypeVar

        class Base:
            pass

        class Child(Base):
            pass

        T = TypeVar("T", bound=Base)

        def takes_child(x: Child) -> None:
            pass

        def capybara(x: T) -> T:
            takes_child(x)  # E: incompatible_argument
            return x

    @assert_passes()
    def test_covariant_bound_inference_prefers_precise_solution(self):
        from collections.abc import Iterator, ValuesView
        from typing import Protocol, TypeVar

        from typing_extensions import assert_type

        T_co = TypeVar("T_co", bound=Iterator[object], covariant=True)

        class HasIter(Protocol[T_co]):
            def __iter__(self) -> T_co: ...

        def capybara(x: HasIter[T_co]) -> T_co:
            return x.__iter__()

        def check(values: ValuesView[int]) -> None:
            assert_type(capybara(values), Iterator[int])

    @assert_passes()
    def test_constrained_typevar_binop_preserves_typevar(self):
        from typing import TypeVar

        AnyStr = TypeVar("AnyStr", str, bytes)

        def concat(x: AnyStr, y: AnyStr) -> AnyStr:
            return x + y

        def capybara(s: str, b: bytes) -> None:
            assert_type(concat(s, s), str)
            assert_type(concat(b, b), bytes)
            concat(s, b)  # E: incompatible_call
            concat(b, s)  # E: incompatible_call

    @assert_passes()
    def test_rigid_typevar_target_rejects_concrete_assignment(self):
        from typing import TypeVar

        T = TypeVar("T", int, str)

        def capybara(x: T) -> T:
            y: T = 1  # E: incompatible_assignment
            print(y)
            return x

    @assert_passes()
    def test_constraint_cannot_contain_typevars(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class C(Generic[T]):
            BadConstraint = TypeVar(
                "BadConstraint", str, list[T]  # E: invalid_annotation
            )

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
    def test_rigid_type_typevar_is_not_assignable_to_narrower_type_object(self):
        from typing import TypeVar

        class Base:
            pass

        class Child(Base):
            pass

        T = TypeVar("T", bound=Base)

        def takes_child_type(x: type[Child]) -> None:
            pass

        def capybara(cls: type[T]) -> type[T]:
            takes_child_type(cls)  # E: incompatible_argument
            return cls

    @assert_passes()
    def test_rigid_typevar_uses_upper_bound_as_source(self):
        from typing import TypeVar

        class Base:
            pass

        T = TypeVar("T", bound=Base)

        def takes_base(x: Base) -> None:
            pass

        def capybara(x: T) -> T:
            takes_base(x)
            return x

    @assert_passes()
    def test_type_typevar_inference_still_works(self):
        from typing import TypeVar

        from typing_extensions import assert_type

        class Base:
            pass

        class Child(Base):
            pass

        T = TypeVar("T", bound=Base)

        def keep_type(cls: type[T]) -> type[T]:
            return cls

        def capybara() -> None:
            assert_type(keep_type(Child), type[Child])

    @assert_passes()
    def test_min_enum(self):
        import enum

        class E(enum.IntEnum):
            a = 1
            b = 2

        def capybara():
            m = min(E)
            assert_type(m, E)

    @assert_passes()
    def test_constraints(self):
        from typing import List, TypeVar

        LT = TypeVar("LT", List[int], List[str])

        def g(x: LT) -> LT:
            return x

        def pacarana() -> None:
            assert_is_value(g([]), AnyValue(AnySource.inference))
            assert_type(g([1]), list[int])

    @assert_passes()
    def test_redundant_constraints(self):
        from typing import TypeVar

        from typing_extensions import SupportsIndex

        T = TypeVar("T", int, float, SupportsIndex)

        def f(x: T) -> T:
            return x

        def capybara(si: SupportsIndex):
            assert_type(f(1), int)
            assert_type(f(si), SupportsIndex)
            assert_type(f(1.0), float | int)

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
            assert_type(x, Annotated[int, 42])
            assert_type(f(x), Annotated[int, 42])


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

    @skip_before((3, 12))
    def test_reject_subscripted_generic_or_protocol_bases_with_pep695_syntax(self):
        self.assert_passes(
            """
            from typing import Generic, Protocol

            class GoodProtocol[T](Protocol):
                ...

            def capybara() -> None:
                class BadGeneric[T](Generic[T]):  # E: invalid_base
                    ...

                class BadProtocol[T](Protocol[T]):  # E: invalid_base
                    ...
        """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 12))
    def test_protocol_base_validation_with_duplicate_and_generic_bases(self):
        self.assert_passes(
            """
            from typing import Generic, Protocol, TypeVar

            T_co = TypeVar("T_co", covariant=True)

            class GoodGenericBase(Protocol[T_co], Generic[T_co]):
                ...

            def capybara() -> None:
                class BadDuplicate(Protocol[T_co, T_co]):
                    ...
        """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 12))
    def test_pep695_protocol_variance_is_inferred_without_errors(self):
        self.assert_passes("""
            from typing import Protocol

            class Reader[T](Protocol):
                def get(self) -> T: ...

            class Writer[T](Protocol):
                def put(self, value: T) -> None: ...
        """)

    @skip_before((3, 12))
    def test_pep695_type_parameter_forward_refs_in_bounds(self):
        self.assert_passes("""
            class Box[T: ForwardReference[int]]:
                ...

            class Choice[T: (ForwardReference[int], "ForwardReference[str]", bytes)]:
                ...

            class ForwardReference[T]:
                ...
        """)

    @skip_before((3, 12))
    def test_pep695_type_parameter_bound_and_constraint_validation(self):
        self.assert_passes("""
            class Outer[V]:
                class BadBound[T: dict[str, V]]:  # E: invalid_annotation
                    ...

            class BadConstraints0[T: ()]:  # E: invalid_annotation
                ...

            class BadConstraints1[T: (str,)]:  # E: invalid_annotation
                ...
        """)

    @skip_before((3, 12))
    def test_pep695_type_parameter_dependent_bounds_and_constraints(self):
        self.assert_passes("""
            from typing import Sequence

            class BadBound[S, T: Sequence[S]]:  # E: invalid_annotation
                ...

            class BadConstraint[S, T: (Sequence[S], bytes)]:  # E: invalid_annotation
                ...
        """)

    @skip_before((3, 12))
    def test_pep695_type_parameter_default_must_reference_earlier_params(self):
        self.assert_passes(
            """
            from typing import Generic

            from typing_extensions import TypeVar

            U = TypeVar("U")
            T = TypeVar("T", default=tuple[U])

            def capybara() -> None:
                class Bad(Generic[T, U]):  # E: invalid_type_parameter
                    ...
        """,
            run_in_both_module_modes=True,
        )

    @skip_before((3, 13))
    def test_pep696_type_parameter_defaults_match_bounds_and_constraints(self):
        self.assert_passes("""
            class GoodBound[T: float = int]:
                ...

            class BadBound[T: str = int]:  # E: invalid_annotation
                ...

            class GoodConstraint[T: (int, str) = int]:
                ...

            class BadConstraint[T: (float, str) = int]:  # E: invalid_annotation
                ...

            class GoodConstraintTypeVar[Base: (int, str), T: (int, str, bool) = Base]:
                ...

            class BadConstraintTypeVar[
                # E: invalid_annotation
                Base: (int, str), T: (bool, complex) = Base
            ]:
                ...
        """)

    @skip_before((3, 13))
    def test_pep696_type_parameter_default_with_bound_typevar(self):
        self.assert_passes("""
            class AcceptsBoundTypeVar[Base: int, T: (int, str) = Base]:
                ...

            class RejectsBoundTypeVar[Base: bytes, T: (int, str) = Base]:  # E: invalid_annotation
                ...
        """)

    @assert_passes()
    def test_runtime_typevar_default_matches_literal_constraints_and_bound_typevar(
        self,
    ):
        from typing import Literal

        from typing_extensions import TypeVar

        LiteralOk = TypeVar("LiteralOk", Literal[1], Literal[2], default=Literal[1])
        LiteralBad = TypeVar(
            "LiteralBad",
            Literal[1],
            Literal[2],
            default=Literal[3],  # E: incompatible_call
        )
        BoundInt = TypeVar("BoundInt", bound=int)
        BoundDefaultOk = TypeVar(
            "BoundDefaultOk", int, str, default=BoundInt  # E: incompatible_call
        )
        BoundDefaultBad = TypeVar(
            "BoundDefaultBad", str, bytes, default=BoundInt  # E: incompatible_call
        )

        print(LiteralOk, LiteralBad, BoundDefaultOk, BoundDefaultBad)

    @skip_before((3, 12))
    def test_pep695_typevartuple_syntax_in_class_and_function(self):
        self.assert_passes(
            """
            from typing_extensions import assert_type

            class Array[*Ts]:
                def pack(self, *args: *Ts) -> tuple[*Ts]:
                    return args

            def capybara() -> None:
                assert_type(Array[int, str]().pack(1, "x"), tuple[int, str])
        """,
            run_in_both_module_modes=True,
        )

    @assert_passes(run_in_both_module_modes=True)
    def test_generic_base_must_cover_all_class_type_params(self):
        from typing import Generic, TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class PairBase(Generic[T, U]):
            pass

        def capybara() -> None:
            class Good(PairBase[T, U], Generic[T, U]):
                pass

            class Bad(PairBase[T, U], Generic[T]):  # E: invalid_base
                pass

    @skip_before((3, 11))
    @assert_passes()
    def test_manual_runtime_type_params_align_legacy_generic_kinds(self):
        from typing import Callable, Generic

        from typing_extensions import (
            ParamSpec,
            TypeVar,
            TypeVarTuple,
            Unpack,
            assert_type,
        )

        T = TypeVar("T")
        P = ParamSpec("P")
        Ts = TypeVarTuple("Ts")

        # This is intentionally odd: it forces the runtime-alignment path to
        # see `__type_params__` even for legacy `Generic[...]` syntax.
        class Box(Generic[T]):
            __type_params__ = (T,)
            value: T

            def __init__(self, value: T) -> None:
                self.value = value

        class Wrapper(Generic[P]):
            __type_params__ = (P,)

            def call(
                self, fn: Callable[P, int], *args: P.args, **kwargs: P.kwargs
            ) -> int:
                return fn(*args, **kwargs)

        class Array(Generic[Unpack[Ts]]):
            __type_params__ = (Ts,)

            def pack(self, *args: Unpack[Ts]) -> tuple[Unpack[Ts]]:
                return args

        def capybara(
            box: Box[int], wrapper: Wrapper[[int, str]], array: Array[int, str]
        ) -> None:
            assert_type(box.value, int)
            assert_type(wrapper, Wrapper[[int, str]])
            assert_type(array, Array[int, str])

    @assert_passes()
    def test_manual_runtime_type_params_append_unmatched_declared_param(self):
        from typing import Generic

        from typing_extensions import TypeVar

        T = TypeVar("T")
        U = TypeVar("U")

        class Weird(Generic[T]):
            __type_params__ = (T, U)

        def capybara() -> None:
            print(Weird)

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

            Node[int].label = 1  # E: incompatible_assignment
            Node[int].label  # E: undefined_attribute
            Node.label = 1  # E: incompatible_assignment
            Node.label  # E: undefined_attribute
            type(n1).label  # E: undefined_attribute

    @skip_before((3, 12))
    @assert_passes(allow_import_failures=True)
    def test_legacy_typevar_infer_variance_after_import_failure(self):
        from typing import Generic, Iterator, TypeVar

        X1 = TypeVar("X1", covariant=True, contravariant=True)  # E: incompatible_call
        T = TypeVar("T", infer_variance=True)

        class Co(Generic[T]):
            def __iter__(self) -> Iterator[T]:
                raise NotImplementedError

        class Contra(Generic[T]):
            def set_value(self, value: T) -> None:
                raise NotImplementedError

        co_ok: Co[object] = Co[int]()
        co_bad: Co[int] = Co[object]()  # E: incompatible_assignment

        contra_bad: Contra[object] = Contra[int]()  # E: incompatible_assignment
        contra_ok: Contra[int] = Contra[object]()

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
    def test_infer_variance_from_inferred_variance_base(self):
        self.assert_passes("""
            from typing import Iterator

            class Parent[T]:
                def __iter__(self) -> Iterator[T]:
                    raise NotImplementedError

            class Child[T](Parent[T]):
                pass

            a: Child[float] = Child[int]()
            b: Child[int] = Child[float]()  # E: incompatible_assignment
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
            wrapper(1)  # E: incompatible_argument

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

    @assert_passes()
    def test_impl(self):
        from typing_extensions import TypeVar

        T = TypeVar("T")
        U = TypeVar("U", default=42)  # E: incompatible_argument
        V = TypeVar("V", bound=int, default=str)  # E: incompatible_call
        W = TypeVar("W", int, str, default=bool)  # E: incompatible_call
