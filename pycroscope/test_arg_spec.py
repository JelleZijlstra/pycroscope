# static analysis: ignore
import functools
from dataclasses import dataclass
from typing import List, NewType, TypeVar

import pytest

from .checker import Checker
from .maybe_asynq import asynq
from .signature import BoundMethodSignature, ParameterKind, Signature, SigParameter
from .stacked_scopes import Composite
from .test_name_check_visitor import (
    ConfiguredNameCheckVisitor,
    TestNameCheckVisitorBase,
)
from .test_node_visitor import assert_passes
from .value import (
    AnySource,
    AnyValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    TypedValue,
    assert_is_value,
)

T = TypeVar("T")
NT = NewType("NT", int)


class ClassWithCall(object):
    def __init__(self, name):
        pass

    def __call__(self, arg):
        pass

    @classmethod
    def normal_classmethod(cls):
        pass

    @staticmethod
    def normal_staticmethod(arg):
        pass


def function(capybara, hutia=3, *tucotucos, **proechimys):
    pass


def wrapped(args: int, kwargs: str) -> None:
    pass


def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


@dataclass
class AllTheAttrs:
    x: List[str]

    def __getattr__(self, attr: str) -> "AllTheAttrs":
        return AllTheAttrs([*self.x, attr])


def test_get_argspec():
    checker = Checker()
    visitor = ConfiguredNameCheckVisitor(
        __file__, "", {}, fail_after_first=False, checker=checker
    )
    cwc_typed = TypedValue(ClassWithCall)

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):
        asc = checker.arg_spec_cache

        # there's special logic for this in signature_from_value; TODO move that into
        # ExtendedArgSpec
        assert Signature.make(
            [SigParameter("arg")], callable=ClassWithCall.__call__
        ) == visitor.signature_from_value(cwc_typed)

        ata = AllTheAttrs([])
        assert asc.get_argspec(ata) is None

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls")],
                callable=ClassWithCall.normal_classmethod.__func__,
            ),
            Composite(KnownValue(ClassWithCall)),
        ) == asc.get_argspec(ClassWithCall.normal_classmethod)
        assert Signature.make(
            [SigParameter("arg")], callable=ClassWithCall.normal_staticmethod
        ) == asc.get_argspec(ClassWithCall.normal_staticmethod)

        assert Signature.make(
            [
                SigParameter("capybara"),
                SigParameter("hutia", default=KnownValue(3)),
                SigParameter("tucotucos", ParameterKind.VAR_POSITIONAL),
                SigParameter("proechimys", ParameterKind.VAR_KEYWORD),
            ],
            callable=function,
        ) == asc.get_argspec(function)

        assert Signature.make(
            [
                SigParameter("args", annotation=TypedValue(int)),
                SigParameter("kwargs", annotation=TypedValue(str)),
            ],
            KnownValue(None),
            callable=wrapped,
        ) == asc.get_argspec(wrapped)
        decorated = decorator(wrapped)
        assert Signature.make(
            [
                SigParameter(
                    "args",
                    ParameterKind.VAR_POSITIONAL,
                    annotation=AnyValue(AnySource.inference),
                ),
                SigParameter(
                    "kwargs",
                    ParameterKind.VAR_KEYWORD,
                    annotation=AnyValue(AnySource.inference),
                ),
            ],
            callable=decorated,
        ) == asc.get_argspec(decorated)
        assert Signature.make(
            [
                SigParameter(
                    "x", ParameterKind.POSITIONAL_ONLY, annotation=TypedValue(int)
                )
            ],
            NewTypeValue("NT", TypedValue(int), NT),
            callable=NT,
        ) == asc.get_argspec(NT)


def test_get_argspec_asynq():
    if asynq is None:
        pytest.skip("asynq is not available")
    from pycroscope.asynq_tests import ClassWithCallAsynq, async_function

    checker = Checker()

    # test everything twice because calling qcore.get_original_fn has side effects
    for _ in range(2):
        asc = checker.arg_spec_cache

        assert Signature.make(
            [SigParameter("x"), SigParameter("y")],
            callable=async_function.fn,
            is_asynq=True,
        ) == asc.get_argspec(async_function)

        assert Signature.make(
            [SigParameter("x"), SigParameter("y")],
            callable=async_function.fn,
            is_asynq=True,
        ) == asc.get_argspec(async_function.asynq)

        instance = ClassWithCallAsynq(1)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("self"), SigParameter("x")],
                callable=instance.async_method.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(instance)),
        ) == asc.get_argspec(instance.async_method)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("self"), SigParameter("x")],
                callable=instance.async_method.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(instance)),
        ) == asc.get_argspec(instance.async_method.asynq)

        assert Signature.make(
            [SigParameter("y")],
            callable=ClassWithCallAsynq.async_staticmethod.fn,
            is_asynq=True,
        ) == asc.get_argspec(ClassWithCallAsynq.async_staticmethod)

        assert Signature.make(
            [SigParameter("y")],
            callable=ClassWithCallAsynq.async_staticmethod.fn,
            is_asynq=True,
        ) == asc.get_argspec(ClassWithCallAsynq.async_staticmethod.asynq)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("z")],
                callable=ClassWithCallAsynq.async_classmethod.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCallAsynq)),
        ) == asc.get_argspec(ClassWithCallAsynq.async_classmethod)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("z")],
                callable=ClassWithCallAsynq.async_classmethod.decorator.fn,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCallAsynq)),
        ) == asc.get_argspec(ClassWithCallAsynq.async_classmethod.asynq)

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("ac")],
                callable=ClassWithCallAsynq.pure_async_classmethod.decorator.fn,
            ),
            Composite(KnownValue(ClassWithCallAsynq)),
        ) == asc.get_argspec(ClassWithCallAsynq.pure_async_classmethod)

        # This behaves differently in 3.9 through 3.12 than in earlier and later
        # versions. The behavior change was made in
        # https://github.com/python/cpython/issues/63272
        # and undone in https://github.com/python/cpython/issues/89519
        if hasattr(ClassWithCallAsynq.classmethod_before_async, "decorator"):
            callable = ClassWithCallAsynq.classmethod_before_async.decorator.fn
        else:
            callable = ClassWithCallAsynq.classmethod_before_async.__func__.fn

        assert BoundMethodSignature(
            Signature.make(
                [SigParameter("cls"), SigParameter("ac")],
                callable=callable,
                is_asynq=True,
            ),
            Composite(KnownValue(ClassWithCallAsynq)),
        ) == asc.get_argspec(ClassWithCallAsynq.classmethod_before_async)


def test_positional_only():
    def f(__x, _f__x):
        pass

    class Y:
        def f(self, __x):
            pass

        class X:
            def f(self, __x, _Y__x):
                pass

    asc = Checker().arg_spec_cache
    assert asc.get_argspec(f) == Signature.make(
        [
            SigParameter("__x", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_f__x", ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
        callable=f,
    )
    assert asc.get_argspec(Y.f) == Signature.make(
        [
            SigParameter("self", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_Y__x", ParameterKind.POSITIONAL_ONLY),
        ],
        callable=Y.f,
    )
    assert asc.get_argspec(Y.X.f) == Signature.make(
        [
            SigParameter("self", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_X__x", ParameterKind.POSITIONAL_ONLY),
            SigParameter("_Y__x", ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
        callable=Y.X.f,
    )


class TestClassInstantiation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_union_with_impl(self):
        def capybara(cond: bool) -> None:
            if cond:
                typ = list
            else:
                typ = tuple
            assert_is_value(typ, KnownValue(list) | KnownValue(tuple))
            assert_is_value(typ([1]), KnownValue([1]) | KnownValue((1,)))

    @assert_passes()
    def test_union_without_impl(self):
        class A:
            pass

        class B:
            pass

        def capybara(cond: bool) -> None:
            if cond:
                cls = A
            else:
                cls = B
            assert_is_value(cls(), MultiValuedValue([TypedValue(A), TypedValue(B)]))

    @assert_passes()
    def test_constructor_impl(self):
        from pycroscope.tests import FailingImpl

        def capybara():
            FailingImpl()  # E: incompatible_call

    @assert_passes()
    def test_subclass_value(self):
        from typing import Type

        class A:
            def __init__(self, x: int) -> None:
                pass

        def capybara(t: Type[A]) -> None:
            assert_is_value(t(1), TypedValue(A))
            t("x")  # E: incompatible_argument

    @assert_passes()
    def test_constructor_forward_refs(self):
        import pathlib

        class Capybara:
            def __init__(self, p: "pathlib.Path") -> None:
                pass

        def capybara():
            Capybara(3)  # E: incompatible_argument
            Capybara(pathlib.Path("x"))

    @assert_passes()
    def test_dunder_signature(self):
        import inspect

        class Cls:
            __signature__ = inspect.Signature(
                [inspect.Parameter("x", inspect.Parameter.KEYWORD_ONLY)]
            )

        def capybara():
            assert_is_value(Cls(x=3), TypedValue(Cls))
            Cls()  # E: incompatible_call
            Cls(1)  # E: incompatible_call


class TestFunctionsSafeToCall(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        def test(self):
            assert_is_value(sorted([3, 1, 2]), KnownValue([1, 2, 3]))


class TestNamedTuple(TestNameCheckVisitorBase):
    @assert_passes()
    def test_args(self):
        from typing import NamedTuple

        class NT(NamedTuple):
            field: int

        class CustomNew:
            def __new__(self, a: int) -> "CustomNew":
                return super().__new__(self)

        def make_nt() -> NT:
            return NT(field=3)

        def capybara():
            NT(filed=3)  # E: incompatible_call
            nt2 = make_nt()
            assert_is_value(nt2, TypedValue(NT))
            assert_is_value(nt2.field, TypedValue(int))

            CustomNew("x")  # E: incompatible_argument
            cn = CustomNew(a=3)
            assert_is_value(cn, TypedValue(CustomNew))


class TestBuiltinMethods(TestNameCheckVisitorBase):
    @assert_passes()
    def test_method_wrapper(self):
        import collections.abc

        def capybara():
            r = range(10)
            assert_is_value(r, KnownValue(range(10)))
            assert_is_value(
                r.__iter__(), GenericValue(collections.abc.Iterator, [TypedValue(int)])
            )


class TestPytestRaises(TestNameCheckVisitorBase):
    @assert_passes()
    def test(self):
        import pytest

        def capybara():
            pytest.raises(TypeError, len, 1)

            with pytest.raises(TypeError):
                pass

            with pytest.raises(TypeError, match="no match"):
                pass

            pytest.raises()  # E: incompatible_call
