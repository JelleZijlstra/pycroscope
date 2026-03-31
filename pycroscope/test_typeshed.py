# static analysis: ignore
import _io
import abc
import collections
import collections.abc
import contextlib
import io
import sys
import tempfile
import textwrap
import time
import typing
import urllib.parse
from collections.abc import Collection, MutableSequence, Reversible, Sequence, Set
from pathlib import Path
from typing import Any, Generic, List, NewType, Type, TypeVar, Union
from urllib.error import HTTPError

import pytest
import typing_extensions
from typeshed_client import Resolver, get_search_context

from .checker import Checker
from .extensions import evaluated
from .options import InvalidConfigOption, Options
from .signature import OverloadedSignature, Signature, SigParameter
from .test_arg_spec import ClassWithCall
from .test_config import TEST_OPTIONS
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .tests import make_simple_sequence
from .typeshed import TypeshedFinder, TypeshedPath
from .value import (
    UNINITIALIZED_VALUE,
    AnySource,
    AnyValue,
    CallableValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    KVPair,
    NewTypeValue,
    SequenceValue,
    SubclassValue,
    SyntheticClassObjectValue,
    TypedDictEntry,
    TypedDictValue,
    TypedValue,
    TypeVarMap,
    TypeVarParam,
    TypeVarValue,
    Value,
    _iter_typevar_map_items,
    assert_is_value,
)

T = TypeVar("T")
NT = NewType("NT", int)


class TestTypeshedClient(TestNameCheckVisitorBase):
    @assert_passes()
    def test_types(self):
        import math
        from typing import Container

        assert_type(math.exp(1.0), float | int)
        assert_type("".isspace(), bool)

        def capybara(x: Container[int]) -> None:
            assert_type(x.__contains__(1), bool)

    @assert_passes()
    def test_dict_update(self):
        def capybara():
            x = {}
            x.update({})  # just check that this doesn't fail

    def test_get_bases(self):
        tsf = TypeshedFinder(Checker(), verbose=True)

        # Typeshed removed Generic[] from some base lists, and TypeVar identities
        # may differ across versions. Assert on base type shape rather than exact
        # TypeVar object identity.
        def assert_with_maybe_generic(
            cls: Type[object], expected_bases: List[type]
        ) -> None:
            actual = tsf.get_bases(cls)
            assert actual is not None
            actual_generic = [base for base in actual if isinstance(base, GenericValue)]
            actual_types = [base.typ for base in actual_generic]

            assert set(actual_types) in (
                set(expected_bases),
                {*expected_bases, Generic},
            )
            for base in actual_generic:
                if base.typ is Generic:
                    continue
                assert len(base.args) == 1
                assert isinstance(base.args[0], TypeVarValue)

        assert_with_maybe_generic(list, [MutableSequence])
        assert_with_maybe_generic(Sequence, [Reversible, Collection])
        assert_with_maybe_generic(Set, [Collection])

    def test_get_direct_symbol_ignores_undotted_local_name(self):
        tsf = TypeshedFinder(Checker(), verbose=True)

        assert tsf.get_direct_symbol("Params", "__new__") is None
        assert tsf.get_bases_for_fq_name("Params") is None
        assert (
            tsf.get_attribute_for_fq_name("Params", "__new__", on_class=True)
            is UNINITIALIZED_VALUE
        )

    def test_newtype(self):
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "typing.pyi").write_text("def NewType(a, b): pass\n")
            (temp_dir / "newt.pyi").write_text(textwrap.dedent("""
                from typing import NewType

                NT = NewType("NT", int)
                Alias = int

                def f(x: NT, y: Alias) -> None:
                    pass
                """))
            (temp_dir / "VERSIONS").write_text("newt: 3.5\ntyping: 3.5\n")
            (temp_dir / "@python2").mkdir()
            tsf = TypeshedFinder(Checker(), verbose=True)
            search_context = get_search_context(typeshed=temp_dir, search_path=[])
            tsf.resolver = Resolver(search_context)

            def runtime_f():
                pass

            sig = tsf.get_argspec_for_fully_qualified_name("newt.f", runtime_f)
            newtype = next(iter(tsf._assignment_cache.values()))
            assert isinstance(newtype, KnownValue)
            ntv = NewTypeValue("NT", TypedValue(int), newtype.val)
            assert sig == (
                Signature.make(
                    [
                        SigParameter("x", annotation=ntv),
                        SigParameter("y", annotation=TypedValue(int)),
                    ],
                    KnownValue(None),
                    callable=runtime_f,
                )
            )

    @assert_passes()
    def test_generic_self(self):
        from typing import Dict

        def capybara(x: Dict[int, str]):
            assert_is_value(
                {k: v for k, v in x.items()},
                DictIncompleteValue(
                    dict, [KVPair(TypedValue(int), TypedValue(str), is_many=True)]
                ),
            )

    @assert_passes()
    def test_str_find(self):
        def capybara(s: str) -> None:
            assert_type(s.find("x"), int)

    @assert_passes()
    def test_str_count(self):
        def capybara(s: str) -> None:
            assert_type(s.count("x"), int)

    def test_get_fq_name_uses_safe_attribute_access(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)

        class ExplodesOnQualname:
            __module__ = "builtins"

            def __getattr__(self, name: str) -> Any:
                if name == "__qualname__":
                    raise RuntimeError("boom")
                raise AttributeError(name)

        assert tsf._get_fq_name(ExplodesOnQualname()) is None

    @assert_passes()
    def test_dict_fromkeys(self):
        def capybara(i: int) -> None:
            assert_is_value(
                dict.fromkeys([i]),
                GenericValue(
                    dict,
                    [TypedValue(int), AnyValue(AnySource.explicit) | KnownValue(None)],
                ),
            )

    @assert_passes()
    def test_datetime(self):
        from datetime import datetime

        def capybara(i: int):
            dt = datetime.fromtimestamp(i)
            assert_type(dt, datetime)

    def test_has_stubs(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert tsf.has_stubs(object)
        assert not tsf.has_stubs(ClassWithCall)

    def test_get_attribute(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert UNINITIALIZED_VALUE is tsf.get_attribute(object, "nope", on_class=False)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )

    def test_get_direct_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "sample.pyi").write_text(textwrap.dedent("""
                from typing import final

                class C:
                    x: int

                    @property
                    def name(self) -> str: ...

                    @classmethod
                    def make(cls) -> "C": ...

                    @staticmethod
                    @final
                    def build() -> int: ...
                """))
            (temp_dir / "VERSIONS").write_text("sample: 3.8\n")
            (temp_dir / "@python2").mkdir()

            tsf = TypeshedFinder(Checker(), verbose=True)
            search_context = get_search_context(typeshed=temp_dir, search_path=[])
            tsf.resolver = Resolver(search_context)

            attr = tsf.get_direct_symbol("sample.C", "x")
            assert attr is not None
            assert attr.annotation == TypedValue(int)
            assert attr.is_instance_only

            prop = tsf.get_direct_symbol("sample.C", "name")
            assert prop is not None
            assert prop.property_info is not None
            assert prop.property_info.getter_type == TypedValue(str)

            make = tsf.get_direct_symbol("sample.C", "make")
            assert make is not None
            assert make.is_method
            assert make.is_classmethod
            assert isinstance(make.initializer, CallableValue)

            build = tsf.get_direct_symbol("sample.C", "build")
            assert build is not None
            assert build.is_method
            assert build.is_staticmethod
            assert tsf.get_direct_symbol("sample.C", "missing") is None

    def test_override_typeshed_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "VERSIONS").write_text("typing: 3.8\n")
            (temp_dir / "typing.pyi").write_text("Alias = int\n")
            (temp_dir / "@python2").mkdir()

            options = Options.from_option_list([TypeshedPath(temp_dir)])
            tsf = TypeshedFinder.make(Checker(), options, verbose=True)

            assert tsf.resolver.ctx.typeshed == temp_dir

    def test_override_typeshed_path_with_stdlib_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            stdlib = temp_dir / "stdlib"
            stdlib.mkdir()
            (stdlib / "VERSIONS").write_text("typing: 3.8\n")
            (stdlib / "typing.pyi").write_text("Alias = int\n")
            (temp_dir / "@python2").mkdir()

            options = Options.from_option_list([TypeshedPath(temp_dir)])
            tsf = TypeshedFinder.make(Checker(), options, verbose=True)

            assert tsf.resolver.ctx.typeshed == stdlib

    def test_override_typeshed_path_without_versions_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            (temp_dir / "stdlib").mkdir()

            options = Options.from_option_list([TypeshedPath(temp_dir)])
            with pytest.raises(InvalidConfigOption, match="typeshed_path.*VERSIONS"):
                TypeshedFinder.make(Checker(), options, verbose=True)


_EXPECTED_TYPED_DICTS = {
    "TD1": TypedDictValue(
        {"a": TypedDictEntry(TypedValue(int)), "b": TypedDictEntry(TypedValue(str))}
    ),
    "TD2": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int), required=False),
            "b": TypedDictEntry(TypedValue(str), required=False),
        }
    ),
    "PEP655": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int), required=False),
            "b": TypedDictEntry(TypedValue(str)),
        }
    ),
    "Inherited": TypedDictValue(
        {
            "a": TypedDictEntry(TypedValue(int)),
            "b": TypedDictEntry(TypedValue(str)),
            "c": TypedDictEntry(TypedValue(float) | TypedValue(int)),
        }
    ),
}


class TestBundledStubs(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import_aliases(self):
        def capybara():
            from _pycroscope_tests.aliases import (
                ExplicitAlias,
                aliased_constant,
                constant,
                explicitly_aliased_constant,
            )

            assert_is_value(ExplicitAlias, KnownValue(int))
            assert_type(constant, int)
            assert_type(aliased_constant, int)
            assert_type(explicitly_aliased_constant, int)

    def test_aliases(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pycroscope_tests.aliases"
        assert tsf.resolve_name(mod, "constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "aliased_constant") == TypedValue(int)
        assert tsf.resolve_name(mod, "explicitly_aliased_constant") == TypedValue(int)

    def test_overloaded(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pycroscope_tests.overloaded"
        val = tsf.resolve_name(mod, "func")
        assert isinstance(val, CallableValue)
        assert isinstance(val.signature, OverloadedSignature)

    @assert_passes()
    def test_pos_only(self):
        def capybara():
            from _pycroscope_tests.posonly import f, g, h, two_pos_only

            f(1, 2)
            f(1, 2, 3)
            f(1, y=2, z=3)
            f(x=1, y=2)  # E: incompatible_call
            f()  # E: incompatible_call

            g()
            g(1)
            g(1, 2)
            g(1, y=2)
            g(x=1)  # E: incompatible_call

            h()  # E: incompatible_call
            h(1)
            h(x=1)  # E: incompatible_call
            h(1, y=2)
            h(1, 2, 3)

            two_pos_only(1, "x")
            two_pos_only(1)
            two_pos_only(x=1)  # E: incompatible_call
            two_pos_only(1, y="x")  # E: incompatible_call

    @assert_passes()
    def test_typevar_with_default(self):
        def capybara(x: int):
            from _pycroscope_tests.typevar import f

            assert_type(f(x), int)

    @assert_passes()
    def test_typing_extensions_paramspec(self):
        def some_func(x: int) -> str:
            return str(x)

        def capybara(x: int):
            from _pycroscope_tests.paramspec import f, g

            assert_type(f(x), int)
            assert_type(g(some_func, x), str)
            # Clue as to where the bug is: reveal_type(g) shows that *args and **kwargs get inferred
            # as Any[error],  so we probably don't handle P.args / P.kwargs correctly in stubs.
            g(some_func, "not an int")  # TODO should error.

    def test_typeddict(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pycroscope_tests.typeddict"

        for name, expected in _EXPECTED_TYPED_DICTS.items():
            assert tsf.resolve_name(mod, name) == SyntheticClassObjectValue(
                name, expected
            )

    @assert_passes()
    def test_cdata(self):
        def capybara():
            from _pycroscope_tests.cdata import f

            assert_is_value(f(), TypedValue("_ctypes._CData"))

    @assert_passes()
    def test_ast(self):
        import ast

        def capybara(x: ast.Yield):
            assert_type(x, ast.Yield)
            assert_type(x.value, ast.expr | None)

    @assert_passes()
    def test_import_typeddicts(self):
        def capybara():
            from _pycroscope_tests.typeddict import PEP655, TD1, TD2, Inherited

            from pycroscope.test_typeshed import _EXPECTED_TYPED_DICTS

            def nested(td1: TD1, td2: TD2, pep655: PEP655, inherited: Inherited):
                assert_is_value(td1, _EXPECTED_TYPED_DICTS["TD1"])
                assert_is_value(td2, _EXPECTED_TYPED_DICTS["TD2"])
                assert_is_value(pep655, _EXPECTED_TYPED_DICTS["PEP655"])
                assert_is_value(inherited, _EXPECTED_TYPED_DICTS["Inherited"])

    def test_evaluated(self):
        tsf = TypeshedFinder.make(Checker(), TEST_OPTIONS, verbose=True)
        mod = "_pycroscope_tests.evaluated"
        assert tsf.resolve_name(mod, "evaluated") == KnownValue(evaluated)

    @assert_passes()
    def test_evaluated_import(self):
        def capybara(unannotated):
            from typing import IO, BinaryIO, TextIO

            from _pycroscope_tests.evaluated import open, open2

            assert_type(open("r"), TextIO)
            assert_type(open("rb"), BinaryIO)
            assert_is_value(
                open(unannotated), GenericValue(IO, [AnyValue(AnySource.explicit)])
            )
            assert_type(open("r" if unannotated else "rb"), TextIO | BinaryIO)
            assert_type(open2("r"), TextIO)
            assert_type(open2("rb"), BinaryIO)
            assert_is_value(
                open2(unannotated), GenericValue(IO, [AnyValue(AnySource.explicit)])
            )
            assert_type(open2("r" if unannotated else "rb"), TextIO | BinaryIO)

    @assert_passes()
    def test_recursive_base(self):
        from typing import Any, ContextManager

        def capybara():
            from _pycroscope_tests.recursion import _ScandirIterator

            def want_cm(cm: ContextManager[Any]) -> None:
                pass

            def f(x: _ScandirIterator):
                want_cm(x)
                len(x)  # E: incompatible_argument

    @assert_passes()
    def test_args_kwargs(self):
        def capybara():
            from _pycroscope_tests.args import f, g, h, i

            f(1)  # E: incompatible_call
            f(1, "x")
            g(x=1)  # E: incompatible_call
            g(x=1, y="x")
            h("x")  # E: incompatible_argument
            h()
            h(1)
            i(x=3)  # E: incompatible_argument
            i(x="x")
            i()

    @assert_passes()
    def test_stub_context_manager(self):
        from typing_extensions import Literal

        def capybara():
            from _pycroscope_tests.contextmanager import cm

            with cm() as f:
                assert_type(f, int)
                x = 3

            assert_type(x, Literal[3])

    @assert_passes()
    def test_stub_defaults(self):
        def capybara():
            from _pycroscope_tests.defaults import many_defaults

            a, b, c, d = many_defaults()
            assert_is_value(
                a, DictIncompleteValue(dict, [KVPair(KnownValue("a"), KnownValue(1))])
            )
            assert_is_value(
                b,
                SequenceValue(
                    list, [(False, KnownValue(1)), (False, SequenceValue(tuple, []))]
                ),
            )
            assert_is_value(
                c,
                SequenceValue(tuple, [(False, KnownValue(1)), (False, KnownValue(2))]),
            )
            assert_is_value(
                d, SequenceValue(set, [(False, KnownValue(1)), (False, KnownValue(2))])
            )


class TestConstructors(TestNameCheckVisitorBase):
    @assert_passes()
    def test_init_new(self):
        def capybara():
            from _pycroscope_tests.initnew import (
                my_enumerate,
                overloadinit,
                overloadnew,
                simple,
                simplenew,
            )

            simple()  # E: incompatible_call
            simple("x")  # E: incompatible_argument
            assert_is_value(simple(1), TypedValue("_pycroscope_tests.initnew.simple"))

            my_enumerate()  # E: incompatible_call
            my_enumerate([1], start="x")  # E: incompatible_argument
            assert_is_value(
                my_enumerate([1]),
                GenericValue("_pycroscope_tests.initnew.my_enumerate", [KnownValue(1)]),
            )

            overloadinit()  # E: incompatible_call
            assert_is_value(
                overloadinit(1, "x", 2),
                GenericValue("_pycroscope_tests.initnew.overloadinit", [KnownValue(2)]),
            )

            simplenew()  # E: incompatible_call
            assert_is_value(
                simplenew(1), TypedValue("_pycroscope_tests.initnew.simplenew")
            )

            overloadnew()  # E: incompatible_call
            assert_is_value(
                overloadnew(1, "x", 2),
                GenericValue("_pycroscope_tests.initnew.overloadnew", [KnownValue(2)]),
            )

    @assert_passes()
    def test_typeshed_constructors(self):
        def capybara(x):
            assert_type(int(x), int)
            assert_is_value(
                frozenset(),
                GenericValue(frozenset, [AnyValue(AnySource.generic_argument)]),
            )

            assert_type(type("x"), type)
            assert_type(type("x", (), {}), type)


class Parent(Generic[T]):
    pass


class Child(Parent[int]):
    pass


class GenericChild(Parent[T]):
    pass


BasesMap = dict[Union[type, str], list[Value]]


class TestGetGenericBases:
    def setup_method(self) -> None:
        checker = Checker()
        self.get_generic_bases = checker.arg_spec_cache.get_generic_bases

    def test_runtime(self):
        assert {
            Parent: TypeVarMap(typevars={T: AnyValue(AnySource.generic_argument)})
        } == self.get_generic_bases(Parent)
        assert {
            Parent: TypeVarMap(typevars={T: TypeVarValue(TypeVarParam(T))})
        } == self.get_generic_bases(Parent, [TypeVarValue(TypeVarParam(T))])
        assert {
            Child: TypeVarMap(),
            Parent: TypeVarMap(typevars={T: TypedValue(int)}),
        } == self.get_generic_bases(Child)
        assert {
            GenericChild: TypeVarMap(
                typevars={T: AnyValue(AnySource.generic_argument)}
            ),
            Parent: TypeVarMap(typevars={T: AnyValue(AnySource.generic_argument)}),
        } == self.get_generic_bases(GenericChild)
        one = KnownValue(1)
        assert {
            GenericChild: TypeVarMap(typevars={T: one}),
            Parent: TypeVarMap(typevars={T: one}),
        } == self.get_generic_bases(GenericChild, [one])

    def _assert_runtime_any_base(self, any_base: object) -> None:
        class ParentFromAny(any_base):
            pass

        class ChildFromAny(ParentFromAny):
            pass

        assert self.get_generic_bases(ParentFromAny) == {ParentFromAny: TypeVarMap()}
        assert self.get_generic_bases(ChildFromAny) == {
            ChildFromAny: TypeVarMap(),
            ParentFromAny: TypeVarMap(),
        }

    def test_runtime_with_typing_any_base(self):
        if sys.version_info < (3, 11):
            return
        self._assert_runtime_any_base(typing.Any)

    def test_runtime_with_typing_extensions_any_base(self):
        self._assert_runtime_any_base(typing_extensions.Any)

    def test_runtime_annotated_special_form(self):
        assert {"typing.Annotated": TypeVarMap()} == self.get_generic_bases(
            "typing.Annotated"
        )

    def check(
        self,
        expected: Union[BasesMap, list[BasesMap]],
        base: Union[type, str],
        args: typing.Sequence[Value] = (),
    ) -> None:
        actual = self.get_generic_bases(base, args)
        cleaned = {
            base: [value for _, value in _iter_typevar_map_items(tv_map)]
            for base, tv_map in actual.items()
        }
        if isinstance(expected, list):
            assert cleaned in expected
        else:
            assert expected == cleaned

    def test_coroutine(self):
        one = KnownValue(1)
        two = KnownValue(2)
        three = KnownValue(3)
        self.check(
            {
                collections.abc.Coroutine: [one, two, three],
                collections.abc.Awaitable: [three],
            },
            collections.abc.Coroutine,
            [one, two, three],
        )

    def test_callable(self):
        self.check({collections.abc.Callable: []}, collections.abc.Callable)

    def test_dict_items(self):
        TInt = TypedValue(int)
        TStr = TypedValue(str)
        TTuple = make_simple_sequence(tuple, [TInt, TStr])
        self.check(
            {
                "_collections_abc.dict_items": [TInt, TStr],
                collections.abc.Iterable: [TTuple],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
                collections.abc.Collection: [TTuple],
                collections.abc.Set: [TTuple],
                collections.abc.MappingView: [],
                collections.abc.ItemsView: [TInt, TStr],
                collections.abc.Sized: [],
            },
            "_collections_abc.dict_items",
            [TInt, TStr],
        )

    def test_struct_time(self):
        expected = {
            time.struct_time: [],
            "_typeshed.structseq": [AnyValue(AnySource.explicit) | TypedValue(int)],
            tuple: [SequenceValue(tuple, [(False, TypedValue(int))] * 9)],
            collections.abc.Collection: [TypedValue(int)],
            collections.abc.Reversible: [TypedValue(int)],
            collections.abc.Iterable: [TypedValue(int)],
            collections.abc.Sequence: [TypedValue(int)],
            collections.abc.Container: [AnyValue(AnySource.explicit)],
        }
        self.check(expected, time.struct_time)

    def test_context_manager(self):
        int_tv = TypedValue(int)
        missing = AnyValue(AnySource.generic_argument)
        self.check(
            [
                {contextlib.AbstractContextManager: [int_tv, missing]},
                {contextlib.AbstractContextManager: [int_tv, missing], abc.ABC: []},
            ],
            contextlib.AbstractContextManager,
            [int_tv],
        )
        self.check(
            [
                {contextlib.AbstractAsyncContextManager: [int_tv, missing]},
                {
                    contextlib.AbstractAsyncContextManager: [int_tv, missing],
                    abc.ABC: [],
                },
            ],
            contextlib.AbstractAsyncContextManager,
            [int_tv],
        )

    def test_collections(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        int_str_tuple = make_simple_sequence(tuple, [int_tv, str_tv])
        self.check(
            {
                collections.abc.ValuesView: [int_tv],
                collections.abc.MappingView: [],
                collections.abc.Iterable: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
                collections.abc.Sized: [],
            },
            collections.abc.ValuesView,
            [int_tv],
        )
        self.check(
            {
                collections.abc.ItemsView: [int_tv, str_tv],
                collections.abc.MappingView: [],
                collections.abc.Set: [int_str_tuple],
                collections.abc.Collection: [int_str_tuple],
                collections.abc.Iterable: [int_str_tuple],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
                collections.abc.Sized: [],
            },
            collections.abc.ItemsView,
            [int_tv, str_tv],
        )

        self.check(
            {
                collections.deque: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            },
            collections.deque,
            [int_tv],
        )
        self.check(
            {
                collections.defaultdict: [int_tv, str_tv],
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            },
            collections.defaultdict,
            [int_tv, str_tv],
        )

    def test_typeshed(self):
        int_tv = TypedValue(int)
        str_tv = TypedValue(str)
        self.check(
            {
                list: [int_tv],
                collections.abc.MutableSequence: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Reversible: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Sequence: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            },
            list,
            [int_tv],
        )
        self.check(
            {
                set: [int_tv],
                collections.abc.MutableSet: [int_tv],
                collections.abc.Set: [int_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            },
            set,
            [int_tv],
        )
        self.check(
            {
                dict: [int_tv, str_tv],
                collections.abc.MutableMapping: [int_tv, str_tv],
                collections.abc.Mapping: [int_tv, str_tv],
                collections.abc.Collection: [int_tv],
                collections.abc.Iterable: [int_tv],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
            },
            dict,
            [int_tv, str_tv],
        )

    def test_io(self):
        self.check(
            [
                {
                    io.BytesIO: [],
                    io.BufferedIOBase: [],
                    io.IOBase: [],
                    typing.BinaryIO: [],
                    typing.IO: [TypedValue(bytes)],
                    collections.abc.Iterator: [TypedValue(bytes)],
                    collections.abc.Iterable: [TypedValue(bytes)],
                },
                {
                    io.BytesIO: [],
                    io.BufferedIOBase: [],
                    io.IOBase: [],
                    typing.BinaryIO: [],
                    typing.IO: [TypedValue(bytes)],
                    _io._IOBase: [],
                    _io._BufferedIOBase: [],
                },
            ],
            io.BytesIO,
        )

    def test_parse_result(self):
        self.check(
            {
                collections.abc.Iterable: [AnyValue(AnySource.generic_argument)],
                collections.abc.Reversible: [AnyValue(AnySource.generic_argument)],
                collections.abc.Container: [AnyValue(AnySource.explicit)],
                collections.abc.Collection: [AnyValue(AnySource.generic_argument)],
                collections.abc.Sequence: [AnyValue(AnySource.generic_argument)],
                urllib.parse.ParseResult: [],
                urllib.parse._ParseResultBase: (
                    [TypedValue(str)] if sys.version_info >= (3, 14) else []
                ),
                tuple: [AnyValue(AnySource.generic_argument)],
                urllib.parse._ResultMixinStr: [],
                urllib.parse._NetlocResultMixinBase: [TypedValue(str)],
                urllib.parse._NetlocResultMixinStr: [],
                urllib.parse._ResultMixinStr: [],
            },
            urllib.parse.ParseResult,
        )

    def test_buffered_reader(self):
        self.check(
            [
                {
                    io.IOBase: [],
                    io.BufferedIOBase: [],
                    collections.abc.Iterable: [TypedValue(bytes)],
                    collections.abc.Iterator: [TypedValue(bytes)],
                    io.BufferedReader: [],
                    typing.BinaryIO: [],
                    typing.IO: [TypedValue(bytes)],
                },
                {
                    io.IOBase: [],
                    io.BufferedIOBase: [],
                    _io._BufferedIOBase: [],
                    io.BufferedReader: [AnyValue(AnySource.generic_argument)],
                    typing.BinaryIO: [],
                    typing.IO: [TypedValue(bytes)],
                    _io._IOBase: [],
                },
            ],
            io.BufferedReader,
        )


class TestCheckerGenericBases:
    def test_register_synthetic_type_bases_tracks_direct_synthetic_base(self):
        checker = Checker()
        base = "test.Base"
        child = "test.Child"
        checker.register_synthetic_type_bases(base, [])
        checker.register_synthetic_type_bases(
            child, [SyntheticClassObjectValue("Base", TypedValue(base))]
        )
        assert checker.get_generic_bases(child) == {
            child: TypeVarMap(),
            base: TypeVarMap(),
        }

    def test_register_synthetic_type_bases_tracks_transitive_synthetic_bases(self):
        checker = Checker()
        grandparent = "test.Grandparent"
        parent = "test.Parent"
        child = "test.Child"
        checker.register_synthetic_type_bases(
            parent, [SyntheticClassObjectValue("Grandparent", TypedValue(grandparent))]
        )
        checker.register_synthetic_type_bases(
            child, [SyntheticClassObjectValue("Parent", TypedValue(parent))]
        )
        assert checker.get_generic_bases(child) == {
            child: TypeVarMap(),
            parent: TypeVarMap(),
            grandparent: TypeVarMap(),
        }

    def test_register_synthetic_type_bases_substitutes_declared_type_params(self):
        checker = Checker()
        base = "test.Base"
        child = "test.Child"
        checker.register_synthetic_type_bases(
            base, [], declared_type_params=[TypeVarParam(T)]
        )
        checker.register_synthetic_type_bases(
            child,
            [SyntheticClassObjectValue("Base", GenericValue(base, [TypedValue(int)]))],
        )
        assert checker.get_generic_bases(child) == {
            child: TypeVarMap(),
            base: TypeVarMap(typevars={T: TypedValue(int)}),
        }

    def test_register_synthetic_type_bases_handles_subclass_generic_base(self):
        checker = Checker()
        base = "test.Base"
        child = "test.Child"
        checker.register_synthetic_type_bases(
            base, [], declared_type_params=[TypeVarParam(T)]
        )
        checker.register_synthetic_type_bases(
            child, [SubclassValue(GenericValue(base, [TypedValue(int)]))]
        )
        assert checker.get_generic_bases(child) == {
            child: TypeVarMap(),
            base: TypeVarMap(typevars={T: TypedValue(int)}),
        }


class TestAttribute:
    def test_basic(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert TypedValue(bool) == tsf.get_attribute(
            staticmethod, "__isabstractmethod__", on_class=False
        )

    def test_property(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert TypedValue(int) == tsf.get_attribute(int, "real", on_class=False)

    def test_http_error(self) -> None:
        tsf = TypeshedFinder(Checker(), verbose=True)
        assert True is tsf.has_attribute(HTTPError, "read")


class TestRange(TestNameCheckVisitorBase):
    @assert_passes()
    def test_iteration(self):
        def capybara(r: range):
            for j in r:
                assert_type(j, int)

            for i in range(10000000):
                assert_type(i, int)


class TestParamSpec(TestNameCheckVisitorBase):
    @assert_passes()
    def test_contextmanager(self):
        import contextlib
        from typing import Generator

        def cm(a: int) -> Generator[str, None, None]:
            yield "hello"

        def capybara():
            wrapped = contextlib.contextmanager(cm)
            assert_is_value(
                wrapped(1),
                GenericValue(contextlib._GeneratorContextManager, [TypedValue(str)]),
            )
            wrapped("x")  # E: incompatible_argument


class TestIntegration(TestNameCheckVisitorBase):
    @assert_passes()
    def test_open(self):
        import io
        from typing import IO, Any, BinaryIO

        def capybara(buffering: int, mode: str):
            assert_type(open("x"), io.TextIOWrapper)
            assert_type(open("x", "r"), io.TextIOWrapper)
            assert_type(open("x", "rb"), io.BufferedReader)
            assert_type(open("x", "rb", buffering=0), io.FileIO)
            assert_type(open("x", "rb", buffering=buffering), BinaryIO)
            assert_type(open("x", mode, buffering=buffering), IO[Any])

    @assert_passes()
    def test_os_walk_path_union_filter(self):
        import os
        from collections.abc import Callable
        from pathlib import Path

        def all_files(
            root: str | Path, filter_function: Callable[[str], bool] | None = None
        ) -> set[str]:
            all_files = set()
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if filter_function is not None and not filter_function(filename):
                        continue
                    all_files.add(os.path.join(dirpath, filename))
            return all_files

    @assert_passes()
    def test_os_walk_bytes(self):
        import os
        from collections.abc import Callable

        def all_files(
            root: bytes, filter_function: Callable[[bytes], bool] | None = None
        ) -> set[bytes]:
            all_files = set()
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    if filter_function is not None and not filter_function(filename):
                        continue
                    all_files.add(os.path.join(dirpath, filename))
            return all_files

    @assert_passes()
    def test_itertools_count(self):
        import itertools

        def capybara():
            assert_is_value(
                itertools.count(1), GenericValue(itertools.count, [TypedValue(int)])
            )


class TestNestedClass(TestNameCheckVisitorBase):
    @assert_passes()
    def test_nested(self):
        def capybara() -> None:
            from _pycroscope_tests.nested import Outer

            Outer.Inner(1)

    @assert_passes()
    def test_with_runtime_object(self):
        import sys
        import types

        class Inner:
            def __init__(self, arg: int) -> None:
                pass

        # The bug here only reproduces if a class that exists at runtime contains
        # a nested class in a stub. So we simulate that by creating a fake runtime
        # module.
        Outer = type("Outer", (), {"Inner": Inner})
        Outer.__module__ = "_pycroscope_tests.nested"
        mod = types.ModuleType("_pycroscope_tests.nested")
        mod.Outer = Outer
        sys.modules["_pycroscope_tests.nested"] = mod

        def capybara() -> None:
            Outer.Inner(1)


class TestDeprecated(TestNameCheckVisitorBase):
    @assert_passes()
    def test_utcnow(self):
        import datetime

        def capybara() -> None:
            assert_type(datetime.datetime.utcnow(), datetime.datetime)
