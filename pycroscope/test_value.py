import ast
import collections.abc
import enum
import io
import pickle
import types
import typing
from typing import NewType
from unittest import mock

from typing_extensions import Protocol, runtime_checkable

from pycroscope.test_node_visitor import skip_if_not_installed

from . import tests, value
from .checker import Checker
from .name_check_visitor import NameCheckVisitor
from .predicates import MinLen
from .relations import _extract_type_form, intersect_values
from .signature import ELLIPSIS_PARAM, Signature
from .stacked_scopes import Composite
from .value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    BoundsMap,
    CallableValue,
    CanAssignError,
    ClassSymbol,
    GenericValue,
    IntersectionValue,
    KnownValue,
    KVPair,
    MultiValuedValue,
    NamedTupleInfo,
    OverlapMode,
    OverlappingValue,
    SequenceValue,
    SubclassValue,
    SuperValue,
    SyntheticClassObjectValue,
    TypedValue,
    TypeIsExtension,
    TypeVarParam,
    TypeVarValue,
    Value,
    concrete_values_from_iterable,
    unite_and_simplify,
    unpack_values,
)

_checker = Checker()
CTX = NameCheckVisitor("", "", ast.parse(""), checker=_checker)


def assert_cannot_assign(left: Value, right: Value) -> None:
    tv_map = left.can_assign(right, CTX)
    assert isinstance(tv_map, CanAssignError)


def assert_can_assign(left: Value, right: Value, bounds_map: BoundsMap = {}) -> None:
    can_assign = left.can_assign(right, CTX)
    if isinstance(can_assign, CanAssignError):
        raise AssertionError(str(can_assign))
    assert can_assign == bounds_map


def test_any_value() -> None:
    any = AnyValue(AnySource.unannotated)
    assert not any.is_type(int)
    assert_can_assign(any, KnownValue(1))
    assert_can_assign(any, TypedValue(int))
    assert_can_assign(any, MultiValuedValue([KnownValue(1), TypedValue(int)]))
    assert str(any) == "Any[unannotated]"
    assert str(AnyValue(AnySource.default)) == "Any"


def test_partial_value_fallback() -> None:
    node = ast.parse("list[int]", mode="eval").body
    assert isinstance(node, ast.Subscript)
    partial = value.PartialValue(
        operation=value.PartialValueOperation.SUBSCRIPT,
        root=KnownValue(list),
        node=node.value,
        members=(TypedValue(int),),
        runtime_value=KnownValue(list[int]),
    )

    assert partial.get_fallback_value() == KnownValue(list[int])
    assert_can_assign(TypedValue(types.GenericAlias), partial)
    assert_cannot_assign(TypedValue(int), partial)


def test_partial_unpack_value() -> None:
    node = ast.parse("(*xs,)", mode="eval").body
    assert isinstance(node, ast.Tuple)
    starred = node.elts[0]
    assert isinstance(starred, ast.Starred)
    partial = value.PartialValue(
        operation=value.PartialValueOperation.UNPACK,
        root=TypedValue(tuple),
        node=starred,
        members=(),
        runtime_value=AnyValue(AnySource.inference),
    )

    assert partial.get_fallback_value() == AnyValue(AnySource.inference)
    assert str(partial) == "Any[inference] (partial from *tuple)"


def test_partial_bitor_value() -> None:
    node = ast.parse("int | str", mode="eval").body
    assert isinstance(node, ast.BinOp)
    partial = value.PartialValue(
        operation=value.PartialValueOperation.BITOR,
        root=TypedValue(int),
        node=node,
        members=(TypedValue(str),),
        runtime_value=KnownValue(int | str),
    )

    assert partial.get_fallback_value() == KnownValue(int | str)
    assert str(partial) == "Literal[int | str] (partial from int | str)"


def test_extract_type_form_from_partial_value() -> None:
    from typing_extensions import TypeForm

    node = ast.parse("TypeForm[int]", mode="eval").body
    assert isinstance(node, ast.Subscript)
    partial = value.PartialValue(
        operation=value.PartialValueOperation.SUBSCRIPT,
        root=KnownValue(TypeForm),
        node=node.value,
        members=(KnownValue(int),),
        runtime_value=AnyValue(AnySource.inference),
    )
    extracted = _extract_type_form(partial, CTX)
    assert extracted == TypedValue(int)


def test_known_value() -> None:
    val = KnownValue(3)
    assert 3 == val.val
    assert str(val) == "Literal[3]"
    assert str(KnownValue("")) == "Literal['']"
    assert str(KnownValue(None)) == "None"
    assert str(KnownValue(str)) == "type 'str'"
    assert str(KnownValue(KnownValue)) == "type 'pycroscope.value.KnownValue'"
    assert (
        str(KnownValue(test_known_value))
        == "function 'pycroscope.test_value.test_known_value'"
    )
    assert str(KnownValue(ast)) == "module 'ast'"
    assert val.is_type(int)
    assert not val.is_type(str)

    assert_cannot_assign(val, KnownValue(1))
    assert_can_assign(val, val)
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, MultiValuedValue([val, AnyValue(AnySource.marker)]))
    assert_cannot_assign(val, val | TypedValue(int))
    assert_cannot_assign(KnownValue(int), SubclassValue(TypedValue(int)))
    assert_cannot_assign(KnownValue(1), KnownValue(True))
    assert_cannot_assign(KnownValue(True), KnownValue(1))

    nan = float("nan")
    assert_can_assign(KnownValue(nan), KnownValue(nan))
    assert_cannot_assign(KnownValue(nan), KnownValue(0.0))


def test_none_and_nonetype_are_equivalent() -> None:
    assert_can_assign(TypedValue(types.NoneType), KnownValue(None))
    assert_can_assign(KnownValue(None), TypedValue(types.NoneType))


@skip_if_not_installed("asynq")
def test_unbound_method_value() -> None:
    from pycroscope.asynq_tests import PropertyObject

    po_composite = Composite(value.TypedValue(PropertyObject))
    val = value.UnboundMethodValue("get_prop_with_get", po_composite)
    assert "<method get_prop_with_get on pycroscope.asynq_tests.PropertyObject>" == str(
        val
    )
    assert "get_prop_with_get" == val.attr_name
    assert TypedValue(PropertyObject) == val.composite.value
    assert None is val.secondary_attr_name
    assert PropertyObject.get_prop_with_get == val.get_method()
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue(
        "get_prop_with_get", po_composite, secondary_attr_name="asynq"
    )
    assert (
        "<method get_prop_with_get.asynq on pycroscope.asynq_tests.PropertyObject>"
        == str(val)
    )
    assert "get_prop_with_get" == val.attr_name
    assert TypedValue(PropertyObject) == val.composite.value
    assert "asynq" == val.secondary_attr_name
    method = val.get_method()
    assert method is not None
    assert method.__name__ in tests.ASYNQ_METHOD_NAMES
    assert PropertyObject.get_prop_with_get == method.__self__
    assert val.is_type(object)
    assert not val.is_type(str)

    val = value.UnboundMethodValue("non_async_method", po_composite)
    assert val.get_method() is not None
    assert val.get_signature(CTX) is not None
    assert_can_assign(val, val)
    assert_cannot_assign(val, KnownValue(1))
    assert_can_assign(val, CallableValue(Signature.make([ELLIPSIS_PARAM])))
    assert_can_assign(val, CallableValue(Signature.make([])))


def test_typed_value() -> None:
    val = TypedValue(str)
    assert val.typ is str
    assert str(val) == "str"
    assert val.is_type(str)
    assert not val.is_type(int)

    assert_can_assign(val, val)
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue("x"))
    assert_can_assign(val, MultiValuedValue([val, KnownValue("x")]))
    assert_cannot_assign(val, MultiValuedValue([KnownValue("x"), TypedValue(int)]))

    literal_string = TypedValue(str, literal_only=True)
    assert literal_string.typ is str
    assert str(literal_string) == "LiteralString"
    assert_can_assign(val, literal_string)
    assert_cannot_assign(literal_string, val)
    assert_can_assign(literal_string, KnownValue("x"))

    float_val = TypedValue(float)
    assert str(float_val) == "float"
    assert_can_assign(float_val, KnownValue(1.0))
    assert_cannot_assign(float_val, KnownValue(1))
    assert_cannot_assign(float_val, KnownValue(""))
    assert_can_assign(float_val, TypedValue(float))
    assert_cannot_assign(float_val, TypedValue(int))
    assert_cannot_assign(float_val, TypedValue(str))
    assert_can_assign(float_val, TypedValue(mock.Mock))

    assert_cannot_assign(float_val, SubclassValue(TypedValue(float)))
    assert_can_assign(TypedValue(type), SubclassValue(TypedValue(float)))


def test_typed_value_type_object_cache_is_context_local() -> None:
    shared = TypedValue("pkg.A")
    checker_with_base = Checker()
    checker_with_base.register_synthetic_type_bases("pkg.A", [TypedValue(int)])
    checker_without_base = Checker()

    assert shared.get_type_object(checker_with_base).is_assignable_to_type(int)
    assert not shared.get_type_object(checker_without_base).is_assignable_to_type(int)


def test_get_generic_args_for_type_with_super() -> None:
    T = typing.TypeVar("T")

    class Base(typing.Generic[T]):
        pass

    class Child(Base[int]):
        pass

    super_value = SuperValue(KnownValue(Child), TypedValue(Child))
    assert super_value.get_fallback_value().get_generic_args_for_type(Base, CTX) is None


@runtime_checkable
class Proto(Protocol):
    def asynq(self) -> None: ...


def test_protocol() -> None:
    tv = TypedValue(Proto)

    def fn_missing_asynq() -> None:
        pass

    assert_cannot_assign(tv, KnownValue(fn_missing_asynq))

    def fn_with_asynq() -> None:
        pass

    fn_with_asynq.asynq = lambda: None
    assert_can_assign(tv, KnownValue(fn_with_asynq))

    class X:
        def asynq(self) -> None:
            pass

    assert_can_assign(tv, TypedValue(X))
    assert_can_assign(tv, KnownValue(X()))


def test_callable() -> None:
    cval = TypedValue(collections.abc.Callable)
    assert_can_assign(cval, cval)

    gen_val = GenericValue(
        collections.abc.Callable, [TypedValue(int), KnownValue(None)]
    )
    assert_can_assign(gen_val, gen_val)


def test_subclass_value() -> None:
    val = SubclassValue(TypedValue(int))
    assert_can_assign(val, KnownValue(int))
    assert_can_assign(val, KnownValue(bool))
    assert_cannot_assign(val, KnownValue(str))
    assert_can_assign(val, TypedValue(type))
    assert_cannot_assign(val, TypedValue(int))
    assert_can_assign(val, SubclassValue(TypedValue(bool)))
    assert_can_assign(val, TypedValue(type))
    assert_cannot_assign(val, SubclassValue(TypedValue(str)))
    val = SubclassValue(TypedValue(str))
    assert str(val) == "type[str]"
    assert TypedValue(str) == val.typ
    assert val.is_type(str)
    assert not val.is_type(int)
    val = SubclassValue(TypedValue(float)) | SubclassValue(TypedValue(int))
    assert_can_assign(val, KnownValue(int))
    assert_can_assign(val, SubclassValue(TypedValue(int)))


def test_subclass_value_make_invalid_literal() -> None:
    assert SubclassValue.make(KnownValue(1)) == AnyValue(AnySource.error)


def test_generic_value() -> None:
    val = GenericValue(list, [TypedValue(int)])
    assert "list[int]" == str(val)
    assert_can_assign(val, TypedValue(list))
    assert_can_assign(val, GenericValue(list, [AnyValue(AnySource.marker)]))
    assert_cannot_assign(val, GenericValue(list, [TypedValue(bool)]))
    assert_cannot_assign(val, GenericValue(list, [TypedValue(str)]))
    assert_cannot_assign(val, GenericValue(set, [TypedValue(int)]))
    assert "tuple[int, ...]" == str(value.GenericValue(tuple, [TypedValue(int)]))

    it = GenericValue(collections.abc.Iterable, [TypedValue(object)])
    assert_can_assign(
        it, GenericValue(types.GeneratorType, [TypedValue(bool) | KnownValue(None)])
    )


def test_sequence_value() -> None:
    val = value.SequenceValue(
        tuple, [(False, TypedValue(int)), (False, TypedValue(str))]
    )
    assert_can_assign(val, TypedValue(tuple))
    assert_can_assign(GenericValue(tuple, [TypedValue(int) | TypedValue(str)]), val)
    assert_cannot_assign(val, GenericValue(tuple, [TypedValue(int) | TypedValue(str)]))
    assert_cannot_assign(val, GenericValue(tuple, [TypedValue(int) | TypedValue(list)]))

    assert_can_assign(val, val)
    assert_cannot_assign(val, value.SequenceValue(tuple, [(False, TypedValue(int))]))
    assert_can_assign(
        val,
        value.SequenceValue(
            tuple, [(False, TypedValue(bool)), (False, TypedValue(str))]
        ),
    )

    assert_can_assign(val, KnownValue((1, "x")))
    assert_cannot_assign(val, KnownValue((1, 2)))
    assert_cannot_assign(val, KnownValue((1, "x", "y")))

    assert str(val) == "tuple[int, str]"
    assert str(value.SequenceValue(tuple, [(False, TypedValue(int))])) == "tuple[int]"
    assert (
        str(
            value.SequenceValue(
                tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
            )
        )
        == "tuple[int, *tuple[str, ...]]"
    )
    assert (
        str(value.SequenceValue(list, [(False, TypedValue(int))]))
        == "<list containing [int]>"
    )


def test_sequence_value_unpack() -> None:
    fmt_map = {"i": int, "s": str, "b": bool, "o": object}

    def s(fmt: str) -> SequenceValue:
        members = []
        for c in fmt:
            is_many = c.isupper()
            members.append((is_many, TypedValue(fmt_map[c.lower()])))
        return SequenceValue(tuple, members)

    # left is empty
    assert_can_assign(s(""), s(""))
    assert_cannot_assign(s(""), s("i"))

    # only single values
    assert_can_assign(s("i"), s("i"))
    assert_cannot_assign(s("i"), s("ii"))
    assert_cannot_assign(s("ii"), s("i"))
    assert_can_assign(s("o"), s("i"))

    # left is one many
    assert_can_assign(s("I"), s("i"))
    assert_can_assign(s("I"), s("ii"))
    assert_can_assign(s("I"), s("iI"))
    assert_can_assign(s("I"), s("iIii"))
    assert_can_assign(s("I"), s("bIb"))
    assert_cannot_assign(s("I"), s("o"))

    # prefix on the left
    assert_can_assign(s("iI"), s("i"))
    assert_can_assign(s("oI"), s("iB"))
    assert_cannot_assign(s("iI"), s("I"))
    assert_can_assign(s("sI"), s("siIi"))
    assert_can_assign(s("Ii"), s("iI"))

    # suffix on the right
    assert_can_assign(s("Ii"), s("i"))
    assert_can_assign(s("Ii"), s("Ii"))
    assert_can_assign(s("Ii"), s("iIi"))
    assert_can_assign(s("Ii"), s("iI"))

    assert_cannot_assign(s("iIi"), s("i"))

    # this fails
    assert_can_assign(s("iIsIi"), s("IisiI"))
    # TODO
    # assert_can_assign(s("IisiI"), s("iIsIi"))


def test_dict_incomplete_value() -> None:
    val = value.DictIncompleteValue(dict, [KVPair(TypedValue(int), KnownValue("x"))])
    assert "<dict containing {int: Literal['x']}>" == str(val)

    val = value.DictIncompleteValue(
        dict,
        [
            KVPair(KnownValue("a"), TypedValue(int)),
            KVPair(KnownValue("b"), TypedValue(str)),
        ],
    )
    assert val.get_value(KnownValue("a"), CTX) == TypedValue(int)


def test_multi_valued_value() -> None:
    val = TypedValue(int) | KnownValue(None)
    assert MultiValuedValue([TypedValue(int), KnownValue(None)]) == val
    assert val == val | KnownValue(None)
    assert MultiValuedValue(
        [TypedValue(int), KnownValue(None), TypedValue(str)]
    ) == val | TypedValue(str)

    assert str(val) == "int | None"
    assert_can_assign(val, KnownValue(1))
    assert_can_assign(val, KnownValue(None))
    assert_cannot_assign(val, KnownValue(""))
    assert_cannot_assign(val, TypedValue(float))
    assert_can_assign(val, val)
    assert_cannot_assign(val, KnownValue(None) | TypedValue(str))
    assert_can_assign(
        val, AnyValue(AnySource.marker) | TypedValue(int) | KnownValue(None)
    )

    assert str(KnownValue(1) | KnownValue(2)) == "Literal[1, 2]"
    assert (
        str(KnownValue(1) | KnownValue(2) | KnownValue(None)) == "Literal[1, 2, None]"
    )
    assert str(TypedValue(int) | TypedValue(str)) == "int | str"
    assert (
        str(TypedValue(int) | TypedValue(str) | KnownValue(None)) == "int | str | None"
    )
    assert (
        str(
            TypedValue(int)
            | TypedValue(str)
            | KnownValue(None)
            | KnownValue(1)
            | KnownValue(2)
        )
        == "int | str | Literal[1, 2] | None"
    )


def test_large_union_optimization() -> None:
    val = MultiValuedValue([*[KnownValue(i) for i in range(10000)], TypedValue(str)])
    assert_can_assign(val, KnownValue(1))
    assert_cannot_assign(val, KnownValue(234234))
    assert_cannot_assign(val, KnownValue(True))
    assert_can_assign(val, KnownValue(""))


class ThriftEnum(object):
    X = 0
    Y = 1

    _VALUES_TO_NAMES = {0: "X", 1: "Y"}

    _NAMES_TO_VALUES = {"X": 0, "Y": 1}


def test_can_assign_thrift_enum() -> None:
    val = TypedValue(ThriftEnum)
    assert_can_assign(val, KnownValue(0))
    assert_cannot_assign(val, KnownValue(2))
    assert_cannot_assign(val, KnownValue(1.0))

    assert_can_assign(val, TypedValue(int))
    assert_can_assign(val, TypedValue(ThriftEnum))
    assert_cannot_assign(val, TypedValue(str))


def test_variable_name_value() -> None:
    uid_val = value.VariableNameValue(["uid", "viewer"])
    varname_map = {
        "uid": uid_val,
        "viewer": uid_val,
        "actor_id": value.VariableNameValue(["actor_id"]),
    }

    assert None is value.VariableNameValue.from_varname("capybaras", varname_map)

    val = value.VariableNameValue.from_varname("uid", varname_map)
    assert None is not val
    assert val is value.VariableNameValue.from_varname("viewer", varname_map)
    assert val is value.VariableNameValue.from_varname("old_uid", varname_map)
    assert val is not value.VariableNameValue.from_varname("actor_id", varname_map)
    assert_can_assign(TypedValue(int), val)
    assert_can_assign(KnownValue(1), val)
    assert_can_assign(val, TypedValue(int))
    assert_can_assign(val, KnownValue(1))


def test_typeddict_value() -> None:
    val = value.TypedDictValue(
        {
            "a": value.TypedDictEntry(TypedValue(int)),
            "b": value.TypedDictEntry(TypedValue(str)),
        }
    )
    # dict iteration order in some Python versions is not deterministic
    assert str(val) in [
        'TypedDict({"a": int, "b": str})',
        'TypedDict({"b": str, "a": int})',
    ]

    assert_can_assign(val, AnyValue(AnySource.marker))
    assert_cannot_assign(val, TypedValue(dict))
    assert_cannot_assign(val, TypedValue(str))

    # KnownValue of dict
    assert_can_assign(val, KnownValue({"a": 1, "b": "2"}))
    # extra keys in a dict literal are rejected for TypedDict construction
    assert_cannot_assign(val, KnownValue({"a": 1, "b": "2", "c": 1}))
    # missing key
    assert_cannot_assign(val, KnownValue({"a": 1}))
    # wrong type
    assert_cannot_assign(val, KnownValue({"a": 1, "b": 2}))

    # TypedDictValue
    assert_can_assign(val, val)
    assert_can_assign(
        val,
        value.TypedDictValue(
            {
                "a": value.TypedDictEntry(TypedValue(int)),
                "b": value.TypedDictEntry(TypedValue(str)),
                "c": value.TypedDictEntry(TypedValue(float)),
            }
        ),
    )
    assert_cannot_assign(
        val,
        value.TypedDictValue(
            {
                "a": value.TypedDictEntry(KnownValue(1)),
                "b": value.TypedDictEntry(TypedValue(str)),
            }
        ),
    )
    assert_cannot_assign(
        val,
        value.TypedDictValue(
            {
                "a": value.TypedDictEntry(KnownValue(1)),
                "b": value.TypedDictEntry(TypedValue(str)),
                "c": value.TypedDictEntry(TypedValue(float)),
            }
        ),
    )
    assert_cannot_assign(
        val,
        value.TypedDictValue(
            {
                "a": value.TypedDictEntry(KnownValue(1)),
                "b": value.TypedDictEntry(TypedValue(int)),
            }
        ),
    )
    assert_cannot_assign(
        val, value.TypedDictValue({"b": value.TypedDictEntry(TypedValue(str))})
    )

    # DictIncompleteValue
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_can_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(str)),
                KVPair(KnownValue("a"), TypedValue(int), is_required=False),
                KVPair(KnownValue("b"), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(str)),
                KVPair(KnownValue("c"), AnyValue(AnySource.marker)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(AnyValue(AnySource.marker), TypedValue(str)),
            ],
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict, [KVPair(AnyValue(AnySource.marker), TypedValue(str))]
        ),
    )
    assert_cannot_assign(
        val,
        value.DictIncompleteValue(
            dict,
            [
                KVPair(KnownValue("a"), TypedValue(int)),
                KVPair(KnownValue("b"), TypedValue(float)),
            ],
        ),
    )


def test_synthetic_class_object_value_typeddict() -> None:
    movie_td = value.TypedDictValue(
        {
            "name": value.TypedDictEntry(TypedValue(str)),
            "year": value.TypedDictEntry(TypedValue(int)),
        }
    )
    movie_cls = value.SyntheticClassObjectValue("Movie", movie_td)
    other_movie_cls = value.SyntheticClassObjectValue("OtherMovie", movie_td)

    assert str(movie_cls) == "<class 'Movie'>"

    # Synthetic TypedDict class values are singleton-like.
    assert_can_assign(movie_cls, movie_cls)
    assert_cannot_assign(movie_cls, other_movie_cls)
    assert_cannot_assign(movie_cls, TypedValue(type))

    # But like other class objects, they can flow to plain type.
    assert_can_assign(TypedValue(type), movie_cls)
    assert_cannot_assign(SubclassValue(TypedValue(dict)), movie_cls)

    # Intersections should behave like other singleton value intersections.
    assert intersect_values(movie_cls, movie_cls, CTX) == movie_cls
    assert intersect_values(movie_cls, other_movie_cls, CTX) == NO_RETURN_VALUE
    assert intersect_values(movie_cls, TypedValue(type), CTX) == movie_cls
    assert intersect_values(movie_cls, TypedValue(int), CTX) == NO_RETURN_VALUE


def test_synthetic_class_object_value_nominal_class() -> None:
    int_cls = value.SyntheticClassObjectValue("int", TypedValue(int))
    other_int_cls = value.SyntheticClassObjectValue("other_int", TypedValue(int))

    assert str(int_cls) == "<class 'int'>"
    assert_can_assign(int_cls, int_cls)
    assert_cannot_assign(int_cls, other_int_cls)

    # Exact class object should satisfy type[int] and plain type.
    assert_can_assign(SubclassValue(TypedValue(int)), int_cls)
    assert_can_assign(TypedValue(type), int_cls)
    assert_cannot_assign(SubclassValue(TypedValue(str)), int_cls)


def test_synthetic_class_object_value_matches_specialized_type() -> None:
    synthetic_cls = value.SyntheticClassObjectValue("C", TypedValue("mod.C"))
    specialized = SubclassValue(GenericValue("mod.C", [TypedValue(int)]))

    # Unspecialized class objects should still match specialized type[...] forms.
    assert_can_assign(specialized, synthetic_cls)
    assert_can_assign(synthetic_cls, specialized)


def test_synthetic_class_object_value_unresolved_nominal_class() -> None:
    unresolved_cls = value.SyntheticClassObjectValue("X", TypedValue("mod.X"))
    other_cls = value.SyntheticClassObjectValue("Y", TypedValue("mod.Y"))

    # Synthetic class objects with unresolved class names should still behave
    # like class objects for assignability checks.
    assert_can_assign(TypedValue(type), unresolved_cls)
    assert_can_assign(SubclassValue(TypedValue("mod.X")), unresolved_cls)
    assert_cannot_assign(SubclassValue(TypedValue("mod.Y")), unresolved_cls)
    assert_cannot_assign(unresolved_cls, other_cls)


def test_synthetic_namedtuple_members_without_runtime_class() -> None:
    checker = Checker()
    point = SyntheticClassObjectValue(
        "Point",
        TypedValue("mod.Point"),
        base_classes=(TypedValue(tuple),),
        namedtuple_info=NamedTupleInfo(
            field_names=("x", "label"),
            default_fields=("label",),
            has_namedtuple_marker_base=True,
        ),
    )
    point.declared_symbols["x"] = ClassSymbol(
        annotation=TypedValue(int), is_instance_only=True, initializer=TypedValue(int)
    )
    point.declared_symbols["label"] = ClassSymbol(
        annotation=TypedValue(str), is_instance_only=True, initializer=TypedValue(str)
    )
    checker.register_synthetic_class(point)

    assert value.ordered_namedtuple_fields_from_synthetic(point) == ("x", "label")
    assert value.tuple_members_from_value(TypedValue("mod.Point"), checker) == (
        (False, TypedValue(int)),
        (False, TypedValue(str)),
    )


def test_exact_tuple_subclass_members_from_runtime_bases() -> None:
    class NotANT(tuple[int, str]):
        pass

    checker = Checker()
    assert value.tuple_members_from_value(TypedValue(NotANT), checker) == (
        (False, TypedValue(int)),
        (False, TypedValue(str)),
    )


def test_tuple_subclass_with_custom_iter_is_not_unpacked_as_exact_tuple() -> None:
    class Weird(tuple[int, str]):
        pass

    def _iter(self: Weird) -> collections.abc.Iterator[int | str]:
        yield self[1]
        yield self[0]

    Weird.__iter__ = _iter

    checker = Checker()
    assert not value._can_unpack_tuple_members_from_value(TypedValue(Weird), checker)


class Capybara(enum.IntEnum):
    hydrochaeris = 1
    isthmius = 2


def test_new_type_value() -> None:
    nt1 = NewType("nt1", int)
    nt1_val = value.NewTypeValue("nt1", TypedValue(int), nt1)
    nt2 = NewType("nt2", int)
    nt2_val = value.NewTypeValue("nt2", TypedValue(int), nt2)
    assert_can_assign(nt1_val, nt1_val)
    assert_cannot_assign(nt1_val, nt2_val)
    # This should eventually return False
    assert_cannot_assign(nt1_val, TypedValue(int))
    assert_can_assign(TypedValue(int), nt1_val)
    assert_cannot_assign(nt1_val, TypedValue(Capybara))
    assert_cannot_assign(nt1_val, KnownValue(Capybara.hydrochaeris))

    assert_can_assign(nt1_val, AnyValue(AnySource.marker))
    assert_can_assign(AnyValue(AnySource.marker), nt1_val)


def test_annotated_value() -> None:
    tv_int = TypedValue(int)
    ext = TypeIsExtension(tv_int)
    assert_cannot_assign(AnnotatedValue(tv_int, [ext]), tv_int)
    assert_can_assign(tv_int, AnnotatedValue(tv_int, [ext]))

    union = TypedValue(int) | TypedValue(float)
    annotated = AnnotatedValue(union, [ext])
    assert_cannot_assign(annotated, union)
    assert_can_assign(union, annotated)


class A:
    pass


class B(A):
    pass


class C(A):
    pass


class D(B, C):
    pass


def test_intersection_value() -> None:
    val = TypedValue(int) & TypedValue(str)
    assert str(val) == "int & str"

    assert_can_assign(val, val)
    assert_can_assign(TypedValue(object), val)

    val = TypedValue(B) & TypedValue(C)
    assert str(val) == "pycroscope.test_value.B & pycroscope.test_value.C"
    assert_can_assign(val, val)
    assert_can_assign(TypedValue(A), val)
    assert_cannot_assign(val, TypedValue(A))

    assert_cannot_assign(TypedValue(D), val)
    assert_can_assign(val, TypedValue(D))

    assert_can_assign(TypedValue(B), val)
    assert_cannot_assign(val, TypedValue(B))

    assert_can_assign(TypedValue(B) | TypedValue(C), val)
    assert_cannot_assign(val, TypedValue(B) | TypedValue(C))

    never = KnownValue(1) & KnownValue(2)
    assert_can_assign(never, never)
    assert_can_assign(NO_RETURN_VALUE, never)
    assert_can_assign(never, NO_RETURN_VALUE)


def test_unannotated_any_intersection_simplifies() -> None:
    unannotated = AnyValue(AnySource.unannotated)
    explicit = AnyValue(AnySource.explicit)
    min_len_1 = value.PredicateValue(MinLen(1))

    assert intersect_values(unannotated, TypedValue(int), CTX) == TypedValue(int)
    assert intersect_values(TypedValue(int), unannotated, CTX) == TypedValue(int)

    assert intersect_values(unannotated, min_len_1, CTX) == value.IntersectionValue(
        (unannotated, min_len_1)
    )
    assert intersect_values(min_len_1, unannotated, CTX) == value.IntersectionValue(
        (min_len_1, unannotated)
    )

    assert intersect_values(explicit, TypedValue(int), CTX) == value.IntersectionValue(
        (explicit, TypedValue(int))
    )
    assert intersect_values(TypedValue(int), explicit, CTX) == value.IntersectionValue(
        (TypedValue(int), explicit)
    )


def test_predicate_intersection() -> None:
    pred = value.PredicateValue(MinLen(1))
    inters = IntersectionValue((TypedValue(str), pred))
    assert_can_assign(TypedValue(str), inters)
    assert_can_assign(pred, inters)


def test_overlapping_value_intersection_simplifies() -> None:
    overlapping_int = OverlappingValue(TypedValue(int))

    assert intersect_values(overlapping_int, TypedValue(int), CTX) == TypedValue(int)
    assert intersect_values(TypedValue(int), overlapping_int, CTX) == TypedValue(int)
    assert intersect_values(overlapping_int, TypedValue(object), CTX) == overlapping_int
    assert intersect_values(TypedValue(object), overlapping_int, CTX) == overlapping_int

    assert intersect_values(
        overlapping_int, TypedValue(str), CTX
    ) == overlapping_int & TypedValue(str)
    assert (
        intersect_values(TypedValue(str), overlapping_int, CTX)
        == TypedValue(str) & overlapping_int
    )

    overlapping_int_or_str = OverlappingValue(TypedValue(int) | TypedValue(str))
    int_or_bytes = TypedValue(int) | TypedValue(bytes)
    assert intersect_values(overlapping_int_or_str, int_or_bytes, CTX) == (
        TypedValue(int) | (TypedValue(bytes) & overlapping_int_or_str)
    )
    assert intersect_values(int_or_bytes, overlapping_int_or_str, CTX) == (
        TypedValue(int) | (TypedValue(bytes) & overlapping_int_or_str)
    )


def test_typeform_intersection_simplification() -> None:
    typeform_int = value.TypeFormValue(TypedValue(int))
    tf_and_int = intersect_values(typeform_int, TypedValue(int), CTX)
    int_and_tf = intersect_values(TypedValue(int), typeform_int, CTX)

    assert tf_and_int == value.IntersectionValue((typeform_int, TypedValue(int)))
    assert int_and_tf == value.IntersectionValue((TypedValue(int), typeform_int))

    tf_and_obj = intersect_values(typeform_int, TypedValue(object), CTX)
    obj_and_tf = intersect_values(TypedValue(object), typeform_int, CTX)
    assert tf_and_obj == typeform_int
    assert obj_and_tf == typeform_int


def test_typevar_intersection_preserves_typevar() -> None:
    typevar_value = TypeVarValue(TypeVarParam(typing.TypeVar("T")))
    narrowed_int = intersect_values(typevar_value, TypedValue(int), CTX)
    narrowed_str = intersect_values(typevar_value, TypedValue(str), CTX)

    assert intersect_values(typevar_value, TypedValue(object), CTX) == typevar_value
    assert narrowed_int == (value.IntersectionValue((typevar_value, TypedValue(int))))
    assert_can_assign(typevar_value, narrowed_int)
    assert_can_assign(typevar_value, MultiValuedValue([narrowed_int, narrowed_str]))


def test_constrained_typevar_intersection_simplifies() -> None:
    anystr_value = TypeVarValue(
        TypeVarParam(
            typing.TypeVar("AnyStr", str, bytes),
            constraints=(TypedValue(str), TypedValue(bytes)),
        )
    )

    class StrSub(str):
        pass

    assert intersect_values(anystr_value, TypedValue(StrSub), CTX) == TypedValue(StrSub)


def test_io() -> None:
    assert_can_assign(
        GenericValue(typing.IO, [AnyValue(AnySource.marker)]), TypedValue(io.BytesIO)
    )


def test_concrete_values_from_iterable() -> None:
    assert isinstance(concrete_values_from_iterable(KnownValue(1), CTX), CanAssignError)
    assert concrete_values_from_iterable(KnownValue(()), CTX) == []
    assert concrete_values_from_iterable(KnownValue((1, 2)), CTX) == [
        KnownValue(1),
        KnownValue(2),
    ]
    assert concrete_values_from_iterable(
        tests.make_simple_sequence(list, [KnownValue(1), KnownValue(2)]), CTX
    ) == [KnownValue(1), KnownValue(2)]
    assert TypedValue(int) == concrete_values_from_iterable(
        GenericValue(list, [TypedValue(int)]), CTX
    )
    assert [
        KnownValue(1) | KnownValue(3),
        KnownValue(2) | KnownValue(4),
    ] == concrete_values_from_iterable(
        MultiValuedValue(
            [
                tests.make_simple_sequence(list, [KnownValue(1), KnownValue(2)]),
                KnownValue((3, 4)),
            ]
        ),
        CTX,
    )
    assert MultiValuedValue(
        [KnownValue(1), KnownValue(2), TypedValue(int)]
    ) == concrete_values_from_iterable(
        MultiValuedValue(
            [
                tests.make_simple_sequence(list, [KnownValue(1), KnownValue(2)]),
                GenericValue(list, [TypedValue(int)]),
            ]
        ),
        CTX,
    )
    assert MultiValuedValue(
        [KnownValue(1), KnownValue(2), KnownValue(3)]
    ) == concrete_values_from_iterable(
        MultiValuedValue(
            [
                tests.make_simple_sequence(list, [KnownValue(1), KnownValue(2)]),
                KnownValue((3,)),
            ]
        ),
        CTX,
    )

    class HasGetItem:
        def __getitem__(self, some_random_name: int) -> str:
            return str(some_random_name)

    assert concrete_values_from_iterable(TypedValue(HasGetItem), CTX) == TypedValue(str)

    class BadGetItem:
        def __getitem__(self, i: int, extra: bool) -> str:
            return str(i) + str(extra)

    assert isinstance(
        concrete_values_from_iterable(TypedValue(BadGetItem), CTX), CanAssignError
    )


def _assert_pickling_roundtrip(obj: object) -> None:
    assert obj == pickle.loads(pickle.dumps(obj))


def test_pickling() -> None:
    _assert_pickling_roundtrip(KnownValue(1))
    _assert_pickling_roundtrip(TypedValue(int))
    _assert_pickling_roundtrip(KnownValue(None) | TypedValue(str))


def test_unite_and_simplify() -> None:
    vals = [GenericValue(list, [TypedValue(int)]), KnownValue([])]
    assert unite_and_simplify(*vals, limit=2) == GenericValue(
        list, [TypedValue(int)]
    ) | GenericValue(list, [AnyValue(AnySource.unreachable)])


def test_unpack_values() -> None:
    t_int = SequenceValue(tuple, [(False, TypedValue(int))])
    assert unpack_values(t_int, CTX, 1, None) == [TypedValue(int)]
    assert unpack_values(t_int, CTX, 1, 0) == [TypedValue(int), SequenceValue(list, [])]
    assert isinstance(unpack_values(t_int, CTX, 1, 1), CanAssignError)
    assert isinstance(unpack_values(t_int, CTX, 2, None), CanAssignError)

    t_int_str = SequenceValue(
        tuple, [(False, TypedValue(int)), (False, TypedValue(str))]
    )
    assert isinstance(unpack_values(t_int_str, CTX, 1, None), CanAssignError)
    assert unpack_values(t_int_str, CTX, 2, None) == [TypedValue(int), TypedValue(str)]
    assert unpack_values(t_int_str, CTX, 2, 0) == [
        TypedValue(int),
        TypedValue(str),
        SequenceValue(list, []),
    ]
    assert unpack_values(t_int_str, CTX, 1, 1) == [
        TypedValue(int),
        SequenceValue(list, []),
        TypedValue(str),
    ]

    t_int_star_str = SequenceValue(
        tuple, [(False, TypedValue(int)), (True, TypedValue(str))]
    )
    assert unpack_values(t_int_star_str, CTX, 1, None) == [TypedValue(int)]
    assert unpack_values(t_int_star_str, CTX, 1, 0) == [
        TypedValue(int),
        SequenceValue(list, [(True, TypedValue(str))]),
    ]
    assert unpack_values(t_int_star_str, CTX, 1, 1) == [
        TypedValue(int),
        GenericValue(list, [TypedValue(str)]),
        TypedValue(str),
    ]
    assert unpack_values(t_int_star_str, CTX, 2, None) == [
        TypedValue(int),
        TypedValue(str),
    ]

    t_int_star_str_float = SequenceValue(
        tuple,
        [(False, TypedValue(int)), (True, TypedValue(str)), (False, TypedValue(float))],
    )
    assert isinstance(unpack_values(t_int_star_str_float, CTX, 1, None), CanAssignError)
    assert unpack_values(t_int_star_str_float, CTX, 2, None) == [
        TypedValue(int),
        TypedValue(float),
    ]
    assert unpack_values(t_int_star_str_float, CTX, 1, 1) == [
        TypedValue(int),
        SequenceValue(list, [(True, TypedValue(str))]),
        TypedValue(float),
    ]
    assert unpack_values(t_int_star_str_float, CTX, 0, 2) == [
        GenericValue(list, [TypedValue(str) | TypedValue(int)]),
        TypedValue(int) | TypedValue(str),
        TypedValue(float),
    ]


def test_can_overlap() -> None:
    assert isinstance(
        TypedValue(str).can_overlap(TypedValue(int), CTX, OverlapMode.EQ),
        CanAssignError,
    )
    assert isinstance(
        TypedValue(str).can_overlap(KnownValue(1), CTX, OverlapMode.EQ), CanAssignError
    )
    assert isinstance(
        TypedValue(str).can_overlap(TypedValue(type(None)), CTX, OverlapMode.EQ),
        CanAssignError,
    )
    assert isinstance(
        TypedValue(str).can_overlap(KnownValue(None), CTX, OverlapMode.EQ),
        CanAssignError,
    )
