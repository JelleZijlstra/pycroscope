# static analysis: ignore

from pycroscope.checker import Checker
from pycroscope.input_sig import FullSignature, InputSigValue
from pycroscope.relations import Relation, _has_relation_for_generic_arg_pair
from pycroscope.signature import ParameterKind, Signature, SigParameter
from pycroscope.test_name_check_visitor import TestNameCheckVisitorBase
from pycroscope.test_node_visitor import assert_passes, skip_before
from pycroscope.value import CanAssignError, TypedValue


def test_mixed_input_sig_generic_relation_does_not_crash():
    sig = Signature.make(
        [SigParameter("x", ParameterKind.POSITIONAL_ONLY, annotation=TypedValue(int))],
        return_annotation=TypedValue(str),
    )
    input_sig = InputSigValue(FullSignature(sig))
    result = _has_relation_for_generic_arg_pair(
        input_sig, TypedValue(int), Relation.ASSIGNABLE, Checker()
    )
    assert isinstance(result, CanAssignError)


class TestRelations(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_unbounded_tuple_unions(self):
        self.assert_passes("""
            from typing import assert_type

            type Eq0 = tuple[()]
            type Eq1 = tuple[int]
            Ge0 = tuple[int, ...]
            type Ge1 = tuple[int, *Ge0]

            def capybara(eq0: Eq0, eq1: Eq1, ge0: Ge0, ge1: Ge1) -> None:
                eq0_ge1__eq0: Eq0 | Ge1 = eq0
                eq0_ge1__eq1: Eq0 | Ge1 = eq1
                eq0_ge1__ge0: Eq0 | Ge1 = ge0
                eq0_ge1__ge1: Eq0 | Ge1 = ge1
                print(eq0_ge1__eq0, eq0_ge1__eq1, eq0_ge1__ge0, eq0_ge1__ge1)

                assert_type(ge0, Eq0 | Ge1)
            """)

    @assert_passes()
    def test_literal_tuple_equivalent_to_tuple_of_literals(self):
        from typing_extensions import Literal, assert_type

        def capybara(
            literal_tuple: Literal[(("x",),)], tuple_of_literals: tuple[Literal["x"]]
        ) -> None:
            assert_type(literal_tuple, tuple[Literal["x"]])
            assert_type(tuple_of_literals, Literal[(("x",),)])


class TestIntersections(TestNameCheckVisitorBase):
    @assert_passes()
    def test_equivalence(self):
        from typing_extensions import Any, Literal, Never, assert_type

        from pycroscope.extensions import Intersection

        class A:
            x: Any

        class B:
            x: int

        def capybara(
            x: Intersection[Literal[1], Literal[2]], y: Intersection[A, B]
        ) -> None:
            assert_type(x, Never)

            assert_type(y, Intersection[A, B])
            assert_type(y.x, Intersection[int, Any])

    @assert_passes()
    def test_nested(self):
        from typing_extensions import Any, Literal, Never, assert_type

        from pycroscope.extensions import Intersection

        def func() -> None:
            class A:
                x: Any

            class B:
                x: int

            def capybara(
                x: Intersection[Literal[1], Literal[2]], y: Intersection[A, B]
            ) -> None:
                assert_type(x, Never)

                assert_type(y, Intersection[A, B])
                assert_type(y.x, Intersection[int, Any])

    @assert_passes()
    def test_nested_annotation_only_attribute(self):
        from typing_extensions import assert_type

        def func() -> None:
            class A:
                x: int

            a = A()
            assert_type(a.x, int)

    @assert_passes()
    def test_nested_protocol_with_annotation_only_member(self):
        from typing import Protocol

        def func() -> None:
            class P(Protocol):
                x: int

            class Good:
                x: int

            class Bad:
                pass

            def takes_p(arg: P) -> None:
                pass

            takes_p(Good())
            takes_p(Bad())  # E: incompatible_argument

    @assert_passes()
    def test_typed_value_intersections(self):
        from typing_extensions import Never, assert_type, final

        from pycroscope.extensions import Intersection

        class A:
            x: int

        @final
        class B:
            x: str

        def capybara(
            ab: Intersection[A, B],
            int_str: Intersection[int, str],
            a_int: Intersection[A, int],
        ) -> None:
            assert_type(ab, Never)
            assert_type(int_str, Never)
            assert_type(a_int, Never)  # E: inference_failure
