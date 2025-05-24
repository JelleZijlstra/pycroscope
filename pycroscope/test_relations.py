# static analysis: ignore

from pycroscope.test_name_check_visitor import TestNameCheckVisitorBase
from pycroscope.test_node_visitor import skip_before


class TestRelations(TestNameCheckVisitorBase):
    @skip_before((3, 12))
    def test_unbounded_tuple_unions(self):
        self.assert_passes(
            """
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

                assert_type(ge0, Eq0 | Ge1)
            """
        )
