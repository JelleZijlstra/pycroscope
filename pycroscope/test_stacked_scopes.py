# static analysis: ignore

from .error_code import ErrorCode
from .name_check_visitor import build_stacked_scopes
from .options import Options
from .stacked_scopes import ScopeType, uniq_chain
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes, skip_if_not_installed
from .value import (
    UNINITIALIZED_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    DictIncompleteValue,
    GenericValue,
    KnownValue,
    MultiValuedValue,
    ReferencingValue,
    SequenceValue,
    TypedValue,
    assert_is_value,
)


# just used for its __dict__
class Module:
    foo = 1
    bar = None


class TestStackedScopes:
    def setup_method(self):
        self.scopes = build_stacked_scopes(Module, options=Options({}))

    def test_scope_type(self):
        assert ScopeType.module_scope == self.scopes.scope_type()

        with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
            assert ScopeType.function_scope == self.scopes.scope_type()

        assert ScopeType.module_scope == self.scopes.scope_type()

    def test_current_and_module_scope(self):
        assert "foo" in self.scopes.current_scope()
        assert "foo" in self.scopes.module_scope()

        with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
            assert "foo" not in self.scopes.current_scope()
            assert "foo" in self.scopes.module_scope()

        assert "foo" in self.scopes.current_scope()
        assert "foo" in self.scopes.module_scope()

    def test_get(self):
        ctx = None
        assert KnownValue(1) == self.scopes.get("foo", None, None, can_assign_ctx=ctx)

        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            self.scopes.set("foo", KnownValue(2), None, None)
            assert KnownValue(2) == self.scopes.get(
                "foo", None, None, can_assign_ctx=ctx
            )

        assert KnownValue(1) == self.scopes.get("foo", None, None, can_assign_ctx=ctx)

        assert UNINITIALIZED_VALUE is self.scopes.get(
            "doesnt_exist", None, None, can_assign_ctx=ctx
        )

        # outer class scopes aren't used
        with self.scopes.add_scope(ScopeType.class_scope, scope_node=None):
            self.scopes.set("cls1", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get(
                "cls1", None, None, can_assign_ctx=ctx
            )

            with self.scopes.add_scope(ScopeType.class_scope, scope_node=None):
                self.scopes.set("cls2", KnownValue(1), None, None)
                assert KnownValue(1) == self.scopes.get(
                    "cls2", None, None, can_assign_ctx=ctx
                )

                assert UNINITIALIZED_VALUE is self.scopes.get(
                    "cls1", None, None, can_assign_ctx=ctx
                )

            assert KnownValue(1) == self.scopes.get(
                "cls1", None, None, can_assign_ctx=ctx
            )

    def test_set(self):
        ctx = None
        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            self.scopes.set("multivalue", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get(
                "multivalue", None, None, can_assign_ctx=ctx
            )
            self.scopes.set("multivalue", KnownValue(2), None, None)
            assert MultiValuedValue([KnownValue(1), KnownValue(2)]) == self.scopes.get(
                "multivalue", None, None, can_assign_ctx=ctx
            )
            self.scopes.set("multivalue", KnownValue(3), None, None)
            assert MultiValuedValue(
                [KnownValue(1), KnownValue(2), KnownValue(3)]
            ) == self.scopes.get("multivalue", None, None, can_assign_ctx=ctx)

            # if the values set are the same, don't make a MultiValuedValue
            self.scopes.set("same", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get(
                "same", None, None, can_assign_ctx=ctx
            )
            self.scopes.set("same", KnownValue(1), None, None)
            assert KnownValue(1) == self.scopes.get(
                "same", None, None, can_assign_ctx=ctx
            )

            # even if they are AnyValue
            any = AnyValue(AnySource.marker)
            self.scopes.set("unresolved", any, None, None)
            assert any is self.scopes.get("unresolved", None, None, can_assign_ctx=ctx)
            self.scopes.set("unresolved", any, None, None)
            assert any is self.scopes.get("unresolved", None, None, can_assign_ctx=ctx)

    def test_referencing_value(self):
        ctx = None
        with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
            outer = self.scopes.current_scope()
            self.scopes.set("reference", KnownValue(1), None, None)
            multivalue = MultiValuedValue([KnownValue(1), KnownValue(2)])

            with self.scopes.add_scope(ScopeType.module_scope, scope_node=None):
                val = ReferencingValue(outer, "reference")
                self.scopes.set("reference", val, None, None)
                assert KnownValue(1) == self.scopes.get(
                    "reference", None, None, can_assign_ctx=ctx
                )
                self.scopes.set("reference", KnownValue(2), None, None)
                assert multivalue == self.scopes.get(
                    "reference", None, None, can_assign_ctx=ctx
                )

            assert multivalue == self.scopes.get(
                "reference", None, None, can_assign_ctx=ctx
            )

            self.scopes.set(
                "nonexistent",
                ReferencingValue(self.scopes.module_scope(), "nonexistent"),
                None,
                None,
            )
            assert UNINITIALIZED_VALUE is self.scopes.get(
                "nonexistent", None, None, can_assign_ctx=ctx
            )

            self.scopes.set("is_none", KnownValue(None), None, None)

            with self.scopes.add_scope(ScopeType.function_scope, scope_node=None):
                self.scopes.set(
                    "is_none", ReferencingValue(outer, "is_none"), None, None
                )
                assert AnyValue(AnySource.inference) == self.scopes.get(
                    "is_none", None, None, can_assign_ctx=ctx
                )

    def test_typed_value_set(self):
        ctx = None
        self.scopes.set("value", TypedValue(dict), None, None)
        assert TypedValue(dict) == self.scopes.get(
            "value", None, None, can_assign_ctx=ctx
        )
        div = DictIncompleteValue(dict, [])  # subclass of TypedValue
        self.scopes.set("value", div, None, None)
        assert div == self.scopes.get("value", None, None, can_assign_ctx=ctx)


class TestScoping(TestNameCheckVisitorBase):
    @assert_passes()
    def test_multiple_assignment(self):
        from typing_extensions import Literal

        def capybara():
            x = 3
            assert_type(x, Literal[3])
            x = 4
            assert_type(x, Literal[4])

    @assert_passes()
    def test_undefined_name(self):
        def capybara():
            return x  # E: undefined_name

    @assert_passes()
    def test_read_before_write(self):
        def capybara():
            print(x)  # E: undefined_name
            x = 3
            print(x)

    @assert_passes()
    def test_function_argument(self):
        from typing_extensions import Literal

        def capybara(x):
            assert_is_value(x, AnyValue(AnySource.unannotated))
            x = 3
            assert_type(x, Literal[3])

    @assert_passes()
    def test_default_arg(self):
        def capybara(x=3):
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(3)])
            )

    @assert_passes()
    def test_args_kwargs(self):
        def capybara(*args, **kwargs):
            assert_is_value(
                args, GenericValue(tuple, [AnyValue(AnySource.unannotated)])
            )
            assert_is_value(
                kwargs,
                GenericValue(dict, [TypedValue(str), AnyValue(AnySource.unannotated)]),
            )

    @assert_passes()
    def test_args_kwargs_annotated(self):
        def capybara(*args: int, **kwargs: int):
            assert_type(args, tuple[int, ...])
            assert_type(kwargs, dict[str, int])

    @assert_passes()
    def test_internal_imports(self):
        # nested import froms are tricky because there is no separate AST node for each name, so we
        # need to use a special trick to represent the distinct definition nodes for each name
        import collections

        def capybara():
            from collections import Counter, defaultdict

            assert_is_value(Counter, KnownValue(collections.Counter))
            assert_is_value(defaultdict, KnownValue(collections.defaultdict))

    @assert_passes()
    def test_return_annotation(self):
        import socket

        class Capybara:
            def socket(self) -> socket.error:
                return socket.error()


class TestIf(TestNameCheckVisitorBase):
    @assert_passes()
    def test_basic(self):
        from typing_extensions import Literal

        def capybara(cond):
            if cond:
                x = 3
                assert_type(x, Literal[3])
            else:
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[3, 4])

    @assert_passes()
    def test_nesting(self):
        from typing_extensions import Literal

        def capybara(cond1, cond2):
            if cond1:
                x = 3
                assert_type(x, Literal[3])
            else:
                if cond2:
                    x = 4
                    assert_type(x, Literal[4])
                else:
                    x = 5
                    assert_type(x, Literal[5])
                assert_type(x, Literal[4, 5])
            assert_type(x, Literal[3, 4, 5])


class TestTry(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_except(self):
        from typing_extensions import Literal

        def capybara():
            try:
                x = 3
                assert_type(x, Literal[3])
            except NameError as e:
                assert_type(e, NameError)
                x = 4
                assert_type(x, Literal[4])
            except (RuntimeError, ValueError) as e:
                assert_type(e, RuntimeError | ValueError)
            assert_is_value(
                x,
                MultiValuedValue(
                    [KnownValue(3), KnownValue(4), AnyValue(AnySource.error)]
                ),
            )

    @assert_passes()
    def test_set_before_try(self):
        from typing_extensions import Literal

        def capybara():
            x = 1
            try:
                x = 2
                assert_type(x, Literal[2])
            except NameError:
                assert_type(x, Literal[1, 2])
                x = 3
                assert_type(x, Literal[3])
            except RuntimeError:
                assert_type(x, Literal[1, 2])
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[2, 3, 4])

    @assert_passes()
    def test_multiple_except(self):
        from typing_extensions import Literal

        def capybara():
            try:
                x = 3
                assert_type(x, Literal[3])
            except NameError:
                x = 4
                assert_type(x, Literal[4])
            except IOError:
                x = 5
                assert_type(x, Literal[5])
            assert_type(x, Literal[3, 4, 5])

    @assert_passes()
    def test_else(self):
        from typing_extensions import Literal

        def capybara():
            x = 1
            try:
                x = 2
                x = 3
                assert_type(x, Literal[3])
            except NameError:
                assert_type(x, Literal[1, 2, 3])
                x = 4
                assert_type(x, Literal[4])
            else:
                assert_type(x, Literal[3])
                x = 5
                assert_type(x, Literal[5])
            assert_type(x, Literal[4, 5])

    @assert_passes()
    def test_multiple_assignment(self):
        from typing_extensions import Literal

        def capybara():
            x = 1
            try:
                x = 2
                x = 3
                assert_type(x, Literal[3])
            except NameError:
                assert_type(x, Literal[1, 2, 3])
            else:
                assert_type(x, Literal[3])
            assert_type(x, Literal[1, 2, 3])

    @assert_passes()
    def test_finally(self):
        from typing_extensions import Literal

        def capybara():
            try:
                x = 3
                assert_type(x, Literal[3])
            finally:
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[4])

    @assert_passes()
    def test_finally_regression():
        import subprocess

        def test_full():
            clients = []
            try:
                clients.append(subprocess.Popen([]))
            finally:
                for client in clients:
                    client.kill()

    @assert_passes()
    def test_finally_plus_if(self):
        # here an approach that simply ignores the assignments in the try block while examining the
        # finally block would fail
        from typing_extensions import Literal

        def capybara():
            x = 0
            assert_type(x, Literal[0])
            try:
                x = 1
                assert_type(x, Literal[1])
            finally:
                assert_type(x, Literal[0, 1])

    @assert_passes()
    def test_finally_plus_return(self):
        from typing_extensions import Literal

        def capybara():
            x = 0
            assert_type(x, Literal[0])
            try:
                x = 1
                assert_type(x, Literal[1])
                return
            finally:
                assert_type(x, Literal[0, 1])

    @assert_passes()
    def test_bad_except_handler(self):
        def capybara():
            try:
                x = 1
            except 42 as fortytwo:  # E: bad_except_handler
                print(fortytwo)
            else:
                print(x)


class TestLoops(TestNameCheckVisitorBase):
    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_conditional_in_loop(self):
        from typing_extensions import Literal

        def capybara():
            for i in range(2):
                if i == 1:
                    print(x)
                    assert_is_value(
                        x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
                    )
                else:
                    x = 3
                    assert_type(x, Literal[3])
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
            )

    @assert_passes()
    def test_second_assignment_in_loop(self):
        from typing_extensions import Literal

        def capybara():
            hide_until = None
            for _ in range(3):
                assert_type(hide_until, tuple[Literal[1], Literal[2]] | None)
                if hide_until:
                    print(hide_until[1])
                hide_until = (1, 2)

    @assert_passes()
    def test_for_else(self):
        from typing_extensions import Literal

        def capybara():
            for _ in range(2):
                x = 3
                assert_type(x, Literal[3])
            else:
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[3, 4])

    @assert_passes()
    def test_for_always_entered(self):
        from typing_extensions import Literal

        def capybara():
            x = 3
            assert_type(x, Literal[3])
            for _ in [0, 1]:
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[4])

        def huge_range():
            for i in range(1000000000):
                # We don't create the whole Union
                assert_type(i, int)
            # But we do recognize that the iterable is nonempty
            assert_type(i, int)

    @assert_passes()
    def test_range_always_entered(self):
        from typing_extensions import Literal

        def capybara():
            for i in range(2):
                assert_type(i, Literal[0, 1])
            assert_type(i, Literal[0, 1])

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_use_after_for(self):
        def capybara(x):
            for _ in range(x):
                y = 4
                break

            assert_is_value(
                y, MultiValuedValue([KnownValue(4), AnyValue(AnySource.error)])
            )

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_use_after_for_conditional(self):
        def capybara(x):
            for _ in range(2):
                if x > 2:
                    y = 4
                    break

            assert_is_value(
                y, MultiValuedValue([KnownValue(4), AnyValue(AnySource.error)])
            )

    @assert_passes(settings={ErrorCode.possibly_undefined_name: False})
    def test_while(self):
        from typing_extensions import Literal

        def capybara():
            while bool():
                x = 3
                assert_type(x, Literal[3])
            assert_is_value(
                x, MultiValuedValue([AnyValue(AnySource.error), KnownValue(3)])
            )

    @assert_passes()
    def test_while_always_entered(self):
        from typing_extensions import Literal

        def capybara():
            while True:
                x = 3
                assert_type(x, Literal[3])
                break
            assert_type(x, Literal[3])

    @assert_passes()
    def test_while_else(self):
        from typing_extensions import Literal

        def capybara():
            while bool():
                x = 3
                assert_type(x, Literal[3])
            else:
                x = 4
                assert_type(x, Literal[4])
            assert_type(x, Literal[3, 4])

    @assert_passes()
    def test_recursive_func_in_loop(self):
        from typing import Iterable

        def capybara(xs: Iterable[int]):
            for x in xs:

                def do_something(y: int) -> int:
                    if x:
                        assert_type(do_something(y), int)
                        do_something("x")  # E: incompatible_argument
                    return y

                assert_type(do_something(x), int)
                do_something("x")  # E: incompatible_argument


class TestUnusedVariable(TestNameCheckVisitorBase):
    @assert_passes()
    def test_used(self):
        def capybara(condition):
            y = 3
            print(y)

            z = 3

            def nested():
                print(z)

            x = 4
            if condition:
                print(x)

    @assert_passes()
    def test_unused(self):
        def capybara():
            y = 3  # E: unused_variable

    @assert_passes()
    def test_unused_annotated_assignment(self):
        def capybara():
            y: int = 3  # E: unused_variable

    def test_replacement(self):
        self.assert_is_changed(
            """
def capybara():
    y = 3
    return 3
""",
            """
def capybara():
    return 3
""",
        )

    @assert_passes()
    def test_unused_then_used(self):
        def capybara():
            y = 3  # E: unused_assignment
            y = 4
            return y

    @assert_passes()
    def test_unused_annotated_assignment_then_used(self):
        def capybara():
            y: int = 3  # E: unused_assignment
            y = 4
            return y

    @assert_passes()
    def test_unused_in_if(self):
        def capybara(condition):
            if condition:
                x = 3  # E: unused_assignment
            x = 4
            return x

    @assert_passes()
    def test_while_loop(self):
        def capybara(condition):
            rlist = condition()
            while rlist:
                rlist = condition()

            num_items = 0
            while num_items < 10:
                if condition:
                    num_items += 1

    @assert_passes(settings={ErrorCode.use_fstrings: False})
    def test_try_finally(self):
        def func():
            return 1

        def capybara():
            x = 0

            try:
                x = func()
            finally:
                print("%d" % x)  # x is a number

    @assert_passes()
    def test_for_may_not_run(self):
        def capybara(iterable):
            # this is not unused, because iterable may be empty
            x = 0
            for x in iterable:
                print(x)
                break
            print(x)

    @assert_passes()
    def test_nesting(self):
        def capybara():
            def inner():
                print(x)

            x = 3
            inner()
            x = 4


class TestUnusedVariableComprehension(TestNameCheckVisitorBase):
    @assert_passes()
    def test_comprehension(self):
        def single_unused():
            return [None for i in range(10)]  # E: unused_variable

        def used():
            return [i for i in range(10)]

        def both_unused(pairs):
            return [None for a, b in pairs]  # E: unused_variable  # E: unused_variable

        def capybara(pairs):
            # this is OK; in real code the name of "b" might serve as useful documentation about
            # what is in "pairs"
            return [a for a, b in pairs]

    def test_replacement(self):
        self.assert_is_changed(
            """
            def capybara():
                return [None for i in range(10)]
            """,
            """
            def capybara():
                return [None for _ in range(10)]
            """,
        )


class TestUnusedVariableUnpacking(TestNameCheckVisitorBase):
    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_unused_in_yield(self):
        from asynq import asynq, result

        @asynq()
        def kerodon(i):
            return i

        @asynq()
        def capybara():
            a, b = yield kerodon.asynq(1), kerodon.asynq(2)  # E: unused_variable
            result(a)

    @skip_if_not_installed("asynq")
    @assert_passes()
    def test_async_returns_pair(self):
        from asynq import asynq, result

        @asynq()
        def returns_pair():
            return 1, 2

        @asynq()
        def capybara():
            a, b = yield returns_pair.asynq()
            result(a)

    @assert_passes()
    def test_all_unused(self):
        def capybara(pair):
            a, b = pair  # E: unused_variable  # E: unused_variable

    @assert_passes()
    def test_some_used(self):
        def capybara(pair):
            a, b = pair
            return a

    @assert_passes()
    def test_multiple_assignment(self):
        def capybara(pair):
            c = a, b = pair  # E: unused_variable  # E: unused_variable
            return c

    @assert_passes()
    def test_used_in_multiple_assignment(self):
        def capybara(pair):
            a, b = c, d = pair
            return a + d

    @assert_passes()
    def test_nested_unpack(self):
        def capybara(obj):
            (a, b), c = obj
            return c

    @assert_passes()
    def test_used_in_annassign(self):
        def capybara(condition):
            x: int
            if condition:
                x = 1
            else:
                x = 2
            return x


class TestLeavesScope(TestNameCheckVisitorBase):
    @assert_passes()
    def test_leaves_scope(self):
        def capybara(cond):
            if cond:
                return
            else:
                x = 3

            print(x)

    @assert_passes()
    def test_try_always_leaves_scope(self):
        def capybara(cond):
            try:
                x = 3
            except ValueError:
                if cond:
                    raise
                else:
                    return None

            print(x)

    @assert_passes()
    def test_try_may_leave_scope(self):
        def capybara(cond):
            try:
                x = 3
            except ValueError:
                if cond:
                    pass
                else:
                    return None

            print(x)  # E: possibly_undefined_name

    @assert_passes()
    def test_assert_false(self):
        def capybara(cond):
            if cond:
                assert False
            else:
                x = 3

            print(x)

    @assert_passes()
    def test_after_assert_false(self):
        from typing_extensions import Literal

        def capybara(cond):
            assert False
            if cond:
                x = True
            else:
                # For some reason in Python 2.7, False gets inferred as Any
                # after the assert False, but True and None still work.
                x = None
            y = None
            assert_type(y, None)
            assert_type(x, Literal[True] | None)

    @assert_passes()
    def test_elif_assert_false(self):
        def capybara(cond):
            if cond == 1:
                x = 3
            elif cond == 2:
                x = 4
            else:
                assert 0

            print(x)

    @assert_passes()
    def test_visit_assert_message(self):
        from typing import Union

        def needs_int(x: int) -> None:
            pass

        def capybara(x: Union[int, str]) -> None:
            assert_type(x, int | str)

            assert isinstance(x, str), needs_int(x)
            assert_type(x, str)

    @assert_passes()
    def test_no_cross_function_propagation(self):
        def capybara(cond):
            if cond == 1:
                x = 3
            else:
                pass

            return x  # E: possibly_undefined_name

        def kerodon():
            # make sure we don't propagate the UNINITIALIZED_VALUE from
            # inside capybara() to here
            y = capybara(2)
            print(y)


class TestConstraints(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assert_truthy(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_type(y, Literal[True, False])
            assert y
            assert_type(y, Literal[True])

    @assert_passes()
    def test_bool_narrowing(self):
        from typing_extensions import Literal

        def capybara(x: bool):
            assert_type(x, bool)
            if x is True:
                assert_type(x, Literal[True])
            else:
                assert_type(x, Literal[False])

    @assert_passes()
    def test_assert_falsy(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_type(y, Literal[True, False])
            assert not y
            assert_type(y, Literal[False])

    @assert_passes()
    def test_no_constraints_from_branches(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False

            if x:
                assert_type(y, Literal[True, False])
                assert y
                assert_type(y, Literal[True])
            # Constraints do not survive past the if block.
            assert_type(y, Literal[True, False])

    @assert_passes()
    def test_if(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False

            assert_type(y, Literal[True, False])
            if y:
                assert_type(y, Literal[True])
            else:
                assert_type(y, Literal[False])
            (assert_type(y, Literal[True]) if y else assert_type(y, Literal[False]))

    @assert_passes()
    def test_isinstance(self):
        class A(object):
            pass

        class B(A):
            pass

        class C(A):
            pass

        def capybara(x):
            assert_is_value(x, AnyValue(AnySource.unannotated))
            if isinstance(x, int):
                assert_type(x, int)
            else:
                assert_is_value(x, AnyValue(AnySource.unannotated))

            y: object = x
            if isinstance(y, A):
                assert_type(y, A)
                if isinstance(y, B):
                    assert_type(y, B)
                    if isinstance(y, C):
                        assert_is_value(y, TypedValue(B) & TypedValue(C))

            z: object = x
            if isinstance(z, B):
                assert_type(z, B)
                if isinstance(z, A):
                    # Less precise constraints are ignored.
                    assert_type(z, B)

            x = B()
            assert_type(x, B)
            if isinstance(x, A):
                # Don't widen the type to A.
                assert_type(x, B)

    @assert_passes()
    def test_isinstance_multiple_types(self):
        def kerodon(cond1, cond2, val, lst: list):  # E: missing_generic_parameters
            if cond1:
                x = int(val)
            elif cond2:
                x = str(val)
            else:
                x = lst
            assert_is_value(
                x,
                MultiValuedValue([TypedValue(int), TypedValue(str), TypedValue(list)]),
            )

            if isinstance(x, (int, str)):
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))
            else:
                assert_type(x, list)

            assert_is_value(
                x,
                MultiValuedValue([TypedValue(int), TypedValue(str), TypedValue(list)]),
            )
            if isinstance(x, int) or isinstance(x, str):
                assert_is_value(x, MultiValuedValue([TypedValue(int), TypedValue(str)]))
            else:
                assert_type(x, list)

    @assert_passes()
    def test_complex_boolean(self):
        from typing_extensions import Literal

        def paca(cond1, cond2):
            if cond1:
                x = True
            elif cond2:
                x = False
            else:
                x = None

            if (x is not True and x is not False) or (x is True):
                assert_is_value(
                    x, MultiValuedValue([KnownValue(None), KnownValue(True)])
                )
            else:
                assert_type(x, Literal[False])

    @assert_passes()
    def test_two_booleans_is(self):
        def capybara(cond1: bool, cond2: bool) -> None:
            from typing import Union

            from typing_extensions import Literal, assert_type

            if (cond1 is True) or (cond2 is True):
                # Ideally we'd elide the "Literal[False]" but this isn't
                # wrong.
                assert_type(cond1, Union[bool, Literal[False]])
                assert_type(cond2, bool)
            else:
                assert_type(cond1, Literal[False])
                assert_type(cond2, Literal[False])

    @assert_passes()
    def test_isinstance_mapping(self):
        import collections.abc
        from typing import Any, Mapping, Union

        from typing_extensions import assert_type

        class A: ...

        def takes_mapping(x: Mapping[str, Any]) -> None: ...

        def foo(x: Union[A, Mapping[str, Any]]) -> None:
            # This is tricky because Mapping is not an instance of type.
            if isinstance(x, Mapping):
                assert_is_value(
                    x,
                    (TypedValue(A) & TypedValue(collections.abc.Mapping))
                    | GenericValue(
                        collections.abc.Mapping,
                        [TypedValue(str), AnyValue(AnySource.explicit)],
                    ),
                )
            else:
                assert_type(x, A)

    @assert_passes()
    def test_isinstance_union(self):
        from typing import Union

        from typing_extensions import assert_type

        def foo(x: Union[int, str, range]) -> None:
            if isinstance(x, int | str):
                assert_type(x, int | str)
            else:
                assert_type(x, range)
            if isinstance(x, Union[int, range]):
                assert_type(x, int | range)
            else:
                assert_type(x, str)

    @assert_passes()
    def test_isinstance_nested_tuple(self):
        from typing import Union

        from typing_extensions import assert_type

        def foo(x: Union[int, str, range]) -> None:
            if isinstance(x, (((int,), (str,)),)):
                assert_type(x, Union[int, str])
            else:
                assert_type(x, range)

    @assert_passes()
    def test_isinstance_bad_arg(self):
        def capybara(x):
            if isinstance(x, 1):  # E: incompatible_argument
                pass

    @assert_passes()
    def test_double_index(self):
        from typing import Optional, Union

        class A:
            attr: Union[int, str]

        class B:
            attr: Optional[A]

        def capybara(b: B):
            assert_type(b, B)
            assert_type(b.attr, A | None)
            if b.attr is not None:
                assert_type(b.attr, A)
                assert_type(b.attr.attr, int | str)
                if isinstance(b.attr.attr, int):
                    assert_type(b.attr.attr, int)

    @assert_passes()
    def test_nested_scope(self):
        class A:
            pass

        class B(A):
            pass

        def capybara(a: A, iterable):
            if isinstance(a, B):
                assert_type(a, B)
                lst = [a for _ in iterable]
                assert_is_value(lst, SequenceValue(list, [(True, TypedValue(B))]))

    @skip_if_not_installed("qcore")
    @assert_passes()
    def test_qcore_asserts(self):
        from qcore.asserts import assert_is_instance
        from typing_extensions import Literal

        def capybara(cond):
            if cond:
                x = True
                y = True
            else:
                x = False
                y = False

            assert_type(x, Literal[True, False])
            assert_type(y, Literal[True, False])
            assert x is True
            assert True is y
            assert_type(x, Literal[True])
            assert_type(y, Literal[True])

        def paca(cond):
            if cond:
                x = True
                y = True
            else:
                x = False
                y = False

            assert_type(x, Literal[True, False])
            assert_type(y, Literal[True, False])
            assert x is not True
            assert True is not y
            assert_type(x, Literal[False])
            assert_type(y, Literal[False])

        def mara(cond, cond2):
            assert_is_value(cond, AnyValue(AnySource.unannotated))
            assert_is_instance(cond, int)
            assert_type(cond, int)

            assert_is_instance(cond2, (int, str))
            assert_type(cond2, int | str)

    @assert_passes()
    def test_is_or_is_not(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False

            assert_type(y, Literal[True, False])
            if y is True:
                assert_type(y, Literal[True])
            else:
                assert_type(y, Literal[False])
            if y is not True:
                assert_type(y, Literal[False])
            else:
                assert_type(y, Literal[True])

    @assert_passes()
    def test_and_or(self):
        from typing import Literal

        true_or_false = MultiValuedValue([KnownValue(True), KnownValue(False)])

        def capybara(x, y):
            if x is True and y is True:
                assert_type(x, Literal[True])
                assert_type(y, Literal[True])
            else:
                # no constraints from the inverse of an AND constraint
                assert_is_value(y, AnyValue(AnySource.unannotated))

        def kerodon(x):
            if x is True and assert_type(x, Literal[True]):
                pass
            # After the if it's either True (if the if branch was taken)
            # or Any (if it wasn't). This is not especially
            # useful in this case, but hopefully harmless.
            assert_is_value(
                x, MultiValuedValue([KnownValue(True), AnyValue(AnySource.unannotated)])
            )

        def paca(x):
            if x:
                y = True
                z = True
            else:
                y = False
                z = False

            if y is True or z is True:
                assert_type(y, bool)
                assert_type(z, bool)
            else:
                assert_type(y, Literal[False])
                assert_type(z, Literal[False])

        def pacarana(x):
            # OR constraints within the conditional
            if x:
                z = True
            else:
                z = False
            if z is True or assert_type(z, Literal[False]):
                pass

        def hutia(x):
            if x:
                y = True
            else:
                y = False

            if x and y:
                assert_type(y, Literal[True])
            else:
                assert_type(y, bool)

        def mara(x):
            if x:
                y = True
                z = True
            else:
                y = False
                z = False

            if not (y is True and z is True):
                assert_type(y, bool)
                assert_type(z, bool)
            else:
                assert_type(y, Literal[True])
                assert_type(z, Literal[True])

        def phoberomys(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False

            if not ((x is False or y is False) or z is True):
                assert_type(x, Literal[True])
                assert_type(y, Literal[True])
                assert_type(z, Literal[False])
            else:
                assert_type(x, bool)
                assert_type(y, bool)
                assert_type(z, bool)

        def llitun(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False
            if x and y and z:
                assert_type(x, Literal[True])
                assert_type(y, Literal[True])
                assert_type(z, Literal[True])
            else:
                assert_type(x, bool)
                assert_type(y, bool)
                assert_type(z, bool)

        def coypu(cond):
            if cond:
                x = True
                y = True
                z = True
            else:
                x = False
                y = False
                z = False
            if x or y or z:
                assert_type(x, bool)
                assert_type(y, bool)
                assert_type(z, bool)
            else:
                assert_type(x, Literal[False])
                assert_type(y, Literal[False])
                assert_type(z, Literal[False])

    @assert_passes()
    def test_set_in_condition(self):
        from typing import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_type(y, bool)
            if not y:
                assert_type(y, Literal[False])
                y = True
            assert_type(y, Literal[True])

    @assert_passes()
    def test_optional_becomes_non_optional(self):
        from typing import Optional

        def capybara(x: Optional[int]) -> None:
            assert_type(x, int | None)
            if not x:
                x = int(0)
            assert_type(x, int)

    @assert_passes()
    def test_reset_on_assignment(self):
        from typing import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False
            if y is True:
                assert_type(y, Literal[True])
                y = bool(x)
                assert_is_value(y, TypedValue(bool), skip_annotated=True)

    @assert_passes()
    def test_constraint_on_arg_type(self):
        from typing import Optional

        def kerodon() -> Optional[int]:
            return 3

        def capybara() -> None:
            x = kerodon()
            assert_type(x, int | None)

            if x:
                assert_type(x, int)
            else:
                assert_type(x, int | None)
            if x is not None:
                assert_type(x, int)
            else:
                assert_type(x, None)

    @assert_passes()
    def test_constraint_in_nested_scope(self):
        from typing import Optional

        def capybara(x: Optional[int], z):
            if x is None:
                return

            assert_type(x, int)

            def nested():
                assert_type(x, int)

            return [assert_type(x, int) for _ in z]

    @assert_passes()
    def test_repeated_constraints(self):
        def capybara(cond):
            if cond:
                x = True
            else:
                x = False
            assert_type(x, bool)

            # Tests that this completes in a reasonable time.
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            if x:
                pass
            assert_type(x, bool)

    @assert_passes()
    def test_nonlocal_unresolved(self):
        def capybara(x):
            def nested():
                while True:
                    assert_is_value(x, AnyValue(AnySource.unannotated))
                    if x:
                        pass

            return nested()

    @assert_passes()
    def test_nonlocal_unresolved_if(self):
        def capybara(x):
            def nested():
                assert_is_value(x, AnyValue(AnySource.unannotated))
                if x:
                    assert_is_value(x, AnyValue(AnySource.unannotated))

            return nested()

    @assert_passes()
    def test_nonlocal_known(self):
        from typing_extensions import Literal

        def capybara(y):
            if y:
                x = True
            else:
                x = False

            def nested():
                assert_type(x, bool)
                if x:
                    assert_type(x, Literal[True])
                else:
                    assert_type(x, Literal[False])

    @assert_passes()
    def test_nonlocal_known_with_write(self):
        from typing_extensions import Literal

        def capybara(y):
            if y:
                x = True
            else:
                x = False

            def nested():
                nonlocal x
                assert_type(x, bool)
                if x:
                    assert_type(x, Literal[True])
                else:
                    assert_type(x, Literal[False])
                    x = True
                    assert_type(x, Literal[True])

    @assert_passes()
    def test_nonlocal_in_loop(self):
        def capybara(x):
            def nested(y):
                for _ in y:
                    if x:
                        pass

    @assert_passes()
    def test_nonlocal_not_unused(self):
        def _get_call_point(x, y):
            frame = x
            while y(frame):
                frame = frame.f_back
            return {"filename": frame.f_code.co_filename, "line_no": frame.f_lineno}

    @assert_passes()
    def test_conditional_assignment_to_global(self):
        _disk_size_with_low_usage = 0

        def _report_boxes_with_low_disk_usage(tier):
            global _disk_size_with_low_usage
            x = 0
            if tier.startswith("lego"):
                _disk_size_with_low_usage = 3
            x += _disk_size_with_low_usage
            _disk_size_with_low_usage = 0
            return x

    @assert_passes()
    def test_comprehension(self):
        def maybe_int(x):
            if x:
                return int(x)
            else:
                return None

        def capybara(x, y):
            assert_type(maybe_int(x), int | None)

            lst = [maybe_int(elt) for elt in y]
            assert_is_value(
                lst, SequenceValue(list, [(True, TypedValue(int) | KnownValue(None))])
            )
            lst2 = [elt for elt in lst if elt]
            assert_is_value(lst2, SequenceValue(list, [(True, TypedValue(int))]))

    @assert_passes()
    def test_comprehension_composite(self):
        from dataclasses import dataclass
        from typing import List, Optional, Tuple

        @dataclass
        class Capybara:
            x: Optional[int]

        def use_attr(c: List[Capybara]) -> None:
            assert_is_value(
                [elt.x for elt in c],
                SequenceValue(list, [(True, TypedValue(int) | KnownValue(None))]),
            )
            assert_is_value(
                [elt.x for elt in c if elt.x is not None],
                SequenceValue(list, [(True, TypedValue(int))]),
            )
            assert_is_value(
                [elt.x for elt in c if elt.x],
                SequenceValue(list, [(True, TypedValue(int))]),
            )

        def use_subscript(d: List[Tuple[int, Optional[int]]]) -> None:
            assert_is_value(
                [pair[1] for pair in d],
                SequenceValue(list, [(True, TypedValue(int) | KnownValue(None))]),
            )
            assert_is_value(
                [pair[1] for pair in d if pair[1] is not None],
                SequenceValue(list, [(True, TypedValue(int))]),
            )
            assert_is_value(
                [pair[1] for pair in d if pair[1]],
                SequenceValue(list, [(True, TypedValue(int))]),
            )

    @assert_passes()
    def test_while(self):
        from typing_extensions import Literal

        def capybara(x):
            if x:
                y = True
            else:
                y = False
            assert_type(y, bool)
            while y:
                assert_type(y, Literal[True])
            assert_type(y, bool)

    @assert_passes()
    def test_while_hasattr(self):
        from typing import Optional

        from typing_extensions import assert_type

        from pycroscope.predicates import HasAttr
        from pycroscope.value import IntersectionValue, PredicateValue, TypedValue

        def capybara(x: Optional[int]):
            assert_type(x, int | None)
            while x is not None and hasattr(x, "name"):
                assert_is_value(
                    x,
                    IntersectionValue(
                        (
                            TypedValue(int),
                            PredicateValue(HasAttr("name", TypedValue(object))),
                        )
                    ),
                )

    @assert_passes()
    def test_hasattr_annotated_unification(self):
        from typing_extensions import assert_type

        from pycroscope.predicates import HasAttr
        from pycroscope.value import PredicateValue, TypedValue

        def capybara(x: int) -> None:
            assert_type(x, int)
            if hasattr(x, "name"):
                assert_is_value(
                    x,
                    TypedValue(int)
                    & PredicateValue(HasAttr("name", TypedValue(object))),
                )
            # TODO this should simplify to just int
            assert_is_value(
                x,
                (TypedValue(int) & PredicateValue(HasAttr("name", TypedValue(object))))
                | TypedValue(int),
            )

    @assert_passes()
    def test_gt_annotated_unification(self):
        from pycroscope.annotated_types import Gt
        from pycroscope.value import CustomCheckExtension

        ext = CustomCheckExtension(Gt(5))

        def capybara(x: int) -> None:
            assert_type(x, int)
            if x > 5:
                assert_is_value(x, AnnotatedValue(TypedValue(int), [ext]))
            assert_is_value(x, TypedValue(int) | AnnotatedValue(TypedValue(int), [ext]))

    @assert_passes()
    def test_unconstrained_composite(self):
        class Foo(object):
            def has_images(self):
                pass

        class InlineEditor:
            def init(self, input, is_qtext=False):
                if is_qtext:
                    value = input
                else:
                    value = ""

                assert_is_value(
                    value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )

                self.value = value

                assert_is_value(
                    self.value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )

            def tree(self):
                assert_is_value(
                    self.value,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue("")]),
                )
                if isinstance(self.value, Foo) and self.value.has_images():
                    assert_type(self.value, Foo)
                else:
                    assert_is_value(
                        self.value,
                        MultiValuedValue(
                            [
                                AnyValue(AnySource.unannotated),
                                TypedValue(Foo),
                                KnownValue(""),
                            ]
                        ),
                    )
                assert_is_value(
                    self.value,
                    MultiValuedValue(
                        [
                            AnyValue(AnySource.unannotated),
                            TypedValue(Foo),
                            KnownValue(""),
                        ]
                    ),
                )

    @assert_passes()
    def test_operator_constraints(self):
        from typing import Union

        from typing_extensions import Literal

        container = {1, 2, 3}

        def capybara(cond):
            x = 1 if cond else 2
            assert_type(x, Literal[1, 2])
            if x == 1:
                assert_type(x, Literal[1])
            else:
                assert_type(x, Literal[2])
            if x in (2,):
                assert_type(x, Literal[2])
            else:
                assert_type(x, Literal[1])
            if "x" in cond:
                assert_is_value(cond, AnyValue(AnySource.unannotated))

        def pacarana(x: Union[Literal["x"], int]):
            assert_type(x, Literal["x"] | int)
            if x == 0:
                assert_type(x, Literal[0])
            elif x == "x":
                assert_type(x, Literal["x"])
            else:
                assert_type(x, int)

        def moco(x: Union[Literal["x"], int]):
            assert_type(x, Literal["x"] | int)
            if x != 0:
                assert_type(x, Literal["x"] | int)
            else:
                assert_type(x, Literal[0])

        def hutia(x: str, y: object):
            if x in ["a", "b"]:
                assert_type(x, Literal["a", "b"])
            if y in container:
                assert_type(y, Literal[1, 2, 3])

    @skip_if_not_installed("annotated_types")
    @assert_passes()
    def test_preserve_annotated(self):
        from typing import Any, Optional

        from annotated_types import Predicate
        from typing_extensions import Annotated

        from pycroscope.annotated_types import AnnotatedPredicate
        from pycroscope.value import CustomCheckExtension

        def pred(x: Any) -> bool:
            return x > 5

        Pred = Predicate(pred)
        Ext = CustomCheckExtension(AnnotatedPredicate(pred))
        AnnotatedUnion = AnnotatedValue(TypedValue(str), [Ext]) | AnnotatedValue(
            KnownValue(None), [Ext]
        )

        def capybara(x: Annotated[Optional[str], Pred]) -> None:
            assert_is_value(x, AnnotatedUnion)

            if x:
                assert_type(x, Annotated[str, Pred])
            else:
                # None or the empty string
                assert_is_value(x, AnnotatedUnion)

        def pacarana(x: Annotated[Optional[str], Pred]) -> None:
            assert_is_value(x, AnnotatedUnion)
            if x is not None:
                assert_type(x, Annotated[str, Pred])
            else:
                assert_type(x, Annotated[None, Pred])

        def agouti(x: Annotated[Optional[str], Pred]) -> None:
            assert_is_value(x, AnnotatedUnion)
            if isinstance(x, str):
                assert_type(x, Annotated[str, Pred])
            else:
                assert_type(x, Annotated[None, Pred])

    @assert_passes()
    def test_possibly_undefined(self):
        from typing_extensions import Literal, assert_type

        def capybara(cond):
            if cond:
                x = 1

            if x:  # E: possibly_undefined_name
                assert_type(x, Literal[1])


class TestComposite(TestNameCheckVisitorBase):
    @assert_passes()
    def test_assignment(self):
        from typing_extensions import Literal

        class Capybara(object):
            def __init__(self, x):
                self.x = x

            def eat(self):
                assert_is_value(
                    self.x,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(1)]),
                )
                self.x = 1
                assert_type(self.x, Literal[1])

                self = Capybara(2)
                assert_is_value(
                    self.x,
                    MultiValuedValue([AnyValue(AnySource.unannotated), KnownValue(1)]),
                )

    @assert_passes()
    def test_conditional_attribute_assign(self):
        class Capybara(object):
            def __init__(self, x):
                self.x = int(x)

            def eat(self, cond, val):
                if cond:
                    self.x = int(val)
                x = self.x
                assert_type(x, int)

    @assert_passes()
    def test_attribute_to_never(self):
        from typing import Union

        class TypedValue:
            typ: Union[type, str]

            def get_generic_args_for_type(self) -> object:
                if isinstance(self.typ, super):
                    return self.typ.__self_class__
                else:
                    assert False

    @assert_passes()
    def test_constraint(self):
        class Capybara(object):
            def __init__(self, x):
                self.x = x

            def eat(self, val):
                self.x = val
                if isinstance(self.x, int):
                    assert_type(self.x, int)

            def eat_no_assign(self):
                if isinstance(self.x, int):
                    assert_type(self.x, int)

    @assert_passes()
    def test_subscript(self):
        from typing import Any, Dict

        from typing_extensions import Literal

        def capybara(x: Dict[str, Any], y) -> None:
            assert_is_value(x["a"], AnyValue(AnySource.explicit))
            x["a"] = 1
            assert_type(x["a"], Literal[1])
            if isinstance(x["c"], int):
                assert_is_value(x["c"], AnyValue(AnySource.explicit) & TypedValue(int))
            if x["b"] is None:
                assert_type(x["b"], None)

    @assert_passes()
    def test_unhashable_subscript(self):
        def capybara(df):
            # make sure this doesn't crash
            df[["a", "b"]] = 42
            print(df[["a", "b"]])


def test_uniq_chain():
    assert [] == uniq_chain([])
    assert list(range(3)) == uniq_chain(range(3) for _ in range(3))
    assert [1] == uniq_chain([1, 1, 1] for _ in range(3))


class TestInvalidation(TestNameCheckVisitorBase):
    @assert_passes()
    def test_still_valid(self) -> None:
        def capybara(x, y):
            condition = isinstance(x, int)
            assert_is_value(x, AnyValue(AnySource.unannotated))
            if condition:
                assert_type(x, int)

            condition = isinstance(y, int) if x else isinstance(y, str)
            assert_is_value(y, AnyValue(AnySource.unannotated))
            if condition:
                assert_type(y, int | str)

    @assert_passes()
    def test_invalidated(self) -> None:
        def capybara(x, y):
            condition = isinstance(x, int)
            assert_is_value(x, AnyValue(AnySource.unannotated))
            x = y
            if condition:
                assert_is_value(x, AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_other_scope(self) -> None:
        def callee(x):
            return isinstance(x, int)

        def capybara(x, y):
            if callee(y):
                assert_is_value(x, AnyValue(AnySource.unannotated))
                assert_is_value(y, AnyValue(AnySource.unannotated))

    @assert_passes()
    def test_while(self) -> None:
        from typing import Optional

        def make_optional() -> Optional[str]:
            return "x"

        def capybara():
            x = make_optional()
            while x:
                assert_type(x, str)
                x = make_optional()

    @assert_passes()
    def test_len_condition(self) -> None:
        from typing_extensions import Any, assert_type

        from pycroscope.predicates import MinLen
        from pycroscope.value import (
            AnySource,
            AnyValue,
            IntersectionValue,
            PredicateValue,
        )

        def capybara(file_list, key, ids):
            has_bias = len(key) > 0
            data = []
            for _ in file_list:
                assert_type(key, Any)
                if has_bias:
                    assert_is_value(
                        key,
                        IntersectionValue(
                            (AnyValue(AnySource.unannotated), PredicateValue(MinLen(1)))
                        ),
                    )
                    data = [ids, data[key]]
                else:
                    data = [ids]

    @assert_passes()
    def test_len_condition_with_type(self) -> None:
        from typing import Optional

        def capybara(file_list, key: Optional[int], ids):
            has_bias = key is not None
            data = []
            for _ in file_list:
                assert_type(key, int | None)
                if has_bias:
                    assert_type(key, int)
                    data = [ids, data[key]]
                else:
                    assert_type(key, None)
                    data = [ids]

    @assert_passes()
    def test_len_condition_attribute_access(self) -> None:
        def capybara(x):
            if len(x) >= 2:
                x.__class__
                x.__str__

    @assert_passes()
    def test_else(self) -> None:
        from typing import Optional

        def capybara(key: Optional[int]):
            has_bias = key is not None
            assert_type(key, int | None)
            if has_bias:
                assert_type(key, int)
            else:
                assert_type(key, None)


class TestClassNesting(TestNameCheckVisitorBase):

    @assert_passes()
    def test_class_in_class(self):
        class Caviids(object):
            class Capybaras(object):
                if False:
                    print(neochoerus)  # E: undefined_name

            def method(self, cap: Capybaras):
                assert_type(cap, Caviids.Capybaras)

    @assert_passes()
    def test_class_in_function(self):
        from typing_extensions import Literal, assert_type

        def get_capybaras(object):
            x = 3

            class Capybaras(object):
                if False:
                    print(neochoerus)  # E: undefined_name

                assert_type(x, Literal[3])

    @assert_passes()
    def test_double_nesting(self):
        from typing_extensions import Literal, assert_type

        def outer():
            outer_var = "outer"

            def inner():
                inner_var = "inner"

                class Nested:
                    assert_type(outer_var, Literal["outer"])
                    assert_type(inner_var, Literal["inner"])

                return Nested

            return inner

    @assert_passes()
    def test_triple_function_nesting(self):
        from typing_extensions import Literal, assert_type

        def outer():
            outer_var = "outer"

            def inner():
                inner_var = "inner"

                def innermost():
                    assert_type(outer_var, Literal["outer"])
                    assert_type(inner_var, Literal["inner"])

                return innermost

            return inner


class TestNestedAttributeNarrowing(TestNameCheckVisitorBase):
    @assert_passes()
    def test_nested_attribute(self):
        from typing import Optional

        from typing_extensions import assert_type

        class A:
            a: Optional[str]

        def f(x: A) -> None:
            if x.a is None:
                return
            any(assert_type(x.a, str) for _ in ["foo"])

    @assert_passes()
    def test_nested_subscript(self):
        from typing import Optional

        from typing_extensions import assert_type

        def f(x: list[Optional[str]]):
            if x[0] is None:
                return
            any(assert_type(x[0], str) for _ in ["foo"])
