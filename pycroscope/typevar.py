"""

TypeVar solver.

"""

from collections.abc import Iterable, Sequence

import pycroscope

from .analysis_lib import Sentinel
from .safe import all_of_type, is_instance_of_typing_name
from .value import (
    AnySource,
    AnyValue,
    Bound,
    BoundsMap,
    CanAssignContext,
    CanAssignError,
    IsOneOf,
    KnownValue,
    LowerBound,
    OrBound,
    SequenceValue,
    TypedValue,
    TypeVarLike,
    TypeVarMap,
    UpperBound,
    Value,
    replace_known_sequence_value,
    unite_values,
)

BOTTOM = Sentinel("<bottom>")
TOP = Sentinel("<top>")


def _as_fixed_tuple_members(value: Value) -> Sequence[Value] | None:
    value = replace_known_sequence_value(value)
    if not isinstance(value, SequenceValue) or value.typ is not tuple:
        return None
    return value.get_member_sequence()


def _merge_typevartuple_lower_bounds(
    lower1: Value, lower2: Value, ctx: CanAssignContext
) -> Value | CanAssignError | None:
    members1 = _as_fixed_tuple_members(lower1)
    members2 = _as_fixed_tuple_members(lower2)
    if members1 is None or members2 is None:
        return None
    if len(members1) != len(members2):
        return CanAssignError(
            "Incompatible bounds on type variable",
            [CanAssignError(str(lower1)), CanAssignError(str(lower2))],
        )
    merged_members = []
    for member1, member2 in zip(members1, members2):
        if member1.is_assignable(member2, ctx):
            merged = member1
        elif member2.is_assignable(member1, ctx):
            merged = member2
        else:
            merged = unite_values(member1, member2)
        merged_members.append((False, merged))
    return SequenceValue(tuple, merged_members)


def resolve_bounds_map(
    bounds_map: BoundsMap,
    ctx: CanAssignContext,
    *,
    all_typevars: Iterable[TypeVarLike] = (),
) -> tuple[TypeVarMap, Sequence[CanAssignError]]:
    tv_map = {tv: AnyValue(AnySource.generic_argument) for tv in all_typevars}
    errors = []
    for tv, bounds in bounds_map.items():
        bounds = tuple(dict.fromkeys(bounds))
        if is_instance_of_typing_name(tv, "ParamSpec"):
            # For ParamSpec, we use a simpler approach
            solution = pycroscope.input_sig.solve_paramspec(bounds, ctx)
        else:
            solution = solve(bounds, ctx)
        if isinstance(solution, CanAssignError):
            errors.append(solution)
            solution = AnyValue(AnySource.error)
        tv_map[tv] = solution
    return tv_map, errors


def solve(bounds: Iterable[Bound], ctx: CanAssignContext) -> Value | CanAssignError:
    from .relations import Relation, has_relation

    bottom = BOTTOM
    top = TOP
    options = None
    is_typevartuple = False

    for bound in bounds:
        if isinstance(bound, LowerBound):
            if is_instance_of_typing_name(bound.typevar, "TypeVarTuple"):
                is_typevartuple = True
                if bottom is not BOTTOM:
                    merged = _merge_typevartuple_lower_bounds(bottom, bound.value, ctx)
                    if isinstance(merged, CanAssignError):
                        return merged
                    if isinstance(merged, Value):
                        bottom = merged
                        continue
                    compatibility = has_relation(
                        bottom, bound.value, Relation.CONSISTENT, ctx
                    )
                    if isinstance(compatibility, CanAssignError):
                        return CanAssignError(
                            "Incompatible bounds on type variable",
                            [
                                CanAssignError(str(bottom)),
                                CanAssignError(str(bound.value)),
                            ],
                        )
            # Ignore lower bounds to Any
            if isinstance(bound.value, AnyValue) and bottom is not BOTTOM:
                continue
            if bottom is BOTTOM or bound.value.is_assignable(bottom, ctx):
                # New bound is more specific. Adopt it.
                bottom = bound.value
            elif bottom.is_assignable(bound.value, ctx):
                # New bound is less specific. Ignore it.
                pass
            else:
                # New bound is separate. We have to satisfy both.
                # TODO shouldn't this use intersection?
                bottom = unite_values(bottom, bound.value)
        elif isinstance(bound, UpperBound):
            if is_instance_of_typing_name(bound.typevar, "TypeVarTuple"):
                is_typevartuple = True
            if top is TOP or top.is_assignable(bound.value, ctx):
                top = bound.value
            elif bound.value.is_assignable(top, ctx):
                pass
            else:

                def _normalize_upper_bound(value: Value) -> Value | None:
                    if isinstance(value, KnownValue):
                        return TypedValue(type(value.val))
                    return value.get_fallback_value()

                top_fallback = (
                    _normalize_upper_bound(top) if isinstance(top, Value) else None
                )
                bound_fallback = _normalize_upper_bound(bound.value)
                if (
                    top_fallback is not None
                    and bound_fallback is not None
                    and (
                        top_fallback.is_assignable(bound_fallback, ctx)
                        or bound_fallback.is_assignable(top_fallback, ctx)
                    )
                ):
                    top = top_fallback
                    continue
                return CanAssignError(
                    "Incompatible upper bounds on type variable",
                    [CanAssignError(str(top)), CanAssignError(str(bound.value))],
                )
        elif isinstance(bound, OrBound):
            # TODO figure out how to handle this
            continue
        elif isinstance(bound, IsOneOf):
            if is_instance_of_typing_name(bound.typevar, "TypeVarTuple"):
                is_typevartuple = True
            options = bound.constraints
        else:
            raise AssertionError(f"unrecognized bound {bound!r}")

    if is_typevartuple and isinstance(bottom, Value) and isinstance(top, Value):
        compatibility = has_relation(bottom, top, Relation.CONSISTENT, ctx)
        if isinstance(compatibility, CanAssignError):
            return CanAssignError(
                "Incompatible bounds on type variable",
                [CanAssignError(str(bottom)), CanAssignError(str(top))],
            )

    if bottom is BOTTOM:
        if top is TOP:
            solution = AnyValue(AnySource.generic_argument)
        else:
            solution = top
    elif top is TOP:
        solution = bottom
    else:
        can_assign = has_relation(top, bottom, Relation.ASSIGNABLE, ctx)
        if isinstance(can_assign, CanAssignError):
            return CanAssignError(
                "Incompatible bounds on type variable",
                [
                    can_assign,
                    CanAssignError(
                        children=[CanAssignError(str(bound)) for bound in bounds]
                    ),
                ],
            )
        solution = bottom

    if options is not None:
        can_assigns = [
            has_relation(option, solution, Relation.ASSIGNABLE, ctx)
            for option in options
        ]
        if all_of_type(can_assigns, CanAssignError):
            return CanAssignError(children=list(can_assigns))
        available = [
            option
            for option, can_assign in zip(options, can_assigns)
            if not isinstance(can_assign, CanAssignError)
        ]
        # If there's only one solution, pick it.
        if len(available) == 1:
            return available[0]
        # If we inferred Any, keep it; all the solutions will be valid, and
        # picking one will lead to weird errors down the line.
        if isinstance(solution, AnyValue):
            return solution
        available = remove_redundant_solutions(available, ctx)
        if len(available) == 1:
            return available[0]
        # If there are still multiple options, we fall back to Any.
        return AnyValue(AnySource.inference)
    return solution


def remove_redundant_solutions(
    solutions: Sequence[Value], ctx: CanAssignContext
) -> Sequence[Value]:
    # This is going to be quadratic, so don't do it when there's too many
    # opttions.
    initial_count = len(solutions)
    if initial_count > 10:
        return solutions

    removed_indexes: set[int] = set()
    for i, sol in enumerate(solutions):
        for j, other in enumerate(solutions):
            if i == j or j in removed_indexes:
                continue
            if sol.is_assignable(other, ctx) and not other.is_assignable(sol, ctx):
                removed_indexes.add(i)
                break
    return [sol for i, sol in enumerate(solutions) if i not in removed_indexes]
