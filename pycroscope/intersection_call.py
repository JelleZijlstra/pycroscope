"""Call checking for intersection types.

The implementation is intentionally direct and may enumerate all subsets of
callable members.
"""

import itertools
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace

from typing_extensions import assert_never

from pycroscope.input_sig import ActualArguments
from pycroscope.relations import intersect_multi, intersect_values
from pycroscope.signature import (
    ANY_SIGNATURE,
    ARGS,
    Argument,
    BoundArgs,
    CallReturn,
    CheckCallContext,
    ConcreteSignature,
    OverloadedSignature,
    ParameterKind,
    Signature,
    preprocess_args,
)
from pycroscope.stacked_scopes import Composite
from pycroscope.value import (
    NO_RETURN_VALUE,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CallableValue,
    CanAssignContext,
    CanAssignError,
    GenericValue,
    IntersectionValue,
    KnownValue,
    MultiValuedValue,
    NewTypeValue,
    NotValue,
    OverlappingValue,
    ParamSpecArgsValue,
    ParamSpecKwargsValue,
    PartialCallValue,
    PartialValue,
    PredicateValue,
    SequenceValue,
    SubclassValue,
    SuperValue,
    SyntheticClassObjectValue,
    SyntheticModuleValue,
    SyntheticTypeFormValue,
    TypeAliasValue,
    TypedDictValue,
    TypedValue,
    TypeFormValue,
    TypeVarTupleBindingValue,
    TypeVarTupleValue,
    TypeVarValue,
    UnboundMethodValue,
    Value,
    Variance,
    gradualize,
    replace_fallback,
    unite_values,
)

GetSignature = Callable[[Value], object]


@dataclass(frozen=True)
class _Alternative:
    signature: Signature
    bound_args: BoundArgs


@dataclass(frozen=True)
class _Member:
    original: ConcreteSignature
    alternatives: tuple[_Alternative, ...]


def check_call(
    callee: IntersectionValue,
    args: Sequence[Argument],
    ctx: CheckCallContext,
    get_signature: GetSignature,
) -> Value:
    """Check a call to an intersection value."""
    actual_args = preprocess_args(args, ctx)
    if actual_args is None:
        return AnyValue(AnySource.error)

    # Step 1. Signature determination.
    signatures = _determine_signatures(callee, ctx, get_signature)
    if signatures is None:
        return AnyValue(AnySource.error)
    if len(signatures) == 1:
        return signatures[0].check_call(args, ctx)

    # Step 2. Parameter matching.
    members = _match_parameters(signatures, actual_args, ctx)
    if members is None:
        return AnyValue(AnySource.error)
    if len(members) == 1:
        return members[0].original.check_call(args, ctx)

    # Step 3. Validity.
    bottom_args = _materialize_actual_arguments(
        actual_args, _bottom_materialization, ctx
    )
    if not _valid(members, bottom_args, ctx):
        ctx.on_error("Cannot call intersection value")
        return AnyValue(AnySource.error)

    # Step 4. Return type.
    return _compute_return_type(members, actual_args, ctx)


def _determine_signatures(
    callee: IntersectionValue, ctx: CheckCallContext, get_signature: GetSignature
) -> tuple[ConcreteSignature, ...] | None:
    signatures = []
    noncallable = []
    for member in callee.vals:
        signature = get_signature(member)
        if signature is ANY_SIGNATURE:
            signatures.append(ANY_SIGNATURE)
        elif isinstance(signature, Signature):
            signatures.append(signature)
        elif isinstance(signature, OverloadedSignature):
            signatures.append(signature)
        elif _is_definitely_non_callable(member, ctx.can_assign_ctx):
            noncallable.append(member)

    if noncallable:
        detail = CanAssignError(
            "Some members of intersection are definitely not callable",
            [CanAssignError(str(member)) for member in noncallable],
        )
        ctx.on_error("Cannot call intersection value", detail=str(detail))
        return None
    if not signatures:
        ctx.on_error("Intersection value has no callable members")
        return None
    return tuple(signatures)


def _is_definitely_non_callable(value: Value, ctx: CanAssignContext) -> bool:
    value = replace_fallback(value)
    if isinstance(value, AnnotatedValue):
        return _is_definitely_non_callable(value.value, ctx)
    if isinstance(value, (AnyValue, PredicateValue, NotValue, TypeVarValue)):
        return False
    if isinstance(value, MultiValuedValue):
        return all(_is_definitely_non_callable(subval, ctx) for subval in value.vals)
    if isinstance(value, IntersectionValue):
        return any(_is_definitely_non_callable(subval, ctx) for subval in value.vals)
    if isinstance(value, SubclassValue):
        return False
    if isinstance(value, TypedValue):
        return value.get_type_object(ctx).is_final()
    if isinstance(value, GenericValue):
        return value.get_type_object(ctx).is_final()
    return False


def _match_parameters(
    signatures: Sequence[ConcreteSignature],
    actual_args: ActualArguments,
    ctx: CheckCallContext,
) -> tuple[_Member, ...] | None:
    members = []
    for signature in signatures:
        alternatives = []
        for concrete in _iter_signature_variants(signature):
            with _catch_call_errors(ctx):
                bound_args = concrete.bind_arguments(actual_args, ctx)
            if bound_args is not None:
                alternatives.append(_Alternative(concrete, bound_args))
        if alternatives:
            members.append(_Member(signature, tuple(alternatives)))

    if not members:
        ctx.on_error("Cannot call intersection value")
        return None
    return tuple(members)


@contextmanager
def _catch_call_errors(ctx: CheckCallContext) -> Generator[None]:
    if ctx.visitor is None:
        yield
        return
    with ctx.visitor.catch_errors():
        yield


def _iter_signature_variants(signature: ConcreteSignature) -> Iterable[Signature]:
    if isinstance(signature, Signature):
        yield signature
    else:
        yield from signature.signatures


def _valid(
    members: Sequence[_Member],
    actual_args: ActualArguments,
    ctx: CheckCallContext,
    *,
    seen: frozenset[tuple[str, ...]] = frozenset(),
) -> bool:
    if any(_member_accepts(member, actual_args, ctx) for member in members):
        return True

    key = _actual_arguments_key(actual_args)
    if key in seen:
        return False
    seen = seen | {key}

    split = _find_split(members, actual_args, ctx)
    if split is None:
        return False
    left, right = split
    return _valid(members, left, ctx, seen=seen) and _valid(
        members, right, ctx, seen=seen
    )


def _member_accepts(
    member: _Member, actual_args: ActualArguments, ctx: CheckCallContext
) -> bool:
    return _apply_member(member, actual_args, ctx) is not None


def _apply_member(
    member: _Member, actual_args: ActualArguments, ctx: CheckCallContext
) -> Value | None:
    if isinstance(member.original, OverloadedSignature):
        return _apply_overload_member(member, actual_args, ctx)

    returns = []
    for alternative in member.alternatives:
        with _catch_call_errors(ctx):
            ret = alternative.signature.check_call_preprocessed(actual_args, ctx)
        if not ret.is_error:
            returns.append(ret.return_value)
            if not ret.used_any_for_match and ret.remaining_arguments is None:
                break
    if not returns:
        return None
    if any(isinstance(ret, AnyValue) for ret in returns):
        return AnyValue(AnySource.multiple_overload_matches)
    return unite_values(*returns)


def _apply_overload_member(
    member: _Member, actual_args: ActualArguments, ctx: CheckCallContext
) -> Value | None:
    assert isinstance(member.original, OverloadedSignature)
    if _overload_member_matched_multiple_arms_with_any(member):
        return AnyValue(AnySource.multiple_overload_matches)

    signatures = member.original._prefer_variadic_matches(
        [alternative.signature for alternative in member.alternatives], actual_args
    )

    any_rets: list[CallReturn] = []
    union_rets: list[CallReturn] = []
    union_and_any_rets: list[CallReturn] = []
    last = len(signatures) - 1
    for i, signature in enumerate(signatures):
        with _catch_call_errors(ctx):
            ret = signature.check_call_preprocessed(
                actual_args, ctx, is_overload=i != last
            )
        if ret.is_error:
            continue
        if ret.remaining_arguments is not None:
            if ret.used_any_for_match:
                union_and_any_rets.append(ret)
            else:
                union_rets.append(ret)
            actual_args = ret.remaining_arguments
        elif ret.used_any_for_match:
            any_rets.append(ret)
        else:
            return member.original._unite_rets(
                any_rets, union_and_any_rets, union_rets, ret, ctx=ctx
            )

    if any_rets or union_rets or union_and_any_rets:
        return member.original._unite_rets(
            any_rets, union_and_any_rets, union_rets, ctx=ctx
        )
    return None


def _overload_member_matched_multiple_arms_with_any(member: _Member) -> bool:
    if len(member.alternatives) < 2:
        return False
    if (
        len({alternative.signature.return_value for alternative in member.alternatives})
        == 1
    ):
        return False
    return any(
        isinstance(composite.value, AnyValue)
        for alternative in member.alternatives
        for _position, composite in alternative.bound_args.values()
    )


def _find_split(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> tuple[ActualArguments, ActualArguments] | None:
    for member in members:
        for alternative in member.alternatives:
            for position, param_type in _iter_argument_parameter_types(alternative):
                current = _get_argument_value(actual_args, position)
                if current is None:
                    continue
                boundary = _top_materialization(param_type, ctx.can_assign_ctx)
                accepted = intersect_values(current, boundary, ctx.can_assign_ctx)
                rejected = intersect_values(
                    current, NotValue(boundary), ctx.can_assign_ctx
                )
                if accepted is NO_RETURN_VALUE or rejected is NO_RETURN_VALUE:
                    continue
                if accepted == current or rejected == current:
                    continue
                return (
                    _replace_argument_value(actual_args, position, accepted),
                    _replace_argument_value(actual_args, position, rejected),
                )
    return None


def _compute_return_type(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> Value:
    possible_regions = _possible_regions(members, actual_args, ctx)
    if not possible_regions:
        return NO_RETURN_VALUE
    possible_returns = [
        ret for _, _, ret in possible_regions if ret is not NO_RETURN_VALUE
    ]
    r_top = unite_values(*possible_returns)

    guaranteed_regions = _guaranteed_regions(members, actual_args, ctx)
    if guaranteed_regions:
        bottom_returns = []
        for region_args, accepting_members in guaranteed_regions:
            bottom_returns.append(
                _guaranteed_result(region_args, accepting_members, ctx)
            )
        r_bottom = unite_values(*bottom_returns)
    else:
        r_bottom = intersect_multi(possible_returns, ctx.can_assign_ctx)

    gradual_top = intersect_values(
        AnyValue(AnySource.explicit), r_top, ctx.can_assign_ctx
    )
    return unite_values(r_bottom, gradual_top)


def _possible_regions(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> list[tuple[ActualArguments, tuple[_Member, ...], Value]]:
    regions = []
    for shape_args in _iter_argument_shape_regions(members, actual_args):
        shape_members = _rematch_members(members, shape_args, ctx)
        if not shape_members:
            continue
        top_args = _materialize_actual_arguments(shape_args, _top_materialization, ctx)
        for subset in _nonempty_subsets(shape_members):
            region_args_list = [top_args]
            for member in shape_members:
                included = member in subset
                next_region_args_list = []
                for region_args in region_args_list:
                    next_region_args_list.extend(
                        _restrict_region(
                            region_args,
                            member,
                            included=included,
                            excluded_materialization=_bottom_materialization,
                            ctx=ctx,
                        )
                    )
                region_args_list = next_region_args_list
                if not region_args_list:
                    break
            for region_args in region_args_list:
                regional_return = _regional_return(subset, region_args, ctx)
                if regional_return is not NO_RETURN_VALUE:
                    regions.append((region_args, subset, regional_return))
    return regions


def _guaranteed_regions(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> list[tuple[ActualArguments, tuple[_Member, ...]]]:
    regions = []
    for shape_args in _iter_argument_shape_regions(members, actual_args):
        shape_members = _rematch_members(members, shape_args, ctx)
        if not shape_members:
            continue
        bottom_args = _materialize_actual_arguments(
            shape_args, _bottom_materialization, ctx
        )
        for subset in _nonempty_subsets(shape_members):
            region_args_list = [bottom_args]
            for member in shape_members:
                included = member in subset
                next_region_args_list = []
                for region_args in region_args_list:
                    next_region_args_list.extend(
                        _restrict_region(
                            region_args,
                            member,
                            included=included,
                            excluded_materialization=_top_materialization,
                            ctx=ctx,
                        )
                    )
                region_args_list = next_region_args_list
                if not region_args_list:
                    break
            for region_args in region_args_list:
                regions.append((region_args, subset))
    return regions


def _restrict_region(
    actual_args: ActualArguments,
    member: _Member,
    *,
    included: bool,
    excluded_materialization: Callable[[Value, CanAssignContext], Value],
    ctx: CheckCallContext,
) -> tuple[ActualArguments, ...]:
    if not included:
        return _exclude_member_region(
            actual_args, member, excluded_materialization, ctx
        )

    narrowed = actual_args
    for position in _argument_positions_for_member(member):
        current = _get_argument_value(narrowed, position)
        if current is None:
            continue
        domain = _member_parameter_domain(
            member, position, _top_materialization, ctx.can_assign_ctx
        )
        new_value = intersect_values(current, domain, ctx.can_assign_ctx)
        if new_value is NO_RETURN_VALUE:
            return ()
        narrowed = _replace_argument_value(narrowed, position, new_value)
    return (narrowed,)


def _exclude_member_region(
    actual_args: ActualArguments,
    member: _Member,
    excluded_materialization: Callable[[Value, CanAssignContext], Value],
    ctx: CheckCallContext,
) -> tuple[ActualArguments, ...]:
    if _has_unrestrictable_accepting_alternative(member):
        return ()

    regions = [actual_args]
    for alternative in member.alternatives:
        next_regions = []
        for region in regions:
            next_regions.extend(
                _exclude_alternative_region(
                    region, alternative, excluded_materialization, ctx
                )
            )
        regions = next_regions
        if not regions:
            break
    return tuple(regions)


def _exclude_alternative_region(
    actual_args: ActualArguments,
    alternative: _Alternative,
    excluded_materialization: Callable[[Value, CanAssignContext], Value],
    ctx: CheckCallContext,
) -> tuple[ActualArguments, ...]:
    regions = []
    prefix = actual_args
    for position, param_type in _iter_argument_parameter_types(alternative):
        current = _get_argument_value(prefix, position)
        if current is None:
            continue
        domain = excluded_materialization(param_type, ctx.can_assign_ctx)
        rejected = intersect_values(current, NotValue(domain), ctx.can_assign_ctx)
        if rejected is not NO_RETURN_VALUE:
            regions.append(_replace_argument_value(prefix, position, rejected))
        accepted = intersect_values(current, domain, ctx.can_assign_ctx)
        if accepted is NO_RETURN_VALUE:
            break
        prefix = _replace_argument_value(prefix, position, accepted)
    return tuple(regions)


def _regional_return(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> Value:
    returns = []
    for member in members:
        ret = _apply_member(member, actual_args, ctx)
        if ret is None:
            return NO_RETURN_VALUE
        returns.append(ret)
    return intersect_multi(returns, ctx.can_assign_ctx)


def _guaranteed_result(
    region_args: ActualArguments,
    accepting_members: Sequence[_Member],
    ctx: CheckCallContext,
) -> Value:
    contributions = []
    for kept in _nonempty_subsets(accepting_members):
        omitted = [member for member in accepting_members if member not in kept]
        if omitted and _valid_with_bottom_domains(omitted, region_args, ctx):
            continue
        contribution = _regional_return(kept, region_args, ctx)
        if contribution is not NO_RETURN_VALUE:
            contributions.append(contribution)
    return unite_values(*contributions)


def _valid_with_bottom_domains(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> bool:
    bottom_domain_members = tuple(
        _Member(
            member.original,
            tuple(
                _Alternative(
                    replace(
                        alternative.signature,
                        parameters={
                            name: replace(
                                param,
                                annotation=_bottom_materialization(
                                    param.annotation, ctx.can_assign_ctx
                                ),
                            )
                            for name, param in alternative.signature.parameters.items()
                        },
                    ),
                    alternative.bound_args,
                )
                for alternative in member.alternatives
            ),
        )
        for member in members
    )
    return _valid(bottom_domain_members, actual_args, ctx)


def _nonempty_subsets(items: Sequence[_Member]) -> Iterable[tuple[_Member, ...]]:
    for size in range(1, len(items) + 1):
        yield from itertools.combinations(items, size)


def _rematch_members(
    members: Sequence[_Member], actual_args: ActualArguments, ctx: CheckCallContext
) -> tuple[_Member, ...]:
    rematched = []
    for member in members:
        alternatives = []
        for alternative in member.alternatives:
            with _catch_call_errors(ctx):
                bound_args = alternative.signature.bind_arguments(actual_args, ctx)
            if bound_args is not None:
                alternatives.append(_Alternative(alternative.signature, bound_args))
        if alternatives:
            rematched.append(_Member(member.original, tuple(alternatives)))
    return tuple(rematched)


def _iter_argument_shape_regions(
    members: Sequence[_Member], actual_args: ActualArguments
) -> Iterable[ActualArguments]:
    shape_regions = []
    if actual_args.star_args is not None:
        for member in members:
            for alternative in member.alternatives:
                shape_region = _star_args_shape_region(alternative, actual_args)
                if shape_region is not None and not any(
                    _actual_arguments_key(existing)
                    == _actual_arguments_key(shape_region)
                    for existing in shape_regions
                ):
                    shape_regions.append(shape_region)

    if shape_regions:
        yield from shape_regions
    else:
        yield actual_args


def _star_args_shape_region(
    alternative: _Alternative, actual_args: ActualArguments
) -> ActualArguments | None:
    if actual_args.star_args is None:
        return None

    star_filled = [
        name
        for name, (position, _composite) in alternative.bound_args.items()
        if position is ARGS
        and alternative.signature.parameters[name].kind
        in (ParameterKind.POSITIONAL_ONLY, ParameterKind.POSITIONAL_OR_KEYWORD)
    ]
    if not star_filled:
        return None

    positionals = list(actual_args.positionals)
    positionals.extend(
        (True, Composite(actual_args.star_args)) for _ in range(len(star_filled))
    )
    residual_star_args = (
        actual_args.star_args
        if any(
            position is ARGS
            and alternative.signature.parameters[name].kind
            is ParameterKind.VAR_POSITIONAL
            for name, (position, _composite) in alternative.bound_args.items()
        )
        else None
    )
    return replace(actual_args, positionals=positionals, star_args=residual_star_args)


def _iter_argument_parameter_types(
    alternative: _Alternative,
) -> Iterable[tuple[int | str, Value]]:
    for name, (position, _composite) in alternative.bound_args.items():
        if isinstance(position, (int, str)):
            yield position, alternative.signature.parameters[name].annotation


def _has_unrestrictable_accepting_alternative(member: _Member) -> bool:
    return any(
        not any(_iter_argument_parameter_types(alternative))
        and all(
            param.kind is not ParameterKind.ELLIPSIS
            for param in alternative.signature.parameters.values()
        )
        for alternative in member.alternatives
    )


def _argument_positions_for_member(member: _Member) -> tuple[int | str, ...]:
    positions = []
    for alternative in member.alternatives:
        for position, _ in _iter_argument_parameter_types(alternative):
            if position not in positions:
                positions.append(position)
    return tuple(positions)


def _member_parameter_domain(
    member: _Member,
    position: int | str,
    materialize: Callable[[Value, CanAssignContext], Value],
    ctx: CanAssignContext,
) -> Value:
    domains = []
    for alternative in member.alternatives:
        for candidate_position, param_type in _iter_argument_parameter_types(
            alternative
        ):
            if candidate_position == position:
                domains.append(materialize(param_type, ctx))
    return unite_values(*domains)


def _get_argument_value(
    actual_args: ActualArguments, position: int | str
) -> Value | None:
    if isinstance(position, int):
        if position >= len(actual_args.positionals):
            return None
        return actual_args.positionals[position][1].value
    if position not in actual_args.keywords:
        return None
    return actual_args.keywords[position][1].value


def _replace_argument_value(
    actual_args: ActualArguments, position: int | str, value: Value
) -> ActualArguments:
    if isinstance(position, int):
        positionals = list(actual_args.positionals)
        definitely_provided, composite = positionals[position]
        positionals[position] = (
            definitely_provided,
            Composite(value, composite.varname, composite.node),
        )
        return replace(actual_args, positionals=positionals)
    keywords = dict(actual_args.keywords)
    definitely_provided, composite = keywords[position]
    keywords[position] = definitely_provided, Composite(
        value, composite.varname, composite.node
    )
    return replace(actual_args, keywords=keywords)


def _materialize_actual_arguments(
    actual_args: ActualArguments,
    materialize: Callable[[Value, CanAssignContext], Value],
    ctx: CheckCallContext,
) -> ActualArguments:
    positionals = [
        (
            definitely_provided,
            Composite(
                materialize(composite.value, ctx.can_assign_ctx),
                composite.varname,
                composite.node,
            ),
        )
        for definitely_provided, composite in actual_args.positionals
    ]
    keywords = {
        name: (
            definitely_provided,
            Composite(
                materialize(composite.value, ctx.can_assign_ctx),
                composite.varname,
                composite.node,
            ),
        )
        for name, (definitely_provided, composite) in actual_args.keywords.items()
    }
    star_args = (
        materialize(actual_args.star_args, ctx.can_assign_ctx)
        if actual_args.star_args is not None
        else None
    )
    star_kwargs = (
        materialize(actual_args.star_kwargs, ctx.can_assign_ctx)
        if actual_args.star_kwargs is not None
        else None
    )
    return replace(
        actual_args,
        positionals=positionals,
        keywords=keywords,
        star_args=star_args,
        star_kwargs=star_kwargs,
    )


def _actual_arguments_key(actual_args: ActualArguments) -> tuple[str, ...]:
    pieces = [str(composite.value) for _, composite in actual_args.positionals]
    pieces.extend(
        f"{name}={composite.value}"
        for name, (_, composite) in actual_args.keywords.items()
    )
    if actual_args.star_args is not None:
        pieces.append(f"*={actual_args.star_args}")
    if actual_args.star_kwargs is not None:
        pieces.append(f"**={actual_args.star_kwargs}")
    return tuple(pieces)


def _top_materialization(value: Value, ctx: CanAssignContext) -> Value:
    value = gradualize(value)
    match value:
        case AnnotatedValue(value=inner):
            return _top_materialization(inner, ctx)
        case AnyValue():
            return TypedValue(object)
        case MultiValuedValue(vals=vals):
            return unite_values(*[_top_materialization(subval, ctx) for subval in vals])
        case IntersectionValue(vals=vals):
            return intersect_multi(
                [_top_materialization(subval, ctx) for subval in vals], ctx
            )
        case NotValue(value=inner):
            return gradualize(NotValue(_bottom_materialization(inner, ctx)))
        case TypedDictValue():
            # TODO: The real materialization behavior for gradual TypedDicts may need
            # to materialize item types while preserving TypedDict structure.
            return value
        case SequenceValue(typ=typ, members=members):
            return SequenceValue(
                typ,
                [
                    (is_many, _top_materialization(member, ctx))
                    for is_many, member in members
                ],
            )
        case CallableValue(signature=signature, typ=fallback):
            return CallableValue(
                _materialize_callable_signature(
                    signature,
                    parameter_materialize=_bottom_materialization,
                    return_materialize=_top_materialization,
                    ctx=ctx,
                ),
                fallback,
            )
        case GenericValue(typ=typ, args=args, weak=weak):
            top_args = []
            for arg, variance in zip(args, _generic_variances(value, ctx)):
                if variance is Variance.COVARIANT:
                    top_args.append(_top_materialization(arg, ctx))
                elif variance is Variance.CONTRAVARIANT:
                    top_args.append(_bottom_materialization(arg, ctx))
                else:
                    arg_top = _top_materialization(arg, ctx)
                    arg_bottom = _bottom_materialization(arg, ctx)
                    if arg_top != arg or arg_bottom != arg:
                        return TypedValue(typ)
                    top_args.append(arg)
            return GenericValue(typ, tuple(top_args), weak=weak)
        case (
            TypeAliasValue()
            | NewTypeValue()
            | OverlappingValue()
            | TypeVarValue()
            | TypeVarTupleBindingValue()
            | TypeVarTupleValue()
            | ParamSpecArgsValue()
            | ParamSpecKwargsValue()
            | PartialValue()
            | PartialCallValue()
            | SyntheticTypeFormValue()
            | SuperValue()
            | TypeFormValue()
        ):
            # TODO: This fallback-based handling preserves legacy pycroscope behavior
            # but flattens distinctions that matter to the materialization model,
            # such as aliases, NewType identity, TypeVars, and synthetic wrappers.
            return _top_materialization(value.get_fallback_value(), ctx)
        case (
            KnownValue()
            | SyntheticClassObjectValue()
            | SyntheticModuleValue()
            | UnboundMethodValue()
            | TypedValue()
            | SubclassValue()
            | PredicateValue()
        ):
            return value
        case _:
            assert_never(value)


def _bottom_materialization(value: Value, ctx: CanAssignContext) -> Value:
    value = gradualize(value)
    match value:
        case AnnotatedValue(value=inner):
            return _bottom_materialization(inner, ctx)
        case AnyValue():
            return NO_RETURN_VALUE
        case MultiValuedValue(vals=vals):
            return unite_values(
                *[_bottom_materialization(subval, ctx) for subval in vals]
            )
        case IntersectionValue(vals=vals):
            return intersect_multi(
                [_bottom_materialization(subval, ctx) for subval in vals], ctx
            )
        case NotValue(value=inner):
            return gradualize(NotValue(_top_materialization(inner, ctx)))
        case TypedDictValue():
            # TODO: The real materialization behavior for gradual TypedDicts may need
            # to materialize item types while preserving TypedDict structure.
            return value
        case SequenceValue(typ=typ, members=members):
            return SequenceValue(
                typ,
                [
                    (is_many, _bottom_materialization(member, ctx))
                    for is_many, member in members
                ],
            )
        case CallableValue(signature=signature, typ=fallback):
            return CallableValue(
                _materialize_callable_signature(
                    signature,
                    parameter_materialize=_top_materialization,
                    return_materialize=_bottom_materialization,
                    ctx=ctx,
                ),
                fallback,
            )
        case GenericValue(typ=typ, args=args, weak=weak):
            bottom_args = []
            for arg, variance in zip(args, _generic_variances(value, ctx)):
                if variance is Variance.COVARIANT:
                    bottom_args.append(_bottom_materialization(arg, ctx))
                elif variance is Variance.CONTRAVARIANT:
                    bottom_args.append(_top_materialization(arg, ctx))
                else:
                    bottom_args.append(_bottom_materialization(arg, ctx))
            return GenericValue(typ, tuple(bottom_args), weak=weak)
        case (
            TypeAliasValue()
            | NewTypeValue()
            | OverlappingValue()
            | TypeVarValue()
            | TypeVarTupleBindingValue()
            | TypeVarTupleValue()
            | ParamSpecArgsValue()
            | ParamSpecKwargsValue()
            | PartialValue()
            | PartialCallValue()
            | SyntheticTypeFormValue()
            | SuperValue()
            | TypeFormValue()
        ):
            # TODO: This fallback-based handling preserves legacy pycroscope behavior
            # but flattens distinctions that matter to the materialization model,
            # such as aliases, NewType identity, TypeVars, and synthetic wrappers.
            return _bottom_materialization(value.get_fallback_value(), ctx)
        case (
            KnownValue()
            | SyntheticClassObjectValue()
            | SyntheticModuleValue()
            | UnboundMethodValue()
            | TypedValue()
            | SubclassValue()
            | PredicateValue()
        ):
            return value
        case _:
            assert_never(value)


def _materialize_callable_signature(
    signature: ConcreteSignature,
    *,
    parameter_materialize: Callable[[Value, CanAssignContext], Value],
    return_materialize: Callable[[Value, CanAssignContext], Value],
    ctx: CanAssignContext,
) -> ConcreteSignature:
    if isinstance(signature, OverloadedSignature):
        return OverloadedSignature(
            [
                _materialize_signature(
                    alternative,
                    parameter_materialize=parameter_materialize,
                    return_materialize=return_materialize,
                    ctx=ctx,
                )
                for alternative in signature.signatures
            ]
        )
    return _materialize_signature(
        signature,
        parameter_materialize=parameter_materialize,
        return_materialize=return_materialize,
        ctx=ctx,
    )


def _materialize_signature(
    signature: Signature,
    *,
    parameter_materialize: Callable[[Value, CanAssignContext], Value],
    return_materialize: Callable[[Value, CanAssignContext], Value],
    ctx: CanAssignContext,
) -> Signature:
    return replace(
        signature,
        parameters={
            name: replace(
                param, annotation=parameter_materialize(param.annotation, ctx)
            )
            for name, param in signature.parameters.items()
        },
        return_value=return_materialize(signature.return_value, ctx),
    )


def _generic_variances(
    value: GenericValue, ctx: CanAssignContext
) -> Sequence[Variance]:
    type_params = ctx.get_type_parameters(value.typ)
    if len(type_params) == len(value.args):
        return [param.variance for param in type_params]
    return (Variance.INVARIANT,) * len(value.args)
