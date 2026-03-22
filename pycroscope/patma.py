"""

Visitor for pattern matching.

"""

import ast
import collections.abc
import enum
import itertools
from collections.abc import Callable, Container, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

import pycroscope

from .analysis_lib import override, set_inferred_value
from .annotations import type_from_value
from .error_code import ErrorCode
from .extensions import CustomCheck
from .predicates import EqualsPredicate, IsAssignablePredicate, MaxLen, MinLen
from .relations import Relation, has_relation, intersect_values
from .signature import MappingValue
from .stacked_scopes import (
    NULL_CONSTRAINT,
    AbstractConstraint,
    AndConstraint,
    Composite,
    Constraint,
    ConstraintType,
    OrConstraint,
    constrain_value,
)
from .value import (
    NO_RETURN_VALUE,
    UNINITIALIZED_VALUE,
    VOID,
    AnnotatedValue,
    AnySource,
    AnyValue,
    CanAssign,
    CanAssignContext,
    CanAssignError,
    CustomCheckExtension,
    DictIncompleteValue,
    KnownValue,
    KVPair,
    OverlapMode,
    PredicateValue,
    SequenceValue,
    SubclassValue,
    TypedValue,
    Value,
    flatten_values,
    kv_pairs_from_mapping,
    len_of_value,
    replace_known_sequence_value,
    unite_values,
    unpack_values,
)

try:
    from ast import (
        MatchAs,
        MatchClass,
        MatchMapping,
        MatchOr,
        MatchSequence,
        MatchSingleton,
        MatchStar,
        MatchValue,
    )
except ImportError:
    # 3.9 and lower
    MatchAs = MatchClass = MatchMapping = Any
    MatchOr = MatchSequence = MatchSingleton = MatchValue = Any

    # Avoid false positive errors on isinstance() in 3.9 self check
    class MatchStar(ast.AST):
        pass


# For these types, a single class subpattern matches the whole thing
_SPECIAL_CLASS_PATTERN_TYPES = {
    bool,
    bytearray,
    bytes,
    dict,
    float,
    frozenset,
    int,
    list,
    set,
    str,
    tuple,
}
SpecialClassPatternValue = unite_values(
    *[SubclassValue(TypedValue(typ)) for typ in _SPECIAL_CLASS_PATTERN_TYPES]
)


class SpecialPositionalMatch(enum.Enum):
    self = 1  # match against self (special behavior for builtins)
    error = 2  # couldn't figure out the attr, match against Any


@dataclass(frozen=True)
class Exclude(CustomCheck):
    """A CustomCheck that excludes certain types."""

    excluded: Value

    def can_assign(self, value: Value, ctx: CanAssignContext) -> CanAssign:
        for subval in flatten_values(value, unwrap_annotated=True):
            if isinstance(subval, AnyValue):
                continue
            can_assign = has_relation(self.excluded, subval, Relation.ASSIGNABLE, ctx)
            if not isinstance(can_assign, CanAssignError):
                return CanAssignError(
                    f"{subval} is compatible with excluded type {self.excluded}"
                )
        return {}

    def __str__(self) -> str:
        return f"Exclude[{self.excluded}]"


T = TypeVar("T")
MatchableSequence = AnnotatedValue(
    TypedValue(collections.abc.Sequence),
    [
        CustomCheckExtension(
            Exclude(TypedValue(str) | TypedValue(bytes) | TypedValue(bytearray))
        )
    ],
)


@dataclass
class LenPredicate:
    expected_length: int
    has_star: bool
    ctx: CanAssignContext

    def __call__(self, value: Value, positive: bool) -> Value | None:
        value_len = len_of_value(value)
        if isinstance(value_len, KnownValue) and isinstance(value_len.val, int):
            if self.has_star:
                match = value_len.val >= self.expected_length
            else:
                match = value_len.val == self.expected_length
            if not positive:
                match = not match
            if match:
                return value
            else:
                return None

        if self.has_star:
            # We don't currently model negative sequence-length constraints from patterns,
            # and positive star-pattern length checks are usually not very informative.
            return value

        if positive:
            narrowed = self._narrow_with_bounds(
                value, self.expected_length, self.expected_length
            )
            if narrowed is None:
                return None
        else:
            # We currently cannot represent "len(value) != N" precisely in general.
            narrowed = value

        return narrowed

    def _narrow_with_bounds(
        self, value: Value, min_len: int | None, max_len: int | None
    ) -> Value | None:
        narrowed: Value = value
        if min_len is not None:
            narrowed = intersect_values(
                narrowed, PredicateValue(MinLen(min_len)), self.ctx
            )
            if narrowed is NO_RETURN_VALUE:
                return None
        if max_len is not None:
            narrowed = intersect_values(
                narrowed, PredicateValue(MaxLen(max_len)), self.ctx
            )
            if narrowed is NO_RETURN_VALUE:
                return None
        return narrowed


class PatternMatchability(enum.Enum):
    NEVER = 1
    MAYBE = 2
    ALWAYS = 3


ClassPatternName = str | SpecialPositionalMatch
ClassPattern = tuple[ClassPatternName, ast.AST]
ClassPatternStatus = CanAssignError | list[ClassPatternName] | None


@dataclass
class StructuralPatternPredicate:
    pattern: ast.AST
    patma_visitor: "PatmaVisitor"

    def __call__(self, value: Value, positive: bool) -> Value | None:
        matching_values = []
        for subval in flatten_values(value, unwrap_annotated=True):
            matchability = self.patma_visitor.pattern_matchability(self.pattern, subval)
            if positive:
                if matchability is not PatternMatchability.NEVER:
                    matching_values.append(subval)
            elif matchability is not PatternMatchability.ALWAYS:
                matching_values.append(subval)
        if not matching_values:
            return None
        return unite_values(*matching_values)


@dataclass
class PatmaVisitor(ast.NodeVisitor):
    visitor: "pycroscope.name_check_visitor.NameCheckVisitor"
    _match_value_cache: dict[int, Value] = field(default_factory=dict)
    _mapping_key_cache: dict[int, list[Value]] = field(default_factory=dict)
    _class_pattern_info_cache: dict[
        int, tuple[Value, Value, list[ClassPattern], ClassPatternStatus]
    ] = field(default_factory=dict)

    def visit(self, node: ast.AST) -> AbstractConstraint:
        constraint = super().visit(node)
        if self.visitor.annotate:
            set_inferred_value(node, VOID)
        return constraint

    def visit_MatchSingleton(self, node: MatchSingleton) -> AbstractConstraint:
        self.check_impossible_pattern(node, KnownValue(node.value))
        return self.make_constraint(
            ConstraintType.predicate,
            EqualsPredicate(node.value, self.visitor, use_is=True),
        )

    def visit_MatchValue(self, node: MatchValue) -> AbstractConstraint:
        pattern_val = self.visitor.visit(node.value)
        self._match_value_cache[id(node)] = pattern_val
        self.check_impossible_pattern(node, pattern_val)
        if not isinstance(pattern_val, KnownValue):
            self.visitor.show_error(
                node,
                f"Match value is not a literal: {pattern_val}",
                ErrorCode.internal_error,
            )
            return NULL_CONSTRAINT

        return self.make_constraint(
            ConstraintType.predicate, EqualsPredicate(pattern_val.val, self.visitor)
        )

    def visit_MatchSequence(self, node: MatchSequence) -> AbstractConstraint:
        self.check_impossible_pattern(node, MatchableSequence)
        starred_index = index_of(node.patterns, lambda pat: isinstance(pat, MatchStar))
        if starred_index is None:
            target_length = len(node.patterns)
            post_starred_length = None
        else:
            target_length = starred_index
            post_starred_length = len(node.patterns) - 1 - target_length
        constraints = [
            self.intersect_with(
                PredicateValue(
                    MinLen(len(node.patterns) - int(starred_index is not None))
                )
            )
        ]
        if starred_index is None:
            constraints.append(
                self.intersect_with(PredicateValue(MaxLen(len(node.patterns))))
            )
        can_assign_seq = has_relation(
            MatchableSequence,
            self.visitor.match_subject.value,
            Relation.ASSIGNABLE,
            self.visitor,
        )
        if isinstance(can_assign_seq, CanAssignError):
            constraints.append(
                self.make_constraint(
                    ConstraintType.predicate,
                    IsAssignablePredicate(
                        MatchableSequence,
                        self.visitor,
                        positive_only=len(node.patterns) > 1 or starred_index is None,
                    ),
                )
            )
        unpacked = unpack_values(
            constrain_value(
                self.visitor.match_subject.value,
                AndConstraint.make(constraints),
                ctx=self.visitor,
            ),
            self.visitor,
            target_length,
            post_starred_length,
        )
        if isinstance(unpacked, CanAssignError):
            unpacked = itertools.repeat(AnyValue(AnySource.generic_argument))
        subjects: Iterable[Value] = unpacked
        for pat, subject in zip(node.patterns, subjects):
            with override(self.visitor, "match_subject", Composite(subject)):
                constraints.append(self.visit(pat))
        return AndConstraint.make(constraints)

    def visit_MatchMapping(self, node: MatchMapping) -> AbstractConstraint:
        self.check_impossible_pattern(node, MappingValue)
        constraint = self.make_constraint(
            ConstraintType.predicate,
            IsAssignablePredicate(
                MappingValue, self.visitor, positive_only=len(node.keys) > 0
            ),
        )
        constraints = [constraint]
        subject = constrain_value(
            self.visitor.match_subject.value, constraint, ctx=self.visitor
        )
        key_values = [self.visitor.visit(key) for key in node.keys]
        self._mapping_key_cache[id(node)] = key_values
        kv_pairs = kv_pairs_from_mapping(subject, self.visitor)
        if isinstance(kv_pairs, CanAssignError):
            kv_pairs = [
                KVPair(
                    AnyValue(AnySource.generic_argument),
                    AnyValue(AnySource.generic_argument),
                )
            ]
        kv_pairs = list(reversed(kv_pairs))
        optional_pairs: set[KVPair] = set()
        removed_pairs: set[KVPair] = set()
        for key_val, pattern in zip(key_values, node.patterns):
            value, new_optional_pairs, new_removed_pairs = get_value_from_kv_pairs(
                kv_pairs, key_val, self.visitor, optional_pairs, removed_pairs
            )
            optional_pairs |= new_optional_pairs
            removed_pairs |= new_removed_pairs
            if value is UNINITIALIZED_VALUE:
                self.visitor.show_error(
                    node,
                    f"Impossible pattern: {self.visitor.match_subject.value} has no"
                    f" key {key_val}",
                    ErrorCode.impossible_pattern,
                )
                value = AnyValue(AnySource.error)
            with override(self.visitor, "match_subject", Composite(value)):
                constraints.append(self.visit(pattern))
        if node.rest is not None:
            new_kv_pairs = []
            for kv_pair in kv_pairs:
                if kv_pair in removed_pairs:
                    continue
                if kv_pair in optional_pairs:
                    kv_pair = replace(kv_pair, is_required=False)
                new_kv_pairs.append(kv_pair)
            val = DictIncompleteValue(dict, list(reversed(new_kv_pairs)))
            self.visitor._set_name_in_scope(node.rest, node, val)
        return AndConstraint.make(constraints)

    def visit_MatchClass(self, node: MatchClass) -> AbstractConstraint:
        cls, matched_type, patterns, match_args_status = self._get_class_pattern_info(
            node
        )
        can_assign = has_relation(
            TypedValue(type), cls, Relation.ASSIGNABLE, self.visitor
        )
        if isinstance(can_assign, CanAssignError):
            self.visitor.show_error(
                node.cls,
                "Class pattern must be a type",
                ErrorCode.bad_match,
                detail=str(can_assign),
            )
        self.check_impossible_pattern(node, matched_type)
        if isinstance(match_args_status, CanAssignError):
            self.visitor.show_error(
                node.cls,
                "Invalid class pattern",
                ErrorCode.bad_match,
                detail=str(match_args_status),
            )
        elif isinstance(match_args_status, list):
            self.visitor.show_error(
                node.cls,
                f"{cls} takes at most {len(match_args_status)} positional subpatterns,"
                f" but {len(node.patterns)} were provided",
                ErrorCode.bad_match,
                detail=str(match_args_status),
            )
        constraint = self.make_constraint(
            ConstraintType.predicate,
            IsAssignablePredicate(
                matched_type,
                self.visitor,
                positive_only=bool(node.patterns or node.kwd_patterns),
            ),
        )
        subject = constrain_value(
            self.visitor.match_subject.value, constraint, ctx=self.visitor
        )
        subject_composite = self.visitor.match_subject._replace(value=subject)

        seen_names = set()
        for name, _ in patterns:
            if isinstance(name, str):
                if name in seen_names:
                    self.visitor.show_error(
                        node, f"Duplicate keyword pattern {name}", ErrorCode.bad_match
                    )
                seen_names.add(name)

        constraints = [constraint]
        for name, subpattern in patterns:
            if name is SpecialPositionalMatch.self:
                subsubject = subject_composite
            elif name is SpecialPositionalMatch.error:
                subsubject = Composite(AnyValue(AnySource.error))
            else:
                assert isinstance(name, str)
                attr = self.visitor.get_attribute(subject_composite, name)
                if attr is UNINITIALIZED_VALUE:
                    # It may exist on a child class, so we don't error here.
                    # This matches pyright's behavior.
                    subsubject = Composite(AnyValue(AnySource.unreachable))
                else:
                    new_varname = self.visitor._extend_composite(
                        subject_composite, name, subpattern
                    )
                    subsubject = Composite(attr, new_varname)
            with override(self.visitor, "match_subject", subsubject):
                constraints.append(self.visit(subpattern))

        return AndConstraint.make(constraints)

    def visit_MatchStar(self, node: MatchStar) -> AbstractConstraint:
        if node.name is not None:
            self.visitor._set_name_in_scope(
                node.name, node, self.visitor.match_subject.value
            )
        return self.intersect_with(TypedValue(object))

    def visit_MatchAs(self, node: MatchAs) -> AbstractConstraint:
        val = self.visitor.match_subject.value
        if node.pattern is None:
            constraint = self.intersect_with(TypedValue(object))
        else:
            constraint = self.visit(node.pattern)

        if node.name is not None:
            val = constrain_value(val, constraint, ctx=self.visitor)
            self.visitor._set_name_in_scope(node.name, node, val)

        return constraint

    def visit_MatchOr(self, node: MatchOr) -> AbstractConstraint:
        subscopes = []
        constraints = []
        for pattern in node.patterns:
            with self.visitor.scopes.subscope() as subscope:
                constraints.append(self.visit(pattern))
                subscopes.append(subscope)
        self.visitor.scopes.combine_subscopes(subscopes)
        return OrConstraint.make(constraints)

    def generic_visit(self, node: ast.AST) -> AbstractConstraint:
        raise NotImplementedError(f"Unsupported pattern node: {node}")

    def make_constraint(self, typ: ConstraintType, value: object) -> AbstractConstraint:
        varname = self.visitor.match_subject.varname
        return Constraint(varname, typ, True, value)

    def intersect_with(self, value: Value) -> AbstractConstraint:
        return self.make_constraint(ConstraintType.intersect_with, value)

    def make_structural_constraint(self, node: ast.AST) -> AbstractConstraint:
        return self.make_constraint(
            ConstraintType.predicate, StructuralPatternPredicate(node, self)
        )

    def pattern_matchability(self, node: ast.AST, value: Value) -> PatternMatchability:
        if isinstance(node, MatchSingleton):
            pattern_val = KnownValue(node.value)
            if (
                value.can_overlap(pattern_val, self.visitor, OverlapMode.MATCH)
                is not None
            ):
                return PatternMatchability.NEVER
            if isinstance(value, KnownValue):
                op = (
                    value.val is node.value
                    if node.value in (None, True, False)
                    else False
                )
                return PatternMatchability.ALWAYS if op else PatternMatchability.NEVER
            if isinstance(value, TypedValue) and (
                (node.value is None and value.typ is type(None))
                or (node.value in (True, False) and value.typ is bool)
            ):
                return PatternMatchability.MAYBE
            return PatternMatchability.MAYBE
        elif isinstance(node, MatchValue):
            pattern_val = self._match_value_cache.get(id(node))
            if pattern_val is None:
                return PatternMatchability.MAYBE
            if (
                value.can_overlap(pattern_val, self.visitor, OverlapMode.MATCH)
                is not None
            ):
                return PatternMatchability.NEVER
            if isinstance(pattern_val, KnownValue) and isinstance(value, KnownValue):
                return (
                    PatternMatchability.ALWAYS
                    if value.val == pattern_val.val
                    else PatternMatchability.NEVER
                )
            return PatternMatchability.MAYBE
        elif isinstance(node, MatchSequence):
            outer = self._outer_matchability(MatchableSequence, value)
            if outer is PatternMatchability.NEVER:
                return outer
            length_matchability = self._sequence_length_matchability(node, value)
            if length_matchability is PatternMatchability.NEVER:
                return length_matchability
            subpattern_matchability = self._sequence_subpattern_matchability(
                node, value
            )
            return self._combine_matchabilities(
                outer, length_matchability, subpattern_matchability
            )
        elif isinstance(node, MatchMapping):
            outer = self._outer_matchability(MappingValue, value)
            if outer is PatternMatchability.NEVER:
                return outer
            key_values = self._mapping_key_cache.get(id(node))
            if key_values is None:
                return PatternMatchability.MAYBE
            kv_pairs = kv_pairs_from_mapping(value, self.visitor)
            if isinstance(kv_pairs, CanAssignError):
                return PatternMatchability.MAYBE
            optional_pairs: set[KVPair] = set()
            removed_pairs: set[KVPair] = set()
            results = []
            for key_val, pattern in zip(key_values, node.patterns):
                subvalue, new_optional_pairs, new_removed_pairs = (
                    get_value_from_kv_pairs(
                        kv_pairs, key_val, self.visitor, optional_pairs, removed_pairs
                    )
                )
                optional_pairs |= new_optional_pairs
                removed_pairs |= new_removed_pairs
                if subvalue is UNINITIALIZED_VALUE:
                    return PatternMatchability.MAYBE
                results.append(self.pattern_matchability(pattern, subvalue))
            return self._combine_matchabilities(outer, *results)
        elif isinstance(node, MatchClass):
            return self._class_pattern_matchability(node, value)
        elif isinstance(node, MatchStar):
            return PatternMatchability.ALWAYS
        elif isinstance(node, MatchAs):
            if node.pattern is None:
                return PatternMatchability.ALWAYS
            return self.pattern_matchability(node.pattern, value)
        elif isinstance(node, MatchOr):
            return self._or_matchability(
                self.pattern_matchability(pattern, value) for pattern in node.patterns
            )
        raise NotImplementedError(f"Unsupported pattern node: {node}")

    def check_impossible_pattern(self, node: ast.AST, value: Value) -> None:
        error = self.visitor.match_subject.value.can_overlap(
            value, self.visitor, OverlapMode.MATCH
        )
        if error is not None:
            self.visitor.show_error(
                node,
                f"Impossible pattern: {self.visitor.match_subject.value} can never"
                f" be {value}",
                ErrorCode.impossible_pattern,
                detail=str(error),
            )

    def _outer_matchability(
        self, pattern_value: Value, value: Value
    ) -> PatternMatchability:
        if (
            value.can_overlap(pattern_value, self.visitor, OverlapMode.MATCH)
            is not None
        ):
            return PatternMatchability.NEVER
        if pattern_value.is_assignable(value, self.visitor):
            return PatternMatchability.ALWAYS
        return PatternMatchability.MAYBE

    def _combine_matchabilities(
        self, *matchabilities: PatternMatchability
    ) -> PatternMatchability:
        if any(
            matchability is PatternMatchability.NEVER for matchability in matchabilities
        ):
            return PatternMatchability.NEVER
        if all(
            matchability is PatternMatchability.ALWAYS
            for matchability in matchabilities
        ):
            return PatternMatchability.ALWAYS
        return PatternMatchability.MAYBE

    def _or_matchability(
        self, matchabilities: Iterable[PatternMatchability]
    ) -> PatternMatchability:
        matchabilities = list(matchabilities)
        if any(
            matchability is PatternMatchability.ALWAYS
            for matchability in matchabilities
        ):
            return PatternMatchability.ALWAYS
        if any(
            matchability is PatternMatchability.MAYBE for matchability in matchabilities
        ):
            return PatternMatchability.MAYBE
        return PatternMatchability.NEVER

    def _sequence_length_matchability(
        self, node: MatchSequence, value: Value
    ) -> PatternMatchability:
        starred_index = index_of(node.patterns, lambda pat: isinstance(pat, MatchStar))
        expected_length = len(node.patterns) - int(starred_index is not None)
        value_len = len_of_value(value)
        if isinstance(value_len, KnownValue) and isinstance(value_len.val, int):
            if starred_index is None:
                matches = value_len.val == expected_length
            else:
                matches = value_len.val >= expected_length
            return PatternMatchability.ALWAYS if matches else PatternMatchability.NEVER
        return PatternMatchability.MAYBE

    def _sequence_subpattern_matchability(
        self, node: MatchSequence, value: Value
    ) -> PatternMatchability:
        starred_index = index_of(node.patterns, lambda pat: isinstance(pat, MatchStar))
        if starred_index is None:
            target_length = len(node.patterns)
            post_starred_length = None
        else:
            target_length = starred_index
            post_starred_length = len(node.patterns) - 1 - target_length
        narrowed = LenPredicate(
            len(node.patterns) - int(starred_index is not None),
            starred_index is not None,
            self.visitor,
        )(value, True)
        if narrowed is None:
            return PatternMatchability.NEVER
        unpacked = unpack_values(
            narrowed, self.visitor, target_length, post_starred_length
        )
        if isinstance(unpacked, CanAssignError):
            return PatternMatchability.MAYBE
        results = [
            self.pattern_matchability(pattern, subject)
            for pattern, subject in zip(node.patterns, unpacked)
        ]
        return self._combine_matchabilities(*results)

    def _class_pattern_matchability(
        self, node: MatchClass, value: Value
    ) -> PatternMatchability:
        info = self._class_pattern_info_cache.get(id(node))
        if info is None:
            return PatternMatchability.MAYBE
        _, matched_type, patterns, _ = info
        outer = self._outer_matchability(matched_type, value)
        if outer is PatternMatchability.NEVER:
            return outer

        subject_composite = Composite(value)
        results = []
        for name, subpattern in patterns:
            if name is SpecialPositionalMatch.self:
                subvalue = value
            elif name is SpecialPositionalMatch.error:
                return PatternMatchability.MAYBE
            else:
                assert isinstance(name, str)
                attr = self.visitor.get_attribute(
                    subject_composite, name, record_reads=False
                )
                if attr is UNINITIALIZED_VALUE:
                    return PatternMatchability.MAYBE
                subvalue = attr
            results.append(self.pattern_matchability(subpattern, subvalue))
        return self._combine_matchabilities(outer, *results)

    def _get_class_pattern_info(
        self, node: MatchClass
    ) -> tuple[Value, Value, list[ClassPattern], ClassPatternStatus]:
        if cached := self._class_pattern_info_cache.get(id(node)):
            return cached

        cls = self.visitor.visit(node.cls)
        matched_type = type_from_value(cls, visitor=self.visitor, node=node.cls)
        patterns: list[ClassPattern] = [
            (attr, pattern) for attr, pattern in zip(node.kwd_attrs, node.kwd_patterns)
        ]
        match_args_status: ClassPatternStatus = None
        if node.patterns:
            match_args = get_match_args(cls, self.visitor)
            if isinstance(match_args, CanAssignError):
                match_args_status = match_args
                patterns = [
                    *[
                        (SpecialPositionalMatch.error, pattern)
                        for pattern in node.patterns
                    ],
                    *patterns,
                ]
            elif len(node.patterns) > len(match_args):
                match_args_status = [*match_args]
                patterns = [
                    *[
                        (SpecialPositionalMatch.error, pattern)
                        for pattern in node.patterns
                    ],
                    *patterns,
                ]
            else:
                patterns = [*zip(match_args, node.patterns), *patterns]

        info = (cls, matched_type, patterns, match_args_status)
        self._class_pattern_info_cache[id(node)] = info
        return info


def index_of(elts: Sequence[T], pred: Callable[[T], bool]) -> int | None:
    for i, elt in enumerate(elts):
        if pred(elt):
            return i
    return None


def get_value_from_kv_pairs(
    kv_pairs: Sequence[KVPair],
    key: Value,
    ctx: CanAssignContext,
    optional_pairs: Container[KVPair],
    removed_pairs: Container[KVPair],
) -> tuple[Value, set[KVPair], set[KVPair]]:
    """Return the :class:`Value` for a specific key."""
    possible_values = []
    covered_keys: set[Value] = set()
    new_optional_pairs: set[KVPair] = set()
    for pair in kv_pairs:
        if pair in removed_pairs:
            continue
        if not pair.is_many:
            if isinstance(pair.key, AnnotatedValue):
                my_key = pair.key.value
            else:
                my_key = pair.key
            if isinstance(my_key, KnownValue):
                is_required = pair.is_required and pair not in optional_pairs
                if my_key == key and is_required:
                    if possible_values:
                        new_optional_pairs.add(pair)
                        new_removed_pairs = set()
                    else:
                        new_removed_pairs = {pair}
                    return (
                        unite_values(*possible_values, pair.value),
                        new_optional_pairs,
                        new_removed_pairs,
                    )
                elif my_key in covered_keys:
                    continue
                elif is_required:
                    covered_keys.add(my_key)
        maybe_error = key.can_overlap(pair.key, ctx, OverlapMode.MATCH)
        if maybe_error is None:
            possible_values.append(pair.value)
            new_optional_pairs.add(pair)
    if not possible_values:
        return UNINITIALIZED_VALUE, set(), set()
    return unite_values(*possible_values), new_optional_pairs, set()


def get_match_args(
    cls: Value, visitor: "pycroscope.name_check_visitor.NameCheckVisitor"
) -> CanAssignError | Sequence[str | SpecialPositionalMatch]:
    if SpecialClassPatternValue.is_assignable(cls, visitor):
        return [SpecialPositionalMatch.self]
    match_args_value = visitor.get_attribute(Composite(cls), "__match_args__")
    if match_args_value is UNINITIALIZED_VALUE:
        return CanAssignError(f"{cls} has no attribute __match_args__")
    match_args_value = replace_known_sequence_value(match_args_value)
    if isinstance(match_args_value, SequenceValue):
        if match_args_value.typ is not tuple:
            return CanAssignError(
                f"__match_args__ must be a literal tuple, not {match_args_value}"
            )
        match_args = []
        for i, (is_many, arg) in enumerate(match_args_value.members):
            if is_many:
                return CanAssignError("Cannot use unpacking in __match_args__")
            if not isinstance(arg, KnownValue) or not isinstance(arg.val, str):
                return CanAssignError(
                    f"__match_args__ element {i} is {arg}, not a string literal"
                )
            match_args.append(arg.val)
        return match_args
    return CanAssignError(
        f"__match_args__ must be a literal tuple, not {match_args_value}"
    )
