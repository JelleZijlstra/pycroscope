# Changelog

## Unreleased

- Tighten `type`-statement alias semantics: type aliases are no longer accepted as class bases or `isinstance`/`issubclass` classinfo arguments, alias metadata attributes like `__value__`/`__type_params__` are handled consistently, `type` statements are rejected inside function scope, and pycroscope now reports alias redeclarations/invalid unguarded alias cycles plus bound/constraint violations when specializing `TypeAliasType` aliases (including ParamSpec list-form arguments); `isinstance`/`issubclass` now also reject TypedDicts, non-`@runtime_checkable` protocols, and parameterized generics in classinfo arguments.
- Improve constructor-call checking through `type[T]`: pycroscope now validates constructor arguments for both unbounded and bounded `TypeVar` class objects, and rejects extra arguments for classes that use the default no-argument object constructor.
- Improve `typing.NewType` handling in static fallback mode (when modules fail at import): pycroscope now preserves NewType constructor results, validates NewType base-type restrictions (for example `Any`, protocols, `TypedDict`, literals, and generic forms), and reports `NewType` assignment-target name mismatches.
- Preserve generic type arguments more consistently: unsubscripted generic aliases now default their parameters to `Any`, and constructor calls through generic aliases (for example `ListAlias[int]()`) keep static type arguments instead of erasing them to bare runtime container types.
- Improve `TypeVar` bound handling by accepting forward-reference string bounds that refer to names defined later in the file, and by reporting `invalid_annotation` when a bound is parameterized by type variables (for example `bound=list[T]`).
- Improve `Final` handling for dataclasses by allowing `ClassVar[Final[T]]` in dataclass bodies and by not requiring default-less `Final` dataclass fields to be explicitly assigned in `__init__`.
- Unify synthetic class handling by storing synthetic generic/protocol metadata on each synthetic class object, and fix nested local-class intersections so annotation-only members (for example `x: int`) are recognized in attribute checks.
- Improve static fallback analysis for unimportable modules: `TypeVar(...)` results are now preserved in annotations, `type[A | B]` unions are analyzed without runtime `|`, `type[...]` now enforces single-argument arity, and `typing.Type` alias values now report undefined attributes correctly.
- Replace the internal `_SubscriptedValue` with a public `PartialValue` type that records partial expression evaluation details (including operation kind and runtime fallback value), improving extensibility for partially evaluated type expressions.
- Improve protocol checking in import-failure fallback mode by preserving synthetic protocol members and restoring protocol-merging checks (including invalid protocol bases and abstract-class instantiation diagnostics), which fixes conformance coverage for `protocols_merging.py`.
- Infer variance for PEP 695 class type parameters from class member usage and generic base classes, improving assignment checks for covariant and contravariant generics.
- Accept union arguments for constrained `TypeVar` parameters when each union member matches at least one constraint, including calls like `re.compile(pattern)` where `pattern` is `str | bytes`.
- Generalize class-call signature inference to use Python-level metaclass `__call__` methods (not just enum classes), improving call checking for custom metaclasses.
- Allow setting `output_format` in `pyproject.toml` so users can choose concise or detailed error output from config files.
- Improve enum analysis in import-failure fallback mode by preserving enum-member semantics for synthetic classes, validating declared enum `_value_` types, and avoiding false `unsafe_comparison` errors for enum identity checks.
- Improve overload checks in import-failure fallback mode by preserving synthetic base-class relationships for override/final validation, fixing false `override_does_not_override` reports and missed final-method override errors.
- Fix internal errors in static fallback analysis for unimportable modules involving zero-argument `super()` and property setters in synthetic classes.
- Improve qualifier checking by rejecting calls to `Annotated` aliases, enforcing `Final` rules more consistently (including decorator semantics and class-member initialization), and preserving these checks when modules cannot be imported.
- Fix an internal error on bare `ParamSpec` return annotations, and now report bare `ParamSpec` annotations as invalid in direct return/parameter/variable contexts instead of crashing.
- Fix a crash in import-failure fallback mode when checking protocols that inherit generic bases (for example `Iterable[T]`), by resolving inherited protocol members statically instead of treating them as missing.
- Annotate `match` pattern AST nodes in annotate mode, so `annotate_code()` and self-check no longer leave `Match*` nodes without `inferred_value`.
- Fix an internal error when checking equality comparisons between dataclass instances in modules that fail at import time.
- Preserve `@overload` behavior in static fallback mode when a module cannot be imported, including overload-aware inference for synthetic class dunder methods like `__getitem__` and consistency checks against overload implementations.
- Preserve generic base information for synthetic classes in static fallback mode, so subclasses like `class D(dict[str, int])` are assignable to `dict[str, int]`.
- Improve tuple typing behavior by validating invalid `tuple[..., ...]`/multi-unbounded-unpack forms, preserving `tuple[T, ...]` semantics when runtime imports fail, and improving tuple narrowing in sequence-pattern `match` cases.
- Improve NamedTuple analysis in both normal and import-failure fallback modes: class-syntax NamedTuple definitions now enforce field/base-class rules, constructor/type inference is more precise (including generics), and tuple indexing/unpacking/type-compat checks now use NamedTuple field types instead of falling back to `Any`.
- Replace internal `**kwargs` TypedDict special-casing with dictionary-entry modeling, so TypedDict dict/Mapping assignability rules stay consistent while dict-method calls on TypedDict values still type-check correctly.
- Tighten TypedDict conformance by enforcing class-syntax and inheritance checks in importable modules (not only fallback analysis), and by ignoring uninhabitable `NotRequired[Never]` keys in `TypedDict.update()`.
- Respect declared `TypeVar` variance in generic assignability and subtyping checks, so covariant, contravariant, and invariant type parameters are enforced correctly.
- Fix callable subtyping checks for `**kwargs` so callable protocols compare keyword value types correctly even with invariant generic relation checking.
- Remove the unused `requirements.txt` contributor setup file; local development setup now uses `uv sync` and `uv run`.
- Improve inference for function-local `collections.namedtuple(...)` definitions by modeling the generated class as a synthetic local class object with a stable qualified name.
- Tighten TypedDict operation checking: declared TypedDict variables now keep TypedDict semantics after reassignment, dict literals with unknown or non-literal keys are rejected when assigning to TypedDicts, and `TypedDict.clear()`/`TypedDict.popitem()` now report errors for non-closed TypedDicts or when required/readonly keys are possible.
- Improve fallback analysis for unimportable modules so synthetically analyzed `TypedDict` declarations keep `extra_items`/`closed` semantics (including functional `TypedDict(...)` forms), which greatly improves conformance coverage for `typeddicts_extra_items.py`.
- Fix functional `TypedDict(...)` field parsing for `Required[]` and `ReadOnly[]`, and restore assignment compatibility between `Protocol[P]` callables and `Callable[P, ...]` aliases in conformance checks.
- Preserve static typing-helper inference for module-scope assignments in importable modules when import-time runtime values would otherwise erase that typing information.
- Avoid runtime deprecation warnings during analysis by using non-deprecated coroutine detection and suppressing speculative-call deprecation warnings, which speeds up large runs like self-check.
- Add a Python 3.12 CI workflow for typing conformance that runs unit tests for the conformance tooling and then fails if pycroscope's conformance outcomes diverge from the known-failing case list.
- Speed up repeated analysis runs (including the test suite) by reusing typeshed resolvers across checker instances when stub search paths are the same.
- Speed up checker setup by loading regex-related default argspecs only when regex functions are analyzed.
- Speed up large analysis runs by memoizing repeated type-relation checks in assignability/subtyping logic.
- Make implicit `TypeForm` checks side-effect-free so relation memoization stays safe and suppresses redundant work.
- Suppress annotation errors while evaluating runtime forward references, so diagnostics are not misattributed to the current module.
- Improve string forward-reference diagnostics by reporting errors on the original annotation lines rather than line 1, avoiding duplicate reports from the collect/check passes, and supporting multiline triple-quoted string annotations (parsed as implicitly parenthesized expressions).
- Validate that overloaded implementations are compatible with their
  `@overload` signatures (including async/decorator-transformed signatures), and
  report overload/implementation mismatches with the new
  `inconsistent_overload` error code.
- Fix callable protocol subtyping when `__call__` is overloaded, so pycroscope uses the declared overload signatures instead of a generic `*args, **kwargs` fallback.
- Fix `assert_type(..., Callable[..., Any])` equivalence checks.
- Fix `type[None]` annotations so `type(None)` is accepted and `None` values are rejected in type-checked calls.
- Fix handling of historical positional-only parameters (`__x`) in source code:
  keyword calls to these parameters now error correctly, and invalid definitions
  like `def f(x, __y): ...` are now reported under a dedicated
  `invalid_positional_only` error code.
- Allow constructor calls to TypedDict classes that are analyzed syntactically (for example when runtime class objects are unavailable), so `MyTypedDict(...)` is type-checked normally in those cases.
- Report an error for `isinstance(obj, SomeTypedDict)` to match TypedDict runtime semantics.
- Report an error when `TypedDict` is used as a `TypeVar` bound.
- Report `invalid_annotation` for nested duplicate qualifiers (for example `Final[Final[int]]`) and for invalid `TypedDict` item qualifier combinations, including conflicting `Required[]`/`NotRequired[]`, nested `ReadOnly[]`, and unsupported qualifiers like `ClassVar[]`.
- Improve TypedDict checking when runtime class objects are unavailable
  (for example after import-time failures or for function-local class
  definitions) by falling back to syntactic TypedDict analysis, so
  `ReadOnly`/`Required`/`NotRequired` annotations and inheritance conflicts
  are still reported.
- Validate functional `TypedDict(...)` declarations more strictly by reporting errors for non-literal field mappings, non-string field names, and mismatched type names in assignments.
- Preserve functional `TypedDict(...)` type information even when runtime keyword-form construction is unavailable (for example on Python 3.13+), avoiding spurious call errors and follow-on annotation failures.
- Report `invalid_base` for synthetic TypedDict classes that mix `TypedDict` with non-`TypedDict` base classes or bare `Generic` (only `Generic[...]` is allowed).
- Improve handling of class objects that come from stubs or unimportable
  modules by tracking them as singleton class values, which improves
  compatibility checks for TypedDict class objects and type-expression
  evaluation.
- Create synthetic class objects for non-TypedDict classes when runtime class
  objects are unavailable (for example after import-time failures), so class
  self-references continue to resolve and nominal class values are preserved.
- Treat synthetic class objects as class objects in assignability checks, so
  APIs expecting `type` (for example `TypedValue(...)`) accept synthetic
  classes.
- Preserve dynamic `Any`-base behavior for synthetic classes while keeping
  declared methods precise, so checks like `ClassA(Any).method1()` retain
  annotated return types and unknown members still behave as `Any`.
- Fix `Self` inference for classmethods on class objects loaded from stubs
  (including unimportable modules), so calls like `X.from_config()` now infer
  instance results correctly.
- Treat `with` blocks as non-suppressing when `__exit__`/`__aexit__` return types include non-`bool` members like `None | bool`, which improves narrowing after the block.
- Report `unused_variable` and `unused_assignment` for annotated assignments
  like `x: int = value` when the assigned value is never read.
- Narrow tuple types after `len()` checks when bounds imply a more specific
  shape, including exact-length refinements and lower-bound refinements for
  tuples with fixed and variadic parts, which simplifies `reveal_type()` output.
- Extend `len()`-based narrowing to use intersection predicates, which also
  improves narrowing for non-tuple cases such as literal strings and impossible
  `TypedDict` length branches.
- Fix false-positive errors in some `len()`-narrowed branches involving
  `Any & Predicate[...]` intersections (including `assert_type(..., Any)` and
  some sequence indexing operations).
- Fix a crash when accessing attributes on `len()` predicate constraints by
  treating `PredicateValue` attributes like attributes on `object`.
- Fix dunder method handling on intersection types so operations like indexing
  `list[...] & Predicate[...]` values no longer produce spurious errors and
  `Any[error]` inference.
- Keep unexpected keyword argument names in call errors in source order,
  so repeated runs produce stable output.
- Make protocol member lists in type incompatibility messages deterministic by
  using definition order when available and sorted order otherwise.
- Fix an internal error on Python 3.12+ when parsing PEP 695 generics that include `**P` (`ParamSpec`) type parameters.
- Fix crash if accessing a module's `__annotations__` raises an error.
- Implement PEP 747 `TypeForm` support, including implicit and explicit
  `TypeForm` evaluation, assignability checks, and conformance tests.
- Require `typing_extensions>=4.13.0`.
- Drop support for Python 3.9 and add official support for Python 3.14.
- Narrow attribute and subscript expressions in nested scopes based on
  narrowing checks in the outer scope.
- Apply the `class_attribute_transformers` plugin also for values that
  have a `__get__` method.
- Fix internal error in certain cases involving custom `__getattr__` methods
  that raise an error.
- Reduce the set of dependencies (`ast_decompiler` is no longer used;
  `tomli` is only used before Python 3.11; `codemod` is an extra).
- Package a `py.typed` file for pycroscope itself.
- Ignore presence of `__slots__` in protocols defined in stubs.
- Change implementation of implicit int/float and float/complex promotion
  in accordance with https://github.com/python/typing/pull/1748. Now,
  annotations of `float` implicitly mean `float | int`.
- Fix assignability for certain combinations of unions, `Annotated`, and `NewType`.
- Reduce more uninhabited intersections to `Never`
- Keep checking files when module import fails, and report `import_failed`
  on the line that triggered the import-time error (so it can be ignored
  with `# static analysis: ignore[import_failed]`).
- Fix crashes on unsupported syntax in string forward references by
  reporting regular `invalid_annotation` errors instead.
- Fix a crash in callable assignability involving `Concatenate[...,]`
  signatures represented as `AnySig`.
- Fix crash when checking certain `TypeAliasType` specializations that include
  unhashable runtime arguments (e.g. ParamSpec argument lists).
- Fix a crash when checking overloaded `@staticmethod` definitions that involve
  `ParamSpec`-based callable signatures.
- Preserve overload-based return inference for `@staticmethod` and
  `@classmethod` definitions.
- Avoid errors in generic-base extraction when runtime annotations include
  `TypeVarTuple` parameters, including `typing_extensions.TypeVarTuple` on
  Python 3.10.
- Suppress `missing_return` for known abstract stub bodies (protocol methods
  and `@abstractmethod` methods) when the body is just `...` or `pass`
  (including optional docstrings), while still reporting `missing_return` for
  `@abstractmethod` methods with nontrivial bodies.
- Fix a crash when checking classes that inherit from `typing.Any`.
- Narrow variables correctly when calling `TypeGuard` or `TypeIs` functions
  defined as `@staticmethod`, including calls through either instances or
  classes.
- Fix a crash when handling `typing.Annotated` on Python 3.14, where stubs expose it as an annotated assignment (`Annotated: _SpecialForm`).

## Version 0.2.0 (June 26, 2025)

- Fix crash on class definition keyword args when the `no_implicit_any` error
  is enabled.
- Fix incorrect treatment of `ParamSpec` in certain contexts.
- Add basic support for intersection types with `pycroscope.extensions.Intersection`.
- Fix crash on checking the boolability of certain complex types.
- Support subtyping between more kinds of heterogeneous tuples.
- Treat `bool` and enum classes as equivalent to the union of all their
  members.
- Add support for unpacked tuple types using native unpack syntax (e.g.,
  `tuple[int, *tuple[int, ...]]`; the alternative syntax with `Unpack`
  was already supported).
- `assert_type()` now checks for type equivalence, not equality of the
  internal representation of the type.
- Improve parsing of annotation expressions as distinct from type expressions.
  Fixes crash on certain combinations of type qualifiers.
- Improve support for recursive type aliases
- Correctly handle type aliases and other types with fallbacks in more places
- Fix edge case in `TypeIs` type narrowing with tuple types
- Rewrite the implementation of assignability to be more in line with the typing
  specification
- Fix handling of `ClassVar` annotations in stubs
- Fix operations on `ParamSpecArgs` and `ParamSpecKwargs` values
- Fix incorrect assignability relation between `TypedDict` types and
  `dict[Any, Any]`; the spec requires that these be considered incompatible
- Fix bug where certain binary operations were incorrectly inferred as Any
- Fix bug with generic self types on overloaded methods in stubs
- Add support for NewTypes over any type, instead of just simple types
- Add support for a concise output format (`--output-format concise`)
- Fix treatment of aliases created through the `type` statement in union
  assignability and in iteration
- Make `asynq` and `qcore` optional dependencies
- Fix use of aliases created through the `type` statement in boolean conditions

## Version 0.1.0 (May 3, 2025)

First release under the pycroscope name.
See [the pyanalyze docs](https://github.com/quora/pyanalyze/blob/master/docs/changelog.md)
for the previous changelog.

Changes relative to pyanalyze 0.13.1:

- Update PEP 728 support to the latest version, using the `extra_items=`
  class argument instead of an `__extra_items__` key in the dict.
- Add support for Python 3.13
- Drop support for Python 3.8
- Flag invalid regexes in arguments to functions like `re.search`.
