# Changelog

## Unreleased

- Improve PEP 695 scoping checks by rejecting type-parameter bounds/constraints that depend on other type parameters, and preserve module execution order in import-failure fallback mode so top-level uses before assignment now report `undefined_name`.
- Improve variance and generic-call checking in static fallback mode for unimportable modules: legacy `TypeVar(..., infer_variance=True)` now infers class variance from member usage, legacy `Generic[T]` classes preserve their type parameters more reliably, and generic `@dataclass` constructor calls retain explicit type arguments.
- Improve type-inference handling of `|` type expressions in value position: pycroscope now preserves and resolves type-form unions (including metaclass types) without relying solely on runtime `__or__`/`__ror__`, fixing `assert_type(...)` checks like `constructors_call_metaclass.py`.
- Fix constructor-call checking with generic metaclass passthrough methods: when a metaclass `__call__` is just `*args/**kwargs -> T`, pycroscope now falls back to class `__new__`/`__init__` signatures instead of incorrectly accepting missing constructor arguments.
- Fix `TypeVarTuple` type-alias specialization for explicit empty and unpacked forms: aliases like `IntTuple[()]` and `TA9[*tuple[int, ...], str]` now preserve variadic bindings instead of defaulting to `Any` or nesting tuple arguments incorrectly.
- Improve generic base handling in static-fallback analysis: pycroscope now preserves `Generic[...]` type-parameter order when imports fail, validates `Generic[...]`/`Protocol[...]` base arguments and coverage, rejects generic metaclass specialization, and preserves constrained `TypeVar` correlations in binary operations like `AnyStr` concatenation.
- Improve generator annotation checking: pycroscope now reports missing returns based on a generator's declared return component (`Generator[..., ReturnT]`) and accepts protocol-based generator return annotations (for example protocols defining `__next__`/`__anext__`).
- Conformance CI runs now disable the `must_use` error code so typing-conformance comparisons ignore intentionally discarded generator-expression values (such as `assert_type(...)` checks).
- Improve `@dataclass_transform` converter support: synthesized constructor argument types and dataclass field assignment checks now use converter input types, and `default_factory` checks now validate against converter inputs instead of converted field types.
- Fix `TypeVarTuple` concatenation in generic-class specialization so annotations like `Array[Batch, *Shape]` now infer as `Array[Batch, ...]` rather than collapsing to a single tuple type argument.
- Improve callback-protocol compatibility for callable metadata members: pycroscope now treats function `__name__`/`__module__`/`__qualname__` as `str`, avoids false protocol override/assignment errors for these members, and reports unknown attribute writes on protocol-typed callables.
- Fix `TypeVarTuple` unpack handling in generic class annotations: pycroscope now correctly type-checks patterns like `Array[Batch, *tuple[Any, ...], Channels]` and `Array[*tuple[Any, ...]]` in both normal and static-fallback analysis.
- Fix protocol constructor-call handling for abstract class objects: pycroscope no longer reports spurious `Cannot instantiate protocol class ...` errors when calling through values typed as `type[Proto]` that may refer to concrete implementers.
- Improve explicit-protocol checks in import-failure fallback mode: pycroscope now rejects abstract `super()` protocol calls without defaults, enforces protocol-declared member assignment types, and correctly flags instantiation of explicit protocol implementers that still have abstract members.
- Improve explicit `TypeAlias` handling: pycroscope now preserves generic alias metadata (including `ParamSpec`) for `Alias: TypeAlias = ...`, reports invalid alias specializations more consistently, and treats unsubscripted `Callable[P, T]` aliases as `Callable[..., T]`.
- Improve recursive type-alias checks: pycroscope now specializes recursive implicit generic aliases correctly, rejects circular `TypeAlias` unions more consistently, and reports runtime calls to union alias values as invalid.
- Reduce runtime-import side effect leakage in module-scope inference for local class instances, improving `assert_type(...)` stability for dataclass `Final` fields even when later assignments mutate runtime objects.
- Fix override checking for class and dataclass attributes so pycroscope now reports `incompatible_override` when `ClassVar` and instance variables override each other across inheritance.
- Fix generic type-erasure class attribute checks: pycroscope now rejects class-object reads/writes of instance-only annotated attributes (including specialized generic aliases like `Node[int].label`), instead of allowing runtime import side effects to mask these errors.
- Improve `LiteralString` inference for f-strings: pycroscope now preserves `LiteralString` for f-strings whose formatted expressions are all `LiteralString`-compatible, and falls back to `str` when any expression is not `LiteralString`.
- Improve `Value` dispatch handling in name-check helper paths so enum `_ignore_` parsing, dataclass default detection, type-parameter identity extraction, and runtime-literal index checks process unions/intersections consistently.
- Fix `@dataclass_transform` marker recognition to use inferred decorator values, so unrelated decorators named `dataclass_transform` are no longer treated as PEP 681 markers. Also improve support for other dynamic variations on dataclass_transform and dataclass decorators.
- Improve `annotated_types` length-check metadata handling for unions/intersections by normalizing through fallback values and computing `MinLen`/`MaxLen` bounds more consistently.
- Apply intersection-based narrowing for positive `isinstance(...)`, `issubclass(...)`, `TypeIs[...]`, and `is_of_type(...)` checks in more paths, aligning these checks with other intersection constraint handling.
- Improve recursive protocol matching with `@classmethod` members: pycroscope now preserves classmethod parameter constraints and `Self` specialization in synthetic static-fallback analysis, fixing false protocol incompatibilities and downstream `Any` inference.
- Improve generic-base-class analysis in static fallback mode for unimportable modules: pycroscope now preserves generic base mappings and type-parameter ordering more accurately, reports duplicate/conflicting generic base type-variable declarations, and rejects `Generic`/`Generic[...]` used as type annotations outside base-class lists.
- Improve PEP 695 generic declaration checks: pycroscope now rejects `Generic[...]`/`Protocol[...]` specialization in `class C[T]` bases, handles forward references in type-parameter bounds/constraints without spurious `undefined_name` errors, and reports invalid type-parameter constraint tuples with fewer than two types.
- Preserve literal type arguments when inferring generic class-syntax `NamedTuple` constructor return types (for example, `Box(1)` now infers `Box[Literal[1]]`).
- Improve `TypeVarTuple` handling for generic classes in static-fallback analysis: pycroscope now preserves variadic argument lengths and `NewType` members during inference, and correctly reports incompatible tuple lengths across repeated uses.
- Fix static-fallback generic-class specialization with `TypeVarTuple` bases so annotations like `Array[Height, Width]` no longer spuriously report wrong type-argument arity after an import-time runtime failure in the same module.
- Generalize repeated `TypeVarTuple` inference so same-length element mismatches merge to per-position unions across call contexts (including `*args: tuple[*Ts]` and callable packs), while still rejecting mismatched tuple lengths.

## Version 0.3.0 (March 1, 2026)

This release includes a large number of changes aimed at improving compliance with the
[Python typing spec](https://typing.python.org/en/latest/spec/index.html). While many features are still
missing or incomplete, there is now some degree of support of all major type system features.

- Improve `TypeVarTuple` solving for repeated tuple-parameter uses (for example `def f(*args: tuple[*Ts])`): incompatible tuple shapes across arguments are now rejected, and same-length element mismatches are inferred as per-position unions.
- Remove additional name-based receiver checks in protocol and call-signature handling, so nonstandard receiver names no longer affect protocol-member binding or TypedDict-backed dict-method checks.
- Reduce name-based method heuristics by checking receiver parameters structurally in more places, which improves consistency for methods that don't use literal `self`/`cls` names.
- Improve generic protocol checking by honoring `Protocol[...]` type-parameter order and rejecting protocol matches with unsatisfiable cross-member type-variable constraints.
- Improve method-receiver handling by using inferred receiver semantics instead of hard-coded `self`/`cls` names, so checks for `Final` instance attributes, protocol-member synthesis, enum `_value_` assignments, and receiver attribute writes now behave correctly with nonstandard receiver names.
- Improve protocol/class-object conformance for class-object assignments: pycroscope now enforces method/property/class-variable compatibility more accurately (including static-fallback analysis for unimportable modules).
- Fix `@dataclass_transform` metadata tracking to live on inferred values instead of scope-name side tables, which avoids internal errors in cases like `ParamSpec` declarations and `global`/`nonlocal` references.
- Improve `ParamSpec` component checking and forwarding: pycroscope now rejects more invalid `P.args`/`P.kwargs` annotation and call forms, accepts valid `Concatenate` forwarding patterns like `foo(1, *args, **kwargs)`, and no longer reports `generics_paramspec_components` as a known conformance failure.
- Improve ParamSpec handling in static-fallback analysis for unimportable modules: generic classes like `Y(Generic[U, P])` now preserve ParamSpec type parameters from `Generic[...]` bases, list-form specializations like `Y[int, [int]]` are interpreted correctly, and callable attributes like `y.f` specialize to concrete signatures.
- Improve descriptor handling for dataclass-like classes (`@dataclass` and `@dataclass_transform`): constructor parameters now use descriptor `__set__` value types for data descriptors, and class/instance attribute reads now follow descriptor `__get__` return types.
- Tighten ParamSpec specialization checks for generic classes: pycroscope now reports `invalid_annotation` for invalid mixed-generic forms like `C[int, int]` when the second parameter is a `ParamSpec`, while still accepting valid forms like `C[int, [int]]`, `C[int, P]`, `C[int, Concatenate[str, P]]`, and `C[int, ...]`.
- Improve `Value` dispatch consistency in suggested-type and class-key inference helpers by normalizing through fallback values and handling unions/intersections more consistently.
- Improve `Value` dispatch robustness in name checking so enum assignment analysis, protocol-base detection, and type-parameter extraction handle unions/intersections consistently and avoid internal errors on non-gradual values.
- Improve intersection-type attribute checks so pycroscope now enforces slot restrictions, frozen-dataclass immutability, and NamedTuple field immutability for `Intersection[...]` instance values.
- Improve class-object resolution for annotated and generic values in core name checking, reducing false negatives in protocol-instantiation, enum-base, and attribute-target checks.
- Improve protocol/class-object compatibility checks by handling more class-value forms consistently (including `SubclassValue` generic wrappers and `type`-typed values).
- Improve `@contextmanager`/`@asynccontextmanager` inference: decorated callables now keep context-manager return types, and `with ... as x` now infers the yielded value type more precisely.
- Improve constraint narrowing consistency by routing `isinstance(...)` and `is ...` checks through relation-based subtype logic, reducing edge-case mismatches across complex value kinds.
- Apply NamedTuple runtime-call suppression consistently by removing the special-case exemption for classes in `pycroscope.*` modules.
- Update internal `@contextmanager` return annotations to use `Generator[...]`, improving compatibility with newer typing/typeshed behavior.
- Improve protocol compatibility in static fallback mode for unimportable modules: pycroscope now enforces ClassVar-vs-instance member distinctions, writable data-member/property rules (including setter requirements), and supports plain writable attributes for read-only protocol properties.
- Fix tuple-literal equivalence checks so `Literal[("x",)]` is treated as equivalent to `tuple[Literal["x"]]` in subtype/equivalence relations.
- Improve dataclass constructor/pattern semantics in static fallback mode: pycroscope now honors class-level `init`/`match_args` settings for `@dataclass` and `@dataclass_transform`, and recognizes `factory=` field specifier defaults when checking generated constructors and `default_factory` return types.
- Improve dataclass conformance in static fallback mode for unimportable modules: pycroscope now validates default-before-non-default field ordering, checks `default_factory` return types against field annotations, preserves callable dataclass fields as values (not bound methods), exposes dataclass metadata (`__dataclass_fields__`), and resolves inherited dataclass constructors more accurately.
- Tighten `TypeVarTuple` validation: pycroscope now reports `invalid_annotation` for unpacking mistakes like `tuple[Ts]`, `*args: Ts`, `Generic[Ts]`, and `Generic[*Ts1, *Ts2]`, and no longer emits spurious `Unrecognized annotation typing.Unpack[...]` errors for annotations like `Array[*Shape]`.
- Improve `TypeVar` default validation: pycroscope now reports `invalid_annotation` when defaults conflict with bounds or constraints, enforces `Generic[...]` default-order rules (including import-failure fallback mode), and treats unspecialized synthetic class objects as compatible with specialized `type[...]` forms.
- Improve constructor checking when modules fail at import time: pycroscope now enforces explicit `__init__` self-annotation compatibility for specialized generic class calls (for example rejecting `Class4[str]()` when `__init__` requires `Class4[int]`), and avoids false positives from synthetic class-subscripting fallback on generic instances.
- Improve `TypeAliasType(...)` handling: variadic alias specialization now supports `TypeVarTuple`, runtime `type_params` scope/literal-tuple and circular-definition checks are enforced, and recursive alias evaluation no longer triggers internal recursion errors.
- Fix variance checks in static fallback mode for unimportable modules by recovering generic type parameters from base annotations, which removes false `invalid_annotation` errors and restores expected nested-alias variance errors (for example `generics_variance.py`).
- Fix a crash in type relation checks when comparing generic arguments that mix ParamSpec call-signature values with non-ParamSpec values; pycroscope now reports a regular type mismatch instead of `internal_error`.
- Accept ParamSpec list specialization syntax in generic classes (for example `C[[int]]` for `class C[**P]`), matching the typing spec’s allowed form.
- Improve static fallback analysis for unimportable modules by pre-registering synthetic methods and persisting `self`/`cls` attribute assignments, reducing false `undefined_attribute` errors in class methods.
- Tighten typing-construct arity validation in annotations (including `ClassVar`/`Final`/`Required` qualifiers and runtime `Callable[...]` parsing), so malformed argument lists are reported more consistently.
- Improve `@dataclass_transform` field specifier default handling: pycroscope now infers implicit `init` values from field-specifier signatures (including overload defaults like `Literal[False]`), fixing constructor checks for cases like `dataclasses_transform_field.py`.
- Improve dataclass `slots` checking (including `@dataclass_transform` classes): pycroscope now surfaces `__slots__` for slotted dataclasses, rejects `slots=True` classes that also define `__slots__`, and reports invalid assignments to attributes not declared in slots (including static fallback mode for unimportable modules).
- Fix callable protocol compatibility for `__call__` signatures in import-failure fallback mode (including `*args: Any, **kwargs: Any` ellipsis-style tails and `Concatenate[..., ...]` interop), improving conformance for `callables_annotation.py`.
- Improve generic alias constructor checking in static fallback mode: calls like `Node[int](...)` now preserve explicit type arguments for inference and enforce constructor argument types.
- Improve protocol conformance by rejecting protocol instantiation and fixing generic/variance protocol subtyping checks, including static fallback handling for protocol generic bases when runtime import metadata is incomplete.
- Improve static fallback analysis for unimportable modules by resolving synthetic class instance methods from synthetic bases in expression contexts, reducing false `undefined_attribute`/`inference_failure` cascades for cases like `Self`-typed methods.
- Improve static fallback analysis for unimportable modules by making synthetic `@classmethod` attributes callable and preserving `Self` return specialization (for example inferring `Circle` from `Circle.from_config(...)`).
- Improve `ClassVar` conformance checks: pycroscope now rejects `ClassVar` outside class-body attribute declarations (including type aliases), reports errors for assignments to class variables through instances, and adds a new `classvar_type_parameters` error code (disabled by default) for rejecting `ClassVar` type parameters (`TypeVar`/`ParamSpec`); conformance CI now enables this stricter check.
- Improve static fallback analysis for unimportable modules by resolving synthetic class instance methods from synthetic bases in expression contexts, reducing false `undefined_attribute`/`inference_failure` cascades for cases like `Self`-typed methods.
- Fix dataclass hashability inference (including `@dataclass_transform` classes): mutable `eq=True` classes are now treated as unhashable unless `unsafe_hash=True` or an explicit `__hash__` is provided, in both normal analysis and import-failure fallback mode.
- Improve dataclass `InitVar` handling: `__post_init__` signatures are now validated against `InitVar` fields (including inherited fields), and `InitVar` members are now correctly rejected as instance attributes, fixing conformance for `dataclasses_postinit.py`.
- Align `@dataclass_transform` metaclass inheritance with the typing spec: classes that directly specify a transform-decorated metaclass are now treated as neither frozen nor non-frozen, which fixes conformance behavior for `dataclasses_transform_meta.py`.
- Replace internal starred-expression handling with `PartialValue` `UNPACK` operations, removing `_StarredValue` and making partial unpack evaluation consistent across annotations and runtime expression analysis.
- Enforce type-variable variance compatibility in class inheritance through generic aliases, so invalid bases like `Base[T_co]` and aliased equivalents now report `invalid_annotation`.
- Tighten `Callable[...]` annotation validation to match the typing spec: pycroscope now correctly rejects invalid forms like `Callable[int]`, `Callable[int, int]`, `Callable[int, int, int]`, and `Callable[[...], T]`.
- Improve dataclass and `@dataclass_transform` constructor/comparison checking: synthetic dataclass constructors no longer inherit non-dataclass base `__init__` parameters, generic transform bases now propagate correctly (including `Base[T]` forms), and `<`/`<=`/`>`/`>=` comparisons now enforce dataclass ordering rules.
- Add initial PEP 681 `@dataclass_transform` support: classes transformed by marked decorators, base classes, or metaclasses now get dataclass-like constructor/frozen semantics in both normal analysis and static fallback mode when imports fail.
- Honor standard `# type: ignore` comments (including top-of-file file-level ignores), so pycroscope now suppresses errors in the same places users expect from typing-spec directives.
- Reduce false-positive `incompatible_override` errors for protocol-style method overrides by comparing callable signatures more consistently in override checks.
- Improve `ParamSpec` handling by rejecting invalid annotation locations more consistently (including bare `ParamSpec` type aliases, `list[P]`, and `Callable[..., P]` return positions), and enforce assignment-target name matching for `TypeVar`, `TypeVarTuple`, `ParamSpec`, `NewType`, and functional `NamedTuple`/`TypedDict` declarations.
- Improve `TypeVarTuple` callable checking by preserving unpacked tuple parameters in callable argument lists, improving inference for variadic `*args` patterns, and keeping constructor-call checking precise for synthetic classes in static fallback analysis.
- Improve protocol runtime-check checks: `issubclass()` now rejects `@runtime_checkable` data protocols, and `isinstance()`/`issubclass()` now report `incompatible_argument` for unsafe-overlap runtime-checkable protocol checks that could succeed at runtime despite incompatible member types.
- Improve synthetic constructor-to-callable inference: pycroscope now synthesizes more accurate callable signatures for overloaded `__init__`/`__new__`, honors custom metaclass `__call__`, preserves local `namedtuple` constructor behavior, and handles dataclass constructor fallback (including class vars and import-failure subclasses) more accurately.
- Improve static fallback analysis for dataclasses with keyword-only fields: `KW_ONLY` pseudo-fields no longer raise `invalid_annotation`, and kw-only constructor arguments are now checked correctly even when modules fail at import time.
- Allow `@override` methods in subclasses of `Any`-derived base classes (for example `class Parent(Any): ...`) instead of incorrectly reporting `override_does_not_override`.
- Enforce PEP 695 generic-syntax compatibility rules for classes and generic functions/methods: pycroscope now reports `invalid_annotation` when old-style `TypeVar`/`ParamSpec`/`TypeVarTuple` declarations are mixed into new `class C[T]` or `def f[T](...)` annotation contexts.
- Add protocol variance validation for both legacy `Protocol[T]` and PEP 695 generic syntax: pycroscope now reports `invalid_annotation` when declared protocol type-variable variance does not match inferred usage (including unused protocol type variables, which default to covariant).
- Add a new `invalid_literal` error code (disabled by default) to flag `Literal[...]` arguments that are outside the typing-spec allowed set, and enable this check in conformance CI runs.
- Tighten `type`-statement alias semantics: type aliases are no longer accepted as class bases or `isinstance`/`issubclass` classinfo arguments, alias metadata attributes like `__value__`/`__type_params__` are handled consistently, `type` statements are rejected inside function scope, and pycroscope now reports alias redeclarations/invalid unguarded alias cycles plus bound/constraint violations when specializing `TypeAliasType` aliases (including ParamSpec list-form arguments); `isinstance`/`issubclass` now also reject TypedDicts, non-`@runtime_checkable` protocols, and parameterized generics in classinfo arguments.
- Improve constructor-call checking through `type[T]`: pycroscope now validates constructor arguments for both unbounded and bounded `TypeVar` class objects, and rejects extra arguments for classes that use the default no-argument object constructor.
- Improve `typing.NewType` handling in static fallback mode (when modules fail at import): pycroscope now preserves NewType constructor results, validates NewType base-type restrictions (for example `Any`, protocols, `TypedDict`, literals, and generic forms), and reports `NewType` assignment-target name mismatches.
- Add frozen-dataclass enforcement in both normal and import-failure fallback analysis: assigning to frozen dataclass instance attributes is now rejected, and mixing frozen/non-frozen dataclass inheritance now reports errors.
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
- Tighten attribute-store checks: assigning to or deleting `NamedTuple` fields through attribute syntax now reports errors, and assignments to existing annotated attributes now report `incompatible_assignment` when types do not match.
- Replace internal `**kwargs` TypedDict special-casing with dictionary-entry modeling, so TypedDict dict/Mapping assignability rules stay consistent while dict-method calls on TypedDict values still type-check correctly.
- Tighten TypedDict conformance by enforcing class-syntax and inheritance checks in importable modules (not only fallback analysis), and by ignoring uninhabitable `NotRequired[Never]` keys in `TypedDict.update()`.
- Respect declared `TypeVar` variance in generic assignability and subtyping checks, so covariant, contravariant, and invariant type parameters are enforced correctly.
- Fix callable subtyping checks for `**kwargs` so callable protocols compare keyword value types correctly even with invariant generic relation checking.
- Tighten `**kwargs: Unpack[...]` callable handling: pycroscope now requires a concrete `TypedDict` inside `Unpack` for `**kwargs`, rejects overlapping named parameters, and disallows assigning plain keyword-only callables to protocols that require unpacked `TypedDict` kwargs.
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
