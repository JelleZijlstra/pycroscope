# Changelog

## Unreleased

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
