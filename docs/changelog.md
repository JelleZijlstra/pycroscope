# Changelog

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
