# Agent Instructions

- If behavior, CLI output/options, or configuration changes, update the relevant docs in `docs/` in the same change.
  Do not update `docs/` for minor changes.
- When asked to make a PR, first rebase your changes on latest `main`, then open a PR. Verify that the PR
  body is formatted correctly, then monitor CI and push fixes for any CI failures or merge conflicts.

## Changelog

- Any change with a user-visible effect must include an entry in `docs/changelog.md`.
- Add changelog entries under `## Unreleased` in `docs/changelog.md`.
- Changelog entries should be single bullets in plain language that explain the user-visible effect (not internal refactors).
- Bugfixes for bugs that were not in any release do not need changelog entries.

## Testing

- Before finishing, run the linting/tests relevant to the files you changed.
- `test_self.py` is often useful for finding regressions. Run it with the `full` extra enabled or it will be skipped.
- Run conformance CI with the same interpreter you want pycroscope itself to use, because `tools/conformance_ci.py`
  invokes `sys.executable -m pycroscope` under the hood.
- Use this form from the repo root: `uv run --python 3.12 python tools/conformance_ci.py --typing-repo ~/py/typing`
  (optionally prefix `UV_CACHE_DIR=/tmp/uv-cache`).
- When fixing regressions found by `test_self.py`, add separate test cases instead of just relying on `test_self.py`.
- When writing test cases, prefer using code samples (`@assert_passes()`) instead of tests that directly
  invoke pycroscope functions. Code samples should represent user-written code that triggers the pycroscope
  feature under test and should not import internal pycroscope functions (except where needed for e.g. `assert_is_value`).
- Prefer `assert_type()` over `assert_is_value()` for type assertions against types where possible; it's OK to keep
  `assert_is_value` for more complicated types that cannot be directly represented in user code.

## Code Conventions

- All type inference should go through the main type inference visitor in `name_check_visitor.py`. Other code should
  generally not perform AST walks. (Exceptions include the pattern matching visitor in `patma.py`, stub visitor in
  `typeshed.py`, and annotation visitor in `annotations.py`.)
- The code should avoid special-casing specific functions or standard library classes outside of impl functions in
  `implementation.py`. Instead of special-casing individual symbols, find more specific solutions.
- `implementation.py` should contain only impl functions used by extended argspecs. Shared non-impl helpers should
  live in the owning module even when they support special-casing logic.
- Code that branches on different subclasses of `Value` should take care to cover all cases. Where possible, replace
  code that dispatches on different kinds of values with a call to a general helper function that already knows how
  to deal with all Values, such as `has_relation`, `is_assignable`, or `is_subtype`. If dispatching on `Value` is
  necessary, use the following procedure:
  - Call `pycroscope.value.gradualize` or `pycroscope.value.replace_fallback` to narrow the Value to a
    fixed set of classes. This will raise `NotAGradualType`
    when it encounters unexpected Value subclasses. Do not catch this exception; instead add specific handling for those
    classes. Non-GradualType Value subclasses should be used only in narrow places.
    `replace_fallback` narrows to a smaller set of Value subclasses (`BasicType` instead of `GradualType`). If you need special handling for some types that are GradualType but not BasicType, do that here.
  - Perform special handing for `MultiValuedValue` (unions) and `IntersectionValue` (intersections), usually calling the
    same logic for each member value and aggregating the results.
  - Now dispatch over all remaining types, which are described by the `SimpleType` union. Use `assert_never()` to ensure
    all types are covered.
- Avoid using `@staticmethod` for local helper functions. Use private module-level functions instead.
- Avoid using function-local imports, except where necessary to avoid an import cycle.
- Never catch `NotAGradualType`. Instead refactor the code so that non-gradual types do not escape narrow parts of the
  checker, or add specific handling for individual types.
- Do not use the names of symbols or parameters (such as `self`) for logic. Instead use type inference or figure out
  whether a parameter (for example) logically represents `self` without relying on the name.
  Exception: opinionated lint-style checks may still enforce spelling conventions for parameter names, such as
  `method_first_arg` requiring `self`/`cls`.
- The behavior in import failure mode (where we cannot load the runtime module) should match that in normal mode
  as much as possible.
- Types and other internal objects should only be represented in one canonical way in pycroscope's internal logic.
  There should not be two equivalent ways to represent the same concept.
