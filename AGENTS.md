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
- When fixing regressions found by `test_self.py`, add separate test cases instead of just relying on `test_self.py`.
- When writing test cases, prefer using code samples (`@assert_passes()`) instead of tests that directly
  invoke pycroscope functions.

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
  to deal with all Values, such as `has_relation`.
- Avoid using `@staticmethod` for local helper functions. Use private module-level functions instead.
- Avoid using function-local imports, except where necessary to avoid an import cycle.
