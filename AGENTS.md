# Agent Instructions

- If behavior, CLI output/options, or configuration changes, update the relevant docs in `docs/` in the same change.
  Do not update `docs/` for minor changes.

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
