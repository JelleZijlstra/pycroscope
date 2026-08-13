# Releasing pycroscope

Release preparation happens in a pull request so that the generated changelog,
version, and lockfile can be reviewed together.

## Prepare the release

1. Start from the latest `main` and choose the new version.
2. Confirm that every user-visible change since the previous release has a
   fragment in `changelog.d/`.
3. Update `project.version` in `pyproject.toml`.
4. Update the lockfile explicitly:

   ```console
   uv lock
   ```

   The project itself is represented in `uv.lock`, so every version bump must
   include this step. Review the diff and make sure unrelated dependencies did
   not change.

5. Collect the fragments into a dated release section:

   ```console
   uv run --locked --group release scriv collect
   ```

   Scriv reads the version from `pyproject.toml`, adds a section to
   `docs/changelog.md`, and removes the collected fragment files. Review the
   generated text and optionally add a short release summary below the heading.

6. Run the relevant tests and build the distributions:

   ```console
   uv run --locked --extra tests --extra asynq --extra codemod pytest pycroscope
   uv build
   ```

7. Commit the version bump, `uv.lock`, generated changelog, and removed
   fragments in the release pull request. Merge it after CI passes.

## Publish the release

GitHub Releases are the authoritative release mechanism. After the release pull
request is merged, open the repository's **Releases** page and draft a new
release:

1. Create a new tag whose name exactly matches the version in `pyproject.toml`
   (for example, `0.5.0`) and target it at the release pull request's merge
   commit on `main`. Do not create or push the tag separately.
2. Use the version as the release title.
3. Copy the corresponding section from `docs/changelog.md` into the release
   notes. Scriv can print only that section for convenient copying:

   ```console
   uv run --locked --group release scriv print --version 0.5.0
   ```

4. Save and review the draft before publishing it.

Publishing the GitHub release creates the tag and triggers
`.github/workflows/publish.yml`, which builds the distributions and publishes
them to PyPI. Confirm that the workflow completes successfully and that the new
version is available on PyPI.
