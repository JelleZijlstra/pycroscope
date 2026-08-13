# Changelog fragments

Add one Scriv fragment for each user-visible change instead of editing
`docs/changelog.md` directly:

```console
uv run --locked --group release scriv create
```

Replace the generated placeholder with a single plain-language bullet. Internal
refactors and fixes for bugs that have never appeared in a release do not need a
fragment. Scriv combines and removes the fragments when a release is prepared.
