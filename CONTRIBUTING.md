Welcome! We'd like to make contributing to pycroscope as painless
as possible. Here is a quick guide.

It's useful to have a virtual environment to work in. I use
commands like these:

```
$ cd pycroscope
$ uv sync --locked --extra tests --extra full --group docs
```

## Black

The code is formatted using [_Black_](https://black.readthedocs.io).
You can run the formatter with:

```
$ uv run --locked --with black black pycroscope
```

## ruff

We use [ruff](https://docs.astral.sh/ruff/) as a linter and import sorter:

```
$ uv run --locked --with ruff ruff check pycroscope
```

## Unit tests

The unit tests are run with [pytest](https://docs.pytest.org/):

```
$ uv run --locked --extra tests pytest -v pycroscope
```

Running all of the tests takes a few minutes, so I often use the
`-k` option to select only the tests I am currently working on.
For example:

```
$ uv run --locked --extra tests pytest -v pycroscope -k PEP673
```

We run tests on all supported Python versions on GitHub Actions,
but usually I don't bother when testing locally. If necessary, you
can install all supported versions with a tool like
[pyenv](https://github.com/pyenv/pyenv).

## Third-party CI

`tools/third_party_ci.py` runs pycroscope against configured third-party
repositories. It automatically reads the gitignored local config file
`tools/third_party_ci_local.toml` if it exists, so you can keep local checkout
paths there:

```toml
[local]
taxonomy = "/path/to/taxonomy"
```

Command-line `--local NAME:PATH` values override the config file. Dependency
installation output is hidden by default; pass `-v` or `--verbose` to show it.

## Changelog

For each user-visible change, create a changelog fragment:

```
$ uv run --locked --group release scriv create
```

Replace the generated placeholder with one plain-language bullet and commit the
fragment with the change. Do not edit `docs/changelog.md` directly; Scriv updates
that shared file when a release is prepared. Internal refactors and fixes for bugs
that have never appeared in a release do not need fragments.

See [the release guide](docs/releasing.md) for the maintainer workflow.
