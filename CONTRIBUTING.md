Welcome! We'd like to make contributing to pycroscope as painless
as possible. Here is a quick guide.

It's useful to have a virtual environment to work in. I use
commands like these:

```
$ cd pycroscope
$ uv sync --frozen --extra tests --extra full --group docs
```

## Black

The code is formatted using [_Black_](https://black.readthedocs.io).
You can run the formatter with:

```
$ uv run --with black black pycroscope
```

## ruff

We use [ruff](https://docs.astral.sh/ruff/) as a linter and import sorter:

```
$ uv run --with ruff ruff check pycroscope
```

## Unit tests

The unit tests are run with [pytest](https://docs.pytest.org/):

```
$ uv run --extra tests pytest -v pycroscope
```

Running all of the tests takes a few minutes, so I often use the
`-k` option to select only the tests I am currently working on.
For example:

```
$ uv run --extra tests pytest -v pycroscope -k PEP673
```

We run tests on all supported Python versions on GitHub Actions,
but usually I don't bother when testing locally. If necessary, you
can install all supported versions with a tool like
[pyenv](https://github.com/pyenv/pyenv).
