# pycroscope

Pycroscope is a semi-static type checker for Python code. Like a static type checker (e.g., mypy or pyright), it
detects type errors in your code so bugs can be found before they reach production. Unlike such tools, however,
it imports the modules it type checks, enabling pycroscope to understand many dynamic constructs that other type
checkers will reject. This property also makes it possible to extend pycroscope with plugins that interact directly
with your code.

Pycroscope is a friendly fork of [pyanalyze](https://github.com/quora/pyanalyze).

## Usage

You can install pycroscope with:

```bash
$ pip install pycroscope
```

Once it is installed, you can run pycroscope on a Python file or package as follows:

```bash
$ python -m pycroscope file.py
$ python -m pycroscope package/
```

But note that this will try to import all Python files it is passed. If you have scripts that perform operations without `if __name__ == "__main__":` blocks, pycroscope may end up executing them.

In order to run successfully, pycroscope needs to be able to import the code it checks. To make this work you may have to manually adjust Python's import path using the `$PYTHONPATH` environment variable.

For quick experimentation, you can also use the `-c` option to directly type check a piece of code:

```
$ python -m pycroscope -c 'import typing; typing.reveal_type(1)'
Runtime type is 'int'

Revealed type is 'Literal[1]' (code: reveal_type)
In <code> at line 1
   1: import typing; typing.reveal_type(1)
                                        ^
```

### Configuration

Pycroscope has a number of command-line options, which you can see by running `python -m pycroscope --help`. Important ones include `-f`, which runs an interactive prompt that lets you examine and fix each error found by pycroscope, and `--enable`/`--disable`, which enable and disable specific error codes.

Configuration through a `pyproject.toml` file is also supported. See
[the documentation](https://pycroscope.readthedocs.io/en/latest/configuration.html) for
details.

### Extending pycroscope

One of the main ways to extend pycroscope is by providing a specification for a particular function. This allows you to run arbitrary code that inspects the arguments to the function and raises errors if something is wrong.

As an example, suppose your codebase contains a function `database.run_query()` that takes as an argument a SQL string, like this:

```python
database.run_query("SELECT answer, question FROM content")
```

You want to detect when a call to `run_query()` contains syntactically invalid SQL or refers to a non-existent table or column. You could set that up with code like this:

```python
from pycroscope.error_code import ErrorCode
from pycroscope.signature import CallContext, Signature, SigParameter
from pycroscope.value import KnownValue, TypedValue, AnyValue, AnySource, Value

from database import run_query, parse_sql


def run_query_impl(ctx: CallContext) -> Value:
    sql = ctx.vars["sql"]
    if not isinstance(sql, KnownValue) or not isinstance(sql.val, str):
        ctx.show_error(
            "Argument to run_query() must be a string literal",
            ErrorCode.incompatible_call,
        )
        return AnyValue(AnySource.error)

    try:
        parsed = parse_sql(sql)
    except ValueError as e:
        ctx.show_error(
            f"Invalid sql passed to run_query(): {e}",
            ErrorCode.incompatible_call,
        )
        return AnyValue(AnySource.error)

    # check that the parsed SQL is valid...

    # pycroscope will use this as the inferred return type for the function
    return TypedValue(list)


# in pyproject.toml, set:
# known_signatures = ["<module>.get_known_argspecs"]
def get_known_argspecs(arg_spec_cache):
    return {
        # This infers the parameter types and names from the function signature
        run_query: arg_spec_cache.get_argspec(
            run_query, impl=run_query_impl
        ),
        # You can also write the signature manually
        run_query: Signature.make(
            [SigParameter("sql", annotation=TypedValue(str))],
            callable=run_query,
            impl=run_query_impl,
        ),
    }
```

### Supported features

Pycroscope generally aims to implement [the Python typing spec](https://typing.readthedocs.io/en/latest/spec/index.html),
but support for some features is incomplete. See [the documentation](https://pycroscope.readthedocs.io/en/latest/)
for details.

### Ignoring errors

Sometimes pycroscope gets things wrong and you need to ignore an error it emits. This can be done as follows:

- Add `# static analysis: ignore` on a line by itself before the line that generates the error.
- Add `# static analysis: ignore` at the end of the line that generates the error.
- Add `# static analysis: ignore` at the top of the file; this will ignore errors in the entire file.

You can add an error code, like `# static analysis: ignore[undefined_name]`, to ignore only a specific error code. This does not work for whole-file ignores. If the `bare_ignore` error code is turned on, pycroscope will emit an error if you don't specify an error code on an ignore comment.

Pycroscope does not currently support the standard `# type: ignore` comment syntax.

### Python version support

Pycroscope supports all versions of Python that have not reached end-of-life. Because it imports the code it checks, you have to run it using the same version of Python you use to run your code.

## Contributing

We welcome your contributions. See [CONTRIBUTING.md](https://github.com/JelleZijlstra/pycroscope/blob/master/CONTRIBUTING.md)
for how to get started.

## Documentation

Documentation is available on [GitHub](https://github.com/JelleZijlstra/pycroscope/tree/master/docs).
