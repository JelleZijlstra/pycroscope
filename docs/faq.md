# Frequently asked questions

## Why is it named pycroscope?

It's like putting your Python code under a microscope and seeing everything that's
happening.

## Why another typechecker?

Pycroscope is a fork of [pyanalyze](https://github.com/quora/pyanalyze), a type
checker developed at Quora. It started as an internal tool somewhere between a linter and a type
checker, but it proved very useful in dealing with [asynq](https://github.com/quora/asynq),
our asynchronous programming framework. This framework uses generators in an unusual way
and pyanalyze made it possible to detect several tricky mistakes in asynq code statically.

For Quora, asynq support remained important, but pyanalyze's architecture has also allowed us
to perform numerous other static checks to make its codebase safer. For example, they use
pycroscope to help keep the codebase safe against SQL injections, to enforce that names of
A/B tests are valid, and to enforce that UI strings are translated correctly.

I (Jelle Zijlstra) was the primary author of pyanalyze, but I left Quora in 2025.
I continued using the project in some personal projects, and I think the ideas are worth
developing further, so I created a fork called pycroscope so I can continue to develop
the type checker under my own account.

## What makes pycroscope different?

The biggest architectural difference between pycroscope and other Python type checkers
is that pycroscope imports the code it checks, while other checkers purely look at the
source code. This allows features that are very
difficult to achieve with a fully static type checker:

- pycroscope requires no special casing to understand the semantics of the `@dataclass`
  decorator, which creates a synthesized `__init__` method. It simply inspects the
  signature of the generated method and uses that for type checking. In general,
  many dynamic constructs unsupported by other type checkers will work immediately with
  pycroscope.
- pycroscope can call back into user code to customize type checking behavior. For example,
  the `CustomCheck` extension provides a way for user code to get very precise control
  over type checking behavior. Possible use cases include allowing only literal values,
  disallowing usage of `Any` for specific APIs, and allowing only values that can be
  pickled at runtime.

But pycroscope is still a static checker, and it has some advantages over a dynamic
(runtime) typechecker:

- All code paths are checked, not just the ones that are hit in a particular run.
- The type system can carry around more information than just the runtime type of a
  value. For example, pycroscope supports `NewType` wrappers around runtime types.
- pycroscope can use type stubs such as those in
  [typeshed](https://github.com/python/typeshed) for type checking.

In addition, pycroscope checks each module mostly independently, keeping the AST for only
one module in memory at once and using runtime function and module objects for computing
signatures and types. This reduces memory usage and makes it easier to deal with circular
dependencies.

However, this approach also has some disadvantages:

- It is difficult to engineer an incremental mode, because that would require reloading
  imported modules. At Quora, we run pycroscope locally on changed files only, which is
  much faster than running it on the entire codebase but does not catch issues where a
  change breaks type checking in an upstream module.
- Scripts that do work on import cannot be usefully checked; you must guard execution
  with `if __name__ == "__main__":`.
- Undefined attributes on instances of user-defined classes cannot be detected with full
  confidence, because the class object does not provide a good way to find out which
  attributes are created in the `__init__` method. Currently, pycroscope works around this
  by deferring detection of undefined attributes until the entire codebase has been checked,
  but this is fragile and not always reliable.
- Initially, the implementation of `@typing.overload` did not provide a way to access the
  overloads at runtime, so there was no obvious way to support overloaded functions at runtime.
  However, this was fixed in Python 3.11.

## When should I use pycroscope?

If you have a complex Python codebase and want to make sure it stays maintainable and
stable, using a type checker is a great option. There are several options for type
checking Python code (listed in
[the typing documentation](https://typing.readthedocs.io/en/latest/)). Unique advantages
of pycroscope include:

- Better support for dynamic constructs and configurability thanks to its semi-static
  architecture (see "What makes pycroscope different?" above).
- Support for specific checks, such as finding missing f prefixes in f-strings,
  finding missing `await` statements, detecting possibly undefined names, and warning
  about conditions on objects of suspicious types.
- Type system extensions such as `CustomCheck`, `ParameterTypeGuard`, and `ExternalType`.
- Strong support for checking code that uses the [asynq](https://github.com/quora/asynq)
  framework.

## What is the history of pycroscope?

[//]: # "First commit is 6d671398f9de24ee8cc1baccadfed2420cba765c"

The first incarnation of pycroscope dates back to July 2015 and merely detected undefined
names. It was soon extended to detect incompatible function calls, find unused
code, and perform some basic type checking.

For context, pytype was started in March 2015 and mypy in 2012. PEP 484 was accepted in 2015.

The initial version was closely tied to
Quora's internal code structure, but in June 2017 it was split off into its own internal
package, now named pyanalyze. By then, it had strong support for `asynq` and supported
customization through implementation functions. After Quora moved to Python 3, pyanalyze
gained support for parsing type annotations and
its typing support moved closer to the standard type system.

The first public release was in May 2020. Since then, work has focused on providing full
support for the Python type system, including `Annotated`, `Callable`, `TypeVar`, and
`TypeGuard`.

After I left Quora in 2025, I created pycroscope in May 2025 as a fork under my
own account.
