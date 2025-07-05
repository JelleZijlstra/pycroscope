Glossary
========

.. glossary::

    constraint
        A :class:`pycroscope.stacked_scopes.Constraint` is a way to
        narrow down the type of a local variable (or other
        :term:`varname`). Constraints are inferred from function
        calls like :func:`isinstance`, conditions like ``is None``,
        and assertions.

    impl
        An impl function is a callback that gets called when the
        type checker encounters a particular function. For example,
        pycroscope contains an impl function for :func:`isinstance`
        that generates a :term:`constraint`.

    phase
        Type checking happens in two phases: *collecting* and
        *checking*. The collecting phase collects all definitions
        and reference; the checking phase checks types. Errors are
        usually emitted only during the checking phase.

    value
        Pycroscope infers and checks types, but the objects used
        to represent types are called :class:`pycroscope.value.Value`.
        Values are pervasive throughout the pycroscope codebase.

    varname
        The object that a :term:`constraint` operates on. This is
        either a string (representing a variable name) or a
        :class:`pycroscope.stacked_scopes.CompositeVariable`,
        representing an attribute or index on a variable.
