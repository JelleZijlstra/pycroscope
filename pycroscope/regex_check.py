"""

Helpers for validating regex values.

"""

import re

from .error_code import ErrorCode
from .value import CanAssignError, KnownValue, Value, flatten_values


def check_regex(pattern: str | bytes) -> CanAssignError | None:
    try:
        re.compile(pattern)
    except re.error as e:
        return CanAssignError(
            f"Invalid regex pattern: {e}", error_code=ErrorCode.invalid_regex
        )
    return None


def check_regex_in_value(value: Value) -> CanAssignError | None:
    errors = []
    for subval in flatten_values(value):
        if not isinstance(subval, KnownValue):
            continue
        if not isinstance(subval.val, (str, bytes)):
            continue
        maybe_error = check_regex(subval.val)
        if maybe_error is not None:
            errors.append(maybe_error)
    if errors:
        if len(errors) == 1:
            return errors[0]
        return CanAssignError("Invalid regex", errors, ErrorCode.invalid_regex)
    return None
