"""

Wrapper for the asynq and qcore modules, which are optional dependencies.

"""

try:
    import asynq as asynq
except ImportError:
    asynq = None
try:
    import qcore as qcore
except ImportError:
    qcore = None
