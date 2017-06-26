from decorator import decorator
from functools import wraps

# From librosa: https://github.com/librosa/librosa
try:
    from numba.decorators import jit as optional_jit
except ImportError:
    # Decorator with optional arguments borrowed from
    # http://stackoverflow.com/a/10288927
    def magical_decorator(decorator):
        @wraps(decorator)
        def inner(*args, **kw):
            if len(args) == 1 and not kw and callable(args[0]):
                return decorator()(args[0])
            else:
                return decorator(*args, **kw)
        return inner

    @magical_decorator
    def optional_jit(*_, **__):
        def __wrapper(func, *args, **kwargs):
            return func(*args, **kwargs)
        return decorator(__wrapper)
