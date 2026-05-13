import functools
import logging
import time

logger = logging.getLogger("ml")


def timing(func):
    """
    Decorator to time a function's execution. Uses the 'ml' logger to print.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"Function '{func.__module__}.{func.__name__}' took {end - start:.4f} seconds")
        return result

    return wrapper
