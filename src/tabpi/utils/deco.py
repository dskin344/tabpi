from __future__ import annotations

from functools import wraps
import time

from rich import print


def timeit(fn: callable) -> callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{fn.__name__} took {elapsed:.6f} seconds")
        return result

    return wrapper
