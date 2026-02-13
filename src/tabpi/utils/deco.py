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


def avgtime(n: int = 10):
    def decorator(fn: callable) -> callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            total_time = 0.0
            for _ in range(n):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                end = time.perf_counter()
                print(f"Run took {end - start:.6f} seconds")
                total_time += end - start
            avg_time = total_time / n
            print(f"{fn.__name__} took an average of {avg_time:.6f} seconds over {n} runs")
            return result

        return wrapper

    return decorator
