import time
from functools import wraps

def clock(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f" {func.__name__}의 실행 시간: {end - start:.2f}초")
        return result
    return wrapper
