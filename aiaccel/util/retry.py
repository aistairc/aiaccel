import time
from functools import wraps


def retry(_MAX_NUM=60, _DELAY=1.0):
    def _retry(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            for i in range(_MAX_NUM):
                try:
                    return func(*args, **kwargs)
                except BaseException as e:
                    if i == _MAX_NUM - 1:
                        raise e
                    time.sleep(_DELAY)
                    continue
        return _wrapper
    return _retry
