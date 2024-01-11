from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def retry(_MAX_NUM: int = 60, _DELAY: float = 1.0) -> Callable[[Any], Any]:
    """Decorator to retry function.

    Args:
        _MAX_NUM (int, optional): Maximum number of retries. Defaults to 60.
        _DELAY (float, optional): Retry interval in seconds. Defaults to 1.0.
    """

    def _retry(func: Callable[[Any], Any]) -> Any:
        @wraps(func)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
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
