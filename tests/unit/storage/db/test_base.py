import pathlib
from functools import wraps

db_path = pathlib.Path("test.db")


def init():
    if db_path.exists():
        db_path.unlink()


def t_base():
    def _test_base(func):
        @wraps(func)
        def _wrapper(*wrgs, **kwargs):
            init()
            func(*wrgs, **kwargs)
            init()
            return

        return _wrapper

    return _test_base
