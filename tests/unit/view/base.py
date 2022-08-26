import pathlib
from functools import wraps
from aiaccel.workspace import Workspace
from aiaccel.config import Config


# db_path = pathlib.Path("test.db")
# ws = Workspace("test_work")
config_path = pathlib.Path('tests/test_data/config.json')
config = Config(config_path)
ws = Workspace(config.workspace.get())
db_path = pathlib.Path("test.db")


def init():
    if ws.exists():
        ws.clean()


def create():
    ws.create()


def t_base():
    def _test_base(func):
        @wraps(func)
        def _wrapper(*wrgs, **kwargs):
            init()
            create()
            try:
                func(*wrgs, **kwargs)
            finally:
                init()
            return
        return _wrapper
    return _test_base
