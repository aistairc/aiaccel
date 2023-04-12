import pathlib
import time
from functools import wraps

from aiaccel.storage import Storage
from aiaccel.workspace import Workspace

ws = Workspace("test_work")
db_path = (ws.path / 'storage/storage.db')
config_path = pathlib.Path('tests/test_data/config.json')


def db_delete():
    if db_path.exists():
        db_path.unlink()


def init():
    if ws.exists():
        ws.clean()
    # if db_path.exists():
    #     db_path.unlink()


def create():
    ws.create()

def get_storage():
    storage = Storage(ws.storage_file_path)
    return storage

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


def dummy_retry(_MAX_NUM, _DELAY):
    def _retry(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            _MAX_NUM = 0
            _DELAY = 0
            for i in range(_MAX_NUM):
                try:
                    print("dummy retry")
                    return func(*args, **kwargs)
                except BaseException as e:
                    if i == _MAX_NUM - 1:
                        raise e
                    time.sleep(_DELAY)
                    continue
        return _wrapper
    return _retry
