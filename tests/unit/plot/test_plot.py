from aiaccel.storage.storage import Storage
from aiaccel.plot import Plotter
import pathlib
from functools import wraps
from aiaccel.workspace import Workspace
from aiaccel.config import Config

ws = Workspace("test_work")
config_path = pathlib.Path('tests/test_data/config.json')

options = {
    'config': str(config_path),
    'resume': None,
    'clean': False,
    'fs': False,
    'process_name': 'test'
}


def init():
    if ws.exists():
        ws.clean()
    # if ws.path.exists():
    #     ws.path.unlink()


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


@t_base()
def test_plot():
    storage = Storage(ws.path)
    config = Config(config_path)

    goal = "minimize"
    config.goal.set(goal)

    # データ無しの場合
    plotter = Plotter(config, options)
    assert plotter.plot() is None

    # 正常
    trial_id = 0
    objective = 0.01

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    plotter = Plotter(config, options)
    assert plotter.plot() is None

