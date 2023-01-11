import pathlib

from aiaccel.cli.plot import Plotter
from aiaccel.config import load_config
from aiaccel.storage.storage import Storage
from aiaccel.workspace import Workspace
from unittest.mock import patch
from pathlib import Path


def test_plot(clean_work_dir, work_dir, create_tmp_config):

    clean_work_dir()
    workspace = Workspace(str(work_dir))
    if workspace.path.exists():
        workspace.clean()
    workspace.create()

<<<<<<< HEAD
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
    config = load_config(config_path)
=======
    config_path = pathlib.Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    config = Config(config_path)
    storage = Storage(ws=Path(config.workspace.get()))
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129

    config.optimize.goal = "minimize"

    # データ無しの場合
    plotter = Plotter(config)
    assert plotter.plot() is None

    # 正常
    trial_id = 0
    objective = 0.01

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    plotter = Plotter(config)
    assert plotter.plot() is None

    # len(objectives) == 0
    with patch.object(plotter.storage.result, 'get_objectives', return_value=[]):
        assert plotter.plot() is None

    # len(objectives) != len(bests)
    with patch.object(plotter.storage.result, 'get_objectives', return_value=[1, 2, 3]):
        with patch.object(plotter.storage.result, 'get_bests', return_value=[1, 2, 3, 4]):
            assert plotter.plot() is None

    # self.cplt.set_colors
    # self.cplt.caption
    # self.cplt.line_plot
    with patch.object(plotter.storage.result, 'get_objectives', return_value=[1, 2, 3]):
        with patch.object(plotter.storage.result, 'get_bests', return_value=[1, 2, 3]):
            assert plotter.plot() is None
