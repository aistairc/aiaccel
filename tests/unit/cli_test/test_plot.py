from pathlib import Path
from unittest.mock import patch

from aiaccel.cli import Plotter
from aiaccel.config import load_config
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace


def test_plot(clean_work_dir, work_dir, create_tmp_config):

    clean_work_dir()
    workspace = Workspace(str(work_dir))
    if workspace.path.exists():
        workspace.clean()
    workspace.create()

    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    config = load_config(config_path)
    config.optimize.goal = ["minimize"]

    # データ無しの場合
    plotter = Plotter(config)
    assert plotter.plot() is None

    # 正常
    trial_id = 0
    objective = [0.01]

    storage = Storage(workspace.storage_file_path)
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
    with patch.object(plotter.storage.result, 'get_objectives', return_value=[[1], [2], [3]]):
        with patch.object(plotter.storage.result, 'get_bests', return_value=[1, 2, 3, 4]):
            assert plotter.plot() is None

    # self.cplt.set_colors
    # self.cplt.caption
    # self.cplt.line_plot
    with patch.object(plotter.storage.result, 'get_objectives', return_value=[[1], [2], [3]]):
        with patch.object(plotter.storage.result, 'get_bests', return_value=[1, 2, 3]):
            assert plotter.plot() is None
