from contextlib import contextmanager
from pathlib import Path

import pytest
from aiaccel.config import Config
from aiaccel.util.filesystem import create_yaml
from aiaccel.workspace import Workspace
import shutil

d0 = {
    "end_time": "11/03/2020 16:07:45",
    "trial_id": "0",
    "parameters":
        [
            {
                "parameter_name": "x1",
                "type": "FLOAT",
                "value": 0.9932890709584586
            },
            {
                "parameter_name": "x10",
                "type": "FLOAT",
                "value": 3.599465287952899
            },
            {
                "parameter_name": "x2",
                "type": "FLOAT",
                "value": -3.791100401941936
            },
            {
                "parameter_name": "x3",
                "type": "FLOAT",
                "value": -1.6730481463987088
            },
            {
                "parameter_name": "x4",
                "type": "FLOAT",
                "value": 2.2148440758326835
            },
            {
                "parameter_name": "x5",
                "type": "FLOAT",
                "value": 2.111917696952796
            },
            {
                "parameter_name": "x6",
                "type": "FLOAT",
                "value": 4.364405867994597
            },
            {
                "parameter_name": "x7",
                "type": "FLOAT",
                "value": -0.7789300003858477
            },
            {
                "parameter_name": "x8",
                "type": "FLOAT",
                "value": 3.30035693274327
            },
            {
                "parameter_name": "x9",
                "type": "FLOAT",
                "value": 1.7030556641407104
            }
        ],
    "result": 73.92756153914445,
    "start_time": "11/03/2020 16:07:38"
}


d1 = {
    "end_time": "11/03/2020 16:07:45",
    "trial_id": "1",
    "parameters":
        [
            {
                "parameter_name": "x1",
                "type": "FLOAT",
                "value": 0.8521885935278827
            },
            {
                "parameter_name": "x10",
                "type": "FLOAT",
                "value": -0.6723293209494665
            },
            {
                "parameter_name": "x2",
                "type": "FLOAT",
                "value": 2.6228008245794197
            },
            {
                "parameter_name": "x3",
                "type": "FLOAT",
                "value": -4.978939466488893
            },
            {
                "parameter_name": "x4",
                "type": "FLOAT",
                "value": -0.546128059451986
            },
            {
                "parameter_name": "x5",
                "type": "FLOAT",
                "value": 2.2154003234078257
            },
            {
                "parameter_name": "x6",
                "type": "FLOAT",
                "value": -2.7123777872954733
            },
            {
                "parameter_name": "x7",
                "type": "FLOAT",
                "value": 4.452706955539224
            },
            {
                "parameter_name": "x8",
                "type": "FLOAT",
                "value": 4.014274576114836
            },
            {
                "parameter_name": "x9",
                "type": "FLOAT",
                "value": -4.694100169664464
            }
        ],
    "result": 103.38599820960606,
    "start_time": "11/03/2020 16:07:38"
}


d2 = {
    "end_time": "11/03/2020 16:07:46",
    "trial_id": 2,
    "parameters":
        [
            {
                "parameter_name": "x1",
                "type": "FLOAT",
                "value": 0.2209278197011611
            },
            {
                "parameter_name": "x10",
                "type": "FLOAT",
                "value": 3.4743373693723267
            },
            {
                "parameter_name": "x2",
                "type": "FLOAT",
                "value": 2.6377461897661405
            },
            {
                "parameter_name": "x3",
                "type": "FLOAT",
                "value": -2.449309742605783
            },
            {
                "parameter_name": "x4",
                "type": "FLOAT",
                "value": -0.04564912908059071
            },
            {
                "parameter_name": "x5",
                "type": "FLOAT",
                "value": -0.505089352112619
            },
            {
                "parameter_name": "x6",
                "type": "FLOAT",
                "value": 1.515929727227629
            },
            {
                "parameter_name": "x7",
                "type": "FLOAT",
                "value": 2.8872335113551317
            },
            {
                "parameter_name": "x8",
                "type": "FLOAT",
                "value": -4.0614041322576515
            },
            {
                "parameter_name": "x9",
                "type": "FLOAT",
                "value": -4.716525234779937
            }
        ],
    "result": 74.70862563400767,
    "start_time": "11/03/2020 16:07:40"
}


class BaseTest(object):

    @pytest.fixture(autouse=True)
    def _setup(self):
        test_data_dir = Path(__file__).resolve().parent.joinpath('test_data')
        self.test_data_dir = test_data_dir
        test_config_json = test_data_dir.joinpath('config.json')
        self.config_random_path = test_data_dir.joinpath('config_random.json')
        self.config_sobol_path = test_data_dir.joinpath('config_sobol.json')
        self.config = Config(test_config_json)
        self.config_random = Config(self.config_random_path)
        self.config_sobol = Config(self.config_sobol_path)
        self.config_json = test_data_dir.joinpath('config.json')
        self.grid_config_json = test_data_dir.joinpath('grid_config.json')
        self.config_yaml = test_data_dir.joinpath('config.yml')
        work_dir = Path(self.config.workspace.get()).resolve()
        self.work_dir = work_dir

        self.dict_lock = work_dir.joinpath('lock')

        self.config_grid = Config(self.grid_config_json)

        self.workspace = Workspace(str(work_dir))
        if self.workspace.path.exists():
            self.workspace.clean()
        self.workspace.create()

        self.test_result_data = []
        self.test_result_data.append(d0)
        self.test_result_data.append(d1)
        self.test_result_data.append(d2)

        for d in self.test_result_data:
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            create_yaml(path, d)

        self.result_comparison = []

    @contextmanager
    def create_main(self, from_file_path=None):
        if from_file_path is None:
            from_file_path = self.test_data_dir.joinpath('original_main.py')
        to_file_path = Path('/tmp/original_main.py')
        shutil.copy(from_file_path, to_file_path)
        yield
        to_file_path.unlink()
        results_file = Path('/tmp/results')
        shutil.rmtree(results_file)

    def get_workspace_path(self):
        return self.work_dir
