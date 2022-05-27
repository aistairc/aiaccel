from aiaccel.config import Config, load_config
from pathlib import Path
import pytest


class BaseTest(object):

    @pytest.fixture(autouse=True)
    def _setup(self):
        test_data_dir = Path(__file__).resolve().parent.joinpath('test_data')
        test_config_json = test_data_dir.joinpath('config.json')
        # self.config = load_config(test_config_json)
        self.config = Config(test_config_json)
        self.config_json = test_data_dir.joinpath('config.json')
        self.grid_config_json = test_data_dir.joinpath('grid_config.json')
        self.config_yaml = test_data_dir.joinpath('config.yml')
        work_dir = Path(self.config.workspace.get()).resolve()
        self.dict_lock = work_dir.joinpath('lock')
        self.dict_state = work_dir.joinpath('state')
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        self.config_grid = Config(self.grid_config_json)

