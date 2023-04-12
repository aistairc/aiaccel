# Test when initial is not set

import subprocess
from pathlib import Path

from aiaccel.common import dict_result, file_final_result
from aiaccel.config import load_config
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from tests.base_test import BaseTest


class NoInitialTest(BaseTest):
    search_algorithm = None

    def test_run(self, work_dir, create_tmp_config):
        test_data_dir = Path(__file__).resolve().parent.joinpath('no_initial_test_benchmark', 'test_data')
        config_file = test_data_dir.joinpath('config_{}.yaml'.format(self.search_algorithm))
        config_file = create_tmp_config(config_file)
        config = load_config(config_file)
        python_file = test_data_dir.joinpath('user.py')

        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)

        with self.create_main(python_file):
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']
                             ).wait(timeout=config.generic.batch_job_timeout)
        self.evaluate(work_dir, config, storage)

    def evaluate(self, work_dir, config, storage):
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == config.optimize.trial_number
        assert ready == 0
        assert running == 0
        final_result = work_dir.joinpath(dict_result, file_final_result)
        assert final_result.exists()
