# Test random generation of random, tpe and neler-mead optimizer.

import subprocess
from pathlib import Path

from aiaccel.config import load_config
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from tests.base_test import BaseTest


class RandomGenerationTest(BaseTest):
    search_algorithm = None

    def test_run(self, create_tmp_config):
        test_data_dir = Path(__file__).resolve().parent.joinpath('random_generation_test_benchmark', 'test_data')
        python_file = test_data_dir.joinpath('user.py')

        # random execution
        config_file = test_data_dir.joinpath('config_random.yaml')
        config_file = create_tmp_config(config_file)
        config = load_config(config_file)

        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)

        with self.create_main(python_file):
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
        final_result_random = self.get_final_result(storage)
        print('random', final_result_random)

        # tpe execution
        config_file = test_data_dir.joinpath('config_tpe.yaml')
        config_file = create_tmp_config(config_file)
        config = load_config(config_file)

        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)

        with self.create_main(python_file):
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
        final_result_tpe = self.get_final_result(storage)
        print('tpe', final_result_tpe)

        assert final_result_random == final_result_tpe

        # nelder-mead execution
        config_file = test_data_dir.joinpath('config_nelder-mead.yaml')
        config_file = create_tmp_config(config_file)
        config = load_config(config_file)

        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)

        with self.create_main(python_file):
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
        final_result_neldermead = self.get_final_result(storage)
        print('nelder-mead', final_result_neldermead)
        assert final_result_random == final_result_neldermead

    def get_final_result(self, storage):
        data = storage.result.get_all_result()
        return [data[trial_id] for trial_id in data.keys()]
