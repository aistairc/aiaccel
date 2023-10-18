import asyncio
import subprocess
from pathlib import Path

import yaml

from aiaccel.common import dict_result, file_final_result
from aiaccel.config import is_multi_objective, load_config
from aiaccel.scheduler import (LocalScheduler, PylocalScheduler,
                               create_scheduler)
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from tests.base_test import BaseTest


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config, tmpdir, work_dir):

        #
        # local test
        #
        # is_multi_objective = isinstance(config.goal.get(), list)

        config = self.load_config_for_test(
            self.configs['config_{}.json'.format(self.search_algorithm)]
        )

        if is_multi_objective(config):
            user_main_file = self.test_data_dir.joinpath('original_main_mo.py')
        else:
            user_main_file = None

        with self.create_main(from_file_path=user_main_file):
            # scheduler
            scheduler = create_scheduler(config.resource.type.value)
            assert scheduler == LocalScheduler

            workspace = Workspace(config.generic.workspace)
            storage = Storage(workspace.storage_file_path)
            print(f'\n{config.config_path}\n')
            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--clean']).wait()
            self.evaluate(config, is_multi_objective(config))

            self.result_comparison.append(storage.result.get_objectives())

        #
        # pylocal test
        #
        with self.create_main(user_main_file):
            config = self.load_config_for_test(
                self.configs['config_{}.json'.format(self.search_algorithm)]
            )
            base_dir = Path(config.config_path).parent
            new_config_file_path = base_dir / f'config_{self.search_algorithm}_pylocal.yaml'

            with open(config.config_path, 'r') as f:
                yml = yaml.load(f, Loader=yaml.SafeLoader)
            yml['resource']['type'] = 'python_local'

            with open(new_config_file_path, 'w') as f:
                f.write(yaml.dump(yml, default_flow_style=False))

            config = load_config(create_tmp_config(new_config_file_path))
            assert config.resource.type.value == 'python_local'

            # scheduler
            scheduler = create_scheduler(config.resource.type.value)
            assert scheduler == PylocalScheduler

            workspace = Workspace(config.generic.workspace)
            storage = Storage(workspace.storage_file_path)

            subprocess.Popen(['aiaccel-start', '--config', str(new_config_file_path), '--clean']).wait()
            self.evaluate(config, is_multi_objective(config))

            print(storage.result.get_objectives())
            self.result_comparison.append(storage.result.get_objectives())

        data_0 = self.result_comparison[0]  # local result
        data_1 = self.result_comparison[1]  # pylocal result
        assert len(data_0) == len(data_1)
        for i in range(len(data_0)):
            assert data_0[i] == data_1[i]

    def evaluate(self, config, is_multi_objective=False):
        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)
        work_dir = Path(config.generic.workspace)
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == config.optimize.trial_number
        assert ready == 0
        assert running == 0

        if not is_multi_objective:
            assert workspace.final_result_file.exists()
