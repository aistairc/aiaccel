import asyncio
import subprocess
from pathlib import Path

import yaml

from aiaccel.common import dict_result
from aiaccel.common import file_final_result
from aiaccel.config import Config, is_multi_objective
from aiaccel.storage import Storage
from aiaccel.master import create_master
from aiaccel.master import LocalMaster
from aiaccel.master import PylocalMaster
from aiaccel.scheduler import create_scheduler
from aiaccel.scheduler import LocalScheduler
from aiaccel.scheduler import PylocalScheduler

from tests.base_test import BaseTest


async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config, tmpdir, work_dir):

        #
        # local test
        #
        config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
        config = Config(config_file)
        #is_multi_objective = isinstance(config.goal.get(), list)

        if is_multi_objective(config):
            user_main_file = self.test_data_dir.joinpath('original_main_mo.py')
        else:
            user_main_file = None
            
        with self.create_main(from_file_path=user_main_file):
            config_file = create_tmp_config(config_file)
            config = Config(config_file)

            # master
            master = create_master(config_file)
            assert master == LocalMaster

            # scheduler
            scheduler = create_scheduler(config_file)
            assert scheduler == LocalScheduler

            storage = Storage(ws=Path(config.workspace.get()))
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
            self.evaluate(work_dir, storage, is_multi_objective(config))

            self.result_comparison.append(storage.result.get_objectives())

        #
        # pylocal test
        #
        with self.create_main(user_main_file):
            config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
            new_config_file = tmpdir.joinpath('config_{}_pylocal.yaml'.format(self.search_algorithm))

            with open(config_file, 'r') as f:
                yml = yaml.load(f, Loader=yaml.SafeLoader)
            yml['resource']['type'] = 'python_local'

            with open(new_config_file, 'w') as f:
                f.write(yaml.dump(yml, default_flow_style=False))

            new_config_file = create_tmp_config(new_config_file)
            config = Config(new_config_file)
            assert config.resource_type.get() == 'python_local'

            # master
            master = create_master(new_config_file)
            assert master == PylocalMaster

            # scheduler
            scheduler = create_scheduler(new_config_file)
            assert scheduler == PylocalScheduler

            storage = Storage(ws=Path(config.workspace.get()))

            subprocess.Popen(['aiaccel-start', '--config', str(new_config_file), '--clean']).wait()
            self.evaluate(work_dir, storage, is_multi_objective(config))

            print(storage.result.get_objectives())
            self.result_comparison.append(storage.result.get_objectives())

        data_0 = self.result_comparison[0]  # local result
        data_1 = self.result_comparison[1]  # pylocal result
        assert len(data_0) == len(data_1)
        for i in range(len(data_0)):
            assert data_0[i] == data_1[i]

    def evaluate(self, work_dir, storage, is_multi_objective=False):
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == self.config.trial_number.get()
        assert ready == 0
        assert running == 0

        if not is_multi_objective:
            final_result = work_dir.joinpath(dict_result, file_final_result)
            assert final_result.exists()
