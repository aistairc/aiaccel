import asyncio
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.master.create import create_master
from aiaccel.master.local_master import LocalMaster
from aiaccel.master.pylocal_master import PylocalMaster
from aiaccel.scheduler.create import create_scheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.scheduler.pylocal_scheduler import PylocalScheduler

from tests.base_test import BaseTest

import yaml

async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):

        #
        # local test
        #
        with self.create_main():
            config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
            config = Config(config_file)

            # master
            master = create_master(config_file)
            assert master == LocalMaster

            # scheduler
            scheduler = create_scheduler(config_file)
            assert scheduler == LocalScheduler

            storage = Storage(ws=Path(config.workspace.get()))
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
            self.evaluate(data_dir, work_dir, storage)

            self.result_comparison.append(storage.result.get_objectives())

        #
        # pylocal test
        #
        with self.create_main():
            config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
            new_config_file = data_dir.joinpath('config_{}_pylocal.yaml'.format(self.search_algorithm))

            with open(config_file, 'r') as f:
                yml = yaml.load(f, Loader=yaml.SafeLoader)
            yml['resource']['type'] = 'python_local'
            
            with open(new_config_file, 'w') as f:
                f.write(yaml.dump(yml, default_flow_style=False))

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
            self.evaluate(data_dir, work_dir, storage)

            new_config_file.unlink()
            print(storage.result.get_objectives())
            self.result_comparison.append(storage.result.get_objectives())
        
        data_0 = self.result_comparison[0]  # local result
        data_1 = self.result_comparison[1]  # pylocal result
        assert len(data_0) == len(data_1) 
        for i in range(len(data_0)):
            assert data_0[i] == data_1[i]

    def evaluate(self, data_dir, work_dir, storage):
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == self.config.trial_number.get()
        assert ready == 0
        assert running == 0
        final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
        assert final_result.exists()
        '''
        testr = load_yaml(
            work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result))
        datar = load_yaml(
            data_dir.joinpath(
                'work',
                aiaccel.dict_result,
                '{}.{}'.format(aiaccel.file_final_result, self.search_algorithm)
            )
        )
        assert math.isclose(testr['result'], datar['result'], abs_tol=1e-10)
        '''
