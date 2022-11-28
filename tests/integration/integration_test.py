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

from tests.base_test import BaseTest


async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):
        with self.create_main():
            config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
            config = Config(config_file)
            storage = Storage(ws=Path(config.workspace.get()))
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
            self.evaluate(data_dir, work_dir, storage)

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
