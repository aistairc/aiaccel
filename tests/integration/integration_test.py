from aiaccel.master.local import LocalMaster
from aiaccel.argument import Arguments
from tests.base_test import BaseTest
import aiaccel
import asyncio
import time
from unittest.mock import patch
import sys
import os
from aiaccel.storage.storage import Storage
from pathlib import Path
import copy


async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


async def wait_finish_wrapper(sleep_time, storage, master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, wait_finish, sleep_time, storage, master)


def wait_finish(sleep_time, storage, master):
    while (
        storage.alive.check_alive(process_name='master') or
        storage.alive.check_alive(process_name='optimizer') or
        storage.alive.check_alive(process_name='scheduler')
    ):
        time.sleep(sleep_time)
        master.print_dict_state()


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):

        print(f"cd: {os.getcwd()}")

        config_file = data_dir.joinpath(
            'config_{}.json'.format(self.search_algorithm)
        )

        print(f"config_file: {str(config_file)}")

        commandline_args = [
            "start.py",
            "--config",
            format(config_file)
        ]

        if self.workspace.path.exists():
            self.workspace.clean()
        self.workspace.create()

        self.create_main()

        with patch.object(sys, 'argv', commandline_args):
            options = Arguments()
            master = LocalMaster(options)

            master.storage.alive.init_alive()
            loop = asyncio.get_event_loop()
            gather = asyncio.gather(
                start_master(master),
                wait_finish_wrapper(1, master.storage, master)
            )
            loop.run_until_complete(gather)
            self.evaluate(data_dir, work_dir, master.storage)

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
