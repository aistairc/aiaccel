from aiaccel.master.local_master import LocalMaster
from aiaccel.util.filesystem import get_file_hp_finished, get_file_hp_ready, \
    get_file_hp_running
from tests.base_test import BaseTest
import aiaccel
import asyncio
import time
from unittest.mock import patch
import sys


async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


async def wait_finish_wrapper(sleep_time, ad, master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, wait_finish, sleep_time, ad, master)


def wait_finish(sleep_time, ad, master):
    while ad.joinpath(aiaccel.alive_master).exists() or \
            ad.joinpath(aiaccel.alive_optimizer).exists() or \
            ad.joinpath(aiaccel.alive_scheduler).exists():
        time.sleep(sleep_time)
        master.print_dict_state()


class IntegrationTest(BaseTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):
        config_file = data_dir.joinpath(
            'config_{}.json'.format(self.search_algorithm)
        )
        commandline_args = [
            "start.py",
            "--config",
            format(config_file)
        ]

        with patch.object(sys, 'argv', commandline_args):
            from aiaccel import start
            master = start.Master()
            loop = asyncio.get_event_loop()
            gather = asyncio.gather(
                start_master(master),
                wait_finish_wrapper(1, work_dir.joinpath(aiaccel.dict_alive), master)
            )
            loop.run_until_complete(gather)
            self.evaluate(data_dir, work_dir)

        # master = LocalMaster(options)
        # loop = asyncio.get_event_loop()
        # gather = asyncio.gather(
        #     start_master(master),
        #     wait_finish_wrapper(1, work_dir.joinpath(aiaccel.dict_alive), master)
        # )
        # loop.run_until_complete(gather)
        # self.evaluate(data_dir, work_dir)

    def evaluate(self, data_dir, work_dir):
        assert len(get_file_hp_finished(work_dir)) == self.config.trial_number.get()
        assert len(get_file_hp_ready(work_dir)) == 0
        assert len(get_file_hp_running(work_dir)) == 0
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
