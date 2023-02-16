import asyncio
import subprocess
from pathlib import Path

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.master.create import create_master
from aiaccel.master.local_master import LocalMaster
from aiaccel.scheduler.create import create_scheduler
from aiaccel.scheduler.local_scheduler import LocalScheduler

from tests.base_test import BaseTest



async def start_master(master):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, master.start)


class TestWrapper(BaseTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config, tmpdir, work_dir):

        #
        # local test
        #
        with self.create_main():
            config_file = data_dir.joinpath('config_sh.yaml')
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
            self.evaluate(work_dir, storage)

            self.result_comparison.append(storage.result.get_objectives())

    def evaluate(self, work_dir, storage):
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
