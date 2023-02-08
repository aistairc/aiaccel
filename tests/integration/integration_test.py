import asyncio
import subprocess
from pathlib import Path

import aiaccel
from aiaccel.config import load_config

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

    def test_run(self, data_dir, create_tmp_config, tmpdir, work_dir):

        #
        # local test
        #
        with self.create_main():
            print(self.search_algorithm)
            config = self.load_config_for_test(
                self.configs['config_{}.json'.format(self.search_algorithm)]
            )

            # master
            master = create_master(config.resource.type)
            assert master == LocalMaster

            # scheduler
            scheduler = create_scheduler(config.resource.type)
            assert scheduler == LocalScheduler

            storage = Storage(ws=Path(config.generic.workspace))
            print(f'\n{config.config_path}\n')
            subprocess.Popen(['cat', str(config.config_path)]).wait()
            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--clean']).wait()
            self.evaluate(config)

            self.result_comparison.append(storage.result.get_objectives())

        #
        # pylocal test
        #
        with self.create_main():
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
            assert config.resource.type == 'python_local'

            # master
            master = create_master(config.resource.type)
            assert master == PylocalMaster

            # scheduler
            scheduler = create_scheduler(config.resource.type)
            assert scheduler == PylocalScheduler

            storage = Storage(ws=Path(config.generic.workspace))

            subprocess.Popen(['cat', str(config.config_path)]).wait()
            subprocess.Popen(['aiaccel-start', '--config', str(new_config_file_path), '--clean']).wait()

            self.evaluate(config)

            print(storage.result.get_objectives())
            self.result_comparison.append(storage.result.get_objectives())

        data_0 = self.result_comparison[0]  # local result
        data_1 = self.result_comparison[1]  # pylocal result
        assert len(data_0) == len(data_1)
        for i in range(len(data_0)):
            assert data_0[i] == data_1[i]

    def evaluate(self, config):
        storage = Storage(ws=Path(config.generic.workspace))
        work_dir = Path(config.generic.workspace)

        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == config.optimize.trial_number
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
