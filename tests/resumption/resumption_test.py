import subprocess
from pathlib import Path

from aiaccel.config import Config
from aiaccel.storage import Storage
from tests.integration.integration_test import IntegrationTest


class ResumptionTest(IntegrationTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config):
        config_file = data_dir.joinpath('config_{}.json'.format(self.search_algorithm))
        config = Config(config_file)
        is_multi_objective = isinstance(config.goal.get(), list)

        if is_multi_objective:
            user_main_file = self.test_data_dir / 'original_main_mo.py'
        else:
            user_main_file = None

        with self.create_main(user_main_file):
            config_file = create_tmp_config(config_file)
            config = Config(config_file)
            storage = Storage(ws=Path(config.workspace.get()))
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
            final_result_at_one_time = self.get_final_result(storage)
            print('at one time', final_result_at_one_time)

        # max trial 5
        with self.create_main(user_main_file):
            config_file = data_dir / f'config_{self.search_algorithm}_resumption.json'
            config_file = create_tmp_config(config_file)
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean']).wait()
            subprocess.Popen(['aiaccel-view', '--config', str(config_file)]).wait()

        # resume
        with self.create_main(user_main_file):
            config_file = data_dir.joinpath(f'config_{self.search_algorithm}.json')
            config_file = create_tmp_config(config_file)
            storage = Storage(ws=Path(config.workspace.get()))
            subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--resume', '3']).wait()
            final_result_resumption = self.get_final_result(storage)
            print('resumption steps finished', final_result_resumption)

        assert final_result_at_one_time == final_result_resumption

    def get_final_result(self, storage):
        data = storage.result.get_all_result()
        return [data[trial_id] for trial_id in data.keys()][-1]
