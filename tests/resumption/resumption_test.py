import subprocess
from pathlib import Path

from aiaccel.config import is_multi_objective
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from tests.integration.integration_test import IntegrationTest

import yaml


class ResumptionTest(IntegrationTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config):
        config = self.load_config_for_test(
            self.configs['config_{}.json'.format(self.search_algorithm)]
        )

        if is_multi_objective(config):
            user_main_file = self.test_data_dir / 'original_main_mo.py'
        else:
            user_main_file = None

        base_dir = Path(config.config_path).parent
        new_config_file_path = base_dir / f'config_{self.search_algorithm}_pylocal.yaml'

        with open(config.config_path, 'r') as f:
            yml = yaml.load(f, Loader=yaml.SafeLoader)
        yml['resource']['type'] = 'python_local'

        with open(new_config_file_path, 'w') as f:
            f.write(yaml.dump(yml, default_flow_style=False))

        with self.create_main(user_main_file):
            workspace = Workspace(config.generic.workspace)
            storage = Storage(workspace.storage_file_path)
            subprocess.Popen(['aiaccel-start', '--config', str(new_config_file_path), '--clean']).wait()

            final_result_at_one_time = self.get_final_result(storage)
            print('at one time', final_result_at_one_time)

        # max trial 5
        with self.create_main(user_main_file):
            config = self.load_config_for_test(
                self.configs['config_{}_resumption.json'.format(self.search_algorithm)]
            )
            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--clean']).wait()
            subprocess.Popen(['aiaccel-view', '--config', str(config.config_path)]).wait()

        # resume
        with self.create_main(user_main_file):
            config = self.load_config_for_test(
                self.configs['config_{}.json'.format(self.search_algorithm)]
            )
            workspace = Workspace(config.generic.workspace)
            storage = Storage(workspace.storage_file_path)
            subprocess.Popen(['aiaccel-start', '--config', str(new_config_file_path), '--resume', '3']).wait()
            final_result_resumption = self.get_final_result(storage)
            print('resumption steps finished', final_result_resumption)

        assert final_result_at_one_time == final_result_resumption

    def get_final_result(self, storage):
        data = storage.result.get_all_result()
        return [data[trial_id] for trial_id in data.keys()][-1]
