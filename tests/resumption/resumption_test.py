import subprocess
from pathlib import Path

from aiaccel.config import load_config
from aiaccel.storage.storage import Storage

from tests.integration.integration_test import IntegrationTest


class ResumptionTest(IntegrationTest):
    search_algorithm = None

    def test_run(self, data_dir, create_tmp_config):
        with self.create_main():
            config = self.configs['config_{}.json'.format(self.search_algorithm)]
            storage = Storage(ws=Path(config.generic.workspace))
            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--clean']).wait()
            final_result_at_one_time = self.get_final_result(storage)
            print('at one time', final_result_at_one_time)

        # max trial 5
        with self.create_main():
            config = self.configs['config_{}_resumption.json'.format(self.search_algorithm)]
            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--clean']).wait()

        # resume
        with self.create_main():
            config = self.configs['config_{}.json'.format(self.search_algorithm)]
            storage = Storage(ws=Path(config.generic.workspace))

            subprocess.Popen(['aiaccel-start', '--config', str(config.config_path), '--resume', '4']).wait()
            final_result_resumption = self.get_final_result(storage)
            print('resumption steps finished', final_result_resumption)

        assert final_result_at_one_time == final_result_resumption

    def get_final_result(self, storage):
        data = storage.result.get_all_result()
        return [d.objective for d in data][-1]
