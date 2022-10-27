# resume の有無でtpe・NMの最適化結果が変わることがあったため、追加したテスト
# 通常実行・初期点計算から再開・初期点計算後から再開の3パターンで結果が変わることがあったので、それぞれ比較

import subprocess
from pathlib import Path

from aiaccel.config import Config
from aiaccel.storage.storage import Storage

from tests.integration.integration_test import IntegrationTest


class AdditionalResumptionTest(IntegrationTest):
    search_algorithm = None

    def test_run(self, cd_work, data_dir, work_dir):
        test_data_dir = Path(__file__).resolve().parent.joinpath('benchmark', 'test_data')
        config_file = test_data_dir.joinpath('config_{}.yaml'.format(self.search_algorithm))
        config = Config(config_file)
        storage = Storage(ws=Path(config.workspace.get()))
        subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--clean'],
                         cwd=test_data_dir).wait()
        final_result_at_one_time = self.get_final_result(storage)
        print('at one time', final_result_at_one_time)

        # resume in initial point
        storage = Storage(ws=Path(config.workspace.get()))
        subprocess.Popen(['aiaccel-start', '--config', str(config_file), '--resume', '2'], cwd=test_data_dir).wait()
        final_result_resumption_in_initial = self.get_final_result(storage)
        print('resumption steps in initial point finished', final_result_resumption_in_initial)

        assert final_result_at_one_time == final_result_resumption_in_initial

        # resume
        storage = Storage(ws=Path(config.workspace.get()))
        subprocess.Popen(['aiaccel-start', '--config', str(config_file),
                          '--resume', '11'], cwd=test_data_dir).wait()
        final_result_resumption = self.get_final_result(storage)
        print('resumption steps finished', final_result_resumption)

        assert final_result_at_one_time == final_result_resumption

    def get_final_result(self, storage):
        data = storage.result.get_all_result()
        return [d.objective for d in data]
