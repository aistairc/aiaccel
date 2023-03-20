# Test when initial is not set

import subprocess
from pathlib import Path

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from tests.base_test import BaseTest


class NoInitialTest(BaseTest):
    search_algorithm = None

    def test_run(self, work_dir, create_tmp_config):
        test_data_dir = Path(__file__).resolve().parent.joinpath("no_initial_test_benchmark", "test_data")
        config_file = test_data_dir.joinpath("config_{}.yaml".format(self.search_algorithm))
        config_file = create_tmp_config(config_file)
        self.config = Config(config_file)
        python_file = test_data_dir.joinpath("user.py")

        with self.create_main(python_file):
            storage = Storage(ws=Path(self.config.workspace.get()))
            subprocess.Popen(["aiaccel-start", "--config", str(config_file), "--clean"]).wait(
                timeout=self.config.batch_job_timeout.get()
            )
        self.evaluate(work_dir, storage)

    def evaluate(self, work_dir, storage):
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished == self.config.trial_number.get()
        assert ready == 0
        assert running == 0
        final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
        assert final_result.exists()
