from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from subprocess import Popen

import pytest
from omegaconf.dictconfig import DictConfig

from aiaccel.common import file_final_result
from aiaccel.config import load_config
from aiaccel.storage.storage import Storage
from aiaccel.workspace import Workspace
from tests.base_test import BaseTest

argnames_test_run = "config_filename, expected_returncode"
argvalues_test_run = [
    ("config_budget-specified-grid.yaml", 0),
    ("config_budget-specified-grid_small_trial_number.yaml", 1),
    ("config_budget-specified-grid_accept_small_trial_number.yaml", 0),
    ("config_budget-specified-grid_excessive_trial_number.yaml", 0),
    ("config_budget-specified-grid_invalid_parameter_range.yaml", 1),
]


class AdditionalBudgetSpecifiedGridTest(BaseTest):
    search_algorithm = None

    @pytest.fixture(autouse=True)
    def setup_test_data_dir(self) -> Generator[None, None, None]:
        self.test_data_dir = Path(__file__).resolve().parent.joinpath(
            'additional_budget_specified_grid_test_benchmark',
            'test_data'
        )
        yield
        self.test_data_dir = None

    def main(self, config_file: Path) -> tuple[DictConfig, Storage, Popen]:
        config = load_config(config_file)
        python_file = self.test_data_dir.joinpath('user.py')
        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)
        with self.create_main(python_file):
            # storage = Storage(ws=Path(config.generic.workspace))
            popen = Popen(
                ['aiaccel-start', '--config', str(config_file), '--clean']
            )
            popen.wait(timeout=config.generic.batch_job_timeout)

        return config, storage, popen

    @pytest.mark.parametrize(argnames_test_run, argvalues_test_run)
    def test_run(
        self,
        work_dir: Path,
        create_tmp_config: Callable[[Path], Path],
        config_filename: str,
        expected_returncode: int
    ) -> None:
        config_file = self.test_data_dir.joinpath(config_filename)
        config_file = create_tmp_config(config_file)
        _, storage, popen = self.main(config_file)
        assert popen.returncode == expected_returncode
        if expected_returncode == 0:
            self.evaluate(work_dir, storage)

    def evaluate(
        self,
        work_dir: Path,
        storage: Storage
    ) -> None:
        num_running = storage.get_num_running()
        num_ready = storage.get_num_ready()
        assert num_running == 0
        assert num_ready == 0
