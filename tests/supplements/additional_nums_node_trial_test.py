"""Suuplemental test for variation of the numbers of node and trials.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from subprocess import Popen

import pytest
import yaml
from omegaconf.dictconfig import DictConfig

from aiaccel.common import dict_result, file_final_result
from aiaccel.config import load_config
from aiaccel.storage.storage import Storage
from aiaccel.workspace import Workspace
from tests.base_test import BaseTest


class AdditionalNumsNodeTrialTest(BaseTest):
    search_algorithm = None
    python_program = 'user.py'

    @pytest.fixture(autouse=True)
    def setup_optimizer(self) -> None:
        self.test_data_dir = Path(__file__).resolve().parent.joinpath(
            'additional_nums_node_trial_test', 'test_data'
        )
        self.python_file = self.test_data_dir.joinpath(self.python_program)

    @pytest.mark.parametrize(
        'num_workers, num_trial',
        [
            (2, 10),  # num_workers < num_grid < num_trials
            (5, 10),  # num_grid < num_workers < num_trials
            (10, 5),  # num_grid < num_trials < num_workers
            (10, 2),  # num_trials < num_grid < num_workers
            (3, 2),   # num_trials < num_workers < num_grid
            (2, 3)    # num_workers < num_trials < num_grid
        ]
    )
    def test_run(
        self,
        num_workers: int,
        num_trial: int,
        work_dir: Path,
        create_tmp_config: Callable[[Path, Path]]
    ) -> None:
        config_file = self.test_data_dir.joinpath(
            'config_{}.yaml'.format(self.search_algorithm)
        )
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        cfg['resource']['num_workers'] = num_workers
        cfg['optimize']['trial_number'] = num_trial
        with open(config_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        config_file = create_tmp_config(config_file)
        config = load_config(config_file)

        workspace = Workspace(config.generic.workspace)
        storage = Storage(workspace.storage_file_path)

        with self.create_main(self.python_file):
            popen = Popen(
                ['aiaccel-start', '--config', str(config_file), '--clean']
            )
            popen.wait(timeout=config.generic.batch_job_timeout)
        self.evaluate(work_dir, config, storage)

    def evaluate(self, work_dir: Path, config: DictConfig, storage: Storage) -> None:
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished <= config.optimize.trial_number
        assert ready == 0
        assert running == 0
