"""Suuplemental test for variation of the numbers of node and trials.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from subprocess import Popen

import pytest
import yaml

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage

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
        'num_node, num_trial',
        [
            (2, 10),  # num_node < num_grid < num_trials
            (5, 10),  # num_grid < num_node < num_trials
            (10, 5),  # num_grid < num_trials < num_node
            (10, 2),  # num_trials < num_grid < num_node
            (3, 2),   # num_trials < num_node < num_grid
            (2, 3)    # num_node < num_trials < num_grid
        ]
    )
    def test_run(
        self,
        num_node: int,
        num_trial: int,
        work_dir: Path,
        create_tmp_config: Callable[[Path, Path]]
    ) -> None:
        self.config_file = self.test_data_dir.joinpath(
            'config_{}.yaml'.format(self.search_algorithm)
        )
        with open(self.config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        cfg['resource']['num_node'] = num_node
        cfg['optimize']['trial_number'] = num_trial
        with open(self.config_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        self.config_file = create_tmp_config(self.config_file)
        self.config = Config(self.config_file)

        with self.create_main(self.python_file):
            storage = Storage(ws=Path(self.config.generic.workspace))
            popen = Popen(
                ['aiaccel-start', '--config', str(self.config_file), '--clean']
            )
            popen.wait(timeout=self.config.generic.batch_job_timeout)
        self.evaluate(work_dir, storage)

    def evaluate(self, work_dir: Path, storage: Storage) -> None:
        running = storage.get_num_running()
        ready = storage.get_num_ready()
        finished = storage.get_num_finished()
        assert finished <= self.config.optimize.trial_number
        assert ready == 0
        assert running == 0
        final_result = work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
        assert final_result.exists()
