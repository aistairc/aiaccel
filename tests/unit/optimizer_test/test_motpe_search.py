import warnings
from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.optimizer.motpe_optimizer import MOTpeOptimizer
from tests.base_test import BaseTest


class TestMOTpeOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.config = self.load_config_for_test(self.configs['config_motpe.json'])
        self.optimizer = MOTpeOptimizer(self.config)
        yield
        self.optimizer = None

    def test_check_result(self, setup_hp_finished, setup_result, work_dir):
        # self.optimizer.inner_loop_main_process()
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            with patch.object(self.optimizer.storage.result, 'get_any_trial_objective', return_value=1):
                self.optimizer.check_result()
        with patch.object(self.optimizer.storage.result, 'get_any_trial_objective', return_value=[0, 1]):
            assert self.optimizer.check_result() is None

    def test_is_startup_trials(self):
        assert self.optimizer.is_startup_trials()

    def test_generate_parameter(self):
        assert len(self.optimizer.generate_parameter()) > 0

        # if ((not self.is_startup_trials()) and (len(self.parameter_pool) >= 1))
        with patch.object(self.optimizer, 'check_result', return_value=None):
            with patch.object(self.optimizer, 'is_startup_trials', return_value=False):
                with patch.object(self.optimizer, 'parameter_pool', [{}, {}, {}]):
                    assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) >= self.config.num_workers.get()
        self.optimizer.config.resource.num_workers = 0
        with patch.object(self.optimizer, 'is_startup_trials', return_value=False):
            assert self.optimizer.generate_parameter() is None

    def test_generate_initial_parameter(self, create_tmp_config):
        self.workspace.clean()
        self.workspace.create()
        config = self.load_config_for_test(create_tmp_config(self.data_dir / 'config_motpe_no_initial_params.json'))
        optimizer = MOTpeOptimizer(config)
        self.workspace.clean()
        self.workspace.create()
        optimizer.__init__(config)
        assert len(optimizer.generate_initial_parameter()) > 0
        assert len(optimizer.generate_initial_parameter()) > 0

    def test_create_study(self):
        self.workspace.clean()
        self.workspace.create()
        assert self.optimizer.create_study() is None

    def test_serialize(self):
        self.workspace.clean()
        self.workspace.create()
        optimizer = MOTpeOptimizer(self.config)
        optimizer.trial_id.initial(num=0)
        optimizer.storage.trial.set_any_trial_state(trial_id=0, state="ready")
        optimizer._rng = np.random.RandomState(0)
        assert optimizer.serialize(trial_id=0) is None

    def test_deserialize(self):
        self.workspace.clean()
        self.workspace.create()
        optimizer = MOTpeOptimizer(self.config)
        optimizer.trial_id.initial(num=0)
        optimizer.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        optimizer.serialize(trial_id=0)
        assert optimizer.deserialize(trial_id=0) is None
