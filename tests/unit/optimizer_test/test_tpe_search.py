import pytest
from aiaccel.config import Config
from aiaccel.optimizer.tpe_optimizer import (TpeOptimizer, TPESamplerWrapper,
                                             create_distributions)
from aiaccel.parameter import load_parameter

from tests.base_test import BaseTest
from unittest.mock import patch

class TestTPESamplerWrapper(BaseTest):

    def test_get_startup_trials(self):
        tpe_sampler_wrapper = TPESamplerWrapper()
        assert tpe_sampler_wrapper.get_startup_trials() == 10


class TestTpeOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir, data_dir):
        options = {
            'config': data_dir / 'config_tpe.json',
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = TpeOptimizer(options)
        self.optimizer.storage.alive.init_alive()
        yield
        self.optimizer = None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_post_process(self):
        self.optimizer.pre_process()
        assert self.optimizer.post_process() is None

    def test_check_result(self, setup_hp_finished, setup_result, work_dir):
        self.optimizer.pre_process()
        assert self.optimizer.check_result() is None

    def test_is_startup_trials(self):
        self.optimizer.pre_process()
        assert self.optimizer.is_startup_trials()

    def test_generate_parameter(self):
        self.optimizer.pre_process()
        assert len(self.optimizer.generate_parameter()) > 0

        # if ((not self.is_startup_trials()) and (len(self.parameter_pool) >= 1))
        with patch.object(self.optimizer, 'check_result', return_value=None):
            with patch.object(self.optimizer, 'is_startup_trials', return_value=False):
                with patch.object(self.optimizer, 'parameter_pool', [{},{},{}]):
                    assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) >= self.config.num_node.get()
        with patch.object(self.optimizer.config.num_node, 'get', return_value=0):
            assert self.optimizer.generate_parameter() is None

    def test_generate_initial_parameter(self):
        self.optimizer.pre_process()
        assert len(self.optimizer.generate_initial_parameter()) > 0

    def test_create_study(self):
        assert self.optimizer.create_study() is None

        with patch.object(self.optimizer, 'study', None):
            assert self.optimizer.create_study() is None

        with patch.object(self.optimizer, 'study', object):
            assert self.optimizer.create_study() is None

    def test_serialize(self):
        self.optimizer.create_study()
        self.optimizer.trial_id.initial(num=0)
        self.optimizer.storage.trial.set_any_trial_state(trial_id=0, state="ready")
        assert self.optimizer._serialize(trial_id=0) is None

    def test_deserialize(self):
        self.optimizer.pre_process()
        self.optimizer.trial_id.initial(num=0)
        self.optimizer.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        self.optimizer._serialize(trial_id=0)
        assert self.optimizer._deserialize(trial_id=0) is None


def test_create_distributions(data_dir):
    config = Config(data_dir / 'config_tpe.json')
    params = load_parameter(config.hyperparameters.get())
    dist = create_distributions(params)
    assert type(dist) is dict
