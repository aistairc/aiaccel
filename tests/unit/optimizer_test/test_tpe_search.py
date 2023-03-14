from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.config import load_config
from aiaccel.optimizer.tpe_optimizer import (TpeOptimizer, TPESamplerWrapper,
                                             create_distributions)
from aiaccel.parameter import HyperParameterConfiguration
from tests.base_test import BaseTest


class TestTPESamplerWrapper(BaseTest):

    def test_get_startup_trials(self):
        tpe_sampler_wrapper = TPESamplerWrapper()
        assert tpe_sampler_wrapper.get_startup_trials() == 10


class TestTpeOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.optimizer = TpeOptimizer(self.load_config_for_test(self.configs['config_tpe.json']))
        yield
        self.optimizer = None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

    def test_post_process(self):
        self.optimizer.pre_process()
        assert self.optimizer.post_process() is None

    def test_check_result(self):
        self.optimizer.pre_process()
        self.optimizer.inner_loop_main_process()
        with patch.object(self.optimizer.storage.result, 'get_any_trial_objective', return_value=1):
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
                with patch.object(self.optimizer, 'parameter_pool', [{}, {}, {}]):
                    assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) >= self.config.resource.num_node
        _tmp_num_node = self.optimizer.config.resource.num_node
        self.optimizer.config.resource.num_node = 0
        assert self.optimizer.generate_parameter() is None
        self.optimizer.config.resource.num_node = _tmp_num_node

    def test_generate_initial_parameter(self):
        optimizer = TpeOptimizer(self.load_config_for_test(self.configs['config_tpe_2.json']))
        (optimizer.ws / 'storage' / 'storage.db').unlink()

        optimizer.__init__(self.load_config_for_test(self.configs['config_tpe_2.json']))
        optimizer.pre_process()
        assert len(optimizer.generate_initial_parameter()) > 0

        optimizer.pre_process()
        assert len(optimizer.generate_initial_parameter()) > 0

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
        self.optimizer._rng = np.random.RandomState(0)
        assert self.optimizer._serialize(trial_id=0) is None

    def test_deserialize(self):
        self.optimizer.pre_process()
        self.optimizer.trial_id.initial(num=0)
        self.optimizer.storage.trial.set_any_trial_state(trial_id=0, state="finished")
        self.optimizer._serialize(trial_id=0)
        assert self.optimizer._deserialize(trial_id=0) is None


def test_create_distributions(data_dir):
    config = load_config(data_dir / 'config_tpe_2.json')
    params = HyperParameterConfiguration(config.optimize.parameters)
    dist = create_distributions(params)
    assert type(dist) is dict

    config = load_config(data_dir / 'config_tpe_categorical.json')
    params = HyperParameterConfiguration(config.optimize.parameters)
    dist = create_distributions(params)
    assert type(dist) is dict

    config = load_config(data_dir / 'config_tpe_invalid_type.json')
    params = HyperParameterConfiguration(config.optimize.parameters)
    with pytest.raises(TypeError):
        create_distributions(params)
