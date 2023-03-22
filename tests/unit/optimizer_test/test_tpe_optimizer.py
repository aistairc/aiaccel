from unittest.mock import patch

import numpy as np
import pytest

from aiaccel.config import Config
from aiaccel.optimizer import TpeOptimizer
from aiaccel.optimizer.tpe_optimizer import TPESamplerWrapper, create_distributions
from aiaccel.parameter import load_parameter
from tests.base_test import BaseTest


class TestTPESamplerWrapper(BaseTest):
    def test_get_startup_trials(self):
        tpe_sampler_wrapper = TPESamplerWrapper()
        assert tpe_sampler_wrapper.get_startup_trials() == 10


class TestTpeOptimizer(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.config_tpe_path = create_tmp_config(self.data_dir / "config_tpe.json")
        self.options = {
            "config": self.config_tpe_path,
            "resume": None,
            "clean": False,
            "fs": False,
            "process_name": "optimizer",
        }
        self.optimizer = TpeOptimizer(self.options)
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
        with patch.object(
            self.optimizer.storage.result, "get_any_trial_objective", return_value=1
        ):
            assert self.optimizer.check_result() is None

    def test_is_startup_trials(self):
        self.optimizer.pre_process()
        assert self.optimizer.is_startup_trials()

    def test_generate_parameter(self):
        self.optimizer.pre_process()
        assert len(self.optimizer.generate_parameter()) > 0

        # if ((not self.is_startup_trials()) and (len(self.parameter_pool) >= 1))
        with patch.object(self.optimizer, "check_result", return_value=None):
            with patch.object(self.optimizer, "is_startup_trials", return_value=False):
                with patch.object(self.optimizer, "parameter_pool", [{}, {}, {}]):
                    assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) >= self.config.num_node.get()
        with patch.object(self.optimizer.config.num_node, "get", return_value=0):
            with patch.object(self.optimizer, "is_startup_trials", return_value=False):
                assert self.optimizer.generate_parameter() is None

    def test_generate_initial_parameter(self, create_tmp_config):
        options = self.options.copy()
        self.config_tpe_path = create_tmp_config(self.data_dir / "config_tpe_2.json")
        optimizer = TpeOptimizer(self.options)
        (optimizer.ws / "storage" / "storage.db").unlink()

        optimizer.__init__(options)
        optimizer.pre_process()
        assert len(optimizer.generate_initial_parameter()) > 0
        assert len(optimizer.generate_initial_parameter()) > 0

    def test_create_study(self):
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
    config = Config(data_dir / "config_tpe_2.json")
    params = load_parameter(config.hyperparameters.get())
    dist = create_distributions(params)
    assert type(dist) is dict

    config = Config(data_dir / "config_tpe_categorical.json")
    params = load_parameter(config.hyperparameters.get())
    dist = create_distributions(params)
    assert type(dist) is dict

    config = Config(data_dir / "config_tpe_invalid_type.json")
    params = load_parameter(config.hyperparameters.get())
    with pytest.raises(TypeError):
        create_distributions(params)
