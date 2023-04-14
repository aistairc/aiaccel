import pytest

from aiaccel.optimizer.sobol_optimizer import SobolOptimizer

from tests.base_test import BaseTest
from unittest.mock import patch


class TestSobolOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
        self.optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol.json"]))
        yield
        self.optimizer = None

    def test_pre_process(self):
        optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol.json"]))
        assert optimizer.pre_process() is None

        optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol.json"]))
        with patch.object(optimizer.storage.trial, 'get_finished', return_value=[0, 1, 2]):
            assert optimizer.pre_process() is None

    def test_generate_parameter(self):
        optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol.json"]))
        optimizer.pre_process()

        assert len(optimizer.generate_parameter()) > 0

        with patch.object(optimizer, 'generate_index', 0):
            assert len(optimizer.generate_parameter()) > 0

    def test_generate_initial_parameter(self):
        optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol.json"]))
        optimizer.pre_process()

        log_strings = []
        with patch.object(optimizer.logger, "warning", lambda s: log_strings.append(s)):
            optimizer.generate_initial_parameter()
            assert len(log_strings) > 0

        optimizer = SobolOptimizer(self.load_config_for_test(self.configs["config_sobol_no_initial.json"]))
        optimizer.pre_process()
        optimizer.generate_initial_parameter()
