import pytest
from aiaccel.optimizer.sobol_optimizer import SobolOptimizer
from aiaccel.config import load_config

from tests.base_test import BaseTest
from unittest.mock import patch


class TestSobolOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, data_dir, create_tmp_config):
        self.data_dir = data_dir
<<<<<<< HEAD
        self.optimizer = SobolOptimizer(self.configs["config_sobol.json"])
=======
        self.config_sobol_path = create_tmp_config(self.config_sobol_path)
        self.options = {
            'config': self.config_sobol_path,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = SobolOptimizer(self.options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        yield
        self.optimizer = None

    def test_pre_process(self):
        optimizer = SobolOptimizer(self.configs["config_sobol.json"])
        assert optimizer.pre_process() is None

<<<<<<< HEAD
        optimizer = SobolOptimizer(self.configs["config_sobol.json"])
        with patch.object(optimizer.storage.trial, 'get_finished', return_value=[0,1,2]):
=======
        options = self.options.copy()
        optimizer = SobolOptimizer(options)
        with patch.object(optimizer.storage.trial, 'get_finished', return_value=[0, 1, 2]):
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
            assert optimizer.pre_process() is None

    def test_generate_parameter(self):
        optimizer = SobolOptimizer(self.configs["config_sobol.json"])
        optimizer.pre_process()

        assert len(optimizer.generate_parameter()) > 0

        with patch.object(optimizer, 'generate_index', None):
            assert len(optimizer.generate_parameter()) > 0

<<<<<<< HEAD

    def test_generate_initial_parameter(self):
        optimizer = SobolOptimizer(self.configs["config_sobol.json"])
        optimizer.pre_process()
        optimizer.generate_initial_parameter()

        optimizer = SobolOptimizer(self.configs["config_sobol_no_initial.json"])
=======
    def test_generate_initial_parameter(self, create_tmp_config):
        options = self.options.copy()
        optimizer = SobolOptimizer(options)
        optimizer.pre_process()
        optimizer.generate_initial_parameter()

        self.config_sobol_path = create_tmp_config(self.data_dir / 'config_sobol_no_initial.json')

        optimizer = SobolOptimizer(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        optimizer.pre_process()
        optimizer.generate_initial_parameter()
