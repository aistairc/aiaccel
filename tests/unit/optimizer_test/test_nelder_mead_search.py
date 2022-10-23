import json

import aiaccel
import pytest
from aiaccel.config import ConfileWrapper
from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.optimizer.nelder_mead_optimizer import NelderMeadOptimizer
from aiaccel.parameter import load_parameter

from tests.base_test import BaseTest


class TestNelderMeadOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = NelderMeadOptimizer(options)
        self.optimizer.storage.alive.init_alive()
        yield
        self.optimizer = None

    def test_pre_process(self):
        assert self.optimizer.pre_process() is None

        with open(self.config_json, 'r') as f:
            json_obj = json.load(f)
        json_obj['optimize']['goal'] = aiaccel.goal_maximize
        config = ConfileWrapper(json_obj, 'json_object')
        json_obj['optimize']['goal'] = aiaccel.goal_maximize
        self.optimizer.config = config
        self.optimizer.post_process()
        assert self.optimizer.pre_process() is None

    def test_check_result(self, setup_result, work_dir):
        self.optimizer.pre_process()
        setup_result(1)
        # params = self.optimizer.nelder_mead.get_ready_parameters()
        params = self.optimizer.get_ready_parameters()
        print(params)
        assert self.optimizer.check_result() is None

    def test_generate_parameter(
        self,
        load_test_config_org,
        setup_result,
        work_dir
    ):
        self.optimizer.pre_process()
        # config = load_test_config()
        config = load_test_config_org()
        self.optimizer.params = load_parameter(
            config.get('optimize',
                       'parameters_for_TestNelderMead'))
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list()
        )
        # params = self.optimizer.nelder_mead.get_ready_parameters()
        params = self.optimizer.get_ready_parameters()
        assert params is not None
        setup_result(len(params))
        assert self.optimizer.generate_parameter() is None
        assert self.optimizer.generate_parameter() is None
        assert self.optimizer.generate_parameter() is None

        self.optimizer.nelder_mead._max_itr = 0
        assert self.optimizer.generate_parameter() is None

    def test_generate_parameter2(
        self,
        load_test_config_org,
        setup_result,
        work_dir
    ):
        self.optimizer.pre_process()
        config = load_test_config_org()
        self.optimizer.params = load_parameter(
            config.get('optimize',
                       'parameters_for_TestNelderMeadSearch'))
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list()
        )
        # params = self.optimizer.nelder_mead.get_ready_parameters()
        # params = self.optimizer.get_ready_parameters()
        params = self.optimizer.nelder_mead._executing
        setup_result(len(params))
        assert self.optimizer.generate_parameter() is None
        assert self.optimizer.generate_parameter() is None

    def test_update_ready_parameter_name(
        self,
        load_test_config_org,
        setup_result,
        work_dir
    ):
        self.optimizer.pre_process()
        config = load_test_config_org()
        self.optimizer.params = load_parameter(
            config.get(
                'optimize',
                'parameters_for_TestNelderMead'
            )
        )
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list()
        )
        self.optimizer.nelder_mead._executing.append({'vertex_id': '001'})
        # assert self.nm.update_ready_parameter_name('001', 'new') is None
        pool_p = {"vertex_id": "001"}
        assert self.optimizer.update_ready_parameter_name(pool_p, 'new') is None

    def test_get_ready_parameters(
        self,
        load_test_config_org,
        setup_result,
        work_dir
    ):
        self.optimizer.pre_process()
        config = load_test_config_org()
        self.optimizer.params = load_parameter(
            config.get(
                'optimize',
                'parameters_for_TestNelderMead'
            )
        )
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list()
        )
        # assert len(self.nm.get_ready_parameters()) == 11
        assert len(self.optimizer.get_ready_parameters()) == 3
