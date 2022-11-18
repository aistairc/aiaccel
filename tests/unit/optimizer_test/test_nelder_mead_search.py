import json
import numpy as np

import aiaccel
import pytest
from aiaccel.config import ConfileWrapper
from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.optimizer.nelder_mead_optimizer import NelderMeadOptimizer
from aiaccel.parameter import load_parameter

from tests.base_test import BaseTest
from unittest.mock import patch


class TestNelderMeadOptimizer(BaseTest):

    @pytest.fixture(autouse=True)
    def setup_optimizer(self, clean_work_dir):
        self.options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        self.optimizer = NelderMeadOptimizer(self.options)
        self.optimizer.storage.alive.init_alive()
        yield
        self.optimizer = None

    def test_generate_initial_parameter(self):
        expected = [
            {'parameter_name': 'x1', 'type': 'FLOAT', 'value': 0.74},
            {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 2.98},
            {'parameter_name': 'x3', 'type': 'FLOAT', 'value': 3.62},
            {'parameter_name': 'x4', 'type': 'FLOAT', 'value': 0.9},
            {'parameter_name': 'x5', 'type': 'FLOAT', 'value': 1.99},
            {'parameter_name': 'x6', 'type': 'FLOAT', 'value': -2.78},
            {'parameter_name': 'x7', 'type': 'FLOAT', 'value': 1.0},
            {'parameter_name': 'x8', 'type': 'FLOAT', 'value': 4.97},
            {'parameter_name': 'x9', 'type': 'FLOAT', 'value': 1.98},
            {'parameter_name': 'x10', 'type': 'FLOAT', 'value': 4.03}
        ]

        _optimizer = NelderMeadOptimizer(self.options)
        _optimizer._rng = np.random.RandomState(0)
        _nelder_mead = _optimizer.generate_initial_parameter()
        self.optimizer._rng = np.random.RandomState(0)

        with patch.object(self.optimizer, "nelder_mead", None):
            assert self.optimizer.generate_initial_parameter() == expected

        with patch.object(self.optimizer, "nelder_mead", _nelder_mead):
            assert self.optimizer.generate_initial_parameter() is None

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
        self.optimizer.generate_initial_parameter()
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
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list(),
            rng=rng
        )
        # params = self.optimizer.nelder_mead.get_ready_parameters()
        params = self.optimizer.get_ready_parameters()
        assert params is not None
        setup_result(len(params))
        assert len(self.optimizer.generate_parameter()) > 0

        self.optimizer.nelder_mead._max_itr = 0
        assert self.optimizer.generate_parameter() is None

        # if len(self.parameter_pool) == 0:
        self.optimizer.nelder_mead = NelderMead(self.optimizer.params.get_parameter_list(), rng=rng)
        self.optimizer.generate_initial_parameter()
        with patch.object(self.optimizer, 'nelder_mead_main', return_value=[]):
            with patch.object(self.optimizer, 'parameter_pool', []):
                assert self.optimizer.generate_parameter() == []


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
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list(),
            rng=rng
        )
        # params = self.optimizer.nelder_mead.get_ready_parameters()
        # params = self.optimizer.get_ready_parameters()
        params = self.optimizer.nelder_mead._executing
        setup_result(len(params))
        assert len(self.optimizer.generate_parameter()) > 0

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
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list(),
            rng=rng
        )
        self.optimizer.nelder_mead._executing.append({'vertex_id': '001'})

        pool_p = {"vertex_id": "001"}
        assert self.optimizer.update_ready_parameter_name(pool_p, 'new') is None

        pool_p = {"vertex_id": "002"}
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
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(
            self.optimizer.params.get_parameter_list(),
            rng=rng
        )
        # assert len(self.nm.get_ready_parameters()) == 11
        assert len(self.optimizer.get_ready_parameters()) == 3

    def test_get_nm_results(self):
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(self.optimizer.params.get_parameter_list(), rng=rng)

        self.optimizer.get_nm_results()

        expected = [
            {
                'vertex_id': 'abc',
                'parameters': [{'parameter_name': 'x1', 'value': -4.87}, {'parameter_name': 'x2', 'value': -0.71}],
                'state': 'WaitInitialize',
                'itr': 1,
                'index': 1,
                'out_of_boundary': False
            },
            {
                'parameters': [{'parameter_name': 'x1', 'value': -4.87}, {'parameter_name': 'x2', 'value': -0.71}],
                'state': 'WaitInitialize',
                'itr': 1,
                'index': 1,
                'out_of_boundary': False
            }
        ]

        result_content = {'trial_id':0, 'result':123}
        with patch.object(self.optimizer.nelder_mead, '_executing', expected):
            with patch.object(self.optimizer.storage.result, 'get_any_trial_objective', return_value=result_content):
                self.optimizer.get_nm_results()

    def test__add_result(self):
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(self.optimizer.params.get_parameter_list(), rng=rng)
        self.optimizer.generate_initial_parameter()
        nm_results = [
            {
                'vertex_id': '0001',
                'parameters': [{'parameter_name': 'x1', 'value': -4.87}, {'parameter_name': 'x2', 'value': -0.71}],
                'state': 'WaitInitialize',
                'itr': 1,
                'index': 1,
                'out_of_boundary': False
            },
        ]
        order = [
            {
                'vertex_id':'0001',
                'parameters': [{'parameter_name': 'x1', 'value': -4.87}, {'parameter_name': 'x2', 'value': -0.71}]
            }
        ]
        order2 = [
            {
                'vertex_id':'invalid',
                'parameters': [{'parameter_name': 'x1', 'value': -4.87}, {'parameter_name': 'x2', 'value': -0.71}]
            }
        ]
        assert self.optimizer._add_result(nm_results) is None

        with patch.object(self.optimizer, 'order', order):
            assert self.optimizer._add_result(nm_results) is None

        with patch.object(self.optimizer, 'order', order2):
            assert self.optimizer._add_result(nm_results) is None


    def test_nelder_mead_main(self):
        rng = np.random.RandomState(0)
        self.optimizer.nelder_mead = NelderMead(self.optimizer.params.get_parameter_list(), rng=rng)
        self.optimizer.generate_initial_parameter()
        self.optimizer.nelder_mead_main()

        with patch.object(self.optimizer.nelder_mead, 'search', return_value=None):
            assert self.optimizer.nelder_mead_main() is None

        with patch.object(self.optimizer.nelder_mead, 'search', return_value=[]):
            assert self.optimizer.nelder_mead_main() is None

    def test__get_all_trial_id(self):
        with patch.object(self.optimizer.storage.trial, 'get_all_trial_id', return_value=None):
            assert self.optimizer._get_all_trial_id() == []

        expected = [1, 2, 3]
        with patch.object(self.optimizer.storage.trial, 'get_all_trial_id', return_value=expected):
            assert self.optimizer._get_all_trial_id() == expected
