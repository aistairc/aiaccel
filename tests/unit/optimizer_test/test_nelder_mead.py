# from ConfigSpace.read_and_write import json as configspace_json
import copy
import json
from re import T
from unittest.mock import patch

import numpy as np
import pytest

import aiaccel
from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.parameter import load_parameter
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import load_yaml
from tests.base_test import BaseTest


class TestNelderMead(BaseTest):
    @pytest.fixture(autouse=True)
    def setup_nelder_mead(self, load_test_config):
        self.config = load_test_config()
        # params = load_parameter(config.get('optimize', 'parameters'))
        params = load_parameter(self.config.hyperparameters.get())
        rng = np.random.RandomState(0)
        nm_coef = NelderMead(
            params.get_parameter_list(), coef={"r": 1.0, "ic": -0.5, "oc": 0.5, "e": 2.0, "s": 0.5}, rng=rng
        )
        self.nm = NelderMead(params.get_parameter_list(), rng=rng)
        yield
        self.nm = None

    def test_init(self):
        assert type(self.nm) is NelderMead

    def test__create_initial_values(self):
        params = load_parameter(self.config.hyperparameters.get())
        hps = params.get_parameter_list()
        initial_parameters = [
            {"parameter_name": "x1", "type": "FLOAT", "value": 1},
            {"parameter_name": "x2", "type": "FLOAT", "value": 1},
            {"parameter_name": "x3", "type": "FLOAT", "value": 1},
            {"parameter_name": "x4", "type": "FLOAT", "value": 1},
            {"parameter_name": "x5", "type": "FLOAT", "value": 1},
            {"parameter_name": "x6", "type": "FLOAT", "value": 1},
            {"parameter_name": "x7", "type": "FLOAT", "value": 1},
            {"parameter_name": "x8", "type": "FLOAT", "value": 1},
            {"parameter_name": "x9", "type": "FLOAT", "value": 1},
            {"parameter_name": "x10", "type": "FLOAT", "value": 1},
        ]
        rng = np.random.RandomState(0)
        NelderMead(hps, initial_parameters=initial_parameters, rng=rng)

        initial_parameters = [
            {"parameter_name": "x1", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x2", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x3", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x4", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x5", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x6", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x7", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x8", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x9", "type": "CATEGORICAL", "value": "1"},
            {"parameter_name": "x10", "type": "CATEGORICAL", "value": "1"},
        ]
        with pytest.raises(TypeError):
            NelderMead(hps, initial_parameters=initial_parameters, rng=rng)

    def test_add_executing(self):
        assert self.nm._add_executing(self.nm.y[0]) is None

        out_y = [[b[0] - 1.0, b[1] + 1.0] for b in self.nm.bdrys]
        assert self.nm._add_executing(out_y[0]) is None

        with patch.object(self.nm, "_is_out_of_boundary", return_value=True):
            with patch.object(self.nm, "_maximize", True):
                assert self.nm._add_executing(out_y[0]) is None

    def test_add_y_history(self):
        assert self.nm._add_y_history() is None

    def test_pop_result(self, work_dir, setup_hp_finished):
        assert self.nm._pop_result() is None

        # params = self.nm.get_ready_parameters()
        params = self.nm._executing

        for p, i in zip(params, range(1, len(self.nm.bdrys) + 2)):
            p["vertex_id"] = "{:03}".format(i)

        storage = Storage(work_dir)
        setup_hp_finished(1)
        for i in range(1):
            storage.result.set_any_trial_objective(trial_id=i, objective=0.0)
            for j in range(2):
                storage.hp.set_any_trial_param(trial_id=i, param_name=f"x{j+1}", param_value=0.0, param_type="float")
        storage.trial.set_any_trial_state(trial_id=1, state="finished")
        #
        # c = load_yaml(work_dir.joinpath(aiaccel.dict_hp_finished, '001.hp'))
        #
        print(storage.get_finished())
        print(storage.result.get_all_result())
        c = storage.get_hp_dict(trial_id=0)
        assert c is not None

        param = copy.copy(params[[p["vertex_id"] for p in params].index("001")])
        param["result"] = c["result"]
        self.nm.add_result_parameters(param)
        self.nm._maximize = True
        v = self.nm._pop_result()
        assert v["vertex_id"] == "001"

        param["vertex_id"] = "002"
        param["out_of_boundary"] = True
        self.nm.add_result_parameters(param)
        self.nm._pop_result()

        param["vertex_id"] = "003"
        param["out_of_boundary"] = True
        self.nm._state = "WaitShrink"
        self.nm.add_result_parameters(param)

        # for d in self.nm._executing:
        #     print(d['vertex_id'])

        try:
            self.nm._pop_result()
            assert False
        except ValueError:
            assert True

        param["vertex_id"] = "invalid"
        self.nm.add_result_parameters(param)
        try:
            self.nm._pop_result()
            assert False
        except ValueError:
            assert True

    def test_change_state(self):
        assert self.nm._change_state("WaitShrink") is None

        try:
            self.nm._change_state("InvalidState")
            assert False
        except ValueError:
            assert True

    def test_wait_initialize(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]

        assert self.nm._wait_initialize(results) is None

        with patch.object(self.nm, "_state", "WaitInitialize"):
            assert self.nm._wait_initialize(results) is None

        with patch.object(self.nm, "_state", "Wait"):
            assert self.nm._wait_initialize(results) is None

    def test_initialize(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        self.nm._wait_initialize(results)
        assert self.nm._initialize() is None

    def test_wait_reflect(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        assert self.nm._wait_reflect(results) is None
        assert self.nm._state == "ReflectBranch"

    def test_reflect_branch(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        self.nm._wait_initialize(results)
        self.nm.yc = 0.1
        self.nm._fr = 0.05
        assert self.nm._reflect_branch() is None

        self.nm._fr = -0.1
        assert self.nm._reflect_branch() is None

        self.nm.f[-1] = 0.2
        self.nm._fr = 0.15
        assert self.nm._reflect_branch() is None

        self.nm._fr = 0.3
        assert self.nm._reflect_branch() is None

        # not applicable
        self.nm.f[0] = 0.22
        self.nm.f[-1] = 0.3
        self.nm.f[-2] = 0.201
        self.nm._fr = 0.21
        assert self.nm._reflect_branch() is None

    def test_wait_expand(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        assert self.nm._wait_expand(results) is None
        assert self.nm._state == "ExpandBranch"

    def test_expand_branch(self):
        self.nm.f = [i * 0.1 for i in range(0, len(self.nm.y))]
        self.nm._fe = 0.1
        self.nm._fr = 0.2
        assert self.nm._expand_branch() is None

        self.nm._fe = 0.3
        self.nm._fr = 0.2
        assert self.nm._expand_branch() is None

    def test_wait_outside_contract(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        assert self.nm._wait_outside_contract(results) is None

    def test_outside_contract_branch(self):
        self.nm._foc = 0.1
        self.nm._fr = 0.2
        self.nm.f = [i * 0.1 for i in range(0, len(self.nm.y))]
        assert self.nm._outside_contract_branch() is None

        self.nm._foc = 0.2
        self.nm._fr = 0.1
        assert self.nm._outside_contract_branch() is None

    def test_wait_inside_contract(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1} for i in range(0, len(self.nm.y))]
        assert self.nm._wait_inside_contract(results) is None

    def test_inside_contract_branch(self):
        self.nm.f = [i * 0.1 for i in range(0, len(self.nm.y))]
        self.nm._fic = 0.1
        assert self.nm._inside_contract_branch() is None

        self.nm._fic = 0.3
        assert self.nm._inside_contract_branch() is None

    def test_wait_shrink(self):
        results = [{"state": "WaitInitialize", "result": i * 0.1, "index": i} for i in range(0, len(self.nm.y))]
        self.nm.f = [i * 0.1 for i in range(0, len(self.nm.y))]
        assert self.nm._wait_shrink(results) is None

        print(results)
        with patch.object(self.nm, "_state", "WaitInitialize"):
            assert self.nm._wait_shrink(results) is None

        with patch.object(self.nm, "_state", "InvalidState"):
            assert self.nm._wait_shrink(results) is None

    def test_finalize(self):
        self.nm.f = [i * 0.1 for i in range(0, len(self.nm.y))]
        assert self.nm._finalize() is None

        self.nm._out_of_boundary = True
        assert self.nm._finalize() is None

    def test_order_by(self):
        self.nm.f = np.array([i * 0.1 for i in range(0, len(self.nm.y))])
        assert self.nm._order_by() is None

    def test_centroid(self):
        self.nm.f = np.array([i * 0.1 for i in range(0, len(self.nm.y))])
        assert self.nm._centroid() is None

    def test_reflect(self):
        self.nm.yc = 0.1
        assert self.nm._reflect() is None

    def test_expand(self):
        self.nm.yc = 0.1
        assert self.nm._expand() is None

    def test_inside_contract(self):
        self.nm.yc = 0.1
        assert self.nm._inside_contract() is None

    def test_shrink(self):
        assert self.nm._shrink() is None

    def test_is_out_of_boundary(self):
        assert not self.nm._is_out_of_boundary(self.nm.y[0])
        assert self.nm._is_out_of_boundary([10.0, -10.0])

    def test_add_result_parameters(self):
        assert self.nm.add_result_parameters({}) is None

    def calc_and_add_results(self):
        # params = self.nm.get_ready_parameters()
        params = self.nm._executing

        for p in params:
            p["result"] = np.sum(np.array([pp["value"] ** 2 for pp in p["parameters"]]))
            self.nm.add_result_parameters(p)

    def test_search(self):
        self.nm.y = np.array([[0.9, 0.9], [-0.9, -0.3], [0.1, 0.5]])
        self.nm._executing = []
        for y in self.nm.y:
            self.nm._add_executing(y)

        for i in range(0, 50):
            self.calc_and_add_results()
            assert type(self.nm.search()) is list

        self.nm._state = "WaitShrink"
        self.calc_and_add_results()
        assert type(self.nm.search()) is list

        self.nm._state = "InvalidState"
        try:
            self.nm.search()
            assert False
        except ValueError:
            assert True

        self.nm._max_itr = 0
        assert self.nm.search() is None


def test_nelder_mead_parameters(load_test_config):
    debug = False
    config = load_test_config()
    params = load_parameter(
        # config.get('optimize', 'parameters')
        config.hyperparameters.get()
    )
    initial_parameters = None
    rng = np.random.RandomState(0)
    nelder_mead = NelderMead(
        params.get_parameter_list(),
        initial_parameters=initial_parameters,
        iteration=100,
        maximize=(config.goal.get().lower() == "maximize"),
        rng=rng,
    )

    c_max = 1000
    c = 0
    c_inside_of_boundary = 0
    c_out_of_boundary = 0

    if debug:
        print()

    while True:
        c += 1

        if debug:
            print(
                c,
                "NelderMead state:",
                nelder_mead._state,
                "executing:",
                nelder_mead._executing_index,
                "evaluated_itr:",
                nelder_mead._evaluated_itr,
            )

        # a functionality of NelderMeadOptimizer::check_result()
        # ready_params = nelder_mead.get_ready_parameters()
        ready_params = nelder_mead._executing

        for rp in ready_params:
            rp["result"] = sum([pp["value"] ** 2 for pp in rp["parameters"]])
            nelder_mead.add_result_parameters(rp)

            if debug:
                print("\tsum:", rp["result"])

        # a functionality of NelderMeadOptimizer::generate_parameter()
        searched_params = nelder_mead.search()

        if searched_params is None:
            if debug:
                print("Reached to max iteration.")
            break

        if len(searched_params) == 0:
            continue

        if debug:
            for sp in searched_params:
                print("\t", sp["name"], sp["state"], sp["itr"], sp["out_of_boundary"])

        for sp in searched_params:
            if sp["out_of_boundary"]:
                c_out_of_boundary += 1
            else:
                c_inside_of_boundary += 1

        if c >= c_max:
            break

    if debug:
        print()
        print("inside of boundary", c_inside_of_boundary)
        print("out of boundary", c_out_of_boundary)

    assert c_out_of_boundary == 0
