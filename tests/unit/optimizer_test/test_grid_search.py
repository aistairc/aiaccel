import functools
from pathlib import Path

from aiaccel.config import load_config
from aiaccel.optimizer.grid_optimizer import (GridOptimizer,
                                              generate_grid_points,
                                              get_grid_options)
from aiaccel.parameter import HyperParameter, load_parameter
from tests.base_test import BaseTest
import pytest

def test_get_grid_options():
    test_data_dir = Path(__file__).resolve().parent.parent.parent.joinpath('test_data')
    grid_config_json = test_data_dir.joinpath('grid_config.json')
    config_grid = Config(grid_config_json)

    base, log, step = get_grid_options('x1', config_grid)
    assert base == 10
    assert log
    assert step == 0.1
    try:
        _, _, _ = get_grid_options('invalid', config_grid)
        assert False
    except KeyError:
        assert True

    # no step
    grid_config_json = test_data_dir / 'config_grid_no_step.json'
    config_grid = load_config(grid_config_json)
    with pytest.raises(KeyError):
        base, log, step = get_grid_options('x1', config_grid)

    # no log
    grid_config_json = test_data_dir / 'config_grid_no_log.json'
    config_grid = load_config(grid_config_json)
    with pytest.raises(KeyError):
        base, log, step = get_grid_options('x1', config_grid)

    # no base
    grid_config_json = test_data_dir / 'config_grid_no_base.json'
    config_grid = load_config(grid_config_json)
    with pytest.raises(KeyError):
        base, log, step = get_grid_options('x1', config_grid)

    # base true/false
    grid_config_json = test_data_dir / 'config_grid_base.json'
    config_grid = load_config(grid_config_json)
    with pytest.raises(KeyError):
        base, log, step = get_grid_options('x1', config_grid)

def test_generate_grid_points(grid_load_test_config):
    config = grid_load_test_config()
    params = load_parameter(config.hyperparameters.get())
    for p in params.get_parameter_list():
        generate_grid_points(p, config)

    int_p = HyperParameter(
        {
            'name': 'x4',
            'type': 'uniform_int',
            'lower': 1,
            'upper': 10,
            'log': False,
            'step': 1,
            'base': 10
        }
    )

    assert generate_grid_points(int_p, config)

    cat_p = HyperParameter(
        {
            'name': 'x3',
            'type': 'categorical',
            'choices': ['red', 'green', 'blue'],
            'log': False,
            'step': 1,
            'base': 10
        }
    )
    cat_grid = generate_grid_points(cat_p, config)
    assert cat_grid['parameters'] == ['red', 'green', 'blue']

    class FakeParameter(object):

        def __init__(self, name, type_name):
            self.name = name
            self.type = type_name

    try:
        generate_grid_points(FakeParameter('1', 'uniform_int'), config)
        assert False
    except TypeError:
        assert True

    cat_p = HyperParameter(
        {
            'name': 'x3',
            'type': 'ordinal',
            'choices': ['red', 'green', 'blue'],
            'log': False,
            'step': 1,
            'base': 10,
            "sequence": [1, 2, 3]
        }
    )
    cat_grid = generate_grid_points(cat_p, config)
    print(cat_grid)

class TestGridOptimizer(BaseTest):

    def test_pre_process(self, clean_work_dir):
        self.workspace.clean()
        self.workspace.create()

        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = GridOptimizer(options)
        optimizer.pre_process()

    def test_get_parameter_index(self, clean_work_dir):
        self.workspace.clean()
        self.workspace.create()

        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = GridOptimizer(options)
        optimizer.pre_process()
        assert optimizer.get_parameter_index() == [0 for _ in range(0, 10)]

        max_index = functools.reduce(
            lambda x, y: x*y,
            [len(p['parameters']) for p in optimizer.ready_params]
        )
        optimizer.generate_index = max_index + 1
        assert optimizer.get_parameter_index() is None

    def test_generate_parameter(self, clean_work_dir):
        self.workspace.clean()
        self.workspace.create()

        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = GridOptimizer(options)
        optimizer.pre_process()
        max_index = functools.reduce(
            lambda x, y: x*y,
            [len(p['parameters']) for p in optimizer.ready_params]
        )

        # All generated
        optimizer.generate_index = max_index + 1
        assert len(optimizer.generate_parameter()) == 0

        optimizer.generate_index = 0
        assert len(optimizer.generate_parameter()) == self.config.trial_number.get()


    def test_generate_initial_parameter(self):
        self.workspace.clean()
        self.workspace.create()

        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        optimizer = GridOptimizer(options)
        optimizer.pre_process()
        optimizer.generate_initial_parameter()
