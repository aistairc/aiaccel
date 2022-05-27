from aiaccel.optimizer.grid.search import GridSearchOptimizer, generate_grid_points
from aiaccel.parameter import HyperParameter, load_parameter
from tests.base_test import BaseTest
import functools


# def test_generate_grid_points(load_test_config):
#     config = load_test_config()
# コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
def test_generate_grid_points(grid_load_test_config):
    config = grid_load_test_config()
    params = load_parameter(
        # config.get('hyperparameter', 'ConfigSpace_hyperparameter')
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        config.hyperparameters.get()
    )
    for p in params.get_parameter_list():
        # generate_grid_points(p, config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
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

    print(int_p)
    # assert generate_grid_points(int_p, config)
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
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
    # cat_grid = generate_grid_points(cat_p, config)
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
    cat_grid = generate_grid_points(cat_p, config)
    assert cat_grid['parameters'] == ['red', 'green', 'blue']

    class FakeParameter(object):

        def __init__(self, name, type_name):
            self.name = name
            self.type = type_name

    try:
        # generate_grid_points(FakeParameter('1', 'uniform_int'), config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        generate_grid_points(FakeParameter('1', 'uniform_int'), config)
        assert False
    except TypeError:
        assert True


class TestGridSearchOptimizer(BaseTest):

    def test_pre_process(self, clean_work_dir):
        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        optimizer = GridSearchOptimizer(options)
        optimizer.pre_process()

    def test_get_parameter_index(self, clean_work_dir):
        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        optimizer = GridSearchOptimizer(options)
        optimizer.pre_process()
        assert optimizer.get_parameter_index() == [0 for _ in range(0, 10)]

        max_index = functools.reduce(
            lambda x, y: x*y,
            [len(p['parameters']) for p in optimizer.ready_params]
        )
        optimizer.generate_index = max_index + 1
        assert optimizer.get_parameter_index() is None

    def test_generate_parameter(self, clean_work_dir):
        options = {
            'config': self.grid_config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'dbg': False,
            'graph': False,
            'process_name': 'optimizer'
        }
        optimizer = GridSearchOptimizer(options)
        optimizer.pre_process()
        max_index = functools.reduce(
            lambda x, y: x*y,
            [len(p['parameters']) for p in optimizer.ready_params]
        )
        optimizer.generate_index = max_index + 1
        assert optimizer.generate_parameter() is None

        optimizer.generate_index = 0
        assert optimizer.generate_parameter() is None
