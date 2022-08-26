from aiaccel.parameter import get_best_parameter
from aiaccel.parameter import get_grid_options
from aiaccel.parameter import get_type
from aiaccel.parameter import load_parameter
from aiaccel.util.filesystem import create_yaml
from tests.base_test import BaseTest
import aiaccel


class TestParameter(BaseTest):

    def test_get_best_parameters(
        self,
        clean_work_dir,
        get_one_parameter,
        work_dir
    ):

        files = list(work_dir.joinpath(aiaccel.dict_result).glob('*.yml'))
        best, best_file = get_best_parameter(
            files,
            aiaccel.goal_maximize,
            work_dir.joinpath(aiaccel.dict_lock)
        )
        assert best is None
        assert best_file is None

        results = [120, 101., 140.]
        # for i in range(0, 3):
        #     name = i
        #     path = work_dir.joinpath(aiaccel.dict_result, '{}.yml'.format(name))
        #     print(path)
        #     content = get_one_parameter()
        #     content['result'] = results[i]
        #     create_yaml(path, content, work_dir.joinpath(aiaccel.dict_lock))

        for i in range(len(self.test_result_data)):
            d = self.test_result_data[i]
            name = f"{d['trial_id']}.yml"
            path = work_dir / 'result' / name
            d['result'] = results[i]
            create_yaml(path, d)

        files = list(work_dir.joinpath(aiaccel.dict_result).glob('*.yml'))
        files.sort()
        best, best_file = get_best_parameter(
            files,
            aiaccel.goal_maximize,
            work_dir.joinpath(aiaccel.dict_lock)
        )
        assert best == 140.
        best, best_file = get_best_parameter(
            files,
            aiaccel.goal_minimize,
            work_dir.joinpath(aiaccel.dict_lock)
        )
        assert best == 101.
        try:
            _, _ = get_best_parameter(
                files,
                'invalid_goal',
                work_dir.joinpath(aiaccel.dict_lock)
            )
            assert False
        except ValueError:
            assert True

    def test_get_grid_options(self):
        # base, log, step = get_grid_options('x1', self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        base, log, step = get_grid_options('x1', self.config_grid)
        assert base == 10
        assert log
        assert step == 0.1
        try:
            # _, _, _ = get_grid_options('invalid', self.config)
            # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
            _, _, _ = get_grid_options('invalid', self.config_grid)
            assert False
        except KeyError:
            assert True

    def test_get_type(self):
        int_p = {'type': 'uniform_int'}
        assert get_type(int_p) == 'INT'
        float_p = {'type': 'uniform_float'}
        assert get_type(float_p) == 'FLOAT'
        cat_p = {'type': 'categorical'}
        assert get_type(cat_p) == 'CATEGORICAL'
        ord_p = {'type': 'ordinal'}
        assert get_type(ord_p) == 'ORDINAL'
        other_p = {'type': 'invalid'}
        assert get_type(other_p) == 'invalid'

    def test_load_parameter(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()
        )
        assert hp.__class__.__name__ == 'HyperParameterConfiguration'

    def test_get_parameter_list(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()

        )
        p = hp.get_parameter_list()
        assert len(p) == 10

    def test_get_hyperparameter(self):
        hp = load_parameter(
            # self.config.get('hyperparameter', 'ConfigSpace_hyperparameter'))
            # self.config.get('optimize', 'parameters')
            self.config.hyperparameters.get()
        )
        ps = hp.get_hyperparameter('x3')
        assert ps.name == 'x3'

    def test_sample(self):
        # json_string = {
        #     'parameters': [
        #         {'name': 'a', 'type': 'uniform_int', 'lower': 0, 'upper': 10},
        #         {'name': 'b', 'type': 'uniform_float', 'lower': 0.,
        #          'upper': 10.},
        #         {'name': 'c', 'type': 'categorical',
        #          'choices': ['red', 'green', 'blue']},
        #         {'name': 'd', 'type': 'ordinal',
        #          'sequence': ['10', '20', '30']}
        #     ]
        # }
        json_string = [
            {
                'name': 'a',
                'type': 'uniform_int',
                'lower': 0,
                'upper': 10
            },
            {
                'name': 'b',
                'type': 'uniform_float',
                'lower': 0.,
                'upper': 10.
            },
            {
                'name': 'c',
                'type': 'categorical',
                'choices': ['red', 'green', 'blue']
            },
            {
                'name': 'd',
                'type': 'ordinal',
                'sequence': ['10', '20', '30']
            }
        ]
        hp = load_parameter(json_string)
        p = hp.sample()
        assert len(p) == 4

        # json_string['parameters'].append({'name': 'e', 'type': 'invalid'})
        json_string.append({'name': 'e', 'type': 'invalid'})
        hp = load_parameter(json_string)

        try:
            hp.sample()
            assert False
        except TypeError:
            assert True
