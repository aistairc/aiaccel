# from aiaccel.util.filesystem import get_dict_files
# from aiaccel.util.serialize import deserialize_state
# from aiaccel.util.serialize import deserialize_native_random
# from aiaccel.util.serialize import deserialize_numpy_random
# from aiaccel.util.serialize import serialize_state
from aiaccel.util.serialize import Serializer
from tests.base_test import BaseTest
import numpy as np
import random
from aiaccel.storage.storage import Storage


class TestRandom(BaseTest):
    """
    ToDo: Refactor this!
    """

    def test_random_optimizer(self, clean_work_dir, work_dir):
        random.seed(1)
        np.random.seed(1)
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'optimizer'
        }
        serialize = Serializer(self.config, 'optimizer',options)
        serialize.storage.alive.init_alive()
        serialize_datas = {
            'num_of_generated_parameter': 42,
            'loop_count': 52
        }
        serialize.serialize(
            trial_id=1,
            optimization_variables=serialize_datas,
            native_random_state=random.getstate(),
            numpy_random_state=np.random.get_state()
        )
        # serialize_state(self.dict_state, 'Optimizer', 1, self.dict_lock)
        native_random_value = random.random()
        numpy_random_value = np.random.random()

        d = serialize.deserialize(trial_id=1)
        loop_count = d['optimization_variables']['loop_count']
        num_of_generated_parameter = d['optimization_variables']['num_of_generated_parameter']
        random.setstate(d['native_random_state'])
        np.random.set_state(d['numpy_random_state'])

        # loop_count = deserialize_state(
        #     self.dict_state, 'Optimizer', self.dict_lock)
        assert num_of_generated_parameter == 42
        assert loop_count == 52
        assert random.random() == native_random_value
        assert np.random.random() == numpy_random_value

    def test_random_scheduler(self, clean_work_dir, work_dir):
        # serialize_state(self.dict_state, 'Scheduler', 1, self.dict_lock)
        serialize_datas = {
            'num_of_generated_parameter': 42,
            'loop_count': 1
        }
        print(clean_work_dir)
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'scheduler'
        }
        serialize = Serializer(self.config, 'scheduler', options)
        serialize.storage.alive.init_alive()

        serialize.serialize(
            trial_id=1,
            optimization_variables=serialize_datas,
            native_random_state=random.getstate(),
            numpy_random_state=np.random.get_state()
        )

        native_random_value = random.random()
        numpy_random_value = np.random.random()
        # loop_count = deserialize_state(
        #     self.dict_state,
        #     'Scheduler',
        #     self.dict_lock
        # )

        d = serialize.deserialize(trial_id=1)
        loop_count = d['optimization_variables']['loop_count']
        random.setstate(d['native_random_state'])
        np.random.set_state(d['numpy_random_state'])

        assert loop_count == 1
        assert random.random() == native_random_value
        assert np.random.random() == numpy_random_value

        # try:
        #     #deserialize_native_random('invalid_file')
        #     deserialize_native_random(Path('invalid_file'))
        #     assert False
        # except FileNotFoundError:
        #     assert True

        # try:
        #     deserialize_numpy_random(Path('invalid_file'))
        #     assert False
        # except FileNotFoundError:
        #     assert True
