from aiaccel.util.filesystem import get_dict_files
from aiaccel.util.serialize import deserialize_state, deserialize_native_random,\
    deserialize_numpy_random, resub_loop_count, serialize_state
from pathlib import Path
from tests.base_test import BaseTest
import aiaccel
import numpy as np
import random


def test_resub_loop_count(data_dir):
    native_pattern = '{}_{}_*.{}'.format(
        aiaccel.module_type_optimizer, aiaccel.file_native_random, aiaccel.extension_pickle
    )
    files = sorted(get_dict_files(data_dir.joinpath('work/state'),
                                  native_pattern))
    assert resub_loop_count(files[-1]) == 14


class TestRandom(BaseTest):
    """
    ToDo: Refactor this!
    """

    def test_random_optimizer(self, clean_work_dir, work_dir):
        random.seed(1)
        np.random.seed(1)
        serialize_state(self.dict_state, 'Optimizer', 1, self.dict_lock)
        native_random_value = random.random()
        numpy_random_value = np.random.random()
        loop_count = deserialize_state(
            self.dict_state, 'Optimizer', self.dict_lock)
        assert loop_count == 1
        assert random.random() == native_random_value
        assert np.random.random() == numpy_random_value

    def test_random_scheduler(self, clean_work_dir, work_dir):

        serialize_state(self.dict_state, 'Scheduler', 1, self.dict_lock)
        native_random_value = random.random()
        numpy_random_value = np.random.random()
        loop_count = deserialize_state(
            self.dict_state, 'Scheduler', self.dict_lock)
        assert loop_count == 1
        assert random.random() == native_random_value
        assert np.random.random() == numpy_random_value

        try:
            #deserialize_native_random('invalid_file')
            deserialize_native_random(Path('invalid_file'))
            assert False
        except FileNotFoundError:
            assert True

        try:
            deserialize_numpy_random(Path('invalid_file'))
            assert False
        except FileNotFoundError:
            assert True
