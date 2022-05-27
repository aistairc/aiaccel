from aiaccel.common import get_module_type_from_class_name, get_file_random
import aiaccel


def test_get_module_type_from_class_name():
    assert get_module_type_from_class_name('Master') == 'master'
    assert get_module_type_from_class_name('Optimizer') == 'optimizer'
    assert get_module_type_from_class_name('Scheduler') == 'scheduler'

    try:
        get_module_type_from_class_name('Invalid')
        assert False
    except TypeError:
        assert True


def test_get_file_random(data_dir):
    try:
        get_file_random('..', 'Optimizer', 1, 'invalid_type')
        assert False
    except TypeError:
        assert True

    state_dir = data_dir.joinpath('work', aiaccel.dict_state)
    assert get_file_random(state_dir, 'Optimizer', 1, 'native_random').exists()
    assert get_file_random(state_dir, 'Optimizer', 1, 'numpy_random').exists()
    assert get_file_random(state_dir, 'Optimizer', 1, 'configspace').exists()
