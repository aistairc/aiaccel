from aiaccel.storage.storage import Storage
from tests.unit.storage.fs.base import t_base
from tests.unit.storage.fs.base import ws
import random
import numpy as np
from tests.unit.storage.fs.base import config_path


# set_any_trial_serialize
@t_base()
def test_set_any_trial_serialize():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    process_name = "test"
    optimization_variable = {'loop_count': 1}
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    assert storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    ) is None

    # ============================

    trial_id = 65536
    process_name = "test2"
    optimization_variable = {'loop_count': 1}
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    assert storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    ) is None

    # ============================

    trial_id = 0
    process_name = "test"
    optimization_variable = {'loop_count': 1}
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    assert storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    ) is None


# get_any_trial_serialize
@t_base()
def test_get_any_trial_serialize():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    assert ws.path.exists() is True
    print(ws.path)

    trial_id = 0
    process_name = "test"
    optimization_variable = {'loop_count': 1}

    storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    )

    serial = storage.serializer.get_any_trial_serialize(
        trial_id=trial_id,
        process_name=process_name
    )

    assert serial[0] == optimization_variable

    native_random_value = random.random()
    random.setstate(serial[1])
    assert random.random() == native_random_value

    numpy_random_value = np.random.random()
    np.random.set_state(serial[2])
    assert np.random.random() == numpy_random_value


    # ============================

    trial_id = 2
    process_name = "test"
    assert storage.serializer.get_any_trial_serialize(
        trial_id=trial_id,
        process_name=process_name
    ) is None


# delete_any_trial_serialize
@t_base()
def test_delete_any_trial_serialize():
    storage = Storage(ws.path)

    trial_id = 0
    process_name = "test0"
    optimization_variable = 0
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    )

    trial_id = 1
    process_name = "test1"
    optimization_variable = 1
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    )

    trial_id = 2
    process_name = "test2"
    optimization_variable = 2
    native_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    storage.serializer.set_any_trial_serialize(
        trial_id=trial_id,
        optimization_variable=optimization_variable,
        process_name=process_name,
        native_random_state=native_random_state,
        numpy_random_state=numpy_random_state
    )

    assert storage.serializer.get_any_trial_serialize(trial_id=0,process_name="test0") is not None
    assert storage.serializer.get_any_trial_serialize(trial_id=1,process_name="test1") is not None
    assert storage.serializer.get_any_trial_serialize(trial_id=2,process_name="test2") is not None

    assert storage.serializer.delete_any_trial_serialize(trial_id=0) is None
    assert storage.serializer.get_any_trial_serialize(trial_id=0,process_name="test0") is None
    assert storage.serializer.get_any_trial_serialize(trial_id=1,process_name="test1") is not None
    assert storage.serializer.get_any_trial_serialize(trial_id=2,process_name="test2") is not None

    assert storage.serializer.delete_any_trial_serialize(trial_id=1) is None
    assert storage.serializer.get_any_trial_serialize(trial_id=0,process_name="test0") is None
    assert storage.serializer.get_any_trial_serialize(trial_id=1,process_name="test1") is None
    assert storage.serializer.get_any_trial_serialize(trial_id=2,process_name="test2") is not None

    assert storage.serializer.delete_any_trial_serialize(trial_id=2) is None
    assert storage.serializer.get_any_trial_serialize(trial_id=0,process_name="test0") is None
    assert storage.serializer.get_any_trial_serialize(trial_id=1,process_name="test1") is None
    assert storage.serializer.get_any_trial_serialize(trial_id=2,process_name="test2") is None
