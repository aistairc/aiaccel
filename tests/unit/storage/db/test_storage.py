from aiaccel.storage.storage import Storage
from base import t_base, ws
from unittest.mock import patch

# set_any_trial_start_time
@t_base()
def test_current_max_trial_number():
    storage = Storage(ws.path)

    assert storage.current_max_trial_number() is None

    states = ["test1", "test2", "test3"]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
        assert storage.current_max_trial_number() == i


# get_ready
@t_base()
def test_get_ready():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_ready() == [0, 1]


# get_running
@t_base()
def test_get_running():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_running() == [2, 3]


# get_finished
@t_base()
def test_get_finished():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_finished() == [4, 5]


# get_num_ready
@t_base()
def test_get_num_ready():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_num_ready() == 2


# get_num_running
@t_base()
def test_get_num_running():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_num_running() == 3


# get_num_finished
@t_base()
def test_get_num_finished():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_num_finished() == 4


# is_ready
@t_base()
def test_is_ready():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    for i in range(len(states)):
        if states[i] == "ready":
            assert storage.is_ready(i) is True
        else:
            assert storage.is_ready(i) is False


# is_running
@t_base()
def test_is_running():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    for i in range(len(states)):
        if states[i] == "running":
            assert storage.is_running(i) is True
        else:
            assert storage.is_running(i) is False


# is_finished
@t_base()
def test_is_finished():
    storage = Storage(ws.path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "running",
        "finished",
        "finished",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    for i in range(len(states)):
        if states[i] == "finished":
            assert storage.is_finished(i) is True
        else:
            assert storage.is_finished(i) is False  


# get_hp_dict
@t_base()
def test_get_hp_dict():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = 0.1
    param_type = "float"
    error = "aaaa"

    assert storage.get_hp_dict(trial_id) is None

    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    storage.hp.set_any_trial_param(
        trial_id=trial_id, param_name=param_name,
        param_value=param_value, param_type=param_type
    )

    storage.error.set_any_trial_error(trial_id=trial_id,error_message=error)

    exp = {
        'trial_id': trial_id,
        'parameters': [{
            "parameter_name": param_name,
            "type": param_type,
            "value": param_value
        }],
        'result': objective,
        'start_time': start_time,
        'end_time': end_time,
        "error": error
    }

    d = storage.get_hp_dict(trial_id)

    for key in d.keys():
        assert exp[key] == d[key]


# get_hp_dict
@t_base()
def test_get_hp_dict_int():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = 42
    param_type = "int"
    error = "aaaa"

    assert storage.get_hp_dict(trial_id) is None

    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    storage.hp.set_any_trial_param(
        trial_id=trial_id, param_name=param_name,
        param_value=param_value, param_type=param_type
    )

    storage.error.set_any_trial_error(trial_id=trial_id,error_message=error)

    exp = {
        'trial_id': trial_id,
        'parameters': [{
            "parameter_name": param_name,
            "type": param_type,
            "value": param_value
        }],
        'result': objective,
        'start_time': start_time,
        'end_time': end_time,
        "error": error
    }

    d = storage.get_hp_dict(trial_id)

    for key in d.keys():
        assert exp[key] == d[key]


@t_base()
def test_get_hp_dict_categorical():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = "red"
    param_type = "categorical"
    error = ""

    assert storage.get_hp_dict(trial_id) is None

    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    storage.hp.set_any_trial_param(
        trial_id=trial_id, param_name=param_name,
        param_value=param_value, param_type=param_type
    )

    storage.error.set_any_trial_error(trial_id=trial_id,error_message=error)

    exp = {
        'trial_id': trial_id,
        'parameters': [{
            "parameter_name": param_name,
            "type": param_type,
            "value": param_value
        }],
        'result': objective,
        'start_time': start_time,
        'end_time': end_time,
    }

    d = storage.get_hp_dict(trial_id)

    for key in d.keys():
        assert exp[key] == d[key]


@t_base()
def test_get_hp_dict_invalid_type():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = "red"
    param_type = "invalid"
    error = ""

    assert storage.get_hp_dict(trial_id) is None

    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    storage.hp.set_any_trial_param(
        trial_id=trial_id, param_name=param_name,
        param_value=param_value, param_type=param_type
    )

    storage.error.set_any_trial_error(trial_id=trial_id,error_message=error)

    exp = {
        'trial_id': trial_id,
        'parameters': [{
            "parameter_name": param_name,
            "type": param_type,
            "value": param_value
        }],
        'result': objective,
        'start_time': start_time,
        'end_time': end_time,
    }

    d = storage.get_hp_dict(trial_id)

    for key in d.keys():
        assert exp[key] == d[key]




# get_result_and_error
@t_base()
def test_get_result_and_error():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    error = "aaaa"

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=error
    )

    assert storage.get_result_and_error(trial_id=trial_id) == (objective, error)


# get_best_trial_dict
@t_base()
def test_get_best_trial_dict():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = 0.1
    param_type = "float"
    error = "aaaa"
    goal = "minimize"

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    storage.timestamp.set_any_trial_start_time(
        trial_id=trial_id,
        start_time=start_time
    )

    storage.timestamp.set_any_trial_end_time(
        trial_id=trial_id,
        end_time=end_time
    )

    storage.hp.set_any_trial_param(
        trial_id=trial_id,
        param_name=param_name,
        param_value=param_value,
        param_type=param_type
    )

    storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=error
    )

    exp = {
        'trial_id': trial_id,
        'parameters': [
            {
                "parameter_name": param_name,
                "type": param_type,
                "value": param_value
                
            }
        ],
        'result': objective,
        'start_time': start_time,
        'end_time': end_time,
        "error": error
    }

    d = storage.get_best_trial_dict(goal)

    for key in d.keys():
        assert exp[key] == d[key]


# get_best_trial
@t_base()
def test_get_best_trial():
    storage = Storage(ws.path)
    trial_ids = [0, 1, 2, 3, 4]
    objectives = [0.00, 0.01, -1, 1, 0.03]

    for i in range(len(trial_ids)):
        storage.result.set_any_trial_objective(
            trial_id=trial_ids[i],
            objective=objectives[i]
        )

    goal = "minimize"
    assert storage.get_best_trial(goal) == (2, -1)

    goal = "maximize"
    assert storage.get_best_trial(goal) == (3, 1)

    goal = "aaaaaaaaaa"
    assert storage.get_best_trial(goal) == (None, None)


# delete_trial_data_after_this
@t_base()
def test_delete_trial_data_after_this():
    storage = Storage(ws.path)
    assert storage.delete_trial_data_after_this(trial_id=1) is None

    def dummy_delete_trial(trial_id :int) -> None:
        pass

    with patch.object(storage, 'current_max_trial_number', return_value=10):
        with patch.object(storage, 'delete_trial', dummy_delete_trial):
            assert storage.delete_trial_data_after_this(trial_id=1) is None


# delete_trial
@t_base()
def test_delete_trial():
    storage = Storage(ws.path)
    assert storage.delete_trial(trial_id=1) is None


# rollback_to_ready
@t_base()
def test_rollback_to_ready():
    storage = Storage(ws.path)

    assert storage.rollback_to_ready(trial_id=1) is None

    with patch.object(storage.hp, 'get_any_trial_params', return_value=object):
        assert storage.rollback_to_ready(trial_id=1) is None
