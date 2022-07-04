from aiaccel.storage.storage import Storage
from base import t_base
from base import ws
from base import config_path


# set_any_trial_start_time
@t_base()
def test_current_max_trial_number():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    states = ["ready", "ready", "ready"]

    for i in range(len(states)):
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )
        assert storage.current_max_trial_number() == i


# get_ready
@t_base()
def test_get_ready():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    # states = [
    #     "ready",
    #     "ready",
    #     "running",
    #     "running",
    #     "finished",
    #     "finished"
    # ]

    states = [
        "ready",
        "ready"
    ]

    for i in range(len(states)):
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_ready() == [0, 1]


# get_running
@t_base()
def test_get_running():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_running() == [2, 3]


# get_finished
@t_base()
def test_get_finished():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    states = [
        "ready",
        "ready",
        "running",
        "running",
        "finished",
        "finished"
    ]

    for i in range(len(states)):
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_finished() == [4, 5]


# get_num_ready
@t_base()
def test_get_num_ready():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_num_ready() == 2


# get_num_running
@t_base()
def test_get_num_running():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    
    assert storage.get_num_running() == 3


# get_num_finished
@t_base()
def test_get_num_finished():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )
    assert storage.get_num_finished() == 4


# is_ready
@t_base()
def test_is_ready():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        print(states[i])
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )

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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        param = {
            'trial_id': i,
            'parameter_name': f'x{i}',
            'value': (i * 0.1),
            'type': float
        }
        storage.hp.set_any_trial_params(
            trial_id=i,
            params=param
        )
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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    objective = 0.01
    start_time = "00/00/00:00:00"
    end_time = "11/11/11:11:11"
    param_name = "x1"
    param_value = 0.1
    param_type = "float"
    error = "aaaa"

    assert storage.get_hp_dict(str(trial_id)) is None

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
        'trial_id': str(trial_id),
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

    d = storage.get_hp_dict(str(trial_id))

    for key in d.keys():
        assert exp[key] == d[key]


# get_result_and_error
@t_base()
def test_get_result_and_error():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
        'trial_id': str(trial_id),
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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
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
