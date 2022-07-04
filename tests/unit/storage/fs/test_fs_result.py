from aiaccel.storage.storage import Storage
from base import t_base
from base import db_path
from base import ws
from base import config_path


# set_any_trial_objective
@t_base()
def test_set_any_trial_objective():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    objective = 0.01

    assert storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    ) is None


# get_any_trial_objective
@t_base()
def test_get_any_trial_objective():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    trial_id = 0
    objective = 0.01

    storage.result.set_any_trial_objective(
        trial_id=trial_id,
        objective=objective
    )

    o = storage.result.get_any_trial_objective(trial_id)
    assert objective == o


# get_all_result
@t_base()
def test_get_all_result():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(
            trial_id=ids[i],
            objective=objectives[i]
        )
    
    data = storage.result.get_all_result()
    assert [d.objective for d in data] == objectives


# get_objectives
@t_base()
def test_get_objectives():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(
            trial_id=ids[i],
            objective=objectives[i]
        )
    
    data = storage.result.get_objectives()
    assert objectives == data


# get_bests
@t_base()
def test_get_bests():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    objectives = [1, -5, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(
            trial_id=ids[i],
            objective=objectives[i]
        )
    
    assert storage.result.get_bests('minimize') == [1, -5, -5, -5]
    assert storage.result.get_bests('maximize') == [1, 1, 3, 3]


# get_result_trial_id_list
@t_base()
def test_get_result_trial_id_list():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(
            trial_id=ids[i],
            objective=objectives[i]
        )
    
    assert storage.result.get_result_trial_id_list() == list(ids)


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(
            trial_id=ids[i],
            objective=objectives[i]
        )
    
    assert storage.result.get_any_trial_objective(0) == 1
    assert storage.result.all_delete() is None
    assert storage.result.get_any_trial_objective(0) is None
