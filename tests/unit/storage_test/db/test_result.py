from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy.exc import SQLAlchemyError
from undecorated import undecorated

from aiaccel.storage import Storage
from tests.unit.storage_test.db.base import init, t_base, ws


# set_any_trial_objective
@t_base()
def test_set_any_trial_objective():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01

    assert storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective) is None

    # update
    objective = 0.02
    assert storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective) is None


# set_any_trial_objective exception


@t_base()
def test_set_any_trial_objective_exception():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_objective = undecorated(storage.result.set_any_trial_objective)
        set_any_trial_objective(storage.result, trial_id=trial_id, objective=objective)


@t_base()
def test_get_any_trial_objective_and_best_value():
    storage = Storage(ws.path)

    # Test when objectives is None
    trial_id = 1
    goals = ["minimize"]
    mock_get_any_trial_objective = MagicMock(return_value=None)
    with patch.object(storage.result, "get_any_trial_objective", mock_get_any_trial_objective):
        objectives, best_values = storage.result.get_any_trial_objective_and_best_value(trial_id, goals)
        assert objectives is None
        assert best_values is None

    # Test when objectives is not None
    trial_id = 2
    goals = ["minimize", "maximize"]
    mock_get_any_trial_objective = MagicMock(return_value=[0.9, 0.8])
    mock_get_bests = MagicMock(return_value=[0.95, 0.85])
    with patch.object(storage.result, "get_any_trial_objective", mock_get_any_trial_objective):
        with patch.object(storage.result, "get_bests", mock_get_bests):
            objectives, best_values = storage.result.get_any_trial_objective_and_best_value(trial_id, goals)
            assert objectives == [0.9, 0.8]
            assert best_values == [0.95, 0.85]


# get_any_trial_objective
@t_base()
def test_get_any_trial_objective():
    storage = Storage(ws.path)

    trial_id = 0
    objective = 0.01

    storage.result.set_any_trial_objective(trial_id=trial_id, objective=objective)

    o = storage.result.get_any_trial_objective(trial_id)
    assert objective == o


# get_all_result
@t_base()
def test_get_all_result():
    storage = Storage(ws.path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    data = storage.result.get_all_result()
    assert [data[trial_id] for trial_id in data.keys()] == objectives


# get_objectives
@t_base()
def test_get_objectives():
    storage = Storage(ws.path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    data = storage.result.get_objectives()
    assert objectives == data


@t_base()
def test_get_bests():
    storage = Storage(ws.path)
    objectives = [[1], [-5], [3], [1.23]]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    with pytest.raises(ValueError):
        storage.result.get_bests(["invalid"])

    bests = storage.result.get_bests(["minimize"])
    for i in range(len(bests)):
        assert bests[i] == np.min(objectives)

    bests = storage.result.get_bests(["maximize"])
    for i in range(len(bests)):
        assert bests[i] == np.max(objectives)


# get_result_trial_id_list
@t_base()
def test_get_result_trial_id_list():
    storage = Storage(ws.path)

    assert storage.result.get_result_trial_id_list() is None

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    assert storage.result.get_result_trial_id_list() == list(ids)


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    assert storage.result.get_any_trial_objective(0) == 1
    assert storage.result.all_delete() is None
    assert storage.result.get_any_trial_objective(0) is None


# all_delete exception
@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)

    objectives = [1, 2, 3, 1.23]
    ids = range(len(objectives))

    for i in range(len(objectives)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.result.all_delete)
        all_delete(storage.result)


# delete_any_trial_objective
@t_base()
def test_delete_any_trial_objective():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    objectives = [0.01, 0.02, 0.03]

    for i in range(len(ids)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    assert storage.result.get_any_trial_objective(ids[0]) is not None
    assert storage.result.get_any_trial_objective(ids[1]) is not None
    assert storage.result.get_any_trial_objective(ids[2]) is not None

    assert storage.result.delete_any_trial_objective(ids[0]) is None
    assert storage.result.get_any_trial_objective(ids[0]) is None
    assert storage.result.get_any_trial_objective(ids[1]) is not None
    assert storage.result.get_any_trial_objective(ids[2]) is not None

    assert storage.result.delete_any_trial_objective(ids[1]) is None
    assert storage.result.get_any_trial_objective(ids[0]) is None
    assert storage.result.get_any_trial_objective(ids[1]) is None
    assert storage.result.get_any_trial_objective(ids[2]) is not None

    assert storage.result.delete_any_trial_objective(ids[2]) is None
    assert storage.result.get_any_trial_objective(ids[0]) is None
    assert storage.result.get_any_trial_objective(ids[1]) is None
    assert storage.result.get_any_trial_objective(ids[2]) is None


# delete_any_trial_objective exception
@t_base()
def test_delete_any_trial_objective_exception():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    objectives = [0.01, 0.02, 0.03]

    for i in range(len(ids)):
        storage.result.set_any_trial_objective(trial_id=ids[i], objective=objectives[i])

    init()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_objective = undecorated(storage.result.delete_any_trial_objective)
        delete_any_trial_objective(storage.result, trial_id=0)
