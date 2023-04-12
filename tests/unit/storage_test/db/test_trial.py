import pytest
from sqlalchemy.exc import SQLAlchemyError
from undecorated import undecorated

from aiaccel.storage import Storage
from tests.unit.storage_test.db.base import get_storage, init, t_base, ws


# set_any_trial_state
@t_base()
def test_set_any_trial_state():
    storage = get_storage()

    states = ["ready", "running", "finished"]

    for i in range(len(states)):
        assert storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        ) is None

    # update
    for i in range(len(states)):
        assert storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        ) is None


# set_any_trial_state exception
@t_base()
def test_set_any_trial_state_exception():
    storage = get_storage()

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_state = undecorated(storage.trial.set_any_trial_state)
        set_any_trial_state(
            storage.trial,
            trial_id=0,
            state="ready"
        )

# test_get_any_trial_state


@t_base()
def test_get_any_trial_state():
    storage = get_storage()

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
        assert storage.trial.get_any_trial_state(i) == states[i]


# get_any_state_list
@t_base()
def test_get_any_state_list():
    storage = get_storage()

    assert storage.trial.get_any_state_list("ready") is None

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

    assert storage.trial.get_any_state_list("ready") == [0, 1]
    assert storage.trial.get_any_state_list("running") == [2, 3, 4]
    assert storage.trial.get_any_state_list("finished") == [5, 6, 7, 8]


# all_delete
@t_base()
def test_all_delete():
    storage = get_storage()

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
        assert storage.trial.get_any_trial_state(i) == states[i]

    assert storage.trial.all_delete() is None

    for i in range(len(states)):
        assert storage.trial.get_any_trial_state(i) is None


# all_delete exception
@t_base()
def test_all_delete_exception():
    storage = get_storage()

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
        assert storage.trial.get_any_trial_state(i) == states[i]

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.trial.all_delete)
        all_delete(storage.trial)


# get_ready
@t_base()
def test_get_ready():
    storage = get_storage()

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

    assert storage.trial.get_ready() == [0, 1]


# get_running
@t_base()
def test_get_running():
    storage = get_storage()

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

    assert storage.trial.get_running() == [2, 3, 4]


# get_finished
@t_base()
def test_get_finished():
    storage = get_storage()

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

    assert storage.trial.get_finished() == [5, 6, 7, 8]


# get_all_trial_id
@t_base()
def test_get_all_trial_id():
    storage = get_storage()

    assert storage.trial.get_all_trial_id() is None

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

    assert storage.trial.get_all_trial_id() == [0, 1, 2, 3, 4, 5, 6, 7, 8]


# delete_any_trial_state
@t_base()
def test_delete_any_trial_state():
    storage = get_storage()

    states = [
        "ready",
        "running",
        "finished",
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )

    for i in range(len(states)):
        assert storage.trial.get_any_trial_state(i) == states[i]

    assert storage.trial.get_any_trial_state(trial_id=0) is not None
    assert storage.trial.get_any_trial_state(trial_id=1) is not None
    assert storage.trial.get_any_trial_state(trial_id=2) is not None

    assert storage.trial.delete_any_trial_state(trial_id=0) is None
    assert storage.trial.get_any_trial_state(trial_id=0) is None
    assert storage.trial.get_any_trial_state(trial_id=1) is not None
    assert storage.trial.get_any_trial_state(trial_id=2) is not None

    assert storage.trial.delete_any_trial_state(trial_id=1) is None
    assert storage.trial.get_any_trial_state(trial_id=0) is None
    assert storage.trial.get_any_trial_state(trial_id=1) is None
    assert storage.trial.get_any_trial_state(trial_id=2) is not None

    assert storage.trial.delete_any_trial_state(trial_id=2) is None
    assert storage.trial.get_any_trial_state(trial_id=0) is None
    assert storage.trial.get_any_trial_state(trial_id=1) is None
    assert storage.trial.get_any_trial_state(trial_id=2) is None


# delete_any_trial_state exception
@t_base()
def test_delete_any_trial_state_exception():
    storage = get_storage()

    states = [
        "ready",
        "running",
        "finished",
    ]

    for i in range(len(states)):
        storage.trial.set_any_trial_state(
            trial_id=i,
            state=states[i]
        )

    for i in range(len(states)):
        assert storage.trial.get_any_trial_state(i) == states[i]

    init()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_state = undecorated(storage.trial.delete_any_trial_state)
        delete_any_trial_state(storage.trial, trial_id=0)
