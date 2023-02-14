import pytest
from undecorated import undecorated
from sqlalchemy.exc import SQLAlchemyError

from aiaccel.storage import Storage
from base import t_base, ws, init


# set_any_trial_state
@t_base()
def test_set_any_trial_state():
    storage = Storage(ws.path)

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
    storage = Storage(ws.path)

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
        assert storage.trial.get_any_trial_state(i) == states[i]


# get_any_state_list
@t_base()
def test_get_any_state_list():
    storage = Storage(ws.path)

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
        assert storage.trial.get_any_trial_state(i) == states[i]

    assert storage.trial.all_delete() is None

    for i in range(len(states)):
        assert storage.trial.get_any_trial_state(i) is None


# all_delete exception
@t_base()
def test_all_delete_exception():
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
        assert storage.trial.get_any_trial_state(i) == states[i]

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.trial.all_delete)
        all_delete(storage.trial)


# get_ready
@t_base()
def test_get_ready():
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

    assert storage.trial.get_ready() == [0, 1]


# get_running
@t_base()
def test_get_running():
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

    assert storage.trial.get_running() == [2, 3, 4]


# get_finished
@t_base()
def test_get_finished():
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

    assert storage.trial.get_finished() == [5, 6, 7, 8]


# get_all_trial_id
@t_base()
def test_get_all_trial_id():
    storage = Storage(ws.path)

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
    storage = Storage(ws.path)

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
    storage = Storage(ws.path)

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
