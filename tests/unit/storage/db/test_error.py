from aiaccel.storage.storage import Storage
from base import db_path, t_base, ws, init
from sqlalchemy.exc import SQLAlchemyError

from undecorated import undecorated
import pytest

# set_any_trial_error
@t_base()
def test_set_any_trial_error():
    storage = Storage(ws.path)

    trial_id = 0
    message = "hoge"
    assert storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=message
    ) is None

    # update
    assert storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=message
    ) is None


# set_any_trial_error exception
@t_base()
def test_set_any_trial_error_exception():
    storage = Storage(ws.path)

    trial_id = 0
    message = "hoge"

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_error = undecorated(storage.error.set_any_trial_error)
        set_any_trial_error(storage.error, trial_id=trial_id, error_message=message)


# get_any_trial_error
@t_base()
def test_get_any_trial_error():
    storage = Storage(ws.path)

    trial_id = 0
    message = "hoge"
    storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=message
    )

    get_mess = storage.error.get_any_trial_error(trial_id)
    assert message == get_mess


# get_error_trial_id
@t_base()
def test_get_error_trial_id():
    storage = Storage(ws.path)
    assert storage.error.get_error_trial_id() == []

    assert storage.error.get_error_trial_id() == []

    ids = [0, 1, 2]
    mess = [
        "hoge_0",
        "hoge_1",
        "hoge_2"
    ]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=mess[i]
        )

    assert storage.error.get_error_trial_id() == ids


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    mess = [
        "hoge_0",
        "hoge_1",
        "hoge_2"
    ]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=mess[i]
        )

    assert storage.error.all_delete() is None
    for id in ids:
        assert storage.error.get_any_trial_error(id) is None


# all_delete
@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    mess = [
        "hoge_0",
        "hoge_1",
        "hoge_2"
    ]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=mess[i]
        )

    (ws.path / 'storage/storage.db').unlink()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.error.all_delete)
        all_delete(storage.error)


# delete_any_trial_error
@t_base()
def test_delete_any_trial_error():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    messages = ["hoge0", "hoge1", "hoge2"]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=messages[i]
        )

    assert storage.error.get_any_trial_error(trial_id=0) is not None
    assert storage.error.get_any_trial_error(trial_id=1) is not None
    assert storage.error.get_any_trial_error(trial_id=2) is not None

    assert storage.error.delete_any_trial_error(trial_id=0) is None
    assert storage.error.get_any_trial_error(trial_id=0) is None
    assert storage.error.get_any_trial_error(trial_id=1) is not None
    assert storage.error.get_any_trial_error(trial_id=2) is not None

    assert storage.error.delete_any_trial_error(trial_id=1) is None
    assert storage.error.get_any_trial_error(trial_id=0) is None
    assert storage.error.get_any_trial_error(trial_id=1) is None
    assert storage.error.get_any_trial_error(trial_id=2) is not None

    assert storage.error.delete_any_trial_error(trial_id=2) is None
    assert storage.error.get_any_trial_error(trial_id=0) is None
    assert storage.error.get_any_trial_error(trial_id=1) is None
    assert storage.error.get_any_trial_error(trial_id=2) is None


# delete_any_trial_error exception
@t_base()
def test_delete_any_trial_error_exception():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    messages = ["hoge0", "hoge1", "hoge2"]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=messages[i]
        )

    (ws.path / 'storage/storage.db').unlink()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_error = undecorated(storage.error.delete_any_trial_error)
        delete_any_trial_error(storage.error, trial_id=0)
