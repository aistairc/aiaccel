import pytest
from sqlalchemy.exc import SQLAlchemyError
from undecorated import undecorated

from tests.unit.storage_test.db.base import get_storage, init, t_base, ws


# set_any_trial_error
@t_base()
def test_set_any_trial_error():
    storage = get_storage()

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
    storage = get_storage()

    trial_id = 0
    message = "hoge"

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_error = undecorated(storage.error.set_any_trial_error)
        set_any_trial_error(storage.error, trial_id=trial_id, error_message=message)


# get_any_trial_error
@t_base()
def test_get_any_trial_error():
    storage = get_storage()

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
    storage = get_storage()

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


# set_any_trial_exitcode
@t_base()
def test_set_any_trial_exitcode():
    storage = get_storage()

    trial_id = 0
    exitcode = 0
    assert storage.error.set_any_trial_exitcode(
        trial_id=trial_id,
        exitcode=exitcode
    ) is None

    # update
    assert storage.error.set_any_trial_exitcode(
        trial_id=trial_id,
        exitcode=exitcode
    ) is None


# set_any_trial_exitcode exception
@t_base()
def test_set_any_trial_exitcode_exception():
    storage = get_storage()

    trial_id = 0
    exitcode = 0

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_exitcode = undecorated(storage.error.set_any_trial_exitcode)
        set_any_trial_exitcode(storage.error, trial_id=trial_id, exitcode=exitcode)


# get_any_trial_exitcode
@t_base()
def test_get_any_trial_exitcode():
    storage = get_storage()

    trial_id = 0
    exitcode = 0
    storage.error.set_any_trial_exitcode(
        trial_id=trial_id,
        exitcode=exitcode
    )

    get_mess = storage.error.get_any_trial_exitcode(trial_id)
    assert exitcode == get_mess


# get_failed_exitcode_trial_id
@t_base()
def test_get_failed_exitcode_trial_id():
    storage = get_storage()

    assert storage.error.get_failed_exitcode_trial_id() == []
    assert storage.error.get_failed_exitcode_trial_id() == []

    ids = [0, 1, 2]
    codes = [
        0,
        0,
        1
    ]

    for i in range(len(ids)):
        storage.error.set_any_trial_exitcode(
            trial_id=ids[i],
            exitcode=codes[i]
        )

    assert storage.error.get_failed_exitcode_trial_id() == [2]


# all_delete
@t_base()
def test_all_delete():
    storage = get_storage()

    ids = [0, 1, 2]
    mess = [
        "hoge_0",
        "hoge_1",
        "hoge_2"
    ]
    codes = [
        0,
        0,
        1
    ]

    for i in range(len(ids)):
        storage.error.set_any_trial_error(
            trial_id=ids[i],
            error_message=mess[i]
        )
        storage.error.set_any_trial_exitcode(
            trial_id=ids[i],
            exitcode=codes[i]
        )

    assert storage.error.all_delete() is None
    for id in ids:
        assert storage.error.get_any_trial_error(id) is None


# all_delete
@t_base()
def test_all_delete_exception():
    storage = get_storage()

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
    storage = get_storage()

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
    storage = get_storage()

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
