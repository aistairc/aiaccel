from aiaccel.storage.storage import Storage
from base import db_path, t_base, ws, init
from sqlalchemy.exc import SQLAlchemyError

from undecorated import undecorated
import pytest

# set_any_trial_return_code
@t_base()
def test_set_any_trial_return_code():
    storage = Storage(ws.path)

    trial_id = 0
    assert storage.returncode.set_any_trial_return_code(
        trial_id=trial_id,
        return_code=1
    ) is None

    # update
    assert storage.returncode.set_any_trial_return_code(
        trial_id=trial_id,
        return_code=0
    ) is None


# set_any_trial_return_code exception
@t_base()
def test_set_any_trial_return_code_exception():
    storage = Storage(ws.path)

    trial_id = 0

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_return_code = undecorated(storage.returncode.set_any_trial_return_code)
        set_any_trial_return_code(storage.returncode, trial_id=trial_id, return_code=1)


# get_any_trial_return_code
@t_base()
def test_get_any_trial_return_code():
    storage = Storage(ws.path)

    trial_id = 0
    storage.returncode.set_any_trial_return_code(
        trial_id=trial_id,
        return_code=1
    )

    get_returncode = storage.returncode.get_any_trial_return_code(trial_id)
    assert get_returncode == 1


# get_return_code_trial_id
@t_base()
def test_get_return_code_trial_id():
    storage = Storage(ws.path)
    assert storage.returncode.get_return_code_trial_id() == []

    assert storage.returncode.get_return_code_trial_id() == []

    ids = [0, 1, 2]
    mess = [0, 1, 0]

    for i in range(len(ids)):
        storage.returncode.set_any_trial_return_code(
            trial_id=ids[i],
            return_code=mess[i]
        )

    assert storage.returncode.get_return_code_trial_id() == ids


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    mess = [0, 1, 0]

    for i in range(len(ids)):
        storage.returncode.set_any_trial_return_code(
            trial_id=ids[i],
            return_code=mess[i]
        )

    assert storage.returncode.all_delete() is None
    for id in ids:
        assert storage.returncode.get_any_trial_return_code(id) is None


# all_delete
@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    mess = [0, 1, 0]

    for i in range(len(ids)):
        storage.returncode.set_any_trial_return_code(
            trial_id=ids[i],
            return_code=mess[i]
        )

    (ws.path / 'storage/storage.db').unlink()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.returncode.all_delete)
        all_delete(storage.returncode)


# delete_any_trial_return_code
@t_base()
def test_delete_any_trial_return_code():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    messages = [0, 1, 0]

    for i in range(len(ids)):
        storage.returncode.set_any_trial_return_code(
            trial_id=ids[i],
            return_code=messages[i]
        )

    assert storage.returncode.get_any_trial_return_code(trial_id=0) is not None
    assert storage.returncode.get_any_trial_return_code(trial_id=1) is not None
    assert storage.returncode.get_any_trial_return_code(trial_id=2) is not None

    assert storage.returncode.delete_any_trial_return_code(trial_id=0) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=0) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=1) is not None
    assert storage.returncode.get_any_trial_return_code(trial_id=2) is not None

    assert storage.returncode.delete_any_trial_return_code(trial_id=1) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=0) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=1) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=2) is not None

    assert storage.returncode.delete_any_trial_return_code(trial_id=2) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=0) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=1) is None
    assert storage.returncode.get_any_trial_return_code(trial_id=2) is None


# delete_any_trial_return_code exception
@t_base()
def test_delete_any_trial_return_code_exception():
    storage = Storage(ws.path)

    ids = [0, 1, 2]
    messages = [0, 1, 0]

    for i in range(len(ids)):
        storage.returncode.set_any_trial_return_code(
            trial_id=ids[i],
            return_code=messages[i]
        )

    (ws.path / 'storage/storage.db').unlink()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_return_code = undecorated(storage.returncode.delete_any_trial_return_code)
        delete_any_trial_return_code(storage.returncode, trial_id=0)
