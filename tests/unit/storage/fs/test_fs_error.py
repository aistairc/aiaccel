from aiaccel.storage.storage import Storage
from tests.unit.storage.fs.base import t_base
from tests.unit.storage.fs.base import db_path
from tests.unit.storage.fs.base import ws
from tests.unit.storage.fs.base import config_path


# set_any_trial_error
@t_base()
def test_set_any_trial_error():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)


    trial_id = 0
    message = "hoge"
    assert storage.error.set_any_trial_error(
        trial_id=trial_id,
        error_message=message
    ) is None


# get_any_trial_error
@t_base()
def test_get_any_trial_error():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
    storage = Storage(ws.path, fsmode=True, config_path=config_path)

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
