import pytest
from base import init, t_base, ws
from sqlalchemy.exc import SQLAlchemyError
from undecorated import undecorated

from aiaccel.storage.storage import Storage


# set_any_trial_start_time
@t_base()
def test_set_any_trial_start_time():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"

    assert storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time) is None

    assert storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time) is None


# set_any_trial_start_time exception
@t_base()
def test_set_any_trial_start_time_exception():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_start_time = undecorated(storage.timestamp.set_any_trial_start_time)
        set_any_trial_start_time(storage.timestamp, trial_id=trial_id, start_time=start_time)


# set_any_trial_end_time
@t_base()
def test_set_any_trial_end_time():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    assert storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time) is None

    assert storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time) is None


# set_any_trial_end_time exception
@t_base()
def test_set_any_trial_end_time_exception():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_trial_end_time = undecorated(storage.timestamp.set_any_trial_end_time)
        set_any_trial_end_time(storage.timestamp, trial_id=trial_id, end_time=end_time)


@t_base()
def test_set_any_trial_end_time_assersion():
    storage = Storage(ws.path)

    trial_id = 0
    end_time = "10:00"

    with pytest.raises(AssertionError):
        set_any_trial_end_time = undecorated(storage.timestamp.set_any_trial_end_time)
        set_any_trial_end_time(storage.timestamp, trial_id=trial_id, end_time=end_time)


# get_any_trial_start_time
@t_base()
def test_get_any_trial_start_time():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    assert storage.timestamp.get_any_trial_start_time(trial_id) == start_time
    assert storage.timestamp.get_any_trial_start_time(65535) is None

    # if data.start_time == ''
    trial_id = 1
    start_time = ""
    end_time = "10:00"
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)
    assert storage.timestamp.get_any_trial_start_time(trial_id) is None


# get_any_trial_end_time
@t_base()
def test_get_any_trial_end_time():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    assert storage.timestamp.get_any_trial_end_time(trial_id) == end_time
    assert storage.timestamp.get_any_trial_end_time(65535) is None

    trial_id = 2
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    assert storage.timestamp.get_any_trial_end_time(trial_id) is None

    trial_id = 3
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time="")
    assert storage.timestamp.get_any_trial_end_time(trial_id) is None


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    assert storage.timestamp.get_any_trial_start_time(trial_id) == start_time
    assert storage.timestamp.get_any_trial_end_time(trial_id) == end_time
    assert storage.timestamp.all_delete() is None
    assert storage.timestamp.get_any_trial_start_time(trial_id) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id) is None


# all_delete exception
@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)

    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.timestamp.all_delete)
        all_delete(storage.timestamp)


# delete_any_trial_timestamp
@t_base()
def test_delete_any_trial_timestamp():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    trial_id = 1
    start_time = "00:00"
    end_time = "10:00"
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    trial_id = 2
    start_time = "00:00"
    end_time = "10:00"
    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    assert storage.timestamp.get_any_trial_start_time(trial_id=0) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=0) is not None
    assert storage.timestamp.get_any_trial_start_time(trial_id=1) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=1) is not None
    assert storage.timestamp.get_any_trial_start_time(trial_id=2) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=2) is not None

    assert storage.timestamp.delete_any_trial_timestamp(trial_id=0) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=1) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=1) is not None
    assert storage.timestamp.get_any_trial_start_time(trial_id=2) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=2) is not None

    assert storage.timestamp.delete_any_trial_timestamp(trial_id=1) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=1) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=1) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=2) is not None
    assert storage.timestamp.get_any_trial_end_time(trial_id=2) is not None

    assert storage.timestamp.delete_any_trial_timestamp(trial_id=2) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=0) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=1) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=1) is None
    assert storage.timestamp.get_any_trial_start_time(trial_id=2) is None
    assert storage.timestamp.get_any_trial_end_time(trial_id=2) is None


# delete_any_trial_timestamp excedption
@t_base()
def test_delete_any_trial_timestamp_excedption():
    storage = Storage(ws.path)

    trial_id = 0
    start_time = "00:00"
    end_time = "10:00"

    storage.timestamp.set_any_trial_start_time(trial_id=trial_id, start_time=start_time)
    storage.timestamp.set_any_trial_end_time(trial_id=trial_id, end_time=end_time)

    init()
    with pytest.raises(SQLAlchemyError):
        delete_any_trial_timestamp = undecorated(storage.timestamp.delete_any_trial_timestamp)
        delete_any_trial_timestamp(storage.timestamp, trial_id=0)
