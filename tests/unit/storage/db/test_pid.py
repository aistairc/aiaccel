from aiaccel.storage.storage import Storage
from base import db_path, t_base, ws, init
import pytest
from undecorated import undecorated
from sqlalchemy.exc import SQLAlchemyError

# set_any_process_pid
@t_base()
def test_set_any_process_pid():
    storage = Storage(ws.path)
    
    process_name = "master"
    pid = 12345

    assert storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    ) is None

    # update 
    assert storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    ) is None


# set_any_process_pid exception
@t_base()
def test_set_any_process_pid_exception():
    storage = Storage(ws.path)
    
    process_name = "test_process"
    pid = 12345

    init()
    with pytest.raises(SQLAlchemyError):
        set_any_process_pid = undecorated(storage.pid.set_any_process_pid)
        set_any_process_pid(
            storage.pid,
            process_name=process_name,
            pid=pid
        )


# get_any_process_pid
@t_base()
def test_get_any_process_pid():
    storage = Storage(ws.path)
    
    process_name = "test_process"
    pid = 12345

    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(process_name) == pid


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)
    
    process_name = "test_process"
    pid = 12345

    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(process_name) == pid

    storage.pid.all_delete()
    assert storage.pid.get_any_process_pid(process_name) is None


# all_delete exception
@t_base()
def test_all_delete_exception():
    storage = Storage(ws.path)
    
    process_name = "test_process"
    pid = 12345

    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    init()
    with pytest.raises(SQLAlchemyError):
        all_delete = undecorated(storage.pid.all_delete)
        all_delete(storage.pid)
    


# delete_any_process_pid
@t_base()
def test_delete_any_process_pid():
    storage = Storage(ws.path)
    
    process_name = "test_process_0"
    pid = 10000
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    process_name = "test_process_1"
    pid = 10001
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    process_name = "test_process_2"
    pid = 10002
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid("test_process_0") is not None
    assert storage.pid.get_any_process_pid("test_process_1") is not None
    assert storage.pid.get_any_process_pid("test_process_2") is not None

    assert storage.pid.delete_any_process_pid("test_process_0") is None
    assert storage.pid.get_any_process_pid("test_process_0") is None
    assert storage.pid.get_any_process_pid("test_process_1") is not None
    assert storage.pid.get_any_process_pid("test_process_2") is not None

    assert storage.pid.delete_any_process_pid("test_process_1") is None
    assert storage.pid.get_any_process_pid("test_process_0") is None
    assert storage.pid.get_any_process_pid("test_process_1") is None
    assert storage.pid.get_any_process_pid("test_process_2") is not None

    assert storage.pid.delete_any_process_pid("test_process_2") is None
    assert storage.pid.get_any_process_pid("test_process_0") is None
    assert storage.pid.get_any_process_pid("test_process_1") is None
    assert storage.pid.get_any_process_pid("test_process_2") is None


# delete_any_process_pid exception
@t_base()
def test_delete_any_process_pid_exception():
    storage = Storage(ws.path)
    
    process_name = "test_process_0"
    pid = 10000
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    process_name = "test_process_1"
    pid = 10001
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    process_name = "test_process_2"
    pid = 10002
    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    init()
    with pytest.raises(SQLAlchemyError):
        delete_any_process_pid = undecorated(storage.pid.delete_any_process_pid)
        delete_any_process_pid(storage.pid, "test_process_0")
