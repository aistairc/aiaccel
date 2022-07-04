from aiaccel.storage.storage import Storage
from base import t_base
from base import db_path
from base import ws
from base import config_path


# set_any_process_pid
@t_base()
def test_set_any_process_pid():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
    
    process_name = "master"
    pid = 12345

    assert storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    ) is None


# get_any_process_pid
@t_base()
def test_get_any_process_pid():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
    
    process_name = "master"
    pid = 12345

    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(process_name) == pid


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path, fsmode=True, config_path=config_path)
    
    process_name = "master"
    pid = 12345

    storage.pid.set_any_process_pid(
        process_name=process_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(process_name) == pid

    storage.pid.all_delete()
    assert storage.pid.get_any_process_pid(process_name) is None
