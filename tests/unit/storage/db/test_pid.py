from aiaccel.storage.storage import Storage
from base import t_base
from base import db_path
from base import ws


# set_any_process_pid
@t_base()
def test_set_any_process_pid():
    storage = Storage(ws.path)
    
    module_name = "test_process"
    pid = 12345

    assert storage.pid.set_any_process_pid(
        module_name=module_name,
        pid=pid
    ) is None


# get_any_process_pid
@t_base()
def test_get_any_process_pid():
    storage = Storage(ws.path)
    
    module_name = "test_process"
    pid = 12345

    storage.pid.set_any_process_pid(
        module_name=module_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(module_name) == pid


# all_delete
@t_base()
def test_all_delete():
    storage = Storage(ws.path)
    
    module_name = "test_process"
    pid = 12345

    storage.pid.set_any_process_pid(
        module_name=module_name,
        pid=pid
    )

    assert storage.pid.get_any_process_pid(module_name) == pid

    storage.pid.all_delete()
    assert storage.pid.get_any_process_pid(module_name) is None


# delete_any_process_pid
@t_base()
def test_delete_any_process_pid():
    storage = Storage(ws.path)
    
    module_name = "test_process_0"
    pid = 10000
    storage.pid.set_any_process_pid(
        module_name=module_name,
        pid=pid
    )

    module_name = "test_process_1"
    pid = 10001
    storage.pid.set_any_process_pid(
        module_name=module_name,
        pid=pid
    )

    module_name = "test_process_2"
    pid = 10002
    storage.pid.set_any_process_pid(
        module_name=module_name,
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
