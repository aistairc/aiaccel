from aiaccel.storage.storage import Storage
from base import t_base
from base import db_path
from base import ws

# init_alive
@t_base()
def test_init_alive():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.init_alive() is None
    alives = storage.alive.get_state()
    assert alives['optimizer'] == 0
    assert alives['scheduler'] == 0


# set_any_process_state
@t_base()
def test_set_any_provess_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.set_any_process_state('optimizer', 1) is None
    alives = storage.alive.get_state()
    assert alives['optimizer'] == 1


# get_any_process_state
@t_base()
def test_get_any_process_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.get_any_process_state('optimizer') == 0
    storage.alive.set_any_process_state('optimizer', 1)
    assert storage.alive.get_any_process_state('optimizer') == 1


# get_state
@t_base()
def test_get_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('optimizer', 1)
    alives = storage.alive.get_state()
    assert alives['optimizer'] == 1
    assert alives['scheduler'] == 0


    storage.alive.set_any_process_state('scheduler', 1)
    alives = storage.alive.get_state()
    assert alives['optimizer'] == 1
    assert alives['scheduler'] == 1


# stop_any_process_state
@t_base()
def test_stop_any_process():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('optimizer', 1)
    assert storage.alive.get_any_process_state('optimizer') == 1
    storage.alive.stop_any_process('optimizer')
    assert storage.alive.get_any_process_state('optimizer') == 0
    

# check_alive
@t_base()
def test_check_alive():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('optimizer', 0)
    assert storage.alive.check_alive('optimizer') is False
    storage.alive.set_any_process_state('optimizer', 1)
    assert storage.alive.check_alive('optimizer') is True
