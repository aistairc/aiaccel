from aiaccel.storage.storage import Storage

from base import db_path, t_base, ws
import pytest
from sqlalchemy.exc import SQLAlchemyError
from unittest.mock import patch

import sqlalchemy

from unittest.mock import MagicMock

# init_alive
@t_base()
def test_init_alive():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.init_alive() is None
    alives = storage.alive.get_state()
    assert alives['master'] == 0
    assert alives['optimizer'] == 0
    assert alives['scheduler'] == 0


# set_any_process_state
@t_base()
def test_set_any_process_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.set_any_process_state('master', 1) is None
    alives = storage.alive.get_state()
    assert alives['master'] == 1

# # set_any_process_state : Exception
# @t_base()
# @patch("aiaccel.storage.alive.Alive.session.commit", MagicMock(side_effect=SQLAlchemyError()))
# def test_set_any_process_state_exception():
#     storage = Storage(ws.path)
#     # storage.alive.init_alive()
#     with pytest.raises(SQLAlchemyError):
#         storage.alive.set_any_process_state('master', 1)

# get_any_process_state
@t_base()
def test_get_any_process_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    assert storage.alive.get_any_process_state('master') == 0
    storage.alive.set_any_process_state('master', 1)
    assert storage.alive.get_any_process_state('master') == 1


# # get_any_process_state exception
# @t_base()
# def test_get_any_process_state_exception():
#     from aiaccel.storage.alive import Alive
#     alive = Alive(ws.path / 'storage' / 'storage.db')
#     with pytest.raises(AssertionError):
#         alive.get_any_process_state('invalid')

# get_state
@t_base()
def test_get_state():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('master', 1)
    alives = storage.alive.get_state()
    assert alives['master'] == 1
    assert alives['optimizer'] == 0
    assert alives['scheduler'] == 0

    storage.alive.set_any_process_state('optimizer', 1)
    alives = storage.alive.get_state()
    assert alives['master'] == 1
    assert alives['optimizer'] == 1
    assert alives['scheduler'] == 0

    storage.alive.set_any_process_state('scheduler', 1)
    alives = storage.alive.get_state()
    assert alives['master'] == 1
    assert alives['optimizer'] == 1
    assert alives['scheduler'] == 1


# stop_any_process_state
@t_base()
def test_stop_any_process():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('master', 1)
    assert storage.alive.get_any_process_state('master') == 1
    storage.alive.stop_any_process('master')
    assert storage.alive.get_any_process_state('master') == 0
    

# check_alive
@t_base()
def test_check_alive():
    storage = Storage(ws.path)
    storage.alive.init_alive()

    storage.alive.set_any_process_state('master', 0)
    assert storage.alive.check_alive('master') is False
    storage.alive.set_any_process_state('master', 1)
    assert storage.alive.check_alive('master') is True
