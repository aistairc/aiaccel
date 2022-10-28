import pathlib
import shutil

from aiaccel.workspace import Workspace


def test_create():
    workspace = Workspace("test_workspace")
    workspace.clean()
    # ===  test start ===
    assert workspace.create() is True
    assert workspace.create() is False
    # ===  test end ===
    workspace.clean()


def test_exists():
    workspace = Workspace("test_workspace")
    workspace.clean()
    workspace.create()
    # ===  test start ===
    assert workspace.exists() is True
    # ===  test end ===
    workspace.clean()


def test_clean():
    workspace = Workspace("test_workspace")
    workspace.clean()
    workspace.create()
    # ===  test start ===
    assert workspace.clean() is None
    assert workspace.clean() is None
    # ===  test end ===
    workspace.clean()


def test_check_consists():
    workspace = Workspace("test_workspace")
    workspace.clean()
    workspace.create()
    # ===  test start ===
    assert workspace.check_consists() is True
    workspace.clean()
    assert workspace.check_consists() is False
    # ===  test end ===


def test_move_completed_data():
    workspace = Workspace("test_workspace")
    workspace.clean()
    workspace.create()
    # ===  test start ===
    assert type(workspace.move_completed_data()) == pathlib.PosixPath
    assert workspace.move_completed_data() is None
    # ===  test end ===
    shutil.rmtree("./results")
    workspace.clean()
