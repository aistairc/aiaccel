import shutil
from pathlib import Path

from aiaccel.storage import Storage
from aiaccel.util import (
    create_yaml,
    file_create,
    file_delete,
    file_read,
    get_dict_files,
    get_file_result,
    get_file_result_hp,
    load_yaml,
    make_directories,
    make_directory,
)
from aiaccel.workspace import Workspace


def test_create_yaml(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath("lock")
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    assert create_yaml(path, {}) is None
    file_delete(path)
    assert create_yaml(path, {}, dict_lock) is None


def test_file_create(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    file_create(path, "hello")
    assert path.exists()


def test_file_delete(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    dict_lock = work_dir.joinpath("lock")
    file_create(path, "hello")
    file_delete(path)
    assert not path.exists()
    file_create(path, "hello")
    file_delete(path, dict_lock)
    assert not path.exists()
    assert file_delete(Path("invalid_path")) is None


def test_file_read(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    dict_lock = work_dir.joinpath("lock")
    file_create(path, "hello")
    assert file_read(path) == "hello"
    assert file_read(path, dict_lock) == "hello"
    assert file_read(Path("invalid_path")) is None


def test_get_dict_files(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    dict_lock = work_dir.joinpath("lock")
    file_create(path, "hello")
    assert get_dict_files(alive_dir, "*.yml") == [path]
    assert get_dict_files(alive_dir, "*.yml", dict_lock) == [path]

    assert get_dict_files(Path("invalid_path"), "*.yml", dict_lock) is None


def test_get_file_result(clean_work_dir, setup_result, work_dir):
    content = {}
    create_yaml((work_dir / "result" / "001.result"), content)
    assert get_file_result(work_dir) == [work_dir.joinpath("result/001.result")]


def test_load_yaml(clean_work_dir, work_dir):
    clean_work_dir()
    alive_dir = work_dir.joinpath("alive")
    path = alive_dir.joinpath("master.yml")
    dict_lock = work_dir.joinpath("lock")
    create_yaml(path, {})
    assert load_yaml(path) == {}
    assert load_yaml(path, dict_lock) == {}


def test_make_directory(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath("lock")
    assert make_directory(work_dir.joinpath("result")) is None
    assert make_directory(work_dir.joinpath("new"), dict_lock) is None
    assert make_directory(work_dir.joinpath("new"), dict_lock) is None  # exists


def test_make_directories(clean_work_dir, work_dir):
    clean_work_dir()
    dict_lock = work_dir.joinpath("lock")
    ds = [work_dir.joinpath("hp", "ready"), work_dir.joinpath("hp", "exist"), work_dir.joinpath("hp", "new")]

    if work_dir.joinpath("hp", "exist").is_dir():
        work_dir.joinpath("hp", "exist").rmdir()

    create_yaml(work_dir.joinpath("hp", "exist"), {})
    assert make_directories(ds) is None
    shutil.rmtree(work_dir.joinpath("hp", "exist"))
    file_create(work_dir.joinpath("hp", "exist"), "hello", dict_lock)
    assert make_directories(ds, dict_lock) is None


def test_get_file_result_hp(work_dir):
    # no files
    assert get_file_result_hp(work_dir) == []

    file_path = work_dir / "result" / "test.hp"
    create_yaml(file_path, {})

    assert get_file_result_hp(work_dir) == [Path(file_path)]
