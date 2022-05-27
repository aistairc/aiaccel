from aiaccel.util.filesystem import check_alive_file, copy_directory, create_yaml,\
    file_create, file_delete, file_read, get_basename, get_dict_files,\
    get_file_hp_ready, get_file_hp_running, get_file_hp_finished,\
    get_file_result, load_yaml, make_directories, make_directory, move_file
import shutil


def test_check_alive_file(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    file_create(path, 'hello')
    assert check_alive_file(path)
    assert check_alive_file(path, dict_lock)


def test_copy_directory(clean_work_dir, work_dir):
    dict_lock = work_dir.joinpath('lock')
    alive_dir = work_dir.joinpath('alive')
    to_directory = work_dir.joinpath('copied')
    assert copy_directory(alive_dir, to_directory) is None
    shutil.rmtree(to_directory)
    assert copy_directory(alive_dir, to_directory, dict_lock) is None


def test_create_yaml(clean_work_dir, work_dir):
    dict_lock = work_dir.joinpath('lock')
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    assert create_yaml(path, {}) is None
    file_delete(path)
    assert create_yaml(path, {}, dict_lock) is None


def test_file_create(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    file_create(path, 'hello')
    assert path.exists()


def test_file_delete(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    file_create(path, 'hello')
    file_delete(path)
    assert not path.exists()
    file_create(path, 'hello')
    file_delete(path, dict_lock)
    assert not path.exists()


def test_file_read(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    file_create(path, 'hello')
    assert file_read(path) == 'hello'
    assert file_read(path, dict_lock) == 'hello'


def test_get_basename(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    file_create(path, 'hello')
    assert get_basename(path) == 'master'


def test_get_dict_files(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    file_create(path, 'hello')
    assert get_dict_files(alive_dir, '*.yml') == [path]
    assert get_dict_files(alive_dir, '*.yml', dict_lock) == [path]


def test_get_file_hp_finished(clean_work_dir, setup_hp_finished, work_dir):
    setup_hp_finished(1)
    assert get_file_hp_finished(work_dir) == [
        work_dir.joinpath('hp/finished/001.hp')
    ]


def test_get_file_hp_ready(clean_work_dir, setup_hp_ready, work_dir):
    setup_hp_ready(1)
    assert get_file_hp_ready(work_dir) == [
        work_dir.joinpath('hp/ready/001.hp')
    ]


def test_get_file_hp_running(clean_work_dir, setup_hp_running, work_dir):
    setup_hp_running(1)
    assert get_file_hp_running(work_dir) == [
        work_dir.joinpath('hp/running/001.hp')
    ]


def test_get_file_result(clean_work_dir, setup_result, work_dir):
    setup_result(1)
    assert get_file_result(work_dir) == [
        work_dir.joinpath('result/001.result')
    ]


def test_load_yaml(clean_work_dir, work_dir):
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    dict_lock = work_dir.joinpath('lock')
    create_yaml(path, {})
    assert load_yaml(path) == {}
    assert load_yaml(path, dict_lock) == {}


def test_make_directory(clean_work_dir, work_dir):
    dict_lock = work_dir.joinpath('lock')
    assert make_directory(work_dir.joinpath('result')) is None
    assert make_directory(work_dir.joinpath('new'), dict_lock) is None


def test_make_directories(clean_work_dir, work_dir):
    dict_lock = work_dir.joinpath('lock')
    ds = [
        work_dir.joinpath('hp', 'ready'),
        work_dir.joinpath('hp', 'exist'),
        work_dir.joinpath('hp', 'new')
    ]

    if work_dir.joinpath('hp', 'exist').is_dir():
        work_dir.joinpath('hp', 'exist').rmdir()

    create_yaml(work_dir.joinpath('hp', 'exist'), {})
    assert make_directories(ds) is None
    shutil.rmtree(work_dir.joinpath('hp', 'exist'))
    file_create(work_dir.joinpath('hp', 'exist'), 'hello', dict_lock)
    assert make_directories(ds, dict_lock) is None


def test_move_file(clean_work_dir, work_dir):
    dict_lock = work_dir.joinpath('lock')
    alive_dir = work_dir.joinpath('alive')
    path = alive_dir.joinpath('master.yml')
    create_yaml(path, {})
    move_path = alive_dir.joinpath('optimizer.yml')
    assert move_file(path, move_path, dict_lock) is None
