import tempfile
import os
import shutil
from pathlib import Path

import fasteners
import pytest
from aiaccel.config import Config, load_config
from aiaccel.storage import Storage
from aiaccel.util import create_yaml
from aiaccel.util import load_yaml
from aiaccel.util import interprocess_lock_file
import json

WORK_SUB_DIRECTORIES = [
    'abci_output', 'alive', 'hp', 'hp/finished', 'hp/ready', 'hp/running',
    'lock', 'log', 'resource', 'result', 'runner', 'state', 'verification'
]
WORK_FILES = [
    'config.json',
    'config.yml',
    'hyperparameter.json',
    'original_main.py',
    'wrapper.py',
    'wrapper_abci.sh',
    'grid_config.json'
]


def clean_directory(path: Path, exclude_dir: list = None,
                    exclude_file: list = None, dict_lock: Path = None) -> bool:
    """Remove all files in the directory recursively.

    Args:
        path (Path): Path for the directory
        exclude_dir (list): Exclude directories. All files in directories, are
            not removed.
        exclude_file (list): Exclude files not to be removed.
        dict_lock (Path): A directory saving lock files.

    Returns:
        None
    """
    path = to_path(path)

    if not path.is_dir():
        return False

    if exclude_dir is None:
        exclude_dir = []

    if exclude_file is None:
        exclude_file = []

    if dict_lock is None:
        for p in path.glob('**/*'):
            if p.is_file():
                if True not in [p in d.parts for d in
                                exclude_dir + exclude_file]:
                    p.unlink()
    else:
        with fasteners.InterProcessLock(interprocess_lock_file(path, dict_lock)):
            for p in path.glob('**/*'):
                if p.is_file():
                    if True not in [p in d.parts for d in exclude_dir + exclude_file]:
                        p.unlink()


def to_path(path):
    """Convert to Path object.

    Args:
        path (Any): An any path object.

    Returns:
        Path: A converted path object.
    """
    if isinstance(path, Path):
        return path.resolve()

    return Path(path).resolve()


@pytest.fixture
def cd_work(tmpdir):
    cwd = Path.cwd().resolve()
    os.chdir(tmpdir)
    yield
    os.chdir(cwd)


@pytest.fixture(scope="session")
def config_json(data_dir):
    return data_dir.joinpath('config.json')


@pytest.fixture(scope="session")
def grid_config_json(data_dir):
    return data_dir.joinpath('grid_config.json')


@pytest.fixture
def config_yaml(data_dir):
    return data_dir.joinpath('config.yml')


@pytest.fixture(scope="session")
def data_dir(root_dir):
    return root_dir.joinpath('test_data')


@pytest.fixture(scope="session")
def get_one_parameter(work_dir):
    def _get_one_parameter():
        path = work_dir.joinpath('result/0.yml')
        return load_yaml(path)

    return _get_one_parameter


@pytest.fixture(scope="session")
def load_test_config(config_json):

    def _load_test_config():
        return Config(config_json)

    return _load_test_config


@pytest.fixture(scope="session")
def load_test_config_org(config_json):

    def _load_test_config():
        return load_config(config_json)

    return _load_test_config


@pytest.fixture(scope="session")
def grid_load_test_config(grid_config_json):

    def _load_test_config():
        return Config(grid_config_json)

    return _load_test_config


@pytest.fixture(scope="session")
def root_dir():
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def work_dir(tmpdir):
    return tmpdir.joinpath('work')


@pytest.fixture(scope="session")
def tmpdir():
    tmpdir = tempfile.TemporaryDirectory()
    yield Path(tmpdir.name).resolve()
    tmpdir.cleanup()


@pytest.fixture(scope="session")
def create_tmp_config(data_dir, tmpdir, work_dir):
    def _create_tmp_config(conf_path=None):
        if conf_path is None:
            conf_path = data_dir.joinpath('config.yaml')

        if conf_path.suffix == ".yaml" or conf_path.suffix == ".yml":
            yml = load_yaml(conf_path)
            yml['generic']['workspace'] = str(work_dir)
            tmp_conf_path = tmpdir.joinpath('config.yaml')
            create_yaml(tmp_conf_path, yml)
        elif conf_path.suffix == ".json":
            with open(conf_path, 'r') as f:
                json_obj = json.load(f)
                json_obj['generic']['workspace'] = str(work_dir)
                tmp_conf_path = tmpdir.joinpath('config.json')
            with open(tmp_conf_path, 'w') as f:
                json.dump(json_obj, f)

        return tmp_conf_path

    return _create_tmp_config


@pytest.fixture(scope="session")
def setup_hp_files(work_dir):
    def _setup_hp_files(hp_type, n=1):
        if hp_type not in ['ready', 'running', 'finished']:
            hp_type = 'ready'

        # hp_files = list(data_dir.joinpath('work/hp/finished').glob('*.hp'))
        # n = min(n, len(hp_files))

        # for i in range(1, n+1):
        #     shutil.copyfile(
        #         data_dir.joinpath('work/hp/finished/{:03}.hp'.format(i)),
        #         work_dir.joinpath('hp/{}/{:03}.hp'.format(hp_type, i))
        #     )

        storage = Storage(work_dir)
        for i in range(n):
            storage.trial.set_any_trial_state(trial_id=i, state=hp_type)
    return _setup_hp_files


@pytest.fixture(scope="session")
def setup_hp_ready(setup_hp_files):
    def _setup_hp_ready(n=1):
        setup_hp_files('ready', n=n)

    return _setup_hp_ready


@pytest.fixture(scope="session")
def setup_hp_running(setup_hp_files):
    def _setup_hp_running(n=1):
        setup_hp_files('running', n=n)

    return _setup_hp_running


@pytest.fixture(scope="session")
def setup_hp_finished(setup_hp_files):
    def _setup_hp_finished(n=1):
        setup_hp_files('finished', n=n)

    return _setup_hp_finished


@pytest.fixture(scope="session")
def setup_result(work_dir):
    def _setup_result(n=1):
        storage = Storage(work_dir)
        running = storage.get_running()
        print('dbg')
        print(running)
        for trial_id in running:
            storage.result.set_any_trial_objective(trial_id=trial_id, objective=0)

    return _setup_result


@pytest.fixture(scope="session")
def database_remove(work_dir):
    def _database_remove():
        p = work_dir / 'storage' / 'storage.db'
        if p.exists():
            p.unlink()
    return _database_remove


@pytest.fixture
def clean_work_dir(work_dir, data_dir):
    def _clean_work_dir():
        while not work_dir.is_dir():
            if not work_dir.exists():
                work_dir.mkdir()

            if not work_dir.is_dir():
                work_dir.unlink()

        valid_dir = [work_dir.joinpath(wsd) for wsd in WORK_SUB_DIRECTORIES]
        valid_dir += [work_dir]
        work_files = [work_dir.joinpath(wf) for wf in WORK_FILES]

        clean_directory(
            work_dir, exclude_file=work_files
        )

        for p in work_dir.glob('**/*'):
            # TODO: this part can be replaced using PurePath.is_relative_to()
            #  from version 3.9
            if p.is_dir() and \
                    p not in [work_dir.joinpath(wsd) for wsd in WORK_SUB_DIRECTORIES] and \
                    any([str(work_dir.joinpath(wsd)) in str(p) for wsd in WORK_SUB_DIRECTORIES]):
                shutil.rmtree(p)

        for d in valid_dir:
            while d.exists() and not d.is_dir():
                d.unlink()
                d.mkdir()

        for wsd in WORK_SUB_DIRECTORIES:
            if not work_dir.joinpath(wsd).exists():
                work_dir.joinpath(wsd).mkdir()

        for wf in WORK_FILES:
            if not work_dir.joinpath(wf).exists():
                shutil.copyfile(data_dir.joinpath(wf), work_dir.joinpath(wf))

    return _clean_work_dir


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "noautousefixture: Marks test without autouse-fixture of test class.")
