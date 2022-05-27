from aiaccel.config import load_config
from aiaccel.config import Config
from aiaccel.util.filesystem import clean_directory, load_yaml
from pathlib import Path
import os
import pytest
import shutil

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


@pytest.fixture
def cd_work(work_dir):
    cwd = Path.cwd().resolve()
    os.chdir(work_dir)
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
def get_one_parameter(data_dir):
    def _get_one_parameter():
        path = data_dir.joinpath('work/hp/finished/001.hp')
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
def work_dir(load_test_config, data_dir):
    test_config = load_test_config()
    return Path(test_config.workspace.get()).resolve()


@pytest.fixture(scope="session")
def setup_hp_files(data_dir, work_dir):
    def _setup_hp_files(hp_type, n=1):
        if hp_type not in ['ready', 'running', 'finished']:
            hp_type = 'ready'

        hp_files = list(data_dir.joinpath('work/hp/finished').glob('*.hp'))
        n = min(n, len(hp_files))

        for i in range(1, n+1):
            shutil.copyfile(
                data_dir.joinpath('work/hp/finished/{:03}.hp'.format(i)),
                work_dir.joinpath('hp/{}/{:03}.hp'.format(hp_type, i))
            )

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
def setup_result(data_dir: Path, work_dir: Path):
    def _setup_result(n=1):
        files = list(data_dir.joinpath('work/result').glob('*.result'))
        n = min(n, len(files))

        for i in range(1, n+1):
            shutil.copyfile(
                data_dir.joinpath('work/result/{:03}.result'.format(i)),
                work_dir.joinpath('result/{:03}.result'.format(i))
            )

    return _setup_result


@pytest.fixture(scope="session", autouse=True)
def setup_session_work_dir(base_clean_work_dir, data_dir, work_dir):
    base_clean_work_dir()
    for wf in WORK_FILES:
        shutil.copyfile(data_dir.joinpath(wf), work_dir.joinpath(wf))


@pytest.fixture(scope="session", autouse=True)
def teardown_session_work_dir(base_clean_work_dir):
    base_clean_work_dir()


@pytest.fixture(scope="session")
def base_clean_work_dir(data_dir, work_dir):
    def _base_clean_work_dir():
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

    return _base_clean_work_dir


@pytest.fixture
def clean_work_dir(work_dir, base_clean_work_dir):
    base_clean_work_dir()
    yield
    base_clean_work_dir()

