from pathlib import Path
from unittest.mock import patch

from aiaccel.config import Config
from aiaccel.util import TrialId


def test_trial_id_init(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    trial_id = TrialId(config_path)
    trial_id_2 = TrialId(config_path)

    assert trial_id.__init__(config_path) is None
    assert trial_id_2.__init__(config_path) is None


def test_zero_padding_any_trial_id(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    trial_id = TrialId(config_path)
    config = Config(config_path)

    name_length = config.name_length.get()
    file_hp_count_fmt = f'%0{name_length}d'
    assert trial_id.zero_padding_any_trial_id(trial_id=1) == file_hp_count_fmt % 1


def test_increment_1(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    config = Config(config_path)
    workspace = Path(config.workspace.get())
    trial_id = TrialId(config_path)
    pre = trial_id.get()
    assert trial_id.increment() is None
    now = trial_id.get()
    assert now == pre + 1

    with patch.object(trial_id.lock, 'acquire', return_value=False):
        assert trial_id.increment() is None

    (workspace / 'hp' / 'count.txt').unlink()
    assert trial_id.increment() is None


def test_get(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    trial_id = TrialId(config_path)
    trial_id.initial(42)
    assert trial_id.get() == 42


def test_initial(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    trial_id = TrialId(config_path)
    assert trial_id.initial(num=5) is None
    assert trial_id.get() == 5

    with patch.object(trial_id.lock, 'acquire', return_value=False):
        assert trial_id.initial(num=5) is None
        assert trial_id.get() == 5


def test_integer(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    trial_id = TrialId(config_path)
    trial_id.initial(num=42)
    assert trial_id.integer == 42


def test_string(create_tmp_config):
    config_path = Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)
    config = Config(config_path)
    name_length = config.name_length.get()
    file_hp_count_fmt = f'%0{name_length}d'

    trial_id = TrialId(config_path)
    trial_id.initial(num=42)
    assert trial_id.string == file_hp_count_fmt % 42
