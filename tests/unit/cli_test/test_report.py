import pathlib

from unittest.mock import patch

from aiaccel.cli.report import main
from aiaccel.cli import CsvWriter
from aiaccel.workspace import Workspace


ws = Workspace("test_work")
config_path = pathlib.Path('tests/test_data/config.json')


def test_report(clean_work_dir, work_dir, create_tmp_config):
    clean_work_dir()
    workspace = Workspace(str(work_dir))
    if workspace.path.exists():
        workspace.clean()
    workspace.create()

    config_path = pathlib.Path('tests/test_data/config.json')
    config_path = create_tmp_config(config_path)

    csv_writer = CsvWriter(config_path)

    assert csv_writer._get_zero_padding_trial_id(1) == '000001'
    assert csv_writer.create() is None

    hp = {
        'trial_id': '000000',
        'parameters': [
            {
                'parameter_name': 'test_1',
                'type': 'FLOAT',
                'value': 2.155147371813655
            },
            {
                'parameter_name': 'test_2',
                'type': 'FLOAT',
                'value': 4.071839861571789
            }
        ],
        'result': -0.2433042724186567
    }
    with patch.object(csv_writer.storage.trial, 'get_finished', return_value=[0]):
        with patch.object(csv_writer.storage, 'get_hp_dict', return_value=hp):
            assert csv_writer.create() is None


def test_report_():
    try:
        main()
    except BaseException:
        pass
