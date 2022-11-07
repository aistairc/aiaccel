from aiaccel.util.report import CreationReport
from aiaccel.workspace import Workspace
from aiaccel.config import Config
import pathlib
from unittest.mock import patch
from aiaccel.cli.report import main

ws = Workspace("test_work")
config_path = pathlib.Path('tests/test_data/config.json')

def test_report():
    # config = Config(config_path)
    report = CreationReport(config_path)

    assert report.get_zero_padding_trial_id(1) == '000001'
    assert report.create() is None

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
    with patch.object(report.storage.trial, 'get_finished', return_value = [0]):
        with patch.object(report.storage, 'get_hp_dict', return_value = hp):
            assert report.create() is None

def test_report_():
    try:
        main()
    except:
        pass
