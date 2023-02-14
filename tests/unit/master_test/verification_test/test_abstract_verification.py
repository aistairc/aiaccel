from unittest.mock import patch

from aiaccel.common import dict_verification
from aiaccel.common import extension_verification
from aiaccel.master.verification.abstract_verification import \
    AbstractVerification
from aiaccel.util import load_yaml
from tests.base_test import BaseTest


class TestAbstractVerification(BaseTest):

    def test_init(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }

        verification = AbstractVerification(options)
        assert verification.is_verified

    def test_verify(self, setup_hp_finished, work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }

        verification = AbstractVerification(options)
        verification.is_verified = False
        assert verification.verify() is None
        verification.is_verified = True
        setup_hp_finished(10)

        for i in range(10):
            verification.storage.result.set_any_trial_objective(
                trial_id=i,
                objective=(i * 1.0)
            )
            verification.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            verification.storage.hp.set_any_trial_params(
                trial_id=i,
                params=[
                    {'parameter_name': f'x{j+1}', 'value': 0.0, 'type': 'float'}
                    for j in range(2)
                ]
            )

        verification.verify()
        file_path = work_dir / dict_verification / f'1.{extension_verification}'
        assert file_path.exists()

        with patch.object(verification, 'finished_loop', None):
            assert verification.verify() is None

        with patch.object(verification, 'finished_loop', 65535):
            assert verification.verify() is None

    def test_make_verification(
        self,
        setup_hp_finished,
        work_dir
    ):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }

        verification = AbstractVerification(options)
        setup_hp_finished(10)

        for i in range(10):
            verification.storage.result.set_any_trial_objective(
                trial_id=i,
                objective=(i * 1.0)
            )
            verification.storage.trial.set_any_trial_state(trial_id=i, state='finished')
            for j in range(2):
                verification.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f'x{j+1}',
                    param_value=0.0,
                    param_type='float'
                )

        verification.verify()
        file_path = work_dir / dict_verification / f'5.{extension_verification}'
        assert file_path.exists()
        yml = load_yaml(file_path)
        for y in yml:
            if y['loop'] == 1 or y['loop'] == 5:
                assert y['passed']

        d0 = {
            'result': float('-inf')
        }
        d1 = {
            'result': 0
        }
        with patch.object(verification.storage, 'get_best_trial_dict', return_value=d0):
            with patch.object(verification.storage, 'get_num_finished', return_value=1):
                assert verification.make_verification(0, 0) is None
        with patch.object(verification.storage, 'get_best_trial_dict', return_value=d1):
            with patch.object(verification.storage, 'get_num_finished', return_value=1):
                assert verification.make_verification(0, 0) is None

    def test_print(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        verification = AbstractVerification(options)
        verification.is_verified = False
        assert verification.print() is None
        verification.is_verified = True
        assert verification.print() is None

    def test_save(self, work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        verification = AbstractVerification(options)
        verification.is_verified = False
        assert verification.save(1) is None
        verification.is_verified = True
        # setup_hp_finished(1)

        for i in range(1):
            verification.storage.result.set_any_trial_objective(
                trial_id=i,
                objective=i * 1.0

            )
            for j in range(2):
                verification.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f"x{j}",
                    param_value=0.0,
                    param_type='float'
                )

        verification.verify()
        path = work_dir.joinpath(
            dict_verification,
            f'1.{extension_verification}'
        )

        if path.exists():
            path.unlink()

        verification.save(1)
        assert path.exists()
