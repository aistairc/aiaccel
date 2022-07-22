from aiaccel.master.verification.abstract import\
    AbstractVerification
from aiaccel.util.filesystem import load_yaml
from tests.base_test import BaseTest
import aiaccel
from aiaccel.storage.storage import Storage



class TestAbstractVerification(BaseTest):

    def test_init(self):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'master'
        }
        
        verification = AbstractVerification(options)
        assert verification.is_verified

    def test_verify(self, clean_work_dir, setup_hp_finished, work_dir):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)

        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
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
            # for j in range(2):
            # verification.storage.hp.set_any_trial_param(
            #    trial_id=i,
            #    param_name=f'x{j+1}',
            #    param_value=0.0,
            #    param_type='float'
            # )
            verification.storage.hp.set_any_trial_params(
                trial_id=i,
                params=[
                    {'parameter_name': f'x{j+1}', 'value': 0.0, 'type': 'float'}
                    for j in range(2)
                ]
            )

        verification.verify()
        file_path = work_dir / aiaccel.dict_verification / f'1.{aiaccel.extension_verification}'
        assert file_path.exists()

    def test_make_verification(
        self,
        clean_work_dir,
        setup_hp_finished,
        work_dir
    ):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)

        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
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
        file_path = work_dir / aiaccel.dict_verification / f'5.{aiaccel.extension_verification}'
        assert file_path.exists()
        yml = load_yaml(file_path)
        for y in yml:
            if y['loop'] == 1 or y['loop'] == 5:
                assert y['passed']

    def test_print(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'master'
        }

        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(options)
        verification.is_verified = False
        assert verification.print() is None
        verification.is_verified = True
        assert verification.print() is None

    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'nosave': False,
            'fs': False,
            'process_name': 'master'
        }

        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
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
            aiaccel.dict_verification,
            '1.{}'.format(aiaccel.extension_verification)
        )

        if path.exists():
            path.unlink()

        verification.save(1)
        assert path.exists()
