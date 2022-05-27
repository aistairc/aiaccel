from aiaccel.master.verification.abstract_verification import\
    AbstractVerification
from aiaccel.util.filesystem import load_yaml
from tests.base_test import BaseTest
import aiaccel


class TestAbstractVerification(BaseTest):

    def test_init(self):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(self.config_json)
        assert verification.is_verified

    def test_verify(self, clean_work_dir, setup_hp_finished, work_dir):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(self.config_json)
        verification.is_verified = False
        assert verification.verify() is None
        verification.is_verified = True
        setup_hp_finished(1)
        verification.verify()
        assert work_dir.joinpath(
            aiaccel.dict_verification,
            '1.{}'.format(aiaccel.extension_verification)).exists()

    def test_make_verification(
        self,
        clean_work_dir,
        setup_hp_finished,
        work_dir
    ):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(self.config_json)
        setup_hp_finished(5)
        verification.verify()
        yml = load_yaml(
            work_dir.joinpath(
                aiaccel.dict_verification,
                '5.{}'.format(aiaccel.extension_verification)
            )
        )

        for y in yml:
            if y['loop'] == 1 or y['loop'] == 5:
                assert y['passed']

    def test_print(self):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(self.config_json)
        verification.is_verified = False
        assert verification.print() is None
        verification.is_verified = True
        assert verification.print() is None

    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        # verification = AbstractVerification(self.config)
        # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
        verification = AbstractVerification(self.config_json)
        verification.is_verified = False
        assert verification.save(1) is None
        verification.is_verified = True
        setup_hp_finished(1)
        verification.verify()
        path = work_dir.joinpath(
            aiaccel.dict_verification,
            '1.{}'.format(aiaccel.extension_verification)
        )

        if path.exists():
            path.unlink()

        verification.save(1)
        assert path.exists()
