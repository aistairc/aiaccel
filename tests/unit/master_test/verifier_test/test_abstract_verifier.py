from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from aiaccel.common import dict_verification
from aiaccel.common import extension_verification
from aiaccel.master import AbstractVerifier

from tests.base_test import BaseTest


class TestAbstractVerifier(BaseTest):
    @pytest.fixture(autouse=True)
    def setupt_verifier(
        self,
        request: pytest.FixtureRequest,
        monkeypatch: pytest.MonkeyPatch
    ) -> Generator[None, None, None]:
        if "noautousefixture" in request.keywords:
            yield
        else:
            options = {
                'config': self.config_json,
                'resume': None,
                'clean': False,
                'fs': False,
                'process_name': 'master'
            }
            with monkeypatch.context() as m:
                m.setattr(AbstractVerifier, "__abstractmethods__", set())
                self.verifier = AbstractVerifier(options)
                yield
        self.verifier = None

    @pytest.mark.noautousefixture
    def test_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        with pytest.raises(TypeError):
            _ = AbstractVerifier(options)

        with monkeypatch.context() as m:
            m.setattr(AbstractVerifier, "__abstractmethods__", set())
            verifier = AbstractVerifier(options)
            assert verifier.is_verified

    def test_print(self):
        self.verifier.is_verified = False
        assert self.verifier.print() is None
        self.verifier.is_verified = True
        assert self.verifier.print() is None

    def test_save(self, work_dir: Path) -> None:
        self.verifier.is_verified = False
        assert self.verifier.save(1) is None
        self.verifier.is_verified = True
        # setup_hp_finished(1)

        for i in range(1):
            self.verifier.storage.result.set_any_trial_objective(
                trial_id=i,
                objective=i * 1.0
            )
            for j in range(2):
                self.verifier.storage.hp.set_any_trial_param(
                    trial_id=i,
                    param_name=f"x{j}",
                    param_value=0.0,
                    param_type='float'
                )

        self.verifier.verify()
        path = work_dir / dict_verification / f'1.{extension_verification}'
        if path.exists():
            path.unlink()

        self.verifier.save(1)
        assert path.exists()
