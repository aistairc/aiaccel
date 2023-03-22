from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from aiaccel.common import dict_verification
from aiaccel.common import extension_verification
from aiaccel.master import SingleObjectiveVerifier
from aiaccel.util import load_yaml

from tests.base_test import BaseTest


class TestSingleObjectiveVerifier(BaseTest):
    @pytest.fixture(autouse=True)
    def setupt_verifier(self) -> Generator[None, None, None]:
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        self.verifier = SingleObjectiveVerifier(options)
        yield
        self.verifier = None

    def test_init(self) -> None:
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        verifier = SingleObjectiveVerifier(options)
        assert verifier.is_verified

    def test_verify(self, monkeypatch: pytest.MonkeyPatch, work_dir: Path) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.verifier, 'is_verified', False)
            assert self.verifier.verify() is None

            m.setattr(self.verifier, 'is_verified', True)

            m.setattr(self.verifier, '_is_loop_verifiable', lambda _: False)
            m.setattr(self.verifier, '_verified_loops', dummy_verified_loops := [])
            assert self.verifier.verify() is None
            assert len(dummy_verified_loops) == 0

            m.setattr(self.verifier, '_is_loop_verifiable', lambda _: True)

            m.setattr(self.verifier, '_is_loop_verified', lambda _: True)
            m.setattr(self.verifier, '_verified_loops', dummy_verified_loops := [])
            assert self.verifier.verify() is None
            assert len(dummy_verified_loops) == 0

            m.setattr(self.verifier, '_is_loop_verified', lambda _: False)

            m.setattr(self.verifier, '_find_best_objective_before_target_loop', lambda *_: 0)
            m.setattr(self.verifier, '_make_verification', lambda *_: 'verified')
            m.setattr(self.verifier, '_verified_loops', dummy_verified_loops := [])
            assert self.verifier.verify() is None
            assert len(dummy_verified_loops) == len(self.verifier.condition)

            m.setattr(self.verifier, '_make_verification', lambda *_: '')
            m.setattr(self.verifier, '_verified_loops', dummy_verified_loops := [])
            assert self.verifier.verify() is None
            assert len(dummy_verified_loops) == 0

        with monkeypatch.context() as m:
            for trial_id in range(10):
                self.verifier.storage.result.set_any_trial_objective(
                    trial_id=trial_id,
                    objective=trial_id * 1.0
                )
                self.verifier.storage.trial.set_any_trial_state(
                    trial_id=trial_id,
                    state='finished'
                )
                for j in range(1, 3):
                    self.verifier.storage.hp.set_any_trial_param(
                        trial_id=trial_id,
                        param_name=f'x{j}',
                        param_value=0.0,
                        param_type='float'
                    )
            self.verifier.verify()
            file_path = work_dir / dict_verification / f'5.{extension_verification}'
            assert file_path.exists()
            yml = load_yaml(file_path)
            for y in yml:
                if y['loop'] == 1 or y['loop'] == 5:
                    assert y['passed']

        with monkeypatch.context() as m:
            m.setattr(
                self.verifier, '_verified_loops', dummy_verified_loops := []
            )
            m.setattr(
                self.verifier.storage, 'get_finished',
                lambda: [5, 6, 7, 8, 9]
            )

            m.setattr(
                self.verifier.storage.result, 'get_any_trial_objective',
                lambda x: [None, None, None, None, None, 0, 0, 0, 0, 0][x]
            )
            assert self.verifier.verify() is None
            assert dummy_verified_loops == [5]

            with monkeypatch.context() as m:
                m.setattr(
                    self.verifier, '_verified_loops', dummy_verified_loops := []
                )
            m.setattr(
                self.verifier.storage, 'get_finished',
                lambda: [5, 6, 7, 8, 9]
            )
            m.setattr(
                self.verifier.storage.result, 'get_any_trial_objective',
                lambda x: [None, None, None, None, None, 65, 65, 65, 65, 65][x]
            )
            assert self.verifier.verify() is None
            assert dummy_verified_loops == []

    def test_is_loop_verifiable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.verifier.config.trial_number, 'get', lambda: 1)
            assert self.verifier._is_loop_verifiable(0) is True
            assert self.verifier._is_loop_verifiable(2) is False

    def test_is_loop_verified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.verifier, '_verified_loops', [0])
            assert self.verifier._is_loop_verified(0) is True
            assert self.verifier._is_loop_verified(1) is False

    def test_find_best_objective_before_target_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(self.verifier.storage.result, 'get_any_trial_objective', lambda _: 0)
            current_best = self.verifier._find_best_objective_before_target_loop([1], 0)
            assert current_best == self.verifier._current_best_start
            assert self.verifier._verified_trial_ids == []

            current_best = self.verifier._find_best_objective_before_target_loop([0], 0)
            assert current_best == 0
            assert self.verifier._verified_trial_ids == [0]

    def test_make_verification(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with monkeypatch.context() as m:
            m.setattr(
                self.verifier,
                'verification_result',
                dummy_verification_result := [{}] * len(self.verifier.condition)
            )
            assert self.verifier._make_verification(0, 0) == 'verified'
            assert dummy_verification_result[0]['passed'] is True

        with monkeypatch.context() as m:
            m.setattr(self.verifier, '_verified_trial_ids', [0, 1])
            m.setattr(
                self.verifier,
                'verification_result',
                dummy_verification_result := [{}] * len(self.verifier.condition)
            )
            assert self.verifier._make_verification(-1, 0) == 'verified'
            assert dummy_verification_result[0]['passed'] is False

        with monkeypatch.context() as m:
            m.setattr(self.verifier, '_verified_trial_ids', [])
            m.setattr(
                self.verifier,
                'verification_result',
                dummy_verification_result := [{}] * len(self.verifier.condition)
            )
            assert self.verifier._make_verification(-1, 0) == ''
            assert hasattr(dummy_verification_result, 'passed') is False
