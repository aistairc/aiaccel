from __future__ import annotations

from collections.abc import Generator

import pytest

from aiaccel.master import MultiObjectiveVerifier

from tests.base_test import BaseTest


class TestMultiObjectiveVerifier(BaseTest):
    @pytest.fixture(autouse=True)
    def setupt_verifier(self) -> Generator[None, None, None]:
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        self.verifier = MultiObjectiveVerifier(options)
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
        verifier = MultiObjectiveVerifier(options)
        assert verifier.is_verified

    def test_verify(self) -> None:
        pass
