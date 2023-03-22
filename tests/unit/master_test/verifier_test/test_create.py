from __future__ import annotations

from aiaccel.master import create_verifier
from aiaccel.master import SingleObjectiveVerifier
from aiaccel.master import MultiObjectiveVerifier


def test_create_verifier() -> None:
    config_single_objective = "tests/test_data/config.json"
    assert create_verifier(config_single_objective) == SingleObjectiveVerifier

    config_multi_objective = "tests/test_data/config_motpe.json"
    assert create_verifier(config_multi_objective) == MultiObjectiveVerifier
