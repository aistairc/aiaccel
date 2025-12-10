from __future__ import annotations

from typing import Callable

import copy
import pytest
from pathlib import Path


@pytest.fixture
def demo_scenario_dict() -> dict[str, object]:
    return {
        "name": "demo",
        "train_macro_trials": 2,
        "train_micro_trials": 2,
        "eval_macro_trials": 2,
        "eval_micro_trials": 2,
        "train_objective": {
            "target": "tests.hpo.modelbridge.sample_objective.objective",
        },
        "eval_objective": {
            "target": "tests.hpo.modelbridge.sample_objective.objective",
        },
        "train_params": {
            "macro": {"x": {"low": 0, "high": 1}},
            "micro": {"y": {"low": 0, "high": 1}},
        },
        "eval_params": {
            "macro": {"x": {"low": 0, "high": 1}},
            "micro": {"y": {"low": 0, "high": 1}},
        },
    }


@pytest.fixture
def make_bridge_config(demo_scenario_dict: dict[str, object]) -> Callable[[Path | str, int, int], dict[str, object]]:
    def _factory(output_dir: Path | str, train_runs: int = 1, eval_runs: int = 1) -> dict[str, object]:
        return {
            "hpo": {
                "base_config": "dummy_base_config.yaml",
            },
            "bridge": {
                "output_dir": str(output_dir),
                "seed": 5,
                "train_runs": train_runs,
                "eval_runs": eval_runs,
                "scenarios": [copy.deepcopy(demo_scenario_dict)],
            },
        }

    return _factory