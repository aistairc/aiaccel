from __future__ import annotations

from pathlib import Path

from aiaccel.hpo.modelbridge.config import ObjectiveConfig
from aiaccel.hpo.modelbridge.evaluators import build_evaluator, command_objective
from aiaccel.hpo.modelbridge.types import TrialContext


def _context() -> TrialContext:
    return TrialContext(
        scenario="demo",
        phase="macro",
        trial_index=0,
        params={"x": 0.5},
        seed=42,
        output_dir=Path("."),
    )


def test_build_evaluator_python_callable() -> None:
    config = ObjectiveConfig(target="tests.hpo.modelbridge.sample_objective.objective")
    evaluator = build_evaluator(config, base_env={"FROM": "TEST"})
    result = evaluator(_context())
    assert result.metrics["mae"] == 0.5
    assert result.payload["base_env"] == {"FROM": "TEST"}


def test_command_objective_runs_subprocess() -> None:
    payload = {"objective": 1.23, "metrics": {"mae": 1.23}}
    command = [
        "python",
        "-c",
        ("import json, sys; payload = {'objective': 1.23, 'metrics': {'mae': 1.23}}; json.dump(payload, sys.stdout)"),
    ]
    context = _context()
    result = command_objective(context, command=command, timeout=5.0, base_env={})
    assert result.objective == payload["objective"]
    assert result.metrics == payload["metrics"]
