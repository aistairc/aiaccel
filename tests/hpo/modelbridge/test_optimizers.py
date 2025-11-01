from __future__ import annotations

from pathlib import Path

from aiaccel.hpo.modelbridge.config import ObjectiveConfig, ParameterBounds
from aiaccel.hpo.modelbridge.evaluators import build_evaluator
from aiaccel.hpo.modelbridge.optimizers import run_phase


def test_run_phase_collects_trials(tmp_path: Path) -> None:
    config = ObjectiveConfig(target="tests.hpo.modelbridge.sample_objective.stateless_objective")
    evaluator = build_evaluator(config)
    outcome = run_phase(
        scenario="demo",
        phase="macro",
        trials=2,
        space={"x": ParameterBounds(low=0.0, high=1.0)},
        evaluator=evaluator,
        seed=0,
        output_dir=tmp_path,
    )

    assert len(outcome.trials) == 2
    assert outcome.best_value is not None
    assert "x" in outcome.best_params
