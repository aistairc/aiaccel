from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.runner import execute_pipeline, plan_pipeline, run_pipeline
from aiaccel.hpo.modelbridge.optimizers import PhaseOutcome
from aiaccel.hpo.modelbridge.types import TrialResult, TrialContext, EvaluationResult


def _config(tmp_path: Path, make_bridge_config) -> dict[str, object]:
    data = make_bridge_config(tmp_path / "outputs")
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_macro_trials"] = 3
    scenario["train_micro_trials"] = 3
    scenario["eval_macro_trials"] = 3
    scenario["eval_micro_trials"] = 3
    scenario["regression"] = {"degree": 1}
    data["bridge"]["seed"] = 7
    return data


def test_run_pipeline_dry_run(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    plan = plan_pipeline(bridge_config)
    payload = execute_pipeline(plan, dry_run=True)

    # 4 HPO phases + Regress + Evaluate + Summary + DataAssimilation = 8 contexts
    assert len(payload["contexts"]) == 8
    hpo_contexts = [ctx for ctx in payload["contexts"] if ctx["phase"] == "hpo"]
    assert any(ctx["role"] == "train" and ctx["target"] == "macro" for ctx in hpo_contexts)
    assert not (tmp_path / "outputs").exists()


def test_run_pipeline(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    
    # Mock run_hpo to return fake trials
    mock_trials = [
        TrialResult(
            context=MagicMock(params={"x": 0.5, "y": 0.5}),
            evaluation=EvaluationResult(objective=0.1, metrics={"mae": 0.1}, payload={}),
            state="COMPLETE"
        )
    ]
    mock_outcome = PhaseOutcome(study=MagicMock(), trials=mock_trials)
    
    # We also need to mock _load_best_trials_from_storage because RegressionRunner calls it
    # and it won't find DBs created by our mock run_hpo.
    with patch("aiaccel.hpo.modelbridge.runner.run_hpo", return_value=mock_outcome), \
         patch("aiaccel.hpo.modelbridge.runner._load_best_trials_from_storage", return_value={0: mock_trials[0]}):
        summary = run_pipeline(bridge_config)

    summary_path = tmp_path / "outputs" / "summary.json"
    assert summary_path.exists()
    
    # Check artifacts existence (json files)
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    assert (scenario_dir / "regression_train.json").exists()


def test_run_pipeline_partial_phases(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    
    mock_trials = [
        TrialResult(
            context=MagicMock(params={"x": 0.5, "y": 0.5}),
            evaluation=EvaluationResult(objective=0.1, metrics={"mae": 0.1}, payload={}),
            state="COMPLETE"
        )
    ]
    mock_outcome = PhaseOutcome(study=MagicMock(), trials=mock_trials)
    
    with patch("aiaccel.hpo.modelbridge.runner.run_hpo", return_value=mock_outcome):
        plan = plan_pipeline(bridge_config, phases=("hpo",))
        partial = execute_pipeline(plan)
        assert "contexts" in partial
        
        # Now run regress phase. Needs mocked storage load.
        with patch("aiaccel.hpo.modelbridge.runner._load_best_trials_from_storage", return_value={0: mock_trials[0]}):
            execute_pipeline(plan_pipeline(bridge_config, phases=("regress",)))
            
        assert (scenario_dir / "regression_train.json").exists()