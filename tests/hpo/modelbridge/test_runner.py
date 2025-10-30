from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.runner import run_pipeline


def _config(tmp_path: Path) -> Dict[str, object]:
    return {
        "hpo": {"optimizer": "optuna", "sampler": "tpe"},
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "seed": 7,
            "scenarios": [
                {
                    "name": "demo",
                    "macro_trials": 3,
                    "micro_trials": 3,
                    "objective": {
                        "target": "tests.hpo.modelbridge.sample_objective.objective",
                    },
                    "params": {
                        "macro": {"x": {"low": 0.0, "high": 1.0}},
                        "micro": {"y": {"low": 0.0, "high": 1.0}},
                    },
                    "regression": {"degree": 1},
                }
            ],
        },
    }


def test_run_pipeline(tmp_path: Path) -> None:
    bridge_config = load_bridge_config(_config(tmp_path))
    summary = run_pipeline(bridge_config)

    summary_path = tmp_path / "outputs" / "summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_file: Dict[str, Any] = json.load(handle)
    assert summary_file == summary

    scenario_summary = summary["demo"]
    assert scenario_summary["macro_trials"] == 3
    assert scenario_summary["micro_trials"] == 3
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    macro_csv = scenario_dir / "macro_trials.csv"
    micro_csv = scenario_dir / "micro_trials.csv"
    predictions_csv = scenario_dir / "predictions.csv"
    assert macro_csv.exists() and micro_csv.exists()
    macro_contents = macro_csv.read_text(encoding="utf-8")
    assert "objective" in macro_contents and "x" in macro_contents
    micro_contents = micro_csv.read_text(encoding="utf-8")
    assert "metric_mae" in micro_contents
    assert predictions_csv.exists()
    predictions_content = predictions_csv.read_text(encoding="utf-8")
    assert "actual_y" in predictions_content and "pred_y" in predictions_content
