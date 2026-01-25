from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.pipeline import run_pipeline


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    data = make_bridge_config(str(tmp_path / "outputs"))
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_macro_trials"] = 3
    scenario["train_micro_trials"] = 3
    scenario["eval_macro_trials"] = 3
    scenario["eval_micro_trials"] = 3
    scenario["regression"] = {"degree": 1}
    data["bridge"]["seed"] = 7
    return data


def test_run_pipeline(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))

    with (
        patch("aiaccel.hpo.modelbridge.pipeline.run_hpo_phase") as mock_hpo,
        patch("aiaccel.hpo.modelbridge.pipeline.run_regression") as mock_reg,
        patch("aiaccel.hpo.modelbridge.pipeline.run_evaluation") as mock_eval,
        patch("aiaccel.hpo.modelbridge.pipeline.run_summary") as mock_sum,
        patch("aiaccel.hpo.modelbridge.pipeline.run_external_command") as _,
    ):
        manifest = run_pipeline(bridge_config)

        # Verify calls
        assert mock_hpo.called
        assert mock_reg.called
        assert mock_eval.called
        assert mock_sum.called

        # Manifest should be populated
        artifact_paths = [str(a["path"]) for a in manifest["artifacts"]]
        # manifest.json is written after collection, so it might not be in the list returned.
        # But log file should be there.
        assert any("aiaccel_modelbridge.log" in p for p in artifact_paths)
        assert manifest["scenarios"]["demo"]["status"] == "completed"
