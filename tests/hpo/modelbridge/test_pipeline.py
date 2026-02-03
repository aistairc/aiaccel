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
        patch("aiaccel.hpo.modelbridge.pipeline.run_setup_train_step") as mock_setup_train,
        patch("aiaccel.hpo.modelbridge.pipeline.run_setup_eval_step") as mock_setup_eval,
        patch("aiaccel.hpo.modelbridge.pipeline.run_regression_step") as mock_reg,
        patch("aiaccel.hpo.modelbridge.pipeline.run_evaluate_model_step") as mock_eval_model,
        patch("aiaccel.hpo.modelbridge.pipeline.run_summary_step") as mock_sum,
        patch("aiaccel.hpo.modelbridge.pipeline.run_da_step") as _mock_da,
    ):
        manifest = run_pipeline(bridge_config)

        # Verify calls
        assert mock_setup_train.called
        assert mock_setup_eval.called
        assert mock_reg.called
        assert mock_eval_model.called
        assert mock_sum.called

        assert isinstance(manifest, dict)


def test_run_pipeline_steps(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))

    with (
        patch("aiaccel.hpo.modelbridge.pipeline.run_setup_train_step") as mock_setup_train,
        patch("aiaccel.hpo.modelbridge.pipeline.run_setup_eval_step") as mock_setup_eval,
        patch("aiaccel.hpo.modelbridge.pipeline.run_regression_step") as mock_reg,
        patch("aiaccel.hpo.modelbridge.pipeline.run_evaluate_model_step") as mock_eval_model,
        patch("aiaccel.hpo.modelbridge.pipeline.run_summary_step") as mock_sum,
    ):
        # Run only setup_train and regression
        run_pipeline(bridge_config, steps=["setup_train", "regression"])

        assert mock_setup_train.called
        assert mock_reg.called
        assert not mock_setup_eval.called
        assert not mock_eval_model.called
        assert not mock_sum.called
