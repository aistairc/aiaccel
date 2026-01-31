from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from aiaccel.hpo.modelbridge.config import HpoSettings, ScenarioConfig
from aiaccel.hpo.modelbridge.ops import run_setup_train_step


def test_run_setup_train_step(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    data = make_bridge_config(str(tmp_path / "outputs"))

    # Create a base config file
    base_config_path = tmp_path / "base_config.yaml"
    base_config_path.write_text("optimize:\n  goal: minimize\n", encoding="utf-8")

    hpo_settings = HpoSettings(
        base_config=base_config_path,
        optimize_command=["aiaccel-hpo", "optimize"],
        job_runner_command=["aiaccel-job", "local", "cpu"],
        macro_overrides={},
        micro_overrides={}
    )

    # We need to construct ScenarioConfig objects manually or via validation if not mocking everything
    # But since ops uses config objects, we can just mock the object attributes
    scenario = MagicMock(spec=ScenarioConfig)
    scenario.name = "test_sc"
    scenario.train_macro_trials = 2
    scenario.train_micro_trials = 2

    # Configure the mocked train_params
    train_params_mock = MagicMock()
    train_params_mock.macro = {"x": MagicMock(low=0.0, high=1.0, log=False, step=None)}
    train_params_mock.micro = {"y": MagicMock(low=0.0, high=1.0, log=False, step=None)}
    scenario.train_params = train_params_mock

    output_dir = tmp_path / "outputs" / "test_sc"

    # Call the setup function
    # We should NOT need to mock subprocess because it shouldn't be called.
    # If it is called, the test will likely fail (or hang/do weird things), but better to spy/fail.
    with patch("subprocess.run") as mock_subprocess:
        run_setup_train_step(
            settings=hpo_settings,
            scenario=scenario,
            runs=1,
            seed_base=42,
            scenario_dir=output_dir
        )

        assert not mock_subprocess.called

    # Check artifacts
    # Structure: runs/train/000/macro and micro
    run_dir = output_dir / "runs" / "train" / "000"
    assert (run_dir / "macro" / "config.yaml").exists()
    assert (run_dir / "micro" / "config.yaml").exists()

    # Verify config content
    with open(run_dir / "macro" / "config.yaml") as f:
        conf = yaml.safe_load(f)
        assert conf["n_trials"] == 2
        assert "x" in conf["params"]
        assert conf["working_directory"] == str(run_dir / "macro")
