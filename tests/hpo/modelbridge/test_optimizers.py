from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

from omegaconf import OmegaConf

from aiaccel.hpo.modelbridge.config import HpoSettings, ParameterBounds
from aiaccel.hpo.modelbridge.optimizers import run_hpo


def test_run_hpo_success(tmp_path: Path) -> None:
    # Setup dummy base config
    base_config = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create({"optimize": {"goal": "min"}}), base_config)

    hpo_settings = HpoSettings(base_config=base_config)
    output_dir = tmp_path / "output"
    
    mock_study = MagicMock()
    mock_study.trials = []
    mock_study.best_trial = None
    
    with patch("subprocess.run") as mock_run, \
         patch("optuna.load_study", return_value=mock_study) as mock_load:
        
        outcome = run_hpo(
            hpo_settings=hpo_settings,
            scenario="demo",
            phase="macro",
            trials=2,
            space={"x": ParameterBounds(low=0.0, high=1.0)},
            seed=0,
            output_dir=output_dir,
        )
        
        assert mock_run.called
        assert mock_load.called
        assert outcome.study == mock_study
        
        # Verify config injection
        temp_conf_path = output_dir / "optimize_config.yaml"
        assert temp_conf_path.exists()
        loaded_conf = OmegaConf.load(temp_conf_path)
        assert loaded_conf.n_trials == 2
        # Check injected params structure (HparamsManager)
        assert loaded_conf.params.x._target_ == "aiaccel.hpo.optuna.hparams.Float"
        assert loaded_conf.params.x.low == 0.0
