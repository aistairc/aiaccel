from __future__ import annotations

from pathlib import Path

import yaml

from aiaccel.hpo.apps.modelbridge import main as cli_main


def test_data_assimilation_mock(tmp_path: Path) -> None:
    cfg = {
        "hpo": {"optimizer": "optuna", "sampler": "tpe"},
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "train_runs": 0,
            "eval_runs": 0,
            "scenarios": [],
        },
        "data_assimilation": {
            "output_root": str(tmp_path / "da_outputs"),
            "micro_model": "microDemo",
            "macro_model": "macroDemo",
            "scenarios": 2,
            "allow_mock": True,
            "agent_sizes": {"naive": 1, "rational": 0, "ruby": 0},
            "trials": {"micro": 2, "macro_train": 2, "macro_test": 2},
            "seeds": {"micro": 0, "macro_train": 0, "macro_test": 0},
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main(
        [
            "data-assimilation",
            "--config",
            str(cfg_path),
            "--quiet",
            "--no-log",
        ]
    )

    summary_path = tmp_path / "da_outputs" / "data_assimilation_summary.json"
    manifest_path = tmp_path / "da_outputs" / "data_assimilation_manifest.json"
    assert summary_path.exists()
    assert manifest_path.exists()
