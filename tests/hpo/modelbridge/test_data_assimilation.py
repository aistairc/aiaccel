from __future__ import annotations

import json
from pathlib import Path

import yaml

from aiaccel.hpo.apps.modelbridge import main as cli_main


def test_data_assimilation_generic_command(tmp_path: Path) -> None:
    # Create a dummy script that writes a summary file
    output_root = tmp_path / "da_outputs"
    output_root.mkdir()
    summary_file = output_root / "data_assimilation_summary.json"

    # We use a simple python one-liner as the command
    cmd = [
        "python",
        "-c",
        f"import json; print('Running DA'); open('{summary_file}', 'w').write(json.dumps({{'status': 'ok'}}))",
    ]

    cfg = {
        "hpo": {},
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "train_runs": 0,
            "eval_runs": 0,
            "scenarios": [],
        },
        "data_assimilation": {
            "enabled": True,
            "command": cmd,
            "cwd": str(tmp_path),
            "output_root": str(output_root),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--quiet",
            "--no-log",
        ]
    )

    assert summary_file.exists()
    manifest_path = output_root / "data_assimilation_manifest.json"
    assert manifest_path.exists()

    with summary_file.open() as f:
        data = json.load(f)
    assert data["status"] == "ok"


def test_data_assimilation_disabled(tmp_path: Path, capsys: object) -> None:
    cfg = {
        "hpo": {},
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "scenarios": [],
        },
        "data_assimilation": {
            "enabled": False,
            "command": "echo 'should not run'",
            "output_root": str(tmp_path / "da_outputs"),
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cli_main(["run", "--config", str(cfg_path), "--quiet", "--no-log"])

    assert not (tmp_path / "da_outputs").exists()
