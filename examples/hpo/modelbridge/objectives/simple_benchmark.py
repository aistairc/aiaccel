"""Minimal modelbridge benchmark for prepare/external/analyze flow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from aiaccel.hpo.modelbridge.config import (
    BridgeConfig,
    BridgeSettings,
    HpoSettings,
    ObjectiveConfig,
    ParameterBounds,
    ParameterSpace,
    RegressionConfig,
    ScenarioConfig,
)
from aiaccel.hpo.modelbridge.execution import emit_commands
from aiaccel.hpo.modelbridge.pipeline import run_pipeline
from aiaccel.hpo.modelbridge.toolkit.io import read_json


def build_config(base_dir: Path) -> BridgeConfig:
    """Return a BridgeConfig matching the simple benchmark scenario."""
    objective = ["python", "examples/hpo/modelbridge/simple_objective.py", "{out_filename}", "--x={x}", "--y={y}"]
    scenario = ScenarioConfig(
        name="simple",
        train_macro_trials=12,
        train_micro_trials=12,
        eval_macro_trials=6,
        eval_micro_trials=6,
        train_objective=ObjectiveConfig(command=objective),
        eval_objective=ObjectiveConfig(command=objective),
        train_params=ParameterSpace(
            macro={
                "x": ParameterBounds(low=-1.0, high=1.0),
                "y": ParameterBounds(low=-1.0, high=1.0),
            },
            micro={
                "x": ParameterBounds(low=-1.0, high=1.0),
                "y": ParameterBounds(low=-1.0, high=1.0),
            },
        ),
        eval_params=ParameterSpace(
            macro={
                "x": ParameterBounds(low=-1.0, high=1.0),
                "y": ParameterBounds(low=-1.0, high=1.0),
            },
            micro={
                "x": ParameterBounds(low=-1.0, high=1.0),
                "y": ParameterBounds(low=-1.0, high=1.0),
            },
        ),
        regression=RegressionConfig(kind="linear", degree=1),
        metrics=["mae"],
    )

    return BridgeConfig(
        bridge=BridgeSettings(
            output_dir=base_dir / "work" / "modelbridge" / scenario.name,
            seed=32,
            train_runs=1,
            eval_runs=1,
            scenarios=[scenario],
        ),
        hpo=HpoSettings(
            base_config=base_dir / "examples" / "hpo" / "modelbridge" / "optimize_config.yaml",
        ),
    )


def main() -> None:
    """Run prepare, external optimize commands, and analyze profile."""
    config = build_config(Path.cwd())

    run_pipeline(config, profile="prepare")
    train_cmd_json = emit_commands(config, role="train", fmt="json")
    eval_cmd_json = emit_commands(config, role="eval", fmt="json")

    _run_emitted_commands(train_cmd_json)
    _run_emitted_commands(eval_cmd_json)

    result = run_pipeline(config, profile="analyze")
    print(json.dumps(result.to_dict(), indent=2, default=str))

    scenario_dir = config.bridge.output_dir / config.bridge.scenarios[0].name
    predictions_path = scenario_dir / "test_predictions.csv"
    if predictions_path.exists():
        print(f"\nSample predictions from {predictions_path}:")
        for line in predictions_path.read_text(encoding="utf-8").splitlines()[:5]:
            print(line)
    else:
        print("\nNo test_predictions.csv found.")


def _run_emitted_commands(path: Path) -> None:
    payload = read_json(path)
    commands = payload.get("commands", []) if isinstance(payload, dict) else []
    for item in commands:
        command = item.get("command", [])
        if not isinstance(command, list):
            raise RuntimeError(f"Malformed command entry in {path}")
        subprocess.run([str(token) for token in command], check=True)


if __name__ == "__main__":
    main()
