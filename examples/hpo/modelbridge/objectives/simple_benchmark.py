"""Minimal end-to-end benchmark for the rev04 modelbridge flow."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

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


def build_config(base_dir: Path) -> BridgeConfig:
    """Build a small in-process benchmark config.

    Args:
        base_dir: Repository root directory.

    Returns:
        BridgeConfig: Ready-to-run benchmark configuration.
    """
    objective = [
        "python3",
        str(base_dir / "examples" / "hpo" / "modelbridge" / "objectives" / "simple_objective.py"),
        "{out_filename}",
        "--x={x}",
        "--y={y}",
    ]
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
    """Run prepare, external execution, and analyze/publish."""
    root_dir = Path(__file__).resolve().parents[4]
    config = build_config(root_dir)

    run_pipeline(config, profile="prepare")
    train_cmd_json = emit_commands(config, role="train", fmt="json")
    eval_cmd_json = emit_commands(config, role="eval", fmt="json")

    _run_emitted_commands(train_cmd_json)
    _run_emitted_commands(eval_cmd_json)

    result = run_pipeline(config, profile="analyze")
    print(json.dumps(_pipeline_result_to_dict(result), indent=2, default=str))

    scenario_dir = config.bridge.output_dir / config.bridge.scenarios[0].name
    predictions_path = scenario_dir / "test_predictions.csv"
    if predictions_path.exists():
        print(f"\nSample predictions from {predictions_path}:")
        for line in predictions_path.read_text(encoding="utf-8").splitlines()[:5]:
            print(line)
    else:
        print("\nNo test_predictions.csv found.")


def _run_emitted_commands(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    commands = payload.get("commands", []) if isinstance(payload, dict) else []
    for item in commands:
        command = item.get("command", [])
        if not isinstance(command, list):
            raise RuntimeError(f"Malformed command entry in {path}")
        subprocess.run([str(token) for token in command], check=True)


def _pipeline_result_to_dict(result: object) -> dict[str, object]:
    payload = asdict(result)
    payload["results"] = [item["step"] for item in payload.get("results", [])]
    return payload


if __name__ == "__main__":
    main()
