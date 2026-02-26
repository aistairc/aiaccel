"""Minimal end-to-end benchmark for the rev04 modelbridge flow."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from aiaccel.hpo.modelbridge import api
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
    """Run prepare, HPO, analyze, and publish steps."""
    root_dir = Path(__file__).resolve().parents[4]
    config = build_config(root_dir)

    steps = [
        api.prepare_train_step(config),
        api.prepare_eval_step(config),
        api.hpo_train_step(config),
        api.hpo_eval_step(config),
        api.collect_train_step(config),
        api.collect_eval_step(config),
        api.fit_regression_step(config),
        api.evaluate_model_step(config),
        api.publish_summary_step(config),
    ]
    print(json.dumps({"results": [asdict(step) for step in steps]}, indent=2, default=str))

    scenario_dir = config.bridge.output_dir / config.bridge.scenarios[0].name
    predictions_path = scenario_dir / "test_predictions.csv"
    if predictions_path.exists():
        print(f"\nSample predictions from {predictions_path}:")
        for line in predictions_path.read_text(encoding="utf-8").splitlines()[:5]:
            print(line)
    else:
        print("\nNo test_predictions.csv found.")


if __name__ == "__main__":
    main()
