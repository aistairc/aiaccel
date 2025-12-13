"""Minimal end-to-end run of the lightweight modelbridge pipeline (train+eval).

Run with ``python examples/hpo/modelbridge/simple_benchmark.py`` to execute
train/eval HPO pairs, fit the bridge regression (macro→micro parameters), and
inspect predictions. Results are written under ``./work/modelbridge/simple``
by default.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from aiaccel.hpo.modelbridge.runner import run_pipeline


def build_config(base_dir: Path) -> BridgeConfig:
    """Return a :class:`BridgeConfig` matching the simple benchmark (train+eval) setup."""

    command = ["python", "examples/hpo/modelbridge/simple_objective.py", "{out_filename}", "--x={x}", "--y={y}"]

    scenario = ScenarioConfig(
        name="simple",
        train_macro_trials=12,
        train_micro_trials=12,
        eval_macro_trials=6,
        eval_micro_trials=6,
        train_objective=ObjectiveConfig(command=command),
        eval_objective=ObjectiveConfig(command=command),
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
        metrics=("mae",),
    )

    settings = BridgeSettings(
        output_dir=base_dir / "work" / "modelbridge" / scenario.name,
        seed=32,
        train_runs=1,
        eval_runs=1,
        scenarios=[scenario],
    )

    base_config_path = base_dir / "examples" / "hpo" / "modelbridge" / "optimize_config.yaml"
    hpo_settings = HpoSettings(base_config=base_config_path)

    return BridgeConfig(bridge=settings, hpo=hpo_settings)


def main() -> None:
    """Build the configuration, execute the pipeline, and print a summary."""

    config = build_config(Path.cwd())
    summary = run_pipeline(config)
    print("Pipeline summary:")
    print(json.dumps(summary, indent=2, default=str))

    scenario_dir = config.bridge.output_dir / "scenarios" / config.bridge.scenarios[0].name
    predictions_path = scenario_dir / "predictions.csv"
    if predictions_path.exists():
        print(f"\nSample predictions from {predictions_path}:")
        preview = predictions_path.read_text(encoding="utf-8").splitlines()[:5]
        for line in preview:
            print(line)
    else:
        print("\nNo predictions.csv found (check configuration).")


if __name__ == "__main__":
    main()