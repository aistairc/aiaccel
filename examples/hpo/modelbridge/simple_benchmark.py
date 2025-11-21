"""Minimal end-to-end run of the lightweight modelbridge pipeline (train+eval).

Run with ``python examples/hpo/modelbridge/simple_benchmark.py`` to execute
train/eval HPO pairs, fit the bridge regression (macro→micro parameters), and
inspect predictions. Results are written under ``./work/modelbridge/simple``
by default.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from aiaccel.hpo.modelbridge.config import (
    BridgeConfig,
    BridgeSettings,
    ObjectiveConfig,
    ParameterBounds,
    ParameterSpace,
    RegressionConfig,
    ScenarioConfig,
)
from aiaccel.hpo.modelbridge.runner import run_pipeline
from aiaccel.hpo.modelbridge.types import EvaluationResult, TrialContext


def objective(context: TrialContext, env: Mapping[str, str] | None = None) -> EvaluationResult:  # noqa: ARG001
    """Synthetic macro/micro target that stays differentiable and bounded."""

    if context.phase == "macro":
        x = context.params.get("macro_x", 0.0)
        y = context.params.get("macro_y", 0.0)
    else:
        x = context.params.get("micro_x", 0.0)
        y = context.params.get("micro_y", 0.0)

    micro_score = (x - 0.6) ** 2 + (y + 0.3) ** 2
    mae = abs(micro_score - 0.25)
    return EvaluationResult(objective=micro_score, metrics={"mae": mae})


def build_config(base_dir: Path) -> BridgeConfig:
    """Return a :class:`BridgeConfig` matching the simple benchmark (train+eval) setup."""

    scenario = ScenarioConfig(
        name="simple",
        train_macro_trials=12,
        train_micro_trials=12,
        eval_macro_trials=6,
        eval_micro_trials=6,
        train_objective=ObjectiveConfig(target="examples.hpo.modelbridge.simple_benchmark.objective"),
        eval_objective=ObjectiveConfig(target="examples.hpo.modelbridge.simple_benchmark.objective"),
        train_params=ParameterSpace(
            macro={
                "macro_x": ParameterBounds(low=-1.0, high=1.0),
                "macro_y": ParameterBounds(low=-1.0, high=1.0),
            },
            micro={
                "micro_x": ParameterBounds(low=-1.0, high=1.0),
                "micro_y": ParameterBounds(low=-1.0, high=1.0),
            },
        ),
        eval_params=ParameterSpace(
            macro={
                "macro_x": ParameterBounds(low=-1.0, high=1.0),
                "macro_y": ParameterBounds(low=-1.0, high=1.0),
            },
            micro={
                "micro_x": ParameterBounds(low=-1.0, high=1.0),
                "micro_y": ParameterBounds(low=-1.0, high=1.0),
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

    return BridgeConfig(bridge=settings)


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
