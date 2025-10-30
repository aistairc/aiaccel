"""Minimal end-to-end run of the lightweight modelbridge pipeline.

Run this script with ``python examples/hpo/modelbridge/simple_benchmark.py`` to
launch paired macro/micro HPO, fit the bridge regression (macro→micro
parameters), and inspect the generated predictions. Results are written under
``./work/modelbridge/simple`` by default.
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


def objective(context: TrialContext, base_env: Mapping[str, str] | None = None) -> EvaluationResult:
    """Synthetic macro/micro target that stays differentiable and bounded."""

    del base_env  # The example keeps the signature but does not use the value.
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
    """Return a :class:`BridgeConfig` matching the simple benchmark setup."""

    scenario = ScenarioConfig(
        name="simple",
        macro_trials=12,
        micro_trials=12,
        objective=ObjectiveConfig(target="examples.hpo.modelbridge.simple_benchmark.objective"),
        params=ParameterSpace(
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
