"""Bridge sphere/rastrigin/griewank benchmarks using the modelbridge pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    micro: str
    macro: str
    regression: RegressionConfig
    trials: int



Vector = NDArray[np.float64]


def sphere(vec: Vector) -> float:
    """Sphere benchmark."""

    return float(np.sum(vec**2))


def rastrigin(vec: Vector) -> float:
    """Rastrigin benchmark."""

    return float(10 * vec.size + np.sum(vec**2 - 10.0 * np.cos(2 * np.pi * vec)))


def griewank(vec: Vector) -> float:
    """Griewank benchmark."""

    denom = np.sqrt(np.arange(1, vec.size + 1, dtype=float))
    return float(np.sum(vec**2) / 4000.0 - np.prod(np.cos(vec / denom)) + 1.0)


FUNCTIONS: dict[str, Callable[[Vector], float]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "griewank": griewank,
}

SCENARIOS = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            name="sphere_to_rastrigin",
            micro="sphere",
            macro="rastrigin",
            regression=RegressionConfig(kind="linear", degree=2),
            trials=80,
        ),
        ScenarioSpec(
            name="rastrigin_to_sphere",
            micro="rastrigin",
            macro="sphere",
            regression=RegressionConfig(kind="linear", degree=1),
            trials=100,
        ),
        ScenarioSpec(
            name="griewank_to_sphere",
            micro="griewank",
            macro="sphere",
            regression=RegressionConfig(kind="linear", degree=2),
            trials=120,
        ),
    )
}


def benchmark_objective(context: TrialContext, base_env: Mapping[str, str] | None = None) -> EvaluationResult:
    """Evaluate the configured benchmark for the given scenario/phase."""

    del base_env
    spec = SCENARIOS[context.scenario]
    phase = "macro" if context.phase == "macro" else "micro"
    fn_name = spec.macro if phase == "macro" else spec.micro
    fn = FUNCTIONS[fn_name]

    prefix = "macro" if phase == "macro" else "micro"
    vector = np.array([context.params[f"{prefix}_x1"], context.params[f"{prefix}_x2"]], dtype=float)
    score = fn(vector)
    return EvaluationResult(objective=score, metrics={"mae": abs(score)})


def build_config(base_dir: Path) -> BridgeConfig:
    """Construct the bridge configuration covering all benchmark scenarios."""

    scenario_configs: list[ScenarioConfig] = []
    for spec in SCENARIOS.values():
        scenario_configs.append(
            ScenarioConfig(
                name=spec.name,
                macro_trials=spec.trials,
                micro_trials=spec.trials,
                objective=ObjectiveConfig(
                    target="examples.hpo.modelbridge.multi_function_benchmark.benchmark_objective"
                ),
                params=ParameterSpace(
                    macro={
                        "macro_x1": ParameterBounds(low=-5.0, high=5.0),
                        "macro_x2": ParameterBounds(low=-5.0, high=5.0),
                    },
                    micro={
                        "micro_x1": ParameterBounds(low=-5.0, high=5.0),
                        "micro_x2": ParameterBounds(low=-5.0, high=5.0),
                    },
                ),
                regression=spec.regression,
                metrics=("mae",),
            )
        )

    settings = BridgeSettings(
        output_dir=base_dir / "work" / "modelbridge" / "multi_function",
        seed=123,
        scenarios=scenario_configs,
    )

    return BridgeConfig(bridge=settings)


def main() -> None:
    """Run the multi-function benchmark and print summaries."""

    config = build_config(Path.cwd())
    summary = run_pipeline(config)

    print("ModelBridge multi-function benchmark summary:")
    print(json.dumps(summary, indent=2, default=str))

    for scenario in config.bridge.scenarios:
        predictions_path = config.bridge.output_dir / "scenarios" / scenario.name / "predictions.csv"
        if predictions_path.exists():
            print(f"\nPreview of predictions for {scenario.name}:")
            for line in predictions_path.read_text(encoding="utf-8").splitlines()[:5]:
                print(line)


if __name__ == "__main__":
    main()
