"""Bridge sphere/rastrigin/griewank benchmarks using the modelbridge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    micro: str
    macro: str
    regression: RegressionConfig
    train_trials: int
    eval_trials: int


SCENARIOS = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            name="sphere_to_rastrigin",
            micro="sphere",
            macro="rastrigin",
            regression=RegressionConfig(kind="linear", degree=2),
            train_trials=80,
            eval_trials=40,
        ),
        ScenarioSpec(
            name="rastrigin_to_sphere",
            micro="rastrigin",
            macro="sphere",
            regression=RegressionConfig(kind="linear", degree=1),
            train_trials=100,
            eval_trials=50,
        ),
        ScenarioSpec(
            name="griewank_to_sphere",
            micro="griewank",
            macro="sphere",
            regression=RegressionConfig(kind="linear", degree=2),
            train_trials=120,
            eval_trials=60,
        ),
    )
}


def build_config(base_dir: Path) -> BridgeConfig:
    """Construct the bridge configuration covering all benchmark scenarios."""

    scenario_configs: list[ScenarioConfig] = []
    for spec in SCENARIOS.values():
        train_cmd = [
            "python",
            "examples/hpo/modelbridge/objectives/multi_objective.py",
            "{out_filename}",
            f"--function={spec.macro}",  # macro function
            "--x1={x1}",
            "--x2={x2}",
        ]
        eval_cmd = [
            "python",
            "examples/hpo/modelbridge/objectives/multi_objective.py",
            "{out_filename}",
            f"--function={spec.micro}",  # micro function
            "--x1={x1}",
            "--x2={x2}",
        ]

        params = ParameterSpace(
            macro={
                "x1": ParameterBounds(low=-5.0, high=5.0),
                "x2": ParameterBounds(low=-5.0, high=5.0),
            },
            micro={
                "x1": ParameterBounds(low=-5.0, high=5.0),
                "x2": ParameterBounds(low=-5.0, high=5.0),
            },
        )

        scenario_configs.append(
            ScenarioConfig(
                name=spec.name,
                train_macro_trials=spec.train_trials,
                train_micro_trials=spec.train_trials,
                eval_macro_trials=spec.eval_trials,
                eval_micro_trials=spec.eval_trials,
                train_objective=ObjectiveConfig(command=train_cmd),
                eval_objective=ObjectiveConfig(command=eval_cmd),
                train_params=params,
                eval_params=params,
                regression=spec.regression,
            )
        )

    return BridgeConfig(bridge=BridgeSettings(output_dir=base_dir, scenarios=scenario_configs))
