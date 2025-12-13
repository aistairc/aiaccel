"""Bridge sphere/rastrigin/griewank benchmarks using the modelbridge pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aiaccel.hpo.modelbridge.config import (
    BridgeConfig,
    BridgeSettings,
    RegressionConfig,
    ScenarioConfig,
)
from aiaccel.hpo.modelbridge.runner import run_pipeline


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
            "python", "examples/hpo/modelbridge/multi_objective.py",
            "{out_filename}",
            f"--function={spec.macro}", # macro function
            "--x1={x1}", "--x2={x2}"
        ]
        pass

    return BridgeConfig(bridge=BridgeSettings(output_dir=base_dir, scenarios=[])) # Dummy
