"""Scenario planning helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import ParameterBounds, ScenarioConfig

ParamDict = dict[str, ParameterBounds]


@dataclass
class PhasePlan:
    """Plan describing a single optimisation phase."""

    name: str
    phase: str
    trials: int
    space: ParamDict


@dataclass
class ScenarioPlan:
    """Container mapping a scenario to its macro/micro plans."""

    config: ScenarioConfig
    macro: PhasePlan
    micro: PhasePlan


def build_plan(config: ScenarioConfig) -> ScenarioPlan:
    """Create a :class:`ScenarioPlan` from ``config``."""

    macro = PhasePlan(
        name=f"{config.name}-macro",
        phase="macro",
        trials=config.macro_trials,
        space=config.params.macro,
    )
    micro = PhasePlan(
        name=f"{config.name}-micro",
        phase="micro",
        trials=config.micro_trials,
        space=config.params.micro,
    )
    return ScenarioPlan(config=config, macro=macro, micro=micro)


__all__ = ["PhasePlan", "ScenarioPlan", "build_plan"]
