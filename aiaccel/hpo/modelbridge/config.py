"""Configuration dataclasses and helpers for the redesigned modelbridge pipeline."""

from __future__ import annotations

from typing import Any

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .exceptions import ValidationError


@dataclass
class ParameterBounds:
    """Numeric parameter definition for Optuna suggestions."""

    low: float
    high: float
    step: float | None = None
    log: bool = False


@dataclass
class ParameterSpace:
    """Collection of macro/micro parameter bounds."""

    macro: dict[str, ParameterBounds] = field(default_factory=dict)
    micro: dict[str, ParameterBounds] = field(default_factory=dict)


@dataclass
class ObjectiveConfig:
    """Definition of the evaluation entry point."""

    target: str
    command: list[str] | None = None
    timeout: float | None = None


@dataclass
class RegressionConfig:
    """Regression strategy configuration."""

    kind: str = "linear"
    degree: int = 1


@dataclass
class ScenarioConfig:
    """Configuration for a single bridging scenario."""

    name: str
    macro_trials: int
    micro_trials: int
    objective: ObjectiveConfig
    params: ParameterSpace
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    metrics: Sequence[str] = field(default_factory=lambda: ("mae", "mse", "r2"))


@dataclass
class BridgeSettings:
    """Common settings applied to every scenario run."""

    output_dir: Path
    working_directory: Path | None = None
    seed: int = 0
    log_level: str = "INFO"
    scenarios: list[ScenarioConfig] = field(default_factory=list)


@dataclass
class HpoSettings:
    """HPO backend knobs."""

    optimizer: str = "optuna"
    sampler: str = "tpe"


@dataclass
class BridgeConfig:
    """Top level configuration consumed by ``run_pipeline``."""

    hpo: HpoSettings = field(default_factory=HpoSettings)
    bridge: BridgeSettings = field(default_factory=lambda: BridgeSettings(output_dir=Path("./work/modelbridge")))


def _to_parameter_bounds(mapping: Mapping[str, Any]) -> dict[str, ParameterBounds]:
    """Convert ``mapping`` into a dictionary of :class:`ParameterBounds`."""

    bounds: dict[str, ParameterBounds] = {}
    for key, value in mapping.items():
        if not isinstance(value, Mapping):
            raise ValidationError(f"Parameter '{key}' definition must be a mapping")
        value_map = dict(value)
        try:
            low = float(value_map["low"])
            high = float(value_map["high"])
        except (KeyError, TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValidationError(f"Parameter '{key}' requires 'low' and 'high' floats") from exc
        step = value_map.get("step")
        log = bool(value_map.get("log", False))
        bounds[key] = ParameterBounds(low=low, high=high, step=step, log=log)
    return bounds


def _coerce_trials(name: str, data: Mapping[str, Any]) -> tuple[int, int]:
    try:
        macro_trials = int(data["macro_trials"])
    except KeyError as exc:
        raise ValidationError("Scenario requires 'macro_trials'") from exc
    if macro_trials <= 0:
        raise ValidationError(f"Scenario '{name}' requires macro_trials > 0")

    micro_trials = int(data.get("micro_trials", macro_trials))
    if micro_trials <= 0:
        raise ValidationError(f"Scenario '{name}' requires micro_trials > 0")
    return macro_trials, micro_trials


def _coerce_objective(name: str, data: Mapping[str, Any]) -> ObjectiveConfig:
    try:
        objective_raw = data["objective"]
    except KeyError as exc:  # noqa: PERF203
        raise ValidationError(f"Scenario '{name}' requires an 'objective' section") from exc
    if not isinstance(objective_raw, Mapping):
        raise ValidationError(f"Scenario '{name}' objective must be a mapping")

    try:
        target = str(objective_raw["target"])
    except KeyError as exc:
        raise ValidationError(f"Scenario '{name}' objective requires 'target'") from exc

    command = None
    if "command" in objective_raw:
        command_val = objective_raw["command"]
        if not isinstance(command_val, Sequence):  # noqa: PERF203
            raise ValidationError(f"Scenario '{name}' objective 'command' must be a sequence")
        command = [str(item) for item in command_val]

    timeout = None
    if "timeout" in objective_raw:
        timeout = float(objective_raw["timeout"])

    return ObjectiveConfig(target=target, command=command, timeout=timeout)


def _coerce_params(name: str, data: Mapping[str, Any]) -> ParameterSpace:
    try:
        params_raw = data["params"]
    except KeyError as exc:
        raise ValidationError(f"Scenario '{name}' requires 'params' section") from exc
    if not isinstance(params_raw, Mapping):
        raise ValidationError(f"Scenario '{name}' params must be a mapping")

    macro_raw = params_raw.get("macro", {})
    micro_raw = params_raw.get("micro", {})
    if not isinstance(macro_raw, Mapping) or not isinstance(micro_raw, Mapping):
        raise ValidationError(f"Scenario '{name}' params.macro/micro must be mappings")
    if not macro_raw or not micro_raw:
        raise ValidationError(f"Scenario '{name}' requires both macro and micro parameters")

    return ParameterSpace(
        macro=_to_parameter_bounds(macro_raw),
        micro=_to_parameter_bounds(micro_raw),
    )


def _coerce_regression(name: str, data: Mapping[str, Any]) -> RegressionConfig:
    regression_raw = data.get("regression", {})
    if not isinstance(regression_raw, Mapping):
        raise ValidationError(f"Scenario '{name}' regression must be a mapping")
    return RegressionConfig(
        kind=str(regression_raw.get("type", "linear")),
        degree=int(regression_raw.get("degree", 1)),
    )


def _coerce_metrics(name: str, data: Mapping[str, Any]) -> tuple[str, ...]:
    metrics_raw = data.get("metrics", ("mae", "mse", "r2"))
    if not isinstance(metrics_raw, (list, tuple)):
        raise ValidationError(f"Scenario '{name}' metrics must be a sequence")
    return tuple(str(item) for item in metrics_raw)


def _coerce_scenario(data: Mapping[str, Any]) -> ScenarioConfig:
    """Build a :class:`ScenarioConfig` from a raw mapping."""

    try:
        name = str(data["name"])
    except KeyError as exc:
        raise ValidationError("Scenario requires 'name'") from exc

    macro_trials, micro_trials = _coerce_trials(name, data)
    objective = _coerce_objective(name, data)
    params = _coerce_params(name, data)
    regression = _coerce_regression(name, data)
    metrics = _coerce_metrics(name, data)

    return ScenarioConfig(
        name=name,
        macro_trials=macro_trials,
        micro_trials=micro_trials,
        objective=objective,
        params=params,
        regression=regression,
        metrics=metrics,
    )


def _coerce_settings(data: Mapping[str, Any]) -> BridgeSettings:
    """Convert bridge level settings into :class:`BridgeSettings`."""

    output_raw = data.get("output_dir", "./work/modelbridge")
    output_dir = Path(output_raw).expanduser()
    working_directory = data.get("working_directory")
    working_path = Path(working_directory).expanduser() if working_directory else None
    seed = int(data.get("seed", 0))
    log_level = str(data.get("log_level", "INFO"))

    scenarios_raw = data.get("scenarios", [])
    if not isinstance(scenarios_raw, Iterable):
        raise ValidationError("bridge.scenarios must be iterable")
    scenarios = [_coerce_scenario(item) for item in scenarios_raw]

    return BridgeSettings(
        output_dir=output_dir,
        working_directory=working_path,
        seed=seed,
        log_level=log_level,
        scenarios=scenarios,
    )


def _coerce_hpo(data: Mapping[str, Any]) -> HpoSettings:
    """Create :class:`HpoSettings` from ``data``."""

    optimizer = str(data.get("optimizer", "optuna"))
    sampler = str(data.get("sampler", "tpe"))
    return HpoSettings(optimizer=optimizer, sampler=sampler)


def load_bridge_config(payload: Mapping[str, Any]) -> BridgeConfig:
    """Convert a mapping (or OmegaConf) into a :class:`BridgeConfig`."""

    if isinstance(payload, OmegaConf):
        payload = OmegaConf.to_container(payload, resolve=True)  # type: ignore[assignment]
    if not isinstance(payload, Mapping):
        raise ValidationError("Configuration payload must be a mapping")

    hpo_raw = payload.get("hpo", {})
    if not isinstance(hpo_raw, Mapping):
        raise ValidationError("hpo section must be a mapping")
    bridge_raw = payload.get("bridge")
    if not isinstance(bridge_raw, Mapping):
        raise ValidationError("bridge section must be provided")

    hpo = _coerce_hpo(hpo_raw)
    bridge = _coerce_settings(bridge_raw)
    return BridgeConfig(hpo=hpo, bridge=bridge)


__all__ = [
    "BridgeConfig",
    "BridgeSettings",
    "HpoSettings",
    "ObjectiveConfig",
    "ParameterBounds",
    "ParameterSpace",
    "RegressionConfig",
    "ScenarioConfig",
    "load_bridge_config",
]
