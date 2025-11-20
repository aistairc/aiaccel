"""Configuration dataclasses and helpers for the redesigned modelbridge pipeline."""

from __future__ import annotations

from typing import Any

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


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
    kernel: str | None = None
    noise: float | None = None


@dataclass
class ScenarioConfig:
    """Configuration for a single bridging scenario."""

    name: str
    train_macro_trials: int
    train_micro_trials: int
    eval_macro_trials: int
    eval_micro_trials: int
    objective: ObjectiveConfig
    params: ParameterSpace
    train_objective: ObjectiveConfig | None = None
    eval_objective: ObjectiveConfig | None = None
    train_params: ParameterSpace | None = None
    eval_params: ParameterSpace | None = None
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    metrics: Sequence[str] = field(default_factory=lambda: ("mae", "mse", "r2"))


@dataclass
class BridgeSettings:
    """Common settings applied to every scenario run."""

    output_dir: Path
    working_directory: Path | None = None
    seed: int = 0
    log_level: str = "INFO"
    train_runs: int = 1
    eval_runs: int = 0
    storage: str | None = None
    write_csv: bool = False
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
            raise ValueError(f"Parameter '{key}' definition must be a mapping")
        value_map = dict(value)
        try:
            low = float(value_map["low"])
            high = float(value_map["high"])
        except (KeyError, TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError(f"Parameter '{key}' requires 'low' and 'high' floats") from exc
        step = value_map.get("step")
        log = bool(value_map.get("log", False))
        bounds[key] = ParameterBounds(low=low, high=high, step=step, log=log)
    return bounds


def _coerce_trials(name: str, data: Mapping[str, Any], prefix: str) -> tuple[int, int]:
    macro_key = f"{prefix}_macro_trials"
    micro_key = f"{prefix}_micro_trials"
    try:
        macro_trials = int(data.get(macro_key, data.get("macro_trials")))
    except KeyError as exc:
        raise ValueError(f"Scenario requires '{macro_key}'") from exc
    if macro_trials <= 0:
        raise ValueError(f"Scenario '{name}' requires {macro_key} > 0")

    micro_trials = int(data.get(micro_key, macro_trials))
    if micro_trials <= 0:
        raise ValueError(f"Scenario '{name}' requires {micro_key} > 0")
    return macro_trials, micro_trials


def _coerce_objective(name: str, data: Mapping[str, Any]) -> tuple[ObjectiveConfig, ObjectiveConfig | None, ObjectiveConfig | None]:
    base_raw = data.get("objective")
    if not isinstance(base_raw, Mapping):
        raise ValueError(f"Scenario '{name}' requires an 'objective' mapping")
    base = _parse_objective_mapping(name, base_raw, "objective", require_target=True)
    train_raw = data.get("train_objective")
    eval_raw = data.get("eval_objective")
    train = _build_objective_with_override(name, base, train_raw, "train_objective")
    eval_obj = _build_objective_with_override(name, base, eval_raw, "eval_objective")
    return ObjectiveConfig(**base), train, eval_obj


def _parse_objective_mapping(
    name: str,
    payload: Mapping[str, Any],
    label: str,
    *,
    require_target: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"Scenario '{name}' {label} must be a mapping")

    target = payload.get("target")
    if require_target:
        if target is None:
            raise ValueError(f"Scenario '{name}' {label} requires 'target'")
        target = str(target)
    elif target is not None:
        target = str(target)

    command_raw = payload.get("command")
    command = None
    if command_raw is not None:
        if not isinstance(command_raw, Sequence) or isinstance(command_raw, (str, bytes)):  # noqa: PERF203
            raise ValueError(f"Scenario '{name}' {label} 'command' must be a sequence")
        command = [str(item) for item in command_raw]

    timeout_raw = payload.get("timeout")
    timeout = float(timeout_raw) if timeout_raw is not None else None

    result = {"target": target, "command": command, "timeout": timeout}
    return result


def _build_objective_with_override(
    name: str,
    base: dict[str, Any],
    override_raw: Mapping[str, Any] | None,
    label: str,
) -> ObjectiveConfig | None:
    if override_raw is None:
        return None
    override = _parse_objective_mapping(name, override_raw, label, require_target=False)
    merged = {
        "target": override["target"] or base["target"],
        "command": override["command"] if override["command"] is not None else base["command"],
        "timeout": override["timeout"] if override["timeout"] is not None else base["timeout"],
    }
    return ObjectiveConfig(**merged)


def _coerce_params(
    name: str,
    data: Mapping[str, Any],
    key: str = "params",
    *,
    base: ParameterSpace | None = None,
    allow_partial: bool = False,
) -> ParameterSpace:
    params_raw = data.get(key)
    if params_raw is None:
        if allow_partial and base is not None:
            return base
        raise ValueError(f"Scenario '{name}' requires '{key}' section")
    if not isinstance(params_raw, Mapping):
        raise ValueError(f"Scenario '{name}' {key} must be a mapping")

    macro_raw = params_raw.get("macro")
    micro_raw = params_raw.get("micro")
    if not allow_partial:
        if not isinstance(macro_raw, Mapping) or not isinstance(micro_raw, Mapping):
            raise ValueError(f"Scenario '{name}' {key}.macro/micro must be mappings")
        if not macro_raw or not micro_raw:
            raise ValueError(f"Scenario '{name}' requires both macro and micro parameters in {key}")
    macro_bounds = _to_parameter_bounds(macro_raw) if isinstance(macro_raw, Mapping) else {}
    micro_bounds = _to_parameter_bounds(micro_raw) if isinstance(micro_raw, Mapping) else {}

    if allow_partial and base is not None:
        macro_bounds = macro_bounds or base.macro
        micro_bounds = micro_bounds or base.micro
    if not macro_bounds or not micro_bounds:
        raise ValueError(f"Scenario '{name}' requires both macro and micro parameters in {key}")

    return ParameterSpace(macro=macro_bounds, micro=micro_bounds)


def _coerce_regression(name: str, data: Mapping[str, Any]) -> RegressionConfig:
    regression_raw = data.get("regression", {})
    if not isinstance(regression_raw, Mapping):
        raise ValueError(f"Scenario '{name}' regression must be a mapping")
    if not regression_raw:
        return RegressionConfig()

    def _coerce_int(value: Any, field: str, default: int) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError(f"Scenario '{name}' regression.{field} must be an integer") from exc

    def _coerce_float(value: Any, field: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError(f"Scenario '{name}' regression.{field} must be a float") from exc

    kind_value = regression_raw.get("kind", regression_raw.get("type", "linear"))
    kind = str(kind_value)

    degree_value = regression_raw.get("degree")
    poly_raw = regression_raw.get("poly")
    if isinstance(poly_raw, Mapping) and "degree" in poly_raw:
        degree_value = poly_raw.get("degree")
    degree = _coerce_int(degree_value, "degree", 1)

    kernel_value = regression_raw.get("kernel")
    gpr_raw = regression_raw.get("gpr")
    if isinstance(gpr_raw, Mapping) and "kernel" in gpr_raw:
        kernel_value = gpr_raw.get("kernel")
    kernel = str(kernel_value) if kernel_value not in (None, "") else None

    noise_value = regression_raw.get("noise")
    alpha_value = regression_raw.get("alpha")
    noise = _coerce_float(noise_value, "noise")
    alpha = _coerce_float(alpha_value, "alpha")

    if isinstance(gpr_raw, Mapping):
        gpr_noise = _coerce_float(gpr_raw.get("noise"), "gpr.noise")
        gpr_alpha = _coerce_float(gpr_raw.get("alpha"), "gpr.alpha")
        if gpr_noise is not None:
            noise = gpr_noise
        if gpr_alpha is not None:
            alpha = gpr_alpha

    if alpha is not None:
        noise = alpha

    return RegressionConfig(kind=kind, degree=degree, kernel=kernel, noise=noise)


def _coerce_metrics(name: str, data: Mapping[str, Any]) -> tuple[str, ...]:
    metrics_raw = data.get("metrics", ("mae", "mse", "r2"))
    if not isinstance(metrics_raw, (list, tuple)):
        raise ValueError(f"Scenario '{name}' metrics must be a sequence")
    allowed = {"mae", "mse", "r2"}
    metrics = tuple(str(item) for item in metrics_raw)
    invalid = [metric for metric in metrics if metric not in allowed]
    if invalid:
        raise ValueError(f"Scenario '{name}' metrics contains unsupported values: {', '.join(invalid)}")
    return metrics


def _coerce_scenario(data: Mapping[str, Any]) -> ScenarioConfig:
    """Build a :class:`ScenarioConfig` from a raw mapping."""

    try:
        name = str(data["name"])
    except KeyError as exc:
        raise ValueError("Scenario requires 'name'") from exc

    train_macro_trials, train_micro_trials = _coerce_trials(name, data, "train")
    eval_macro_trials, eval_micro_trials = _coerce_trials(name, data, "eval")
    objective_base, train_objective, eval_objective = _coerce_objective(name, data)
    params = _coerce_params(name, data)
    train_params = _coerce_params(name, data, "train_params", base=params, allow_partial=True)
    eval_params = _coerce_params(name, data, "eval_params", base=params, allow_partial=True)
    regression = _coerce_regression(name, data)
    metrics = _coerce_metrics(name, data)

    return ScenarioConfig(
        name=name,
        train_macro_trials=train_macro_trials,
        train_micro_trials=train_micro_trials,
        eval_macro_trials=eval_macro_trials,
        eval_micro_trials=eval_micro_trials,
        objective=objective_base,
        params=params,
        train_objective=train_objective,
        eval_objective=eval_objective,
        train_params=train_params,
        eval_params=eval_params,
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
    train_runs = int(data.get("train_runs", 1))
    eval_runs = int(data.get("eval_runs", 0))
    storage_raw = data.get("storage")
    storage = None
    if storage_raw is not None:
        storage = str(storage_raw)
        if "://" not in storage:
            storage = f"sqlite:///{Path(storage).expanduser().resolve()}"

    write_csv = bool(data.get("write_csv", False))

    scenarios_raw = data.get("scenarios", [])
    if not isinstance(scenarios_raw, Iterable):
        raise ValueError("bridge.scenarios must be iterable")
    scenarios = [_coerce_scenario(item) for item in scenarios_raw]

    return BridgeSettings(
        output_dir=output_dir,
        working_directory=working_path,
        seed=seed,
        log_level=log_level,
        train_runs=train_runs,
        eval_runs=eval_runs,
        storage=storage,
        write_csv=write_csv,
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
        raise ValueError("Configuration payload must be a mapping")

    hpo_raw = payload.get("hpo", {})
    if not isinstance(hpo_raw, Mapping):
        raise ValueError("hpo section must be a mapping")
    bridge_raw = payload.get("bridge")
    if not isinstance(bridge_raw, Mapping):
        raise ValueError("bridge section must be provided")

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
