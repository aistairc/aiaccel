"""Configuration models and helpers for the redesigned modelbridge pipeline."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ParameterBounds(BaseModel):
    """Numeric parameter definition for Optuna suggestions."""

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    step: float | None = None
    log: bool = False

    @field_validator("high")
    @classmethod
    def _ensure_range(cls, high: float, info) -> float:  # type: ignore[override]
        low = info.data.get("low", high)
        if high <= low:
            raise ValueError("high must be greater than low")
        return high


class ParameterSpace(BaseModel):
    """Collection of macro/micro parameter bounds."""

    model_config = ConfigDict(extra="forbid")

    macro: dict[str, ParameterBounds] = Field(default_factory=dict)
    micro: dict[str, ParameterBounds] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_params(self) -> "ParameterSpace":
        if not self.macro or not self.micro:
            raise ValueError("macro and micro parameter spaces must be provided")
        return self


class ObjectiveConfig(BaseModel):
    """Definition of the evaluation entry point."""

    model_config = ConfigDict(extra="forbid")

    target: str
    command: list[str] | None = None
    timeout: float | None = None

    @field_validator("command")
    @classmethod
    def _coerce_command(cls, command: list[Any] | None) -> list[str] | None:
        if command is None:
            return None
        if not isinstance(command, list):
            raise ValueError("command must be a list")
        return [str(item) for item in command]


class RegressionConfig(BaseModel):
    """Regression strategy configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: str = "linear"
    degree: int = 1
    kernel: str | None = None
    noise: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        kind_alias = payload.pop("type", None)
        if kind_alias:
            payload["kind"] = kind_alias
        poly = payload.pop("poly", None)
        if isinstance(poly, Mapping) and "degree" in poly:
            payload["degree"] = poly.get("degree")
        gpr = payload.pop("gpr", None)
        if isinstance(gpr, Mapping):
            if "kernel" in gpr:
                payload["kernel"] = gpr.get("kernel")
            if "noise" in gpr:
                payload["noise"] = gpr.get("noise")
            if "alpha" in gpr:
                payload["noise"] = gpr.get("alpha")
        if "alpha" in payload and payload.get("noise") is None:
            payload["noise"] = payload["alpha"]
            payload.pop("alpha", None)
        return payload

    @field_validator("degree")
    @classmethod
    def _validate_degree(cls, value: int) -> int:
        if value < 1:
            raise ValueError("degree must be >= 1")
        return value


class ScenarioConfig(BaseModel):
    """Configuration for a single bridging scenario."""

    model_config = ConfigDict(extra="forbid")

    name: str
    train_macro_trials: int
    train_micro_trials: int
    eval_macro_trials: int
    eval_micro_trials: int
    train_objective: ObjectiveConfig
    eval_objective: ObjectiveConfig
    train_params: ParameterSpace
    eval_params: ParameterSpace
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    metrics: Sequence[str] = Field(default_factory=lambda: ("mae", "mse", "r2"))

    @field_validator("train_macro_trials", "train_micro_trials", "eval_macro_trials", "eval_micro_trials")
    @classmethod
    def _validate_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("trial counts must be > 0")
        return value

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, metrics: Sequence[str]) -> tuple[str, ...]:
        allowed = {"mae", "mse", "r2"}
        metrics_tuple = tuple(str(item) for item in metrics)
        invalid = [metric for metric in metrics_tuple if metric not in allowed]
        if invalid:
            raise ValueError(f"metrics contains unsupported values: {', '.join(invalid)}")
        return metrics_tuple


class HpoSettings(BaseModel):
    """HPO backend knobs."""

    model_config = ConfigDict(extra="forbid")

    optimizer: str = "optuna"
    sampler: str = "tpe"


class BridgeSettings(BaseModel):
    """Common settings applied to every scenario run."""

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    working_directory: Path | None = None
    seed: int = 0
    log_level: str = "INFO"
    train_runs: int = 1
    eval_runs: int = 0
    write_csv: bool = False
    scenarios: list[ScenarioConfig] = Field(default_factory=list)

    @field_validator("train_runs", "eval_runs")
    @classmethod
    def _validate_runs(cls, value: int) -> int:
        if value < 0:
            raise ValueError("run counts must be >= 0")
        return value

    @model_validator(mode="after")
    def _default_workdir(self) -> "BridgeSettings":
        if self.working_directory is None:
            self.working_directory = self.output_dir
        return self


class DataAssimilationScaling(BaseModel):
    """Scaling rules for MAS-Bench parameters."""

    model_config = ConfigDict(extra="forbid")

    sigma_scale: tuple[float, float] = (97.0, 3.0)  # (multiplier, offset)
    mu_scale: float = 300.0
    pi_complement: bool = True  # fill last pi as 1 - sum(previous)


class DataAssimilationSamplers(BaseModel):
    """Sampler choices for micro/macro optimisation."""

    model_config = ConfigDict(extra="forbid")

    micro: str = "random"
    macro_train: str = "cmaes"
    macro_test: str = "cmaes"


class DataAssimilationTrials(BaseModel):
    """Trial counts for each optimisation stage."""

    model_config = ConfigDict(extra="forbid")

    micro: int = 4
    macro_train: int = 5
    macro_test: int = 5

    @field_validator("micro", "macro_train", "macro_test")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("trial counts must be > 0")
        return value


class DataAssimilationSeeds(BaseModel):
    """Seeds for reproducible optimisation."""

    model_config = ConfigDict(extra="forbid")

    micro: int = 1
    macro_train: int = 1
    macro_test: int = 1

    @field_validator("micro", "macro_train", "macro_test")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value


class DataAssimilationConfig(BaseModel):
    """Configuration for MAS-Bench data assimilation workflow."""

    model_config = ConfigDict(extra="forbid")

    output_root: Path = Path("./work/modelbridge/data_assimilation")
    mas_bench_jar: Path | None = None
    dataset_root: Path | None = None
    micro_model: str
    macro_model: str
    scenarios: int = 4
    regression_degree: int = 1
    allow_mock: bool = False  # allow skipping jar execution (tests/CI)
    agent_sizes: dict[str, int] | None = None  # optional override {"naive":..,"rational":..,"ruby":..}
    samplers: DataAssimilationSamplers = Field(default_factory=DataAssimilationSamplers)
    trials: DataAssimilationTrials = Field(default_factory=DataAssimilationTrials)
    seeds: DataAssimilationSeeds = Field(default_factory=DataAssimilationSeeds)
    scaling: DataAssimilationScaling = Field(default_factory=DataAssimilationScaling)

    @field_validator("scenarios")
    @classmethod
    def _scenarios_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("scenarios must be > 0")
        return value

    @field_validator("regression_degree")
    @classmethod
    def _degree_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("regression_degree must be >= 1")
        return value


class BridgeConfig(BaseModel):
    """Top level configuration consumed by ``run_pipeline``."""

    model_config = ConfigDict(extra="ignore")

    hpo: HpoSettings = Field(default_factory=HpoSettings)
    bridge: BridgeSettings = Field(default_factory=lambda: BridgeSettings(output_dir=Path("./work/modelbridge")))
    data_assimilation: DataAssimilationConfig | None = None




def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _deep_merge(base[key], value)  # type: ignore[index]
        else:
            merged[key] = value
    return merged


def load_bridge_config(payload: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Convert a mapping (or OmegaConf) into a :class:`BridgeConfig`.

    ``overrides`` (if provided) is merged over the payload (payload < overrides).
    """

    if isinstance(payload, OmegaConf):
        payload = OmegaConf.to_container(payload, resolve=True)  # type: ignore[assignment]
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration payload must be a mapping")

    layered = dict(payload)
    if overrides:
        layered = _deep_merge(layered, overrides)

    try:
        bridge = BridgeConfig.model_validate(layered)
        return bridge
    except ValidationError as exc:  # noqa: TRY003
        raise ValueError(f"Invalid bridge configuration: {exc}") from exc


def generate_schema() -> dict[str, Any]:
    """Return JSON schema for the bridge configuration."""

    return BridgeConfig.model_json_schema()


__all__ = [
    "BridgeConfig",
    "BridgeSettings",
    "DataAssimilationConfig",
    "DataAssimilationScaling",
    "DataAssimilationSamplers",
    "DataAssimilationTrials",
    "DataAssimilationSeeds",
    "HpoSettings",
    "ObjectiveConfig",
    "ParameterBounds",
    "ParameterSpace",
    "RegressionConfig",
    "ScenarioConfig",
    "load_bridge_config",
    "generate_schema",
]
