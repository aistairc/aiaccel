"""Configuration models for modelbridge."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from pathlib import Path

from omegaconf import OmegaConf

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator


class ParameterBounds(BaseModel):
    """Numeric parameter definition.

    Attributes:
        low (float): Lower bound.
        high (float): Upper bound.
        step (float | None): Step size for discretization.
        log (bool): Whether to use logarithmic scale.
    """

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    step: float | None = None
    log: bool = False

    @field_validator("high")
    @classmethod
    def _ensure_range(cls, high: float, info: ValidationInfo) -> float:
        low = info.data.get("low", high)
        if high <= low:
            raise ValueError("high must be greater than low")
        return high


class ParameterSpace(BaseModel):
    """Collection of macro/micro parameter bounds.

    Attributes:
        macro (dict[str, ParameterBounds]): Macro parameter definitions.
        micro (dict[str, ParameterBounds]): Micro parameter definitions.
    """

    model_config = ConfigDict(extra="forbid")

    macro: dict[str, ParameterBounds] = Field(default_factory=dict)
    micro: dict[str, ParameterBounds] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_params(self) -> ParameterSpace:
        if not self.macro or not self.micro:
            raise ValueError("macro and micro parameter spaces must be provided")
        return self


class ObjectiveConfig(BaseModel):
    """Definition of the evaluation entry point.

    Attributes:
        command (list[str]): Command line to execute.
    """

    model_config = ConfigDict(extra="forbid")

    command: list[str]

    @field_validator("command", mode="before")
    @classmethod
    def _coerce_command(cls, command: Any) -> list[str]:
        if not command:
            raise ValueError("command must be provided")
        if isinstance(command, str):
            import shlex

            return shlex.split(command)
        if not isinstance(command, list):
            raise ValueError("command must be a list or string")
        return [str(item) for item in command]


class RegressionConfig(BaseModel):
    """Regression strategy configuration.

    Attributes:
        kind (str): Regression algorithm ('linear', 'polynomial', 'gpr').
        degree (int): Polynomial degree (default: 1).
        kernel (str | None): GPR kernel name ('RBF', 'MATERN32', 'MATERN52').
        noise (float | None): GPR noise variance.
    """

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
        # Normalize aliases for compatibility or ease of use
        if "type" in payload:
            payload["kind"] = payload.pop("type")

        # Flatten structure if nested configs were provided in old spec (poly, gpr)
        # Spec v13 defines flat structure but we support loading old style if needed?
        # Spec v13 says strict validation. Let's stick to flat.
        # But user YAML might use aliases.
        return payload

    @field_validator("degree")
    @classmethod
    def _validate_degree(cls, value: int) -> int:
        if value < 1:
            raise ValueError("degree must be >= 1")
        return value


class ScenarioConfig(BaseModel):
    """Configuration for a single bridging scenario.

    Attributes:
        name (str): Scenario identifier.
        train_macro_trials (int): Number of trials for training macro HPO.
        train_micro_trials (int): Number of trials for training micro HPO.
        eval_macro_trials (int): Number of trials for evaluation macro HPO.
        eval_micro_trials (int): Number of trials for evaluation micro HPO.
        train_objective (ObjectiveConfig): Objective function for training.
        eval_objective (ObjectiveConfig): Objective function for evaluation.
        train_params (ParameterSpace): Parameter search space for training.
        eval_params (ParameterSpace): Parameter search space for evaluation.
        regression (RegressionConfig): Regression model settings.
        metrics (Sequence[str]): Evaluation metrics ('mae', 'mse', 'r2').
    """

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
    metrics: Sequence[str] = Field(default_factory=lambda: ["mae", "mse", "r2"])

    @field_validator("train_macro_trials", "train_micro_trials", "eval_macro_trials", "eval_micro_trials")
    @classmethod
    def _validate_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("trial counts must be > 0")
        return value

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, metrics: Sequence[str]) -> list[str]:
        allowed = {"mae", "mse", "r2"}
        metrics_list = [str(item) for item in metrics]
        invalid = [metric for metric in metrics_list if metric not in allowed]
        if invalid:
            raise ValueError(f"metrics contains unsupported values: {', '.join(invalid)}")
        return metrics_list


class HpoSettings(BaseModel):
    """HPO backend configuration.

    Attributes:
        base_config (Path): Path to the base aiaccel configuration file.
        macro_overrides (dict[str, Any]): OmegaConf overrides for macro HPO.
        micro_overrides (dict[str, Any]): OmegaConf overrides for micro HPO.
    """

    model_config = ConfigDict(extra="forbid")

    base_config: Path
    macro_overrides: dict[str, Any] = Field(default_factory=dict)
    micro_overrides: dict[str, Any] = Field(default_factory=dict)


class BridgeSettings(BaseModel):
    """Common settings applied to every scenario run.

    Attributes:
        output_dir (Path): Root directory for outputs.
        seed (int): Random seed base.
        log_level (str): Logging level (INFO, DEBUG, etc.).
        json_log (bool): Enable JSON structured logging.
        train_runs (int): Number of training runs per scenario.
        eval_runs (int): Number of evaluation runs per scenario.
        strict_mode (bool): Fail hard on missing/ambiguous required inputs.
        scenarios (list[ScenarioConfig]): List of scenarios to execute.
    """

    model_config = ConfigDict(extra="forbid")

    output_dir: Path
    seed: int = 0
    log_level: str = "INFO"
    json_log: bool = False
    train_runs: int = 1
    eval_runs: int = 0
    strict_mode: bool = False
    scenarios: list[ScenarioConfig] = Field(default_factory=list)

    @field_validator("train_runs", "eval_runs")
    @classmethod
    def _validate_runs(cls, value: int) -> int:
        if value < 0:
            raise ValueError("run counts must be >= 0")
        return value


class BridgeConfig(BaseModel):
    """Top level configuration.

    Attributes:
        bridge (BridgeSettings): General bridge settings.
        hpo (HpoSettings): HPO backend settings.
    """

    model_config = ConfigDict(extra="forbid")

    bridge: BridgeSettings
    hpo: HpoSettings


def load_bridge_config(payload: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Convert a mapping into a BridgeConfig.

    Args:
        payload (Mapping[str, Any]): The configuration dictionary (or OmegaConf object).
        overrides (Mapping[str, Any] | None): Optional overrides to merge.

    Returns:
        BridgeConfig: The validated configuration object.

    Raises:
        ValueError: If validation fails.
    """
    if isinstance(payload, OmegaConf):
        payload = OmegaConf.to_container(payload, resolve=True)  # type: ignore

    if overrides:
        payload = _deep_merge(dict(payload), overrides)

    try:
        return BridgeConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid bridge configuration: {exc}") from exc


def generate_schema() -> dict[str, Any]:
    """Return JSON schema for the bridge configuration.

    Returns:
        dict[str, Any]: The JSON schema.
    """
    return BridgeConfig.model_json_schema()


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(base[key], value)
        else:
            merged[key] = value
    return merged
