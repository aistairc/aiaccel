"""Pydantic configuration schema for modelbridge."""

from __future__ import annotations

from typing import Any, Literal, cast

from collections.abc import Mapping, Sequence
from pathlib import Path
import shlex

from omegaconf import OmegaConf

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

SeedMode = Literal["auto_increment", "user_defined"]
ExecutionTarget = Literal["local", "abci"]
JobMode = Literal["cpu", "gpu", "mpi", "train"]
MetricName = Literal["mae", "mse", "r2"]


def _default_metrics() -> list[MetricName]:
    """Return default metric list for scenario settings."""
    return ["mae", "mse", "r2"]


def _to_tokens(value: Any, *, field_name: str) -> list[str]:
    """Normalize token input from string or sequence."""
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    raise ValueError(f"{field_name} must be a list or string")


class ParameterBounds(BaseModel):
    """Numeric parameter range.

    Args:
        low: Lower bound.
        high: Upper bound.
        step: Optional discretization step.
        log: Whether to apply logarithmic sampling.
    """

    model_config = ConfigDict(extra="forbid")
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @field_validator("high")
    @classmethod
    def _ensure_range(cls, high: float, info: ValidationInfo) -> float:
        if high <= info.data.get("low", high):
            raise ValueError("high must be greater than low")
        return high


class ParameterSpace(BaseModel):
    """Role parameter spaces.

    Args:
        macro: Macro-target parameter bounds.
        micro: Micro-target parameter bounds.
    """

    model_config = ConfigDict(extra="forbid")
    macro: dict[str, ParameterBounds] = Field(min_length=1)
    micro: dict[str, ParameterBounds] = Field(min_length=1)


class ObjectiveConfig(BaseModel):
    """Objective command configuration.

    Args:
        command: Objective command tokens.
    """

    model_config = ConfigDict(extra="forbid")
    command: list[str]

    @field_validator("command", mode="before")
    @classmethod
    def _coerce_command(cls, value: Any) -> list[str]:
        tokens = _to_tokens(value, field_name="command")
        if not tokens:
            raise ValueError("command must be provided")
        return tokens


class SeedUserValues(BaseModel):
    """User-defined seeds grouped by role and target.

    Args:
        train_macro: Seeds for train/macro runs.
        train_micro: Seeds for train/micro runs.
        eval_macro: Seeds for eval/macro runs.
        eval_micro: Seeds for eval/micro runs.
    """

    model_config = ConfigDict(extra="forbid")
    train_macro: list[int] = Field(default_factory=list)
    train_micro: list[int] = Field(default_factory=list)
    eval_macro: list[int] = Field(default_factory=list)
    eval_micro: list[int] = Field(default_factory=list)


class SeedPolicyConfig(BaseModel):
    """Seed policy for one stream.

    Args:
        mode: Seed mode.
        base: Base seed for auto mode.
        user_values: Explicit seeds for user-defined mode.
    """

    model_config = ConfigDict(extra="forbid")
    mode: SeedMode = "auto_increment"
    base: int | None = None
    user_values: SeedUserValues | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> SeedPolicyConfig:
        if self.mode == "user_defined" and self.user_values is None:
            raise ValueError("user_values must be provided when mode=user_defined")
        return self


class SeedPolicySet(BaseModel):
    """Seed policies for sampler and optimizer streams.

    Args:
        sampler: Sampler seed policy.
        optimizer: Optimizer seed policy.
    """

    model_config = ConfigDict(extra="forbid")
    sampler: SeedPolicyConfig = Field(default_factory=SeedPolicyConfig)
    optimizer: SeedPolicyConfig = Field(default_factory=SeedPolicyConfig)


class ExecutionTargetConfig(BaseModel):
    """Execution target options.

    Args:
        target: Execution target name.
        emit_on_prepare: Whether prepare emits commands.
        job_profile: Job launcher profile.
        job_mode: Job launcher mode.
        job_walltime: Optional walltime.
        job_log_dir: Optional log directory override.
        job_extra_args: Additional job arguments.
    """

    model_config = ConfigDict(extra="forbid")
    target: ExecutionTarget = "local"
    emit_on_prepare: bool = False
    job_profile: str | None = None
    job_mode: JobMode = "cpu"
    job_walltime: str | None = None
    job_log_dir: Path | None = None
    job_extra_args: list[str] = Field(default_factory=list)

    @field_validator("job_extra_args", mode="before")
    @classmethod
    def _coerce_extra_args(cls, value: Any) -> list[str]:
        return _to_tokens(value, field_name="job_extra_args")

    @model_validator(mode="after")
    def _normalize_defaults(self) -> ExecutionTargetConfig:
        if self.job_profile is None:
            self.job_profile = "sge" if self.target == "abci" else "local"
        return self


class RegressionConfig(BaseModel):
    """Regression model options.

    Args:
        kind: Regression algorithm name.
        degree: Polynomial degree.
        kernel: Optional GPR kernel name.
        noise: Optional GPR noise value.
    """

    model_config = ConfigDict(extra="forbid")
    kind: str = "linear"
    degree: int = Field(default=1, ge=1)
    kernel: str | None = None
    noise: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or "type" not in data:
            return data
        payload = dict(data)
        payload["kind"] = payload.pop("type")
        return payload


class ScenarioConfig(BaseModel):
    """One scenario configuration.

    Args:
        name: Scenario name.
        *_trials: Trial count per role/target.
        train_objective: Train objective command.
        eval_objective: Eval objective command.
        train_params: Train parameter space.
        eval_params: Eval parameter space.
        regression: Regression settings.
        metrics: Metric names.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    train_macro_trials: int = Field(gt=0)
    train_micro_trials: int = Field(gt=0)
    eval_macro_trials: int = Field(gt=0)
    eval_micro_trials: int = Field(gt=0)
    train_objective: ObjectiveConfig
    eval_objective: ObjectiveConfig
    train_params: ParameterSpace
    eval_params: ParameterSpace
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    metrics: list[MetricName] = Field(default_factory=_default_metrics)


class HpoSettings(BaseModel):
    """HPO base config and override payloads.

    Args:
        base_config: Base optimize config YAML path.
        macro_overrides: Overrides applied to macro target configs.
        micro_overrides: Overrides applied to micro target configs.
        abci_overrides: Additional overrides when execution target is ``abci``.
    """

    model_config = ConfigDict(extra="forbid")
    base_config: Path
    macro_overrides: dict[str, Any] = Field(default_factory=dict)
    micro_overrides: dict[str, Any] = Field(default_factory=dict)
    abci_overrides: dict[str, Any] = Field(default_factory=dict)


class BridgeSettings(BaseModel):
    """Top-level runtime settings for modelbridge.

    Args:
        output_dir: Root output directory.
        seed: Base seed fallback value.
        seed_policy: Seed policies for sampler and optimizer.
        execution: Execution target settings.
        log_level: Root logging level name.
        json_log: Whether to emit JSON-formatted logs.
        train_runs: Number of train runs per scenario.
        eval_runs: Number of eval runs per scenario.
        strict_mode: Whether scenario issues should fail a step.
        scenarios: Scenario configuration list.
    """

    model_config = ConfigDict(extra="forbid")
    output_dir: Path
    seed: int = 0
    seed_policy: SeedPolicySet = Field(default_factory=SeedPolicySet)
    execution: ExecutionTargetConfig = Field(default_factory=ExecutionTargetConfig)
    log_level: str = "INFO"
    json_log: bool = False
    train_runs: int = Field(default=1, ge=0)
    eval_runs: int = Field(default=0, ge=0)
    strict_mode: bool = False
    scenarios: list[ScenarioConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_seed_user_values_length(self) -> BridgeSettings:
        expected = {
            "train_macro": self.train_runs,
            "train_micro": self.train_runs,
            "eval_macro": self.eval_runs,
            "eval_micro": self.eval_runs,
        }
        for name, policy in {
            "sampler": self.seed_policy.sampler,
            "optimizer": self.seed_policy.optimizer,
        }.items():
            if policy.mode != "user_defined":
                continue
            if policy.user_values is None:
                raise ValueError(f"{name} seed policy requires user_values")
            values = policy.user_values.model_dump()
            for key, expected_len in expected.items():
                actual_len = len(values[key])
                if actual_len != expected_len:
                    raise ValueError(f"{name} seed user_values.{key} length must be {expected_len}, got {actual_len}")
        return self


class BridgeConfig(BaseModel):
    """Validated modelbridge config root.

    Args:
        bridge: Bridge runtime settings.
        hpo: HPO generation settings.
    """

    model_config = ConfigDict(extra="forbid")
    bridge: BridgeSettings
    hpo: HpoSettings


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested mappings.

    Args:
        base: Base mapping.
        override: Override mapping applied over ``base``.

    Returns:
        dict[str, Any]: Merged mapping payload.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def load_bridge_config(payload: Mapping[str, Any] | Any, overrides: Mapping[str, Any] | None = None) -> BridgeConfig:
    """Validate raw payload and return ``BridgeConfig``.

    Args:
        payload: Raw mapping or OmegaConf payload.
        overrides: Optional override mapping merged before validation.

    Returns:
        BridgeConfig: Validated configuration.

    Raises:
        ValueError: If payload is invalid or schema validation fails.
    """
    if OmegaConf.is_config(payload):
        payload = OmegaConf.to_container(payload, resolve=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Bridge configuration must be a mapping")

    merged_payload = deep_merge_mappings(cast(Mapping[str, Any], payload), overrides) if overrides else dict(payload)
    try:
        return BridgeConfig.model_validate(merged_payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid bridge configuration: {exc}") from exc


def generate_schema() -> dict[str, Any]:
    """Return JSON schema for modelbridge config.

    Returns:
        dict[str, Any]: JSON schema payload.
    """
    return BridgeConfig.model_json_schema()
