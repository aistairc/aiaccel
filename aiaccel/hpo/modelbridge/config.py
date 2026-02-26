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
BridgeRole = Literal["train", "eval"]
BridgeTarget = Literal["macro", "micro"]


def _default_metrics() -> list[MetricName]:
    """Return default metric list for scenario evaluation."""
    return ["mae", "mse", "r2"]


def _to_tokens(value: Any, *, field_name: str) -> list[str]:
    """Normalize string/list values to token lists."""
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    raise ValueError(f"{field_name} must be a list or string")


class ParameterBounds(BaseModel):
    """Define numeric parameter bounds.

    This model describes one tunable numeric parameter range used in HPO space
    generation.

    Args:
        low: Lower bound.
        high: Upper bound.
        step: Optional discretization step.
        log: Whether to sample in log space.

    Returns:
        ParameterBounds: Validated parameter bound model.

    Raises:
        ValueError: If ``high`` is not greater than ``low``.
    """

    model_config = ConfigDict(extra="forbid")
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @field_validator("high")
    @classmethod
    def _ensure_range(cls, high: float, info: ValidationInfo) -> float:
        """Validate that upper bound is greater than lower bound."""
        if high <= info.data.get("low", high):
            raise ValueError("high must be greater than low")
        return high


class ParameterSpace(BaseModel):
    """Define role-specific parameter spaces.

    This model stores macro and micro parameter-bound mappings for one role.

    Args:
        macro: Macro-target parameter bounds.
        micro: Micro-target parameter bounds.

    Returns:
        ParameterSpace: Validated parameter-space model.

    Raises:
        ValueError: If either mapping is empty or invalid.
    """

    model_config = ConfigDict(extra="forbid")
    macro: dict[str, ParameterBounds] = Field(min_length=1)
    micro: dict[str, ParameterBounds] = Field(min_length=1)


class ObjectiveConfig(BaseModel):
    """Define objective command configuration.

    This model normalizes objective command inputs into command-token lists.

    Args:
        command: Objective command tokens.

    Returns:
        ObjectiveConfig: Validated objective configuration.

    Raises:
        ValueError: If command cannot be tokenized or is empty.
    """

    model_config = ConfigDict(extra="forbid")
    command: list[str]

    @field_validator("command", mode="before")
    @classmethod
    def _coerce_command(cls, value: Any) -> list[str]:
        """Normalize and validate objective command tokens."""
        tokens = _to_tokens(value, field_name="command")
        if not tokens:
            raise ValueError("command must be provided")
        return tokens


class SeedUserValues(BaseModel):
    """Define user-provided seeds grouped by role and target.

    This model holds explicit seed lists for all role/target combinations.

    Args:
        train_macro/train_micro: Seeds for train runs.
        eval_macro/eval_micro: Seeds for eval runs.

    Returns:
        SeedUserValues: Validated user seed payload.

    Raises:
        ValueError: If a seed list contains invalid values.
    """

    model_config = ConfigDict(extra="forbid")
    train_macro: list[int] = Field(default_factory=list)
    train_micro: list[int] = Field(default_factory=list)
    eval_macro: list[int] = Field(default_factory=list)
    eval_micro: list[int] = Field(default_factory=list)


class SeedPolicyConfig(BaseModel):
    """Define one seed-policy stream.

    This model configures either auto-increment or user-defined seed behavior
    for sampler or optimizer streams.

    Args:
        mode: Seed mode.
        base: Base seed for auto-increment mode.
        user_values: Explicit seeds for user-defined mode.

    Returns:
        SeedPolicyConfig: Validated seed-policy model.

    Raises:
        ValueError: If ``mode`` is ``user_defined`` without ``user_values``.
    """

    model_config = ConfigDict(extra="forbid")
    mode: SeedMode = "auto_increment"
    base: int | None = None
    user_values: SeedUserValues | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> SeedPolicyConfig:
        """Validate required user values for user-defined mode."""
        if self.mode == "user_defined" and self.user_values is None:
            raise ValueError("user_values must be provided when mode=user_defined")
        return self


class SeedPolicySet(BaseModel):
    """Define seed policies for sampler and optimizer streams.

    This model groups both sampler and optimizer seed policy definitions.

    Args:
        sampler: Sampler seed policy.
        optimizer: Optimizer seed policy.

    Returns:
        SeedPolicySet: Validated seed-policy set model.

    Raises:
        ValueError: If any nested policy is invalid.
    """

    model_config = ConfigDict(extra="forbid")
    sampler: SeedPolicyConfig = Field(default_factory=SeedPolicyConfig)
    optimizer: SeedPolicyConfig = Field(default_factory=SeedPolicyConfig)


class ExecutionTargetConfig(BaseModel):
    """Define execution-target options.

    This model configures local/ABCI execution behavior and job-wrapper options.

    Args:
        target: Execution target name.
        emit_on_prepare: Whether prepare should emit command artifacts.
        job_profile/job_mode/job_walltime/job_log_dir/job_extra_args: Job launcher options.

    Returns:
        ExecutionTargetConfig: Validated execution-target model.

    Raises:
        ValueError: If execution values are invalid.
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
        """Normalize extra job arguments."""
        return _to_tokens(value, field_name="job_extra_args")

    @model_validator(mode="after")
    def _normalize_defaults(self) -> ExecutionTargetConfig:
        """Fill target-specific default job profile."""
        if self.job_profile is None:
            self.job_profile = "sge" if self.target == "abci" else "local"
        return self


class RegressionConfig(BaseModel):
    """Define regression model options.

    This model controls regression backend and backend-specific options.

    Args:
        kind: Regression algorithm name.
        degree: Polynomial degree.
        kernel: Optional GPR kernel name.
        noise: Optional GPR noise value.

    Returns:
        RegressionConfig: Validated regression configuration.

    Raises:
        ValueError: If alias normalization or value validation fails.
    """

    model_config = ConfigDict(extra="forbid")
    kind: str = "linear"
    degree: int = Field(default=1, ge=1)
    kernel: str | None = None
    noise: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        """Map alias fields onto canonical schema keys."""
        if not isinstance(data, Mapping) or "type" not in data:
            return data
        payload = dict(data)
        payload["kind"] = payload.pop("type")
        return payload


class ScenarioConfig(BaseModel):
    """Define one modelbridge scenario.

    This model combines trial counts, objective commands, parameter spaces, and
    regression/metric configuration for one scenario.

    Args:
        name: Scenario name.
        *_trials: Trial counts for each role/target pair.
        train_objective: Train objective command settings.
        eval_objective: Eval objective command settings.
        train_params: Train parameter space.
        eval_params: Eval parameter space.
        regression: Regression model settings.
        metrics: Evaluation metric names.

    Returns:
        ScenarioConfig: Validated scenario configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
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
    """Define HPO base config and override payloads.

    This model holds the base optimize config path and target-specific override
    mappings.

    Args:
        base_config: Base optimize config YAML path.
        macro_overrides: Macro-target config overrides.
        micro_overrides: Micro-target config overrides.
        abci_overrides: Extra overrides for `abci` target.

    Returns:
        HpoSettings: Validated HPO settings model.

    Raises:
        ValueError: If settings payload is malformed.
    """

    model_config = ConfigDict(extra="forbid")
    base_config: Path
    macro_overrides: dict[str, Any] = Field(default_factory=dict)
    micro_overrides: dict[str, Any] = Field(default_factory=dict)
    abci_overrides: dict[str, Any] = Field(default_factory=dict)


class ExternalHpoImportEntry(BaseModel):
    """Define one external HPO artifact mapping entry.

    This model maps one user-generated HPO artifact pair into a modelbridge
    run slot.

    Args:
        scenario: Scenario name in bridge config.
        role: Train or eval role.
        run_id: Run identifier.
        target: Macro or micro target.
        source_hpo_config: Source optimize config path.
        source_optuna_db: Source Optuna DB path.

    Returns:
        ExternalHpoImportEntry: Validated import mapping entry.

    Raises:
        ValueError: If entry payload values are invalid.
    """

    model_config = ConfigDict(extra="forbid")
    scenario: str
    role: BridgeRole
    run_id: int = Field(ge=0)
    target: BridgeTarget
    source_hpo_config: Path
    source_optuna_db: Path


class ExternalHpoImportConfig(BaseModel):
    """Define external HPO import settings.

    This model controls optional import of user-generated HPO artifacts into
    modelbridge run layout.

    Args:
        enabled: Whether import behavior is active.
        entries: External artifact mapping entries.

    Returns:
        ExternalHpoImportConfig: Validated import configuration.

    Raises:
        ValueError: If enabled import has no entries.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    entries: list[ExternalHpoImportEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_entries(self) -> ExternalHpoImportConfig:
        """Validate enabled import settings."""
        if self.enabled and not self.entries:
            raise ValueError("external_hpo_import.entries must be provided when enabled=true")
        return self


class BridgeSettings(BaseModel):
    """Define top-level runtime settings for modelbridge.

    This model stores output, seed, execution, scenario, and import settings
    used by all modelbridge steps.

    Args:
        output_dir: Root output directory.
        seed: Fallback base seed for auto mode.
        seed_policy/execution: Seed and execution settings.
        log_level/json_log: Logging settings.
        *runs: Train/eval run counts.
        strict_mode: Whether issues fail steps.
        scenarios: Scenario configurations.

    Returns:
        BridgeSettings: Validated bridge runtime settings.

    Raises:
        ValueError: If nested seed settings do not match run counts.
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
    external_hpo_import: ExternalHpoImportConfig = Field(default_factory=ExternalHpoImportConfig)

    @model_validator(mode="after")
    def _validate_seed_user_values_length(self) -> BridgeSettings:
        """Validate user-defined seed lengths against run counts."""
        expected = {"train_macro": self.train_runs, "train_micro": self.train_runs}
        expected.update({"eval_macro": self.eval_runs, "eval_micro": self.eval_runs})
        _validate_seed_lengths("sampler", self.seed_policy.sampler, expected)
        _validate_seed_lengths("optimizer", self.seed_policy.optimizer, expected)
        return self


class BridgeConfig(BaseModel):
    """Define validated modelbridge configuration root.

    This model is the root schema consumed by modelbridge runtime steps.

    Args:
        bridge: Runtime bridge settings.
        hpo: HPO generation settings.

    Returns:
        BridgeConfig: Validated root configuration.

    Raises:
        ValueError: If required sections are missing or invalid.
    """

    model_config = ConfigDict(extra="forbid")
    bridge: BridgeSettings
    hpo: HpoSettings


def _validate_seed_lengths(name: str, policy: SeedPolicyConfig, expected: Mapping[str, int]) -> None:
    """Validate user-defined seed list lengths against expected run counts."""
    if policy.mode != "user_defined":
        return
    if policy.user_values is None:
        raise ValueError(f"{name} seed policy requires user_values")
    values = policy.user_values.model_dump()
    for key, expected_len in expected.items():
        actual_len = len(values[key])
        if actual_len != expected_len:
            raise ValueError(f"{name} seed user_values.{key} length must be {expected_len}, got {actual_len}")


def deep_merge_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested mappings.

    Args:
        base: Base mapping.
        override: Override mapping.

    Returns:
        dict[str, Any]: Recursively merged mapping.

    Raises:
        ValueError: If mapping merge cannot be completed.
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
        payload: Raw payload mapping or OmegaConf object.
        overrides: Optional override mapping.

    Returns:
        BridgeConfig: Validated configuration.

    Raises:
        ValueError: If payload is not a mapping or schema validation fails.
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

    Args:
        None.

    Returns:
        dict[str, Any]: JSON schema payload.

    Raises:
        ValueError: If schema generation fails.
    """
    return BridgeConfig.model_json_schema()
