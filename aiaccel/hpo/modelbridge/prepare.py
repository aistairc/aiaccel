"""Prepare steps for modelbridge."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .config import BridgeConfig, HpoSettings, ParameterBounds, ScenarioConfig
from .execution import emit_commands
from .layout import Role, Target, eval_run_dir, plan_path, scenario_dir, train_run_dir
from .toolkit.io import write_json
from .toolkit.results import StepResult, StepStatus, write_step_state
from .toolkit.seeding import resolve_seed


def prepare_train(config: BridgeConfig) -> StepResult:
    """Generate training configs and the train plan.

    Args:
        config: Parsed modelbridge configuration.

    Returns:
        StepResult: Step execution result for `prepare_train`.
    """
    return _prepare_role(config, "train")


def prepare_eval(config: BridgeConfig) -> StepResult:
    """Generate evaluation configs and the eval plan.

    Args:
        config: Parsed modelbridge configuration.

    Returns:
        StepResult: Step execution result for `prepare_eval`.
    """
    return _prepare_role(config, "eval")


def _prepare_role(config: BridgeConfig, role: Role) -> StepResult:
    """Prepare one role by generating per-run optimize configs and a plan."""
    output_dir = config.bridge.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = config.bridge.train_runs if role == "train" else config.bridge.eval_runs
    entries: list[dict[str, Any]] = []
    config_paths: list[str] = []
    emitted_command_path: str | None = None

    for scenario in config.bridge.scenarios:
        scenario_path = scenario_dir(output_dir, scenario.name)
        scenario_path.mkdir(parents=True, exist_ok=True)
        entries.extend(
            _prepare_scenario_role(
                settings=config.hpo,
                scenario=scenario,
                scenario_path=scenario_path,
                role=role,
                runs=runs,
                config=config,
                config_paths=config_paths,
            )
        )

    plan_payload = {
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
    plan_file = write_json(plan_path(output_dir, role), plan_payload)

    if entries:
        status: StepStatus = "success"
        reason = None
        if config.bridge.execution.emit_on_prepare:
            emitted = emit_commands(
                config,
                role=role,
                fmt="shell",
                execution_target=config.bridge.execution.target,
            )
            emitted_command_path = str(emitted)
    else:
        status = "skipped"
        reason = f"No {role} runs configured."

    result = StepResult(
        step=f"prepare_{role}",
        status=status,
        inputs={"role": role, "runs": runs, "execution_target": config.bridge.execution.target},
        outputs={
            "plan_path": str(plan_file),
            "num_entries": len(entries),
            "config_paths": config_paths,
            "command_path": emitted_command_path,
        },
        reason=reason,
    )
    write_step_state(output_dir, result)
    return result


def _prepare_scenario_role(
    settings: HpoSettings,
    scenario: ScenarioConfig,
    scenario_path: Path,
    role: Role,
    runs: int,
    config: BridgeConfig,
    config_paths: list[str],
) -> list[dict[str, Any]]:
    """Prepare all run/target entries for a scenario and role."""
    if runs <= 0:
        return []

    entries: list[dict[str, Any]] = []
    targets: tuple[Target, Target] = ("macro", "micro")

    for run_idx in range(runs):
        for target in targets:
            current_dir = (
                train_run_dir(scenario_path, run_idx, target)
                if role == "train"
                else eval_run_dir(scenario_path, run_idx, target)
            )
            current_dir.mkdir(parents=True, exist_ok=True)

            trials = _select_trials(scenario, role, target)
            command = _select_objective_command(scenario, role)
            params = _select_param_space(scenario, role, target)
            overrides = settings.macro_overrides if target == "macro" else settings.micro_overrides
            sampler_seed = resolve_seed(
                config.bridge.seed_policy.sampler,
                role=role,
                target=target,
                run_id=run_idx,
                fallback_base=config.bridge.seed,
            )
            optimizer_seed = resolve_seed(
                config.bridge.seed_policy.optimizer,
                role=role,
                target=target,
                run_id=run_idx,
                fallback_base=config.bridge.seed,
            )
            seed_mode = _resolve_seed_mode(config)
            study_name = f"{scenario.name}-{role}-{target}-{run_idx:03d}"

            config_file = _generate_hpo_config(
                settings=settings,
                space=params,
                trials=trials,
                sampler_seed=sampler_seed,
                optimizer_seed=optimizer_seed,
                output_dir=current_dir,
                study_name=study_name,
                command=command,
                execution_target=config.bridge.execution.target,
                overrides=overrides,
            )
            config_paths.append(str(config_file))

            entries.append(
                {
                    "scenario": scenario.name,
                    "role": role,
                    "run_id": run_idx,
                    "target": target,
                    "config_path": str(config_file),
                    "expected_db_path": str(current_dir / "optuna.db"),
                    "study_name": study_name,
                    "seed": sampler_seed,
                    "sampler_seed": sampler_seed,
                    "optimizer_seed": optimizer_seed,
                    "seed_mode": seed_mode,
                    "sampler_seed_mode": config.bridge.seed_policy.sampler.mode,
                    "optimizer_seed_mode": config.bridge.seed_policy.optimizer.mode,
                    "execution_target": config.bridge.execution.target,
                    "objective_command": list(command),
                }
            )

    return entries


def _resolve_seed_mode(config: BridgeConfig) -> str:
    """Resolve summary seed mode from sampler/optimizer policies."""
    if (
        config.bridge.seed_policy.sampler.mode == "user_defined"
        or config.bridge.seed_policy.optimizer.mode == "user_defined"
    ):
        return "user_defined"
    return "auto_increment"


def _select_trials(scenario: ScenarioConfig, role: Role, target: Target) -> int:
    """Select trial count for the requested role/target."""
    if role == "train":
        return scenario.train_macro_trials if target == "macro" else scenario.train_micro_trials
    return scenario.eval_macro_trials if target == "macro" else scenario.eval_micro_trials


def _select_objective_command(scenario: ScenarioConfig, role: Role) -> Sequence[str]:
    """Select objective command for the requested role."""
    if role == "train":
        return scenario.train_objective.command
    return scenario.eval_objective.command


def _select_param_space(scenario: ScenarioConfig, role: Role, target: Target) -> dict[str, ParameterBounds]:
    """Select parameter space for the requested role/target."""
    params = scenario.train_params if role == "train" else scenario.eval_params
    return params.macro if target == "macro" else params.micro


def _generate_hpo_config(
    settings: HpoSettings,
    space: dict[str, ParameterBounds],
    trials: int,
    sampler_seed: int,
    optimizer_seed: int,
    output_dir: Path,
    *,
    study_name: str,
    command: Sequence[str],
    execution_target: str,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Generate one `aiaccel-hpo optimize` config file from the base config.

    Args:
        settings: HPO setting block including base config path and overrides.
        space: Parameter space for the target.
        trials: Number of trials to run.
        sampler_seed: Sampler seed.
        optimizer_seed: Optimizer seed.
        output_dir: Working directory for one run/target.
        study_name: Study name used in Optuna storage.
        command: Objective command template.
        execution_target: Execution target (`local` or `abci`).
        overrides: Optional role/target specific OmegaConf overrides.

    Returns:
        Path: Written config file path.
    """
    db_path = output_dir / "optuna.db"
    storage_uri = f"sqlite:///{db_path.resolve()}"

    base_conf = cast(DictConfig, OmegaConf.load(settings.base_config))
    if overrides:
        base_conf = cast(DictConfig, OmegaConf.merge(base_conf, OmegaConf.create(overrides)))
    if execution_target == "abci" and settings.abci_overrides:
        base_conf = cast(DictConfig, OmegaConf.merge(base_conf, OmegaConf.create(settings.abci_overrides)))

    OmegaConf.update(base_conf, "n_trials", trials)
    OmegaConf.update(base_conf, "working_directory", str(output_dir))
    OmegaConf.update(base_conf, "command", list(command))
    OmegaConf.update(base_conf, "optimize.rand_seed", optimizer_seed, merge=True)

    optimize_conf = base_conf.get("optimize", {})
    goal = optimize_conf.get("goal", "minimize")
    direction = "minimize" if goal == "minimize" else "maximize"

    params_def: dict[str, Any] = {"_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager"}
    for name, bounds in space.items():
        params_def[name] = {
            "_target_": "aiaccel.hpo.optuna.hparams.Float",
            "low": bounds.low,
            "high": bounds.high,
            "log": bounds.log,
            "step": bounds.step,
        }
    OmegaConf.update(base_conf, "params", params_def)

    OmegaConf.update(
        base_conf,
        "study",
        {
            "_target_": "optuna.create_study",
            "study_name": study_name,
            "storage": storage_uri,
            "direction": direction,
            "load_if_exists": True,
            "sampler": {
                "_target_": "optuna.samplers.TPESampler",
                "seed": sampler_seed,
            },
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(base_conf, config_path)
    return config_path
