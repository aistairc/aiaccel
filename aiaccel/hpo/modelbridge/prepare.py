"""Prepare steps: generate per-run optimize configs and role plans."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .common import (
    Role,
    StepResult,
    Target,
    plan_path,
    resolve_seed,
    run_path,
    scenario_path,
    write_json,
    write_step_state,
)
from .config import BridgeConfig, ExecutionTarget, HpoSettings, ParameterBounds, ScenarioConfig
from .execution import emit_commands


def prepare_train(config: BridgeConfig) -> StepResult:
    """Prepare optimize configs and plan entries for train role."""
    return _prepare_role(config, "train")


def prepare_eval(config: BridgeConfig) -> StepResult:
    """Prepare optimize configs and plan entries for eval role."""
    return _prepare_role(config, "eval")


def _prepare_role(config: BridgeConfig, role: Role) -> StepResult:
    """Prepare all runs for one role and write a role plan."""
    output_dir = config.bridge.output_dir
    runs = config.bridge.train_runs if role == "train" else config.bridge.eval_runs
    config_paths: list[str] = []
    entries: list[dict[str, Any]] = []

    for scenario in config.bridge.scenarios:
        entries.extend(
            _prepare_run_target(
                config=config,
                scenario=scenario,
                scenario_output=scenario_path(output_dir, scenario.name),
                role=role,
                run_id=run_id,
                target=cast(Target, target),
                config_paths=config_paths,
            )
            for run_id in range(runs)
            for target in ("macro", "micro")
        )

    plan_file = plan_path(output_dir, role)
    write_json(plan_file, {"role": role, "created_at": datetime.now(timezone.utc).isoformat(), "entries": entries})

    emitted_command_path = (
        str(emit_commands(config, role=role, fmt="shell", execution_target=config.bridge.execution.target))
        if entries and config.bridge.execution.emit_on_prepare
        else None
    )
    result = StepResult(
        step=f"prepare_{role}",
        status="success" if entries else "skipped",
        inputs={"role": role, "runs": runs, "execution_target": config.bridge.execution.target},
        outputs={
            "plan_path": str(plan_file),
            "num_entries": len(entries),
            "config_paths": config_paths,
            "command_path": emitted_command_path,
        },
        reason=None if entries else f"No {role} runs configured",
    )
    write_step_state(output_dir, result)
    return result


def _prepare_run_target(
    *,
    config: BridgeConfig,
    scenario: ScenarioConfig,
    scenario_output: Path,
    role: Role,
    run_id: int,
    target: Target,
    config_paths: list[str],
) -> dict[str, Any]:
    """Build one plan entry and per-run optimize config."""
    current_dir = run_path(scenario_output, role, run_id, target)
    current_dir.mkdir(parents=True, exist_ok=True)
    objective_command = list(getattr(scenario, f"{role}_objective").command)

    sampler_seed = resolve_seed(
        config.bridge.seed_policy.sampler,
        role=role,
        target=target,
        run_id=run_id,
        fallback_base=config.bridge.seed,
    )
    optimizer_seed = resolve_seed(
        config.bridge.seed_policy.optimizer,
        role=role,
        target=target,
        run_id=run_id,
        fallback_base=config.bridge.seed,
    )
    policy_modes = {config.bridge.seed_policy.sampler.mode, config.bridge.seed_policy.optimizer.mode}
    seed_mode = "user_defined" if "user_defined" in policy_modes else "auto_increment"

    config_path = _generate_hpo_config(
        settings=config.hpo,
        space=dict(getattr(getattr(scenario, f"{role}_params"), target)),
        trials=int(getattr(scenario, f"{role}_{target}_trials")),
        sampler_seed=sampler_seed,
        optimizer_seed=optimizer_seed,
        output_dir=current_dir,
        study_name=f"{scenario.name}-{role}-{target}-{run_id:03d}",
        command=objective_command,
        execution_target=config.bridge.execution.target,
        overrides=config.hpo.macro_overrides if target == "macro" else config.hpo.micro_overrides,
    )
    config_paths.append(str(config_path))

    return {
        "scenario": scenario.name,
        "role": role,
        "run_id": run_id,
        "target": target,
        "config_path": str(config_path),
        "expected_db_path": str(current_dir / "optuna.db"),
        "study_name": f"{scenario.name}-{role}-{target}-{run_id:03d}",
        "seed": sampler_seed,
        "sampler_seed": sampler_seed,
        "optimizer_seed": optimizer_seed,
        "seed_mode": seed_mode,
        "sampler_seed_mode": config.bridge.seed_policy.sampler.mode,
        "optimizer_seed_mode": config.bridge.seed_policy.optimizer.mode,
        "execution_target": config.bridge.execution.target,
        "objective_command": objective_command,
    }


def _generate_hpo_config(
    *,
    settings: HpoSettings,
    space: dict[str, ParameterBounds],
    trials: int,
    sampler_seed: int,
    optimizer_seed: int,
    output_dir: Path,
    study_name: str,
    command: Sequence[str],
    execution_target: ExecutionTarget,
    overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Render one optimize configuration file for a run/target."""
    db_path = output_dir / "optuna.db"
    storage_uri = f"sqlite:///{db_path.resolve()}"

    conf = OmegaConf.load(settings.base_config)
    if not isinstance(conf, DictConfig):
        raise ValueError(f"Base config must be a mapping: {settings.base_config}")
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.create(dict(overrides)))
    if execution_target == "abci" and settings.abci_overrides:
        conf = OmegaConf.merge(conf, OmegaConf.create(settings.abci_overrides))

    for key, value in {
        "n_trials": trials,
        "working_directory": str(output_dir),
        "command": list(command),
    }.items():
        OmegaConf.update(conf, key, value)
    OmegaConf.update(conf, "optimize.rand_seed", optimizer_seed, merge=True)

    goal = OmegaConf.select(conf, "optimize.goal", default="minimize")
    direction = "maximize" if goal == "maximize" else "minimize"
    params_def: dict[str, Any] = {"_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager"}
    params_def.update(
        {
            name: {
                "_target_": "aiaccel.hpo.optuna.hparams.Float",
                "low": bounds.low,
                "high": bounds.high,
                "log": bounds.log,
                "step": bounds.step,
            }
            for name, bounds in space.items()
        }
    )
    OmegaConf.update(conf, "params", params_def)
    OmegaConf.update(
        conf,
        "study",
        {
            "_target_": "optuna.create_study",
            "study_name": study_name,
            "storage": storage_uri,
            "direction": direction,
            "load_if_exists": True,
            "sampler": {"_target_": "optuna.samplers.TPESampler", "seed": sampler_seed},
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(conf, config_path)
    return config_path
