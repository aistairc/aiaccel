"""Prepare steps: generate per-run optimize configs and role plans."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import shutil

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
    """Prepare optimize configs and plan entries for train role.

    This step materializes train-role run directories, run-level optimize
    config files, and the train plan manifest.

    Args:
        config: Validated modelbridge configuration.

    Returns:
        StepResult: Execution result for ``prepare_train``.

    Raises:
        FileNotFoundError: If referenced base config or import artifacts are missing.
        ValueError: If resolved config payloads are malformed.
    """
    return _prepare_role(config, "train")


def prepare_eval(config: BridgeConfig) -> StepResult:
    """Prepare optimize configs and plan entries for eval role.

    This step materializes eval-role run directories, run-level optimize
    config files, and the eval plan manifest.

    Args:
        config: Validated modelbridge configuration.

    Returns:
        StepResult: Execution result for ``prepare_eval``.

    Raises:
        FileNotFoundError: If referenced base config or import artifacts are missing.
        ValueError: If resolved config payloads are malformed.
    """
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
    imported_entries = _import_external_hpo_results(config=config, role=role)

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
            "num_imported_entries": len(imported_entries),
            "imported_entries": imported_entries,
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


def _import_external_hpo_results(config: BridgeConfig, *, role: Role) -> list[dict[str, str | int]]:
    """Import user-provided HPO artifacts into modelbridge run layout."""
    import_cfg = config.bridge.external_hpo_import
    if not import_cfg.enabled:
        return []

    scenario_names = {scenario.name for scenario in config.bridge.scenarios}
    max_runs = config.bridge.train_runs if role == "train" else config.bridge.eval_runs
    imported: list[dict[str, str | int]] = []

    for entry in import_cfg.entries:
        if entry.role != role:
            continue
        if entry.scenario not in scenario_names:
            raise ValueError(f"external_hpo_import scenario not found: {entry.scenario}")
        if entry.run_id >= max_runs:
            raise ValueError(f"external_hpo_import run_id out of range for role={role}: {entry.run_id}")
        if not entry.source_hpo_config.exists():
            raise FileNotFoundError(f"external_hpo_import source_hpo_config not found: {entry.source_hpo_config}")
        if not entry.source_optuna_db.exists():
            raise FileNotFoundError(f"external_hpo_import source_optuna_db not found: {entry.source_optuna_db}")

        run_dir = run_path(
            scenario_path(config.bridge.output_dir, entry.scenario),
            role=role,
            run_id=entry.run_id,
            target=entry.target,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        config_dest = run_dir / "config.yaml"
        db_dest = run_dir / "optuna.db"
        shutil.copy2(entry.source_hpo_config, config_dest)
        shutil.copy2(entry.source_optuna_db, db_dest)
        imported.append(
            {
                "scenario": entry.scenario,
                "role": entry.role,
                "run_id": entry.run_id,
                "target": entry.target,
                "config_path": str(config_dest),
                "db_path": str(db_dest),
            }
        )
    return imported
