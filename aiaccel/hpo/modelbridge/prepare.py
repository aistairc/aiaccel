"""Prepare steps for modelbridge."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .config import BridgeConfig, HpoSettings, ParameterBounds, ScenarioConfig
from .layout import Role, Target, plan_path, run_dir, scenario_dir
from .toolkit.io import write_json
from .toolkit.results import StepResult, StepStatus, write_step_state

RUN_ID_OFFSET_STRIDE = 100000


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
                seed_base=config.bridge.seed,
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
    else:
        status = "skipped"
        reason = f"No {role} runs configured."

    result = StepResult(
        step=f"prepare_{role}",
        status=status,
        inputs={"role": role, "runs": runs},
        outputs={
            "plan_path": str(plan_file),
            "num_entries": len(entries),
            "config_paths": config_paths,
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
    seed_base: int,
    config_paths: list[str],
) -> list[dict[str, Any]]:
    """Prepare all run/target entries for a scenario and role."""
    if runs <= 0:
        return []

    entries: list[dict[str, Any]] = []
    targets: tuple[Target, Target] = ("macro", "micro")

    for run_idx in range(runs):
        for target in targets:
            current_dir = run_dir(scenario_path, role, run_idx, target)
            current_dir.mkdir(parents=True, exist_ok=True)

            trials = _select_trials(scenario, role, target)
            command = _select_objective_command(scenario, role)
            params = _select_param_space(scenario, role, target)
            overrides = settings.macro_overrides if target == "macro" else settings.micro_overrides
            seed = _compute_seed(seed_base, role, target, run_idx)
            study_name = f"{scenario.name}-{role}-{target}-{run_idx:03d}"

            config_file = _generate_hpo_config(
                settings=settings,
                space=params,
                trials=trials,
                seed=seed,
                output_dir=current_dir,
                study_name=study_name,
                command=command,
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
                    "seed": seed,
                    "objective_command": list(command),
                }
            )

    return entries


def _compute_seed(seed_base: int, role: Role, target: Target, run_idx: int) -> int:
    """Compute deterministic seed offset by role, target, and run index."""
    group_idx = 0
    if role == "eval":
        group_idx += 2
    if target == "micro":
        group_idx += 1
    return seed_base + group_idx * RUN_ID_OFFSET_STRIDE + run_idx


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
    seed: int,
    output_dir: Path,
    *,
    study_name: str,
    command: Sequence[str],
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Generate one `aiaccel-hpo optimize` config file from the base config.

    Args:
        settings: HPO setting block including base config path and overrides.
        space: Parameter space for the target.
        trials: Number of trials to run.
        seed: Sampler seed.
        output_dir: Working directory for one run/target.
        study_name: Study name used in Optuna storage.
        command: Objective command template.
        overrides: Optional role/target specific OmegaConf overrides.

    Returns:
        Path: Written config file path.
    """
    db_path = output_dir / "optuna.db"
    storage_uri = f"sqlite:///{db_path.resolve()}"

    base_conf = cast(DictConfig, OmegaConf.load(settings.base_config))
    if overrides:
        base_conf = cast(DictConfig, OmegaConf.merge(base_conf, OmegaConf.create(overrides)))

    OmegaConf.update(base_conf, "n_trials", trials)
    OmegaConf.update(base_conf, "working_directory", str(output_dir))
    OmegaConf.update(base_conf, "command", list(command))

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
                "seed": seed,
            },
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(base_conf, config_path)
    return config_path
