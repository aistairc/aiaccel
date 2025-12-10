"""External HPO execution via aiaccel-hpo optimize."""

from __future__ import annotations

import contextlib
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import optuna
from omegaconf import OmegaConf

from .config import HpoSettings, ParameterBounds
from .types import TrialResult, TrialContext, EvaluationResult


@dataclass(slots=True)
class PhaseOutcome:
    """Result of a completed optimisation phase."""

    study: optuna.Study
    trials: list[TrialResult]

    @property
    def best_params(self) -> dict[str, float]:
        if self.study.best_trial is None:
            return {}
        return dict(self.study.best_trial.params)

    @property
    def best_value(self) -> float | None:
        if self.study.best_trial is None:
            return None
        return float(self.study.best_value)


def run_hpo(
    *,
    hpo_settings: HpoSettings,
    scenario: str,
    phase: str,  # "macro" or "micro"
    trials: int,
    space: dict[str, ParameterBounds],
    seed: int,
    output_dir: Path,
    storage: str | None = None,
    study_name: str | None = None,
    command: list[str] | None = None,
) -> PhaseOutcome:
    """Execute aiaccel-hpo optimize as a subprocess."""

    if hpo_settings.base_config is None:
        raise ValueError("base_config is required for external execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_study_name = study_name or f"{scenario}-{phase}"
    resolved_storage = storage
    if resolved_storage is None:
        db_path = output_dir / "optuna.db"
        resolved_storage = f"sqlite:///{db_path.resolve()}"

    base_conf = OmegaConf.load(hpo_settings.base_config)

    # Merge overrides
    overrides = hpo_settings.macro_overrides if phase == "macro" else hpo_settings.micro_overrides
    if overrides:
        base_conf = OmegaConf.merge(base_conf, OmegaConf.create(overrides))

    # Force critical settings for aiaccel-hpo optimize
    OmegaConf.update(base_conf, "n_trials", trials)
    OmegaConf.update(base_conf, "n_max_jobs", 1)  # Sequential execution per run_id
    OmegaConf.update(base_conf, "working_directory", str(output_dir))
    
    # Inject params (HparamsManager style)
    _inject_params(base_conf, space)

    # Inject study config
    # aiaccel-hpo optimize requires 'study' key with hydra instantiation config
    goal = base_conf.get("optimize", {}).get("goal", "minimize")
    direction = "minimize" if goal == "minimize" else "maximize"
    
    study_conf = {
        "_target_": "optuna.create_study",
        "study_name": resolved_study_name,
        "storage": resolved_storage,
        "direction": direction,
        "load_if_exists": True,
        "sampler": {
            "_target_": "optuna.samplers.TPESampler",
            "seed": seed,
        }
    }
    OmegaConf.update(base_conf, "study", study_conf)

    # Write temp config
    temp_config_path = output_dir / "optimize_config.yaml"
    OmegaConf.save(base_conf, temp_config_path)

    cmd = hpo_settings.optimize_command
    cmd_args = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    # Use absolute path for config
    cmd_args.extend(["--config", str(temp_config_path.resolve())])
    
    # Pass objective command as positional arguments if provided
    if command:
        cmd_args.append("--")
        cmd_args.extend(command)

    # Run the process
    # We execute from current directory (project root) to preserve relative paths
    try:
        subprocess.run(cmd_args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Optimization process failed: {exc.stderr}") from exc

    # Load results
    # Try to load from the passed storage first
    with contextlib.suppress(Exception):
        study = optuna.load_study(study_name=resolved_study_name, storage=resolved_storage)

    # Re-loading study from the storage URI provided
    try:
        study = optuna.load_study(study_name=resolved_study_name, storage=resolved_storage)
    except Exception as exc:
        raise RuntimeError(f"Could not load study '{resolved_study_name}' from '{resolved_storage}' after optimization") from exc

    trials_payload = collect_trial_results(
        study=study,
        scenario=scenario,
        phase=phase,
        output_dir=output_dir,
    )

    return PhaseOutcome(study=study, trials=trials_payload)


def _inject_params(conf: Any, space: dict[str, ParameterBounds]) -> None:
    params_def = {"_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager"}
    for name, bounds in space.items():
        params_def[name] = {
            "_target_": "aiaccel.hpo.optuna.hparams.Float",
            "low": bounds.low,
            "high": bounds.high,
            "log": bounds.log,
            "step": bounds.step,
        }
    OmegaConf.update(conf, "params", params_def)


def collect_trial_results(
    *,
    study: optuna.Study,
    scenario: str,
    phase: str,
    output_dir: Path,
) -> list[TrialResult]:
    results: list[TrialResult] = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        context = TrialContext(
            scenario=scenario,
            phase=phase,
            trial_index=trial.number,
            params={k: float(v) for k, v in trial.params.items()},
            seed=trial.user_attrs.get("seed", 0),
            output_dir=output_dir,
        )
        metrics = {str(k): float(v) for k, v in trial.user_attrs.get("metrics", {}).items()}
        payload = trial.user_attrs.get("payload", {})
        if trial.value is None:
            continue
        evaluation = EvaluationResult(objective=float(trial.value), metrics=metrics, payload=payload)
        results.append(TrialResult(context=context, evaluation=evaluation, state=str(trial.state)))

    results.sort(key=lambda item: item.context.trial_index)
    return results


__all__ = ["PhaseOutcome", "collect_trial_results", "run_hpo"]
