"""Optuna integration helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import optuna

from .config import ParameterBounds
from .types import EvaluationResult, TrialContext, TrialResult


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


def run_phase(
    *,
    scenario: str,
    phase: str,
    trials: int,
    space: dict[str, ParameterBounds],
    evaluator: Callable[[TrialContext], EvaluationResult],
    seed: int,
    output_dir: Path,
    storage: str | None = None,
    study_name: str | None = None,
    write_csv: bool = False,
) -> PhaseOutcome:
    """Execute an optimisation phase and return the collected trials."""

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_study_name = study_name or f"{scenario}-{phase}"
    resolved_storage = storage
    if resolved_storage is None:
        db_path = output_dir / f"{phase}_optuna.db"
        resolved_storage = f"sqlite:///{db_path.resolve()}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=resolved_study_name,
            storage=resolved_storage,
            load_if_exists=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to create or load study '{resolved_study_name}'") from exc

    collected: list[TrialResult] = []

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, space)
        context = TrialContext(
            scenario=scenario,
            phase=phase,
            trial_index=trial.number,
            params=params,
            seed=seed,
            output_dir=output_dir,
        )
        evaluation = evaluator(context)
        trial.set_user_attr("metrics", evaluation.metrics)
        trial.set_user_attr("payload", evaluation.payload)
        trial.set_user_attr("seed", context.seed)
        collected.append(TrialResult(context=context, evaluation=evaluation, state="COMPLETE"))
        return evaluation.objective

    remaining = max(0, trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    if remaining > 0:
        try:
            study.optimize(objective, n_trials=remaining, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Phase '{scenario}:{phase}' terminated unexpectedly") from exc

    trials_payload = collect_trial_results(
        study=study,
        scenario=scenario,
        phase=phase,
        output_dir=output_dir,
    )

    if write_csv:
        _persist_trials(output_dir / f"{phase}_trials.csv", trials_payload)

    return PhaseOutcome(study=study, trials=trials_payload)


def _suggest_params(trial: optuna.Trial, space: dict[str, ParameterBounds]) -> dict[str, float]:
    """Sample parameters from ``space`` using Optuna's suggestion API."""

    params: dict[str, float] = {}
    for name, bounds in space.items():
        if bounds.step is not None:
            params[name] = trial.suggest_float(name, bounds.low, bounds.high, step=float(bounds.step))
        elif bounds.log:
            params[name] = trial.suggest_float(name, bounds.low, bounds.high, log=True)
        else:
            params[name] = trial.suggest_float(name, bounds.low, bounds.high)
    return params


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
        evaluation = EvaluationResult(objective=float(trial.value), metrics=metrics, payload=payload)
        results.append(TrialResult(context=context, evaluation=evaluation, state=str(trial.state)))

    results.sort(key=lambda item: item.context.trial_index)
    return results


def _persist_trials(path: Path, trials: list[TrialResult]) -> None:
    rows = []
    for result in trials:
        row = {"trial_index": result.context.trial_index, **result.context.params}
        row["objective"] = result.evaluation.objective
        for metric_name, metric_value in result.evaluation.metrics.items():
            row[f"metric_{metric_name}"] = metric_value
        rows.append(row)
    if rows:
        from .io import write_csv

        write_csv(path, rows)


__all__ = ["PhaseOutcome", "collect_trial_results", "run_phase"]
