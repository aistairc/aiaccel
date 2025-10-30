"""Optuna integration helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import optuna

from .config import ParameterBounds
from .exceptions import ExecutionError
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
) -> PhaseOutcome:
    """Execute an optimisation phase and return the collected trials."""

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
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
        collected.append(TrialResult(context=context, evaluation=evaluation, state="COMPLETE"))
        return evaluation.objective

    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
    except Exception as exc:  # noqa: BLE001
        raise ExecutionError(f"Phase '{scenario}:{phase}' terminated unexpectedly") from exc

    return PhaseOutcome(study=study, trials=collected)


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


__all__ = ["PhaseOutcome", "run_phase"]
