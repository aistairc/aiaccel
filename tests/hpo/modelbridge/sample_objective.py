"""Sample objective functions used in modelbridge tests."""

from __future__ import annotations

from collections.abc import Mapping

from aiaccel.hpo.modelbridge.types import EvaluationResult, TrialContext


def objective(context: TrialContext, base_env: Mapping[str, str] | None = None) -> EvaluationResult:
    """Return linear score using the trial parameters."""

    score = sum(context.params.values())
    payload = {"base_env": dict(base_env or {})}
    return EvaluationResult(objective=score, metrics={"mae": score}, payload=payload)


def stateless_objective(context: TrialContext) -> EvaluationResult:
    """Objective that ignores ``base_env`` entirely."""

    score = len(context.params)
    return EvaluationResult(objective=float(score), metrics={"mae": float(score)})
