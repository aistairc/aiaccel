"""Lightweight modelbridge pipeline integration."""

from .config import (
    BridgeConfig,
    BridgeSettings,
    DataAssimilationConfig,
    ObjectiveConfig,
    ParameterSpace,
    ScenarioConfig,
    generate_schema,
)
from .data_assimilation import run_data_assimilation
from .runner import PipelinePlan, execute_pipeline, plan_pipeline, run_pipeline
from .types import EvaluationResult, EvaluatorFn, PhaseContext, RunContext, RunnerFn, TrialContext, TrialResult

__all__ = [
    "BridgeConfig",
    "BridgeSettings",
    "DataAssimilationConfig",
    "ObjectiveConfig",
    "ParameterSpace",
    "ScenarioConfig",
    "generate_schema",
    "run_data_assimilation",
    "EvaluationResult",
    "EvaluatorFn",
    "RunnerFn",
    "RunContext",
    "PhaseContext",
    "TrialContext",
    "TrialResult",
    "PipelinePlan",
    "plan_pipeline",
    "execute_pipeline",
    "run_pipeline",
]
