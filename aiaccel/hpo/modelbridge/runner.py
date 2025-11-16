"""High level orchestration for the modelbridge pipeline."""

from __future__ import annotations

from typing import Any, Literal

from collections.abc import Iterable, Sequence
import itertools
from pathlib import Path

from .config import BridgeConfig, ScenarioConfig
from .evaluators import build_evaluator
from .exceptions import ExecutionError, ValidationError
from .io import read_csv, read_json, write_csv, write_json
from .logging import configure_logging, get_logger
from .optimizers import run_phase
from .regression import evaluate as evaluate_regression
from .regression import fit as fit_regression
from .scenario import build_plan
from .summary import ScenarioSummary, SummaryBuilder
from .types import EvaluationResult, RegressionSample, TrialContext, TrialResult

PhaseName = Literal["macro", "micro", "regress", "summary"]
PHASE_ORDER: tuple[PhaseName, ...] = ("macro", "micro", "regress", "summary")
SCENARIO_SUMMARY_FILE = "scenario_summary.json"


def run_pipeline(
    config: BridgeConfig,
    *,
    phases: Sequence[PhaseName] | None = None,
    scenarios: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Execute the modelbridge pipeline described by ``config``."""

    requested_phases = _normalize_phases(phases)
    scenario_filter = set(scenarios or [])

    settings = config.bridge
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(settings.log_level, output_dir)
    logger = get_logger(__name__)
    logger.info("Starting modelbridge pipeline -> output_dir=%s", output_dir)

    run_summary = "summary" in requested_phases
    summary_builder = SummaryBuilder(output_dir=output_dir) if run_summary else None

    base_env = {
        "AIACCEL_OUTPUT_DIR": str(output_dir),
    }
    if settings.working_directory:
        base_env["AIACCEL_WORK_DIR"] = str(settings.working_directory)

    processed: list[str] = []

    for scenario_cfg in settings.scenarios:
        if scenario_filter and scenario_cfg.name not in scenario_filter:
            continue
        processed.append(scenario_cfg.name)
        logger.info("Running scenario %s (phases=%s)", scenario_cfg.name, ",".join(requested_phases))
        summary = _execute_scenario(
            scenario_cfg,
            settings.seed,
            requested_phases,
            base_env,
            output_dir,
        )
        if summary_builder and summary:
            summary_builder.add(scenario_cfg.name, summary)

    if not processed:
        raise ValidationError("No scenarios matched the requested filters")

    if summary_builder:
        summary_final = summary_builder.finalize()
        logger.info("Pipeline summary generated")
        return summary_final

    logger.info("Pipeline phases completed: %s", ",".join(requested_phases))
    return {"phases": list(requested_phases), "scenarios": processed}


def _execute_scenario(
    config: ScenarioConfig,
    seed: int,
    phases: Sequence[PhaseName],
    base_env: dict[str, str],
    output_dir: Path,
) -> ScenarioSummary | None:
    plan = build_plan(config)
    scenario_dir = output_dir / "scenarios" / config.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    evaluator = build_evaluator(config.objective, base_env=base_env)

    macro_trials: list[TrialResult] | None = None
    micro_trials: list[TrialResult] | None = None

    if "macro" in phases:
        macro = run_phase(
            scenario=config.name,
            phase=plan.macro.phase,
            trials=plan.macro.trials,
            space=plan.macro.space,
            evaluator=evaluator,
            seed=seed,
            output_dir=scenario_dir,
        )
        macro_trials = macro.trials
        _persist_trials(scenario_dir / "macro_trials.csv", macro.trials)
    elif "regress" in phases:
        macro_trials = _load_trials_from_csv(scenario_dir / "macro_trials.csv", config.name, "macro", scenario_dir)

    if "micro" in phases:
        micro = run_phase(
            scenario=config.name,
            phase=plan.micro.phase,
            trials=plan.micro.trials,
            space=plan.micro.space,
            evaluator=evaluator,
            seed=seed + 1,
            output_dir=scenario_dir,
        )
        micro_trials = micro.trials
        _persist_trials(scenario_dir / "micro_trials.csv", micro.trials)
    elif "regress" in phases:
        micro_trials = _load_trials_from_csv(scenario_dir / "micro_trials.csv", config.name, "micro", scenario_dir)

    scenario_summary: ScenarioSummary | None = None

    if "regress" in phases:
        if macro_trials is None or micro_trials is None:
            raise ExecutionError(f"Scenario '{config.name}' requires macro/micro trials before regression")
        scenario_summary = _run_regression(
            config,
            macro_trials,
            micro_trials,
            scenario_dir,
        )

    if "summary" in phases and scenario_summary is None:
        scenario_summary = _load_scenario_summary(scenario_dir, config.name)

    return scenario_summary


def _run_regression(
    config: ScenarioConfig,
    macro_trials: list[TrialResult],
    micro_trials: list[TrialResult],
    scenario_dir: Path,
) -> ScenarioSummary:
    regression_samples = _compose_samples(macro_trials, micro_trials)
    model = fit_regression(regression_samples, degree=config.regression.degree)
    metrics_raw = evaluate_regression(model, regression_samples)
    metrics = {name: float(metrics_raw.get(name, 0.0)) for name in config.metrics}

    predictions = [model.predict(sample.features) for sample in regression_samples]

    write_json(scenario_dir / "regression.json", model.to_dict())
    _persist_predictions(scenario_dir / "predictions.csv", regression_samples, predictions)

    summary = ScenarioSummary(
        macro_trials=len(macro_trials),
        micro_trials=len(micro_trials),
        macro_best=_best_params(macro_trials),
        micro_best=_best_params(micro_trials),
        metrics=metrics,
    )
    _persist_scenario_summary(scenario_dir, summary)
    return summary


def _normalize_phases(phases: Sequence[PhaseName] | None) -> tuple[PhaseName, ...]:
    if not phases:
        return PHASE_ORDER
    normalized: list[PhaseName] = []
    for phase in phases:
        if phase not in PHASE_ORDER:
            raise ValidationError(f"Unknown phase '{phase}'")
        if phase not in normalized:
            normalized.append(phase)
    normalized.sort(key=lambda name: PHASE_ORDER.index(name))
    return tuple(normalized)


def _load_trials_from_csv(
    path: Path,
    scenario: str,
    phase: str,
    output_dir: Path,
) -> list[TrialResult]:
    if not path.exists():
        raise ExecutionError(f"Missing trials file for scenario '{scenario}' phase '{phase}'")
    rows = read_csv(path)
    results: list[TrialResult] = []
    for index, row in enumerate(rows):
        trial_index = int(row.get("trial_index", index))
        params: dict[str, float] = {}
        metrics: dict[str, float] = {}
        objective = float(row["objective"])
        for key, value in row.items():
            if key in {"trial_index", "objective"}:
                continue
            if key.startswith("metric_"):
                metrics[key.removeprefix("metric_")] = float(value)
            elif value != "":
                params[key] = float(value)
        context = TrialContext(
            scenario=scenario,
            phase=phase,
            trial_index=trial_index,
            params=params,
            seed=0,
            output_dir=output_dir,
        )
        evaluation = EvaluationResult(objective=objective, metrics=metrics)
        results.append(TrialResult(context=context, evaluation=evaluation, state="COMPLETE"))
    return results


def _load_scenario_summary(path: Path, scenario: str) -> ScenarioSummary:
    summary_path = path / SCENARIO_SUMMARY_FILE
    if not summary_path.exists():
        raise ExecutionError(f"Scenario '{scenario}' does not have a saved regression summary")
    payload = read_json(summary_path)
    return ScenarioSummary(
        macro_trials=int(payload["macro_trials"]),
        micro_trials=int(payload["micro_trials"]),
        macro_best=dict(payload["macro_best"]),
        micro_best=dict(payload["micro_best"]),
        metrics=dict(payload["metrics"]),
    )


def _persist_scenario_summary(path: Path, summary: ScenarioSummary) -> None:
    write_json(
        path / SCENARIO_SUMMARY_FILE,
        {
            "macro_trials": summary.macro_trials,
            "micro_trials": summary.micro_trials,
            "macro_best": summary.macro_best,
            "micro_best": summary.micro_best,
            "metrics": summary.metrics,
        },
    )


def _best_params(trials: Iterable[TrialResult]) -> dict[str, float]:
    best_result = min(trials, key=lambda result: result.evaluation.objective)
    return dict(best_result.context.params)


def _compose_samples(macro: Iterable[TrialResult], micro: Iterable[TrialResult]) -> list[RegressionSample]:
    """Pair macro and micro trials into regression samples."""

    macro_sorted = sorted(macro, key=lambda item: item.context.trial_index)
    micro_sorted = sorted(micro, key=lambda item: item.context.trial_index)
    pairs = itertools.zip_longest(macro_sorted, micro_sorted, fillvalue=None)
    samples: list[RegressionSample] = []
    for macro_result, micro_result in pairs:
        if macro_result is None or micro_result is None:
            break
        samples.append(
            RegressionSample(
                features=dict(macro_result.context.params),
                target=dict(micro_result.context.params),
            )
        )
    if not samples:
        raise ExecutionError("No trials collected for regression")
    return samples


def _persist_trials(path: Path, trials: Iterable[TrialResult]) -> None:
    """Serialise ``trials`` to ``path`` as a CSV file."""

    rows = []
    for result in trials:
        row = {"trial_index": result.context.trial_index, **result.context.params}
        row["objective"] = result.evaluation.objective
        for metric_name, metric_value in result.evaluation.metrics.items():
            row[f"metric_{metric_name}"] = metric_value
        rows.append(row)
    write_csv(path, rows)


def _persist_predictions(
    path: Path,
    samples: Iterable[RegressionSample],
    predictions: Iterable[dict[str, float]],
) -> None:
    """Serialise predicted vs actual micro parameters."""

    rows = []
    for sample, predicted in zip(samples, predictions, strict=True):
        row = {}
        for name, value in sample.target.items():
            row[f"actual_{name}"] = value
        for name, value in predicted.items():
            row[f"pred_{name}"] = value
        for name, value in sample.features.items():
            row[f"macro_{name}"] = value
        rows.append(row)
    if rows:
        write_csv(path, rows)


__all__ = ["run_pipeline", "PHASE_ORDER"]
