"""High level orchestration for the modelbridge pipeline."""

from __future__ import annotations

from collections.abc import Iterable
import itertools
from pathlib import Path
from typing import Any

from .config import BridgeConfig
from .evaluators import build_evaluator
from .exceptions import ExecutionError
from .io import write_csv, write_json
from .logging import configure_logging, get_logger
from .optimizers import run_phase
from .regression import evaluate as evaluate_regression
from .regression import fit as fit_regression
from .scenario import build_plan
from .summary import ScenarioSummary, SummaryBuilder
from .types import RegressionSample, TrialResult


def run_pipeline(config: BridgeConfig) -> dict[str, Any]:
    """Execute the modelbridge pipeline described by ``config``."""

    settings = config.bridge
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(settings.log_level, output_dir)
    logger = get_logger(__name__)
    logger.info("Starting modelbridge pipeline -> output_dir=%s", output_dir)

    summary_builder = SummaryBuilder(output_dir=output_dir)
    base_env = {
        "AIACCEL_OUTPUT_DIR": str(output_dir),
    }
    if settings.working_directory:
        base_env["AIACCEL_WORK_DIR"] = str(settings.working_directory)

    for scenario_cfg in settings.scenarios:
        logger.info("Running scenario %s", scenario_cfg.name)
        plan = build_plan(scenario_cfg)
        scenario_dir = output_dir / "scenarios" / scenario_cfg.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        evaluator = build_evaluator(scenario_cfg.objective, base_env=base_env)
        macro = run_phase(
            scenario=scenario_cfg.name,
            phase=plan.macro.phase,
            trials=plan.macro.trials,
            space=plan.macro.space,
            evaluator=evaluator,
            seed=settings.seed,
            output_dir=scenario_dir,
        )
        micro = run_phase(
            scenario=scenario_cfg.name,
            phase=plan.micro.phase,
            trials=plan.micro.trials,
            space=plan.micro.space,
            evaluator=evaluator,
            seed=settings.seed + 1,
            output_dir=scenario_dir,
        )

        regression_samples = _compose_samples(macro.trials, micro.trials)
        model = fit_regression(regression_samples, degree=scenario_cfg.regression.degree)
        metrics = evaluate_regression(model, regression_samples)
        metrics = {name: metrics.get(name, 0.0) for name in scenario_cfg.metrics}

        predictions = [model.predict(sample.features) for sample in regression_samples]

        _persist_trials(scenario_dir / "macro_trials.csv", macro.trials)
        _persist_trials(scenario_dir / "micro_trials.csv", micro.trials)
        write_json(scenario_dir / "regression.json", model.to_dict())
        _persist_predictions(scenario_dir / "predictions.csv", regression_samples, predictions)

        summary_builder.add(
            scenario_cfg.name,
            ScenarioSummary(
                macro_trials=len(macro.trials),
                micro_trials=len(micro.trials),
                macro_best=macro.best_params,
                micro_best=micro.best_params,
                metrics=metrics,
            ),
        )

    summary = summary_builder.finalize()
    logger.info("Pipeline completed successfully")
    return summary


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
        row = dict(result.context.params)
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


__all__ = ["run_pipeline"]
