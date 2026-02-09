"""Analyze steps for modelbridge."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from .config import BridgeConfig
from .layout import metrics_dir, models_dir, scenario_dir
from .modeling import _evaluate_metrics, _evaluate_metrics_from_preds, _fit_regression, _predict_regression
from .toolkit.io import read_csv, read_json, write_csv, write_json
from .toolkit.results import StepResult, StepStatus, write_step_state


def fit_regression(config: BridgeConfig) -> StepResult:
    """Fit regression models from collected training pairs.

    Args:
        config: Parsed modelbridge configuration.

    Returns:
        StepResult: Step execution result for `fit_regression`.

    Raises:
        RuntimeError: When strict mode is enabled and required artifacts are missing.
    """
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        scenario_path = scenario_dir(output_dir, scenario.name)
        train_pairs_path = scenario_path / "train_pairs.csv"
        if not train_pairs_path.exists():
            issues.append(f"{scenario.name}: missing train_pairs.csv")
            scenario_outputs[scenario.name] = {"status": "missing_pairs"}
            continue

        parsed = _parse_pairs_csv(train_pairs_path)
        if not parsed:
            issues.append(f"{scenario.name}: train_pairs.csv has no valid rows")
            scenario_outputs[scenario.name] = {"status": "empty_pairs", "train_pairs_csv": str(train_pairs_path)}
            continue

        features = [item[1] for item in parsed]
        targets = [item[2] for item in parsed]
        model = _fit_regression(features, targets, scenario.regression)
        metrics = _evaluate_metrics(model, features, targets, list(scenario.metrics))

        model_file = write_json(models_dir(scenario_path) / "regression_model.json", model)
        metrics_file = write_json(metrics_dir(scenario_path) / "train_metrics.json", metrics)
        scenario_outputs[scenario.name] = {
            "status": "success",
            "model_path": str(model_file),
            "metrics_path": str(metrics_file),
        }

    return _finalize_analyze_step(
        config=config,
        step_name="fit_regression",
        outputs={"scenarios": scenario_outputs},
        issues=issues,
    )


def evaluate_model(config: BridgeConfig) -> StepResult:
    """Evaluate regression models using collected evaluation pairs.

    Args:
        config: Parsed modelbridge configuration.

    Returns:
        StepResult: Step execution result for `evaluate_model`.

    Raises:
        RuntimeError: When strict mode is enabled and required artifacts are missing.
    """
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        scenario_path = scenario_dir(output_dir, scenario.name)
        model_path = models_dir(scenario_path) / "regression_model.json"
        eval_pairs_path = scenario_path / "test_pairs.csv"
        if not model_path.exists():
            issues.append(f"{scenario.name}: missing regression_model.json")
            scenario_outputs[scenario.name] = {"status": "missing_model"}
            continue
        if not eval_pairs_path.exists():
            issues.append(f"{scenario.name}: missing test_pairs.csv")
            scenario_outputs[scenario.name] = {"status": "missing_eval_pairs"}
            continue

        pairs = _parse_pairs_csv(eval_pairs_path)
        if not pairs:
            issues.append(f"{scenario.name}: test_pairs.csv has no valid rows")
            scenario_outputs[scenario.name] = {"status": "empty_eval_pairs"}
            continue

        model = read_json(model_path)
        features = [item[1] for item in pairs]
        targets = [item[2] for item in pairs]
        predictions = _predict_regression(model, features)
        metrics = _evaluate_metrics_from_preds(targets, predictions, list(scenario.metrics))

        metrics_file = write_json(metrics_dir(scenario_path) / "eval_metrics.json", metrics)
        prediction_rows: list[dict[str, Any]] = []
        for index, (run_id, macro, micro) in enumerate(pairs):
            row: dict[str, Any] = {"run_id": run_id}
            row.update({f"macro_{name}": value for name, value in macro.items()})
            row.update({f"actual_{name}": value for name, value in micro.items()})
            row.update({f"pred_{name}": value for name, value in predictions[index].items()})
            prediction_rows.append(row)

        predictions_file = write_csv(scenario_path / "test_predictions.csv", prediction_rows)
        scenario_outputs[scenario.name] = {
            "status": "success",
            "metrics_path": str(metrics_file),
            "predictions_path": str(predictions_file),
        }

    return _finalize_analyze_step(
        config=config,
        step_name="evaluate_model",
        outputs={"scenarios": scenario_outputs},
        issues=issues,
    )


def _finalize_analyze_step(
    *,
    config: BridgeConfig,
    step_name: str,
    outputs: dict[str, Any],
    issues: list[str],
) -> StepResult:
    """Finalize status and persist state for analyze-related steps."""
    output_dir = config.bridge.output_dir
    success_count = sum(
        1 for scenario_output in outputs["scenarios"].values() if scenario_output.get("status") == "success"
    )
    total = len(config.bridge.scenarios)

    if issues and config.bridge.strict_mode:
        failure_reason = "; ".join(issues)
        result = StepResult(step=step_name, status="failed", outputs=outputs, reason=failure_reason)
        write_step_state(output_dir, result)
        raise RuntimeError(failure_reason)

    status: StepStatus
    reason: str | None
    if success_count == total:
        status = "success"
        reason = None
    elif success_count == 0:
        status = "skipped"
        reason = "; ".join(issues) if issues else f"{step_name} skipped"
    else:
        status = "partial"
        reason = "; ".join(issues)

    result = StepResult(step=step_name, status=status, outputs=outputs, reason=reason)
    write_step_state(output_dir, result)
    return result


def _parse_pairs_csv(path: Path) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    """Parse macro/micro pair rows from a CSV file."""
    rows = read_csv(path)
    parsed: list[tuple[int, dict[str, float], dict[str, float]]] = []

    for row in rows:
        try:
            run_id = int(row.get("run_id", "0"))
        except ValueError:
            continue

        macro: dict[str, float] = {}
        micro: dict[str, float] = {}
        for key, raw_value in row.items():
            if raw_value in {"", None}:
                continue
            if key.startswith("macro_"):
                macro[key.removeprefix("macro_")] = float(raw_value)
            elif key.startswith("micro_"):
                micro[key.removeprefix("micro_")] = float(raw_value)
        if macro and micro:
            parsed.append((run_id, macro, micro))

    return parsed
