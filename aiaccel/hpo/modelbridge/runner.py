"""High level orchestration for the modelbridge pipeline."""

from __future__ import annotations

from typing import Any, Literal

from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
import optuna

from .config import BridgeConfig, BridgeSettings, ParameterBounds, ParameterSpace, ScenarioConfig
from .evaluators import build_evaluator
from .io import read_json, write_csv, write_json
from .logging import configure_logging, get_logger
from .optimizers import collect_trial_results, run_phase
from .regression import RegressionModel, evaluate_regression, fit_regression
from .summary import ScenarioSummary, SummaryBuilder
from .types import RegressionSample, TrialResult

PhaseName = Literal["hpo", "regress", "evaluate", "summary"]
PHASE_ORDER: tuple[PhaseName, ...] = ("hpo", "regress", "evaluate", "summary")
SCENARIO_SUMMARY_FILE = "scenario_summary.json"


@dataclass(slots=True)
class RunLayout:
    """Path/seed helper for a scenario."""

    output_root: Path
    scenario: str

    def scenario_dir(self) -> Path:
        return self.output_root / "scenarios" / self.scenario

    def run_dir(self, role: str, target: str, run_idx: int) -> Path:
        return self.output_root / "runs" / self.scenario / role / target / f"{run_idx:03d}"

    def storage_uri(self, role: str, target: str, run_idx: int, *, ensure_dir: bool = True) -> str:
        run_dir = self.run_dir(role, target, run_idx)
        if ensure_dir:
            run_dir.mkdir(parents=True, exist_ok=True)
        db_path = run_dir / "optuna.db"
        return f"sqlite:///{db_path.resolve()}"

    def seed(self, base: int, role: str, target: str, run_idx: int) -> int:
        return base + _seed_offset(role, target) + run_idx


@dataclass(slots=True)
class PipelineState:
    """Container for scenario execution."""

    best: dict[str, dict[str, dict[int, TrialResult]]] = field(
        default_factory=lambda: {"train": {"macro": {}, "micro": {}}, "eval": {"macro": {}, "micro": {}}}
    )
    model: RegressionModel | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)


def run_pipeline(
    config: BridgeConfig,
    *,
    phases: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    role: str | None = None,
    target: str | None = None,
    run_id: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Execute the modelbridge pipeline described by ``config``."""

    requested_phases = _normalize_phases(phases)
    scenario_filter = set(scenarios or [])

    settings = config.bridge
    output_root = settings.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    if not quiet:
        silent_env = os.environ.get("AIACCEL_LOG_SILENT")
        console_enabled = silent_env not in {"1", "true", "True"}
        configure_logging(settings.log_level, output_root, console=console_enabled)
    logger = get_logger(__name__)
    logger.info("Starting modelbridge pipeline -> output_dir=%s", output_root)

    summary_builder = SummaryBuilder(output_dir=output_root) if "summary" in requested_phases else None
    base_env = {"AIACCEL_OUTPUT_DIR": str(output_root)}
    if settings.working_directory:
        base_env["AIACCEL_WORK_DIR"] = str(settings.working_directory)

    processed: list[str] = []
    for scenario_cfg in settings.scenarios:
        if scenario_filter and scenario_cfg.name not in scenario_filter:
            continue
        processed.append(scenario_cfg.name)
        logger.info("Running scenario %s (phases=%s)", scenario_cfg.name, ",".join(requested_phases))
        layout = RunLayout(output_root=output_root, scenario=scenario_cfg.name)
        state = PipelineState()
        summary = _execute_scenario(
            layout,
            scenario_cfg,
            settings,
            requested_phases,
            base_env,
            settings.write_csv,
            role,
            target,
            run_id,
            state,
        )
        if summary_builder and summary:
            summary_builder.add(scenario_cfg.name, summary)

    if not processed:
        raise ValueError("No scenarios matched the requested filters")

    if summary_builder:
        payload = summary_builder.finalize()
        logger.info("Pipeline summary generated")
        return payload

    logger.info("Pipeline phases completed: %s", ",".join(requested_phases))
    return {"phases": list(requested_phases), "scenarios": processed}


def _execute_scenario(
    layout: RunLayout,
    config: ScenarioConfig,
    settings: BridgeSettings,
    phases: Sequence[PhaseName],
    base_env: dict[str, str],
    write_csv: bool,
    role_filter: str | None,
    target_filter: str | None,
    run_id: int | None,
    state: PipelineState,
) -> ScenarioSummary | None:
    scenario_dir = layout.scenario_dir()
    scenario_dir.mkdir(parents=True, exist_ok=True)

    train_evaluator = build_evaluator(config.train_objective or config.objective, base_env=base_env)
    eval_evaluator = build_evaluator(config.eval_objective or config.objective, base_env=base_env)
    train_params = config.train_params or config.params
    eval_params = config.eval_params or config.params

    if "hpo" in phases:
        _run_hpo_phase(
            layout,
            config,
            settings,
            train_params,
            eval_params,
            train_evaluator,
            eval_evaluator,
            state,
            role_filter,
            target_filter,
            run_id,
            write_csv,
        )

    if "regress" in phases:
        state.model, state.train_metrics = _run_regression_phase(
            layout,
            config,
            settings,
            train_params,
            state,
            write_csv,
        )

    if "evaluate" in phases:
        state.eval_metrics = _run_evaluation_phase(
            layout,
            config,
            settings,
            eval_params,
            state,
            write_csv,
            strict=True,
        )

    if "summary" in phases:
        return _build_summary(
            layout,
            config,
            settings,
            state,
        )

    return None


def _run_hpo_phase(
    layout: RunLayout,
    config: ScenarioConfig,
    settings: BridgeSettings,
    train_params: ParameterSpace,
    eval_params: ParameterSpace,
    train_evaluator,
    eval_evaluator,
    state: PipelineState,
    role_filter: str | None,
    target_filter: str | None,
    run_id: int | None,
    write_csv: bool,
) -> None:
    base_seed = getattr(settings, "seed", 0)
    roles = [role_filter] if role_filter else ("train", "eval")
    targets = [target_filter] if target_filter else ("macro", "micro")
    for role in roles:
        runs = _resolve_runs(settings, role, run_id)
        if not runs:
            continue
        evaluator = train_evaluator if role == "train" else eval_evaluator
        params = train_params if role == "train" else eval_params
        for target in targets:
            trials = _get_trials(config, role, target)
            space = params.macro if target == "macro" else params.micro
            results = _run_hpo_runs(
                layout=layout,
                role=role,
                target=target,
                runs=runs,
                trials=trials,
                space=space,
                evaluator=evaluator,
                base_seed=base_seed,
                write_csv=write_csv,
            )
            state.best[role][target].update(results)


def _run_regression_phase(
    layout: RunLayout,
    config: ScenarioConfig,
    settings: BridgeSettings,
    train_params: ParameterSpace,
    state: PipelineState,
    write_csv: bool,
) -> tuple[RegressionModel | None, dict[str, float]]:
    macro_best = _ensure_best(config, settings, layout, "train", "macro", state.best["train"]["macro"])
    micro_best = _ensure_best(config, settings, layout, "train", "micro", state.best["train"]["micro"])
    state.best["train"]["macro"] = macro_best
    state.best["train"]["micro"] = micro_best
    samples = _compose_samples(macro_best, micro_best)
    regression_samples = [sample for _, sample in samples]
    model = fit_regression(regression_samples, config.regression)
    metrics = {
        k: float(v)
        for k, v in evaluate_regression(model, regression_samples, metrics=config.metrics).items()
    }
    write_json(layout.scenario_dir() / "regression_train.json", model.to_dict())
    if write_csv:
        _persist_best_pairs(layout.scenario_dir() / "train_pairs.csv", samples)
    return model, metrics


def _run_evaluation_phase(
    layout: RunLayout,
    config: ScenarioConfig,
    settings: BridgeSettings,
    eval_params: ParameterSpace,
    state: PipelineState,
    write_csv: bool,
    *,
    strict: bool = True,
) -> dict[str, float]:
    model = state.model
    if model is None:
        model = _load_regression_model(layout.scenario_dir() / "regression_train.json")
    macro_best = _ensure_best(config, settings, layout, "eval", "macro", state.best["eval"]["macro"])
    micro_best = _ensure_best(config, settings, layout, "eval", "micro", state.best["eval"]["micro"])
    state.best["eval"]["macro"] = macro_best
    state.best["eval"]["micro"] = micro_best
    if not macro_best or not micro_best:
        if strict:
            raise RuntimeError(f"Scenario '{config.name}' requires eval macro/micro runs before evaluation")
        return {}
    pairs = _compose_samples(macro_best, micro_best)
    samples = [sample for _, sample in pairs]
    predictions = [model.predict(sample.features) for sample in samples]
    metrics = {
        k: float(v)
        for k, v in evaluate_regression(model, samples, metrics=config.metrics).items()
    }
    if write_csv:
        _persist_predictions(layout.scenario_dir() / "test_predictions.csv", pairs, predictions)
    write_json(layout.scenario_dir() / "regression_test_metrics.json", metrics)
    state.model = model
    return metrics


def _build_summary(
    layout: RunLayout,
    config: ScenarioConfig,
    settings: BridgeSettings,
    state: PipelineState,
) -> ScenarioSummary:
    scenario_dir = layout.scenario_dir()
    scenario_dir.mkdir(parents=True, exist_ok=True)
    train_macro_best = _ensure_best(config, settings, layout, "train", "macro", state.best["train"]["macro"])
    train_micro_best = _ensure_best(config, settings, layout, "train", "micro", state.best["train"]["micro"])
    eval_macro_best = _ensure_best(config, settings, layout, "eval", "macro", state.best["eval"]["macro"])
    eval_micro_best = _ensure_best(config, settings, layout, "eval", "micro", state.best["eval"]["micro"])

    train_pairs_samples = _compose_samples_optional(train_macro_best, train_micro_best)
    eval_pairs_samples = _compose_samples_optional(eval_macro_best, eval_micro_best)

    model = state.model
    train_metrics = dict(state.train_metrics)
    eval_metrics = dict(state.eval_metrics)

    if model is None and train_pairs_samples:
        regression_samples = [sample for _, sample in train_pairs_samples]
        model = fit_regression(regression_samples, config.regression)
        train_metrics = {
            k: float(v)
            for k, v in evaluate_regression(model, regression_samples, metrics=config.metrics).items()
        }

    if model is not None and not eval_metrics and eval_pairs_samples:
        eval_metrics = {
            k: float(v)
            for k, v in evaluate_regression(
                model, [sample for _, sample in eval_pairs_samples], metrics=config.metrics
            ).items()
        }

    summary = ScenarioSummary(
        train_pairs=len(train_pairs_samples),
        eval_pairs=len(eval_pairs_samples),
        train_macro_best=[train_macro_best[run].context.params for run in sorted(train_macro_best)],
        train_micro_best=[train_micro_best[run].context.params for run in sorted(train_micro_best)],
        eval_macro_best=[eval_macro_best[run].context.params for run in sorted(eval_macro_best)],
        eval_micro_best=[eval_micro_best[run].context.params for run in sorted(eval_micro_best)],
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )
    _persist_scenario_summary(scenario_dir, summary)
    return summary


def _normalize_phases(phases: Sequence[str] | None) -> tuple[PhaseName, ...]:
    if not phases or "full" in phases:
        return PHASE_ORDER
    normalized: list[PhaseName] = []
    for phase in phases:
        if phase not in PHASE_ORDER:
            raise ValueError(f"Unknown phase '{phase}'")
        if phase not in normalized:
            normalized.append(phase)
    normalized.sort(key=lambda name: PHASE_ORDER.index(name))
    return tuple(normalized)


def _resolve_runs(settings, role: str, run_id: int | None) -> list[int]:
    total = getattr(settings, "train_runs" if role == "train" else "eval_runs", 0)
    if total <= 0:
        return []
    if run_id is not None:
        if run_id < 0 or run_id >= total:
            raise ValueError(f"run_id {run_id} is out of range for role '{role}'")
        return [run_id]
    return list(range(total))


def _get_trials(config: ScenarioConfig, role: str, target: str) -> int:
    if role == "train" and target == "macro":
        return config.train_macro_trials
    if role == "train" and target == "micro":
        return config.train_micro_trials
    if role == "eval" and target == "macro":
        return config.eval_macro_trials
    return config.eval_micro_trials


def _run_hpo_runs(
    *,
    layout: RunLayout,
    role: str,
    target: str,
    runs: Sequence[int],
    trials: int,
    space: dict[str, ParameterBounds],
    evaluator,
    base_seed: int,
    write_csv: bool,
) -> dict[int, TrialResult]:
    best: dict[int, TrialResult] = {}
    for run_idx in runs:
        output_dir = layout.run_dir(role, target, run_idx)
        output_dir.mkdir(parents=True, exist_ok=True)
        storage_uri = layout.storage_uri(role, target, run_idx, ensure_dir=False)
        outcome = run_phase(
            scenario=layout.scenario,
            phase=f"{role}-{target}-{run_idx:03d}",
            trials=trials,
            space=space,
            evaluator=evaluator,
            seed=layout.seed(base_seed, role, target, run_idx),
            output_dir=output_dir,
            storage=storage_uri,
            study_name=f"{layout.scenario}-{role}-{target}-{run_idx:03d}",
            write_csv=False,
        )
        best[run_idx] = _best_trial(outcome.trials)
    if write_csv and best:
        csv_path = layout.scenario_dir() / f"{role}_{target}_best.csv"
        rows = []
        for run_idx in sorted(best):
            row = {"run_id": run_idx} | {f"param_{k}": v for k, v in best[run_idx].context.params.items()}
            rows.append(row)
        write_csv(csv_path, rows)
    return best


def _ensure_best(
    config: ScenarioConfig,
    settings: BridgeSettings,
    layout: RunLayout,
    role: str,
    target: str,
    cache: dict[int, TrialResult],
) -> dict[int, TrialResult]:
    if cache:
        return cache
    runs = _resolve_runs(settings, role, None)
    return _load_best_trials_from_storage(layout, role, target, runs)


def _load_best_trials_from_storage(
    layout: RunLayout,
    role: str,
    target: str,
    runs: Sequence[int],
) -> dict[int, TrialResult]:
    best: dict[int, TrialResult] = {}
    for run_idx in runs:
        storage_uri = layout.storage_uri(role, target, run_idx, ensure_dir=False)
        db_path = storage_uri.removeprefix("sqlite:///")
        if not Path(db_path).exists():
            continue
        try:
            study = optuna.load_study(study_name=f"{layout.scenario}-{role}-{target}-{run_idx:03d}", storage=storage_uri)
        except Exception:
            continue
        trials = collect_trial_results(study=study, scenario=layout.scenario, phase=f"{role}-{target}", output_dir=Path(db_path).parent)
        if trials:
            best[run_idx] = _best_trial(trials)
    return best


def _compose_samples(
    macro: dict[int, TrialResult],
    micro: dict[int, TrialResult],
) -> list[tuple[int, RegressionSample]]:
    samples: list[tuple[int, RegressionSample]] = []
    for run_idx in sorted(set(macro.keys()) & set(micro.keys())):
        macro_tr = macro[run_idx]
        micro_tr = micro[run_idx]
        samples.append(
            (
                run_idx,
                RegressionSample(
                    features=dict(macro_tr.context.params),
                    target=dict(micro_tr.context.params),
                ),
            )
        )
    if not samples:
        raise RuntimeError("No overlapping macro/micro best trials available")
    return samples


def _persist_best_pairs(path: Path, samples: list[tuple[int, RegressionSample]]) -> None:
    rows = []
    for run_idx, sample in samples:
        row = {"run_id": run_idx}
        row |= {f"macro_{k}": v for k, v in sample.features.items()}
        row |= {f"micro_{k}": v for k, v in sample.target.items()}
        rows.append(row)
    write_csv(path, rows)


def _persist_predictions(
    path: Path,
    samples: list[tuple[int, RegressionSample]],
    predictions: list[dict[str, float]],
) -> None:
    rows = []
    for (run_idx, sample), pred in zip(samples, predictions, strict=True):
        row = {"run_id": run_idx}
        row |= {f"macro_{k}": v for k, v in sample.features.items()}
        row |= {f"actual_{k}": v for k, v in sample.target.items()}
        row |= {f"pred_{k}": v for k, v in pred.items()}
        rows.append(row)
    write_csv(path, rows)


def _compose_samples_optional(
    macro: dict[int, TrialResult],
    micro: dict[int, TrialResult],
) -> list[tuple[int, RegressionSample]]:
    try:
        return _compose_samples(macro, micro)
    except RuntimeError:
        return []


def _load_regression_model(path: Path) -> RegressionModel:
    if not path.exists():
        raise RuntimeError(f"Regression model missing at {path}")
    payload = read_json(path)
    return RegressionModel.from_dict(payload)


def _best_trial(trials: list[TrialResult]) -> TrialResult:
    return min(trials, key=lambda t: t.evaluation.objective)


def _persist_scenario_summary(path: Path, summary: ScenarioSummary) -> None:
    write_json(
        path / SCENARIO_SUMMARY_FILE,
        {
            "train_pairs": summary.train_pairs,
            "eval_pairs": summary.eval_pairs,
            "train_macro_best": summary.train_macro_best,
            "train_micro_best": summary.train_micro_best,
            "eval_macro_best": summary.eval_macro_best,
            "eval_micro_best": summary.eval_micro_best,
            "train_metrics": summary.train_metrics,
            "eval_metrics": summary.eval_metrics,
        },
    )


def _seed_offset(role: str, target: str) -> int:
    if role == "train" and target == "micro":
        return 1
    if role == "eval" and target == "macro":
        return 100
    if role == "eval" and target == "micro":
        return 101
    return 0


__all__ = ["run_pipeline", "PHASE_ORDER"]
