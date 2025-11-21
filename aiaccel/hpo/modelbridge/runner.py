"""Planning and execution utilities for the modelbridge pipeline."""

from __future__ import annotations

from typing import Any, Callable, Literal

from dataclasses import dataclass, field
from collections.abc import Sequence
import os
from pathlib import Path
import optuna

from .config import BridgeConfig, BridgeSettings, ScenarioConfig
from .evaluators import build_evaluator
from .io import read_json, write_csv, write_json
from .logging import configure_logging, get_logger
from .optimizers import collect_trial_results, run_phase
from .regression import RegressionModel, evaluate_regression, fit_regression
from .summary import ScenarioSummary, SummaryBuilder
from .types import EvaluatorFn, PhaseContext, RegressionSample, RunnerFn, TrialResult

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

    def storage_uri(self, role: str, target: str, run_idx: int) -> str:
        db_path = self.run_dir(role, target, run_idx) / "optuna.db"
        return f"sqlite:///{db_path.resolve()}"

    def seed(self, base: int, role: str, target: str, run_idx: int) -> int:
        return base + _seed_offset(role, target) + run_idx


@dataclass(slots=True)
class PipelinePlan:
    """Executable plan containing ordered phase contexts."""

    contexts: list[PhaseContext]
    settings: BridgeSettings
    scenarios: dict[str, ScenarioConfig]

    def serializable(self) -> dict[str, object]:
        return {
            "contexts": [ctx.serializable() for ctx in self.contexts],
            "output_dir": str(self.settings.output_dir),
            "working_directory": str(self.settings.working_directory or self.settings.output_dir),
            "log_level": self.settings.log_level,
        }


@dataclass(slots=True)
class ScenarioState:
    """Captured state during execution."""

    best: dict[str, dict[str, dict[int, TrialResult]]] = field(
        default_factory=lambda: {"train": {"macro": {}, "micro": {}}, "eval": {"macro": {}, "micro": {}}}
    )
    model: RegressionModel | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PlanBuilder:
    """Helper to construct PhaseContexts for a scenario."""

    layout: RunLayout
    settings: BridgeSettings
    scenario: ScenarioConfig
    base_seed: int
    contexts: list[PhaseContext] = field(default_factory=list)

    def add_hpo(self, role: str | None, target: str | None, run_id: int | None) -> None:
        roles = [role] if role else ("train", "eval")
        targets = [target] if target else ("macro", "micro")
        for phase_role in roles:
            runs = _resolve_runs(self.settings, phase_role, run_id)
            for phase_target in targets:
                for idx in runs:
                    ctx = PhaseContext(
                        scenario=self.scenario.name,
                        phase="hpo",
                        role=phase_role,
                        target=phase_target,
                        run_id=idx,
                        seed=self.layout.seed(self.base_seed, phase_role, phase_target, idx),
                        output_dir=self.layout.run_dir(phase_role, phase_target, idx),
                        working_directory=self.settings.working_directory or self.settings.output_dir,
                    )
                    ctx.runner = _make_hpo_runner(
                        layout=self.layout,
                        scenario=self.scenario,
                        settings=self.settings,
                        role=phase_role,
                        target=phase_target,
                        run_idx=idx,
                    )
                    self.contexts.append(ctx)

    def add_phase(self, phase: PhaseName, runner: Callable[..., Any]) -> None:
        ctx = PhaseContext(
            scenario=self.scenario.name,
            phase=phase,
            output_dir=self.layout.scenario_dir(),
            working_directory=self.settings.working_directory or self.settings.output_dir,
        )
        ctx.runner = runner
        self.contexts.append(ctx)


def plan_pipeline(
    config: BridgeConfig,
    *,
    phases: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    role: str | None = None,
    target: str | None = None,
    run_id: int | None = None,
) -> PipelinePlan:
    """Create an executable plan without running it."""

    requested_phases = _normalize_phases(phases)
    scenario_filter = set(scenarios or [])
    settings = config.bridge
    contexts: list[PhaseContext] = []
    scenario_map = {cfg.name: cfg for cfg in settings.scenarios}

    for scenario_cfg in settings.scenarios:
        if scenario_filter and scenario_cfg.name not in scenario_filter:
            continue
        layout = RunLayout(output_root=settings.output_dir, scenario=scenario_cfg.name)
        builder = PlanBuilder(layout=layout, settings=settings, scenario=scenario_cfg, base_seed=settings.seed)
        for phase in requested_phases:
            if phase == "hpo":
                builder.add_hpo(role, target, run_id)
            elif phase == "regress":
                builder.add_phase("regress", _make_regression_runner(layout, scenario_cfg, settings))
            elif phase == "evaluate":
                builder.add_phase("evaluate", _make_evaluation_runner(layout, scenario_cfg, settings))
            elif phase == "summary":
                builder.add_phase("summary", _make_summary_runner(layout, scenario_cfg, settings))
        contexts.extend(builder.contexts)

    if not contexts:
        raise ValueError("No scenarios matched the requested filters")

    return PipelinePlan(contexts=contexts, settings=settings, scenarios=scenario_map)


def execute_pipeline(
    plan: PipelinePlan,
    *,
    dry_run: bool = False,
    quiet: bool = True,
    log_to_file: bool = False,
) -> dict[str, Any]:
    """Execute the provided plan. When ``dry_run`` is True, only the plan payload is returned."""

    if dry_run:
        return plan.serializable()

    output_root = plan.settings.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    silent_env = os.environ.get("AIACCEL_LOG_SILENT")
    console_enabled = not quiet and silent_env not in {"1", "true", "True"}
    if log_to_file or console_enabled:
        configure_logging(
            plan.settings.log_level,
            output_root,
            reset_handlers=True,
            console=console_enabled,
            file=log_to_file,
        )
    logger = get_logger(__name__)
    logger.info("Executing modelbridge plan with %d contexts", len(plan.contexts))

    summary_builder = SummaryBuilder(output_dir=output_root) if any(ctx.phase == "summary" for ctx in plan.contexts) else None
    states: dict[str, ScenarioState] = {}

    for ctx in plan.contexts:
        scenario_state = states.setdefault(ctx.scenario, ScenarioState())
        runner = ctx.runner
        scenario_cfg = plan.scenarios[ctx.scenario]
        if not callable(runner):
            raise RuntimeError(f"No runner associated with context {ctx}")
        result = runner(ctx, scenario_cfg, plan.settings, scenario_state)
        if isinstance(result, ScenarioSummary) and summary_builder:
            summary_builder.add(ctx.scenario, result)

    if summary_builder:
        return summary_builder.finalize()
    return {"contexts": [c.serializable() for c in plan.contexts]}


def run_pipeline(
    config: BridgeConfig,
    *,
    phases: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    role: str | None = None,
    target: str | None = None,
    run_id: int | None = None,
    dry_run: bool = False,
    quiet: bool = True,
    log_to_file: bool = False,
) -> dict[str, Any]:
    """Convenience wrapper that plans then executes."""

    plan = plan_pipeline(
        config,
        phases=phases,
        scenarios=scenarios,
        role=role,
        target=target,
        run_id=run_id,
    )
    return execute_pipeline(plan, dry_run=dry_run, quiet=quiet, log_to_file=log_to_file)


def _make_hpo_runner(
    *,
    layout: RunLayout,
    scenario: ScenarioConfig,
    settings: BridgeSettings,
    role: str,
    target: str,
    run_idx: int,
) -> RunnerFn:
    evaluator = build_evaluator(scenario.train_objective if role == "train" else scenario.eval_objective, base_env=_base_env(settings))
    params = scenario.train_params if role == "train" else scenario.eval_params
    trials = _get_trials(scenario, role, target)

    def _runner(
        ctx: PhaseContext,
        scenario_cfg: ScenarioConfig,
        bridge_settings: BridgeSettings,
        state: ScenarioState,
    ) -> None:
        output_dir = layout.run_dir(role, target, run_idx)
        output_dir.mkdir(parents=True, exist_ok=True)
        storage_uri = layout.storage_uri(role, target, run_idx)
        outcome = run_phase(
            scenario=scenario_cfg.name,
            phase=target,
            trials=trials,
            space=params.macro if target == "macro" else params.micro,
            evaluator=evaluator,
            seed=ctx.seed or bridge_settings.seed,
            output_dir=output_dir,
            storage=storage_uri,
            study_name=f"{scenario_cfg.name}-{role}-{target}-{run_idx:03d}",
            write_csv=False,
        )
        state.best[role][target][run_idx] = _best_trial(outcome.trials)
        _write_best_csv(layout, role, target, state.best[role][target], bridge_settings.write_csv)

    return _runner


def _make_regression_runner(
    layout: RunLayout,
    scenario: ScenarioConfig,
    settings: BridgeSettings,
) -> RunnerFn:
    def _runner(
        ctx: PhaseContext,
        scenario_cfg: ScenarioConfig,
        bridge_settings: BridgeSettings,
        state: ScenarioState,
    ) -> RegressionModel | None:
        macro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "train", "macro", state.best["train"]["macro"])
        micro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "train", "micro", state.best["train"]["micro"])
        state.best["train"]["macro"] = macro_best
        state.best["train"]["micro"] = micro_best
        samples = _compose_samples(macro_best, micro_best)
        regression_samples = [sample for _, sample in samples]
        model = _ensure_model(layout, state, scenario_cfg, samples_for_fit=regression_samples)
        state.train_metrics = _evaluate_metrics(model, regression_samples, scenario_cfg.metrics)
        layout.scenario_dir().mkdir(parents=True, exist_ok=True)
        write_json(layout.scenario_dir() / "regression_train.json", model.to_dict())
        if settings.write_csv:
            _persist_best_pairs(layout.scenario_dir() / "train_pairs.csv", samples)
        return model

    return _runner


def _make_evaluation_runner(
    layout: RunLayout,
    scenario: ScenarioConfig,
    settings: BridgeSettings,
) -> RunnerFn:
    def _runner(
        ctx: PhaseContext,
        scenario_cfg: ScenarioConfig,
        bridge_settings: BridgeSettings,
        state: ScenarioState,
    ) -> dict[str, float]:
        model = _ensure_model(layout, state, scenario_cfg, samples_for_fit=None)
        macro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "eval", "macro", state.best["eval"]["macro"])
        micro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "eval", "micro", state.best["eval"]["micro"])
        state.best["eval"]["macro"] = macro_best
        state.best["eval"]["micro"] = micro_best
        pairs = _compose_samples(macro_best, micro_best)
        samples = [sample for _, sample in pairs]
        predictions = [model.predict(sample.features) for sample in samples]
        metrics = _evaluate_metrics(model, samples, scenario_cfg.metrics)
        state.model = model
        state.eval_metrics = metrics
        layout.scenario_dir().mkdir(parents=True, exist_ok=True)
        write_json(layout.scenario_dir() / "regression_test_metrics.json", metrics)
        if settings.write_csv:
            _persist_predictions(layout.scenario_dir() / "test_predictions.csv", pairs, predictions)
        return metrics

    return _runner


def _make_summary_runner(
    layout: RunLayout,
    scenario: ScenarioConfig,
    settings: BridgeSettings,
) -> RunnerFn:
    def _runner(
        ctx: PhaseContext,
        scenario_cfg: ScenarioConfig,
        bridge_settings: BridgeSettings,
        state: ScenarioState,
    ) -> ScenarioSummary:
        train_macro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "train", "macro", state.best["train"]["macro"])
        train_micro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "train", "micro", state.best["train"]["micro"])
        eval_macro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "eval", "macro", state.best["eval"]["macro"])
        eval_micro_best = _ensure_best(scenario_cfg, bridge_settings, layout, "eval", "micro", state.best["eval"]["micro"])
        train_pairs_samples = _compose_samples_optional(train_macro_best, train_micro_best)
        eval_pairs_samples = _compose_samples_optional(eval_macro_best, eval_micro_best)

        model = state.model
        train_metrics = dict(state.train_metrics)
        eval_metrics = dict(state.eval_metrics)

        if model is None and train_pairs_samples:
            regression_samples = [sample for _, sample in train_pairs_samples]
            model = _ensure_model(layout, state, scenario_cfg, samples_for_fit=regression_samples)
            train_metrics = _evaluate_metrics(model, regression_samples, scenario_cfg.metrics)

        if model is not None and not eval_metrics and eval_pairs_samples:
            eval_metrics = _evaluate_metrics(model, [sample for _, sample in eval_pairs_samples], scenario_cfg.metrics)

        layout.scenario_dir().mkdir(parents=True, exist_ok=True)
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
        _persist_scenario_summary(layout.scenario_dir(), summary)
        return summary

    return _runner


def _normalize_phases(phases: Sequence[str] | None) -> tuple[PhaseName, ...]:
    if not phases or "full" in phases:
        return PHASE_ORDER
    normalized: list[PhaseName] = []
    for phase in phases:
        if phase not in PHASE_ORDER:
            raise ValueError(f"Unknown phase '{phase}'")
        if phase not in normalized:
            normalized.append(phase)
    return tuple(normalized)


def _resolve_runs(settings: BridgeSettings, role: str, run_id: int | None) -> list[int]:
    total = settings.train_runs if role == "train" else settings.eval_runs
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
    return _load_best_trials_from_storage(layout, config, role, target, runs)


def _load_best_trials_from_storage(
    layout: RunLayout,
    config: ScenarioConfig,
    role: str,
    target: str,
    runs: Sequence[int],
) -> dict[int, TrialResult]:
    best: dict[int, TrialResult] = {}
    for run_idx in runs:
        storage_uri = layout.storage_uri(role, target, run_idx)
        db_path = Path(storage_uri.removeprefix("sqlite:///"))
        if not db_path.exists():
            continue
        try:
            study = optuna.load_study(
                study_name=f"{config.name}-{role}-{target}-{run_idx:03d}",
                storage=storage_uri,
            )
        except Exception:
            continue
        trials = collect_trial_results(
            study=study,
            scenario=config.name,
            phase=target,
            output_dir=db_path.parent,
        )
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


def _compose_samples_optional(
    macro: dict[int, TrialResult],
    micro: dict[int, TrialResult],
) -> list[tuple[int, RegressionSample]]:
    try:
        return _compose_samples(macro, micro)
    except RuntimeError:
        return []


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


def _best_trial(trials: list[TrialResult]) -> TrialResult:
    return min(trials, key=lambda t: t.evaluation.objective)


def _seed_offset(role: str, target: str) -> int:
    if role == "train" and target == "micro":
        return 1
    if role == "eval" and target == "macro":
        return 100
    if role == "eval" and target == "micro":
        return 101
    return 0


def _base_env(settings: BridgeSettings) -> dict[str, str]:
    env = {
        "AIACCEL_OUTPUT_DIR": str(settings.output_dir),
        "AIACCEL_WORK_DIR": str(settings.working_directory or settings.output_dir),
    }
    return env


def _write_best_csv(
    layout: RunLayout,
    role: str,
    target: str,
    best: dict[int, TrialResult],
    enable_csv: bool,
) -> None:
    if not enable_csv or not best:
        return
    csv_path = layout.scenario_dir() / f"{role}_{target}_best.csv"
    rows = []
    for run_idx in sorted(best):
        row = {"run_id": run_idx} | {f"param_{k}": v for k, v in best[run_idx].context.params.items()}
        rows.append(row)
    if rows:
        write_csv(csv_path, rows)


def _ensure_model(
    layout: RunLayout,
    state: ScenarioState,
    scenario_cfg: ScenarioConfig,
    samples_for_fit: list[RegressionSample] | None,
) -> RegressionModel:
    if state.model is not None:
        return state.model
    if samples_for_fit:
        model = fit_regression(samples_for_fit, scenario_cfg.regression)
        state.model = model
        return model
    model_path = layout.scenario_dir() / "regression_train.json"
    payload = read_json(model_path)
    model = RegressionModel.from_dict(payload)
    state.model = model
    return model


def _evaluate_metrics(
    model: RegressionModel,
    samples: list[RegressionSample],
    metrics: tuple[str, ...] | list[str],
) -> dict[str, float]:
    return {k: float(v) for k, v in evaluate_regression(model, samples, metrics=metrics).items()}


__all__ = ["PHASE_ORDER", "PipelinePlan", "plan_pipeline", "execute_pipeline", "run_pipeline"]
