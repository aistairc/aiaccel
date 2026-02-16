"""Pipeline orchestration and canonical step/profile registry."""

from __future__ import annotations

from typing import Literal, cast

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from .analyze import evaluate_model, fit_regression
from .collect import collect_eval, collect_train
from .common import PipelineResult, StepResult, plan_path, read_plan
from .config import BridgeConfig
from .prepare import prepare_eval, prepare_train
from .publish import publish_summary

StepName = Literal[
    "prepare_train",
    "prepare_eval",
    "collect_train",
    "collect_eval",
    "fit_regression",
    "evaluate_model",
    "publish_summary",
]
PipelineProfile = Literal["prepare", "analyze", "full"]
PIPELINE_PROFILES: tuple[PipelineProfile, ...] = ("prepare", "analyze", "full")

StepAction = Callable[
    [
        BridgeConfig,
        tuple[Path, ...],
        tuple[Path, ...],
        tuple[tuple[Path, Path], ...],
        tuple[tuple[Path, Path], ...],
    ],
    StepResult,
]
StepDefinition = tuple[StepName, str, tuple[PipelineProfile, ...], StepAction]

STEP_DEFINITIONS: tuple[StepDefinition, ...] = (
    (
        "prepare_train",
        "prepare-train",
        ("prepare", "full"),
        lambda config, _tp, _ep, _tpp, _epp: prepare_train(config),
    ),
    (
        "prepare_eval",
        "prepare-eval",
        ("prepare", "full"),
        lambda config, _tp, _ep, _tpp, _epp: prepare_eval(config),
    ),
    (
        "collect_train",
        "collect-train",
        ("analyze", "full"),
        lambda config, train_paths, _ep, train_pairs, _epp: collect_train(
            config,
            db_paths=train_paths or None,
            db_pairs=train_pairs or None,
        ),
    ),
    (
        "collect_eval",
        "collect-eval",
        ("analyze", "full"),
        lambda config, _tp, eval_paths, _tpp, eval_pairs: collect_eval(
            config,
            db_paths=eval_paths or None,
            db_pairs=eval_pairs or None,
        ),
    ),
    (
        "fit_regression",
        "fit-regression",
        ("analyze", "full"),
        lambda config, _tp, _ep, _tpp, _epp: fit_regression(config),
    ),
    (
        "evaluate_model",
        "evaluate-model",
        ("analyze", "full"),
        lambda config, _tp, _ep, _tpp, _epp: evaluate_model(config),
    ),
    (
        "publish_summary",
        "publish-summary",
        ("analyze", "full"),
        lambda config, _tp, _ep, _tpp, _epp: publish_summary(config),
    ),
)

STEP_SPECS: tuple[tuple[StepName, str, tuple[PipelineProfile, ...]], ...] = tuple(
    (step_name, cli_command, profiles) for step_name, cli_command, profiles, _action in STEP_DEFINITIONS
)
STEP_DEFINITION_BY_NAME: dict[StepName, StepDefinition] = {definition[0]: definition for definition in STEP_DEFINITIONS}
STEP_ACTIONS: dict[StepName, StepAction] = {
    step_name: definition[3] for step_name, definition in STEP_DEFINITION_BY_NAME.items()
}


def _run_step(
    step_name: StepName,
    config: BridgeConfig,
    train_paths: tuple[Path, ...],
    eval_paths: tuple[Path, ...],
    train_pairs: tuple[tuple[Path, Path], ...],
    eval_pairs: tuple[tuple[Path, Path], ...],
) -> StepResult:
    """Execute one canonical step definition."""
    definition = STEP_DEFINITION_BY_NAME.get(step_name)
    if definition is None:
        raise ValueError(f"Unknown step: {step_name}")
    return definition[3](config, train_paths, eval_paths, train_pairs, eval_pairs)


def steps_for_profile(profile: PipelineProfile) -> list[StepName]:
    """Return ordered step names for one profile."""
    return [step for step, _cli, profiles in STEP_SPECS if profile in profiles]


def normalize_steps(steps: Iterable[str]) -> list[StepName]:
    """Normalize and validate explicit step names."""
    normalized: list[StepName] = []
    valid = set(STEP_ACTIONS)
    for step in steps:
        if step not in valid:
            raise ValueError(f"Unknown step: {step}")
        normalized.append(cast(StepName, step))
    return normalized


def run_pipeline(
    config: BridgeConfig,
    steps: Sequence[str] | None = None,
    *,
    profile: str | None = None,
    train_db_paths: Sequence[Path] | None = None,
    eval_db_paths: Sequence[Path] | None = None,
    train_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    eval_db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> PipelineResult:
    """Run selected steps or profile."""
    if steps is not None and profile is not None:
        raise ValueError("steps and profile are mutually exclusive")
    if profile is not None and profile not in PIPELINE_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")

    train_paths = tuple(train_db_paths or ())
    eval_paths = tuple(eval_db_paths or ())
    train_pairs = tuple(train_db_pairs or ())
    eval_pairs = tuple(eval_db_pairs or ())
    selected = _select_steps(steps=steps, profile=profile)

    results = [_run_step(step, config, train_paths, eval_paths, train_pairs, eval_pairs) for step in selected]
    if profile == "full":
        _ensure_full_profile_ready(config)
        results.extend(
            _run_step(step, config, train_paths, eval_paths, train_pairs, eval_pairs)
            for step in steps_for_profile("analyze")
        )

    summary_path, manifest_path = _extract_publish_paths(results)
    return PipelineResult(results=results, summary_path=summary_path, manifest_path=manifest_path)


def _ensure_full_profile_ready(config: BridgeConfig) -> None:
    """Validate that external optimize outputs exist before analyze steps."""
    missing: list[Path] = []
    for role in ("train", "eval"):
        path = plan_path(config.bridge.output_dir, role)
        if not path.exists():
            continue
        plan_role, entries = read_plan(path)
        if plan_role != role:
            raise RuntimeError(f"Plan role mismatch: expected {role}, got {plan_role} ({path})")
        missing.extend(
            [expected for expected in (Path(entry["expected_db_path"]) for entry in entries) if not expected.exists()]
        )

    if not missing:
        return
    preview = ", ".join(str(path) for path in missing[:5])
    if len(missing) > 5:
        preview = f"{preview}, ... (+{len(missing) - 5} more)"
    raise RuntimeError(
        f"Full profile requires external HPO outputs before collect/analyze; missing optuna DB files: {preview}"
    )


def _select_steps(*, steps: Sequence[str] | None, profile: str | None) -> list[StepName]:
    """Select canonical steps from explicit list or profile."""
    if steps is not None:
        return normalize_steps(steps)
    if profile == "full" or profile is None:
        return steps_for_profile("prepare")
    return steps_for_profile(cast(PipelineProfile, profile))


def _extract_publish_paths(results: Sequence[StepResult]) -> tuple[Path | None, Path | None]:
    """Extract summary and manifest paths from publish step outputs."""
    summary_path: Path | None = None
    manifest_path: Path | None = None
    for result in results:
        if result.step != "publish_summary":
            continue
        summary = result.outputs.get("summary_path")
        manifest = result.outputs.get("manifest_path")
        if isinstance(summary, str):
            summary_path = Path(summary)
        if isinstance(manifest, str):
            manifest_path = Path(manifest)
    return summary_path, manifest_path
