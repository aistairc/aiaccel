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

STEP_SPECS: tuple[tuple[StepName, str, tuple[PipelineProfile, ...]], ...] = (
    ("prepare_train", "prepare-train", ("prepare", "full")),
    ("prepare_eval", "prepare-eval", ("prepare", "full")),
    ("collect_train", "collect-train", ("analyze", "full")),
    ("collect_eval", "collect-eval", ("analyze", "full")),
    ("fit_regression", "fit-regression", ("analyze", "full")),
    ("evaluate_model", "evaluate-model", ("analyze", "full")),
    ("publish_summary", "publish-summary", ("analyze", "full")),
)
STEP_NAME_BY_CLI_COMMAND: dict[str, StepName] = {cli: step for step, cli, _profiles in STEP_SPECS}

STEP_ACTIONS: dict[StepName, Callable[..., StepResult]] = {
    "prepare_train": lambda config, _tp, _ep, _tpp, _epp: prepare_train(config),
    "prepare_eval": lambda config, _tp, _ep, _tpp, _epp: prepare_eval(config),
    "collect_train": lambda config, train_paths, _ep, train_pairs, _epp: collect_train(
        config,
        db_paths=train_paths or None,
        db_pairs=train_pairs or None,
    ),
    "collect_eval": lambda config, _tp, eval_paths, _tpp, eval_pairs: collect_eval(
        config,
        db_paths=eval_paths or None,
        db_pairs=eval_pairs or None,
    ),
    "fit_regression": lambda config, _tp, _ep, _tpp, _epp: fit_regression(config),
    "evaluate_model": lambda config, _tp, _ep, _tpp, _epp: evaluate_model(config),
    "publish_summary": lambda config, _tp, _ep, _tpp, _epp: publish_summary(config),
}


def steps_for_profile(profile: PipelineProfile) -> list[StepName]:
    """Return ordered step names for one profile.

    Args:
        profile: Pipeline profile name.

    Returns:
        list[StepName]: Ordered step names.
    """
    return [step for step, _cli, profiles in STEP_SPECS if profile in profiles]


def normalize_steps(steps: Iterable[str]) -> list[StepName]:
    """Normalize and validate explicit step names.

    Args:
        steps: Raw step names.

    Returns:
        list[StepName]: Validated step names preserving input order.

    Raises:
        ValueError: If any step name is unknown.
    """
    normalized: list[StepName] = []
    valid = set(STEP_ACTIONS)
    for step in steps:
        if step not in valid:
            raise ValueError(f"Unknown step: {step}")
        normalized.append(cast(StepName, step))
    return normalized


def run_pipeline(  # noqa: C901
    config: BridgeConfig,
    steps: Sequence[str] | None = None,
    *,
    profile: str | None = None,
    train_db_paths: Sequence[Path] | None = None,
    eval_db_paths: Sequence[Path] | None = None,
    train_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    eval_db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> PipelineResult:
    """Run selected steps or profile.

    Args:
        config: Validated modelbridge configuration.
        steps: Optional explicit step list.
        profile: Optional profile name.
        train_db_paths: Optional train DB paths for collect override.
        eval_db_paths: Optional eval DB paths for collect override.
        train_db_pairs: Optional train DB pairs for collect override.
        eval_db_pairs: Optional eval DB pairs for collect override.

    Returns:
        PipelineResult: Aggregated step results.

    Raises:
        ValueError: If argument combinations are invalid.
        RuntimeError: If full profile readiness check fails.
    """
    if steps is not None and profile is not None:
        raise ValueError("steps and profile are mutually exclusive")
    if profile is not None and profile not in PIPELINE_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")

    train_paths = tuple(train_db_paths or ())
    eval_paths = tuple(eval_db_paths or ())
    train_pairs = tuple(train_db_pairs or ())
    eval_pairs = tuple(eval_db_pairs or ())
    if steps is not None:
        selected = normalize_steps(steps)
    elif profile == "full" or profile is None:
        selected = steps_for_profile("prepare")
    else:
        selected = steps_for_profile(cast(PipelineProfile, profile))

    results = [STEP_ACTIONS[step](config, train_paths, eval_paths, train_pairs, eval_pairs) for step in selected]
    if profile == "full":
        _ensure_full_profile_ready(config)
        results.extend(
            STEP_ACTIONS[step](config, train_paths, eval_paths, train_pairs, eval_pairs)
            for step in steps_for_profile("analyze")
        )

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
