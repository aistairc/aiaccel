"""Modelbridge pipeline orchestration."""

from __future__ import annotations

from typing import Literal

from collections.abc import Sequence
from pathlib import Path

from .analyze import evaluate_model, fit_regression
from .collect import collect_eval, collect_train
from .config import BridgeConfig
from .layout import plan_path
from .prepare import prepare_eval, prepare_train
from .publish import publish_summary
from .toolkit.io import read_json
from .toolkit.results import PipelineResult, StepResult

PipelineProfile = Literal["prepare", "analyze", "full"]


PROFILE_STEPS: dict[PipelineProfile, list[str]] = {
    "prepare": ["prepare_train", "prepare_eval"],
    "analyze": [
        "collect_train",
        "collect_eval",
        "fit_regression",
        "evaluate_model",
        "publish_summary",
    ],
    "full": [
        "prepare_train",
        "prepare_eval",
        "collect_train",
        "collect_eval",
        "fit_regression",
        "evaluate_model",
        "publish_summary",
    ],
}

VALID_STEPS = {
    "prepare_train",
    "prepare_eval",
    "collect_train",
    "collect_eval",
    "fit_regression",
    "evaluate_model",
    "publish_summary",
}


class ModelBridgePipeline:
    """Orchestrate modelbridge lifecycle steps.

    Args:
        config: Parsed modelbridge configuration.
        train_db_paths: Optional explicit DB paths for train collection.
        eval_db_paths: Optional explicit DB paths for eval collection.
        train_db_pairs: Optional explicit train DB pairs.
        eval_db_pairs: Optional explicit eval DB pairs.
    """

    def __init__(
        self,
        config: BridgeConfig,
        *,
        train_db_paths: Sequence[Path] | None = None,
        eval_db_paths: Sequence[Path] | None = None,
        train_db_pairs: Sequence[tuple[Path, Path]] | None = None,
        eval_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    ) -> None:
        self.config = config
        self.train_db_paths = list(train_db_paths or [])
        self.eval_db_paths = list(eval_db_paths or [])
        self.train_db_pairs = list(train_db_pairs or [])
        self.eval_db_pairs = list(eval_db_pairs or [])

    def run_step(self, step_name: str) -> StepResult:
        """Execute one lifecycle step.

        Args:
            step_name: Canonical step name.

        Returns:
            StepResult: Step execution result.

        Raises:
            ValueError: If the requested step name is unknown.
        """
        match step_name:
            case "prepare_train":
                return prepare_train(self.config)
            case "prepare_eval":
                return prepare_eval(self.config)
            case "collect_train":
                return collect_train(
                    self.config,
                    db_paths=self.train_db_paths or None,
                    db_pairs=self.train_db_pairs or None,
                )
            case "collect_eval":
                return collect_eval(
                    self.config,
                    db_paths=self.eval_db_paths or None,
                    db_pairs=self.eval_db_pairs or None,
                )
            case "fit_regression":
                return fit_regression(self.config)
            case "evaluate_model":
                return evaluate_model(self.config)
            case "publish_summary":
                return publish_summary(self.config)
            case _:
                raise ValueError(f"Unknown step: {step_name}")

    def run_profile(self, profile_name: PipelineProfile) -> list[StepResult]:
        """Execute a named profile.

        Args:
            profile_name: Profile key (`prepare`, `analyze`, `full`).

        Returns:
            list[StepResult]: Ordered step results for the profile.

        Raises:
            ValueError: If the profile is unknown.
        """
        if profile_name not in PROFILE_STEPS:
            raise ValueError(f"Unknown profile: {profile_name}")
        results: list[StepResult] = []
        if profile_name == "full":
            # `full` crosses an external execution boundary between prepare and analyze.
            # Ensure the expected Optuna DB files already exist before continuing.
            results.extend([self.run_step("prepare_train"), self.run_step("prepare_eval")])
            self._ensure_external_hpo_ready()
            for step in PROFILE_STEPS["analyze"]:
                results.append(self.run_step(step))
            return results

        for step in PROFILE_STEPS[profile_name]:
            results.append(self.run_step(step))
        return results

    def _ensure_external_hpo_ready(self) -> None:
        """Validate that required Optuna DB artifacts exist for full profile execution.

        Raises:
            RuntimeError: If the plan payload is malformed or expected DB files are missing.
        """
        missing_db_paths = self._collect_missing_expected_db_paths()

        if missing_db_paths:
            preview = ", ".join(str(path) for path in missing_db_paths[:5])
            if len(missing_db_paths) > 5:
                preview = f"{preview}, ... (+{len(missing_db_paths) - 5} more)"
            raise RuntimeError(
                f"Full profile requires external HPO outputs before collect/analyze; missing optuna DB files: {preview}"
            )

    def _collect_missing_expected_db_paths(self) -> list[Path]:
        """Collect missing expected DB paths from role plans."""
        missing_db_paths: list[Path] = []
        for role in ("train", "eval"):
            role_plan_path = plan_path(self.config.bridge.output_dir, role)
            if not role_plan_path.exists():
                continue
            entries = self._load_plan_entries(role_plan_path)
            missing_db_paths.extend(self._find_missing_db_paths(entries, role_plan_path))
        return missing_db_paths

    @staticmethod
    def _load_plan_entries(role_plan_path: Path) -> list[object]:
        """Load and validate plan entry list."""
        payload = read_json(role_plan_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Malformed plan payload: {role_plan_path}")
        entries = payload.get("entries", [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Malformed plan entries in: {role_plan_path}")
        return entries

    @staticmethod
    def _find_missing_db_paths(entries: list[object], role_plan_path: Path) -> list[Path]:
        """Resolve missing expected DB paths from plan entries."""
        missing_db_paths: list[Path] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RuntimeError(f"Malformed plan entry at {role_plan_path}#{index}")
            expected_db_path = entry.get("expected_db_path")
            if not isinstance(expected_db_path, str) or not expected_db_path:
                raise RuntimeError(f"Plan entry missing expected_db_path at {role_plan_path}#{index}")
            db_path = Path(expected_db_path)
            if not db_path.exists():
                missing_db_paths.append(db_path)
        return missing_db_paths


def run_pipeline(
    config: BridgeConfig,
    steps: Sequence[str] | None = None,
    *,
    profile: PipelineProfile | None = None,
    train_db_paths: Sequence[Path] | None = None,
    eval_db_paths: Sequence[Path] | None = None,
    train_db_pairs: Sequence[tuple[Path, Path]] | None = None,
    eval_db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> PipelineResult:
    """Run selected steps or a profile and return the aggregated result.

    Args:
        config: Parsed modelbridge configuration.
        steps: Optional explicit step list.
        profile: Optional named profile.
        train_db_paths: Optional explicit DB paths for train collection.
        eval_db_paths: Optional explicit DB paths for eval collection.
        train_db_pairs: Optional explicit train DB pairs.
        eval_db_pairs: Optional explicit eval DB pairs.

    Returns:
        PipelineResult: Aggregated pipeline result.

    Raises:
        ValueError: If arguments are inconsistent or unknown step/profile is requested.
    """
    if steps is not None and profile is not None:
        raise ValueError("steps and profile are mutually exclusive")

    pipeline = ModelBridgePipeline(
        config,
        train_db_paths=train_db_paths,
        eval_db_paths=eval_db_paths,
        train_db_pairs=train_db_pairs,
        eval_db_pairs=eval_db_pairs,
    )

    results: list[StepResult]
    if steps is None and profile is None:
        results = pipeline.run_profile("prepare")
    elif profile is not None:
        results = pipeline.run_profile(profile)
    else:
        unknown = [step for step in steps or [] if step not in VALID_STEPS]
        if unknown:
            raise ValueError(f"Unknown step(s): {', '.join(unknown)}")
        results = [pipeline.run_step(step) for step in (steps or [])]

    summary_path: Path | None = None
    manifest_path: Path | None = None
    for result in results:
        if result.step == "publish_summary":
            summary_str = result.outputs.get("summary_path")
            manifest_str = result.outputs.get("manifest_path")
            if isinstance(summary_str, str):
                summary_path = Path(summary_str)
            if isinstance(manifest_str, str):
                manifest_path = Path(manifest_str)

    return PipelineResult(results=results, summary_path=summary_path, manifest_path=manifest_path)
