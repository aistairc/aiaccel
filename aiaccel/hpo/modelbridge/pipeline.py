"""Modelbridge pipeline orchestrator."""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence
from pathlib import Path

from .config import BridgeConfig
from .ops import (
    run_evaluation,
    run_external_command,
    run_hpo_phase,
    run_regression,
    run_summary,
)
from .utils import hash_file, setup_logging, write_json


def run_pipeline(config: BridgeConfig, steps: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute the modelbridge pipeline.

    This function orchestrates the HPO, regression, evaluation, and summary phases
    for all configured scenarios.

    Args:
        config (BridgeConfig): Validated configuration object.
        steps (Sequence[str] | None): List of steps to execute.
            If None, all steps are executed.
            Choices: "train", "eval", "regression", "evaluation", "summary", "da".

    Returns:
        dict[str, Any]: Manifest dictionary containing execution status and artifact paths.
    """
    target_steps = set(steps) if steps else None

    def should_run(step_name: str) -> bool:
        return target_steps is None or step_name in target_steps

    # Setup logging
    setup_logging(config.bridge.log_level, config.bridge.output_dir, json_logs=config.bridge.json_log)

    manifest: dict[str, Any] = {
        "config": config.model_dump(mode="json"),
        "scenarios": {},
        "artifacts": [],
    }

    # Execute scenarios
    for scenario in config.bridge.scenarios:
        scenario_dir = config.bridge.output_dir / scenario.name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: HPO (Train)
        if should_run("train"):
            run_hpo_phase(
                settings=config.hpo,
                scenario=scenario,
                role="train",
                runs=config.bridge.train_runs,
                seed_base=config.bridge.seed,
                scenario_dir=scenario_dir,
            )

        # Step 2: HPO (Eval)
        if should_run("eval") and config.bridge.eval_runs > 0:
            run_hpo_phase(
                settings=config.hpo,
                scenario=scenario,
                role="eval",
                runs=config.bridge.eval_runs,
                seed_base=config.bridge.seed,
                scenario_dir=scenario_dir,
            )

        # Step 3: Regression
        if should_run("regression"):
            run_regression(scenario, scenario_dir)

        # Step 4: Evaluation
        if should_run("evaluation"):
            run_evaluation(scenario, scenario_dir)

        # Manifest update per scenario
        manifest["scenarios"][scenario.name] = {"status": "completed", "dir": str(scenario_dir)}

    # Step 5: Summary
    if should_run("summary"):
        run_summary(config.bridge.scenarios, config.bridge.output_dir)

    # Step 6: Data Assimilation
    if should_run("da") and config.data_assimilation and config.data_assimilation.enabled:
        run_external_command(config.data_assimilation)

    # Finalize Manifest
    manifest["artifacts"] = _collect_artifacts(config.bridge.output_dir)
    write_json(config.bridge.output_dir / "manifest.json", manifest)

    return manifest


def _collect_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []

    # Common artifacts
    for name in ["summary.json", "manifest.json", "aiaccel_modelbridge.log"]:
        p = output_dir / name
        if p.exists():
            artifacts.append({"path": str(p), "sha256": hash_file(p), "size": p.stat().st_size})

    # Scan scenarios
    for s_dir in output_dir.iterdir():
        if not s_dir.is_dir() or s_dir.name in ["logs", "data_assimilation"]:
            continue

        # Models and metrics
        for cat in ["models", "metrics"]:
            d = s_dir / cat
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        artifacts.append({"path": str(f), "sha256": hash_file(f), "size": f.stat().st_size})

    # Data assimilation
    da_manifest = output_dir / "data_assimilation_manifest.json"
    if da_manifest.exists():
        artifacts.append(
            {"path": str(da_manifest), "sha256": hash_file(da_manifest), "size": da_manifest.stat().st_size}
        )

    return artifacts
