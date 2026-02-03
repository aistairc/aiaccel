"""Modelbridge pipeline orchestrator."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from pathlib import Path

from .config import BridgeConfig, DataAssimilationConfig
from .ops import (
    run_da_step,
    run_evaluate_model_step,
    run_regression_step,
    run_setup_eval_step,
    run_setup_train_step,
    run_summary_step,
)
from .utils import hash_file, setup_logging, write_json


class ModelBridgePipeline:
    """Orchestrates the modelbridge pipeline steps."""

    def __init__(self, config: BridgeConfig):
        """Initialize the pipeline.

        Args:
            config (BridgeConfig): Validated configuration object.
        """
        self.config = config
        # Setup logging immediately upon initialization
        setup_logging(
            config.bridge.log_level,
            config.bridge.output_dir,
            json_logs=config.bridge.json_log,
        )

    def run_step(self, step_name: str) -> None:
        """Execute a single pipeline step.

        Args:
            step_name (str): The name of the step to execute.
        """
        match step_name:
            case "setup_train":
                self.run_setup_train()
            case "setup_eval":
                self.run_setup_eval()
            case "regression":
                self.run_regression()
            case "evaluate_model":
                self.run_evaluate_model()
            case "summary":
                self.run_summary()
            case "da":
                self.run_da()
            case _:
                raise ValueError(f"Unknown step: {step_name}")

    def run_setup_train(self) -> None:
        """Execute Setup Train step for all scenarios."""
        for scenario in self.config.bridge.scenarios:
            scenario_dir = self.config.bridge.output_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            run_setup_train_step(
                settings=self.config.hpo,
                scenario=scenario,
                runs=self.config.bridge.train_runs,
                seed_base=self.config.bridge.seed,
                scenario_dir=scenario_dir,
            )

    def run_setup_eval(self) -> None:
        """Execute Setup Eval step for all scenarios."""
        if self.config.bridge.eval_runs <= 0:
            return

        for scenario in self.config.bridge.scenarios:
            scenario_dir = self.config.bridge.output_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            run_setup_eval_step(
                settings=self.config.hpo,
                scenario=scenario,
                runs=self.config.bridge.eval_runs,
                seed_base=self.config.bridge.seed,
                scenario_dir=scenario_dir,
            )

    def run_regression(self) -> None:
        """Execute Regression step for all scenarios."""
        for scenario in self.config.bridge.scenarios:
            scenario_dir = self.config.bridge.output_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            run_regression_step(scenario, scenario_dir)

    def run_evaluate_model(self) -> None:
        """Execute Evaluation step for all scenarios."""
        for scenario in self.config.bridge.scenarios:
            scenario_dir = self.config.bridge.output_dir / scenario.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            run_evaluate_model_step(scenario, scenario_dir)

    def run_summary(self) -> None:
        """Execute Summary step."""
        run_summary_step(self.config.bridge.scenarios, self.config.bridge.output_dir)
        self._create_manifest()

    def run_da(self) -> None:
        """Execute Data Assimilation step."""
        if self.config.data_assimilation and self.config.data_assimilation.enabled:
            if self.config.data_assimilation.output_root is None:
                self.config.data_assimilation.output_root = self.config.bridge.output_dir / "data_assimilation"
            run_da_step(self.config.data_assimilation)

    def _create_manifest(self) -> None:
        """Create and save manifest.json."""
        manifest: dict[str, Any] = {
            "config": self.config.model_dump(mode="json"),
            "scenarios": {},
            "artifacts": _collect_artifacts(self.config.bridge.output_dir, self.config.data_assimilation),
        }

        for scenario in self.config.bridge.scenarios:
            s_dir = self.config.bridge.output_dir / scenario.name
            if s_dir.exists():
                manifest["scenarios"][scenario.name] = {"status": "exist", "dir": str(s_dir)}

        write_json(self.config.bridge.output_dir / "manifest.json", manifest)


def run_pipeline(config: BridgeConfig, steps: Sequence[str] | None = None) -> dict[str, Any]:
    """Execute the modelbridge pipeline.

    Args:
        config (BridgeConfig): Validated configuration object.
        steps (Sequence[str] | None): List of steps to execute.
            If None, all steps are executed.

    Returns:
        dict[str, Any]: Manifest dictionary (reloaded from disk if available).
    """
    pipeline = ModelBridgePipeline(config)

    if steps is None:
        # Default order
        steps_to_run = ["setup_train", "setup_eval", "regression", "evaluate_model", "summary", "da"]
    else:
        steps_to_run = list(steps)

    for step in steps_to_run:
        pipeline.run_step(step)

    # Return manifest if it exists
    manifest_path = config.bridge.output_dir / "manifest.json"
    if manifest_path.exists():
        import json

        with manifest_path.open() as f:
            return cast(dict[str, Any], json.load(f))
    return {}


def _collect_artifacts(
    output_dir: Path, data_assimilation: DataAssimilationConfig | None = None
) -> list[dict[str, Any]]:
    artifacts = []
    artifacts.extend(_collect_common_artifacts(output_dir))
    artifacts.extend(_collect_scenario_artifacts(output_dir))
    artifacts.extend(_collect_da_artifacts(output_dir, data_assimilation))
    return artifacts


def _collect_common_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for name in ["summary.json", "manifest.json", "aiaccel_modelbridge.log"]:
        path = output_dir / name
        if path.exists():
            artifacts.append({"path": str(path), "sha256": hash_file(path), "size": path.stat().st_size})
    return artifacts


def _collect_scenario_artifacts(output_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for s_dir in output_dir.iterdir():
        if not s_dir.is_dir() or s_dir.name in ["logs", "data_assimilation"]:
            continue
        for cat in ["models", "metrics"]:
            cat_dir = s_dir / cat
            if not cat_dir.exists():
                continue
            for file_path in cat_dir.iterdir():
                if file_path.is_file():
                    artifacts.append(
                        {"path": str(file_path), "sha256": hash_file(file_path), "size": file_path.stat().st_size}
                    )
    return artifacts


def _collect_da_artifacts(output_dir: Path, data_assimilation: DataAssimilationConfig | None) -> list[dict[str, Any]]:
    if data_assimilation is None:
        return []
    da_root = data_assimilation.output_root or (output_dir / "data_assimilation")
    da_manifest = da_root / "data_assimilation_manifest.json"
    if not da_manifest.exists():
        return []
    return [{"path": str(da_manifest), "sha256": hash_file(da_manifest), "size": da_manifest.stat().st_size}]
