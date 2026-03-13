"""Run modelbridge benchmarks across multiple function pair scenarios."""

from __future__ import annotations

from typing import Any

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    macro_function: str
    micro_function: str


FUNCTION_IDS = {
    "sphere": 0,
    "rastrigin": 1,
    "griewank": 2,
}

SCENARIOS = (
    ScenarioSpec(name="sphere_to_rastrigin", macro_function="sphere", micro_function="rastrigin"),
    ScenarioSpec(name="rastrigin_to_sphere", macro_function="rastrigin", micro_function="sphere"),
    ScenarioSpec(name="griewank_to_sphere", macro_function="griewank", micro_function="sphere"),
)


def _base_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing_pythonpath else f"{repo_root}:{existing_pythonpath}"
    return env


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(cwd), env=env)


def _run_optimize(config_path: Path, *, repo_root: Path) -> None:
    _run(
        [sys.executable, str(repo_root / "aiaccel" / "hpo" / "apps" / "optimize.py"), "--config", str(config_path)],
        cwd=repo_root,
        env=_base_env(repo_root),
    )


def _run_modelbridge_tool(tool_name: str, *, repo_root: Path, args: list[str]) -> None:
    _run(
        [sys.executable, str(repo_root / "aiaccel" / "hpo" / "modelbridge" / tool_name), *args],
        cwd=repo_root,
        env=_base_env(repo_root),
    )


def _build_space(function_id: int) -> dict[str, dict[str, float]]:
    return {
        "x1": {"low": -5.0, "high": 5.0},
        "x2": {"low": -5.0, "high": 5.0},
        "function_id": {"low": float(function_id), "high": float(function_id)},
    }


def _build_config(
    *,
    objective_script: Path,
    scenario: ScenarioSpec,
    n_train: int,
    n_test: int,
    n_trials: int,
) -> dict[str, Any]:
    macro_id = FUNCTION_IDS[scenario.macro_function]
    micro_id = FUNCTION_IDS[scenario.micro_function]
    return {
        "n_train": n_train,
        "n_test": n_test,
        "objective_command": [sys.executable, str(objective_script)],
        "train_macro_trials": n_trials,
        "train_micro_trials": n_trials,
        "test_macro_trials": n_trials,
        "test_micro_trials": n_trials,
        "train_params": {
            "macro": _build_space(macro_id),
            "micro": _build_space(micro_id),
        },
        "test_params": {
            "macro": _build_space(macro_id),
            "micro": _build_space(micro_id),
        },
    }


def run_scenario(
    *,
    workspace_root: Path,
    scenario: ScenarioSpec,
    n_train: int,
    n_test: int,
    n_trials: int,
) -> dict[str, Any]:
    scenario_workspace = workspace_root / scenario.name
    scenario_workspace.mkdir(parents=True, exist_ok=True)

    objective_script = (REPO_ROOT / "examples" / "hpo" / "modelbridge" / "objectives" / "multi_objective.py").resolve()
    config = _build_config(
        objective_script=objective_script,
        scenario=scenario,
        n_train=n_train,
        n_test=n_test,
        n_trials=n_trials,
    )
    config_path = scenario_workspace / "benchmark_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    _run_modelbridge_tool(
        "prepare.py",
        repo_root=REPO_ROOT,
        args=["--config", str(config_path), "--workspace", str(scenario_workspace)],
    )

    for phase in ("train", "test"):
        for run_config in sorted((scenario_workspace / "runs" / phase).rglob("config.yaml")):
            _run_optimize(run_config, repo_root=REPO_ROOT)

    _run_modelbridge_tool(
        "collect.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(scenario_workspace), "--phase", "train"],
    )
    _run_modelbridge_tool(
        "collect.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(scenario_workspace), "--phase", "test"],
    )
    _run_modelbridge_tool(
        "fit_model.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(scenario_workspace)],
    )
    _run_modelbridge_tool(
        "evaluate.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(scenario_workspace)],
    )
    summary_path = scenario_workspace / "models" / "summary.json"

    summary: dict[str, Any] = (
        {"summary": None} if not summary_path.exists() else json.loads(summary_path.read_text(encoding="utf-8"))
    )
    summary.update(
        {
            "scenario": scenario.name,
            "macro_function": scenario.macro_function,
            "micro_function": scenario.micro_function,
            "workspace": str(scenario_workspace),
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-function modelbridge benchmarks.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT / "examples" / "hpo" / "modelbridge" / "workspace" / "benchmark_multi",
        help="Workspace root for scenario artifacts.",
    )
    parser.add_argument("--scenario", type=str, default="all", help="Scenario name or 'all'.")
    parser.add_argument("--n-train", type=int, default=2, help="Number of train runs.")
    parser.add_argument("--n-test", type=int, default=1, help="Number of test runs.")
    parser.add_argument("--trials", type=int, default=8, help="Number of Optuna trials per target.")
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the workspace root before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace_root = args.workspace.resolve()
    if bool(args.clean) and workspace_root.exists():
        shutil.rmtree(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    selected_scenarios = [
        scenario for scenario in SCENARIOS if args.scenario == "all" or scenario.name == args.scenario
    ]
    if not selected_scenarios:
        available = ", ".join(scenario.name for scenario in SCENARIOS)
        raise SystemExit(f"Unknown scenario: {args.scenario}. Available: all, {available}")

    results = [
        run_scenario(
            workspace_root=workspace_root,
            scenario=scenario,
            n_train=int(args.n_train),
            n_test=int(args.n_test),
            n_trials=int(args.trials),
        )
        for scenario in selected_scenarios
    ]
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
