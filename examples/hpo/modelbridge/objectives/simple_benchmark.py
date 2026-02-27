"""Run a small end-to-end benchmark with the Spec17 modelbridge pipeline."""

from __future__ import annotations

from typing import Any

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


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


def _build_config(
    *,
    objective_script: Path,
    n_train: int,
    n_test: int,
    n_trials: int,
) -> dict[str, Any]:
    return {
        "n_train": n_train,
        "n_test": n_test,
        "objective_command": [sys.executable, str(objective_script)],
        "train_macro_trials": n_trials,
        "train_micro_trials": n_trials,
        "test_macro_trials": n_trials,
        "test_micro_trials": n_trials,
        "train_params": {
            "macro": {"x": {"low": -1.0, "high": 1.0}, "y": {"low": -1.0, "high": 1.0}},
            "micro": {"x": {"low": -1.0, "high": 1.0}, "y": {"low": -1.0, "high": 1.0}},
        },
        "test_params": {
            "macro": {"x": {"low": -1.0, "high": 1.0}, "y": {"low": -1.0, "high": 1.0}},
            "micro": {"x": {"low": -1.0, "high": 1.0}, "y": {"low": -1.0, "high": 1.0}},
        },
    }


def run_benchmark(*, workspace: Path, n_train: int, n_test: int, n_trials: int, clean: bool) -> dict[str, Any]:
    if clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    objective_script = (REPO_ROOT / "examples" / "hpo" / "modelbridge" / "objectives" / "simple_objective.py").resolve()
    config = _build_config(
        objective_script=objective_script,
        n_train=n_train,
        n_test=n_test,
        n_trials=n_trials,
    )
    config_path = workspace / "benchmark_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    _run_modelbridge_tool(
        "prepare.py",
        repo_root=REPO_ROOT,
        args=["--config", str(config_path), "--workspace", str(workspace)],
    )

    for phase in ("train", "test"):
        for run_config in sorted((workspace / "runs" / phase).rglob("config.yaml")):
            _run_optimize(run_config, repo_root=REPO_ROOT)

    _run_modelbridge_tool(
        "collect.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(workspace), "--phase", "train"],
    )
    _run_modelbridge_tool(
        "collect.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(workspace), "--phase", "test"],
    )
    _run_modelbridge_tool(
        "fit_model.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(workspace)],
    )
    _run_modelbridge_tool(
        "evaluate.py",
        repo_root=REPO_ROOT,
        args=["--workspace", str(workspace)],
    )
    summary_path = workspace / "models" / "summary.json"

    summary: dict[str, Any]
    if not summary_path.exists():
        summary = {"summary": None, "workspace": str(workspace)}
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["workspace"] = str(workspace)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spec17 simple benchmark pipeline.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT / "examples" / "hpo" / "modelbridge" / "workspace" / "benchmark_simple",
        help="Workspace directory for benchmark artifacts.",
    )
    parser.add_argument("--n-train", type=int, default=2, help="Number of train runs.")
    parser.add_argument("--n-test", type=int, default=1, help="Number of test runs.")
    parser.add_argument("--trials", type=int, default=6, help="Number of Optuna trials per target.")
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the workspace before running.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_benchmark(
        workspace=args.workspace.resolve(),
        n_train=int(args.n_train),
        n_test=int(args.n_test),
        n_trials=int(args.trials),
        clean=bool(args.clean),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
