"""Generate per-run optimize configs for the modelbridge workflow."""

from __future__ import annotations

from typing import Any

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

DEFAULT_SEED_BASES: dict[str, int] = {
    "train_macro": 42,
    "train_micro": 142,
    "test_macro": 1042,
    "test_micro": 1142,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the prepare step.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Modelbridge Prepare Step")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    return parser.parse_args(argv)


def _resolve_objective_command(command: Sequence[str], *, config_path: Path) -> list[str]:
    """Resolve relative objective script paths against config-neighbor directories.

    Args:
        command: Original objective command tokens.
        config_path: Configuration file used to derive search roots.

    Returns:
        Resolved command tokens. The second token is rewritten when a matching script path exists.
    """
    resolved = [str(token) for token in command]
    if len(resolved) < 2:
        return resolved

    candidate = Path(resolved[1]).expanduser()
    if candidate.is_absolute():
        return resolved

    # Prefer config dir and its parent so examples/config/config.yaml can reference
    # objectives under examples/hpo/modelbridge/objectives.
    search_roots = (config_path.parent, config_path.parent.parent)
    for root in search_roots:
        script_path = (root / candidate).resolve()
        if script_path.exists():
            resolved[1] = str(script_path)
            return resolved
    return resolved


def create_hpo_config(
    output_dir: Path,
    *,
    role: str,
    target: str,
    run_id: int,
    sampler_seed_base: int,
    n_trials: int,
    target_params: Mapping[str, Mapping[str, Any]],
    objective_command: Sequence[str],
) -> Path:
    """Generate one ``aiaccel-hpo optimize`` config file for a specific run.

    Args:
        output_dir: Run directory where ``config.yaml`` and artifacts are created.
        role: Phase role name (for example ``train`` or ``test``).
        target: Optimization target name (``macro`` or ``micro``).
        run_id: Zero-based run identifier.
        sampler_seed_base: Seed base offset for deterministic Optuna sampling.
        n_trials: Number of optimization trials.
        target_params: Parameter-space bounds for the selected target.
        objective_command: Objective command template before parameter placeholders are appended.

    Returns:
        Generated config path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    command = list(objective_command)
    for param_name in target_params:
        command.append(f"--{param_name}={{{param_name}}}")
    command.append("{out_filename}")

    db_path = output_dir / "optuna.db"
    hpo_config: dict[str, Any] = {
        "n_max_jobs": 1,
        "n_trials": n_trials,
        "working_directory": str(output_dir.resolve()),
        "command": command,
        "study": {
            "_target_": "optuna.create_study",
            "study_name": f"{role}-{target}-{run_id:03d}",
            "storage": f"sqlite:///{db_path.resolve()}",
            "direction": "minimize",
            "load_if_exists": True,
            "sampler": {
                "_target_": "optuna.samplers.TPESampler",
                "seed": sampler_seed_base + run_id,
            },
        },
        "params": {
            "_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager",
        },
    }
    for param_name, bounds in target_params.items():
        hpo_config["params"][param_name] = {
            "_target_": "aiaccel.hpo.optuna.hparams.Float",
            "low": bounds.get("low", 0.0),
            "high": bounds.get("high", 1.0),
            "log": bounds.get("log", False),
        }

    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(hpo_config, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return config_path


def _as_int(value: Any, *, key: str) -> int:
    """Parse a non-negative integer from config input.

    Args:
        value: Raw configuration value.
        key: Human-readable key name for error messages.

    Returns:
        Parsed non-negative integer value.

    Raises:
        ValueError: If the value is negative or cannot be converted to an integer.
    """
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{key} must be >= 0")
    return parsed


def _load_seed_bases(config: Mapping[str, Any]) -> dict[str, int]:
    """Load sampler seed-base defaults and optional overrides from config.

    Args:
        config: Root prepare configuration mapping.

    Returns:
        Seed base mapping for all train/test and macro/micro combinations.

    Raises:
        ValueError: If ``seed_defaults`` is malformed or contains unsupported keys.
    """
    raw = config.get("seed_defaults", {})
    if raw is None:
        return dict(DEFAULT_SEED_BASES)
    if not isinstance(raw, Mapping):
        raise ValueError("seed_defaults must be a mapping")

    seed_bases = dict(DEFAULT_SEED_BASES)
    for key, value in raw.items():
        if key in seed_bases:
            seed_bases[key] = _as_int(value, key=f"seed_defaults.{key}")
        elif key in {"train", "test"}:
            phase = str(key)
            phase_seed = _as_int(value, key=f"seed_defaults.{phase}")
            seed_bases[f"{phase}_macro"] = phase_seed
            seed_bases[f"{phase}_micro"] = phase_seed + 100
        else:
            raise ValueError(f"Unsupported seed_defaults key: {key}")
    return seed_bases


def run_prepare(config_path: Path, workspace: Path) -> tuple[int, int]:
    """Generate train/test optimize configs under ``workspace/runs``.

    Args:
        config_path: Path to the modelbridge workflow config YAML.
        workspace: Workspace root where run configs are generated.

    Returns:
        Tuple of ``(n_train, n_test)`` generated run counts.

    Raises:
        ValueError: If config values are malformed or unsupported.
    """
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config: dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    n_train = _as_int(config.get("n_train", 0), key="n_train")
    n_test = _as_int(config.get("n_test", 0), key="n_test")
    raw_command = config.get("objective_command", ["python", "objective.py"])
    if not isinstance(raw_command, list) or any(not isinstance(token, str) for token in raw_command):
        raise ValueError("objective_command must be a list[str]")
    objective_command = _resolve_objective_command(raw_command, config_path=config_path)
    seed_bases = _load_seed_bases(config)
    runs_dir = workspace / "runs"

    for run_id in range(n_train):
        params = config.get("train_params", {})
        for target in ("macro", "micro"):
            target_params = params.get(target, {}) if isinstance(params, dict) else {}
            n_trials = _as_int(config.get(f"train_{target}_trials", 10), key=f"train_{target}_trials")
            create_hpo_config(
                runs_dir / "train" / target / f"{run_id:03d}",
                role="train",
                target=target,
                run_id=run_id,
                sampler_seed_base=seed_bases[f"train_{target}"],
                n_trials=n_trials,
                target_params=target_params,
                objective_command=objective_command,
            )

    for run_id in range(n_test):
        params = config.get("test_params", {})
        for target in ("macro", "micro"):
            target_params = params.get(target, {}) if isinstance(params, dict) else {}
            n_trials = _as_int(config.get(f"test_{target}_trials", 10), key=f"test_{target}_trials")
            create_hpo_config(
                runs_dir / "test" / target / f"{run_id:03d}",
                role="test",
                target=target,
                run_id=run_id,
                sampler_seed_base=seed_bases[f"test_{target}"],
                n_trials=n_trials,
                target_params=target_params,
                objective_command=objective_command,
            )
    return n_train, n_test


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prepare step CLI.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Process exit code. ``0`` on success.
    """
    args = parse_args(argv)
    n_train, n_test = run_prepare(config_path=Path(args.config), workspace=Path(args.workspace))
    print(f"[Prepare] Scattered configs for {n_train} train runs and {n_test} test runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
