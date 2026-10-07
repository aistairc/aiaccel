# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

"""Wrapper script to run MAS-Bench data assimilation logic using aiaccel-hpo optimize."""

from __future__ import annotations

from typing import Any, Protocol

import argparse
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import subprocess
import sys

import pandas as pd

from mas_bench_utils import MASBenchExecutor, get_logger, mock_error, scale_params, write_input_csv, write_json
import optuna
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import yaml


def _resolve_config_path(path_value: Any, *, config_dir: Path, prefer_existing: bool = False) -> str:
    """Resolve a config path value.

    Relative paths are resolved from config directory by default.
    When prefer_existing=True, also try current working directory and pick the
    first existing candidate.
    """
    path_str = str(path_value)
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path)

    config_based = (config_dir / path).resolve()
    if not prefer_existing:
        return str(config_based)

    cwd_based = path.resolve()
    if config_based.exists():
        return str(config_based)
    if cwd_based.exists():
        return str(cwd_based)
    return str(config_based)


def _normalize_runtime_config(
    config: dict[str, Any],
    *,
    config_path: Path,
    output_root_override: str | None,
) -> tuple[dict[str, Any], Path]:
    """Normalize path-like entries to absolute paths for stable subprocess execution."""
    normalized = dict(config)
    config_dir = config_path.parent.resolve()

    if "dataset_root" in normalized:
        normalized["dataset_root"] = _resolve_config_path(
            normalized["dataset_root"], config_dir=config_dir, prefer_existing=True
        )
    if "mas_bench_jar" in normalized:
        normalized["mas_bench_jar"] = _resolve_config_path(
            normalized["mas_bench_jar"], config_dir=config_dir, prefer_existing=True
        )

    output_root_raw = output_root_override if output_root_override is not None else normalized.get("output_root")
    if output_root_raw is None:
        output_root_raw = "./work/modelbridge/data_assimilation"
    output_root = Path(_resolve_config_path(output_root_raw, config_dir=config_dir))
    normalized["output_root"] = str(output_root)
    return normalized, output_root


class Executor(Protocol):
    """Interface of ``MASBenchExecutor`` used by this wrapper."""

    def agent_sizes(self) -> tuple[int, int, int]: ...

    def run_simulation(self, model: str, run_dir: Path, input_csv: Path, mock: bool, error_value: float) -> float: ...


@dataclass(frozen=True)
class Scenario:
    """One observation condition shared by the micro and macro studies of a regression sample.

    For real MAS-Bench runs, the condition is the observation dataset of ``micro_model`` and
    ``macro_model``. For mock runs, it is ``mock_target``, the parameter value that minimizes the
    mock error.
    """

    id: str
    micro_model: str
    macro_model: str
    mock_target: float


def _scenario_model(raw: dict[str, Any], *, config: dict[str, Any], name: str, key: str) -> str:
    """Return the scenario's model name, falling back to the top-level value only when it is absent."""
    for source in (raw, config):
        if name in source:
            return str(source[name])
    raise ValueError(f"{key} entry {raw['id']!r} has no '{name}' and no top-level '{name}' default")


def _parse_scenario(raw: Any, *, config: dict[str, Any], key: str) -> Scenario:
    if not isinstance(raw, dict) or "id" not in raw:
        raise ValueError(f"{key} entries must be mappings with an 'id', got {raw!r}")
    return Scenario(
        id=str(raw["id"]),
        micro_model=_scenario_model(raw, config=config, name="micro_model", key=key),
        macro_model=_scenario_model(raw, config=config, name="macro_model", key=key),
        mock_target=float(raw.get("mock_target", 0.0)),
    )


def _load_scenarios(config: dict[str, Any], *, mock: bool) -> tuple[list[Scenario], Scenario]:
    """Load train scenarios and the test scenario, rejecting duplicated observation conditions."""
    if "scenarios" in config:
        raise ValueError("'scenarios' was replaced by 'train_scenarios' and 'test_scenario'; see README.md")
    raw_train = config.get("train_scenarios")
    if not isinstance(raw_train, list) or not raw_train:
        raise ValueError("train_scenarios must be a non-empty list")
    train = [_parse_scenario(raw, config=config, key="train_scenarios") for raw in raw_train]
    test = _parse_scenario(config.get("test_scenario"), config=config, key="test_scenario")

    ids = [scenario.id for scenario in [*train, test]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Scenario ids must be unique, got {ids}")
    # A mock run is conditioned only by mock_target; a real run only by its models.
    conditions = [scenario.mock_target if mock else (scenario.micro_model, scenario.macro_model) for scenario in train]
    if len(set(conditions)) != len(conditions):
        field_name = "mock_target" if mock else "micro_model/macro_model"
        raise ValueError(f"train_scenarios must differ in {field_name}, got {conditions}")
    return train, test


def _get_sampler_config(name: str, seed: int) -> dict[str, Any]:
    name = name.lower()
    if name == "random":
        return {"_target_": "optuna.samplers.RandomSampler", "seed": seed}
    if name == "tpe":
        return {"_target_": "optuna.samplers.TPESampler", "seed": seed}
    if name in {"cmaes", "cma-es"}:
        if importlib.util.find_spec("cmaes") is None:
            logging.getLogger(__name__).warning("cmaes not found, falling back to RandomSampler")
            return {"_target_": "optuna.samplers.RandomSampler", "seed": seed}
        return {"_target_": "optuna.samplers.CmaEsSampler", "seed": seed}
    raise ValueError(f"Unsupported sampler '{name}'")


def _generate_params_config(
    agent_sizes: tuple[int, int, int],
) -> tuple[dict[str, Any], list[str]]:
    """Generates the params section for aiaccel config and the list of param names."""
    naive, rational, ruby = agent_sizes
    total_agents = naive + rational + ruby

    params_def: dict[str, Any] = {
        "_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager",
    }

    param_names = []

    cnt = 0
    for prefix, count in [("naive", naive), ("rational", rational), ("ruby", ruby)]:
        for i in range(count):
            for p_type in ["sigma", "mu"]:
                name = f"{p_type}_{prefix}{i}"
                params_def[name] = {
                    "_target_": "aiaccel.hpo.optuna.hparams.Float",
                    "low": 0.0,
                    "high": 1.0,
                }
                param_names.append(name)

            if cnt + 1 < total_agents:
                name = f"pi_{prefix}{i}"
                params_def[name] = {
                    "_target_": "aiaccel.hpo.optuna.hparams.Float",
                    "low": 0.0,
                    "high": 1.0,
                }
                param_names.append(name)
            cnt += 1

    return params_def, param_names


def _run_aiaccel_optimization(
    study_name: str,
    output_dir: Path,
    sampler_name: str,
    seed: int,
    n_trials: int,
    model_name: str,
    scenario: Scenario,
    config_path: Path,
    mock: bool,
    agent_sizes: tuple[int, int, int],
    logger: logging.Logger,
) -> optuna.Study:
    work_dir = output_dir / study_name
    work_dir.mkdir(parents=True, exist_ok=True)

    db_path = work_dir / "optuna.db"
    storage_url = f"sqlite:///{db_path.resolve()}"

    params_conf, param_names = _generate_params_config(agent_sizes)

    objective_script = Path(__file__).parent / "mas_bench_objective.py"
    cmd = [
        sys.executable,
        str(objective_script.resolve()),
        "--config",
        str(config_path.resolve()),
        "--model",
        model_name,
        "--mock_target",
        str(scenario.mock_target),
        "--out_dir",
        "{out_filename}_dir",  # aiaccel uses out_filename as json path, we append _dir
        "--out_file",
        "{out_filename}",
        "--trial_id",
        "{job_name}",
    ]
    if mock:
        cmd.append("--mock")
    cmd.extend(f"--{name}={{{name}}}" for name in param_names)

    aiaccel_config = {
        "study": {
            "_target_": "optuna.create_study",
            "study_name": study_name,
            "direction": "minimize",
            "storage": storage_url,
            "load_if_exists": True,
            "sampler": _get_sampler_config(sampler_name, seed),
        },
        "params": params_conf,
        "n_trials": n_trials,
        "n_max_jobs": 1,
        "command": cmd,
    }

    aiaccel_config_path = work_dir / "aiaccel_config.yaml"
    aiaccel_config_path.write_text(yaml.safe_dump(aiaccel_config, sort_keys=False), encoding="utf-8")
    logger.info("Running optimization '%s' with config %s", study_name, aiaccel_config_path)

    log_file = work_dir / "aiaccel_job_sub.log"
    subprocess.run(
        [
            "aiaccel-job",
            "local",
            "cpu",
            str(log_file),
            "--",
            "aiaccel-hpo",
            "optimize",
            "--config",
            str(aiaccel_config_path),
        ],
        check=True,
    )
    return optuna.load_study(study_name=study_name, storage=storage_url)


def _optimize_scenarios(
    config: dict[str, Any],
    *,
    stage: str,
    scenarios: list[Scenario],
    executor: Executor,
    mock: bool,
    output_root: Path,
    config_path: Path,
) -> dict[str, dict[str, float]]:
    """Run one study per scenario and return the best parameters keyed by scenario id.

    ``stage`` is ``micro``, ``macro_train``, or ``macro_test``; it selects the sampler, seed base,
    trial count, and model of each scenario.
    """
    sampler_name = str(config.get("samplers", {}).get(stage, "random"))
    seed_base = int(config.get("seeds", {}).get(stage, 0))
    n_trials = int(config.get("trials", {}).get(stage, 1))
    agent_sizes = executor.agent_sizes()
    logger = get_logger(__name__)

    results: dict[str, dict[str, float]] = {}
    for index, scenario in enumerate(scenarios):
        model = scenario.micro_model if stage == "micro" else scenario.macro_model
        seed = seed_base + index
        study = _run_aiaccel_optimization(
            f"{model}-{stage}-{scenario.id}-{sampler_name}-{n_trials}-{seed}",
            output_root / stage,
            sampler_name,
            seed,
            n_trials,
            model,
            scenario,
            config_path,
            mock,
            agent_sizes,
            logger,
        )
        completed = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if completed:
            results[scenario.id] = {name: float(value) for name, value in study.best_trial.params.items()}
        else:
            logger.warning("No completed trial for %s scenario %s", stage, scenario.id)
    return results


def _run_regression(
    config: dict[str, Any],
    micro_best: dict[str, dict[str, float]],
    macro_train_best: dict[str, dict[str, float]],
    macro_test_best: dict[str, float],
    output_root: Path,
) -> dict[str, Any]:
    """Fit macro->micro regression on scenarios that have both results, then predict the test scenario."""
    scenario_ids = [scenario_id for scenario_id in macro_train_best if scenario_id in micro_best]
    if not scenario_ids or not macro_test_best:
        raise RuntimeError("Insufficient data for regression: need paired train scenarios and a macro test result")
    macro_df = pd.DataFrame([macro_train_best[scenario_id] for scenario_id in scenario_ids]).sort_index(axis=1)
    micro_df = pd.DataFrame([micro_best[scenario_id] for scenario_id in scenario_ids]).sort_index(axis=1)
    macro_test_df = pd.DataFrame([macro_test_best])[macro_df.columns]

    degree = int(config.get("regression_degree", 1))
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(macro_df.to_numpy(), micro_df.to_numpy())

    y_pred_train = model.predict(macro_df.to_numpy())
    mae = mean_absolute_error(micro_df.to_numpy(), y_pred_train)
    r2 = r2_score(micro_df.to_numpy(), y_pred_train) if len(micro_df) > 1 else None
    predicted_micro = model.predict(macro_test_df.to_numpy())[0].tolist()

    regression_payload = {
        "degree": degree,
        "train_scenarios": scenario_ids,
        "mae_train": float(mae),
        "r2_train": float(r2) if r2 is not None else None,
        "predicted_micro": dict(zip(micro_df.columns, predicted_micro, strict=True)),
    }
    write_json(output_root / "data_assimilation_regression.json", regression_payload)
    return regression_payload


def _run_predicted_micro(
    config: dict[str, Any],
    executor: Executor,
    scenario: Scenario,
    predicted_micro: dict[str, float],
    mock: bool,
    output_root: Path,
) -> float:
    """Run the micro model of the test scenario once with the predicted micro parameters."""
    naive, rational, ruby = executor.agent_sizes()
    total_agents = naive + rational + ruby
    sigma: list[float] = []
    mu: list[float] = []
    pi: list[float] = []
    header: list[str] = []
    for prefix, count in (("naive", naive), ("rational", rational), ("ruby", ruby)):
        for i in range(count):
            sigma.append(predicted_micro.get(f"sigma_{prefix}{i}", 0.0))
            mu.append(predicted_micro.get(f"mu_{prefix}{i}", 0.0))
            # The last agent has no pi parameter; its share is the complement below.
            if len(sigma) < total_agents:
                pi.append(predicted_micro.get(f"pi_{prefix}{i}", 0.0))
            header.extend([f"sigma_{prefix}{i}", f"mu_{prefix}{i}", f"pi_{prefix}{i}"])
    pi.append(max(0.0, min(1.0, 1.0 - sum(pi))) if total_agents > 0 else 1.0)

    sigma_scaled, mu_scaled = scale_params(sigma, mu, config)
    run_dir = output_root / "bridge_predict" / "run_0"
    input_csv = write_input_csv(run_dir, 0, sigma_scaled, mu_scaled, pi, header)
    error = mock_error(predicted_micro.values(), scenario.mock_target)
    return executor.run_simulation(scenario.micro_model, run_dir, input_csv, mock, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS-Bench Wrapper")
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output directory override. If omitted, use output_root in config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    config, output_root = _normalize_runtime_config(
        loaded,
        config_path=config_path.resolve(),
        output_root_override=args.output_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_config_path = output_root / "resolved_mas_bench_config.yaml"
    runtime_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    logger = get_logger(__name__)
    executor = MASBenchExecutor(config, logger=logger)
    mock = bool(config.get("allow_mock", False))
    train_scenarios, test_scenario = _load_scenarios(config, mock=mock)

    logger.info("Starting MAS-Bench data assimilation (mock=%s)", mock)
    logger.info("Using resolved config: %s", runtime_config_path)
    common = {"executor": executor, "mock": mock, "output_root": output_root, "config_path": runtime_config_path}

    # Phase 1-3: micro and macro studies per train scenario, then the macro study of the test scenario.
    micro_best = _optimize_scenarios(config, stage="micro", scenarios=train_scenarios, **common)
    macro_train_best = _optimize_scenarios(config, stage="macro_train", scenarios=train_scenarios, **common)
    macro_test_best = _optimize_scenarios(config, stage="macro_test", scenarios=[test_scenario], **common)
    logger.info("Completed studies: micro=%d, macro_train=%d", len(micro_best), len(macro_train_best))

    # Phase 4: regression macro->micro over scenarios paired by id.
    regression_payload = _run_regression(
        config, micro_best, macro_train_best, macro_test_best.get(test_scenario.id, {}), output_root
    )
    logger.info("Completed regression; predicted micro params: %s", regression_payload["predicted_micro"])

    # Phase 5: run the test scenario's micro model with the predicted micro parameters.
    bridged_error = _run_predicted_micro(
        config, executor, test_scenario, regression_payload["predicted_micro"], mock, output_root
    )
    logger.info("Completed bridged simulation with error %.4f", bridged_error)

    summary = {
        "mock": mock,
        "test_scenario": test_scenario.id,
        "micro_best": micro_best,
        "macro_train_best": macro_train_best,
        "macro_test_best": macro_test_best,
        "regression": regression_payload,
        "bridged_error": bridged_error,
    }
    write_json(output_root / "data_assimilation_summary.json", summary)
    logger.info("Success")


if __name__ == "__main__":
    main()
