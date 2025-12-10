"""Wrapper script to run MAS-Bench data assimilation logic using aiaccel-hpo optimize."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from mas_bench_utils import MASBenchExecutor, get_logger, scale_params, write_input_csv, write_json

# Add aiaccel package path to find config resources if needed, 
# but we are generating full config here.


import importlib.util

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
    
    params_def = {
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
    
    # Construct command args
    # python mas_bench_objective.py --config ... --model ... --out_dir ... --trial_id ... --mock ... --param={param}
    
    objective_script = Path(__file__).parent / "mas_bench_objective.py"
    cmd = [
        sys.executable,
        str(objective_script.resolve()),
        "--config", str(config_path.resolve()),
        "--model", model_name,
        "--out_dir", "{out_filename}_dir", # aiaccel uses out_filename as json path, we append _dir
        "--out_file", "{out_filename}",
        "--trial_id", "{job_name}",
    ]
    if mock:
        cmd.append("--mock")
    
    # Append dynamic params
    for p in param_names:
        cmd.append(f"--{p}={{{p}}}")

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
        "n_max_jobs": 1, # Sequential execution as per original logic? Or parallel? Original was sequential implicitly via Optuna default? Optuna is seq unless n_jobs != 1.
        # Original wrapper ran loop sequentially? No, study.optimize is sequential by default.
        "command": cmd,
    }

    aiaccel_config_path = work_dir / "aiaccel_config.yaml"
    with aiaccel_config_path.open("w") as f:
        yaml.dump(aiaccel_config, f)
        
    logger.info(f"Running optimization '{study_name}' with config {aiaccel_config_path}")
    logger.info(f"Config content:\n{yaml.dump(aiaccel_config)}")
    
    subprocess.run(
        ["aiaccel-hpo", "optimize", "--config", str(aiaccel_config_path)],
        check=True,
    )
    
    # Load results
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return study


def _run_micro_optimization(
    config: dict[str, Any], executor: MASBenchExecutor, mock: bool, output_root: Path, config_path: Path
) -> list[dict[str, float]]:
    samplers_cfg = config.get("samplers", {})
    seeds_cfg = config.get("seeds", {})
    trials_cfg = config.get("trials", {})
    
    sampler_name = samplers_cfg.get("micro", "random")
    seed = int(seeds_cfg.get("micro", 0))
    n_trials = int(trials_cfg.get("micro", 1))
    model = config["micro_model"]
    
    study_name = f"{model}-micro-{n_trials}-{seed}"
    output_dir = output_root / "micro"
    
    logger = get_logger(__name__)
    study = _run_aiaccel_optimization(
        study_name, output_dir, sampler_name, seed, n_trials, model, config_path, mock, executor.agent_sizes(), logger
    )
    
    best_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    return [trial.params for trial in best_trials]


def _run_macro_train(
    config: dict[str, Any], executor: MASBenchExecutor, mock: bool, output_root: Path, config_path: Path
) -> list[dict[str, float]]:
    samplers_cfg = config.get("samplers", {})
    seeds_cfg = config.get("seeds", {})
    trials_cfg = config.get("trials", {})
    
    sampler_name = samplers_cfg.get("macro_train", "cmaes")
    seed = int(seeds_cfg.get("macro_train", 0))
    n_trials = int(trials_cfg.get("macro_train", 1))
    model = config["macro_model"]
    
    results: list[dict[str, float]] = []
    output_dir = output_root / "macro_train"
    logger = get_logger(__name__)

    for idx in range(int(config.get("scenarios", 1))):
        study_name = f"{model}-train-{idx}-{sampler_name}-{n_trials}-{seed}"
        
        study = _run_aiaccel_optimization(
            study_name, output_dir, sampler_name, seed, n_trials, model, config_path, mock, executor.agent_sizes(), logger
        )
        
        if study.best_trial:
            results.append(study.best_trial.params)
            
    return results


def _run_macro_test(
    config: dict[str, Any], executor: MASBenchExecutor, mock: bool, output_root: Path, config_path: Path
) -> dict[str, float]:
    samplers_cfg = config.get("samplers", {})
    seeds_cfg = config.get("seeds", {})
    trials_cfg = config.get("trials", {})
    
    sampler_name = samplers_cfg.get("macro_test", "cmaes")
    seed = int(seeds_cfg.get("macro_test", 0))
    n_trials = int(trials_cfg.get("macro_test", 1))
    model = config["macro_model"]
    
    study_name = f"{model}-test-{sampler_name}-{n_trials}-{seed}"
    output_dir = output_root / "macro_test"
    logger = get_logger(__name__)
    
    study = _run_aiaccel_optimization(
        study_name, output_dir, sampler_name, seed, n_trials, model, config_path, mock, executor.agent_sizes(), logger
    )

    if study.best_trial:
        return study.best_trial.params
    return {}


def _run_regression(
    config: dict[str, Any],
    micro_params: list[dict[str, float]],
    macro_train_params: list[dict[str, float]],
    macro_test_best: dict[str, float],
    output_root: Path,
) -> dict[str, Any]:
    if not micro_params or not macro_train_params:
        raise RuntimeError("Insufficient data for regression")
    micro_df = pd.DataFrame(micro_params)
    macro_train_df = pd.DataFrame(macro_train_params)
    macro_test_df = pd.DataFrame([macro_test_best])

    degree = config.get("regression_degree", 1)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(macro_train_df.to_numpy(), micro_df.to_numpy())

    y_pred_train = model.predict(macro_train_df.to_numpy())
    mae = mean_absolute_error(micro_df.to_numpy(), y_pred_train)
    r2 = r2_score(micro_df.to_numpy(), y_pred_train) if len(micro_df) > 1 else None
    
    # Predict for test
    if not macro_test_df.empty and not macro_test_df.isna().all().all():
        predicted_micro = model.predict(macro_test_df.to_numpy())[0].tolist()
    else:
        # Fallback if no test result
        predicted_micro = [0.0] * len(micro_df.columns)

    predicted_dict = {col: predicted_micro[idx] for idx, col in enumerate(micro_df.columns)}

    regression_payload = {
        "degree": degree,
        "mae_train": float(mae),
        "r2_train": float(r2) if r2 is not None else None,
        "predicted_micro": predicted_dict,
    }
    write_json(output_root / "data_assimilation_regression.json", regression_payload)
    return regression_payload


def _run_predicted_micro(
    config: dict[str, Any],
    executor: MASBenchExecutor,
    predicted_micro: dict[str, float],
    mock: bool,
    output_root: Path,
) -> float:
    # This uses direct executor logic, not aiaccel, as it's a single validation run
    naive, rational, ruby = executor.agent_sizes()
    sigma: list[float] = []
    mu: list[float] = []
    pi: list[float] = []
    header: list[str] = []
    idx = 0

    def _extract(prefix: str, count: int) -> None:
        nonlocal idx
        for i in range(count):
            s = predicted_micro.get(f"sigma_{prefix}{i}", 0.0)
            m = predicted_micro.get(f"mu_{prefix}{i}", 0.0)
            p = predicted_micro.get(f"pi_{prefix}{i}", 0.0)
            sigma.append(s)
            mu.append(m)
            if idx + 1 < naive + rational + ruby:
                pi.append(p)
            header.extend([f"sigma_{prefix}{i}", f"mu_{prefix}{i}", f"pi_{prefix}{i}"])
            idx += 1

    _extract("naive", naive)
    _extract("rational", rational)
    _extract("ruby", ruby)
    if (naive + rational + ruby) > 0:
        pi_last = max(0.0, min(1.0, 1.0 - sum(pi)))
        pi.append(pi_last)
    else:
        pi = [1.0]

    sigma_scaled, mu_scaled = scale_params(sigma, mu, config)
    run_dir = output_root / "bridge_predict" / "run_0"
    input_csv = write_input_csv(run_dir, 0, sigma_scaled, mu_scaled, pi, header)
    error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
    return executor.run_simulation(config["micro_model"], run_dir, input_csv, mock, error)


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS-Bench Wrapper")
    parser.add_argument("--config", required=True, help="Path to configuration YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    output_root = Path(config.get("output_root", "./work/modelbridge/data_assimilation"))
    output_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__)
    executor = MASBenchExecutor(config, logger=logger)
    mock = bool(config.get("allow_mock", False))

    logger.info("Starting MAS-Bench data assimilation")

    # Phase 1: micro scenarios
    micro_results = _run_micro_optimization(config, executor, mock, output_root, config_path)
    logger.info("Completed micro scenarios: %d trials", len(micro_results))

    # Phase 2: macro train assimilation across scenarios
    macro_train_results = _run_macro_train(config, executor, mock, output_root, config_path)
    logger.info("Completed macro train assimilation: %d scenarios", len(macro_train_results))

    # Phase 3: macro test assimilation
    macro_test_result = _run_macro_test(config, executor, mock, output_root, config_path)
    logger.info("Completed macro test assimilation")

    # Phase 4: regression macro->micro
    regression_payload = _run_regression(config, micro_results, macro_train_results, macro_test_result, output_root)
    logger.info("Completed regression; predicted micro params: %s", regression_payload["predicted_micro"])

    # Phase 5: run predicted micro
    bridged_error = _run_predicted_micro(config, executor, regression_payload["predicted_micro"], mock, output_root)
    logger.info("Completed bridged simulation with error %.4f", bridged_error)

    summary = {
        "micro_best": micro_results,
        "macro_train_best": macro_train_results,
        "macro_test_best": macro_test_result,
        "regression": regression_payload,
        "bridged_error": bridged_error,
    }
    summary_path = output_root / "data_assimilation_summary.json"
    write_json(summary_path, summary)

    logger.info("Success")


if __name__ == "__main__":
    main()