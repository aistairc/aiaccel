"""Data assimilation pipeline for MAS-Bench-style macro/micro bridging."""

from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import subprocess

import optuna
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .config import DataAssimilationConfig
from .io import hash_file, write_json
from .logging import configure_logging, get_logger


@dataclass(slots=True)
class MASBenchExecutor:
    """Helper to run MAS-Bench or emit mock results."""

    config: DataAssimilationConfig
    logger: Any = field(default_factory=lambda: get_logger(__name__))

    def agent_sizes(self) -> tuple[int, int, int]:
        if self.config.agent_sizes:
            return (
                int(self.config.agent_sizes.get("naive", 0)),
                int(self.config.agent_sizes.get("rational", 0)),
                int(self.config.agent_sizes.get("ruby", 0)),
            )
        dataset_root = self.config.dataset_root
        if not dataset_root:
            raise FileNotFoundError("dataset_root is not set and agent_sizes are not provided")
        script_path = Path(dataset_root) / self.config.micro_model / "agent_size.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"agent_size.sh not found at {script_path}")
        # shell out to source and echo variables
        shell_script = f"source {script_path} && echo $NAIVE_AGENT $RATIONAL_AGENT $RUBY_AGENT"
        output = subprocess.check_output(["bash", "-c", shell_script], text=True).strip()
        naive, rational, ruby = map(int, output.split())
        return naive, rational, ruby

    def run_simulation(self, model: str, run_dir: Path, input_csv: Path, mock: bool, error_value: float) -> float:
        if mock:
            self._write_mock_fitness(run_dir, error_value)
            return error_value
        jar = self.config.mas_bench_jar
        if not jar or not Path(jar).exists():
            raise FileNotFoundError("MAS-Bench.jar is missing and allow_mock is False")
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["java", "-jar", str(jar), model, str(run_dir), str(input_csv)],
            check=True,
        )
        fitness = self._read_fitness(run_dir)
        return fitness

    def _write_mock_fitness(self, run_dir: Path, error: float) -> None:
        analyze_dir = run_dir / "analyze"
        analyze_dir.mkdir(parents=True, exist_ok=True)
        fitness_path = analyze_dir / "Fitness.csv"
        with fitness_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["AllError"])
            writer.writerow([error])

    def _read_fitness(self, run_dir: Path) -> float:
        fitness_path = run_dir / "analyze" / "Fitness.csv"
        if not fitness_path.exists():
            raise FileNotFoundError(f"Fitness.csv not found under {run_dir}")
        with fitness_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            row = next(reader, None)
            if not row:
                raise ValueError(f"Fitness.csv is empty at {fitness_path}")
            return float(row[0])


def run_data_assimilation(
    config: DataAssimilationConfig,
    *,
    dry_run: bool = False,
    quiet: bool = True,
    log_to_file: bool = False,
    json_logs: bool = False,
) -> dict[str, Any]:
    """Execute the data assimilation workflow."""

    if dry_run:
        return {
            "phases": [
                "generate_micro",
                "assimilate_macro_train",
                "assimilate_macro_test",
                "bridge_regress",
                "bridge_predict_and_run",
            ]
        }

    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    console_enabled = not quiet
    if log_to_file or console_enabled:
        configure_logging(
            "INFO",
            output_root,
            reset_handlers=True,
            console=console_enabled,
            file=log_to_file,
            json_logs=json_logs,
        )
    logger = get_logger(__name__)
    executor = MASBenchExecutor(config, logger=logger)
    mock = bool(config.allow_mock)

    # Phase 1: micro scenarios
    micro_results = _run_micro_optimization(config, executor, mock)
    logger.info("Completed micro scenarios: %d trials", len(micro_results))

    # Phase 2: macro train assimilation across scenarios
    macro_train_results = _run_macro_train(config, executor, mock)
    logger.info("Completed macro train assimilation: %d scenarios", len(macro_train_results))

    # Phase 3: macro test assimilation
    macro_test_result = _run_macro_test(config, executor, mock)
    logger.info("Completed macro test assimilation")

    # Phase 4: regression macro->micro
    regression_payload = _run_regression(config, micro_results, macro_train_results, macro_test_result)
    logger.info("Completed regression; predicted micro params: %s", regression_payload["predicted_micro"])

    # Phase 5: run predicted micro
    bridged_error = _run_predicted_micro(config, executor, regression_payload["predicted_micro"], mock)
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
    _write_da_manifest(output_root, summary_path)
    return summary


def _suggest_params(trial: optuna.Trial, agent_sizes: tuple[int, int, int], cfg: DataAssimilationConfig) -> tuple[list[float], list[float], list[float], list[str]]:
    naive, rational, ruby = agent_sizes
    total_agents = naive + rational + ruby
    sigma: list[float] = []
    mu: list[float] = []
    pi: list[float] = []
    header: list[str] = []
    cnt = 0
    for i in range(naive):
        sigma.append(trial.suggest_float(f"sigma_naive{i}", 0.0, 1.0))
        mu.append(trial.suggest_float(f"mu_naive{i}", 0.0, 1.0))
        if cnt + 1 < total_agents:
            pi.append(trial.suggest_float(f"pi_naive{i}", 0.0, 1.0))
        header += [f"sigma_naive{i}", f"mu_naive{i}", f"pi_naive{i}"]
        cnt += 1
    for i in range(rational):
        sigma.append(trial.suggest_float(f"sigma_rational{i}", 0.0, 1.0))
        mu.append(trial.suggest_float(f"mu_rational{i}", 0.0, 1.0))
        if cnt + 1 < total_agents:
            pi.append(trial.suggest_float(f"pi_rational{i}", 0.0, 1.0))
        header += [f"sigma_rational{i}", f"mu_rational{i}", f"pi_rational{i}"]
        cnt += 1
    for i in range(ruby):
        sigma.append(trial.suggest_float(f"sigma_ruby{i}", 0.0, 1.0))
        mu.append(trial.suggest_float(f"mu_ruby{i}", 0.0, 1.0))
        if cnt + 1 < total_agents:
            pi.append(trial.suggest_float(f"pi_ruby{i}", 0.0, 1.0))
        header += [f"sigma_ruby{i}", f"mu_ruby{i}", f"pi_ruby{i}"]
        cnt += 1
    if total_agents > 0:
        pi_sum = sum(pi)
        last_pi = max(0.0, min(1.0, 1.0 - pi_sum))
        pi.append(last_pi)
    else:
        pi = [1.0]
    return sigma, mu, pi, header


def _scale_params(sigma: list[float], mu: list[float], scaling: DataAssimilationConfig) -> tuple[list[float], list[float]]:
    s_mult, s_offset = scaling.scaling.sigma_scale
    sigma_scaled = [s * s_mult + s_offset for s in sigma]
    mu_scaled = [m * scaling.scaling.mu_scale for m in mu]
    return sigma_scaled, mu_scaled


def _write_input_csv(run_dir: Path, trial_number: int, sigma: list[float], mu: list[float], pi: list[float], header: list[str]) -> Path:
    input_dir = run_dir.parent / "input_parameters"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_csv = input_dir / f"input_parameter_{trial_number}.csv"
    row: list[float] = []
    total_agents = len(pi)
    for i in range(total_agents):
        row.extend([sigma[i], mu[i], pi[i]])
    with input_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(row)
    return input_csv


def _run_micro_optimization(config: DataAssimilationConfig, executor: MASBenchExecutor, mock: bool) -> list[dict[str, float]]:
    naive, rational, ruby = executor.agent_sizes()
    sampler = _make_sampler(config.samplers.micro, config.seeds.micro)
    study_name = f"{config.micro_model}-micro-{config.trials.micro}-{config.seeds.micro}"
    storage = None

    def objective(trial: optuna.Trial) -> float:
        sigma, mu, pi, header = _suggest_params(trial, (naive, rational, ruby), config)
        sigma_scaled, mu_scaled = _scale_params(sigma, mu, config)
        run_dir = config.output_root / "micro" / study_name / f"{trial.number}"
        input_csv = _write_input_csv(run_dir, trial.number, sigma_scaled, mu_scaled, pi, header)
        error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
        return executor.run_simulation(config.micro_model, run_dir, input_csv, mock, error)

    study = optuna.create_study(sampler=sampler, direction="minimize", storage=storage, study_name=study_name, load_if_exists=False)
    study.optimize(objective, n_trials=config.trials.micro)
    best_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    return [trial.params for trial in best_trials]


def _run_macro_train(config: DataAssimilationConfig, executor: MASBenchExecutor, mock: bool) -> list[dict[str, float]]:
    sampler_name = config.samplers.macro_train
    sampler_seed = config.seeds.macro_train
    trial_count = config.trials.macro_train
    results: list[dict[str, float]] = []

    for idx in range(config.scenarios):
        sampler = _make_sampler(sampler_name, sampler_seed)
        study_name = f"{config.macro_model}-train-{idx}-{sampler_name}-{trial_count}-{sampler_seed}"
        storage = None

        def objective(trial: optuna.Trial) -> float:
            naive, rational, ruby = executor.agent_sizes()
            sigma, mu, pi, header = _suggest_params(trial, (naive, rational, ruby), config)
            sigma_scaled, mu_scaled = _scale_params(sigma, mu, config)
            run_dir = config.output_root / "macro_train" / study_name / f"{trial.number}"
            input_csv = _write_input_csv(run_dir, trial.number, sigma_scaled, mu_scaled, pi, header)
            error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
            return executor.run_simulation(config.macro_model, run_dir, input_csv, mock, error)

        study = optuna.create_study(
            sampler=sampler,
            direction="minimize",
            storage=storage,
            study_name=study_name,
            load_if_exists=False,
        )
        study.optimize(objective, n_trials=trial_count)
        if study.best_trial:
            results.append(study.best_trial.params)
    return results


def _run_macro_test(config: DataAssimilationConfig, executor: MASBenchExecutor, mock: bool) -> dict[str, float]:
    sampler = _make_sampler(config.samplers.macro_test, config.seeds.macro_test)
    study_name = f"{config.macro_model}-test-{config.samplers.macro_test}-{config.trials.macro_test}-{config.seeds.macro_test}"
    storage = None

    def objective(trial: optuna.Trial) -> float:
        naive, rational, ruby = executor.agent_sizes()
        sigma, mu, pi, header = _suggest_params(trial, (naive, rational, ruby), config)
        sigma_scaled, mu_scaled = _scale_params(sigma, mu, config)
        run_dir = config.output_root / "macro_test" / study_name / f"{trial.number}"
        input_csv = _write_input_csv(run_dir, trial.number, sigma_scaled, mu_scaled, pi, header)
        error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
        return executor.run_simulation(config.macro_model, run_dir, input_csv, mock, error)

    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
        storage=storage,
        study_name=study_name,
        load_if_exists=False,
    )
    study.optimize(objective, n_trials=config.trials.macro_test)
    if study.best_trial:
        return study.best_trial.params
    return {}


def _run_regression(
    config: DataAssimilationConfig,
    micro_params: list[dict[str, float]],
    macro_train_params: list[dict[str, float]],
    macro_test_best: dict[str, float],
) -> dict[str, Any]:
    if not micro_params or not macro_train_params:
        raise RuntimeError("Insufficient data for regression")
    micro_df = pd.DataFrame(micro_params)
    macro_train_df = pd.DataFrame(macro_train_params)
    macro_test_df = pd.DataFrame([macro_test_best])

    degree = config.regression_degree
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(macro_train_df.to_numpy(), micro_df.to_numpy())

    y_pred_train = model.predict(macro_train_df.to_numpy())
    mae = mean_absolute_error(micro_df.to_numpy(), y_pred_train)
    r2 = r2_score(micro_df.to_numpy(), y_pred_train) if len(micro_df) > 1 else None
    predicted_micro = model.predict(macro_test_df.to_numpy())[0].tolist()
    predicted_dict = {col: predicted_micro[idx] for idx, col in enumerate(micro_df.columns)}

    regression_payload = {
        "degree": degree,
        "mae_train": float(mae),
        "r2_train": float(r2) if r2 is not None else None,
        "predicted_micro": predicted_dict,
    }
    write_json(config.output_root / "data_assimilation_regression.json", regression_payload)
    return regression_payload


def _run_predicted_micro(
    config: DataAssimilationConfig,
    executor: MASBenchExecutor,
    predicted_micro: dict[str, float],
    mock: bool,
) -> float:
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

    sigma_scaled, mu_scaled = _scale_params(sigma, mu, config)
    run_dir = config.output_root / "bridge_predict" / "run_0"
    input_csv = _write_input_csv(run_dir, 0, sigma_scaled, mu_scaled, pi, header)
    error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
    return executor.run_simulation(config.micro_model, run_dir, input_csv, mock, error)


def _make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name in {"cmaes", "cma-es"}:
        try:
            import importlib.util

            if importlib.util.find_spec("cmaes") is None:
                return optuna.samplers.RandomSampler(seed=seed)
            return optuna.samplers.CmaEsSampler(seed=seed)
        except Exception:
            return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unsupported sampler '{name}'")


def _write_da_manifest(output_root: Path, summary_path: Path) -> None:
    artifacts = []
    for path in [
        summary_path,
        output_root / "data_assimilation_regression.json",
    ]:
        if path.exists():
            artifacts.append(
                {"path": str(path), "sha256": hash_file(path), "size": path.stat().st_size, "algorithm": "sha256"}
            )
    write_json(output_root / "data_assimilation_manifest.json", {"artifacts": artifacts})


__all__ = ["run_data_assimilation"]
