"""Utility functions and classes for MAS-Bench data assimilation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def hash_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def scale_params(sigma: list[float], mu: list[float], config: dict[str, Any]) -> tuple[list[float], list[float]]:
    scaling = config.get("scaling", {})
    s_mult, s_offset = scaling.get("sigma_scale", (1.0, 0.0))
    sigma_scaled = [s * s_mult + s_offset for s in sigma]
    mu_scaled = [m * scaling.get("mu_scale", 1.0) for m in mu]
    return sigma_scaled, mu_scaled


def write_input_csv(
    run_dir: Path, trial_number: int, sigma: list[float], mu: list[float], pi: list[float], header: list[str]
) -> Path:
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


@dataclass(slots=True)
class MASBenchExecutor:
    """Helper to run MAS-Bench or emit mock results."""

    config: dict[str, Any]
    logger: logging.Logger = field(default_factory=lambda: get_logger(__name__))

    def agent_sizes(self) -> tuple[int, int, int]:
        sizes = self.config.get("agent_sizes")
        if sizes:
            return (
                int(sizes.get("naive", 0)),
                int(sizes.get("rational", 0)),
                int(sizes.get("ruby", 0)),
            )
        dataset_root = Path(self.config["dataset_root"]).resolve()
        script_path = dataset_root / self.config["micro_model"] / "agent_size.sh"
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

        jar = Path(self.config["mas_bench_jar"]).resolve()
        if not jar.exists():
            raise FileNotFoundError(f"MAS-Bench.jar is missing at {jar}")

        run_dir_abs = run_dir.resolve()
        run_dir_abs.mkdir(parents=True, exist_ok=True)
        input_csv_abs = input_csv.resolve()

        # Determine working directory for Java execution.
        dataset_root = Path(self.config["dataset_root"]).resolve()
        cwd = dataset_root

        try:
            result = subprocess.run(
                ["java", "-jar", str(jar), model, str(run_dir_abs), str(input_csv_abs)],
                check=True,
                cwd=cwd,
                capture_output=True,
                text=True
            )
            # MAS-Bench sometimes returns 0 even on error, so check stderr
            if "ITK_Error" in result.stderr or "Exception" in result.stderr:
                 self.logger.error(f"MAS-Bench execution error detected in stderr: {result.stderr}")
                 raise RuntimeError(f"MAS-Bench execution failed with errors in stderr: {result.stderr[:500]}...")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"MAS-Bench execution failed: {e.stderr}")
            raise

        fitness = self._read_fitness(run_dir_abs)
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
