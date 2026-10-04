"""Objective script for MAS-Bench data assimilation executed by aiaccel-hpo."""

import argparse
import json
from pathlib import Path

from mas_bench_utils import MASBenchExecutor, mock_error, scale_params, write_input_csv
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mock_target", type=float, default=0.0, help="Scenario condition used by mock runs")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--mock", action="store_true")

    # All other ``--name=value`` arguments are hyperparameters.
    args, unknown = parser.parse_known_args()
    params = {}
    for arg in unknown:
        if arg.startswith("--"):
            key, val = arg.lstrip("-").split("=", 1)
            params[key] = float(val)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    executor = MASBenchExecutor(config)
    naive, rational, ruby = executor.agent_sizes()
    total_agents = naive + rational + ruby

    sigma: list[float] = []
    mu: list[float] = []
    pi: list[float] = []
    header: list[str] = []
    for prefix, count in (("naive", naive), ("rational", rational), ("ruby", ruby)):
        for i in range(count):
            sigma.append(params.get(f"sigma_{prefix}{i}", 0.0))
            mu.append(params.get(f"mu_{prefix}{i}", 0.0))
            # The last agent has no pi parameter; its share is the complement below.
            if len(sigma) < total_agents:
                pi.append(params.get(f"pi_{prefix}{i}", 0.0))
            header.extend([f"sigma_{prefix}{i}", f"mu_{prefix}{i}", f"pi_{prefix}{i}"])
    pi.append(max(0.0, min(1.0, 1.0 - sum(pi))) if total_agents > 0 else 1.0)

    sigma_scaled, mu_scaled = scale_params(sigma, mu, config)
    run_dir = Path(args.out_dir)
    trial_id = int(args.trial_id.removeprefix("trial_"))
    input_csv = write_input_csv(run_dir, trial_id, sigma_scaled, mu_scaled, pi, header)

    error = mock_error(params.values(), args.mock_target)
    result = executor.run_simulation(args.model, run_dir, input_csv, args.mock, error)

    # aiaccel-hpo reads the objective value as JSON.
    Path(args.out_file).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
