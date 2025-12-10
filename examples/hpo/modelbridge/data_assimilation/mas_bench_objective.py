"""Objective script for MAS-Bench data assimilation executed by aiaccel-hpo."""

import argparse
import sys
from pathlib import Path
import yaml

from mas_bench_utils import MASBenchExecutor, scale_params, write_input_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--trial_id", required=True)
    parser.add_argument("--mock", action="store_true")
    
    # Capture all other arguments as hyperparameters
    args, unknown = parser.parse_known_args()
    
    # Parse unknown args as key=value pairs
    params = {}
    for arg in unknown:
        if arg.startswith("--"):
            key, val = arg.lstrip("-").split("=", 1)
            params[key] = float(val)

    config_path = Path(args.config)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    executor = MASBenchExecutor(config)
    naive, rational, ruby = executor.agent_sizes()
    total_agents = naive + rational + ruby

    # Reconstruct sigma, mu, pi lists
    sigma = []
    mu = []
    pi = []
    header = []

    def _collect(prefix, count):
        for i in range(count):
            sigma.append(params.get(f"sigma_{prefix}{i}", 0.0))
            mu.append(params.get(f"mu_{prefix}{i}", 0.0))
            # pi is present for all except the very last agent overall?
            # Re-checking logic:
            # In wrapper:
            # if cnt + 1 < total_agents: append pi
            # So the last agent overall does not have a pi param.
            
            # We need to know global index to decide if we look for pi
            current_idx = len(sigma) - 1 # 0-based index of current agent
            
            if current_idx < total_agents - 1:
                pi.append(params.get(f"pi_{prefix}{i}", 0.0))
                header.extend([f"sigma_{prefix}{i}", f"mu_{prefix}{i}", f"pi_{prefix}{i}"])
            else:
                # Last agent
                header.extend([f"sigma_{prefix}{i}", f"mu_{prefix}{i}", f"pi_{prefix}{i}"])

    _collect("naive", naive)
    _collect("rational", rational)
    _collect("ruby", ruby)

    # Calculate last pi
    if total_agents > 0:
        last_pi = max(0.0, min(1.0, 1.0 - sum(pi)))
        pi.append(last_pi)
    else:
        pi = [1.0]

    sigma_scaled, mu_scaled = scale_params(sigma, mu, config)
    
    run_dir = Path(args.out_dir)
    
    trial_id_str = args.trial_id
    if trial_id_str.startswith("trial_"):
        trial_id_str = trial_id_str[6:]
    trial_id = int(trial_id_str)
    
    input_csv = write_input_csv(run_dir, trial_id, sigma_scaled, mu_scaled, pi, header)
    
    # Mock error calculation matches wrapper
    mock_error = sum(sigma_scaled) + sum(mu_scaled) + sum(pi)
    
    result = executor.run_simulation(args.model, run_dir, input_csv, args.mock, mock_error)
    
    with open(args.out_file, "w") as f:
        # aiaccel expects a JSON value (float or list of floats)
        import json
        json.dump(result, f)

if __name__ == "__main__":
    main()
