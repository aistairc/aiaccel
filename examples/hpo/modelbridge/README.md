# Modelbridge Examples

## Overview
This directory contains examples for the Modelbridge workflow in aiaccel-hpo.

### File Structure
- **Makefile.template**: Template for running the pipeline using `aiaccel-job`.
- **modelbridge.yaml**: Main configuration file for the example scenario.
- **optimize_config.yaml**: Base optimization configuration (algorithm, resource).
- **objectives/**: Directory containing Python objective functions.
  - `simple_objective.py`: A simple quadratic function used in the example.
- **data_assimilation/**: Example for data assimilation workflow.
  - `Makefile.template`: Makefile for the data assimilation example.
  - `data_assimilation.yaml`: Configuration for data assimilation.
  - `mas_bench_*.py`: Source codes for the benchmark.

## Quick Start
You can run the pipeline using the provided Makefile or directly via the CLI.

### 1. Using Makefile (Recommended)
Copy `examples/hpo/modelbridge/Makefile.template` next to your configuration or call it directly.

**Run the full pipeline:**
```bash
make -f examples/hpo/modelbridge/Makefile.template \
  run \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
  MODELBRIDGE_OUTPUT=./work/modelbridge/simple
```

**Run individual steps:**
```bash
# Setup: Generate directories and config files for HPO
make -f examples/hpo/modelbridge/Makefile.template setup ...

# Train: Run HPO loop for training data (requires setup)
make -f examples/hpo/modelbridge/Makefile.template train ...

# Eval: Run HPO loop for evaluation data (requires setup)
make -f examples/hpo/modelbridge/Makefile.template eval ...

# Train regression model
make -f examples/hpo/modelbridge/Makefile.template regression ...

# Evaluate regression model
make -f examples/hpo/modelbridge/Makefile.template evaluate_model ...

# Create summary
make -f examples/hpo/modelbridge/Makefile.template summary ...

# Run Data Assimilation (if configured)
make -f examples/hpo/modelbridge/Makefile.template da ...
```

### 2. Using CLI Directly
You can invoke the modelbridge CLI directly.

**Input/Output Arguments:**
- `--config <path>`: Path to the bridge configuration YAML (Required).
- `--output_dir <path>`: Output directory. Overrides config value if specified.

**Options:**
- `--steps <list>`: Comma-separated list of steps to execute.
  - Choices: `setup_train`, `setup_eval`, `regression`, `evaluate_model`, `summary`, `da`.
  - Default: All steps.
- `--json-log`: Emit JSON structured logs.
- `--quiet`: Suppress console logs.

**Example:**
```bash
# Run only the setup_train step (generates configs)
aiaccel-hpo modelbridge run \
  --config examples/hpo/modelbridge/modelbridge.yaml \
  --output_dir ./work/modelbridge/simple \
  --steps setup_train
```

## Resuming / Skipping HPO
The pipeline skips HPO execution for a specific run if its `optuna.db` already exists in the expected location.
To skip computation (e.g., if HPO was run on a cluster):
1. Place the `optuna.db` file in `work/modelbridge/simple/runs/<role>/<run_id>/<target>/` (adjust output path as needed).
   - Example: `work/modelbridge/simple/runs/train/000/macro/optuna.db`
2. **Constraint**: The Optuna Study Name stored *inside* the DB file MUST match the naming convention:
   - `{scenario}-{role}-{target}-{run_id}` (run_id is zero-padded 3 digits)
   - Example: `simple-train-macro-000`
   If the study name mismatches, the pipeline will typically create a new study (ignoring existing data) or fail depending on Optuna behavior.

## Seed Generation and Limitations
To ensure reproducibility, random seeds for HPO trials are generated deterministically using the following formula:
`seed = seed_base + (group_offset * 100000) + run_idx`

Where `group_offset` is determined by the phase and target:
- `train` / `macro`: 0
- `train` / `micro`: 1
- `eval` / `macro`: 2 (previously `eval` role)
- `eval` / `micro`: 3

**Limitation**: The stride between groups is **100,000**. If the number of runs (`train_runs` or `eval_runs`) exceeds 100,000, the seed values will collide with the next group.

## Optional Gaussian Process Regression (GPy)
- Install the optional dependency with `python -m pip install .[gpy]`.
- Update the scenario’s `regression` block:
  ```yaml
  regression:
    kind: gpr
    kernel: RBF
    noise: 1.0e-5
  ```

## Data Assimilation Example
- A MAS-Bench-inspired data assimilation workflow lives under `examples/hpo/modelbridge/data_assimilation/`.
- Run with:
  ```bash
  make -f examples/hpo/modelbridge/data_assimilation/Makefile.template run
  ```

## Schema & Validation
- Print the JSON Schema for the Pydantic v2 configuration:
  ```bash
  aiaccel-hpo modelbridge schema
  ```
- Validate the configuration:
  ```bash
  aiaccel-hpo modelbridge validate --config ./modelbridge.yaml
  ```