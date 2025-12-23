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
1. Copy `examples/hpo/modelbridge/Makefile.template` next to your configuration or call it directly with `make -f`.
2. Execute the pipeline:
   ```bash
   make -f examples/hpo/modelbridge/Makefile.template \
     pipeline \
     MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
     MODELBRIDGE_OUTPUT=./work/modelbridge/simple \
     MODELBRIDGE_SCENARIO=simple
   ```
3. Inspect artifacts under `work/modelbridge/simple`.
   - `manifest.json` lists generated artifacts with sizes and sha256 hashes.
   - Set `MODELBRIDGE_JSON_LOG=1` to emit JSON-formatted pipeline logs.

## Resuming / Skipping HPO
The pipeline skips HPO execution for a specific run if its `optuna.db` already exists in the expected location.
To skip computation (e.g., if HPO was run on a cluster):
1. Place the `optuna.db` file in `work/modelbridge/simple/runs/<role>/<run_id>/<target>/` (adjust output path as needed).
   - Example: `work/modelbridge/simple/runs/train/000/macro/optuna.db`
2. **Constraint**: The Optuna Study Name stored *inside* the DB file MUST match the naming convention:
   - `{scenario}-{role}-{target}-{run_id}` (run_id is zero-padded 3 digits)
   - Example: `simple-train-macro-000`
   If the study name mismatches, the pipeline will typically create a new study (ignoring existing data) or fail depending on Optuna behavior.

## Makefile Template Highlights
- The `pipeline` target chains all phases for a full run by calling `aiaccel-hpo modelbridge run`.
- Tunable variables:
  - `MODELBRIDGE_CONFIG`: path to the YAML configuration.
  - `MODELBRIDGE_OUTPUT`: destination for scenario results.
  - `MODELBRIDGE_LOG_DIR`: directory for logs (defaults to `$(MODELBRIDGE_OUTPUT)/logs`).
  - `MODELBRIDGE_SET`: additional CLI overrides (space-separated `--set key=value` entries).
  - `MODELBRIDGE_JSON_LOG`: set to `1`/`true` to enable structured logs.
  - `AIACCEL_CMD`: command wrapper (defaults to `aiaccel-job` logging to `$(MODELBRIDGE_LOG_DIR)/aiaccel_job.log`).

## PYTHONPATH
- The Makefile.template prepends the repository root to `PYTHONPATH` before invoking the CLI.

## Optional Gaussian Process Regression (GPy)
- Install the optional dependency with `python -m pip install .[gpy]`.
- Update the scenario’s `regression` block:
  ```yaml
  regression:
    kind: gpr
    kernel: RBF
    noise: 1.0e-5
  ```

## Objective Configuration
- The objective must be an executable command string or list of strings.
- Placeholders like `{out_filename}` are supported (passed by `aiaccel-hpo`).
- Example:
  ```yaml
  train_objective:
    command: ["python", "examples/hpo/modelbridge/objectives/simple_objective.py", "{out_filename}"]
  ```

## Data Assimilation Example
- A MAS-Bench-inspired data assimilation workflow lives under `examples/hpo/modelbridge/data_assimilation/`.
- Run with:
  ```bash
  make -f examples/hpo/modelbridge/data_assimilation/Makefile.template data-assimilation
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
