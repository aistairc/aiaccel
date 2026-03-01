# Modelbridge Examples

## Overview
Modelbridge is Makefile-first and shell-first.

Orchestration lives in:
- `Makefile`
- `scripts/*.sh`

Python under `aiaccel/hpo/modelbridge/` is limited to stateless step tools:
- `prepare.py`
- `collect.py`
- `fit_model.py`
- `evaluate.py`

## Directory Roles
- `config/config.yaml`: local run settings (direct `simple_objective.py` call).
- `config/config_abci.yaml`: ABCI-oriented settings (`objective.sh` wrapper call).
- `config/job_config_abci.yaml`: `aiaccel-job pbs` configuration template for ABCI.
- `scripts/`: runtime wrappers called by Make.
- `objective.sh`: ABCI-ready objective wrapper script (module/venv aware).
- `objectives/`: objective and benchmark helper scripts.
- `workspace/`: generated artifacts (configs, Optuna DBs, pairs, models, sentinels).

## Local Run (Default)
```bash
cd examples/hpo/modelbridge
make all
```

Stage aliases:
- `make prepare`
- `make hpo-train`
- `make hpo-test`
- `make collect`
- `make fit`
- `make evaluate`

Reset:
```bash
make clean
```

Local run with explicit config file:
```bash
make clean
make all CONFIG_FILE=config/config.yaml
```

## ABCI Run (Using `objective.sh`)
1. Edit `config/job_config_abci.yaml`.
- Set `path_to_env` to your venv activation script.
- Set `path_to_venv` to your venv root directory.
- Set `job_group` to your ABCI group.
- Adjust module names if needed.

2. Prepare configs with ABCI objective wrapper:
```bash
cd examples/hpo/modelbridge
make prepare CONFIG_FILE=config/config_abci.yaml
```

3. Run the full pipeline in one PBS job:
```bash
aiaccel-job pbs --config config/job_config_abci.yaml cpu --walltime 1:00:00 \
  workspace/logs/modelbridge_abci.log -- \
  make all CONFIG_FILE=config/config_abci.yaml
```

Alternative: submit only train/test HPO stages separately:
```bash
aiaccel-job pbs --config config/job_config_abci.yaml cpu --walltime 1:00:00 \
  workspace/logs/hpo_train.log -- \
  bash scripts/run_hpo.sh train

aiaccel-job pbs --config config/job_config_abci.yaml cpu --walltime 1:00:00 \
  workspace/logs/hpo_test.log -- \
  bash scripts/run_hpo.sh test
```

## Using Files in `objectives/`
Commands below assume current directory is `examples/hpo/modelbridge`.

- `simple_objective.py`:
  - Used by `config/config.yaml` (local direct call).
  - Can also be used via `objective.sh` by setting:
    - `MODELBRIDGE_OBJECTIVE_SCRIPT=objectives/simple_objective.py`
- `multi_objective.py`:
  - Supports `--function` and `--function_id`.
  - For fixed-function optimization, include `function_id` as a parameter with identical `low`/`high` bounds.
- `simple_benchmark.py`:
  - Runs a small end-to-end benchmark.
  - Example:
    - `python objectives/simple_benchmark.py --n-train 2 --n-test 1 --trials 6`
- `multi_function_benchmark.py`:
  - Runs multiple function-pair scenarios with tools.
  - Example:
    - `python objectives/multi_function_benchmark.py --scenario all --n-train 2 --n-test 1 --trials 8`

Run benchmark scripts on ABCI with `aiaccel-job pbs`:
```bash
aiaccel-job pbs --config config/job_config_abci.yaml cpu --walltime 1:00:00 \
  workspace/logs/simple_benchmark.log -- \
  python objectives/simple_benchmark.py --workspace workspace/benchmark_simple
```

## Expected Outputs
- `workspace/runs/{train,test}/{macro,micro}/<run_id>/config.yaml`
- `workspace/runs/{train,test}/{macro,micro}/<run_id>/optuna.db`
- `workspace/pairs/train_pairs.csv`
- `workspace/pairs/test_pairs.csv`
- `workspace/pairs/test_predictions.csv`
- `workspace/models/regression_model.pkl`
- `workspace/models/model_meta.json`
- `workspace/models/summary.json`
- `workspace/state/*.done`
