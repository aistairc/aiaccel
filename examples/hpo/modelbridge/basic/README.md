# Basic Modelbridge Example

## Getting started

From the repository root, run the complete local pipeline with Pixi:

```bash
pixi run make -C examples/hpo/modelbridge/basic all
```

## Overview
Modelbridge is Makefile-first.

Orchestration lives in:
- `Makefile`
- direct calls to the Pixi-provided `aiaccel-modelbridge` CLI from Make recipes
- direct `aiaccel-hpo optimize` execution from Make recipes
- stage helpers resolved through the Pixi-provided `aiaccel-workflow template` CLI

Python under `aiaccel/modelbridge/apps/` contains the modelbridge CLI and its stateless step tools:
- `prepare.py`
- `collect.py`
- `fit_model.py`
- `evaluate.py`

## Directory Roles
- `config/config.yaml`: local run settings (direct `simple_objective.py` call).
- `config/config_abci.yaml`: ABCI-oriented settings (`objective.sh` wrapper call).
- `config/job_config_abci.yaml`: `aiaccel-job pbs` configuration template for ABCI.
- `objective.sh`: ABCI-ready objective wrapper script (module/virtualenv aware).
- `objectives/`: objective and benchmark helper scripts.
- `workspace/`: generated artifacts (configs, Optuna DBs, pairs, models, sentinels).

## Setup
Pixi resolves aiaccel and the modelbridge dependencies from the repository root:

```bash
pixi install
```

The commands below run through Pixi and use its console scripts:
- `aiaccel-modelbridge` for modelbridge steps
- `aiaccel-hpo` for HPO optimization
- `aiaccel-job` for local or PBS job wrapping
- `aiaccel-workflow` for Make stage templates
- `aiaccel-config` for inspecting YAML values, for example:
  ```bash
  pixi run aiaccel-config get-value examples/hpo/modelbridge/basic/config/config.yaml n_train
  ```

## Local Run (Default)
```bash
pixi run make -C examples/hpo/modelbridge/basic all
```

Stage aliases:
- `pixi run make -C examples/hpo/modelbridge/basic prepare`
- `pixi run make -C examples/hpo/modelbridge/basic hpo-train`
- `pixi run make -C examples/hpo/modelbridge/basic hpo-test`
- `pixi run make -C examples/hpo/modelbridge/basic collect`
- `pixi run make -C examples/hpo/modelbridge/basic fit`
- `pixi run make -C examples/hpo/modelbridge/basic evaluate`

Reset:
```bash
pixi run make -C examples/hpo/modelbridge/basic clean
```

Local run with explicit config file:
```bash
pixi run make -C examples/hpo/modelbridge/basic clean
pixi run make -C examples/hpo/modelbridge/basic all CONFIG_FILE=config/config.yaml
```

## ABCI Run (Using `objective.sh`)
1. Edit `config/job_config_abci.yaml`.
- Set `job_group` to your ABCI group.
- Set `modelbridge_venv` to the Pixi environment that contains aiaccel, or leave it empty if the selected modules
  already expose the correct `python` and `aiaccel-*` commands on `PATH`.
- Adjust module names and `modelbridge_python` if needed.

2. Prepare configs with ABCI objective wrapper:
```bash
pixi run make -C examples/hpo/modelbridge/basic prepare CONFIG_FILE=config/config_abci.yaml
```

3. Dispatch each pipeline command through PBS from Make:
```bash
pixi run make -C examples/hpo/modelbridge/basic all CONFIG_FILE=config/config_abci.yaml \
  cmd="aiaccel-job pbs --config config/job_config_abci.yaml" job_ops="--walltime 1:00:00"
```

Alternative: submit only train/test HPO stages separately:
```bash
pixi run make -C examples/hpo/modelbridge/basic hpo-train CONFIG_FILE=config/config_abci.yaml \
  cmd="aiaccel-job pbs --config config/job_config_abci.yaml" job_ops="--walltime 1:00:00"

pixi run make -C examples/hpo/modelbridge/basic hpo-test CONFIG_FILE=config/config_abci.yaml \
  cmd="aiaccel-job pbs --config config/job_config_abci.yaml" job_ops="--walltime 1:00:00"
```

## Using Files in `objectives/`
Commands below are run from the repository root through Pixi.

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
    - `pixi run python examples/hpo/modelbridge/basic/objectives/simple_benchmark.py --n-train 2 --n-test 1 --trials 6`
- `multi_function_benchmark.py`:
  - Runs multiple function-pair scenarios with tools.
  - Example:
    - `pixi run python examples/hpo/modelbridge/basic/objectives/multi_function_benchmark.py --scenario all --n-train 2 --n-test 1 --trials 8`

Run benchmark scripts through Pixi. For ABCI, dispatch a Make target with `cmd` as shown above:
```bash
pixi run python examples/hpo/modelbridge/basic/objectives/simple_benchmark.py \
  --workspace examples/hpo/modelbridge/basic/workspace/benchmark_simple
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
