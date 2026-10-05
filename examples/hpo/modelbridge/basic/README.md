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

## Re-running in an Existing Workspace
Make re-runs `prepare` whenever the config file is newer than `workspace/state/01_prepare.done`.
The HPO and collect stages process every run under `workspace/runs/<phase>`, and the generated
studies use `load_if_exists: true`, so an existing workspace keeps and resumes its Optuna DBs.

- Increasing `n_train`/`n_test` or the number of trials resumes the existing studies.
- Decreasing `n_train`/`n_test` is rejected by `prepare`, because the old runs would otherwise be
  optimized and collected again. Run `make clean` or pass a new `WORKSPACE_DIR`.
- Changing the search space or objective is not detected; run `make clean` or use a new `WORKSPACE_DIR`
  so old trials are not mixed into the new studies.

## Data Contracts
- `pairs/<phase>_pairs.csv` has a `run_id` column followed by sorted `macro_<param>` and `micro_<param>`
  columns. Only runs with a completed best trial for both macro and micro are written.
- When no run qualifies, `collect` rewrites the CSV as an empty file. `fit-model` and `evaluate` then
  skip with a warning and leave the existing model and summary files untouched, so check the logs.
- `models/model_meta.json` stores the feature and target column order used by `evaluate`.
- `models/summary.json` is strict JSON. `metrics.r2` is `null` when there are fewer than two test samples,
  because R2 is undefined for a single sample.
- `models/regression_model.pkl` is a pickle; load only models produced in your own workspace.

## Regression Settings
`fit-model` reads the `regression` section from the first existing config in this order:
`--config`, `MODELBRIDGE_CONFIG_FILE`, `CONFIG_FILE`, `<workspace>/config.yaml`, `<workspace>/../config/config.yaml`.
The Makefile exports `MODELBRIDGE_CONFIG_FILE`, so `CONFIG_FILE=...` also selects the regression settings.

```yaml
regression:
  kind: linear      # linear (default), polynomial, or gpr
  degree: 2         # polynomial degree (integer >= 1)
  noise: 1.0e-4     # gpr only: fixed noise variance (> 0)
```

`gpr` fits one GPy `GPRegression` per micro target with an RBF kernel. The kernel hyperparameters are
GPy defaults and are not optimized, so scale the parameters to a comparable range or prefer `linear`/`polynomial`.

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
