# Data Assimilation Example (MAS-Bench)

## Getting started

From the repository root, run the mock workflow with Pixi:

```bash
pixi run make -C examples/hpo/modelbridge/data_assimilation all
```

The bundled `mas_bench_config.yaml` sets `allow_mock: true`, so this command needs no MAS-Bench assets.
This example provides an external data-assimilation workflow for current modelbridge environments.
It does not use removed legacy commands such as `aiaccel-modelbridge run`.

## Requirements
- Run `pixi install` from the repository root.
- MAS-Bench assets (only for real simulation, `allow_mock: false`):
  - `MAS-Bench.jar` (https://github.com/MAS-Bench/MAS-Bench)
  - `masbench-resources/Dataset/<model>/agent_size.sh`

## Scenarios
A scenario is one observation condition. For every train scenario, the wrapper optimizes the micro model
and the macro model under that condition, and pairs the two best-parameter sets by scenario id to form one
regression sample. The test scenario is used for the macro test study and the bridged micro run.

```yaml
train_scenarios:
  - {id: s0, mock_target: 0.1}
  - {id: s1, mock_target: 0.3, micro_model: FL1-1, macro_model: FS1-1}
test_scenario: {id: t0, mock_target: 0.4}
```

- `micro_model` / `macro_model` default to the top-level keys. In real runs, the observation data comes from
  the MAS-Bench dataset of these models, so train scenarios must differ in their models.
- `mock_target` is the condition of mock runs: the mock error is the squared distance of each raw parameter
  from `mock_target`. Train scenarios must differ in `mock_target` in mock runs.
- Duplicated scenario ids or conditions are rejected, because identical scenarios would repeat the same
  optimization and add no information to the regression.
- All scenarios share `agent_sizes` (or the `agent_size.sh` of the top-level `micro_model`), so the
  parameter names match across scenarios.

The sampler seed of each study is `seeds.<stage> + scenario index`.

## Quick Start (mock execution)
```bash
pixi run make -C examples/hpo/modelbridge/data_assimilation all
```

Main outputs are created under `work/modelbridge/data_assimilation/`:
- `data_assimilation_summary.json`: best parameters per scenario id, regression, and bridged error
- `data_assimilation_regression.json`: train scenarios used, train MAE/R2, and predicted micro parameters
- `micro/`, `macro_train/`, `macro_test/` (Optuna studies and per-trial files)
- `state/01_data_assimilation.done`

The mock objective only demonstrates the data flow; its numbers do not measure data-assimilation quality.

Optional overrides:
```bash
pixi run make -C examples/hpo/modelbridge/data_assimilation all \
  CONFIG_FILE=/path/to/mas_bench_config.yaml OUTPUT_ROOT=/path/to/output
```

To dispatch the workflow through PBS, change the Make dispatcher instead of wrapping Make:

```bash
pixi run make -C examples/hpo/modelbridge/data_assimilation all \
  cmd="aiaccel-job pbs --config /path/to/job_config.yaml" job_ops="--walltime 1:00:00"
```

## Resume with Existing Optuna DB
The wrapper uses `load_if_exists=True`. If matching studies already exist, execution resumes from them.
Run `make clean` (or choose a new `OUTPUT_ROOT`) after changing the objective or scenario conditions.

Study name pattern: `{model}-{stage}-{scenario_id}-{sampler}-{trials}-{seed}`, where `stage` is
`micro`, `macro_train`, or `macro_test`.

Place DBs at:
- `work/modelbridge/data_assimilation/micro/<study_name>/optuna.db`
- `work/modelbridge/data_assimilation/macro_train/<study_name>/optuna.db`
- `work/modelbridge/data_assimilation/macro_test/<study_name>/optuna.db`

## Real MAS-Bench Run
Use a copy of `mas_bench_config.yaml` with `allow_mock: false` and configure:
```yaml
mas_bench_jar: /path/to/MAS-Bench.jar
dataset_root: /path/to/masbench-resources/Dataset
micro_model: FL1-1
macro_model: FS1-1
train_scenarios:
  - {id: s0}  # uses the top-level models
  - {id: s1, micro_model: <micro model B>, macro_model: <macro model B>}
test_scenario: {id: t0, micro_model: <micro model C>, macro_model: <macro model C>}
```

Replace the placeholders with MAS-Bench models whose datasets you have, and ensure the dataset/model
directories and `agent_size.sh` are present for every selected model.
