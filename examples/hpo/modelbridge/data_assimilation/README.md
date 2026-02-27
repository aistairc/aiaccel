# Data Assimilation Example (MAS-Bench)

This example provides an external data-assimilation workflow for current modelbridge environments.
It does not use removed legacy commands such as `aiaccel-hpo modelbridge run`.

## Requirements
- `python -m pip install ".[modelbridge]"`
- MAS-Bench assets (for real simulation):
  - `MAS-Bench.jar` (https://github.com/MAS-Bench/MAS-Bench)
  - `masbench-resources/Dataset/<model>/agent_size.sh`
- For quick local/CI execution, keep `allow_mock: true` in `mas_bench_config.yaml`.

## Quick Start (mock execution)
```bash
cd examples/hpo/modelbridge/data_assimilation
make all
```

Main outputs are created under `work/modelbridge/data_assimilation/`:
- `data_assimilation_summary.json`
- `data_assimilation_regression.json`
- `micro/`, `macro_train/`, `macro_test/` (Optuna studies and per-trial files)
- `state/01_data_assimilation.done`

Optional overrides:
```bash
make all CONFIG_FILE=/path/to/mas_bench_config.yaml OUTPUT_ROOT=/path/to/output
```

## Resume with Existing Optuna DB
The wrapper uses `load_if_exists=True`. If matching studies already exist, execution resumes from them.

Study name patterns:
- micro: `{micro_model}-micro-{scenario_idx}-{sampler}-{trials}-{seed}`
- macro train: `{macro_model}-train-{scenario_idx}-{sampler}-{trials}-{seed}`
- macro test: `{macro_model}-test-{sampler}-{trials}-{seed}`

Place DBs at:
- `work/modelbridge/data_assimilation/micro/<study_name>/optuna.db`
- `work/modelbridge/data_assimilation/macro_train/<study_name>/optuna.db`
- `work/modelbridge/data_assimilation/macro_test/<study_name>/optuna.db`

## Real MAS-Bench Run
Set `allow_mock: false` and configure:
```yaml
mas_bench_jar: /path/to/MAS-Bench.jar
dataset_root: /path/to/masbench-resources/Dataset
micro_model: FL1-1
macro_model: FS1-1
```

Ensure dataset/model directories and `agent_size.sh` are present for selected models.
