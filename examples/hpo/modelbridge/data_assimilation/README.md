# Data Assimilation Example (MAS-Bench)

This example demonstrates the data assimilation + model-bridge workflow inspired by `RapidDataAssimilation.py`, using the new `aiaccel-hpo modelbridge data-assimilation` subcommand.

## Requirements
- `python -m pip install ".[modelbridge]"` (scikit-learn, optuna included).
- MAS-Bench assets (if running real simulations):
  - `MAS-Bench.jar` (build from https://github.com/MAS-Bench/MAS-Bench or reuse `../modelbridge-cc/example/mas_bench/MAS-Bench.jar`).
  - `masbench-resources/Dataset/<model>/agent_size.sh` and related files (set `dataset_root` and `mas_bench_jar` in the YAML).
- For quick local/CI use, `allow_mock: true` in `data_assimilation.yaml` runs a synthetic path without JAR execution.

## Quick Start (mock execution)
```bash
make -f examples/hpo/modelbridge/data_assimilation/Makefile.template plan \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/data_assimilation/data_assimilation.yaml

make -f examples/hpo/modelbridge/data_assimilation/Makefile.template data-assimilation \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/data_assimilation/data_assimilation.yaml \
  MODELBRIDGE_JSON_LOG=1
```
Outputs land in `./work/modelbridge/data_assimilation/` (`data_assimilation_summary.json`, `data_assimilation_manifest.json`, regression details, per-phase inputs/outputs).

## Real MAS-Bench run
- Set in `data_assimilation.yaml`:
  ```yaml
  data_assimilation:
    mas_bench_jar: /path/to/MAS-Bench.jar
    dataset_root: /path/to/masbench-resources/Dataset
    micro_model: FL1-1   # adjust to match your Dataset directory names
    macro_model: FS1-1
    allow_mock: false
  ```
- Ensure the dataset contains `agent_size.sh` for `micro_model`/`macro_model`. Execution will fail early if the JAR or dataset is missing.

## What the pipeline does
1. Generate micro scenarios (Optuna) and record input/output CSV (mock or MAS-Bench Fitness).
2. Assimilate macro train scenarios (CMA-ES by default).
3. Assimilate macro test scenario.
4. Fit polynomial regression (degree configurable) from macro→micro best params.
5. Predict micro params for the test macro and re-run MAS-Bench once to measure error.

## Tuning
- `data_assimilation.trials.*`: number of trials per stage.
- `data_assimilation.samplers.*`: `random`, `tpe`, `cmaes`.
- `data_assimilation.regression_degree`: polynomial degree.
- `data_assimilation.agent_sizes`: override agent counts instead of reading `agent_size.sh`.
- `MODELBRIDGE_SET`: additional CLI overrides (e.g., `MODELBRIDGE_SET="data_assimilation.allow_mock=false data_assimilation.mas_bench_jar=/path/to/MAS-Bench.jar"`).
