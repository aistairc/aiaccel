# Data Assimilation Example (MAS-Bench)

This example demonstrates the data assimilation + model-bridge workflow inspired by `RapidDataAssimilation.py`.

## Requirements
- `python -m pip install "[modelbridge]"` (scikit-learn, optuna included).
- MAS-Bench assets (if running real simulations):
  - `MAS-Bench.jar` (build from https://github.com/MAS-Bench/MAS-Bench or reuse `../modelbridge-cc/example/mas_bench/MAS-Bench.jar`).
  - `masbench-resources/Dataset/<model>/agent_size.sh` and related files (set `dataset_root` and `mas_bench_jar` in `mas_bench_config.yaml`).
- For quick local/CI use, `allow_mock: true` in `mas_bench_config.yaml` runs a synthetic path without JAR execution.

## Quick Start (mock execution)
```bash
make -f examples/hpo/modelbridge/data_assimilation/Makefile.template run \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/data_assimilation/data_assimilation.yaml \
  MODELBRIDGE_JSON_LOG=1
```
Outputs land in `./work/modelbridge/data_assimilation/` (`data_assimilation_summary.json`, `data_assimilation_manifest.json`, regression details, per-phase inputs/outputs).

## Resuming / Skipping Steps
You can skip optimization steps by placing existing `optuna.db` files in the expected directories.
The wrapper checks for existing studies (using `load_if_exists=True` behavior of underlying HPO call).

1. **Micro Optimization**:
   - Path: `work/modelbridge/data_assimilation/micro/<study_name>/optuna.db`
   - Study Name: `{micro_model}-micro-{trials}-{seed}` (e.g. `FL1-1-micro-2-0`)
2. **Macro Train**:
   - Path: `work/modelbridge/data_assimilation/macro_train/<study_name>/optuna.db`
   - Study Name: `{macro_model}-train-{scenario_idx}-{sampler}-{trials}-{seed}` (e.g. `FS1-1-train-0-cmaes-2-0`)
3. **Macro Test**:
   - Path: `work/modelbridge/data_assimilation/macro_test/<study_name>/optuna.db`
   - Study Name: `{macro_model}-test-{sampler}-{trials}-{seed}`

**Constraint**: The Study Name stored inside the DB file MUST match the folder name/expected convention.

## Real MAS-Bench run
- Set in `examples/hpo/modelbridge/data_assimilation/mas_bench_config.yaml`:
  ```yaml
  mas_bench_jar: /path/to/MAS-Bench.jar
  dataset_root: /path/to/masbench-resources/Dataset
  micro_model: FL1-1   # adjust to match your Dataset directory names
  macro_model: FS1-1
  allow_mock: false
  ```
- Ensure the dataset contains `agent_size.sh` for `micro_model`/`macro_model`. Execution will fail early if the JAR or dataset is missing.

## What the pipeline does
1. The pipeline invokes `mas_bench_wrapper.py` as an external hook configured in `data_assimilation.yaml`.
2. The wrapper performs:
   - Generate micro scenarios (Optuna) and record input/output CSV (mock or MAS-Bench Fitness).
   - Assimilate macro train scenarios (CMA-ES by default).
   - Assimilate macro test scenario.
   - Fit polynomial regression (degree configurable) from macro→micro best params.
   - Predict micro params for the test macro and re-run MAS-Bench once to measure error.

## Tuning
- `mas_bench_config.yaml`:
  - `trials.*`: number of trials per stage.
  - `samplers.*`: `random`, `tpe`, `cmaes`.
  - `scaling`: parameter scaling factors.
  - `agent_sizes`: override agent counts instead of reading `agent_size.sh`.