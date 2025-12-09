# Modelbridge Examples

## Quick Start
1. Copy `examples/hpo/modelbridge/Makefile.template` next to your configuration or call it directly with `make -f`.
2. Plan the run if needed (uses CLI `--set` overrides to place outputs under `MODELBRIDGE_OUTPUT` and filter `MODELBRIDGE_SCENARIO`):
   ```bash
   make -f examples/hpo/modelbridge/Makefile.template plan \
     MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
     MODELBRIDGE_OUTPUT=./work/modelbridge/simple \
     MODELBRIDGE_SCENARIO=simple
   ```
3. Execute the staged pipeline (run range set to 0–1 here):
   ```bash
   make -f examples/hpo/modelbridge/Makefile.template \
     pipeline \
     MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
     MODELBRIDGE_OUTPUT=./work/modelbridge/simple \
     MODELBRIDGE_SCENARIO=simple \
     TRAIN_RUN_RANGE=0..1 \
     EVAL_RUN_RANGE=0..1
   ```
4. Inspect artifacts under `work/modelbridge/simple` (Optuna DBs in `runs/*/*/<run>/optuna.db`, regression model in `regression_train.json`, summaries in `summary.json`).
   - `manifest.json` now lists generated artifacts with sizes and sha256 hashes.
   - Set `MODELBRIDGE_JSON_LOG=1` to emit JSON-formatted pipeline logs.

## Makefile Template Highlights
- Targets `train-macro`, `train-micro`, `eval-macro`, and `eval-micro` each materialize the Optuna databases for every run listed in `TRAIN_RUN_IDS` / `EVAL_RUN_IDS` by calling `aiaccel-hpo modelbridge run --phase hpo --role <role> --target <target> --run-id <run>`.
- Artifacts are addressed directly (e.g. `work/modelbridge/simple/runs/<scenario>/train/macro/000/optuna.db`), so rerunning `make train-macro` only re-executes the missing run directories.
- `regress`, `evaluate`, and `summary` wrap the corresponding phases, while `pipeline` chains all phases for a full run. A `plan` target prints the PhaseContext list without executing.
- Tunable variables:
  - `MODELBRIDGE_CONFIG`: path to the YAML configuration (uses strict train/eval objectives/params).
  - `MODELBRIDGE_OUTPUT`: destination for scenario results (passed via CLI `--set bridge.output_dir=...`).
  - `MODELBRIDGE_SCENARIO`: scenario name under `bridge.scenarios` (passed to CLI `--scenario`).
  - `MODELBRIDGE_SET`: additional CLI overrides (space-separated `--set key=value` entries, e.g. `MODELBRIDGE_SET="bridge.log_level=DEBUG bridge.seed=7"`).
  - `MODELBRIDGE_JSON_LOG`: set to `1`/`true` to enable `--json-log` structured logs.
  - `TRAIN_RUN_RANGE` / `EVAL_RUN_RANGE`: range for run IDs in `start..end[..step]` form (default 0..1).
  - `TRAIN_RUN_IDS` / `EVAL_RUN_IDS`: enumerate run IDs explicitly (preferred over `RUN_RANGE`). If unset and outputs exist, detected run IDs are reused.
  - `RUN_PAD_WIDTH`: zero-padding width for run IDs (default 3).
  - `JOB_TEMPLATE`: job template path. Defaults to `jobs/modelbridge/local.sh` executed via `bash`.
  - `AIACCEL_CMD` / `JOB_OPTS`: job execution command (default: `bash $(JOB_TEMPLATE)`). For PBS/SGE use `jobs/modelbridge/pbs.sh` / `sge.sh`.
  - `MODELBRIDGE_LOG_DIR`: location for per-phase logs.

## PYTHONPATH
- The Makefile.template prepends the repository root to `PYTHONPATH` before invoking the CLI so local `examples.*` can be imported.
- When calling `aiaccel-hpo modelbridge` directly, set `PYTHONPATH=$(pwd)` (or your repo path) likewise.

## Optional Gaussian Process Regression (GPy)
- Install the optional dependency with `python -m pip install .[gpy]` (or `pip install aiaccel[gpy]`).
- Update the scenario’s `regression` block to select the backend:
  ```yaml
  regression:
    kind: gpr
    gpr:
      kernel: RBF      # or MATERN32 / MATERN52
      noise: 1.0e-5    # alias: alpha
  ```
- The CLI raises a friendly error when `kind: gpr` is requested without the dependency.

## Example Objective
The default configuration uses `examples.hpo.modelbridge.simple_benchmark.objective` to generate synthetic macro/micro search spaces, executes 12 Optuna trials per run, trains the regression model, and summarizes the accuracy metrics inside `work/modelbridge/simple`.

## Objective Options
- Python callable: set `train_objective.target` / `eval_objective.target` to an import path (e.g., `examples.hpo.modelbridge.simple_benchmark.objective`).
- Command adapter: set `target` to `aiaccel.hpo.modelbridge.evaluators.command_objective` and supply `command: ["python", "path/to/script.py"]`; the command must emit JSON with `objective`, optional `metrics`, and `payload`.
- HTTP adapter: set `target` to an `http(s)://...` endpoint; the runner POSTs `{scenario, phase, trial_index, params, env}` where `env` includes `AIACCEL_PARAM_*`. The endpoint should respond with JSON containing `objective`, optional `metrics`, and `payload`.

## Data Assimilation Example
- A MAS-Bench-inspired data assimilation workflow lives under `examples/hpo/modelbridge/data_assimilation/` with its own `Makefile.template` and `data_assimilation.yaml`.
- Run with `aiaccel-hpo modelbridge data-assimilation --config <path>` (Makefile wraps this). `allow_mock: true` in the sample YAML avoids needing MAS-Bench.jar; set `mas_bench_jar`/`dataset_root` for real runs.

## Schema & Validation
- Print the JSON Schema for the Pydantic v2 configuration:
  ```bash
  aiaccel-hpo modelbridge schema
  ```
- Validate + view the resolved configuration after CLI overrides:
  ```bash
  aiaccel-hpo modelbridge validate --config ./modelbridge.yaml --set bridge.output_dir=./work/modelbridge/simple --print-config
  ```

## Outputs & Logs
- `manifest.json` and per-scenario `checkpoint.json` include sha256 hashes for produced artifacts.
- Use `--json-log` (or `MODELBRIDGE_JSON_LOG=1` in the Makefile) for structured logging; disable console output with `AIACCEL_LOG_SILENT=1` if running in batch environments.
