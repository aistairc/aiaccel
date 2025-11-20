# Modelbridge Examples

## Quick Start
1. Copy `examples/hpo/modelbridge/Makefile.template` next to your configuration or call it directly with `make -f`.
2. Execute the staged pipeline (ここでは run range で 0〜1 を指定):
   ```bash
   make -f examples/hpo/modelbridge/Makefile.template \
     pipeline \
     MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
     MODELBRIDGE_OUTPUT=./work/modelbridge/simple \
     MODELBRIDGE_SCENARIO=simple \
     TRAIN_RUN_RANGE=0..1 \
     EVAL_RUN_RANGE=0..1
   ```
3. Inspect artifacts under `work/modelbridge/simple` (Optuna DBs in `runs/*/*/<run>/optuna.db`, regression model in `regression_train.json`, summaries in `summary.json`).

## Makefile Template Highlights
- Targets `train-macro`, `train-micro`, `eval-macro`, and `eval-micro` each materialize the Optuna databases for every run listed in `TRAIN_RUN_IDS` / `EVAL_RUN_IDS` by calling `aiaccel-hpo modelbridge --phase hpo --role <role> --target <target> --run-id <run>`.
- Artifacts are addressed directly (e.g. `work/modelbridge/simple/runs/<scenario>/train/macro/000/optuna.db`), so rerunning `make train-macro` only re-executes the missing run directories.
- `regress`, `evaluate`, and `summary` wrap the corresponding phases, while `pipeline` chains all phases for a full run.
- Tunable variables:
  - `MODELBRIDGE_CONFIG`: path to the YAML configuration.
  - `MODELBRIDGE_OUTPUT`: destination for scenario results.
  - `MODELBRIDGE_SCENARIO`: scenario name under `bridge.scenarios`.
  - `TRAIN_RUN_RANGE` / `EVAL_RUN_RANGE`: `start..end[..step]` 形式で run ID を範囲指定（デフォルト 0..1）。
  - `TRAIN_RUN_IDS` / `EVAL_RUN_IDS`: 明示的に run ID を列挙したい場合に利用（`RUN_RANGE` より優先）。未指定かつ既存成果物がある場合は自動検出された run ID を使用。
  - `RUN_PAD_WIDTH`: run ID のゼロ埋め桁数（デフォルト 3）。
  - `JOB_TEMPLATE`: ジョブテンプレートのパス。デフォルトは `jobs/modelbridge/local.sh` を `bash` で実行。
  - `AIACCEL_CMD` / `JOB_OPTS`: ジョブ実行コマンド（デフォルト: `bash $(JOB_TEMPLATE)`）。PBS/SGE 用テンプレートは `jobs/modelbridge/pbs.sh` / `sge.sh` を指定。
  - `MODELBRIDGE_LOG_DIR`: location for per-phase logs.

## PYTHONPATH について
- Makefile.template は `PYTHONPATH` にリポジトリルートを自動で追加してから CLI を起動します（ローカルの `examples.*` を読み込むため）。
- `aiaccel-hpo modelbridge` を Makefile なしで直接叩く場合は、同様に `PYTHONPATH=$(pwd)`（またはリポジトリパス）を指定してから実行してください。

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
