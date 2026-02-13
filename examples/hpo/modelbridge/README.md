# Modelbridge Examples

## Overview
This directory contains rev04-compatible examples for `aiaccel-hpo modelbridge`.
The canonical lifecycle is:
- `prepare` (generate plans + per-run optimize config)
- external execution (`aiaccel-hpo optimize`)
- `collect -> fit-regression -> evaluate-model -> publish-summary`

## Files
- `Makefile.template`: External execution workflow (`prepare -> emit -> optimize -> analyze`).
- `modelbridge.yaml`: Modelbridge scenario settings.
- `optimize_config.yaml`: Base optimize config (`_base_` from aiaccel default).
- `objectives/simple_objective.py`: Objective used by train/eval optimize loops.
- `objectives/simple_benchmark.py`: Small end-to-end script using modelbridge API.
- `data_assimilation/`: External tool example (not core modelbridge steps).

## 1. Using Makefile
```bash
make -f examples/hpo/modelbridge/Makefile.template \
  run \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
  MODELBRIDGE_OUTPUT=./work/modelbridge/simple
```

Targets:
- `prepare`: Generate run configs and `workspace/{train,eval}_plan.json`.
- `emit_train`: Emit `workspace/commands/train.sh`.
- `hpo_train`: Execute `train.sh` (calls `aiaccel-hpo optimize`).
- `emit_eval`: Emit `workspace/commands/eval.sh`.
- `hpo_eval`: Execute `eval.sh`.
- `analyze`: `collect -> fit -> evaluate -> publish`.

## 2. Using CLI Directly
Common options:
- `--config <path>` required.
- `--output_dir <path>` optional override.
- `--set key=value` repeatable overrides.
- Examples below use `python -m aiaccel.hpo.apps.modelbridge` so the local repository code is used.
- Run commands from the repository root because example objectives are referenced by relative path in YAML.

### 2.1 Commands used by `make ... run`
```bash
CONFIG=examples/hpo/modelbridge/modelbridge.yaml
OUT=./work/modelbridge/simple

# 1) prepare
python -m aiaccel.hpo.apps.modelbridge run --config "$CONFIG" --output_dir "$OUT" --profile prepare

# 2) emit train commands
python -m aiaccel.hpo.apps.modelbridge emit-commands --config "$CONFIG" --output_dir "$OUT" --role train --format shell

# 3) external train execution
bash "$OUT/workspace/commands/train.sh"

# 4) emit eval commands
python -m aiaccel.hpo.apps.modelbridge emit-commands --config "$CONFIG" --output_dir "$OUT" --role eval --format shell

# 5) external eval execution
bash "$OUT/workspace/commands/eval.sh"

# 6) analyze
python -m aiaccel.hpo.apps.modelbridge run --config "$CONFIG" --output_dir "$OUT" --profile analyze
```

Alternative: emit commands during prepare profile:
```bash
python -m aiaccel.hpo.apps.modelbridge run --config "$CONFIG" --output_dir "$OUT" --profile prepare \
  --prepare-emit-commands --prepare-execution-target local
```

### 2.2 Step-by-step single command execution
```bash
CONFIG=examples/hpo/modelbridge/modelbridge.yaml
OUT=./work/modelbridge/simple

python -m aiaccel.hpo.apps.modelbridge prepare-train --config "$CONFIG" --output_dir "$OUT"
python -m aiaccel.hpo.apps.modelbridge prepare-eval --config "$CONFIG" --output_dir "$OUT"

python -m aiaccel.hpo.apps.modelbridge collect-train --config "$CONFIG" --output_dir "$OUT"
python -m aiaccel.hpo.apps.modelbridge collect-eval --config "$CONFIG" --output_dir "$OUT"

python -m aiaccel.hpo.apps.modelbridge fit-regression --config "$CONFIG" --output_dir "$OUT"
python -m aiaccel.hpo.apps.modelbridge evaluate-model --config "$CONFIG" --output_dir "$OUT"
python -m aiaccel.hpo.apps.modelbridge publish-summary --config "$CONFIG" --output_dir "$OUT"
```

Optional explicit DB input for collect:
```bash
python -m aiaccel.hpo.apps.modelbridge collect-train --config "$CONFIG" --output_dir "$OUT" \
  --train-db-path /path/to/train_macro.db --train-db-path /path/to/train_micro.db

python -m aiaccel.hpo.apps.modelbridge collect-eval --config "$CONFIG" --output_dir "$OUT" \
  --eval-db-path /path/to/eval_macro.db --eval-db-path /path/to/eval_micro.db
```
Collect input priority is:
1. explicit DB pairs (API)
2. explicit DB paths (CLI/API)
3. role plan (`workspace/{train,eval}_plan.json`)
4. layout scan fallback

## 3. Outputs
Generated under `<output_dir>`:
- `workspace/train_plan.json`, `workspace/eval_plan.json`
- `workspace/commands/{train,eval}.{sh,json}`
- `workspace/state/*.json` (one per step)
- `<scenario>/train_pairs.csv`, `<scenario>/test_pairs.csv`, `<scenario>/test_predictions.csv`
- `<scenario>/models/regression_model.json`
- `<scenario>/metrics/train_metrics.json`, `<scenario>/metrics/eval_metrics.json`
- `summary.json`, `manifest.json`

## 4. Validation and Schema
```bash
python -m aiaccel.hpo.apps.modelbridge validate --config examples/hpo/modelbridge/modelbridge.yaml
python -m aiaccel.hpo.apps.modelbridge schema
```

## 5. Benchmark Script
```bash
python3 examples/hpo/modelbridge/objectives/simple_benchmark.py
```

## 6. Optional GPR
Install GPy and switch regression kind in YAML:
```yaml
regression:
  kind: gpr
  kernel: RBF
  noise: 1.0e-5
```
