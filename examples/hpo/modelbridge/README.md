# Modelbridge Examples

## Overview
This directory contains v16-compatible examples for `aiaccel-hpo modelbridge`.
The workflow is Makefile-first and stage-based:
1. `stage0`: `prepare-train`, `prepare-eval`
2. `stage1`: `hpo-train`
3. `stage2`: `hpo-eval`
4. `stage3`: `collect-train`, `collect-eval`
5. `stage4`: `fit-regression`
6. `stage5`: `evaluate-model`
7. `stage6`: `publish-summary`

## Files
- `Makefile.template`: Stage pipeline with `.done` sentinels.
- `modelbridge.yaml`: Modelbridge scenario settings.
- `optimize_config.yaml`: Base optimize config (`_base_` from aiaccel default).
- `objectives/simple_objective.py`: Objective used by train/eval optimize loops.
- `objectives/simple_benchmark.py`: Small end-to-end script using step APIs.
- `data_assimilation/`: External tool example (not core modelbridge steps).

## 1. Using Makefile
```bash
make -f examples/hpo/modelbridge/Makefile.template \
  run \
  MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml \
  MODELBRIDGE_OUTPUT=./work/modelbridge/simple
```

Useful targets:
- `stage0` ... `stage6`: Execute one stage.
- `skip-stageN`: Mark one stage done without execution.
- `validate`: Validate config.
- `schema`: Print config schema.
- `clean`: Remove output and `.done` files.

## 2. Using CLI Directly
Common options:
- `--config <path>` required.
- `--output_dir <path>` optional override.
- `--set key=value` repeatable overrides.
- Run from repository root because objective paths in YAML are relative.

Step commands:
```bash
CONFIG=examples/hpo/modelbridge/modelbridge.yaml
OUT=./work/modelbridge/simple

aiaccel-hpo modelbridge prepare-train --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge prepare-eval --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge hpo-train --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge hpo-eval --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge collect-train --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge collect-eval --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge fit-regression --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge evaluate-model --config "$CONFIG" --output_dir "$OUT"
aiaccel-hpo modelbridge publish-summary --config "$CONFIG" --output_dir "$OUT"
```

Optional explicit DB input for collect:
```bash
aiaccel-hpo modelbridge collect-train --config "$CONFIG" --output_dir "$OUT" \
  --train-db-path /path/to/train_macro.db --train-db-path /path/to/train_micro.db

aiaccel-hpo modelbridge collect-eval --config "$CONFIG" --output_dir "$OUT" \
  --eval-db-path /path/to/eval_macro.db --eval-db-path /path/to/eval_micro.db
```

## 3. Outputs
Generated under `<output_dir>`:
- `workspace/train_plan.json`, `workspace/eval_plan.json`
- `workspace/commands/{train,eval}.{sh,json}`
- `workspace/state/*.json` (one per executed step)
- `<scenario>/runs/<role>/<run_id>/<target>/{config.yaml,run.sh,run.json}`
- `<scenario>/train_pairs.csv`, `<scenario>/test_pairs.csv`, `<scenario>/test_predictions.csv`
- `<scenario>/models/regression_model.json`
- `<scenario>/metrics/train_metrics.json`, `<scenario>/metrics/eval_metrics.json`
- `summary.json`, `manifest.json`

## 4. Validation and Schema
```bash
aiaccel-hpo modelbridge validate --config examples/hpo/modelbridge/modelbridge.yaml
aiaccel-hpo modelbridge schema
```

## 5. Benchmark Script
```bash
python3 examples/hpo/modelbridge/objectives/simple_benchmark.py
```

## 6. Optional GPR
Install `GPy` and switch regression kind in YAML:
```yaml
regression:
  kind: gpr
  kernel: RBF
  noise: 1.0e-5
```
