# Modelbridge Examples

## Getting started

From the repository root, run the basic pipeline with Pixi:

```bash
pixi run make -C examples/hpo/modelbridge/basic all
```

## What modelbridge does

Modelbridge learns a mapping between two optimization problems that share a parameter space:
a cheap *macro* model and an expensive *micro* model.

1. For each run, `aiaccel-hpo optimize` finds the best macro parameters and the best micro parameters.
2. `collect` pairs the two best-parameter sets of the same run into `pairs/<phase>_pairs.csv`.
3. `fit-model` trains a regression from `macro_*` columns (features) to `micro_*` columns (targets).
4. `evaluate` predicts micro parameters for held-out test runs and reports MSE, MAE, and R2.

The fit and evaluate steps need scikit-learn, and `regression.kind: gpr` additionally needs GPy.
Both are part of the Pixi environment; outside Pixi, install them next to aiaccel yourself.

This directory contains two modelbridge examples:

- `basic/`: the standard Makefile-first modelbridge pipeline.
- `data_assimilation/`: the MAS-Bench data-assimilation workflow.

Pixi provides the console scripts on `PATH`, including `aiaccel-job`, `aiaccel-config`,
`aiaccel-hpo`, `aiaccel-modelbridge`, and `aiaccel-workflow`.

See `data_assimilation/README.md` before running the MAS-Bench workflow because real execution requires MAS-Bench assets.
