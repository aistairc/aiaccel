# Modelbridge Examples

## Quick Start (Makefile pipeline)
- Template config: `examples/hpo/modelbridge/modelbridge.yaml`
- Command: `make modelbridge-run MODELBRIDGE_CONFIG=examples/hpo/modelbridge/modelbridge.yaml`
- Output: results under `./work/modelbridge/simple/` (`summary.json`, `scenarios/simple/*`)

## What this example does
Runs the modelbridge pipeline end-to-end using the in-repo objective `examples.hpo.modelbridge.simple_benchmark.objective`. Macro/micro Optuna phases each evaluate 12 trials, fit a linear regression from macro to micro parameters, and write the predictions, regression model, and aggregated summary into `work/modelbridge/simple`.
