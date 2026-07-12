# Modelbridge Examples

## Getting started

From the repository root, run the basic pipeline with Pixi:

```bash
pixi run make -C examples/hpo/modelbridge/basic all
```

This directory contains two modelbridge examples:

- `basic/`: the standard Makefile-first modelbridge pipeline.
- `data_assimilation/`: the MAS-Bench data-assimilation workflow.

Pixi provides the console scripts on `PATH`, including `aiaccel-job`, `aiaccel-config`,
`aiaccel-hpo`, and `aiaccel-workflow`.

See `data_assimilation/README.md` before running the MAS-Bench workflow because real execution requires MAS-Bench assets.
