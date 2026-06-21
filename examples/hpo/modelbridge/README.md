# Modelbridge Examples

This directory contains two modelbridge examples:

- `basic/`: the standard Makefile-first modelbridge pipeline.
- `data_assimilation/`: the MAS-Bench data-assimilation workflow.

Install aiaccel with pip before running examples. From the repository root, the basic example can be started with:

```bash
python -m pip install -e ".[dev,github-actions,modelbridge]"
make -C examples/hpo/modelbridge/basic all
```

The examples use the installed console scripts on `PATH`, including `aiaccel-job`, `aiaccel-config`,
`aiaccel-hpo`, and `aiaccel-workflow`.

See `data_assimilation/README.md` before running the MAS-Bench workflow because real execution requires MAS-Bench assets.
