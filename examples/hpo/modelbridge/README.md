# Modelbridge Examples

This directory contains two modelbridge examples:

- `basic/`: the standard Makefile-first modelbridge pipeline.
- `data_assimilation/`: the MAS-Bench data-assimilation workflow.

Use the repository pixi environment before running examples. The basic example can be started with:

```bash
pixi install
pixi run make -C examples/hpo/modelbridge/basic all
```

See `data_assimilation/README.md` before running the MAS-Bench workflow because real execution requires MAS-Bench assets.
