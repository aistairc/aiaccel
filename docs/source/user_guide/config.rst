OmegaConf Utilities
===================

Typical usage is as follows:

.. code-block:: python

    from argparse import ArgumentParser

    from omegaconf import OmegaConf as oc  # noqa: N813
    from aiaccel.config import (
        load_config,
        overwrite_omegaconf_dumper,
        pathlib2str_config,
        print_config,
        resolve_inherit,
    )

    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = oc.merge(
        load_config(
            args.config,
            {
                "config_path": args.config,
                "working_directory": str(Path(args.config).parent.resolve()),
            },
        ),
        oc.from_cli(unk_args),
    )

    print_config(config)

    config = resolve_inherit(config)
