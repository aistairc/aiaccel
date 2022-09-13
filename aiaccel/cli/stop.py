from aiaccel.argument import Arguments
from aiaccel.config import Config
from aiaccel.workspace import Workspace
from aiaccel.storage.storage import Storage
from pathlib import Path


def main() -> None:  # pragma: no cover
    options = Arguments()
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        return

    config = Config(options['config'])
    workspace = config.workspace.get()

    if Path(workspace).exists() is False:
        print(f"{workspace} is not found.")
        return

    storage = Storage(
        Path(workspace).resolve(),
        fsmode=options['fs'],
        config_path=options['config']
    )

    storage.alive.init_alive()


if __name__ == "__main__":  # pragma: no cover
    main()
