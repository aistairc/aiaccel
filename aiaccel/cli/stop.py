from aiaccel.argument import Arguments
from aiaccel.config import Config
from aiaccel.workspace import Workspace
from aiaccel.storage.storage import Storage


def main() -> None:  # pragma: no cover
    options = Arguments()
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        return

    config = Config(options['config'])
    workspace = config.workspace.get()

    ws = Workspace(workspace)
    if ws.exists() is False:
        print(f"{workspace} is not found.")
        return

    storage = Storage(
        workspace,
        fsmode=config.filesystem_mode.get(),
        config_path=options['config']
    )

    storage.alive.init_alive()


if __name__ == "__main__":  # pragma: no cover
    main()
