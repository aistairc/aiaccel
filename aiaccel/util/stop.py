import pathlib
import aiaccel


def stop_opt(workspace: pathlib.Path) -> None:
    """ Forcibly stops optimization.

    Returns:
        None
    """
    if not workspace.exists():
        return
    print("all stop")
    files = [
        aiaccel.alive_master,
        aiaccel.alive_scheduler,
        aiaccel.alive_optimizer
    ]
    for file in files:
        alive_file = workspace / aiaccel.dict_alive / file
        if alive_file.exists():
            print(
                'Delete alive file: {}.'
                .format(alive_file)
            )
            alive_file.unlink()
    return
