import argparse

from aiaccel.config.config import load_config, overwrite_omegaconf_dumper, resolve_inherit
from aiaccel.config.git import collect_git_status_from_config, print_git_status


def main() -> None:
    overwrite_omegaconf_dumper()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file path", default=None)

    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    config = resolve_inherit(config)

    if len(git_status := collect_git_status_from_config(config)) > 0:
        print_git_status(git_status)

        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
