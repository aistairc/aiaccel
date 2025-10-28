import argparse

from aiaccel.config.config import load_config, overwrite_omegaconf_dumper, resolve_inherit


def main() -> None:
    overwrite_omegaconf_dumper()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument("--key", nargs="*", help="Target key in configration file")

    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    config = resolve_inherit(config)

    value = config
    for k in args.key:
        if value is not None:
            value = value.get(k)

    print(value)


if __name__ == "__main__":
    main()
