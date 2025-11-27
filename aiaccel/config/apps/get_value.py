import argparse

from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.config.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument("key", help="Target key in the configration file")

    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    print(oc.select(config, args.key))


if __name__ == "__main__":
    main()
