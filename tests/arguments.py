from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--clean", nargs="?", const=True, default=False)

    return vars(parser.parse_args())
