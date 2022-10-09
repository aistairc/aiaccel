from argparse import ArgumentParser

from aiaccel.util import Reporter


def main():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    args = parser.parse_args()

    reporter = Reporter(args.config)
    reporter.create()


if __name__ == "__main__":
    main()
