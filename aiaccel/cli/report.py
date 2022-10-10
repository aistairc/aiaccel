from argparse import ArgumentParser

from aiaccel.util.report import CreationReport


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default="config.yml")
    args = parser.parse_args()

    report = CreationReport(args.config)
    report.create()


if __name__ == "__main__":
    main()
