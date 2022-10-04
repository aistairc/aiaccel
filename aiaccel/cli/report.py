import argparse
from aiaccel.util.report import CreationReaport
import sys


args = argparse.ArgumentParser(allow_abbrev=False)
args.add_argument('--config', '-c', type=str)
args.add_argument('--resume', type=int, default=None)
args.add_argument('--clean', nargs='?', const=True, default=False)
args.add_argument('--fs', nargs='?', const=True, default=False)


def main():
    options = vars(args.parse_args())
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        sys.exit()

    report = CreationReaport(options)
    report.create()


if __name__ == "__main__":
    main()
