from aiaccel.argument import Arguments
from aiaccel.util.report import CreationReaport
import sys


def main():
    options = Arguments()
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        sys.exit()

    report = CreationReaport(options)
    report.create()


if __name__ == "__main__":
    main()
