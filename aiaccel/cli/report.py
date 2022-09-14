from aiaccel.argument import Arguments
from aiaccel.util.report import CreationReaport
import sys

if __name__ == "__main__":
    options = Arguments()
    if "config" not in options.keys():
        print("Specify the config file path with the --config option.")
        sys.exit()

    report = CreationReaport(options['config'])
    report.create()
