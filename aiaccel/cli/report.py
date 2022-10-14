from logging import getLogger, StreamHandler, DEBUG

import sys

from aiaccel.argument import Arguments
from aiaccel.util.report import CreationReaport

logger = getLogger(__name__)
logger.setLevel(DEBUG)
logger.addHandler(StreamHandler())


def main():
    options = Arguments()
    if "config" not in options.keys():
        logger.info("Specify the config file path with the --config option.")
        sys.exit()

    report = CreationReaport(options)
    report.create()


if __name__ == "__main__":
    main()
