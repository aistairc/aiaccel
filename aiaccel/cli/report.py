import os
from argparse import ArgumentParser
from logging import StreamHandler, getLogger

from aiaccel.cli import CsvWriter

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


def main() -> None:  # pragma: no cover
    """Parses command line options and reports the result."""
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    args = parser.parse_args()

    csv_writer = CsvWriter(args.config)
    csv_writer.create()


if __name__ == "__main__":  # pragma: no cover
    main()
