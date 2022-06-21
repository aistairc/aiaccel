# -*- coding: utf-8-unix -*-

import argparse

from wd.daemon.qsub import Qsub


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    args = parser.parse_args()
    Qsub(args.config).start()


if __name__ == '__main__':
    main()
