# -*- coding: utf-8-unix -*-

import argparse

from wd.script.cui import Cui


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    args = parser.parse_args()
    Cui(args.config).start()


if __name__ == '__main__':
    main()
