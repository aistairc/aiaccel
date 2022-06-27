# -*- coding: utf-8-unix -*-

import argparse

from wd.daemon.nd import Nd


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('node_name', type=str, nargs=1)
    args = parser.parse_args()
    Nd(args.config, args.node_name).start()


if __name__ == '__main__':
    main()
