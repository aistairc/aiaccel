# -*- coding: utf-8-unix -*-

import argparse

from wd.script.qreq import Qreq


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('-s', '--small', action='store_true',
                        help='true(small) or false(large) default:false')
    args = parser.parse_args()
    Qreq(args.config, args.small).start()


if __name__ == '__main__':
    main()
