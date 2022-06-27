# -*- coding: utf-8-unix -*-

import argparse

from wd.script.nreq import Nreq


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-c', '--config', type=str, default='config.yml')
    parser.add_argument('node_name',
        help='実行する計算ノード名。')
    parser.add_argument('python_path',
        help=('計算ノードで実行する*.pyファイルのpath。\n'
              'または-o指定の場合は書き込みファイル名のpath。'
              'このpathには書かないrequestファイルの名前として使用。'))
    parser.add_argument('-o', '--optno', type=int,
        help='run_aiaccel.pyファイルを作成。*-optの番号を指定。')
    args = parser.parse_args()
    Nreq(args.config, args.node_name, args.python_path, args.optno).start()


if __name__ == '__main__':
    main()
