# -*- coding: utf-8-unix -*-

import sys

import wd
from wd.wrapper.opt import Opt


def main():
    #wd.global_error('debug00 wd.bin.opt')
    pn = wd.get_config_path_for_wd_bin_opt()
    if pn is None:
        #wd.global_error('debug01 wd.bin.opt')
        print('wd.bin.opt: optはwdからの実行ではありません。')
        print('-1', end='')
        return
    args = sys.argv
    #wd.global_error('debug02 wd.bin.opt')
    Opt('../config.yml', args).start()


if __name__ == '__main__':
    main()
