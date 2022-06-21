# -*- coding: utf-8-unix -*-

import pathlib
import shutil

import wd
from wd.base import AbstractBase


class Nreq(AbstractBase):
        
    def __init__(self, config, node_name, python_path, optno):
        super().__init__(config)
        self.node_name = node_name
        self.python_path = pathlib.Path(python_path)
        self.optno = optno

    def init(self):
        self.logger = wd.set_logger(
            'root.nreq', wd.log_level,
            self.dict_log/wd.nreq_log_file,
            'Nreq     ')
        self.dict = self.dict_node/self.node_name
        '''# -*- coding: utf-8-unix -*-

import os
import subprocess

import wd

dst_opt = os.environ['SGE_LOCALDIR']+'/%s/%s-opt'
cmd = 'mkdir -p '+dst_opt
os.system(cmd)
src_opt = os.getcwd()+'/%s-opt'
cmd = 'cp '+src_opt+'/* '+dst_opt
os.system(cmd)

os.chdir(dst_opt)
cmd = 'python -m aiaccel.start'
subprocess.run(cmd.split(' '))
cmd = 'rsync -avh '+dst_opt+'/ '+src_opt
os.system(cmd)
'''
        self.base_fmt = '''# -*- coding: utf-8-unix -*-

import os
import subprocess

import wd

src_opt = wd.get_opt_path(%d)
#wd.global_error('debug: {}'.format(src_opt))
os.chdir(src_opt)
cmd = 'python -m aiaccel.start --config config.yml'
subprocess.run(cmd.split(' '))
'''
        self.logger.info('start')

    def base_main(self):
        self.init()
        self.node_check()
        name = self.get_cnt_file_name(self.python_path.name)
        tp = self.dict/(name+wd.cat_tmp)
        rp = self.dict/(name+wd.cat_request)
        if self.optno is None:
            shutil.copy(self.python_path, tp)
        else:
            str = self.make_run_aiaccel_py()
            tp.write_text(str)
        tp.rename(rp)
        self.logger.info(rp.name)

    def node_check(self):
        p = self.dict_qsub/(self.node_name+wd.cat_running)
        if not p.is_file():
            self.error_exit('計算ノード名が違います。'+str(p))
        if not self.dict.is_dir():
            self.error_exit('wd.bin.ndが実行されていません。'+str(p))

    def get_cnt_file_name(self, base):
        cnt = 0
        while cnt < wd.max_nd_run_same_file:
            name = base+(wd.fmt_nd_run_same_file % cnt)
            cnt += 1
            # 連続して同じファイル名で瞬時に投げこまれることが無いという前提。
            if (self.dict/(name+wd.cat_finished)).is_file():
                continue
            if (self.dict/(name+wd.cat_running)).is_file():
                continue
            if (self.dict/(name+wd.cat_request)).is_file():
                continue
            if (self.dict/(name+wd.cat_tmp)).is_file():
                continue
            return name
        self.error_exit('not cnt(%d) < %s' % (cnt, wd.max_nd_run_same_file))

    def make_run_aiaccel_py(self):
        return self.base_fmt % self.optno
