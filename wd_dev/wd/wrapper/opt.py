# -*- coding: utf-8-unix -*-

import re
import pathlib
import subprocess
import time

import fasteners

import wd
from wd.base import AbstractBase


class Opt(AbstractBase):

    def __init__(self, config, params):
        super().__init__(config)
        self.params = params

    def get_num_opt(self):
        p = wd.opt_pt+wd.delimiter
        m = re.match(p, self.num_opt_path.name)
        if m is None:
            self.error_exit('path=%s p=%s' % (self.num_opt_path, p))
        s = m.groups()[0]
        return s

    def init(self):
        self.num_opt_path = pathlib.Path.cwd()  # *-optから実行されるので。
        self.num_opt = self.get_num_opt()
        self.logger = wd.set_logger(
            'root.opt_%s' % self.num_opt, 'DEBUG',
            self.dict_log/(wd.opt_log_file_fmt % self.num_opt),
            'Opt     ',
            stream=False)
        self.sleep_sec = wd.opt_sleep_sec
        self.gpu_opt = self.dict_gpu/self.num_opt_path.name
        self.gpu_opt.mkdir(exist_ok=True)
        #self.count_path = self.gpu_opt/wd.opt_count_file
        #self.lock_path = self.gpu_opt/wd.opt_count_lock_file
        self.logger.info('start params=%s' % ' '.join(self.params))

    def finished_path_exists(self):
        a = list(self.gpu_opt.glob(self.finished_file_pt))
        n = len(a)
        if n == 1:
            self.finished_path = a[0]
            return True
        elif n == 0:
            return False
        else:
            self.error_exit('n=%d pt=%s gpu_opt=%s'
                            % (n, self.finished_file_pt, self.gpu_opt))

    def make_path_etc(self, n):
        self.num_str = wd.request_fmt % n
        self.tmp_path = self.gpu_opt/(self.num_str+wd.cat_tmp)
        self.request_path = self.gpu_opt/(self.num_str+wd.cat_request)
        self.finished_file_pt = self.num_str+wd.delimiter+'*'+wd.cat_finished

    '''
    def get_count(self):
        p = self.count_path
        lock = fasteners.InterProcessLock(str(self.lock_path))
        if not lock.acquire(timeout=wd.opt_lock_wait_sec):
            self.error_exit('lock timeout '+p)
        n = 0
        if p.exists():
            n = int(p.read_text())
            n += 1
        p.write_text('%d' % n)
        lock.release()
        return n
    '''

    # daemon/ai.pyを参考に。
    def exec_ai_this_node_for_debug(self):
        command_line = ' '.join(self.params[2:])
        ai_exec_file_path = (
            self.path_config/self.config.get('wd', 'ai_exec_file_path'))
        cmd = str(ai_exec_file_path)
        cmd += ' "'+command_line+'"'
        subprocess.run(cmd, shell=True)
    
    def base_main(self):
        self.init()
        if self.params[1] == wd.wrapper_opt_get_num_node_all_param_name:
            nopt, ntotal, ntry, nlarge, nsmall = self.calc_node_num()
            print(str(ntotal), end='')
            return
        if not self.path_qsub.is_file():
            self.exec_ai_this_node_for_debug()
            return
        # /home/acb11523fz/opt/wd/bin/opt.py python user.py --index 000000 --config /home/acb11523fz/opt/wd/sample/config/00-1grid-0gpu/000-grid4/config.json --x1=0.0 --x2=0.0
        n = int(self.params[4])
        self.make_path_etc(n)
        s = '%s :request\n' % wd.now_str()
        s += ' '.join(self.params[1:])
        self.tmp_path.write_text(s)
        self.tmp_path.rename(self.request_path)

        while True:
            if self.finished_path_exists():
                break
            time.sleep(self.sleep_sec)

        res = self.finished_path.read_text()
        p = '(objective_y:\S+)\n'
        m = re.search(p, res)
        ret = 'None'
        if m is None:
            self.logger.error('path=%s p=%s' % (self.num_opt_path, p))
        else:
            ret = m.groups()[0]
        self.logger.info('end result=%s params=%s count=%d'
                         % (ret, ' '.join(self.params),
                            n))
        print(ret)
