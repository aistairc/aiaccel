# -*- coding: utf-8-unix -*-

import pathlib
import os
import traceback
import sys
import re

from aiaccel.config import load_config

import wd


class AbstractBase(object):

    def __init__(self, config):
        try:
            self.path_config = pathlib.Path(config).parent.resolve()
            self.config = load_config(config)
            self.logger = None

            self.ws = self.path_config/wd.workspace
            self.dict_opt_config = self.path_config/wd.dict_opt_config
            self.dict_log = self.ws/wd.dict_log
            self.dict_var = self.ws/wd.dict_var

            self.dict_qsub = self.ws/wd.dict_qsub
            self.dict_qsub_output = self.ws/wd.dict_qsub/wd.dict_output
            self.dict_node = self.ws/wd.dict_node
            self.dict_gpu = self.ws/wd.dict_gpu
            self.dict_tmp = self.ws/wd.dict_tmp

            self.path_qsub = self.ws/wd.dict_var/wd.file_qsub
            self.hostname = os.uname()[1]
            self.make_dict_work()
        except:
            wd.global_error_exit('logical error:\n'+traceback.format_exc())

    def make_dict_work(self):
        a = [self.ws,
             self.dict_log,
             self.dict_var,
             self.dict_qsub,
             self.dict_qsub_output,
             self.dict_node,
             self.dict_gpu,
             self.dict_tmp]
        for p in a:
            p.mkdir(exist_ok=True)

    def error_exit(self, message, exit_code=wd.exit_code_error):
        self.set_base_logger_if_none(message)
        self.logger.error(message)
        sys.exit(exit_code)

    def normal_exit(self, message='end', exit_code=wd.exit_code_normal):
        self.set_base_logger_if_none(message)
        self.logger.info(message)
        sys.exit(exit_code)

    def base_main(self):
        raise NotImplementedError

    def start(self):
        try:
            self.base_main()
        except SystemExit:
            pass
        except:
            self.set_base_logger_if_none()
            self.logger.error('logical error:\n'+traceback.format_exc())

    def set_base_logger_if_none(self, message='none'):
        if self.logger is None:
            try:
                self.logger = wd.set_logger(
                    'root.base', wd.log_level,
                    self.dict_log/wd.base_log_file,
                    'Base    ')
            except:
                wd.global_error_exit(message
                    + '\nlogical error:\n'+traceback.format_exc())

    def calc_node_num(self):
        path = self.dict_opt_config
        s = wd.opt_pt+wd.delimiter
        pa = [p for p in path.iterdir() if re.search(s, str(p))]
        fname = self.config.get('wd', 'name_aiaccel_config')
        nopt = len(pa)
        ntotal = 0
        ntry = 0
        for p in pa:
            s = str(p/fname)
            c = load_config(s)
            n = int(c.get('resource', 'num_node'))
            nt = int(c.get('optimize', 'trial_number'))
            ntotal += n
            ntry += nt
        nlarge = int(ntotal/4)
        nsmall = int(ntotal)%4
        return nopt, ntotal, ntry, nlarge, nsmall
