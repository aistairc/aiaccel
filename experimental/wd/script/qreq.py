# -*- coding: utf-8-unix -*-

import os
import re

import wd
from wd.base import AbstractBase


class Qreq(AbstractBase):
        
    def __init__(self, config, small):
        super().__init__(config)
        self.small = small

    def init(self):
        self.logger = wd.set_logger(
            'root.qreq', wd.log_level,
            self.dict_log/wd.qreq_log_file,
            'Qreq     ')
        self.base_fmt = '''#!/bin/bash
# -*- coding: utf-8-unix; mode: Text; -*-

#$ -l rt_G.%s=1
#$ -l h_rt=%s
#$ -j y
#$ -cwd

source %s/tools/make_wd_env_for_dev.source
cd %s
python -m wd.bin.nd %s
'''
        self.logger.info('start')

    def write_script(self, path, qno):
        if self.small:
            qtype = 'small'
            m = int(eval(self.config.get('wd', 'max_small_minutes')))
            if m > 167*60:
                m = 167*60
        else:
            qtype = 'large'
            m = int(eval(self.config.get('wd', 'max_large_minutes')))
            if m > 71*60:
                m = 71*60
        h_rt = self.get_h_rt_str(m)
        s = self.base_fmt % (qtype, h_rt, os.environ[wd.wd_dev_dir_env_name],
                             str(self.path_config), qno)
        path.write_text(s)

    def base_main(self):
        self.init()
        n = self.get_qno()
        qno = wd.qsub_fmt % n
        tp = self.dict_qsub/(qno+wd.cat_tmp)
        rp = self.dict_qsub/(qno+wd.cat_request)
        self.write_script(tp, qno)
        tp.rename(rp)
        self.logger.info(rp.name)

    def get_qno(self):
        p = wd.qsub_pt+wd.delimiter
        a = []
        for path in self.dict_qsub.glob('**/*'):
            m = re.match(p, path.name)
            if m is not None:
                a.append(int(m.groups()[0]))
        if len(a) == 0:
            return 0
        a.sort()
        return a[-1]+1

    def get_h_rt_str(self, minutes):
        h = minutes/60
        m = minutes%60
        return '%d:%02d:00' % (h, m)
