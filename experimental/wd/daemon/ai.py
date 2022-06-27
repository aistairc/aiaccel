# -*- coding: utf-8-unix -*-

import signal
import pathlib
import os
import re
import subprocess
import time

import wd
from wd.base import AbstractBase


class Ai(AbstractBase):

    def __init__(self, config, node_name):
        super().__init__(config)
        self.node_name = node_name[0]

    def init(self):
        self.logger = wd.set_logger(
            'root.ai', wd.log_level,
            self.dict_log/(wd.ai_log_file_fmt % self.node_name),
            'Ai       ')
        signal.signal(signal.SIGHUP, self.exit_by_signal)
        signal.signal(signal.SIGTERM, self.exit_by_signal)
        dict = self.dict_node/self.node_name
        if not dict.is_dir():
            self.error_exit('「%s」ディレクトリ無し' % dict)
        self.sleep_sec = wd.ai_sleep_sec
        self.procs_info = []  # [proc, path]
        self.logger.info('start')

    def base_main(self):
        self.init()
        while True:
            self.sub_main()

    def exit_by_signal(self, signum, frame):
        self.normal_exit('exit_by_signal signum=%d' % signum)

    def get_command_line(self, path_request):
        txt = path_request.read_text()
        return txt.split('\n')[-1]

    def set_gpu(self, gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    def exec_ai(self, path_request):
        p = wd.gpu_pt+wd.cat_request+'$'
        m = re.search(p, path_request.name)
        if m is None:
            self.error_exit('p=%s path_request=%s' % (p, path_request))
        gpu = m.groups()[0]
        self.set_gpu(gpu)
        command_line = self.get_command_line(path_request)  # index, x1, x2, ...
        running = wd.rename_file_category(path_request.name, wd.cat_running)
        dst = path_request.parent/running
        path_request.rename(dst)
        ai_path = self.path_config/path_request.parent.name  # *-wd/*-opt
        cmd = '(cd '+str(ai_path)+'; '+command_line+')'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        self.procs_info.append([proc, dst])
        self.logger.info('start pid=%d dst=%s cmd=%s'
                         % (proc.pid, dst, cmd))

    def get_path_requests(self):
        a = list(self.dict_gpu.glob('**/*'+wd.cat_request))
        p = (wd.request_pt+wd.delimiter
             + self.node_name+wd.delimiter
             + wd.gpu_pt+wd.cat_request
             + '$')
        b = [d for d in a if re.match(p, d.name)]
        return b

    def write_result(self, path, outd, errd):
        with path.open(mode='at') as f:
            f.write('\n%s :finished\n' % wd.now_str())
            if errd is not None:
                f.write(errd.decode('utf8'))
            f.write(outd.decode('utf8'))

    def sub_main(self):
        for pi in self.procs_info[:]:
            if pi[0].poll() is not None:
                outd, errd = pi[0].communicate()
                self.write_result(pi[1], outd, errd)
                src = pi[1]
                sdst = wd.rename_file_category(str(src), wd.cat_finished)
                src.rename(pathlib.Path(sdst))
                self.logger.info('stop pid=%d %s' % (pi[0].pid, sdst))
                self.procs_info.remove(pi)

        for path_request in self.get_path_requests():
            self.exec_ai(path_request)

        time.sleep(self.sleep_sec)
