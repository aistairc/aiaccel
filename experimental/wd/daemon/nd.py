# -*- coding: utf-8-unix -*-

import signal
import pathlib
import subprocess
import time

import wd
from wd.base import AbstractBase


class Nd(AbstractBase):

    def __init__(self, config, node_name):
        super().__init__(config)
        self.node_name = node_name[0]

    def init(self):
        self.logger = wd.set_logger(
            'root.nd', wd.log_level,
            self.dict_log / (wd.nd_log_file_fmt % self.node_name),
            'Nd       ')
        signal.signal(signal.SIGHUP, self.exit_by_signal)
        signal.signal(signal.SIGTERM, self.exit_by_signal)
        if not self.dict_node.is_dir():
            self.error_exit('「%s」ディレクトリ無し' % self.dict_node)
        self.dict = self.dict_node / self.node_name
        self.dict.mkdir(exist_ok=True)
        self.sleep_sec = wd.nd_sleep_sec
        self.procs_info = []  # [proc, path]
        self.logger.info('start')

    def base_main(self):
        self.init()
        while True:
            self.sub_main()

    def exit_by_signal(self, signum, frame):
        self.normal_exit('exit_by_signal signum=%d' % signum)

    def exit_sub(self, message):
        dst = pathlib.Path(str(self.dict)+wd.cat_finished)
        if self.dict.is_dir():
            if dst.exists():
                super().error_exit('dst.exists() %s %s' % (dst, message))
            self.dict.rename(dst)

    def error_exit(self, message):
        self.exit_sub(message)
        super().error_exit(message)

    def normal_exit(self, message):
        self.exit_sub(message)
        super().normal_exit(message)

    def sub_main(self):
        for pi in self.procs_info[:]:
            if pi[0].poll() is not None:
                src = pi[1]
                dst = wd.rename_file_category(str(src), wd.cat_finished)
                src.rename(pathlib.Path(dst))
                self.logger.info('stop pid=%d %s' % (pi[0].pid, dst))
                self.procs_info.remove(pi)

        for src in self.dict.glob('*'+wd.cat_request):
            dst = pathlib.Path(
                wd.rename_file_category(str(src), wd.cat_running))
            src.rename(dst)
            cmd = ['python', str(dst), self.node_name, str(self.path_config)]
            proc = subprocess.Popen(cmd)
            self.procs_info.append([proc, dst])
            self.logger.info('start pid=%d %s' % (proc.pid, dst))

        time.sleep(self.sleep_sec)
