# -*- coding: utf-8-unix -*-

import signal
import subprocess
import re
import time

from aiaccel.config import load_config

import wd
from wd.base import AbstractBase


class Qsub(AbstractBase):

    def __init__(self, config):
        super().__init__(config)

    def init(self):
        self.logger = wd.set_logger(
            'root.qsub', wd.log_level,
            self.dict_log/wd.qsub_log_file,
            'Qsub     ')
        if self.path_qsub.is_file():
            msg = ('%s ファイルが有るので、既に起動しています。\n'
                   '2重起動はできませんので、何もしないで、終了します。'
                   % self.path_qsub)
            self.normal_exit(msg)
        else:
            self.path_qsub.write_text(self.hostname)
        signal.signal(signal.SIGHUP, self.log_for_signal)
        signal.signal(signal.SIGTERM, self.log_for_signal)
        self.group_name = self.get_group_name()
        self.sleep_sec = wd.qsub_sleep_sec

        nopt, ntotal, ntry, nlarge, nsmall = self.calc_node_num()
        self.opt_total = nopt

        self.logger.info('start')

    def base_main(self):
        self.init()
        while True:
            self.sub_main()

    def log_for_signal(self, signum, frame):
        self.logger.info('シグナルは無視します。signum=%d' % signum)

    def error_exit(self, message):
        if self.path_qsub.is_file():
            self.path_qsub.unlink()
        super().error_exit(message)

    def get_group_name(self):
        dicts = list(self.dict_opt_config.glob(
            (wd.opt_fmt % 0)+wd.delimiter+'*'))
        if len(dicts) != 1:
            self.error_exit(
                'len(dicts) != 1 dict_opt_config=%s'
                % self.dict_opt_config)
        path = (self.dict_opt_config/dicts[0]
                / self.config.get('wd', 'name_opt_config'))
        try:
            ret = load_config(path).get('ABCI', 'group')
        except:
            self.error_exit("load_config(%s).get('ABCI', 'group')" % path)
        # debug。後で修正のこと。2021.11.25
        return ret[1:-1]
        #return ret

    def do_qsub(self, path_request):
        file_request = path_request.name
        file_output = wd.get_file_name_no_category(file_request)
        path_output = self.dict_qsub_output/file_output
        file_running = wd.rename_file_category(file_request, wd.cat_running)
        path_running = self.dict_qsub/file_running
        path_request.rename(path_running)
        if wd.test_no_qsub:  # qsubは実行せずに、qsubファイルをprocessとして実行。
            cmd = '/bin/bash %s' % path_running
            self.logger.info(cmd)
            proc = subprocess.Popen(cmd, shell=True)
            self.logger.info('after Popen')
        else:  # 通常実行。
            cmd = 'qsub -g %s -o %s %s'\
                  % (self.group_name, path_output, path_running)
            self.logger.info(cmd)
            proc = subprocess.run(cmd.split(' '), stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
            res = proc.stdout.decode('utf8')
            self.logger.info(res)

    def get_path_requests(self):
        ret = []
        p = wd.qsub_pt+wd.cat_request+'$'
        for path in self.dict_qsub.glob('**/*'+wd.cat_request):
            if re.match(p, path.name):
                ret.append(path)
        return ret

    def stop_all_qsub(self):
        proc = subprocess.run(
            'qstat', shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        res = proc.stdout.decode('utf8')
        if len(res) == 0:
            self.logger.error('stop_all_qsub: 実行されているqsubがありませんでした。')
            return
        ret = ''
        p = r'\s+(\d+).+'
        p += wd.qsub_pt
        for line in res.split('\n'):
            m = re.match(p, line)
            if m is not None:
                cmd = 'qdel '+m.groups()[0]
                subprocess.run(cmd, shell=True)
                self.logger.info('stop_all_qsub: '+cmd)

    def check_all_opt_end(self):
        a1 = []
        for f1 in self.dict_node.glob(
                '**/%s-*%s' %
                (wd.file_run_aiaccel_py, wd.cat_finished)):
            a1.append(f1)
        if len(a1) == self.opt_total:
            return True
        return False

    def sub_main(self):
        if not self.path_qsub.is_file():
            msg = ('stop %s\n%s ファイルが有りません。\n'
                   '終了指示と認識し、終了します。'
                   % (self.hostname, self.path_qsub))
            self.normal_exit(msg)

        if self.check_all_opt_end():
            msg = ('optが全て終了しました。\n'
                   '全てのqsubを終了します。\n'
                   '本daemonも終了します。')
            self.stop_all_qsub()
            self.path_qsub.unlink()
            self.normal_exit(msg)
            
        for path_request in self.get_path_requests():
            self.do_qsub(path_request)

        time.sleep(self.sleep_sec)
