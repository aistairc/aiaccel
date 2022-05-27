# -*- coding: utf-8-unix -*-

import signal
import re
import time

import wd
from wd.base import AbstractBase


class Gpu(AbstractBase):

    def __init__(self, config, node_name):
        super().__init__(config)
        self.node_name = node_name[0]

    def init(self):
        self.logger = wd.set_logger(
            'root.gpu', wd.log_level,
            self.dict_log/(wd.gpu_log_file_fmt % self.node_name),
            'Gpu      ')
        signal.signal(signal.SIGHUP, self.exit_by_signal)
        signal.signal(signal.SIGTERM, self.exit_by_signal)
        dict = self.dict_node/self.node_name
        if not dict.is_dir():
            self.error_exit('「%s」ディレクトリ無し' % dict)
        self.sleep_sec = wd.gpu_sleep_sec
        self.qsub_max_dic = {}  # qsub番号: gpu数
        self.gpu_max = 0
        self.logger.info('start')

    def exit_by_signal(self, signum, frame):
        self.normal_exit('exit_by_signal signum=%d' % signum)

    def base_main(self):
        self.init()
        self.set_paramters()
        while True:
            self.sub_main()

    def set_paramters(self):
        nopt, ntotal, ntry, nlarge, nsmall = self.calc_node_num()
        cnt = 0
        for n in range(nlarge):
            self.qsub_max_dic[cnt] = 4
            cnt += 1
            self.gpu_max += 4
        for n in range(nsmall):
            self.qsub_max_dic[cnt] = 1
            cnt += 1
            self.gpu_max += 1

    def get_path_requests(self):
        ret = []
        for path in self.dict_gpu.glob('**/*'+wd.cat_request):
            p = wd.request_pt+wd.cat_request+'$'
            if re.match(p, path.name):
                ret.append(path)
        return ret

    def set_qsub_dic(self, qsub_dic, qsub, gpu):
        if qsub not in qsub_dic:
            qsub_dic[qsub] = []
        qsub_dic[qsub].append(gpu)
        qsub_dic[qsub].sort()

    def get_gpu(self, a):
        a.sort()
        n = 0
        for d in a:
            if d != n:
                return n
            n += 1
        return n

    def get_qsub_gpu(self, qsub_dic):
        for km in sorted(self.qsub_max_dic.keys()):
            if km not in qsub_dic:
                qsub = km
                gpu = 0
                return (qsub, gpu)
            for k, v in qsub_dic.items():
                if len(v) < self.qsub_max_dic[k]:
                    qsub = k
                    gpu = self.get_gpu(v)
                    return (qsub, gpu)
        self.error_exit('qsub_dic=%s' % qsub_dic)

    def get_file_name_gpu_request(self, path_request):
        a = list(self.dict_gpu.glob('**/*'+wd.cat_request))
        a += list(self.dict_gpu.glob('**/*'+wd.cat_running))
        p = (wd.request_pt+wd.delimiter
             + wd.qsub_pt+wd.delimiter
             + wd.gpu_pt+wd.delimiter)
        n = 0
        qsub_dic = {}  # qsub番号: gpuリスト[]
        for d in a:
            m = re.match(p, d.name)
            if m is None:
                continue
            g = m.groups()
            self.set_qsub_dic(qsub_dic, int(g[1]), int(g[2]))
            n += 1
        if n >= self.gpu_max:
            return None
        (qsub, gpu) = self.get_qsub_gpu(qsub_dic)
        fmt = (wd.delimiter
               + wd.qsub_fmt+wd.delimiter
               + wd.gpu_fmt+wd.cat_request)
        fn = path_request.name[:wd.request_width]
        fn += fmt % (qsub, gpu)
        return fn

    def sub_main(self):
        for path_request in self.get_path_requests():
            name = self.get_file_name_gpu_request(path_request)
            if name is None:
                break
            path = path_request.parent/name
            path_request.rename(path)

        time.sleep(self.sleep_sec)
