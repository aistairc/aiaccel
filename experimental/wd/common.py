# -*- coding: utf-8-unix -*-

'''Common variables and methods.

    * Import this as follows:
    * import aiaccel.wd as wd
'''

import logging
import sys
import datetime
import pathlib
import re
import os

wrapper_opt_get_num_node_all_param_name = 'get_num_node_all'

wd_dev_dir_env_name = 'WD_DEV_DIR'
venv_activate_env_name = 'WD_VENV_ACTIVATE'

# test_*は、テスト用の設定。通常実行時は、全てFalseに。

# False: 通常実行。
# True: qsubは実行せずに、qsubファイルをprocessとして実行。
test_no_qsub = False

#rd_env_name = 'WD_RD'
aiaccel_wd = '/aiaccel/wd'  # とりあえず、最初のwd-devコミットでは。後で修正予定。
config_file_name = 'config.yml'
dict_wd_config = 'sample/config'
#dict_ai_work = 'ai-work'

exit_code_normal = 0
exit_code_error = 1

workspace = 'work'  # config.jsonのcwd(例えば00-wd)からの相対パス。
dict_opt_config = '.'  # config.jsonのcwd(例えば00-wd)からの相対パス。

log_level = 'DEBUG'
file_log_level = 'DEBUG'
stream_log_level = 'DEBUG'

global_error_log_file = 'global_error.log'
global_error_logger = None

base_log_file = 'base.log'
start_log_file = 'start.log'
cmd_log_file = 'cmd.log'
cui_log_file = 'cui.log'
nd_log_file_fmt = 'nd_%s.log'
qsub_log_file = 'qsub.log'
opt_log_file_fmt = 'opt_%s.log'
gpu_log_file_fmt = 'gpu_%s.log'
ai_log_file_fmt = 'ai_%s.log'
qreq_log_file = 'qreq.log'
nreq_log_file = 'nreq.log'

delimiter = '-'

wd_width = 2
wd_fmt = '%0{}d'.format(wd_width)
wd_pt = r'(\d{%d})' % wd_width

opt_width = 3
opt_fmt = '%0{}d'.format(opt_width)
opt_pt = r'(\d{%d})' % opt_width

request_width = 4
request_fmt = '%0{}d'.format(request_width)
request_pt = r'(\d{%d})' % request_width

qsub_width = 4
qsub_fmt = 'q%0{}d'.format(qsub_width)
qsub_pt = r'q(\d{%d})' % qsub_width

gpu_width = 1
gpu_fmt = 'g%0{}d'.format(gpu_width)
gpu_pt = r'g(\d{%d})' % gpu_width

#opt_count_file = 'count.txt'
#opt_count_lock_file = 'count.lock'
#opt_lock_wait_sec = 10

max_nd_run_same_file = 1000
fmt_nd_run_same_file = '{}%03d'.format(delimiter)

dict_qsub = 'qsub'  # directory
dict_node = 'node'
dict_gpu = 'gpu'
dict_var = 'var'
dict_log = 'log'
dict_tmp = 'tmp'
dict_output = 'output'  # qsub/

cat_tmp = '{}tmp'.format(delimiter)
cat_request = '{}request'.format(delimiter)
cat_running = '{}running'.format(delimiter)
cat_finished = '{}finished'.format(delimiter)

file_qsub = 'qsub'
file_run_aiaccel_py = 'run_aiaccel.py'

qsub_sleep_sec = 3  # dict_qsubのチェック間隔(秒)。
nd_sleep_sec = 3  # dict_nodeのチェック間隔(秒)。
opt_sleep_sec = 3  # opt(wrapper)の*-finishedファイルのチェック間隔。
gpu_sleep_sec = 10  # gpuの空きチェック*-requestの間隔。
ai_sleep_sec = 3  # 計算nodeでの*-q*-g*-requestのチェック間隔。


def set_logger(name, level, log_file, module_type, stream=True):
    logger = logging.getLogger(name)
    logger.setLevel(eval('logging.%s' % level))

    file_level = eval('logging.%s' % file_log_level)
    stream_level = eval('logging.%s' % stream_log_level)

    fh = logging.FileHandler(str(log_file), mode='a')
    fh_formatter = '%(asctime)s %(levelname)-8s %(filename)-12s line ' \
                   '%(lineno)-4s %(funcName)s %(message)s'
    fh_formatter = logging.Formatter(fh_formatter)
    fh.setFormatter(fh_formatter)
    fh.setLevel(file_level)
    logger.addHandler(fh)

    if stream:
        ch = logging.StreamHandler()
        ch_formatter = '{} %(levelname)-8s %(message)s'.format(module_type)
        ch_formatter = logging.Formatter(ch_formatter)
        ch.setFormatter(ch_formatter)
        ch.setLevel(stream_level)
        logger.addHandler(ch)

    return logger


def set_global_error_logger(stream=True):
    global global_error_logger
    if global_error_logger is None:
        global_error_logger = set_logger(
            'root.global_error', log_level,
            global_error_log_file,
            'GERROR  ',
            stream)


def global_error(message, stream=True):
    set_global_error_logger(stream)
    global_error_logger.error(message)


def global_error_exit(message, exit_code=exit_code_error):
    global_error(message)
    sys.exit(exit_code)


def now_str():
    dt = datetime.datetime.now()
    return dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]


#def get_paths(dict, cat):
#    return [f for f in dict.glob('*'+cat)]


def get_file_name_no_category(src_name):
    i = src_name.rfind(delimiter)
    if i == -1:
        raise Exception(
            'src_name.rfind(delimiter) == -1 %s'
            % src_name)
    return src_name[0:i]


def rename_file_category(src_name, dst_cat):
    return get_file_name_no_category(src_name)+dst_cat


def get_config_path_for_wd_bin_opt():
    # .../*-wd/*-opt
    # cwd.nameが'nnn-*'かチェック。
    p1 = opt_pt+delimiter
    d1 = pathlib.Path.cwd()
    if re.match(p1, d1.name) is None:
        return None
    # cwd.parent.nameが'nn-*'かチェック。
    p2 = wd_pt+delimiter
    d2 = d1.parent
    if re.match(p2, d2.name) is None:
        return None
    # nn-*/config.ymlがあるかどうかチェック。
    f1 = d2/config_file_name
    if not f1.exists():
        return None
    return f1


def get_opt_path(optno):
    d1 = pathlib.Path.cwd()
    s1 = (opt_fmt % optno)+delimiter+'*'
    a1 = list(d1.glob(s1))
    if len(a1) == 1:
        return str(a1[0])
    return None
