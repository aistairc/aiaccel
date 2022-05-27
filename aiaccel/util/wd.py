# -*- coding: utf-8-unix -*-

import subprocess

num_node_all = -2  # -2: Not checked. -1: Checked not via wd. 1 and above: via wd.


def get_num_node_all_from_wd():
    cmd = 'python -m wd.bin.opt get_num_node_all'
    proc = subprocess.run(
        cmd.split(' '),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    res = proc.stdout.decode('utf8')
    if len(res) == 0:
        return -1
    return int(res.split('\n')[-1])  # -1 unless via wd. num_node_all via wd.


def check_num_node_all():
    global num_node_all
    if num_node_all == -2:
        num_node_all = get_num_node_all_from_wd()


def get_num_node_all():
    check_num_node_all()
    if num_node_all >= 1:
        return num_node_all
    return -1


def get_cmd_array():
    check_num_node_all()
    if num_node_all >= 1:
        return 'python -m wd.bin.opt'.split(' ')
    return None
