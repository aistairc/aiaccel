# -*- coding: utf-8-unix -*-

import sys
import os
import subprocess

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]
os.chdir(nn_wd_path)
cmd = 'python -m wd.bin.gpu %s' % node_name
subprocess.run(cmd.split(' '))
