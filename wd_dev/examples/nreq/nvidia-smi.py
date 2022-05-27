import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]

cmd = ('nvidia-smi >> '
       '%s/work/tmp/%s-nvidia-smi.txt' % (nn_wd_path, node_name))
os.system(cmd)
