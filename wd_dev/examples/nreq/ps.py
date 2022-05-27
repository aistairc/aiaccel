import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]

cmd = ('ps -f >> '
       '%s/work/tmp/%s-ps.txt' % (nn_wd_path, node_name))
os.system(cmd)
