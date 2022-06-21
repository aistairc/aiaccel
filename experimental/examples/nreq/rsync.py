import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]  # ここでは未使用。

cmd = ('rsync -avh $SGE_LOCALDIR/ %s/work/tmp/rsync'
       ' >> %s/work/tmp/%s-rsync.txt' % (nn_wd_path, nn_wd_path, node_name))
os.system(cmd)
