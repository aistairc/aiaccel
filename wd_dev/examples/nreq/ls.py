import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]

cmd = ('ls -R $SGE_LOCALDIR >> '
       '%s/work/tmp/%s-ls-R.txt' % (nn_wd_path, node_name))
os.system(cmd)
