# SGE_LOCALDIRを定期的にHOMEなどにrsyncする方法
qsubで実行中の計算ノードの$SGE_LOCALDIRの内容を$HOMEなどに定期的にrsyncする方法を説明します。

ユーザプログラムが終了した後に、再度1度だけrsyncしています。

## テスト時のqsub.sh
~~~
#!/bin/bash

#$ -l rt_C.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13

python3 rsync.py loop &
python3 test.py
kill $!
sleep 1
python3 rsync.py
~~~

## テスト時のrsync.py
~~~
import os
import subprocess
import sys
import signal
import time

sleep_time = 15
src_dir = os.environ['SGE_LOCALDIR']
dst_dir = '~/test_rsync_dir'
flag = True
proc = None


def do_rsync():
    global proc
    proc = subprocess.Popen(f'exec rsync -avh {src_dir}/ {dst_dir}', shell=True)
    proc.wait()


def handler(signum, frame):
    global flag
    flag = False
    if proc is not None:
        proc.terminate()
    sys.exit()


signal.signal(signal.SIGTERM, handler)
a = sys.argv
if len(a) == 2 and a[1] == 'loop':
    while True:
        time.sleep(sleep_time)
        if flag:
            do_rsync()
else:
    do_rsync()
~~~

## テスト時のユーザプログラムtest.py
~~~
import os
import pathlib
import time

sleep_time = 5
src_path = pathlib.Path(os.environ['SGE_LOCALDIR'])


for i in range(10):
    (src_path/f'{i}-touch.txt').touch()
    time.sleep(sleep_time)
~~~

## テスト時のコマンド
~~~
$ cd your_directory # with the above three files
$ qsub -g your_group_id qsub.sh
# After the end
$ ls ~/test_rsync_dir
0-touch.txt  2-touch.txt  4-touch.txt  6-touch.txt  8-touch.txt
1-touch.txt  3-touch.txt  5-touch.txt  7-touch.txt  9-touch.txt
$ vi your_qsub_output_file
~~~

## テスト時のvi your_qsub_output_fileの表示結果
~~~
sending incremental file list
created directory /your_home_directory/test_rsync_dir
./
0-touch.txt
1-touch.txt
2-touch.txt
3-touch.txt

sent 342 bytes  received 149 bytes  982.00 bytes/sec
total size is 0  speedup is 0.00
sending incremental file list
./
4-touch.txt
5-touch.txt
6-touch.txt

sent 378 bytes  received 76 bytes  908.00 bytes/sec
total size is 0  speedup is 0.00
sending incremental file list
./
7-touch.txt
8-touch.txt
9-touch.txt

sent 453 bytes  received 76 bytes  1.06K bytes/sec
total size is 0  speedup is 0.00
sending incremental file list

sent 329 bytes  received 12 bytes  682.00 bytes/sec
total size is 0  speedup is 0.00
~~~
