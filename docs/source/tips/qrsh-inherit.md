# qsubで実行中の計算ノードでコマンドを実行する方法
qsubで実行中の計算ノードでコマンドを実行する方法を説明します。

参照資料<br>
[インタラクティブノードからのqrsh -inherit](https://github.com/aistairc/abci-docs/issues/246)

## テスト時のqsub.sh
~~~
#!/bin/bash

#$ -l rt_C.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

sleep 3600
~~~

## テスト時のコマンド
~~~
> qsub -g your_group_id qsub.sh
Your job 10607396 ("qsub.sh") has been submitted

> qstat  # with some omissions
job-ID    prior  name     user  state  submit/start at  queue
10608572  prior  qsub.sh  user  r      submit/start at  gpu@g0001

> JOB_ID=10608572 SGE_TASK_ID=undefined qrsh -inherit g0001 /bin/bash
~~~
promptが表示されませんが、例えば、
~~~
> ls
~~~
とコマンドを入力すると、結果が出力されます。

終了時は必ず
~~~
> exit
~~~

## /bin/bashなどのshellの場合の注意事項

1. キーボードからのControl-Cで、job自体の実行が停止されるようです。<br>
   そのため、終了には、exitコマンドを使用して下さい。

2. コマンドプロンプトが出ないようです。<br>
   違和感が有りますが、コマンド-Enterでコマンドを実行できます。

3. 使用できないコマンド、出力が通常と違うコマンドが有ります。
