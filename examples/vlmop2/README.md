# 多目的最適化の使用方法


## 1. 目的関数がPythonの場合

多目的最適化関数をaiaccelで最適化する場合，目的関数の返り値はリストである必要があります．

- objective example
~~~ python
import aiaccel
import numpy as np

def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"], p["x6"], p["x7"], p["x8"], p["x9"], p["x10"]])
    n = len(x)
    y1 = 1 - np.exp(-sum([(x[i] - 1 / np.sqrt(n))**2 for i in range(n)]))
    y2 = 1 - np.exp(-sum([(x[i] + 1 / np.sqrt(n))**2 for i in range(n)]))

    return [y1, y2]


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
~~~

コンフィグファイルの `optimize.goal` には，目的関数の最小化・最大化を指定します．目的関数の返り値のリストの要素数と同じ数の要素を持つリストを指定します．


- config example
~~~ yaml
generic:
  workspace: ./work
  job_command: python vlmop2.py
  enabled_variable_name_argumentation: True

resource:
  type: local
  num_workers: 4

# ABCI:
#   group: your-group-id
#   job_execution_options: ""
#   job_script_preamble: |
#     #!/bin/bash

#     #$-l rt_C.small=1
#     #$-j y
#     #$-cwd

#     source /etc/profile.d/modules.sh
#     module load gcc/12.2.0
#     module load python/3.11

job_setting:
    job_timeout_seconds: 600
    max_failure_retries: 0
    trial_id_digits: 7

optimize:
  search_algorithm: aiaccel.optimizer.MOTpeOptimizer
  goal: [minimize, minimize]
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: x1
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x2
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x3
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x4
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x5
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x6
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x7
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x8
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x9
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x10
      type: uniform_float
      lower: 0.0
      upper: 1.0
~~~



## 2. Python以外の場合 (fortran 95 を例に解説)
目的関数がPythonでは無い場合，次の二点に注意してください．
1. 目的関数の入力はコマンドライン引数で与えられます．目的関数がコマンドライン引数を取得できるように変更してください・
2. 結果を標準出力に出力してください．複数の値を出力する場合，値ごとに改行して出力してください．


- objective example
~~~ fortran
program vlmop2
    implicit none
    integer, parameter :: n = 10

    character(len=20) :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
    real(kind=8) :: x1, x2, x3, x4, x5, x6, x7, x8, x9, x10
    REAL(KIND=8) :: x(n)
    real(kind=8) :: y1, y2

    ! read command line arguments
    call get_command_argument(1, arg1)
    call get_command_argument(2, arg2)
    call get_command_argument(3, arg3)
    call get_command_argument(4, arg4)
    call get_command_argument(5, arg5)
    call get_command_argument(6, arg6)
    call get_command_argument(7, arg7)
    call get_command_argument(8, arg8)
    call get_command_argument(9, arg9)
    call get_command_argument(10, arg10)

    read(arg1, *) x1
    read(arg2, *) x2
    read(arg3, *) x3
    read(arg4, *) x4
    read(arg5, *) x5
    read(arg6, *) x6
    read(arg7, *) x7
    read(arg8, *) x8
    read(arg9, *) x9
    read(arg10, *) x10

    x = (/x1, x2, x3, x4, x5, x6, x7, x8, x9, x10/)

    y1 = 1.0 - EXP(-SUM((x - 1.0 / SQRT(REAL(n, KIND=8)))**2))
    y2 = 1.0 - EXP(-SUM((x + 1.0 / SQRT(REAL(n, KIND=8)))**2))

    PRINT *, y1
    PRINT *, y2

end program vlmop2
~~~


job_command は，コンパイルしたオブジェクトファイルを実行するコマンドを指定します．<br>
ここでは，`a.out` を指定しています．

`generic.enabled_variable_name_argumentation` を `False` にしてください.<br>
`enabled_variable_name_argumentation` の `True`/`False` によって，コマンドライン引数の与え方が次のように変わります．

- generic.enabled_variable_name_argumentation = True の場合
~~~ bash
./out --x1=0.01 --x2=0.02 ... --x10=0.10 --trial_id=1 --config=config.yaml
~~~

- generic.enabled_variable_name_argumentation = False の場合
~~~ bash
./out 0.01 0.02 ... 0.10 1 config.yaml
~~~


- config example
~~~ yaml
generic:
  workspace: ./work
  job_command: ./a.out
  enabled_variable_name_argumentation: False

resource:
  type: local
  num_workers: 4

# ABCI:
#   group: your-group-id
#   job_execution_options: ""
#   job_script_preamble: |
#     #!/bin/bash

#     #$-l rt_C.small=1
#     #$-j y
#     #$-cwd

#     source /etc/profile.d/modules.sh
#     module load gcc/12.2.0
#     module load python/3.11

job_setting:
    job_timeout_seconds: 600
    max_failure_retries: 0
    trial_id_digits: 7

optimize:
  search_algorithm: aiaccel.optimizer.MOTpeOptimizer
  goal: [minimize, minimize]
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: x1
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x2
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x3
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x4
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x5
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x6
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x7
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x8
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x9
      type: uniform_float
      lower: 0.0
      upper: 1.0
    -
      name: x10
      type: uniform_float
      lower: 0.0
      upper: 1.0
~~~

