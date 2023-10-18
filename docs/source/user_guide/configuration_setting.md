# コンフィグレーションの設定ガイド (WIP)
## **generic:**
### workspace (str, optional):

aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
デフォルトでは "./work" に設定されています．

### job_command (str):
ユーザープログラムを実行するためのコマンドです．

### python_file (str, optional):
ローカル実行のモードの一つである python_local モードを用いる場合に，最適化対象の関数が実装されている python のファイルパスを指定します．
実行モードが ABCI または通常の Local の場合には指定する必要はありません．

### function (str, optional):
ローカル実行のモードの一つである python_local モードを用いる場合に，最適化対象の関数名を指定します．
aiaccel は実行時，python_file に書かれたファイルから，ここで指定された名前の関数をインポートします．
実行モードが ABCI または通常の Local の場合には指定する必要はありません．

### batch_job_timeout (int, optional):
ジョブのタイムアウト時間を秒単位で設定します．
デフォルトでは 600 (秒) に設定されています．

### sleep_time (float, optional):
最適化実行のメインループ 1 周あたりのスリープ時間を秒単位で指定します．
デフォルトでは 0.01 (秒) に設定されています．


<br>


## **resource:**

### type (str):
実行環境を指定します．
aiaccel は以下の 3 つの環境での実行をサポートしています．
- "abci" - ABCI 上で最適化を実行します．
- "local" - ローカル環境で最適化を実行します．
- "python_local" - ローカル環境で最適化を実行します．最適化対象の関数が python で実装されている必要がありますが，通常のローカル実行よりも高速に最適化を実行することが可能です．
デフォルトでは "local" に設定されています．

###  num_workers (int):
使用するノード数を指定します．
デフォルトでは 1 に設定されています．


<br>


## **ABCI:**

### group (str):
ユーザーが所属する ABCI のグループを指定します．

### job_script_preamble (str):
ABCI の設定を記述したシェルスクリプトのファイルを指定します．

### job_execution_options (str | list[str], optional):
aiaccel が ABCI の計算ノード上にジョブを投入する際に付加されるオプションのコマンドです．
デフォルトでは "" (空の文字列) が設定されています．


<br>


## **optimize:**
### search_algorithm (str, optional):
最適化アルゴリズムを設定します．
aiaccel では以下のアルゴリズムをサポートしています．
- "aiaccel.optimizer.NelderMeadOptimizer" (default) - Nelder-Mead 法でパラメータの探索を行います．
- "aiaccel.optimizer.BudgetSpecifiedOptimizer" - コンフィグで設定した試行回数 (`trial_number`) に応じて，探索空間を分割し，その中からパラメータを選択します．
- "aiaccel.optimizer.GridOptimizer" - 分割した探索空間からパラメータを選びます．
- "aiaccel.optimizer.RandomOptimizer" - パラメータをランダムに生成します．
- "aiaccel.optimizer.SobolOptimizer" - Sobol' 列を用いた準モンテカルロ的なサンプリングを行い，パラメータを生成します．
- "aiaccel.optimizer.TpeOptimizer" - ベイズ最適化を用いてパラメータの探索を行います．

デフォルトでは "aiaccel.optimizer.NelderMeadOptimizer" に設定されています．

### goal (str, optional):
最適化の向きを決定します．
- "minimize" - 目的関数が小さくなるようにパラメータを最適化します．
- "maximize" - 目的関数が大きくなるようにパラメータを最適化します．

デフォルトでは "minimize" に設定されています．

### trial_number (int):
試行回数を設定します．

### rand_seed (int, optional):
乱数生成に用いるシードを設定します．設定可能な値の範囲は `numpy.random.default_rng` が取り得る範囲に一致します．
デフォルトでは None に設定されています．

### sobol_scramble (bool, optional):
ソボルオプティマイザを使用する際に，[スクランブル](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html#scipy-stats-qmc-sobol)を使用するかを指定します．
デフォルトでは `true` に設定されています．

### grid_accept_small_trial_number (bool, optional):
`true` に設定すると，バジェット指定型グリッドオプティマイザを使用する際，生成されるグリッド点の数より指定した試行回数が少ない場合にも，強制的に最適化を実行します．`false` に設定した場合，十分な試行回数が設定されていなければ，aiaccel は最適化を行わずに，警告を発して終了します．デフォルトでは `false` に設定されています．

### parameters (list):
パラメータの探索条件をまとめたリストを設定します．
最適化アルゴルズムとパラメータのデータ型に応じて，各要素には以下の項目が含まれます．

- *name* - パラメータの名前を設定します．
- *type* - パラメータのデータ型を設定します．aiaccel は次のデータ型をサポートします．
  - "uniform_float" - 浮動小数点数型
  - "uniform_int" - 整数型
  - "categorical" - カテゴリカル型
  - "ordinal" - オーディナル型
- *lower* - パラメータの最小値を設定します．
- *upper* - パラメータの最大値を設定します．
- *initial* - パラメータの初期値を設定します．
- *step* - パラメータの分解能を設定します．
- *log* - パラメータの探索空間を対数スケールで分割するかを指定します．対数スケールを使用する場合は `true` を，使用しない場合は `false` を設定します．
- *base* - 対数スケールでパラメータの探索空間を分割する場合の対数の基数を指定します．
- *choices* - データ型が categorical の場合に，選択肢のリストを設定します．
- *sequence* - データ型が ordinal の場合に，選択肢のリストを設定します．
- *comment* - コメントを設定します．
- *num_grid_points* - バジェット指定型グリッドオプティマイザのみで使用します．パラメータを何通り考慮するかを指定します．例えば，`lower = 0.0`，`upper = 1.0` のときに `num_grid_points = 5` であれば，対象のパラメータは `0.0, 0.25, 0.50, 0.75, 1.00` の中から選択されます．
<br>


それぞれのアルゴリズムとデータ型で必要なパラメータは以下の通りです．

**Nelder-Mead 法 ("aiaccel.optimizer.NelderMeadOptimizer")**

設定可能なデータ型は "uniform_float"，"uniform_int"，および "ordinal" です．

***"uniform_float" または "uniform_int" の場合***
- *name*
- *type ("uniform_float", "uniform_int")*
- *lower*
- *upper*
- *initial* - 要素数が **パラメータ数 + 1** の配列を設定します．initial の項目が存在しない場合，aiaccel はランダムに初期値の配列を設定します．また，設定された配列の要素数が **パラメータ数 + 1** より少ない場合，aiaccel は足りない初期値をランダムに生成し補います．

***"ordinal" の場合***
- *name*
- *type ("ordinal")*
- *sequence* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．
- *initial* - 要素数が **パラメータ数 + 1** の配列を設定します．initial の項目が存在しない場合，aiaccel はランダムに初期値の配列を設定します．また，設定された配列の要素数が **パラメータ数 + 1** より少ない場合，aiaccel は足りない初期値をランダムに生成し補います．

**バジェット指定型グリッドオプティマイザ (aiaccel.optimizer.BudgetSpecifiedOptimizer)**

設定可能なデータ型は "uniform_float"，"uniform_int"，"categorical"，および "ordinal" です．

***"uniform_float" または "uniform_int" の場合***
- *name*
- *type ("uniform_float", "uniform_int")*
- *lower*
- *upper*
- *log* - (optional) 対数スケールでグリッド点を指定したい場合には `true` を設定します．この値が `true` のときは，`lower` と `upper` の値の符号が一致し，かつ何れも 0 ではない必要があります．指定されなかった場合，デフォルトでは `false` として扱われます．
- *num_grid_points* - (optional) パラメータを何通り考慮するかを指定します．例えば，`lower = 0.0`，`upper = 1.0` のときに `num_grid_points = 5` であれば，対象のパラメータは `0.0, 0.25, 0.50, 0.75, 1.00` の中から選択されます．設定されていない場合，他のパラメータの設定と試行回数から，自動で値が設定されます．

***"categorical" の場合***
- *name*
- *type ("categorical")*
- *choices* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．

***"ordinal" の場合***
- *name*
- *type ("ordinal")*
- *sequence* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．

**グリッドオプティマイザ ("aiaccel.optimizer.GridOptimizer")**

設定可能なデータ型は "uniform_float", "uniform_int", "categorical", および "ordinal" です．
データ型に依らず，初期値は設定できません．

***"uniform_float" または "uniform_int" の場合***
- *name*
- *type ("uniform_float", "uniform_int")*
- *lower*
- *upper*
- *step*
- *log*
- *base*

(注意) `log` が `true` の場合，`lower`，`upper`，および `step` は対数スケールでの値として参照されます．
即ち，探索の下限は ${base}^{lower}$，上限は ${base}^{upper}$ と解釈され， $n\ (=0, 1, \cdots)$ 番目の点は ${base}^{lower} {base}^{n \times step}$ で与えられます．
一方で `log` が `false` の場合，`lower`，`upper`，および `step` は，それぞれ探索の下限，上限，およびステップに直接対応します．
この場合，`base` の値は使用されませんが，何も値を設定していないとエラーが生じます．

***"categorical" の場合***
- *name*
- *type ("categorical")*
- *choices* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．

***"ordinal" の場合***
- *name*
- *type ("ordinal")*
- *sequence* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．

**ランダムオプティマイザ ("aiaccel.optimizer.RandomOptimizer")**

設定可能なデータ型は "uniform_float"，"uniform_int"，"categorical"，および "ordinal" です．

***"uniform_float" または "uniform_int" の場合***
- *name*
- *type ("uniform_float", "uniform_int")*
- *lower*
- *upper*
- *initial*

***"categorical" の場合***
- *name*
- *type ("categorical")*
- *choices* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．
- *initial*

***"ordinal" の場合***
- *name*
- *type ("ordinal")*
- *sequence* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．
- *initial*

**ソボルオプティマイザ ("aiaccel.optimizer.SobolOptimizer")**

設定可能なデータ型は "uniform_float" と "uniform_int" です．
データ型に依らず，初期値は設定できません．
- *name*
- *type ("uniform_float", "uniform_int")*
- *lower*
- *upper*

**TPE オプティマイザ ("aiaccel.optimizer.TpeOptimizer")**

設定可能なデータ型は "uniform_float", "uniform_int", "categorical", および "ordinal" です．

***"uniform_float" または "uniform_int" の場合***
- *name*
- *type ("unform_float", "uniform_int")*
- *lower*
- *upper*
- *initial*
- *log*

***"categorical" の場合***
- *name*
- *type ("categorical")*
- *choices* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．
- *initial*

***"ordinal" の場合***
- *name*
- *type ("ordinal")*
- *sequence* - 選択肢の配列を設定します．配列の要素は float, int, または str 型です．
- *initial*

<br>


## **job_setting:**
### cancel_retry (int, optional):
Max retry counts to transit the state from HpCancelFailed to HpCancelFailure.
Defaults to 3.

### cancel_timeout (int, optional): 
Timeout seconds to transit the state from HpCancelChecking to HpCancelFailed.
Defaults to 60.

### expire_retry (int, optional):
Max retry counts to transit the state from HpExpireFailed to HpExpireFailure.
Defaults to 3.

### expire_timeout (int, optional):
Timeout seconds to transit the state from HpExpireChecking to HpExpireFailed.
Defaults to 60.

### finished_retry (int, optional):
Max retry counts to transit the state from HpFinishedFailed to HpFinishedFailure.
Defaults to 3.

### finished_timeout (int, optional):
Timeout seconds to transit the state from HpFinishedChecking to HpFinishedFailed.
Defaults to 60.


### job_retry (int, optional):
Max retry counts to transit the state from HpCancelFailed to HpCancelFailure.
Defaults to 2.

### job_timeout (int, optional):
Timeout seconds to transit the state from JobChecking to JobFailed.
Defaults to 60.

### kill_retry (int, optional):
Max retry counts to transit the state from KillFailed to KillFailure.
Defaults to 3.

### kill_timeout (int, optional):
Timeout seconds to transit the state from KillChecking to KillFailed.
Defaults to 60.

### result_retry (int, optional):
Max retry counts to transit the state from RunnerFailed to RunnerFailure.
Defaults to 1.

### runner_retry (int, optional): 
Max retry counts to transit the state from RunnerFailed to RunnerFailure.
Defaults to 3.

### runner_timeout (int, optional):
Timeout seconds to transit the state from RunnerChecking to RunnerFailed.
Defaults to 60.

### running_retry (int, optional):
Max retry counts to transit the state from HpRunningFailed to HpRunningFailure.
Defaults to 3.

### running_timeout (int, optional):
Timeout seconds to transit the state from HpRunningChecking to HpRunningFailed.
Defaults to 60.

### init_fail_count (int, optional):
Defaults to 100.

### name_length (int, optional):
文字列としてのジョブ ID の長さです．
この文字列は，結果を .hp ファイルに保存する際にファイル名として使用されます．
デフォルトでは 6 に設定されています．


<br>

## **logger:**


### file:
実行ログの保存先を設定します．


#### optimizer (str, optional):
オプティマイザモジュールのログの保存先を設定します．
デフォルトでは "optimizer.log" に設定されています．

#### scheduler (str, optional):
スケジューラモジュールのログの保存先を設定します．
デフォルトでは "scheduler.log" に設定されています．

### log_level:

#### optimizer (str, optional):
オプティマイザモジュールからのログファイル出力のログレベルを設定します．
デフォルトでは "DEBUG" に設定されています．

A logging level for a log file output of optimizer module.
Defaults to "DEBUG".

Following strings are available;
- 'DEBUG'
- 'INFO'
- 'WARNING'
- 'WARN'
- 'ERROR'
- 'CRITICAL

#### scheduler (str, optional):
オプティマイザモジュールからのログファイル出力のログレベルを設定します．
デフォルトでは "DEBUG" に設定されています．

A logging level for a log file output of scheduler module.
Defaults to "DEBUG".

### stream_level:
#### optimizer (str, optional):
オプティマイザモジュールからのストリーム出力のログレベルを設定します．
デフォルトでは "DEBUG" に設定されています．

A logging level for a stream output of optimizer module.
Defaults to "DEBUG".

#### scheduler (str, optional):
スケジューラモジュールからのストリーム出力のログレベルを設定します．
デフォルトでは "DEBUG" に設定されています．

A logging level for a stream output of scheduler module.
Defaults to "DEBUG".
