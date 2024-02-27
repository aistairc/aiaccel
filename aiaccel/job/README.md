# aiaccel V2: JobDispatcherの設計

aiaccel v2 のジョブディスパッチャーの使用方法と設計について記述する．
- コード: https://github.com/aistairc/aiaccel/tree/feature/v2-draft-jobdispatcher
- branch: feature/v2-draft-jobdispatcher


- `v2-draft-jobdispatcher`の`example`の実行手順
    - インストール
    ~~~ bash
    $ git clone git@github.com:aistairc/aiaccel.git
    $ cd aiaccel
    $ git checkout feature/v2-draft-jobdispatcher
    $ pip install -e .
    ~~~

    - exampleの実行
    ~~~ bash
    $ cd aiaccel/job
    $ python user_program.py
    ~~~

    - 実行結果
    ~~~ bash
    $ python user_program.py
    [I 2024-02-26 12:34:41,331] A new study created in memory with name: no-name-9dd39abe-26df-4e13-896a-1e1089063359
    Running the job with the command: `bash /home/user/aiaccel/aiaccel/job/exsmple/work/hpo-0000.sh x1=3.745401188473625 x2=3.745401188473625`
    {'job_name': 'hpo-0000', 'value': -4.698975879748483, 'hparams': {'x1': 3.745401188473625, 'x2': 3.745401188473625}}
    Running the job with the command: `bash /home/user/aiaccel/aiaccel/job/exsmple/work/hpo-0001.sh x1=9.50714306409916 x2=9.50714306409916`
    {'job_name': 'hpo-0001', 'value': 42.850053920752984, 'hparams': {'x1': 9.50714306409916, 'x2': 9.50714306409916}}
    Running the job with the command: `bash /home/user/aiaccel/aiaccel/job/exsmple/work/hpo-0002.sh x1=7.319939418114051 x2=7.319939418114051`
    {'job_name': 'hpo-0002', 'value': 16.98181599428962, 'hparams': {'x1': 7.319939418114051, 'x2': 7.319939418114051}}
    ...
    ~~~

## 目次
1. [はじめに](#1-はじめに)
2. [使い方](#2-使い方)
    1. [オブジェクティブファイル (目的関数) の作成](#21-オブジェクティブファイル-目的関数-の作成)
    2. [ジョブスクリプト の作成](#22-ジョブスクリプト-の作成)
    3. [ユーザープログラム の作成](#23-ユーザープログラム-の作成)
    4. [実行](#24-実行)
3. [JobCreatorクラスの実装](#3-jobcreatorクラスの実装)
4. [JobDispatcherクラスの実装](#4-jobdispatcherクラスの実装)
5. [改定履歴](#改定履歴)


## 1. はじめに

ユーザーは，`オブジェクティブファイル`, `ジョブファイル`, `ユーザープログラム`を作成する．<br>

- オブジェクティブファイル: 最適化する目的関数を記述したファイル
- ジョブファイル: ジョブを実行するためのスクリプトファイル(ABCIに投入するバッチファイル)
- ユーザープログラム: 最適化を行うためのプログラム


## 2. 使い方

### 2.1 オブジェクティブファイル (目的関数) の作成
オブジェクティブファイルは，最適化する目的関数を記述したファイルを指す．<br>
オブジェクティブファイルは，コマンドライン引数でハイパーパラメーターを受け取り，標準出力に結果を出力する．<br>

以下に，`Python` のオブジェクティブファイルの例と，`Fortran` のオブジェクティブファイルの例を示す．<br>

- `Python` のオブジェクティブファイルの例
    ~~~ python
    from argparse import ArgumentParser


    def func(hparams: dict) -> float:
        x1 = hparams["x1"]
        x2 = hparams["x2"]
        return (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)


    if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--x1", type=float)
        parser.add_argument("--x2", type=float)
        args = parser.parse_args()

        hparams = {
            "x1": args.x1,
            "x2": args.x2,
        }

        print(func(hparams))
    ~~~


- `Fortran95` のオブジェクティブファイルの例
    ~~~ Fortran 95
    program objective
        implicit none
        real :: arg1, arg2, result
        character(len=32) :: arg1_str, arg2_str

        ! Get the input arguments from the command line
        call get_command_argument(1, arg1_str)
        call get_command_argument(2, arg2_str)

        ! Convert the input arguments from string to real
        read(arg1_str, *) arg1
        read(arg2_str, *) arg2

        ! Call the function to optimize
        result = function_to_optimize(arg1, arg2)
        write(*, '(F0.16)') result

    contains

        real function function_to_optimize(x1, x2)
            real, intent(in) :: x1, x2
            function_to_optimize = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        end function function_to_optimize

    end program objective
    ~~~

**[注意]**
- `aiaccel` は，標準出力に出力された値を取得する．必ず，数値のみを出力するようにする．

<br>

### 2.2 ジョブスクリプト の作成

ジョブスクリプト(*.sh)は，オブジェクティブファイルを実行するためのスクリプトファイルを指す．<br>
ジョブスクリプトは，`ABCI` で実行するためのスクリプトファイルである．<br>

- job.sh

    ~~~ bash
    #!/bin/bash

    #$-l rt_C.small=1
    #$-cwd

    source /etc/profile.d/modules.sh
    module load gcc/12.2.0
    module load python/3.10/3.10.10

    python objective.py $@
    ~~~

<br>

### 2.3 ユーザープログラム の作成
ユーザープログラムは，最適化を行うためのプログラムを指す．<br>
ユーザープログラムは，`Python` で記述されたプログラムである．<br>

- user_program.py
    ~~~ python
    import optuna

    from aiaccel import JobDispatcher


    def param_to_args_fn(param: dict) -> str:
        """
        Example:
        param = {
            'x': 0.5,
            'y': 0.3,
            ...
        }
        return "x=0.5 y=0.3 ..."
        """
        return " ".join([f"--{k}={v}" for k, v in param.items()])


    if __name__ == "__main__":
        sampler = optuna.samplers.TPESampler(seed=42)
        # sampler = optuna.samplers.RandomSampler(seed=42)

        study = optuna.create_study(direction="minimize", sampler=sampler)

        n_trials = 50
        n_jobs = 4

        jobs = JobDispatcher("job.sh", n_jobs=n_jobs, param_to_args_fn=param_to_args_fn)

        for n in range(n_trials):
            trial = study.ask()
            hparams = {
                "x1": trial.suggest_float("x", 0, 10),
                "x2": trial.suggest_float("x", 0, 10),
            }

            jobs.submit(hparams, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

            y = jobs.result()
            study.tell(trial, y)
    ~~~

    - objective: 目的関数
    - param_to_args_fn: ハイパーパラメーターをコマンドライン引数に変換する関数 (任意)


    ### Memo

    上記例では `param_to_args_fn` をユーザープログラム内で書いているが，幾つかのパターンを `aiaccel` の組み込み関数として提供するのも良いかもしれない.<br>
    その場合は，`param_to_args_fn=aiaccel.parser.param_to_args_fn`のように記述する．

    - 例：
        ~~~ python
        jobs = JobDispatcher(
            ...,
            param_to_args_fn=aiaccel.parser.param_to_args_fn
        )
        ~~~

<br>

#### 2.3.1 逐次実行

逐次実行の例を以下に示す．逐次実行では，JobDispatcherの引数 `n_jobs` に 1 を指定し，`jobs.result()` メソッドを使ってジョブの結果を取得する．<br>

~~~ python

...(省略)...

if __name__ == "__main__":

    ...(省略)...

    n_jobs = 1
    jobs = JobDispatcher(
        "job.sh",
        n_jobs=n_jobs,
        param_to_args_fn=param_to_args_fn
    )

    for n in range(n_trials):
        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")

        y = jobs.result()
        study.tell(trial, y)
~~~

<br>

#### 2.3.2 並列実行

並列実行の例を以下に示す．並列実行では，JobDispatcherの引数 `n_jobs` に 1以上の値を指定し，`jobs.collect_results()` メソッドを使ってジョブの結果を取得する．<br>

~~~ python

...(省略)...

if __name__ == "__main__":

    ...(省略)...

    n_jobs = 4
    jobs = JobDispatcher(
        "job.sh",
        n_jobs=n_jobs,
        param_to_args_fn=param_to_args_fn
    )

    n = 0
    while True:
        if jobs.finished_job_count >= n_trials:
            break

        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")

        for y, trial in jobs.collect_results():
            study.tell(trial, y)
        n += 1
~~~

<br>

### 2.4 実行
- オブジェクティブファイル: `objective.py`
- ジョブスクリプト: `job.sh`
- ユーザープログラム: `user_program.py`
の3つのファイルを作成し，`user_program.py` を実行する．<br>

~~~ bash
$ python user_program.py
~~~

<br>


## 3. JobCreatorクラスの実装

`JobCreator`は，ジョブの作成，実行，結果の収集を行うクラスである．`JobCreator`は，`JobDispatcher`によって生成され，`JobDispatcher`によって管理される．<br>
`JobCreator`のインスタンスは，`JobDispatcher`がジョブごとに生成する．

主な機能は以下の通りである．
- ジョブファイルの作成
- ジョブの実行
- ジョブの結果の収集
- ジョブの結果をjsonファイルに保存
- ジョブの終了判定
- ジョブの合否判定(エラーの有無)

コード: https://github.com/aistairc/aiaccel/blob/feature/v2-draft-jobdispatcher/aiaccel/job/job_creator.py
~~~ python
class JobCreator:
    def __init__(...):
        ...

    def create(self) -> None:
        """
        ジョブファイルを作成
        """
        ...

    def run(self, hparams_str: str) -> None:
        """
        ジョブを実行

        入力:
            hparams_str: str - ハイパーパラメーターを表す文字列 (e.g. "x1=0.5 x2=0.3")
        """
        ...

    def collect_result(self) -> str | None:
        """
        ジョブの結果を収集

        出力:
            str | None - ジョブの結果
        """
        ...

    def create_result_json(self, result: dict) -> None:
        """
        ジョブの結果をjsonファイルに保存

        入力:
            result: dict - ジョブの結果
        """
        ...

    def is_finished(self) -> bool:
        """
        ジョブが終了したかどうかを判定

        出力:
            bool - ジョブが終了している場合はTrue、そうでない場合はFalse
        """
        ...

    def is_error_free(self) -> bool:
        """
        ジョブがエラーフリーかどうかを判定

        出力:
            bool - ジョブがエラー無しの場合はTrue、そうでない場合はFalse
        """
        ...
~~~

<br>
<br>

## 4. JobDispatcherクラスの実装



`JobDispatcher`は，ジョブの投入と管理を行うクラスである．`JobDispatcher`は，利用可能なワーカー数を管理し，`JobCreator`でジョブを作成，実行，結果の収集を行う．

主な機能は以下の通りである．
- ジョブの投入
- ジョブの結果の収集 (並列実行の場合)
- ジョブの結果の取得 (逐次実行の場合)
- 利用可能なワーカー数の管理
- 完了したジョブ数の管理
- 投入したジョブ数の管理
- 全てのジョブが完了したかどうかの判定
- 全てのジョブの結果の取得

コード: https://github.com/aistairc/aiaccel/blob/feature/v2-draft-jobdispatcher/aiaccel/job/dispatcher.py

~~~ python
class JobDispatcher:
    """
    ジョブのディスパッチと管理を行うクラス
    """
    def __init__(self, func: Callable | str, n_jobs: int, param_to_args_fn: Callable | None = None, ...):
        ...

    def submit(self, hparams: dict, tag: Any = None, job_name: int | None = None) -> None:
        """
        ジョブをディスパッチするメソッド

        入力:
            hparams: dict - ハイパーパラメーターのセット
            tag: Any - ジョブのタグ（任意）
            job_name: int | None - ジョブ名（任意、デフォルトはNone）
        """
        ...

    def collect_results(self) -> list[tuple[float, Any]]:
        """
        結果を収集するメソッド

        出力:
            list[tuple[float, Any]] - 目的関数の結果とそれに対応するタグのリスト
        """
        ...

    def result(self) -> Any:
        """
        ジョブの結果を取得するメソッド. n_jobs=1の場合で使用する

        出力:
            Any - ジョブの結果
        """
        ...

    @property
    def results(self) -> list[dict]:
        """
        全ての結果を取得するプロパティ

        出力:
            list[dict] - 全てのジョブの結果のリスト
        """
        ...

    @property
    def available_worker_count(self) -> int:
        """
        利用可能なワーカー数を取得するプロパティ

        出力:
            int - 利用可能なワーカー数
        """
        ...

    @property
    def finished_job_count(self) -> int:
        """
        完了したジョブ数を取得するプロパティ

        出力:
            int - 完了したジョブ数
        """
        ...

    @property
    def submit_job_count(self) -> int:
        """
        投入したジョブ数を取得するプロパティ

        出力:
            int - 投入済みのジョブ数
        """
        ...

    def all_done(self) -> bool:
        """
        全てのジョブが完了したかどうかを判定するメソッド

        出力:
            bool - 全てのジョブが完了している場合はTrue、そうでない場合はFalse
        """
        ...
~~~

<br>
<br>

## 改定履歴

ver | 改定内容 | 日付
--- | -------- | ----
初版 | - | 2024-02-27
