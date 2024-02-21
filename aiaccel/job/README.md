# aiaccel V2

aiaccel v2 のジョブディスパッチャーの設計について記述する．
- コード: https://github.com/aistairc/aiaccel/tree/feature/v2-draft-jobdispatcher
- commit: 486548f

目次
1. [ユーザープログラムの例](#1-ユーザープログラムの例)
2. [目的関数](#2-目的関数)
3. [ジョブファイル](#3-ジョブファイル)
4. [JobCreatorクラスの実装](#4-jobcreatorクラスの実装)
5. [JobDispatcherクラスの実装](#5-jobdispatcherクラスの実装)


## 1. ユーザープログラムの例

- user_program.py
    ~~~ python
    import optuna

    from aiaccel import JobDispatcher


    def objective(hparams: dict) -> float:
        x1 = hparams["x1"]
        x2 = hparams["x2"]
        return (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)


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
        return " ".join([f"{k}={v}" for k, v in param.items()])


    if __name__ == "__main__":
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        n_trials = 50
        n_jobs = 4
        jobs = JobDispatcher(
            objective,
            n_trials,
            n_jobs=n_jobs,
            param_to_args_fn=param_to_args_fn
        )

        for n in range(n_trials):
            trial = study.ask()
            hparams = {
                "x1": trial.suggest_float("x", 0, 10),
                "x2": trial.suggest_float("x", 0, 10),
            }

            jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}") # ジョブプールが空かないと帰ってこない

            # y = jobs.result()  # n_jobs = 1 の場合
            # study.tell(trial, y)

            for y, trial in jobs.collect_results():  # n_jobs > 1 の場合
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
            objective,
            n_trials,
            n_jobs=n_jobs,
            param_to_args_fn=aiaccel.parser.param_to_args_fn
        )
        ~~~



### 1.1 逐次実行

逐次実行の例を以下に示す．逐次実行では，JobDispatcherの引数 `n_jobs` に 1 を指定し，`jobs.result()` メソッドを使ってジョブの結果を取得する．<br>

~~~ python

...(省略)...

if __name__ == "__main__":

    ...(省略)...

    n_jobs = 1
    jobs = JobDispatcher(
        objective,
        n_trials,
        n_jobs=n_jobs,
        param_to_args_fn=param_to_args_fn
    )

    for n in range(n_trials):

        ...(中略)...

        y = jobs.result()
        study.tell(trial, y)
~~~


### 1.2 並列実行

並列実行の例を以下に示す．並列実行では，JobDispatcherの引数 `n_jobs` に 1以上の値を指定し，`jobs.collect_results()` メソッドを使ってジョブの結果を取得する．<br>

~~~ python

...(省略)...

if __name__ == "__main__":

    ...(省略)...

    n_jobs = 4
    jobs = JobDispatcher(
        objective,
        n_trials,
        n_jobs=n_jobs,
        param_to_args_fn=param_to_args_fn
    )

    for n in range(n_trials):

        ...(中略)...

        for y, trial in jobs.collect_results():
            study.tell(trial, y)
~~~


<br>
<br>

## 2. 目的関数
目的関数は，1. ユーザープログラム内に記述する場合(Pythonコードとして記述)，2. 外部ファイルとして記述する場合 の2種類を想定.<br>
外部ファイルとは，例えば，FortranやC言語で記述されたプログラムを指す．

以下に，`Python` の目的関数の例と，`Fortran` の目的関数の例を示す．<br>


### 2.1 python の場合

~~~ python
def objective(hparams: dict) -> float:
    ...
    return y
~~~

- パラメータは辞書形式で目的関数の引数として渡す
- `aiaccel` に渡す値を `return` する


### 2.2 Python以外

`aiaccel` で `Python` 以外のプログラムを最適化する場合，以下の機能を持つプログラムを作成する．
- パラメータはコマンドライン引数で渡される．
- `aiaccel` に渡す値を標準出力に出力する．

例として `Fortran 95` のコードを以下に示す.

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


#### 2.2.1 ユーザープログラムの例

`Fortran 95` プログラムを最適化させる場合のユーザープログラムの例を示す．<br>
事前に，`a.out` という名前でコンパイルされたプログラムがあると仮定する．

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
    return "0.5 0.3 ..."
    """
    return " ".join([f"{v}" for k, v in param.items()])


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    # sampler = optuna.samplers.RandomSampler(seed=42)

    study = optuna.create_study(direction="minimize", sampler=sampler)

    n_trials = 50
    n_jobs = 4

    jobs = JobDispatcher(
        "./a.out", n_trials, n_jobs=n_jobs, param_to_args_fn=param_to_args_fn
    )

    for n in range(n_trials):
        trial = study.ask()
        hparams = {
            "x1": trial.suggest_float("x", 0, 10),
            "x2": trial.suggest_float("x", 0, 10),
        }

        jobs.submit(hparams, tag=trial, job_name=f"hpo-{n:04}")  # ジョブプールが空かないと帰ってこない

        # y = jobs.result()  # n_jobs = 1 の場合
        # study.tell(trial, y)

        for y, trial in jobs.collect_results():  # n_jobs > 1 の場合
            study.tell(trial, y)
~~~
"`Python` の目的関数を最適化する場合" との違いは，
1. `param_to_args_fn`をFotran向けのコマンドライン引数に変換する関数に変更
2. `JobDispatcher`の初期化時に，`"./a.out"` を目的関数として渡す

<br>
<br>

## 3. ジョブファイル

ジョブファイル(*.sh)は，`aiaccel` がジョブごとに生成し，実行するスクリプトファイルである．

### 3.1 ジョブファイルの例
- python を実行する場合
    ``` bash
    #!/bin/bash

    # モジュールのロード処理など

    python user_program.py -e --params $@
    ```
    - --params 移行の引数はハイパパラメータを表す．(ジョブファイルに対するコマンドライン引数をそのまま渡す)
    - -e オプションは, user_program.py がジョブファイルから実行されたことを示す．-e で実行された場合は，ユーザープログラムで指定したobjective関数を実行して終了する．

        **[補足]**

        ~~~ bash
        $ python user_program.py
        ~~~
        と実行した場合は，最適化処理を実行する．

        ~~~ bash
        $ python user_program.py -e --params x1=0.5 x2=0.3
        ~~~
        と実行した場合は，ユーザープログラムで指定したobjective関数のみを実行し，最適化は行わない．

    **[補足]**

    `-e` オプションと `--params` オプションは，最適化対象が `Python` プログラムの場合にのみ付加する.

- Fortran を実行する場合
    ``` bash
    #!/bin/bash

    # モジュールのロード処理など

    ./a.out $@
    ```

    **[補足]**

    `Python`の事例で挙げたようなオプションは不要．

### 3.2 ジョブファイルの実行
ジョブファイルは，`aiaccel` が `subprocess` モジュールを使って実行する．

- `aiaccel` によるジョブファイルの実行コマンドの例
    ~~~ bash
    $ bash job-0001.sh x1=0.5 x2=0.3
    ~~~


<br>
<br>


## 4. JobCreatorクラスの実装

`JobCreator`は，ジョブの作成，実行，結果の収集を行うクラスである．`JobCreator`は，`JobDispatcher`によって生成され，`JobDispatcher`によって管理される．<br>
`JobCreator`のインスタンスは，`JobDispatcher`がジョブごとに生成する．

主な機能は以下の通りである．
- ジョブファイルの作成
- ジョブの実行
- ジョブの結果の収集
- ジョブの結果をjsonファイルに保存
- ジョブの終了判定
- ジョブの合否判定(エラーの有無)

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

## 5. JobDispatcherクラスの実装

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

~~~ python
class JobDispatcher:
    """
    ジョブのディスパッチと管理を行うクラス
    """
    def __init__(self, func: Callable | str, n_trials: int, ...):
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
