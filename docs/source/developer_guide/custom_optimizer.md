# カスタムオプティマイザー作成ガイド

オプティマイザは，ハイパーパラメータを生成する機能を担うためどのハイパーパラメータの組を選択するかのアルゴリズムにより，最適化のパフォーマンスは大きく変わってきます．
本稿では，ユーザーがaiaccelを用いた独自のオプティマイザを開発するための方法について解説します．

## カスタムオプティマイザの実行確認

それでは，まずカスタムオプティマイザを実行するための動作を確かめてみましょう．
所謂Hello, Worldの位置づけです．
ここで解説する内容は簡単にまとめると下記のようになります．

- カスタムオプティマイザのソースファイルを作成する
- カスタムオプティマイザのソースファイルをaiaccelが読み込めるようにする
- コンフィグレーションファイルを編集し，カスタムオプティマイザを実行できるようにする

カスタムオプティマイザをどのように実行するかは環境に依存しますが，今回は自分で作成したワークスペースディレクトリに追加して実行してみます．
その他にはソースコードに追加するなど，環境次第でやり方は異なりますので各自の環境に合わせて設定してください．

1. 開発環境の確認

まず環境の確認からします．
本ガイドでは，各種インストールは終了しaiaccelが実行できる状態から始めます．
インストールが未だの方はインストールガイドを参照してください．
各ディレクトリ・ファイルは以下のとおりとします．
皆さんの環境に読み替えて参考にしてください．

- aiaccelのソースディレクトリ: /workspace/aiaccel
- ワークスペースディレクトリ: /workspace/aiaccel/work
- 例えばランダムオプティマイザのファイル: /workspace/aiaccel/aiaccel/optimizer/random_optimizer.py

2. カスタムオプティマイザファイルの作成

カスタムオプティマイザのソースファイルを作成します．
今回はランダムオプティマイザをコピーします．

~~~bash
> pwd
/workspace/aiaccel

> mkdir -p work/lib/my_optimizer

> cp aiaccel/optimizer/random_optimizer.py work/lib/my_optimizer/custom_optimizer.py

~~~

これでcustom_optimizer.pyが作成されました．

3. ファイルの編集

このままではcustom_optimizer.pyの内容はランダムオプティマイザと同じですので，少しだけ編集します．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```diff
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


- class RandomOptimizer(AbstractOptimizer):
+ class CustomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

```

上記はdiff表記で削除された行頭に - 追加された行頭に + が付いています．
クラス名をRandomOptimizerからCustomOptimizerに変更しました．
変更したファイルは保存します．

4. パスの設定

今custom_optimizer.pyにはランダムオプティマイザを全く同じ動作をするCustomOptimizerを定義しました．
aiaccelから作成したCustomOptimizerクラスを実行する必要があるので，まず__init__.pyファイルを作成します．
ここではaiaccel/optimizerにある__init__.pyを編集して利用します．

~~~bash
> pwd
/workspace/aiaccel

> cp aiaccel/optimizer/__init__.py work/lib/my_optimizer/

~~~

追加したCustomOptimizerを読み込むように編集します．

***/workspace/aiaccel/work/lib/my_optimizer/__init__.py***

```diff
- from .grid_optimizer import GridOptimizer
- from .nelder_mead_optimizer import NelderMeadOptimizer
- from .random_optimizer import RandomOptimizer
- from .sobol_optimizer import SobolOptimizer
- from .tpe_optimizer import TpeOptimizer
+ from .custom_optimizer import CustomOptimizer

__all__ = [
-    GridOptimizer,
-    RandomOptimizer,
-    SobolOptimizer,
-    NelderMeadOptimizer,
-    TpeOptimizer
+    CustomOptimizer
]

```

次にパスの設定を行います．
PYTHONPATHに，aiaccelと追加したcustom_optimizer.pyのディレクトリを追加します．

~~~bash
> echo $PYTHONPATH

> export PYTHONPATH=/workspace/aiaccel:/workspace/aiaccel/work/lib

> echo $PYTHONPATH
/workspace/aiaccel:/workspace/aiaccel/work/lib

~~~

5. ユーザーファイルの作成

カスタムオプティマイザを作成したので，実際に実行するユーザーファイルを作成します．
今回は/workspace/aiaccel/examples/sphereディレクトリをコピーして作成します．

~~~bash
> pwd
/workspace/aiaccel

> cp -R examples/sphere work/

> cd work/sphere

> pwd
/workspace/aiaccel/work/sphere

> ls
config.yaml       job_script_preamble.sh     user.py

~~~

examples/sphereディレクトリをコピーし，sphereディレクトリに移動しました．
次にコンフィグレーションファイルを編集します．
オプティマイザに今回作成したカスタムオプティマイザを利用したいのでconfig.yamlを編集します．

***/workspace/aiaccel/work/sphere/config.yaml***

```diff
-  search_algorithm: "aiaccel.optimizer.NelderMeadOptimizer"
+  search_algorithm: "my_optimizer.CustomOptimizer"
```

デフォルトのconfig.yamlファイルにはネルダーミードの初期値がリストで設定されているため，これは削除します．

***/workspace/aiaccel/work/sphere/config.yaml***

```yaml:config.yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 600

resource:
  type: "local"
  num_node: 4

ABCI:
  group: "[group]"
  job_script_preamble: "./job_script_preamble.sh"
  job_execution_options: ""

optimize:
  search_algorithm: "my_optimizer.CustomOptimizer"
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
    -
      name: "x2"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
    -
      name: "x3"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
    -
      name: "x4"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
    -
      name: "x5"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0

```

これでコンフィグレーションファイルの編集は一旦終了です．
編集したファイルを保存します．

6. 実行の確認

それでは現在のディレクトリで実行してみます．

~~~bash
> pwd
/workspace/aiaccel/work/sphere

> aiaccel-start --config config.yaml --clean

~~~

正常に実行できれば成功です．
このカスタムオプティマイザの中身はランダムオプティマイザと同じなので，ランダムにハイパーパラメータが選択されます．

ここまででカスタムオプティマイザが実行できることが確認できたので，次節ではオプティマイザのソースコードを編集して実行する点を解説します．

## カスタムオプティマイザの編集

前節でカスタムオプティマイザの実行確認を行いました．
本節では，前節で作成したカスタムオプティマイザを編集しシンプルなアルゴリズムを実装します．
簡単のため前節で作成したワークスペースを流用し５つのfloat型のハイパーパラメータに対し正規分布でハイパーパラメータを生成するオプティマイザを作成してみましょう．

既存のオプティマイザには，ランダム・ソボル列・グリッド・ネルダーミード・TPEがサポートされていますが，ランダムオプティマイザをコピーしたソースファイルから編集を始めます．

1. ランダムオプティマイザの確認

前節でコピーしたカスタムオプティマイザのファイルを見てみましょう．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```python:custom_optimizer.py
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class CustomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        new_params = []
        sample = self.params.sample()

        for s in sample:
            new_param = {
                'parameter_name': s['name'],
                'type': s['type'],
                'value': s['value']
            }
            new_params.append(new_param)

        return new_params

```

CustomOptimizerクラスはAbstractOptimizerを継承し，generate_parameterメソッドのみを実装しています．
generate_parameter以外のオプティマイザとしての機能はAbstractOptimizerに実装されているので，generate_parameterメソッドを実装すれば簡単なオプティマイザなら実装することができます．

2. ハイパーパラメータのソースコードの確認

generate_parameterメソッドを見てみましょう．
self.params.sampleというメソッドを実行しています．
このメソッドは，aiaccel/parameter.pyのHyperParameterConfigurationインスタンスであるself.paramsのsampleメソッドです．
sampleメソッド内では，さらにHyperParameterインスタンスであるvalueから更にsampleメソッドが呼ばれています．
この２度目に呼ばれたsampleメソッドはHyperParameterクラスのメソッドであり，中身を見てみるとハイパーパラメータのタイプごとに処理が分かれていますが，例えばFLOAT型の場合np.random.uniformが実行されます．

***/workspace/aiaccel/aiaccel/parameter.py***

```python:aiaccel/parameter.py
        elif self.type == 'FLOAT':
            value = np.random.uniform(self.lower, self.upper)
```

こうして生成されたランダムなハイパーパラメータを返すことがgenerate_parameterメソッドの役割となります．

3. 正規分布オプティマイザの作成

ではaiaccel/parameter.pyのHyperParameterConfigurationクラスをもう少し詳しく見てみましょう．
sampleメソッドの他に，get_parameter_listというメソッドがあります．
このメソッドは，sampleメソッドでハイパーパラメータをランダムに選択する前のハイパーパラメータのリストを返します．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```diff

        new_params = []
+       hp_list = self.params.get_parameter_list()
-       sample = self.params.sample()

        for s in sample:
```

次に正規分布を用いてハイパーパラメータを生成します．
aiaccel/parameter.pyのHyperParameterクラスではnumpyのrandom.uniformを実行していましたが，今回は正規分布なのでnumpyのrandom.normalを利用します．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```python:aiaccel/optimizer/custom_optimizer.py
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
import numpy as np


class RandomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def generate_parameter(self) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        new_params = []
        hp_list = self.params.get_parameter_list()

        for hp in hp_list:
            value = np.random.normal(0, 0.1)
            value = min(max(value, hp.lower), hp.upper)
            new_param = {
                'parameter_name': hp.name,
                'type': hp.type,
                'value': value
            }
            new_params.append(new_param)

        return new_params

```

正規分布で生成したハイパーパラメータが，最大値・最小値を超えないよう修正を加えています．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```python:custom_optimizer.py
            value = min(max(value, hp.lower), hp.upper)
```

4. 正規分布オプティマイザの実行確認

それでは現在のディレクトリで実行してみます．

~~~bash
> pwd
/workspace/aiaccel/work/sphere

> aiaccel-start --config config.yaml --clean

~~~

正常に終了すれば成功です．

5. オプティマイザへの変数の導入

正規分布のオプティマイザの平均と分散の値はハードコーディングしていました．
このようなオプティマイザに利用する変数はコンフィグレーションファイルや引数として渡したい値です．

ここでは平均と分散をコンフィグレーションファイルから与える方法について解説します．

まずコンフィグレーションファイルに以下の追加をします．

***/workspace/aiaccel/work/sphere/config.yaml***

```diff
optimize:
  search_algorithm: "my_optimizer.CustomOptimizer"
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
+ mu: 3
+ sigma: 0.1
```

muとsigmaが追加されました．
次にcustom_optimizer.pyを編集して，muとsigmaを取得できるようにします．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```python:custom_optimizer.py
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
import numpy as np


class CustomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.mu = self.config.config.get('optimize', 'mu')
        self.sigma = self.config.config.get('optimize', 'sigma')

```

__init__メソッドを追加し，コンフィグレーションからmuとsigmaを取得し変数として保持しました．
あとはrandom.normalを呼ぶ際にmuとsigmaを渡します．

***/workspace/aiaccel/work/lib/my_optimizer/custom_optimizer.py***

```python:custom_optimizer.py
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
import numpy as np


class CustomOptimizer(AbstractOptimizer):
    """An optimizer class with a random algorithm.

    """

    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.mu = self.config.config.get('optimize', 'mu')
        self.sigma = self.config.config.get('optimize', 'sigma')

    def generate_parameter(self) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None
        """

        new_params = []
        hp_list = self.params.get_parameter_list()

        for hp in hp_list:
            value = np.random.normal(self.mu, self.sigma)
            value = min(max(value, hp.lower), hp.upper)
            new_param = {
                'parameter_name': hp.name,
                'type': hp.type,
                'value': value
            }
            new_params.append(new_param)

        return new_params

```

6. 正規分布オプティマイザの実行確認

それでは現在のディレクトリで実行してみます．

~~~bash
> pwd
/workspace/aiaccel/work/sphere

> aiaccel-start --config config.yaml --clean

~~~

/workspace/aiaccel/work/sphere/workディレクトリに実行結果が保存されます．
前回事項した結果と異なることを確認してみてください．

## オプティマイザ内部からの目的関数の値の参照

この節では，オプティマイザの内部から過去に計算した目的関数の値を参照して，次のパラメータを決定する方法を確認します．
aiaccel 上のジョブの ID が  `n` のときの目的関数の値は，`AbstractOptimizer` を継承して作成したカスタムオプティマイザの内部から，次のようにして取得できます．
```
objective_value =  self.storage.result.get_any_trial_objective(n)
```
この処理の後，`objective_value` は `n` で指定した目的関数の値か `None` を保持します．
`None` が保持されるのは，`self.storage.result.get_any_trial_objective()` が呼ばれた時点で，目的関数の値を計算する user program が Storage に 計算結果を保存していない場合です．

### 例: 勾配降下法による最適化

勾配降下法では，着目する試行におけるパラメータが与える目的関数の勾配を元に，次のパラメータを決定します．
一次元の場合， $n$ 試行目のパラメータを $W_n$ とし，目的関数を $f(W_n)$ と書くと， $n+1$ 試行目のパラメータは
$$W_{n+1} = W_n + \gamma f'(W_{n})$$
となります．ここで， $\gamma$ は学習率 (パラメータの更新をどの程度行うかの指標)， $f'(W_n)$ は $W_n$ における目的関数の勾配です．


ここでは $f$ の解析的な形が分からない場合に，勾配を差分で置き換えることを考えます．
簡単のため，前進差分のみを考えると，差分を用いた勾配の近似式は
$$f'(W_n) \approx \frac{f(W_n + \delta) - f(W_n) } { \delta }$$
となります．
従って $n + 1$ 試行目におけるパラメータは
$$W_{n + 1} \approx W_n + \gamma \frac{f(W_n + \delta) - f(W_n) } { \delta }$$
と近似できます．

### オプティマイザの実装

上の例では $n + 1$ 試行目のパラメータを決定するために， $f(W_n)$ と $f(W_{n+1})$ という 2 つの目的関数の値を使用しました．
カスタムオプティマイザでは，これらをメソッド `generate_parameter()` 内で取得する必要があります．
以下に，前進差分を用いたオプティマイザの例を示します．

```python
from enum import Enum, auto
from typing import Dict, List, Optional, Union
import copy

from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class SearchState(Enum):
    PREPARE = auto()
    CALC_FORWARD = auto()
    WAIT_CURRENT_OBJECTIVE = auto()
    WAIT_FORWARD_OBJECTIVE = auto()
    CALC_NEXT_PARAM = auto()


class GradientDescent(AbstractOptimizer):
    def __init__(self, options: Dict) -> None:
        super().__init__(options)

        self.learning_rate = self.config.config.get(
            'optimize', 'learning_rate')
        self.delta: float = 1e-03

        self.current_id: int = 0
        self.current_params: List[Dict[str, Union[str, float]]
                                  ] = self.generate_initial_parameter()
        self.num_parameters = len(self.current_params)
        self.forward_objectives: List[float] = []
        self.num_generated_forwards: int = 0
        self.num_calculated_forward_objectives: int = 0
        self.forward_ids: List[int] = []
        self.state: SearchState = SearchState.CALC_FORWARD

    def generate_parameter(self
                           ) -> Optional[List[Dict[str, Union[str, float]]]]:
        if self.state == SearchState.PREPARE:
            self.current_id = self.current_trial_id - 1
            self.forward_objectives = []
            self.num_generated_forwards = 0
            self.num_calculated_forward_objectives = 0
            self.forward_ids = []

            self.state = SearchState.CALC_FORWARD
            return None

        if self.state == SearchState.CALC_FORWARD:
            new_params = copy.deepcopy(self.current_params)
            param = self.current_params[self.num_generated_forwards]
            forward = {
                'parameter_name': param['parameter_name'],
                'type': param['type'],
                'value': param['value'] + self.delta
            }
            new_params[self.num_generated_forwards] = forward
            self.forward_ids.append(self.current_trial_id)
            self.num_generated_forwards += 1
            if self.num_generated_forwards == self.num_parameters:
                self.state = SearchState.WAIT_CURRENT_OBJECTIVE
            return new_params

        if self.state == SearchState.WAIT_CURRENT_OBJECTIVE:
            self.current_objective = self._get_objective(self.current_id)
            if self.current_objective is not None:
                self.state = SearchState.WAIT_FORWARD_OBJECTIVE
            return None

        if self.state == SearchState.WAIT_FORWARD_OBJECTIVE:
            forward_id = self.forward_ids[
                self.num_calculated_forward_objectives]
            forward_objective = self._get_objective(forward_id)
            if forward_objective is not None:
                self.forward_objectives.append(forward_objective)
                self.num_calculated_forward_objectives += 1
                if (self.num_calculated_forward_objectives ==
                        self.num_parameters):
                    self.state = SearchState.CALC_NEXT_PARAM
            return None

        if self.state == SearchState.CALC_NEXT_PARAM:
            new_params: List[Dict[str, Union[str, float]]] = []
            for param, forward_objective in zip(self.current_params,
                                                self.forward_objectives):
                grad = (forward_objective - self.current_objective
                        ) / self.delta
                value = param['value'] - self.learning_rate * grad
                new_param = {
                    'parameter_name': param['parameter_name'],
                    'type': param['type'],
                    'value': value
                }
                new_params.append(new_param)
            self.current_params = new_params

            self.state = SearchState.PREPARE
            return new_params

    def _get_objective(self, index):
        return self.storage.result.get_any_trial_objective(index)

    @property
    def current_trial_id(self):
        return self.trial_id.integer
```

このオプティマイザは以下で説明する 4 つの状態を取ります．
1. PREPARE: オプティマイザが保持する変数やリストの初期化を行います．
1. CALC_FORWARD: $W_n + \delta$ を計算します．
1. WAIT_CURRENT_OBJECTIVE: Storage に $W_n$ のときの目的関数の値が保存されるまで待機します．
1. WAIT_FORWARD_OBJECTIVE: Storage に $W_n + \delta$ のときの目的関数の値が保存されるまで待機します．
1. CALC_NEXT_PARAM: $n+1$ 試行目のパラメータを計算します．

これらの状態を `Enum` モジュールを用いて実装しています．
```python
class SearchState(Enum):
    PREPARE = auto()
    CALC_FORWARD = auto()
    WAIT_CURRENT_OBJECTIVE = auto()
    WAIT_FORWARD_OBJECTIVE = auto()
    CALC_NEXT_PARAM = auto()
```

オプティマイザが保持する変数は以下の通りです．
- learning_rate: 学習率．
- delta: 目的関数の前進値を計算するための変分 $\delta$.
- current_params: 現在 ( $n$ 試行目) のパラメータ $W_n$．
- num_parameters: 最適化するパラメータの数．
- forward_objectives: $W_n + \delta$ における目的関数の値 $f(W_n + \delta)$．
- num_generated_forwards: 既に生成された $W_n + \delta$ の数．
- num_calculated_forward_objectives: 計算が完了した $f(W_n + \delta)$ の数．
- forward_ids: $f(W_n + \delta)$ の計算が実行される aiaccel 上のジョブ ID (`trial_id`).

### `generate_parameter()` 内の処理の流れ

#### 状態: PREPARE

```python
        if self.state == SearchState.PREPARE:
            self.current_id = self.current_trial_id - 1
            self.forward_objectives = []
            self.num_generated_forwards = 0
            self.num_calculated_forward_objectives = 0
            self.forward_ids = []

            self.state = SearchState.CALC_FORWARD
            return None
```

オプティマイザが保持する変数やリストの初期化を行います．
初期化を行った後，オプティマイザの状態を `PREPARE` から `CALC_FORWARD` に変更します．
この状態では，メソッド `self.generate_parameters()` は必ず `None` を返却します．


#### 状態: CALC_FORWARD 

```python
        if self.state == SearchState.CALC_FORWARD:
            new_params = copy.deepcopy(self.current_params)
            param = self.current_params[self.num_generated_forwards]
            forward = {
                'parameter_name': param['parameter_name'],
                'type': param['type'],
                'value': param['value'] + self.delta
            }
            new_params[self.num_generated_forwards] = forward
            self.forward_ids.append(self.current_trial_id)
            self.num_generated_forwards += 1
            if self.num_generated_forwards == self.num_parameters:
                self.state = SearchState.WAIT_CURRENT_OBJECTIVE
            return new_params
```

パラメータの前進値 $W_n + \delta$ を ***1 回の呼び出しで 1 つだけ***計算し，aiaccel のメインループに値を返却します．

まず，パラメータのリストから前進値を計算するパラメータを一つずつ取り出します．
```python
            new_params = copy.deepcopy(self.current_params)
            param = self.current_params[self.num_generated_forwards]
```
ここで `self.num_generated_forwards` は，既に生成されたパラメータの前進値の総数を表します．

続いて，項目 `value` を $W_n + \delta$ で置き換えます．
```python
            forward = {
                'parameter_name': param['parameter_name'],
                'type': param['type'],
                'value': param['value'] + self.delta
            }
            new_params[self.num_generated_forwards] = forward
```

同時に，返却するパラメータの前進値を計算する aiaccel 上のジョブ ID (`trial_id`) を保持します．
```python
            self.forward_ids.append(self.current_trial_id)
```
`self.current_id` は，この***カスタムオプティマイザ内で***以下のように定義されるプロパティです．
```python
    @property
    def current_trial_id(self):
        return self.trial_id.integer
```

計算された前進値の数 `self.num_generated_forwards` をインクリメントします．
```python
            self.num_generated_forwards += 1
```

全てのパラメータについて，その前進値が計算されたとき，オプティマイザの状態を `CALC_FORWARD` から `WAIT_CURRENT_OBJECTIVE` に変更します．
```python
            if self.num_generated_forwards == self.num_parameters:
                self.state = SearchState.WAIT_CURRENT_OBJECTIVE
            return new_params
```

注意: オプティマイザの状態が `CALC_FORWARD` のとき，メソッド `generate_parameters()` は最適化するパラメータの数と同じ回数だけ aiaccel のメインループに呼ばれます．


#### 状態: WAIT_CURRENT_OBJECTIVE

```python
        if self.state == SearchState.WAIT_CURRENT_OBJECTIVE:
            self.current_objective = self._get_objective(self.current_id)
            if self.current_objective is not None:
                self.state = SearchState.WAIT_FORWARD_OBJECTIVE
            return None
```

現在のパラメータ $W_n$ における目的関数の値 $f(W_n)$ が Storage に保存されるまで待ち，その値を取得します．
```python
            self.current_objective = self._get_objective(self.current_id)
```
`self._get_objective()` は***カスタムオプティマイザ内***で以下のように定義されるメソッドです．
```python
    def _get_objective(self, index):
        return self.storage.result.get_any_trial_objective(index)
```
このメソッドは，取得したい目的関数の値の `trial_id` (aiaccel 上のジョブ ID) を引数に取り，Storage から値を読み出します．
ただし，呼び出された時点で Storage に値が保存されていなければ，`None` を返却します．

メソッド `self._get_objective()` が目的関数の値を返した場合，オプティマイザの状態を `WAIT_CURRENT_OBJECTIVE` から `WAIR_FORWARD_OBJECTIVE` に変更します．
この時点で， $f(W_n)$ の値はメンバ変数 `self.current_objective` に保持されています．
```python
            if self.current_objective is not None:
                self.state = SearchState.WAIT_FORWARD_OBJECTIVE
```

注意: オプティマイザが状態 `WAIT_CURRENT_OBJECTIVE` のとき，メソッド `self.generate_parameters()` は 1 回以上，Storage に対象とする目的関数の値が保存されるまで呼ばれます．
また，Storage から目的関数の値を読み出せたか否かに関わらず，`WAIT_CURRENT_OBJECTIVE` 状態の `self.generate_parameters()` は `None` をメインループに返します．

#### 状態: WAIT_FORWARD_OBJECTIVE

```python
        if self.state == SearchState.WAIT_FORWARD_OBJECTIVE:
            forward_id = self.forward_ids[
                self.num_calculated_forward_objectives]
            forward_objective = self._get_objective(forward_id)
            if forward_objective is not None:
                self.forward_objectives.append(forward_objective)
                self.num_calculated_forward_objectives += 1
                if (self.num_calculated_forward_objectives ==
                        self.num_parameters):
                    self.state = SearchState.CALC_NEXT_PARAM
            return None
```

パラメータの前進値 $W_{n} + \delta$ における目的関数の値 $f(W_n + \delta)$ が Storage に保存されるのを待ち，その値を取得します．

まず，取得する目的関数の aiaccel 上のジョブ ID (`trial_id`) をリストから読み出します．
```python
            forward_id = self.forward_ids[
                self.num_calculated_forward_objectives]
```
ここで，`self.num_calculated_forward_objectives` は取得済みの前進値に対する目的関数の値の総数です．

続いて，メソッド `self._get_objective()` に読み出した ID を渡して，Storage から目的関数の値を読み出します．
```python
            forward_objective = self._get_objective(forward_id)
```

正常な値が返却された場合，返却された値をリストに保持し，取得済みの値の総数 `self.num_calculated_forward_objectives` をインクリメントします．
```python
            if forward_objective is not None:
                self.forward_objectives.append(forward_objective)
                self.num_calculated_forward_objectives += 1
```

このとき，すべての目的関数の値が取得できていれば，オプティマイザの状態を `WAIT_FORWARD_OBJECTIVE` から `CALC_NEXT_PARAM` に変更します．
```python
                if (self.num_calculated_forward_objectives ==
                        self.num_parameters):
                    self.state = SearchState.CALC_NEXT_PARAM
```
注意: オプティマイザが状態 `WAIT_FORWARD_OBJECTIVE` のとき，メソッド `self.generate_parameters()` は少なくとも最適化対象のパラメータの総数以上，Storage にすべての目的関数の値が保存されるまで呼ばれます．また，Storage から目的関数の値が読み出せたか否かや，すべての値の読み出しが完了したか否かに依らず，`WAIT_FORWARD_OBJECTIVE` 状態の `self.generate_parameters()` は `None` をメインループに返します．

#### 状態: CALC_NEXT_PARAM

```python
        if self.state == SearchState.CALC_NEXT_PARAM:
            new_params: List[Dict[str, Union[str, float]]] = []
            for param, forward_objective in zip(self.current_params,
                                                self.forward_objectives):
                grad = (forward_objective - self.current_objective
                        ) / self.delta
                value = param['value'] - self.learning_rate * grad
                new_param = {
                    'parameter_name': param['parameter_name'],
                    'type': param['type'],
                    'value': value
                }
                new_params.append(new_param)
            self.current_params = new_params

            self.state = SearchState.PREPARE
            return new_params
```

Storage から読み出した $W_n$ における目的関数の値 $f(W_n)$ (`self.current_params`) と $W_{n+1}$ における目的関数の値 $f(W_{n+1})$ (`self.forward_objectives`) を用いて勾配を計算します．
```python
            new_params: List[Dict[str, Union[str, float]]] = []
            for param, forward_objective in zip(self.current_params,
                                                self.forward_objectives):
                grad = (forward_objective - self.current_objective
                        ) / self.delta
```

計算した勾配を用いて次のパラメータ $W_{n+1}$ を計算して `Dict` 型オブジェクトを作成し，リストに保持します．
```python
                value = param['value'] - self.learning_rate * grad
                new_param = {
                    'parameter_name': param['parameter_name'],
                    'type': param['type'],
                    'value': value
                }
                new_params.append(new_param)
```

オプティマイザの状態を `CALC_NEXT_PARAM` から `PREPARE` に変更し，作成した次のパラメータをメインループに返却します．
```python
            self.state = SearchState.PREPARE
            return new_params
```


### 注意事項

一般に，パラメータの更新ステップ数 $n$ と aiaccel 上のジョブ ID (`trial_id`) は一致しないことに注意してください．
例えば上の例において，最適化したいパラメータの数が 5 個の場合，パラメータを１度更新するために目的関数を 5 回計算する必要があります．
この場合は 1 回のパラメータ更新で aiaccel の `trial_id` は 5 増加することになります．
従って，config.yaml で指定した `trial_number` が，例えば 30 回の場合，初期値を除いて 4 回しかパラメータは更新されません．

同様な ID の不一致は NelderMeadOptimizer を用いた際にも起こります．
Nelder-Mead 法の 1 ステップに相当する処理が終了したとき，aiaccel 上では `trial_id` が **パラメータ数 + 1** だけ増加します．
