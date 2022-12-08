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
