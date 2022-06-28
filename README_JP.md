# aiaccel
[AI橋渡しクラウドABCI](https://abci.ai/)向けハイパーパラメータ最適化ライブラリ。
サポートされている最適化アルゴリズムは、ランダムサーチ、グリッドサーチ、Sobol列、Nelder-Mead法、およびベイズ最適化法 (TPE)。

# インストール
本ソフトウェアは下記コマンドでインストールできます。
~~~
pip install git+https://github.com/aistairc/aiaccel.git
~~~

# 動作環境
  - python 3 (3.8.13以上)


# examples

## examplesについて

`./examples/` 以下に、5種類のサンプルを提供しています。
ディレクトリ名 (`sphere`, `schwefel`, ...) は、対象のベンチマーク関数名を意味します。

本ソフトウェアは、localとabciの2つの動作環境をサポートしています。
本ソフトウェアは、ランダムサーチ、グリッドサーチ、ソボルシーケンス、ネルダーミード、TPEの5つの探索アルゴリズムをサポートしています。

動作環境とアルゴリズムは、コンフィギュレーションファイルで変更することができます。
- 動作環境
  - local: ローカル上で最適化を実行します。
  - abci: [ABCI](https://abci.ai/)上で最適化を実行します。
- 探索アルゴリズム
  - ランダム: ハイパーパラメータをランダムに探索します。
  - グリッド: ハイパーパラメータを網羅的に探索します。
  - ソボル列: [sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence)に従ってハイパーパラメータを探索します。
  - ネルダーミード: [nelder mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)に従ってハイパーパラメータを探索します。
  - TPE: [TPE(Tree-structed Parzen Estimator Approach)](https://www.lri.fr/~kegl/research/PDFs/BeBaBeKe11.pdf)に従ってハイパーパラメータを探索します。

## ローカル環境でsphere関数の最適化を実行
examples/sphere を実行する方法を説明します。


1. virtualenvをインストールし、仮想環境を作成します。
~~~
    > pip install virtualenv
    > virtualenv venv
    > source venv/bin/activate
~~~

2. 本ソフトウェアをインストールします
~~~
    pip install cython numpy pytest
    pip install git+https://github.com/aistairc/aiaccel.git 
~~~


3. ワークスペースを用意し、sphereディレクトリをコピーします。
~~~
    > cd your_workspace_directory
    > cp -R cloned_directory/aiaccel/examples .
    > cd examples
    > ls
    sphere

    > cd sphere
    > ls
    config.yaml         job_script_preamble.sh         user.py
~~~

4. 実行。
~~~
    > python -m aiaccel.start --config config.yaml
~~~

ワークスペースディレクトリは `--clean` を付加することで実行前に削除できます。
~~~
    > python -m aiaccel.start --config config.yaml --clean
~~~

5. 結果を確認
~~~
    > ls ./work
    abci_output         alive               hp                  lock
    log                 resource            result              runner
    state               verification

    > cat ./work/result/final_result.result
~~~

6. 設定を変更したい場合は、config.yamlファイルを編集してください。
~~~
    vi config.yaml
~~~

7. 最適化を再実行したい場合は、ワークスペースディレクトリを移動してください。
~~~
    > mv /tmp/work /tmp/work/work_aiaccel_200101
~~~


## ABCI環境でsphere関数の最適化を実行
ABCI上でexamples/sphereを実行する手順を説明します。

1. まず、ABCIユーザーズガイドに従って、pythonの環境を構築してください。
~~~
    module load python/3.8/3.8.13
    python3 -m venv work
    source work/bin/activate
~~~

2. config.yamlのresourceをABCIに変更します。
```yaml
resource:
  type: "ABCI"
  num_node: 4
```

3. ワークスペースを用意します．ここからの作業は、[ローカル環境でsphere関数の最適化を実行]の3〜4と同じです。

4. 実行
~~~
    > python -m aiaccel.start --config config.yaml
~~~

5. 実行中のジョブを確認したい場合は、[ABCIユーザーズガイド](https://docs.abci.ai/ja/)を参照してください。


<br>
<hr>

# 補助ツールについて

ABCI上でaiaccelを使用する際、aiaccelの補助ツール(wd)を使用できます。<br>
試作中につき機能は制限していますが、必要ならば是非ともご利用ください。<br>
詳細は[wd_dev/README_JP.md]をご参照ください。


# 補助ツールの機能概要

1. ABCI上でaiaccelを使用するためのCUI機能を利用できます。
2. ABCIのポイント消費量を最大で約25%抑えることができます。

[experimental/README_JP.md]:experimental/README_JP.md

<br>
<hr>


# 謝辞

この成果の一部は、国立研究開発法人新エネルギー・産業技術総合開発機構(NEDO)の委託業務として開発されたものです。<BR>
TPEアルゴリズムは Optuna を利用しました。
