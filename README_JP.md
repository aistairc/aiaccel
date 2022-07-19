# aiaccel: an HPO library for ABCI
[![GitHub license](https://img.shields.io/github/license/aistairc/aiaccel.svg)](https://github.com/aistairc/aiaccel)
[![Supported Python version](https://img.shields.io/badge/Python-3.8-blue)](https://github.com/aistairc/aiaccel)
![CI status](https://github.com/aistairc/aiaccel/actions/workflows/actions.yaml/badge.svg)

[AI橋渡しクラウドABCI](https://abci.ai/)向けハイパーパラメータ最適化ライブラリ。
ランダムサーチ、グリッドサーチ、Sobol列、Nelder-Mead法、およびベイズ最適化法 (TPE)をサポートしています。

# インストール
本ソフトウェアは下記コマンドでインストールできます。
~~~bash
pip install git+https://github.com/aistairc/aiaccel.git
~~~

# 実行例
## ローカル環境で実行する場合

0. (オプション) Virtualenvをインストールし、仮想環境を作成します。
    ~~~bash
    > pip install virtualenv
    > virtualenv venv
    > source venv/bin/activate
    ~~~

1. `aiaccel`をインストールします
    ~~~bash
    pip install git+https://github.com/aistairc/aiaccel.git 
    ~~~


2. ワークスペースを作成し、sphereディレクトリをコピーします。
    ~~~bash
    > mkdir your_workspace_directory
    > cd your_workspace_directory
    > git clone https://github.com/aistairc/aiaccel.git 
    > cp -R cloned_directory/aiaccel/examples .
    > cd examples
    > ls
    sphere

    > cd sphere
    > ls
    config.yaml         job_script_preamble.sh         user.py
    ~~~

3. パラメータ最適化を実行します。
    ~~~bash
    > python -m aiaccel.start --config config.yaml
    ~~~

    Tips: ワークスペースは `--clean` を付加することで実行前に初期化できます。
    ~~~bash
    > python -m aiaccel.start --config config.yaml --clean
    ~~~

4. 結果を確認する。
    ~~~bash
    > ls ./work
    abci_output         alive               hp                  lock
    log                 result              runner              state
    verification

    > cat ./work/result/final_result.result
    ~~~

5. 設定を変更したい場合は、config.yamlファイルを編集してください。
    ~~~bash
    vi config.yaml
    ~~~

## ABCI上で実行する
1. まず、[ABCIユーザーズガイド](https://docs.abci.ai/ja/python)に従って、pythonの環境を構築してください。
    ~~~bash
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

3. ワークスペースを用意します．ここからの作業は、[ローカル環境で実行する場合](https://github.com/aistairc/aiaccel/blob/main/README_JP.md#%E3%83%AD%E3%83%BC%E3%82%AB%E3%83%AB%E7%92%B0%E5%A2%83%E3%81%A7%E5%AE%9F%E8%A1%8C%E3%81%99%E3%82%8B%E5%A0%B4%E5%90%88)の2および3と同じです。

4. 実行
    ~~~bash
    > python -m aiaccel.start --config config.yaml
    ~~~

5. 実行中のジョブを確認したい場合は、[ABCIユーザーズガイド](https://docs.abci.ai/ja/)を参照してください。


# 開発中の機能wdについて
ABCI上で `aiaccel` を実行する場合、HPOを管理する `master` プログラムが常時実行している必要があり、目的関数を計算している間も待機のためにポイントを消費してしまいます。
`./experimental/wd` (watch dog) と呼ばれる開発中の機能を用いれば、このような不必要な消費を抑制することができます。
詳細は[experimental/README_JP.md](experimental/README_JP.md)をご参照ください。


# 謝辞
* この成果の一部は、国立研究開発法人新エネルギー・産業技術総合開発機構(NEDO)の委託業務として開発されたものです。
* TPEアルゴリズムは Optuna を利用しました。
