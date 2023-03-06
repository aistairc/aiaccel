# 概要
git commitをする際，コードを自動フォーマットするための方法について記載する．
pre-commitを用いて，black, isort, flake8によるチェックを行う．

# ソフトウェアのインストール
以下のコマンドから必要なモジュールをインストールする．
~~~bash
> pip install pre-commit black isort flake8
~~~

# コンフィグレーションファイルの確認
git commitを実行する前に pyproject.toml, .pre-commit-config.yaml を確認する．
各フォーマットチェックの設定を追記することができる．

# pre-commitのインストール
以下のコマンドから pre-commit　を有効にする．
~~~bash
> pre-commit install
~~~

# フォーマットのチェック
上記の設定が完了した後，git commit コマンドを実行時自動フォーマットが走り結果を確認することができる．エラーがあった場合 git commit は停止するため，エラーを解消した後再度 git commit を実行する．

# 静的なフォーマットのチェック

## blackの実行
~~~bash
> black --check aiaccel
~~~

## flake8の実行
~~~bash
> flake8 aiaccel
~~~

## isortの実行
~~~bash
> isort --check aiaccel
~~~
