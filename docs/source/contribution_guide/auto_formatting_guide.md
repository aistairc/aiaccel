# pre-commit による自動フォーマット

### 概要
`git commit` する際，コードを自動フォーマットする方法について説明します．
pre-commit を用いて ruff，mypy 等によるチェックを行います．
pre-commit は `git commit` 時に .pre-commit-config.yaml に記載された hooks を実行し，エラーがあった場合は commit を受け付けないようにするツールです．

### ソフトウェアのインストール

`git clone` した aiaccel ディレクトリ直下に移動し，下記コマンドにより pyproject.toml の `[project.optional-dependencies]` に記載されたライブラリ（pre-commit，ruff 等）をインストールします．

~~~bash
cd /path/to/aiaccel
pip install .[dev]
~~~

### コンフィグレーションファイルの確認
`git commit` を実行する前に pyproject.toml と .pre-commit-config.yaml を確認してください．
必要であれば各フォーマットチェックの設定を追記することができます．

### pre-commitの有効化
下記コマンドにより pre-commit を有効にします．

~~~bash
pre-commit install
~~~

以上の手順で `git commit` 時に自動フォーマットできます．エラーがあった場合は `git commit` は停止するため，エラーを解消した後再度 `git commit` を実行してください．

> [!NOTE]
> ### pre-commit のチェックを通さず commit したい場合
>
> ~~~bash
> git commit --no-verify
> ~~~

> [!NOTE]
> ### `git commit` 時以外で pre-commit を実行する方法
>
> ~~~bash
> pre-commit run -a
> ~~~

## 静的フォーマットによるチェック

### ruff の実行
~~~bash
ruff format aiaccel
ruff check --fix aiaccel
~~~

### mypy の実行
~~~bash
mypy aiaccel
~~~
