# Issues

問題を発見した場合や追加機能の要望がある場合，すでに同様の issue が投稿されていないかの確認をお願いします．
同様のテーマが話し合われていない場合，新しい issue を作成してください．

## バグの報告 (Bug report)

バグの報告では，以下の内容についての明確かつ簡潔な説明を含めてください．

- バグの内容
- バグを再現する手順
- あなたが起こると期待したこと
- 実行環境


## 機能リクエスト (Feature request)

機能リクエストを行う際には，以下の内容についての明確かつ簡潔な説明を含めることを推奨します．

- バグが関連する場合，バグの内容
- 実現したい機能の説明
- 検討した実装の説明



# Pull request

aiaccel のコードを修正しリポジトリに反映して欲しい場合，pull request を実行してください．
Pull request を行う際には，以下に注意してください．

## 手順

### 初めて開発に参加する場合
- まず，GitHub 上で aiaccel をフォークします．
- フォークした後，aiaccel のリポジトリを clone します．
    ```bash
    git clone https://github.com/[YOUR USERNAME]/aiaccel.git
    ```

### 開発
- ローカルのリポジトリを最新の状態に更新します．
    ```bash
    git checkout main
    git pull upstream main
    ```
- ブランチを作成します．
    ```bash
    git checkout -b feature/add-new-feature
    ```
- `git add` および `git commit` を使用し，進行に合わせてローカルでコミットします．
    - コミットメッセージでは，変更の動機，バグの性質，または拡張機能の詳細を説明します．
    - メッセージは，コードを見なくても内容を理解できるように記述する必要があります．
### 投稿

*Pull request を行う前に，以下を確認してください*：
- 事前に issue などで他の開発者と議論したか？
- MIT ライセンスで配布できるか？
- 適切な[ユニットテスト](#テスト)は存在するか？
- [ユニットテスト](#テスト)をローカル環境で実行できたか？
- パブリックな関数は docstring を持っているか？
- [ドキュメンテーション](#ドキュメンテーション-wip)は正しくレンダリングされるか？
- [コーディングスタイル](#コーディング規約)は適切か？
- コミットメッセージは適切か？
- 大規模な追加の場合，例 (docs/source/examples) やモジュールレベルの説明はあるか？
- コンパイル済みのコードを追加する場合，setup.py を変更したか？


*上記を確認した後*:
- GitHub 上のフォークに変更をプッシュします．
    ```bash
    git push origin feature/add-new-optimizer
    ```
- GitHub のユーザーネームとパスワードを入力します．
- GitHub に移動します．以下に注意しながらタイトルとメッセージを記述します．
    - **タイトル**
        - 変更を反映した簡単な説明を行うこと．
        - コードはバックフォートでラップすること．
        - ピリオドで終了しないこと．
    - **説明**
        - 動機を書くこと．
        - 変更点を書くこと．
        - 関連する issue を閉じることができる場合，`Close #N` で issue を閉じること．
        - 作業が進行中 (work-in-progress) であるなら，残りのタスクを書くこと．
- Pull request を送信します．


## レビュープロセス
- 他の開発者は，pull request の実装，ドキュメント，コーディングスタイルを改善するためのコメントを投稿します．
- Pull request したコードの更新を行う際は，ローカルリポジトリで変更をコミットし，ローカル環境でのテストが成功した場合にのみフォークへプッシュします．
- aiaccel の開発チームのメンバー 1 人以上が pull request を検証し，承認された場合に main ブランチへマージされます．



# ドキュメンテーション (WIP)

## docstrings
- 実装した関数の基本的な説明，パラメータや返却値の型と意味，使用例を docstrings として記述します．
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に準拠する形で記述してください．
- [コーディング規約](#コーディング規約) も参考にしてください．


## ドキュメント

- ドキュメントのソースファイルは docs の下のディレクトリに作成します．
- ドキュメントのファイル形式はマークダウン形式を推奨しています．
- 大規模な機能の追加があった場合，ドキュメントを作成してください．


## レンダリングの確認

ドキュメントの追加や変更・修正があった場合には，ローカル環境でレンダリングが正常に行われるかを確認してください．

API リファレンスの生成を行うには，aiaccel に移動し，以下のコマンドを実行します．
```bash
cd aiaccel
sphinx-apidoc -f -o docs/source/api_reference aiaccel
```
ドキュメンテーションのレンダリングを確認するには，aiaccel/docs に移動し，HTML ファイルのビルドを行います．
```bash
cd docs
make html
```
ビルドされた HTML 形式のファイルは docs/build/html の下に生成されます．

多言語ドキュメントの生成を行うには，aiaccel/docs で以下のコマンドを実行します．
```bash
make gettext
sphinx-intl update -p build/gettext -l en -l ja
```


# テスト

## テストの追加
- aiaccel では pytest を用いてテストを行います．
- ユニットテストは tests の下のディレクトリに作成します．
    - aiaccel/tests/unit 以下のディレクトリ構造は，config.py などの一部のモジュールを除いて，aiaccel/aiaccel 以下の構造に対応します． 例えば，aiaccel/aiaccel/optimizer/abstract_optimizer.py のテストは aiaccel/tests/unit/optimzier_test/test_abstract_optimizer.py です．
- 新たな機能の追加，またはバグの修正を行った場合，テストコードを作成してください．


## テストの実行 (WIP)
ローカル環境ですべてのテストコードを実行するには，aiaccel に移動し，以下のコマンドを実行します．
```bash
cd aiaccel
pytest
```
特定のテストコードのみを実行したい場合には，ファイル名を引数として指定します．
```bash
pytest tests/unit/optimizer_test/test_abstract_optimizer.py
```
さらに，コーディングスタイルのチェックを行うため，以下のコマンドを実行します．
```bash
pycodestyle aiaccel examples
flake8 aiaccel examples 
```


## 追加コードに対するカバレッジ

コードカバレッジの厳密な基準は設定されていませんが，テストを設計する際にはこの値を十分に考慮します．
特に，以下のような場合は注意が必要です．
- 全体的なスコアが大幅に低下する場合．
- あるクラスやモジュールのカバレッジが異常に低い場合．
- テストが if 文の特定の分岐をカバーしていない場合．

### カバレッジの測定
C0 カバレッジを測定するには，オプション `--cov` を使用して pytest を実行します．
```bash
pytest --cov=aiaccel
```
特定のテストコードのみのカバレッジを測定するには，aiaccel の部分を適切なパスに置き換えます．

C1 カバレッジを測定するには，オプション `--cov` に加えて `--cov-branch` を使用して pytest を実行します．
```bash
pytest --cov=aiaccel --cov-branch
```

# コーディング規約

## 基本的なルール

- aiaccel のソースコードは Python で作成します．
- コーディングスタイルは PEP8 に従います．
    - aiaccel では pycodestyle と flake8 を用いてコーディングスタイルの検証を行います．
    <!-- - オートフォーマッタとして pylint や black の使用を推奨しています． -->
    - 下記の Docstrings についても確認してください．
- aiaccel では型ヒントの検証は行いませんが，できる限り型ヒントを記述してください．
    - aiaccel ではバージョン 3.8 の Python をサポートするため，ビルトインな "`list`" などを型ヒントに使用する際は，future-import を行ってください．
- ランダムな値の生成には [`numpy.random.RandomState`](https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html) を使用して下さい．これは aiaccel が利用しているライブラリ [optuna](https://github.com/optuna/optuna) との互換性を保つためです．


## Docstrings

基本的には [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に準拠する形で docstrings を記述します．
ただし，以下の例外についても注意してください．

- 各モジュールの docstrings は必須ではありません．
- `Args:` セクションでは，パラメータ名の後ろにパラメータの型を括弧で括って記述します．
- 必要に応じて `Example:` セクションを追加します．
- `__init__` メソッドはクラスの docstring に含めます．`__init__` メソッドには記述しません．
- Python オブジェクトへのリンクは *sphinx-style* なリンクを使用します.
- エディタとして vscode を利用する場合，[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) が docstring 生成の役に立ちます．
 
### Example

```python
class ExampleClass:
    """Summary of class.

    There can be additional description(s) of this class.

    Args:
        param1 (type_of_param1): Description of `param1` which
            is given when __init__ method is called.
        param2 (type_of_param2): Description of `param2`.

    Attributions:
        param1 (type_of_param1): Description of `param1`.
        param2 (type_of_param2): Description of `param2`.
        param3 (type_of_param3): Description of 'param3`. 
    """

    def __init__(self, param1: type_of_param1, param2: type_of_param2):
        self.param1 = param1
        self.param2 = param2
        self.param3 = generate_param3()

    def method(self, arg1: type_of_arg1) -> type_of_return:
        """Recieves `type_of_arg1` object and returns return_of_method. 

        Args:
            arg1 (type_of_arg1): Description of `arg1`.
        
        Returns:
            type_of_return: Description of return value. If this method
            returns nothing, this section can be omitted.
        
        Raise:
            TypeOfException: Description of Exception.

        """
        ...
        return return_of_method

```
