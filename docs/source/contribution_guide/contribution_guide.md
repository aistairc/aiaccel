# イシュー (WIP)

問題を発見した場合や，機能のリクエストがあった場合，まずリポジトリにある既存のイシューを確認してください．
同様のテーマが話し合われていない場合，新しいイシューを作成してください．

## バグの報告 (Bug report)

バグの報告を行う際には，以下の内容についての明確かつ簡潔な説明を含めることを推奨します．

- バグの内容
- バグを再現する手順
- あなたが起こる期待したこと
- 実行環境


## 機能リクエスト (Feature request)

機能リクエストでは，以下の内容についての明確かつ簡潔な説明を含めることを推奨します．

- バグが関連する場合，バグの内容
- 実現したい機能の説明
- 検討した実装の説明



# プルリクエスト (WIP)

コードの実装が必要であると判断された場合，コードの作成を行います．
プルリクエストを行う際には，以下に注意してください．

- [ドキュメンテーション](#ドキュメンテーション-wip)
    - 実装した関数の基本的な説明，パラメータや返却値の型と意味，使用例を docstring として記述します．
    [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に準拠する形で記述してください．
    これは，インタープリタやリファレンスガイドで読まれることになります．
    - 大規模な機能の追加があった場合，ドキュメントを準備してください．
    ドキュメントのソースファイルは docs の下のディレクトリにマークダウン形式で作成します．
- [ユニットテスト](#テスト-wip)
    - 実装した全てのコードをテストするテストコードを作成します．
    - Python のバージョンや OS に依らず動くことをローカル環境で確認します．
- コーディングスタイル
    - ソースコードは Python で実装してください．
    - Python のソースコードは PEP8 に準拠して記述してください．
    - aiaccel では flake8 でコーディングスタイルを検証します．
- その他の注意事項
    - [注意事項](#注意事項)にまとめた事項を確認してください．


## 手順
### 初めて開発に参加する場合
- まず，GitHub 上で aiaccel をフォークします．
- フォークした後，aiaccel をダウンロードし，インストールします．
    ```bash
    git clone https://github.com/[YOUR USERNAME]/aiaccel.git
    ```
- ディレクトリを移動し，upstream のリポジトリを追加します．
    ```bash
    cd aiaccel
    git remote add upstream gttps://github.com/aistaic/aiaccel.git
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

- 進行に合わせてローカルでコミットします (`git add` および `git commit` を使用)．

    コミットメッセージでは，変更の動機，バグの性質，または拡張機能の詳細を説明します．
    メッセージは，コードを見なくても内容を理解できるように記述する必要があります．
    Numpy で利用されている以下のような [acronyms](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message) を活用すると，分かりやすいかもしれません．
    > ```
    >API: an (incompatible) API change
    >BENCH: changes to the benchmark suite
    >BLD: change related to building SciPy
    >BUG: bug fix
    >DEP: deprecate something, or remove a deprecated object
    >DEV: development tool or utility
    >DOC: documentation
    >ENH: enhancement
    >MAINT: maintenance commit (refactoring, typos, etc.)
    >REV: revert an earlier commit
    >STY: style fix (whitespace, PEP8)
    >TST: addition or modification of tests
    >REL: related to releasing SciPy
    >```


### 投稿

*プルリクエストを行う前に，以下を確認してください*：
- MIT ライセンスで配布できるか？
- 適切な[ユニットテスト](#テストwip)は存在するか？
- [ユニットテスト](#テストwip)をローカル環境で実行できたか？
- パブリックな関数は docstring を持っているか？
- [ドキュメンテーション](#ドキュメンテーション-wip)は正しくレンダリングされるか？
- [コーディングスタイル](#コーディング規約wip)は適切か？
- コミットメッセージは適切か？
- 大規模な追加の場合，チュートリアル (docs/source/tutorial) やモジュールレベルの説明はあるか？
- コンパイル済みのコードを追加する場合，setup.py を変更したか？


*上記を確認した後*:
- GitHub 上のフォークに変更をプッシュします．

    ```bash
    git push origin feature/add-new-optimizer
    ```
- GitHub のユーザーネームとパスワードを入力します．

- GitHub に移動します．以下に注意しながらタイトルとメッセージを記述します．

    **タイトル**
    - 変更を反映した簡単な説明を行うこと．
    - コードはバックフォートでラップすること．
    - ピリオドで終了しないこと．

    **説明**
    - 動機を書くこと．
    - 変更点を書くこと．
    - 作業が進行中 (work-in-progress) であるなら，残りのタスクを書くこと．

- プルリクエストを送信します．

## 注意事項
### ブランチ名
- ブランチ名は feature/* としてください．



# ドキュメンテーション (WIP)
## docstrings
- 実装した関数の基本的な説明，パラメータや返却値の型と意味，使用例を docstrings として記述します．
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に準拠する形で記述してください．
- [コーディング規約](#コーディング規約-wip) も参考にしてください．


## ドキュメント
- ドキュメントのソースファイルは docs の下のディレクトリにマークダウン形式で作成します．
- 大規模な機能の追加があった場合，ドキュメントを準備してください．


## レンダリングの確認
ドキュメントの追加や変更・修正があった場合には，ローカル環境でレンダリングが正常に行われるかを確認してください．
レンダリングの確認には aiaccel/docs に移動し，HTML ファイルのビルドを行います．
```bash
cd aiaccel/docs
make html
```
docs/build/html の下に，HTML 形式のファイルが生成されます．


# テスト (WIP)

## テストの追加
- ユニットテストは tests の下のディレクトリに作成します．
- 新たな機能の追加, またはバグの修正を行った場合，テストコードを準備してください．
- aiaccel では pytest を用いてテストを行います．
- [コーディング規約](#コーディング規約-wip)  も参考にしてください．

## テストの実行

ローカル環境ですべてのテストコードを実行するには，aiaccel または aiaccel/tests に移動し，以下のコマンドを実行します．
```bash
cd aiaccel/tests
pytest
```
または，特定のテストコードのみを実行するには，ファイル名を引数として指定します．
```bash
pytest tests/unit/optimizer_test/test_abstract_optimizer.py
```


# コーディング規約 (WIP)

## 基本的なルール

- aiaccel のソースコードは Python で作成します．
- コーディングスタイルは PEP8 に従います．
    - aiaccel では flake8 を用いてコーディングスタイルの検証を行います．
- いくつかの例外が存在します．
    - [aiaccel-specific なスタイル](#aiaccel-specific-なスタイル)についても確認してください．
- aiaccel では型ヒントの検証は行いませんが，できる限り型ヒントを記述してください．
    - aiaccel ではバージョン 3.8 の Python をサポートするため，ビルトインな "`list`" などを型ヒントに使用する際は，future-import を行ってください．


## aiaccel-specific なスタイル

### プレフィックス `_` の追加 (プライベートなメソッド，関数，フィールド，及びクラスの名前)

#### Example


### テスト

#### Good Example

#### Bad Example



### Docstrings

基本的には [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) に準拠する形で docstrings を記述します．
ただし，以下の例外についても注意してください．

- 各モジュールの docstrings は必須ではありません．
- `Args` セクションでは，パラメータ名の後ろにパラメータの型を括弧で括って記述します．
- 必要に応じて `Example:` セクションを追加します．
- `__init__` メソッドはクラスの docstring に含めます．`__init__` メソッドには記述しません．
- Python オブジェクトへのリンクは *sphinx-style* なリンクを使用します.

#### Example

