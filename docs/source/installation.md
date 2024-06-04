# インストールガイド (WIP) <a id="top"></a>

ここでは下記の環境におけるaiaccelのインストール方法について解説します．

- [Linux](#Linux)
- [ABCI](#ABCI)
- [Windows](#Windows)
- [macOS](#macOS)


## Linux <a id="Linux"></a>

### Git のインストール
Git のインストール方法は指定しませんが， `apt` 等でインストールして事前に設定を済ませておいてください．
~~~bash
sudo apt update
sudo apt install git
~~~

### 仮想環境の作成

Python の 標準ライブラリである venv を使用して仮想環境を作成します．
下記のコマンドを実行して仮想環境を作成してください．

~~~bash
python3 -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」と表記しますが，仮想環境の名前は任意のものを設定できます．

### 仮想環境のアクティベート

仮想環境を利用するには，下記のコマンドを実行します．

~~~bash
source optenv/bin/activate
~~~

以降の作業はアクティベート済みのものとして進めます．


> [!NOTE] 仮想環境の終了
> 仮想環境を終了する際は下記のコマンドを実行します．
> ~~~bash
> deactivate
> ~~~


### aiaccel のインストール

> [!NOTE]
> 事前に pip と setuptools をアップグレードすることを推奨します．
> `python -m pip --version` コマンドで仮想環境上の pip が表示されることを確認した後，下記のコマンドを実行してください．（システムの Python の pip を使用しないように注意してください．）
>
> ~~~bash
> python -m pip install --upgrade pip setuptools
> ~~~



aiaccel は下記コマンドでインストールできます．
~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Python のインタラクティブモードを使用して，aiaccel がインストールできているか確認します．

~~~bash
$ python
>>> import aiaccel
>>>
~~~

#### （参考）ローカルからのインストール

上記の方法とは別の方法でインストールすることもできます．（aiaccelをダウンロードした後，ローカルからインストールする方法です．）

まず，任意の場所で aiaccel をダウンロードします．

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後，aiaccel ディレクトリに移動します．

~~~bash
cd aiaccel
~~~

下記コマンドを使用して aiaccel をインストールします．（pyproject.toml ファイルを利用しています．）

~~~bash
python -m pip install .
~~~

Python のインタラクティブモードを使用して，aiaccel がインストールできているか確認します．

~~~bash
$ python
>>> import aiaccel
>>>
~~~


> [!NOTE]
> 当ガイドの内容は WSL2 (Windows Subsystem for Linux 2) で正常に動作することを確認済みです．

[back to top](#top)

<br>

## ABCI [(AI Bridging Cloud Infrastructure)](https://abci.ai/ja/about_abci/) <a id="ABCI"></a>
ABCI には計算ノード (V) と計算ノード (A) という２つのノードがあり，それぞれ使用しているディストリビューションが異なります．[ABCI システムの概要](https://docs.abci.ai/ja/system-overview/#software)を参照してください．

### Python 環境の構築
[ABCIユーザーガイド](https://docs.abci.ai/ja/python)に従って Python 環境を用意します．
`module` コマンドを使用して Python 環境をロードします．
~~~bash
module load python/3.10/3.10.10
~~~

### 仮想環境の作成

Python の 標準ライブラリである venv を使用して仮想環境を作成します．
下記のコマンドを実行して仮想環境を作成してください．

~~~bash
python3 -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」と表記しますが，仮想環境の名前は任意のものを設定できます．

### 仮想環境のアクティベート

仮想環境を利用するには，下記のコマンドを実行します．

~~~bash
source optenv/bin/activate
~~~

以降の作業はアクティベート済みのものとして進めます．


> [!NOTE] 仮想環境の終了
> 仮想環境を終了する際は下記のコマンドを実行します．
> ~~~bash
> deactivate
> ~~~


### aiaccel のインストール

> [!NOTE]
> 事前に pip と setuptools をアップグレードすることを推奨します．
> `python -m pip --version` コマンドで仮想環境上の pip が表示されることを確認した後，下記のコマンドを実行してください．（システムの Python の pip を使用しないように注意してください．）
>
> ~~~bash
> python -m pip install --upgrade pip setuptools
> ~~~


aiaccel は下記コマンドでインストールできます．
~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Python のインタラクティブモードを使用して，aiaccel がインストールできているか確認します．

~~~bash
$ python
>>> import aiaccel
>>>
~~~

#### （参考）ローカルからのインストール

上記の方法とは別の方法でインストールすることもできます．（aiaccelをダウンロードした後，ローカルからインストールする方法です．）

まず，任意の場所で aiaccel をダウンロードします．

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後，aiaccel ディレクトリに移動します．

~~~bash
cd aiaccel
~~~

下記コマンドを使用して aiaccel をインストールします．（pyproject.toml ファイルを利用しています．）

~~~bash
python -m pip install .
~~~

Python のインタラクティブモードを使用して，aiaccel がインストールできているか確認します．

~~~bash
$ python
>>> import aiaccel
>>>
~~~

[back to top](#top)

<br>

## Windows <a id="Windows"></a>

### 準備1: Git のインストール

GitHub 経由で aiaccel をインストールする場合，[git](https://gitforwindows.org/) がインストールされている必要があります．
あらかじめインストールしてください．

### 準備2: Execution Polisy の設定

PowerShell を使用して仮想環境を作る場合，セキュリティ設定によって仮想環境をアクティベートするスクリプトが実行できないことがあります．
***以下の手順でセキュリティ設定を変更することが可能ですが，セキュリティリスクが上がることに十分注意してください．詳細は[こちら](https://learn.microsoft.com/ja-jp/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.3)をご確認ください．***

以下のコマンドを実行して，PowerShell の設定を確認します．
```console
> Get-ExecutionPolicy
Restricted
```
この例のように Restricted と表示された場合，以下のコマンドを実行し，設定を変更します．
```console
> Set-ExecutionPolicy RemoteSigned
```

実行後，コマンド `Get-ExecutionPolicy` を実行して，`RemoteSigned` と表示されれば設定完了です．
```console
> Get-ExecutionPolicy
RemoteSigned
```

使用中の PowerShell ウィンドウのみに変更を適用したい場合，以下のようにオプションを追加して設定します．
```console
> Set-ExecutionPolicy RemoteSigned -Scope Process
```

### Python-venv による仮想環境の作成

venv 環境での使用を推奨いたします．
仮想環境を作成するには，下記のコマンドを実行します．

~~~console
> python -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」とし，以後も当仮想環境を「optenv」と表記します．
仮想環境の名前は任意の名前を設定できます．

### アクティベート
仮想環境を利用するには，下記のコマンドを実行します．
~~~console
> .\optenv\Scripts\activate
~~~

以後の作業はアクティベート済みのものとして進めます．

#### （参考）仮想環境の終了

仮想環境を終了するには，下記のコマンドを実行します．
~~~bash
deactivate
~~~


### インストール

```{note}
事前に pip をアップグレードすることを推奨いたします．

~~~console
> python -m pip install --upgrade pip
~~~
```

以下のコマンドを実行します．
~~~console
> python -m pip install git+https://github.com/aistairc/aiaccel.git
~~~

#### （参考）ローカルからのインストール

aiaccel をダウンロードし，ローカルからインストールすることもできます．

まず，aiaccel をダウンロードします．

~~~console
> git clone https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後，aiaccel フォルダに移動します．

~~~console
> cd aiaccel
~~~

依存環境をインストールします.

~~~console
> python -m pip install -r requirements.txt
~~~



setup.py を実行し，aiaccel をインストールします．

~~~console
> python setup.py install
~~~

aiaccel がインポートできることを確認します．

~~~console
> python
>>> import aiaccel
>>>
~~~

> [!NOTE]
> 当ガイドの内容は Windows 11 上で正常に動作することを確認済みです．

[back to top](#top)

<br>

## macOS (WIP) <a id="macOS"></a>

[back to top](#top)