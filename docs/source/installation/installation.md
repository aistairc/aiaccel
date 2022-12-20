# インストールガイド (WIP)

## Linux 向けインストールガイド

### Python-venv による仮想環境の作成

venv 環境での使用を推奨いたします．
仮想環境を作成するには，下記のコマンドを実行します．

~~~bash
python3 -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」とし，以後も当仮想環境を「optenv」と表記します．
仮想環境の名前は任意の名前を設定できます．

### アクティベート

仮想環境を利用するには，下記のコマンドを実行します．
~~~bash
source optenv/bin/activate
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

~~~bash
python3 -m pip install --upgrade pip
~~~
```


aiaccel は下記コマンドでインストールできます．
~~~bash
python3 -m pip install git+https://github.com/aistairc/aiaccel.git
~~~

aiaccel がインポートできることを確認します．

~~~bash
python3
import aiaccel
~~~

#### （参考）ローカルからのインストール

aiaccel をダウンロードし，ローカルからインストールすることもできます．

まず，aiaccel をダウンロードします．

~~~bash
git clone https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後，aiaccel フォルダに移動します．

~~~bash
cd aiaccel
~~~

依存環境をインストールします．

~~~bash
python3 -m pip install -r requirements.txt
~~~

setup.py を実行し，aiaccel をインストールします．

~~~bash
python3 setup.py install
~~~



<br>

## ABCI 向けインストールガイド

### Python 環境の構築
まず，[ABCIユーザーズガイド](https://docs.abci.ai/ja/python)に従って，python の環境を構築してください．
~~~bash
module load gcc/11.2.0
module load python/3.8/3.8.13
python3 -m venv optenv
source optenv/bin/activate
~~~

### Python-venv による仮想環境の作成

venv 環境での使用を推奨いたします．
仮想環境を作成するには，下記のコマンドを実行します．

~~~bash
python3 -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」とし，以後も当仮想環境を「optenv」と表記します．
仮想環境の名前は任意の名前を設定できます．

### アクティベート

仮想環境を利用するには，下記のコマンドを実行します．
~~~bash
source optenv/bin/activate
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

~~~bash
python3 -m pip install --upgrade pip
~~~
```


aiaccel は下記コマンドでインストールできます．
~~~bash
python3 -m pip install git+https://github.com/aistairc/aiaccel.git
~~~

aiaccel がインポートできることを確認します．

~~~bash
python3
import aiaccel
~~~

#### （参考）ローカルからのインストール

aiaccel をダウンロードし，ローカルからインストールすることもできます．

まず，aiaccel をダウンロードします．

~~~bash
git clone https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後，aiaccel フォルダに移動します．

~~~bash
cd aiaccel
~~~

依存環境をインストールします．

~~~bash
python3 -m pip install -r requirements.txt
~~~

setup.py を実行し，aiaccel をインストールします．

~~~bash
python3 setup.py install
~~~


<br>

## Windows 向けインストールガイド


### 準備1: Git のインストール

github 経由で aiaccel をインストールする場合，[git](https://gitforwindows.org/) がインストールされている必要があります．
あらかじめインストールしてください．

### 準備2: Execution Polisy の設定

PowerShell を使用して仮想環境を作る場合，設定によっては仮想環境をアクティベートするスクリプトが実行できせん．

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

<br>

## MacOS 向けインストールガイド (WIP)
