# Linux向けインストールガイド(WIP)

# ABCIでのインストールガイド(WIP)

# Windows向けインストールガイド(WIP)

# MacOS向けインストールガイド(WIP)
numpyのインストール時にエラーで停止する場合、OpenBLASをインストールし以下のパスを環境変数に指定してインストールしてください。

~~~bash
OPENBLAS="$(brew --prefix openblas)" pip install -r requirements.txt
~~~