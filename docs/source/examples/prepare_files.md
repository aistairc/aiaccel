# 作業ファイルの準備

ワークスペースを作成し，移動します．
```console
> mkdir your_workspace_directory
> cd your_workspace_directory
```
リポジトリのクローンを取得し，ディレクトリ examles のコピーを作成します．
```console
> git clone https://github.com/aistairc/aiaccel.git 
> cp -R your_workspace_directory/aiaccel/examples .
```
examples 下には以下のディレクトリが存在します．
- benchmark
- resnet50_cifar10
- schwefel
- sphere
- styblinski-tang
- wrapper_sample

使用するディレクトリに移動して設定を行い， aiaccel を実行します．
例えば sphere に移動するには，以下のようにします．
```console
> cd examples/sphere
```