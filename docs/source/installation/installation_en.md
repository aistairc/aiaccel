# Installation guide

This guide describes how to install aiaccel in the followings.

- [Linux](#linux)
- [ABCI](#abci)
- [Windows](#windows)
- [macOS](#macos)


## Linux <a id="linux"></a>

### Installing Git

You should install Git and set up it in advance. You can install it any way you like, such as using `apt` commands.

~~~bash
sudo apt update
sudo apt install git
~~~

### Creating virtual environments

We recommend using venv, the Python standard library for creating virtual environments.
Execute the following command.

~~~bash
python3 -m venv optenv
~~~

In this guide, the virtual environment is named "optenv", but you can name it something else.

### Activating virtual environments

Execute the following command for activating the virtual environment.

~~~bash
source optenv/bin/activate
~~~

The following procedures assume that the virtual environment is activated.


> [!NOTE]
> You can deactivate virtual environments by using the following command.
> ~~~bash
> deactivate
> ~~~


### Installing aiaccel

> [!NOTE]
> We recommend you upgrade pip and setuptools beforehand.
> Execute the command `python -m pip --version` in order to verify that pip is running in the virtual environment, and then upgrade them. Be careful not to use the pip on other environments.
>
> ~~~bash
> python -m pip install --upgrade pip setuptools
> ~~~

You can install aiaccel by executing the following command.

~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

#### (FYI) Other installation methods

aiaccel can also be installed in other ways. Prepare a virtual environment in advance by following the guide above.

Download aiaccel files from github repository to a directory of your choise.

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~

After doning, move to the directory.

~~~bash
cd aiaccel
~~~

Install aiaccel by executing the following command. The command installs it by using pyproject.toml on local directory.

~~~bash
python -m pip install .
~~~

Do not forget to verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

> [!NOTE]
> The contents of this guide have been verified to work properly with WSL2 (Windows Subsystem for Linux 2).

[back to top](#top)

<br>

## ABCI [(AI Bridging Cloud Infrastructure)](https://abci.ai/en/about_abci/) <a id="abci"></a>

ABCI has Node(V) and Node(A), each running a different Linux distribution. See [ABCI System Overview](https://docs.abci.ai/en/system-overview/#software).

### Creating Python environments

Prepare a Python environment by following [ABCI User Guide](https://docs.abci.ai/en/python).
Set up Python on the ABCI system by using `module` command.

~~~bash
module load python/3.10
~~~

### Creating virtual environments

We recommend using venv, the Python standard library for creating virtual environments.
Execute the following command.

~~~bash
python3 -m venv optenv
~~~

In this guide, the virtual environment is named "optenv", but you can name it something else.

### Activating virtual environments

Execute the following command for activating the virtual environment.

~~~bash
source optenv/bin/activate
~~~

The following procedures assume that the virtual environment is activated.


> [!NOTE]
> You can deactivate virtual environments by using the following command.
> ~~~bash
> deactivate
> ~~~


### Installing aiaccel

> [!NOTE]
> We recommend you upgrade pip and setuptools beforehand.
> Execute the command `python -m pip --version` in order to verify that pip is running in the virtual environment, and then upgrade them. Be careful not to use the pip on other environments.
>
> ~~~bash
> python -m pip install --upgrade pip setuptools
> ~~~

You can install aiaccel by executing the following command.

~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

#### (FYI) Other installation methods

aiaccel can also be installed in other ways. Prepare a virtual environment in advance by following the guide above.

Download aiaccel files from github repository to a directory of your choise.

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~

After doning, move to the directory.

~~~bash
cd aiaccel
~~~

Install aiaccel by executing the following command. The command installs it by using pyproject.toml on local directory.

~~~bash
python -m pip install .
~~~

Do not forget to verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

> [!NOTE]
> The contents of this guide have been verified to work properly with Interactive Node(V) on ABCI.

[back to top](#top)

<br>


## Windows <a id="windows"></a>

> [!NOTE]
> Warning: The following steps would be necessary, depending on your Windows environment.
> ### Setting an Execution Policy
>
> Security settings may prevent the virtual environment from being activated when using PowerShell to create a virtual environment.
> See [descriptions](https://learn.microsoft.com/en/powershell/module/microsoft.powershell.core/about/about_execution_policies?view=powershell-7.3).
>
> Execute the following command in order to confirm PowerShell settings.
> ```console
> > Get-ExecutionPolicy
> Restricted
> ```
> Change setting by executing the following command if `Restricted` is displayed.
> ```console
> > Set-ExecutionPolicy RemoteSigned
> ```
>
> Execute command `Get-ExecutionPolicy` again. Setting is done if `RemoteSigned` is displayed.
> ```console
> > Get-ExecutionPolicy
> RemoteSigned
> ```
>
> If you want to specifiy the scope that is affected by an execution policy, set the following.
> ```console
> > Set-ExecutionPolicy RemoteSigned -Scope Process
> ```

### Installing Git

You should install [Git](https://gitforwindows.org/) and set up it in advance.

### Creating virtual environments

We recommend using venv, the Python standard library for creating virtual environments.
Execute the following command.

~~~bash
python3.exe -m venv optenv
~~~

In this guide, the virtual environment is named "optenv", but you can name it something else.

### Activating virtual environments

Execute the following command for activating the virtual environment.

~~~bash
.\optenv\Script\activate
~~~

The following procedures assume that the virtual environment is activated.


> [!NOTE]
> You can deactivate virtual environments by using the following command.
> ~~~bash
> deactivate
> ~~~

> [!NOTE]
> Warning: Content blocked by Windows security
>
> ご使用のアンチウイルスソフト(Defender 等)の設定によっては先ほど作成した仮想環境上での pip コマンドや Python コマンド等の操作がブロックされる可能性があります．
> The commands `pip` and `python` executed on the virtual environment can be blocked depending on some anti-virus software, such as Windows Defender.

### Installing aiaccel

> [!NOTE]
> We recommend you upgrade pip and setuptools beforehand.
> Execute the command `python.exe -m pip --version` in order to verify that pip is running in the virtual environment, and then upgrade them. Be careful not to use the pip on other environments.
>
> ~~~bash
> python.exe -m pip install --upgrade pip setuptools
> ~~~

You can install aiaccel by executing the following command.

~~~bash
python.exe -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Verify that aiaccel has been successfully installed.

~~~bash
python.exe -m pip show aiaccel
~~~


#### (FYI) Other installation methods

aiaccel can also be installed in other ways. Prepare a virtual environment in advance by following the guide above.

Download aiaccel files from github repository to a directory of your choise.

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~

After doning, move to the directory.

~~~bash
cd .\aiaccel
~~~

Install aiaccel by executing the following command. The command installs it by using pyproject.toml on local directory.

~~~bash
python.exe -m pip install .
~~~

Do not forget to verify that aiaccel has been successfully installed.

~~~bash
python.exe -m pip show aiaccel
~~~

> [!NOTE]
> The contents of this guide have been verified to work properly with Windows 11.

[back to top](#top)

<br>

## macOS  <a id="macos"></a>

### Installing Git

You should install Git and set up it in advance. You can install it any way you like, such as using `homebrew`.

~~~bash
brew update
brew install git
~~~

### Creating virtual environments

We recommend using venv, the Python standard library for creating virtual environments.
Execute the following command.

~~~bash
python3 -m venv optenv
~~~

In this guide, the virtual environment is named "optenv", but you can name it something else.

### Activating virtual environments

Execute the following command for activating the virtual environment.

~~~bash
source optenv/bin/activate
~~~

The following procedures assume that the virtual environment is activated.


> [!NOTE]
> You can deactivate virtual environments by using the following command.
> ~~~bash
> deactivate
> ~~~

### Installing aiaccel

> [!NOTE]
> We recommend you upgrade pip and setuptools beforehand.
> Execute the command `python -m pip --version` in order to verify that pip is running in the virtual environment, and then upgrade them. Be careful not to use the pip on other environments.
>
> ~~~bash
> python -m pip install --upgrade pip setuptools
> ~~~

You can install aiaccel by executing the following command.

~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git@develop/v2
~~~

Verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

#### (FYI) Other installation methods

aiaccel can also be installed in other ways. Prepare a virtual environment in advance by following the guide above.

Download aiaccel files from github repository to a directory of your choise.

~~~bash
git clone --branch develop/v2 --single-branch https://github.com/aistairc/aiaccel.git
~~~

After doning, move to the directory.

~~~bash
cd aiaccel
~~~

Install aiaccel by executing the following command. The command installs it by using pyproject.toml on local directory.

~~~bash
python -m pip install .
~~~

Do not forget to verify that aiaccel has been successfully installed.

~~~bash
python -m pip show aiaccel
~~~

[back to top](#top)