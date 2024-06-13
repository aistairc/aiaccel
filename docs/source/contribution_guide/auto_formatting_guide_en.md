## Auto-formatting by pre-commit

This guide describes how to format automatically.

pre-commit is a tool that check a commit by executing a hook accoring to pre-commit configuration. It automatically points out issues on source code when creating commit by executing the command `git commit`. `.pre-commit-config.yaml` is a configuration file which describes hooks used for auto-formatting, such as ruff and mypy.

### Installing software

Run `git clone` command at any location to download aiaccel, and then move to directly under the aiaccel directory. Execute the following command to install tools such as pre-commit and ruff.

~~~bash
cd /path/to/aiaccel
pip install .[dev]
~~~

### Checking configuration files

Check pyproject.toml and .pre-commit-config.yaml before executing command `git commit`.
Add descriptions in configuration files if you need other hooks.

### Enabling pre-commit

Execute the following command. pre-commit run automatically on `git commit`.

~~~bash
pre-commit install
~~~

Setting is done. Now you can do auto-formatting when running `git commit`. If there are issues on source code, `git commit` stops automatically. Run `git commit` again after fixing some issues.

> [!NOTE]
> ### How to run `git commit` witout pre-commit
>
> ~~~bash
> git commit --no-verify
> ~~~

> [!NOTE]
> ### How to run pre-commit regardless of `git commit`
>
> ~~~bash
> pre-commit run -a
> ~~~

## Static analysis tool

### Running ruff
~~~bash
ruff format aiaccel
ruff check --fix aiaccel
~~~

### Running mypy
~~~bash
mypy aiaccel
~~~