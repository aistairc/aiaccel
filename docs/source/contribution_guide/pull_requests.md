# Pull Requests

When you want to the modified code to be reflected in the repository, please execute a pull request.

### Procedures

- Please fork aiaccel repository on GitHub.
- After forking, run `git clone` command for aiaccel repository.

~~~bash
git clone https://github.com/[YOUR USERNAME]/aiaccel.git
~~~

### Developments
- Update a local repository to the latest version.

~~~bash
git checkout main
git pull upstream main
~~~

- Make a branch.

~~~bash
git checkout -b feature/add-new-feature
~~~

- Commit on local by using `git add` and `git commit` command as you progress.

  - The commit message describes the motivation for the change, the nature of the bug, or details the enhancement.
  - The message should be written in such a way that their contents can be  understood without refering code.


### Submitting

Before submitting a pull request, confirm the following:
- Did you discuss it with other developer on issues in advance?
- Can it be distributed under the MIT licence?
- Is there appropriate [unit tests](#test)?
- Can the [unit tests](#test) be run on local?
- Does the public function have a docstring?
- Can the [documentation](#documentation-wip) be rendered correctly?
- Is the [coding style](#coding-style) appropriate?
- Is the commit message appropriate?
- For larger commit, please provide the example (docs/source/examples) and the description of module level.
- If you are adding complied codes, have you modified setup.py?

After confirming above, do following:
- Push changes to the fork on GitHub.

~~~bash
git push origin feature/add-new-feature
~~~

- Enter your GitHub username and password.
- Move to the GitHub web page and write the title and message, noting the following.
  - Title
    - Briefly describe the changes.
    - Codes should be enclosed in backquotes.
    - Do not end with a period.
  - Descriptions
    - Write the motivation.
    - Write the changes.
    - If the related issues can be closed, please close it with `Close #N`.
    - If work-in-progress, write the remaining tasks.
- Submit the pull request.

## Review processes

- Other developers can contribute comments to improve implementations, documents, and coding styles in the pull request.
- When updating codes in the pull request, please commit the changes in the local repository and push the changes to the fork only if they have been successfully tested in the local environment.
- If the pull request has been reviewed and approved by at least one member of the aiaccel development team, it will be merged into the main branch.

# Documentation (WIP)

## Docstrings

- Write a basic description of the implemented functions, the types and meanings of parameters and return values, and examples of their usage.
- Write in accordance with the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- See also [Coding Conventions](#coding-style).

## Documentation

- Create source files for documentation in a directory under docs.
- The recommended document file format is markdown format.
- Create documentation for any major feature additions.

## Confirming rendering

If you have added, changed, or modified documents, make sure that it renders correctly in the local environment.
Move to the aiaccel directory and execute the following command to generate an API reference.

~~~bash
cd aiaccel
sphinx-apidoc -f -o docs/source/api_reference aiaccel
~~~

Move to aiaccel/docs and build html files to see how the document is rendered.

~~~bash
cd docs
make html
~~~

The built HTML format files are generated under docs/build/html.
Execute the following command in aiaccel/docs to generate multilingual documents.

~~~bash
make gettext
sphinx-intl update -p build/gettext -l en -l ja
~~~

# Tests

## Adding tests

- aiaccel uses pytest for testing.
- Create a directory for unit test under tests directory.
  - The directory structure under aiaccel/tests/unit corresponds to that under aiaccel/aiaccel, except for a few modeules such as config.py. For example, the test for aiaccel/aiaccel/optimizer/abstract_optimizer.py is aiaccel/tests/unit/optimzier_test/test_abstract_optimizer.py.
- If you have added a new feature or bug fix, please create codes for testing.


## Running tests (WIP)

- Move to the aiaccel directory and execute the following command to run all codes for testing on the local environment.

~~~bash
cd aiaccel
pytest
~~~

- Specify a file name as an argument to run only specific code.

~~~bash
pytest tests/unit/optimizer_test/test_abstract_optimizer.py
~~~

- In addition, execute the following command to check coding styles.

~~~bash
pycodestyle aiaccel examples
flake8 aiaccel examples
~~~


## Coverages

No strict criteria for code coverage have been set, but this value should be fully considered when designing test. Plase note the following cases.

- Significantly lower overall score.
- Abnormally low coverage of a class or module.
- Test does not cover a specific branch of the if statement.

### Measurement coverages

Run pytest with the option `--cov` to measure C0 coverage.

~~~bash
pytest --cov=aiaccel
~~~

- Replace `aiaccel` with the appropriate path to measure only the coverage of a specific test code.
- Run pytest with the option `--cov` and `--cov-branch` to measure C1 coverage.

~~~bash
pytest --cov=aiaccel --cov-branch
~~~

# Coding styles

## Basic rules

- Write source codes for aiaccel in Python.
- Coding style should follow PEP8.
  - Validate the coding style by using pycodestyle and flake8 in aiaccel.
  - See also Docstrings below.
- Write type hints whenever possible, but there is no type hint validation in aiaccel.
  - When using a built-in, e.g. `list` as a type hint, run future-import to support Python 3.8 in aiaccel.
- Use [`numpy.random.RandomState`](https://numpy.org/doc/1.16/reference/generated/numpy.random.RandomState.html) to generate a random value and maintain the compatibility with [Optuna](https://github.com/optuna/optuna) used by aiaccel.

## Docstrings

Basically, write docstrings in accordance with the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). However, please note the following exceptions.

- Docstrings for each module are not necessarily required.
- In `Args:` section, a parameter type is described in parentheses after a parameter name.
- Add `Example:` section as needed.
- Include the docstring of `__init__` method in a docstring of class. Do not write it under `__init__`.
- Use sphinx-style links to Python objects.
- Using VSCode as an editor, [autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring) is useful for generating docstrings.

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