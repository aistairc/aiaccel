(coding-style)=
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
