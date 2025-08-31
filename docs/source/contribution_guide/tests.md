(test)=
# Tests

## Adding tests

- aiaccel uses pytest for testing.
- Create a directory for unit test under tests directory.
  - The directory structure under `aiaccel/tests` corresponds to that under `aiaccel/aiaccel`, except for a few modeules such as config.py. For example, the test for `aiaccel/aiaccel/hpo/optuna/suggest_wrapper.py` is `aiaccel/tests/hpo/optuna/test_suggest_wrapper.py`.
- If you have added a new feature or bug fix, please create codes for testing.


## Running tests (WIP)

- Move to the aiaccel directory and execute the following command to run all codes for testing on the local environment.

~~~bash
cd aiaccel
pytest
~~~

- Specify a file name as an argument to run only specific code.

~~~bash
pytest aiaccel/tests/hpo/optuna/test_suggest_wrapper.py
~~~

- In addition, execute the following command to check coding styles.

~~~bash
ruff check
ruff format --check
mypy --config-file mypy.ini .
docstrfmt --check docs/source/
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
