# aiaccel
A hyperparameter optimization library for the ABCI.
Supported optimization algorithms are random search, grid search, sobol sequence, nelder mead and TPE.

# Installation
The software can be installed using `setup.py`.
~~~
python setup.py install
~~~

# Requirements
Require Python 3 (>= 3.8.13).

# Dependencies
Required libraries can be installed using `pip`.
If you install the software using `setup.py`, you need not install using `pip`, because `setup.py` installs the libraries.
~~~
pip install -r requirements.txt
~~~

# How to run examples

## Abstract of examples

In examples directory, five kinds of examples can be tested.
The directory name means the target application to test.
For example, `sphere` directory is sphere function.

This software supports two environments: local and abci, and five search algorithm: random search, grid search, sobol sequence, nelder mead and TPE.
These environments and algorithm can be changed in configuration file.
Please edit the configuration file in the example directory if you wanto to run your choices.

- Environment
  - Environment local means that the software runs on your local computer.
  - Environment abci means that you access to [ABCI](https://abci.ai/) and run your program on it.
- Search algorithm
  - Search algorithm: random search means that the software searches hyperparameters at random.
  - Search algorithm: grid search means that the software searches exhaustive search over hyperparameters.
  - Search algorithm: sobol sequence means that the software searches hyperparameters following the [sobol sequence](https://en.wikipedia.org/wiki/Sobol_sequence).
  - Search algorithm: nelder mead means that the optimizer searches hyperparameters following the [nelder mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method).
  - Search algorithm: TPE means that the optimizer searches hyperparameters following the [TPE(Tree-structed Parzen Estimator Approach)](https://www.lri.fr/~kegl/research/PDFs/BeBaBeKe11.pdf).

## sphere tutorial on local computer
This tutorial describes how to run in examples/sphere directory.


1. Install the virtualenv and create the virtual environment. (Optional)
~~~
    > pip install virtualenv
    > virtualenv venv
    > source venv/bin/activate
~~~

2. Download the repository. This example assumes as running on bash terminal.
~~~
    > git clone https://github.com/aistairc/aiaccel.git

    > ls
    opt
~~~

1. Install the requirements and the software
~~~
    > cd opt
    > pip install cython numpy pytest
    > python setup.py install
~~~

4. Prepare the workspace and copy the sphere directory.
~~~
    > cd your_workspace_directory
    > cp -R cloned_directory/opt/examples .
    > cd examples
    > ls
    sphere

    > cd sphere
    > ls
    config.yaml         job_script_preamble.sh         user.py
~~~

1. Run
~~~
    > python -m aiaccel.start --config config.yaml
~~~

6. After finishing master, check the results.
~~~
    > ls /tmp/work
    abci_output         alive               hp                  lock
    log                 resource            result              runner
    state               verification

    > ls /tmp/work/result/final_result.result
    /tmp/work/result/final_result.result
~~~

7. If you want to change settings, please edit config.yaml file.
~~~
    vi config.yaml
~~~

8. If you want to re-run the optimization, please move the `work_aiaccel` directory.
~~~
    > mv /tmp/work /tmp/work/work_aiaccel_200101
~~~

9. You can clean the workspace directory using `clean_workspace.py`.
~~~
    > python -m aiaccel.bin.clean
~~~

10. If you want to stop the optimization, please use `stop_all.py`.
~~~
    > python -m aiaccel.bin.stop
~~~

## sphere tutorial on ABCI
This tutorial describes how to run examples/sphere on ABCI.

1. At first, please setup python environment following the ABCI Users Guide:
~~~
    module load python/3.8/3.8.13
    python3 -m venv work
    source work/bin/activate
~~~

2. Prepare sphere workspace. Thease processes are same with [sphere tutorial on local computer](#sphere tutorial on local computer) from 2 to 4.

3. Please confirm the configuration file before running master.
```yaml
resource:
  type: "ABCI"
  num_node: 4
```
1. Run
~~~
    > python -m aiaccel.start --config config.yaml
~~~

5. The other processes are same with [sphere tutorial on local computer](#sphere tutorial on local computer) from 6 to 10.
If you want to check the running jobs, please refer the [ABCI User Guide](https://docs.abci.ai/ja/).

# Inputs
The inputs for the system is files as follows:
* config.yaml
  * A input file for the master program of this software.
* job_script_preamble.sh
  * A preamble of shell script to run your target programs on ABCI environment.
* wrapper.py
  * A wrapper program defines how the software sets hyperparameters and inputs, and how the software gets the results from your target programs.
* and your target programs
  * Your target programs which need to get hyperparameters with arguments in CUI environment. Please don't forget the specify your target program in `wrapper.py`.

# Outputs
The outputs of the system is placed on a directory.
The directory can be set in config.yml.
In the directory, the system makes sub directories as follows:

* abci_output
  * stdout/stderr outputs from ABCI are stored in this directory.
* alive
  * The alive files for each module are placed on this directory. If a alive file for master program is deleted, the master program will stop.
  * files
    * master.yml
    * optimizer.yml
    * scheduler.yml
* hp/finished
  * A hyperparameter directory which finished hyperparameters are placed.
  * files
    * for example, ${ID}.yml
* hp/ready
  * A hyperparameter directory which ready to starting hyperparameters are placed.
* hp/running
  * A hyperparameter directory that currently running hyperparameters are placed.
* lock
  * Lock files are stored this directory.
* resource
  * A resource file that some aiaccel modules share, is placed in this directory.
  * files
    * resource.res
* result
  * Results from user target programs are stored in this directory.
* runner
  * The batch job scripts used by ABCI, is placed in this directory.

# Configuration
The most of the settings are written in the configuration file.
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 600

resource:
  type: "local"
  num_node: 4

ABCI:
  group: "[group]"
  job_script_preamble: "./job_script_preamble.sh"
  job_execution_options: ""

optimize:
  search_algorithm: "nelder-mead"
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: 0.0
      upper: 5.0
      initial: 1.0
    -
      name: "x2"
      type: "uniform_float"
      lower: 0.0
      upper: 5.0
      initial: 1.0
```

- generic
  - workspace:
    * The path to the workspace directory.
  - job_command
    * The command to run the user program from the software.
  - batch_job_timeout
    * The timeout period in seconds for all jobs to complete.
- resource:
  - type
    * The type supports `local` and `abci`. It means that the optimization runs on local computer or ABCI.
  - num_node
    * It is the number how many nodes can be used to allocate a job.
- ABCI:
  - job_script_preamble
    * A wrapper schell script to run the software on ABCI.
  - group
    * The group name in ABCI environment. This may be the format such as "gaa****".
- optimizer
  - search_algorithm
    * The search algorithms are available `random`, `grid`,`sobol`, `nelder-mead` and `tpe`.
  - goal
    * The evaluation goal setting. It supports `minimize` and `maximize`.
  - trial_number
    * The number how many hyperparameters are run.
  - rand_seed
    * The random seed for random, and numpy.random.
  - "parameters"
- hyperparameter:
  - name  
    * The name of the hyperparameter．
  - type 
    * The data type of hyperparameter are available `uniform_float`, `uniform_int`, `categorical`, `ordinal` and `sequential`.
  - lower
    * The lower limit of hyperparameter．
  - upper 
    * The upper limit of hyperparameter
  - initial 
    * The initial value of hyperparameter．
  - step  
    * The resolution for grid search．
  - log
    * Logarithmic settings for grid search．
  - base 
    * Logarithmic settings for grid search．
  - comment
    * free writing item．

# Tests
Currently the software supports unit tests, code style tests and integration tests.
To run the tests, the installation is required.

1. Unit test
~~~
    > cd opt
    > pytest -v --cov=aiaccel tests/unit
~~~

2. Code style (pycodestyle) test
~~~
    > cd opt
    > pytest -v --pycodestyle aiaccel examples tests
~~~

3. Code style (flake8) test
~~~
    > cd opt
    > pytest -v --flake8 aiaccel examples tests
~~~

4. Integration tests
The integration tests test optimizer algorithms using `examples/sphere`.
~~~
    > cd opt
    > pytest -v tests/integration
~~~

# Future works
- Get accurate timestamps and measure the running time.
- Parallel execution on one interactive node.
- Support databases to store the states and results.
- Manage multiple master programs.
- Make references such as docstring.
- Implement a dashboard as web applications.
- Implement the API to control the optimization.
- If the optimizer has the state and the total hyperparameters are greater than requested running number, how the optimizer determines the order of generation of the hyperparameters for example in grid search case.
- Error checks for config, parameters and so on. For example, min_value and max_value...
- Add a trouble shooting document, about the [Argparse issue](https://stackoverflow.com/questions/9025204/python-argparse-issue-with-optional-arguments-which-are-negative-numbers).
- Build executable binaries.
- Type hints.
- Support pathlib.
- Synchronization of re-run.
- Error detections for user applications.
- Refactor the codes. Line feeds manner, single quotation and double quotation, and so on.
- Need a good name!

# Acknowledgment
Part of this software was developed in a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO). <BR>
The aiaccel is built with the help of Optuna.
