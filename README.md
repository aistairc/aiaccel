# aiaccel
A hyperparameter optimization library for the ABCI.
The software solves hyperparameter optimizations related to AI technologies such as deep learning and multi-agent simulation.
Currently the software supports five optimization algorithms as follows: random search, grid search, sobol sequence, nelder mead and TPE.

[** README in Japanese **](https://github.com/aistairc/aiaccel/blob/main/README_JP.md)

# Installation
The software can be installed using `setup.py`.
~~~
pip install git+https://github.com/aistairc/aiaccel.git
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

In examples directory, six kinds of examples can be tested.
The directory name means the target application to test.
For example, `sphere` directory is sphere function.

This software supports two environments: local and abci, and five search algorithm: random search, grid search, sobol sequence, nelder mead and TPE.
These environments and algorithm can be changed in configuration file.
Please edit the configuration file in the example directory if you want to run your choices.

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

2. Install the requirements and the software
~~~
    pip install cython numpy pytest
    pip install git+https://github.com/aistairc/aiaccel.git 
~~~

3. Prepare the workspace and copy the sphere directory.
~~~
    > cd your_workspace_directory
    > cp -R cloned_directory/aiaccel/examples .
    > cd examples
    > ls
    sphere

    > cd sphere
    > ls
    config.yaml         job_script_preamble.sh         user.py
~~~

4. Run
~~~
    > python -m aiaccel.start --config config.yaml
~~~

 You can clean the workspace directory using `--clean`.
~~~
    > python -m aiaccel.start --config config.yaml --clean
~~~

5. After finishing master, check the results.
~~~
    > ls /tmp/work
    abci_output         alive               hp                  lock
    log                 resource            result              runner
    state               verification

    > ls /tmp/work/result/final_result.result
    /tmp/work/result/final_result.result
~~~

6. If you want to change settings, please edit config.yaml file.
~~~
    vi config.yaml
~~~

7. If you want to re-run the optimization, please move the `work_aiaccel` directory.
~~~
    > mv /tmp/work /tmp/work/work_aiaccel_200101
~~~

<!-- 9. You can clean the workspace directory using `clean_workspace.py`.
~~~
    > python -m aiaccel.bin.clean
~~~

10. If you want to stop the optimization, please use `stop_all.py`.
~~~
    > python -m aiaccel.bin.stop
~~~ -->

## Sphere tutorial on ABCI
This tutorial describes how to run examples/sphere on ABCI.

1. At first, please setup python environment following the ABCI Users Guide:
~~~
    module load python/3.8/3.8.13
    python3 -m venv work
    source work/bin/activate
~~~

2. Prepare sphere workspace. Thease processes are same with [sphere tutorial on local computer](https://github.com/aistairc/aiaccel#sphere-tutorial-on-local-computer) from 2 to 4.

3. Please confirm the configuration file before running master.
```yaml
resource:
  type: "ABCI"
  num_node: 4
```

4. Run
~~~
    > python -m aiaccel.start --config config.yaml
~~~

5. The other processes are same with [sphere tutorial on local computer](https://github.com/aistairc/aiaccel#sphere-tutorial-on-local-computer) from 6 to 10.
If you want to check the running jobs, please refer the [ABCI User Guide](https://docs.abci.ai/ja/).

# Acknowledgment
aiaccel is built with the help of Optuna.
Part of this software was developed in a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
