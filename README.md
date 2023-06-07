<div align="center"><img src="https://raw.githubusercontent.com/aistairc/aiaccel/master/docs/image/logo_aiaccel.png" width="400"/></div>

# aiaccel: an HPO library for ABCI
[![GitHub license](https://img.shields.io/github/license/aistairc/aiaccel.svg)](https://github.com/aistairc/aiaccel)
[![Supported Python version](https://img.shields.io/badge/Python-3.8-blue)](https://github.com/aistairc/aiaccel)
[![Documentation Status](https://readthedocs.org/projects/aiaccel/badge/?version=latest)](https://aiaccel.readthedocs.io/en/latest/)
![CI status](https://github.com/aistairc/aiaccel/actions/workflows/actions.yaml/badge.svg)

[**日本語  (Japanese)**](https://github.com/aistairc/aiaccel/blob/main/README_JP.md)


A hyperparameter optimization library for [AI Bridging Cloud Infrastructure (ABCI)](https://abci.ai/).
This software solves hyperparameter optimizations related to AI technologies including deep learning and multi-agent simulation.
The software currently supports five optimization algorithms: random search, grid search, sobol sequence, nelder-mead method, and TPE.

# Installation
The software can be installed using `pip`.
~~~bash
> pip install git+https://github.com/aistairc/aiaccel.git
~~~

# Getting started

## Running on a local computer
An example for optimizing a simple function (i.e., sphere function) on a local computer.


0. (Optional) Install [Virtualenv](https://virtualenv.pypa.io/en/latest/) and create a virtual environment. 
    ~~~bash
    > python3 -m venv optenv
    > source optenv/bin/activate
    ~~~

1. Install `aiaccel`
    ~~~bash
    > pip install git+https://github.com/aistairc/aiaccel.git 
    ~~~

2. Create a workspace and copy the sphere example on the repository.
    ~~~bash
    > mkdir your_workspace_directory
    > cd your_workspace_directory
    > git clone https://github.com/aistairc/aiaccel.git 
    > cp -R ./aiaccel/examples .
    > cd examples
    > ls
    sphere

    > cd sphere
    > ls
    config.yaml         user.py
    ~~~

3. Run the parameter optimization
    ~~~bash
    > aiaccel-start --config config.yaml
    ~~~

    or

    ~~~bash
    > python -m aiaccel.cli.start --config config.yaml
    ~~~

    Tips: You can start after cleaning the workspace directory using `--clean`.
    ~~~bash
    > aiaccel-start --config config.yaml --clean
    ~~~

4. Wait for the program to finish and check the optimization results.
    ~~~bash
    > ls ./work
    abci_output         alive               hp                  lock
    log                 result              runner              state

    > cat ./work/result/final_result.result
    ~~~

5. If you want to change configurations, edit `config.yaml`.
    ~~~bash
    > vi config.yaml
    ~~~

## Running on ABCI
This tutorial describes how to run examples/sphere on ABCI.

1. First, setup python environment following [the ABCI Users Guide](https://docs.abci.ai/en/python/):
    ~~~bash
    > module load python/3.11/3.11.2
    > python3 -m venv optenv
    > source optenv/bin/activate
    ~~~

2. Prepare the workspace by following Steps 1 and 2 in [Running on a local computer](https://github.com/aistairc/aiaccel#Running-on-a-local-computer).

3. Please confirm the configuration file before running master.
    ```yaml
    resource:
        type: "abci"
        num_workers: 4
    ```

4. Run on an (interactive) job
    ~~~bash
    > aiaccel-start --config config.yaml
    ~~~

5. If you want to check the running jobs, please refer the [ABCI User Guide](https://docs.abci.ai/en/job-execution/#show-the-status-of-batch-jobs).


## Others
- Check the progress
    ~~~bash
    > aiaccel-view --config config.yaml
    ~~~

- Display simple graphs
    ~~~bash
    > aiaccel-plot --config config.yaml
    ~~~

- Output results to workspace/results.csv.
    ~~~bash
    > aiaccel-report --config config.yaml
    ~~~


# Acknowledgement
* Part of this software was developed in a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
* aiaccel is built with the help of Optuna.
