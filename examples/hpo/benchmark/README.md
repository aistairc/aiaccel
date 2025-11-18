# Verification Code for NelderMeadSampler Using COCO

## 1. File Structure

### nelder-mead
### nelder-mead-subTPE
### TPE

- These directories store the CSV files containing the optimization results for each sampler.

### experiment_coco.py

- This is the main code for validation using COCO.
- It is designed to run with dimensions * 20 steps and 10 parallel executions.
- Upon execution, the results from Optuna are output to `optuna_csv`, and results for each parallel step are output to `step_csv`.

### main_parallel_coco.py

- This code uses `job_dispatcher` to submit jobs for each sampler, function, and dimension.

### objective.sh

- This is a script for qsub submission used by `job_dispatcher`.

### plot.py

- This code uses matplotlib to graph the results of each sampler.
- It references the `optuna_csv` files in each sampler's directory.

### result_bbob_dim_vs_value-fopt_parallel.png

- This is a graph image visualizing the validation results output by running `plot.py`.
- The graph displays 24 benchmark functions with the number of dimensions on the horizontal axis and the mean and standard deviation of optimization results on the vertical axis.

## 2. Execution Instructions

- Install aiaccel and activate the virtual environment.

- Install COCO.
  - For details, please refer to the following GitHub repository:
    https://github.com/numbbo/coco

- Please replace the # activate environment and the `job_group` in `job_config.py` with the appropriate commands and IDs.
- When you run above command on ABCI, the validation for each sampler will be executed.

```
aiaccel-job pbs --config job_config.yaml cpu --walltime 4:00:00 main_parallel_coco.log -- python3.13 main_parallel_coco.py
```

- The results are saved in `optuna_csv` and `step_csv` under each directory.

- To run `plot.py`, you need to install pandas and matplotlib.

```
pip install pandas matplotlib
python plot.py
```

## 3. Checking the Results

- The validation results for each sampler are output to `optuna_csv` and `step_csv` under the corresponding sampler's directory.
- The visualization results from `plot.py` are output to `result_bbob_dim_vs_value-fopt_parallel.png`.
  - From the visualization results, it can be observed that `nelder-mead_subTPE` tends to yield better results during parallel execution. However, in some cases, `nelder-mead` may perform better depending on the function.
