# Verification Code for NelderMeadSampler using coco

## 1. File Structure

### nelder-mead
### nelder-mead_subTPE
### TPE

- These directories contain the verification code for each sampler.

   ### experiment_for_*_sampler_parallel.py

   - The main code for performing the verification using coco.
   - Assumes execution in 20 steps per dimension, with 10 parallel processes.
   - When executed, the results from optuna are output to optuna_csv, and the results for each parallel step are output to step_csv.

   ### main_parallel.py

   - This code submits jobs for each function and dimension using job_dispatcher.

   ### objective.sh

   - Qsub scripts that set the execution time for each dimension.


### plot.py

- This code uses matplotlib to plot the results of each sampler.
- It references optuna_csv in each sampler's directory.

### result_bbob_dim_vs_value-fopt_parallel.png

- A graph image visualizing the verification results.
- The graph displays the number of dimensions on the x-axis and the mean and deviation of optimization results on the y-axis, with 24 benchmark functions lined up.

## 2. Instructions

- Install aiaccel and activate the virtual environment.

- Install coco.
  - For details, refer to the following git repository:
    https://github.com/numbbo/coco

- Execute main_parallel.py in each directory to perform the verification for the corresponding sampler.
- The results will be saved in optuna_csv and step_csv in each directory.

```bash
cd nelder-mead
python main_parallel.py
```

- To run plot.py, you need to install pandas and matplotlib.

```bash
pip install pandas matplotlib
python plot.py
```

## 3. Checking Results

- The verification results for each sampler are output to optuna_csv and step_csv in the corresponding directory.
- The visualization results from plot.py are output to result_bbob_dim_vs_value-fopt_parallel.png.
  - From the visualization results, it can be seen that nelder-mead_subTPE tends to produce better results during parallel execution. However, depending on the function, nelder-mead may yield better results.