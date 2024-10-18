# Examples of NelderMeadSampler

## 1. File Structure

### example.py

- This code demonstrates the general usage of NelderMeadSampler.
- The optimization target is the benchmark function Sphere (this applies to the following examples unless otherwise specified).

### example_parallel.py

- This code demonstrates the usage of NelderMeadSampler in parallel execution.
- Parallel execution is enabled by setting the NelderMeadSampler argument block=True and the study.optimize argument n_jobs=3.
- By enabling parallel execution, the initial point calculation and the computation during shrinking can be parallelized, leading to faster execution compared to serial execution.

### example_enqueue.py

- This code demonstrates the usage of NelderMeadSampler with optuna.study.enqueue_trial.
- Utilizing the ask-tell interface, random parameters are explored using enqueue_trial when NelderMeadSampler fails to output parameters.

### example_sub_sampler.py

- This code demonstrates the usage of the sub_sampler feature of NelderMeadSampler.
- When sub_sampler=optuna.samplers.TPESampler is set as an argument for NelderMeadSampler, TPESampler is used for exploration when NelderMeadSampler fails to output parameters.
- When using the sub_sampler function, the argument block=False must be set even if it is parallel. (Parallel execution is possible even with block=False.)

### coco

- This directory contains code for verifying NelderMeadSampler using the black-box optimization evaluation framework coco.
- For details, please refer to the README.md in the relevant directory.

## 2. Instructions

- After installing aiaccel and activating the virtual environment, run the corresponding file.

```bash
python example.py
```

## 3. Checking Results

- The execution results of the example code are displayed in the standard output.
