# Basic usage of aiaccel-hpo
## Getting started
```bash
aiaccel-hpo optimize params.x1=[0,1] params.x2=[0,1] n_trials=100 -- python objective.py --x1={x1} --x2={x2} {out_filename}
```
A workspace `aiaccel-hpo_***` will be created, and you will get outputs something like:
```plain
[I 2025-08-11 23:19:09,865] A new study created in RDB with name: aiaccel-hpo
[I 2025-08-11 23:19:10,159] Trial 3 finished with value: 1.199387 and parameters: {'x1': 0.7651250790017732, 'x2': 0.7835626174783031}. Best is trial 3 with value: 1.199387.
[I 2025-08-11 23:19:10,179] Trial 7 finished with value: 0.13314 and parameters: {'x1': 0.28734908107070123, 'x2': 0.22487902368959145}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,190] Trial 6 finished with value: 0.854472 and parameters: {'x1': 0.5282599103748785, 'x2': 0.7585598197415366}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,202] Trial 1 finished with value: 0.241872 and parameters: {'x1': 0.490180501594382, 'x2': 0.03993315068224257}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,215] Trial 0 finished with value: 0.267713 and parameters: {'x1': 0.1980697319379605, 'x2': 0.4779975949100864}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,225] Trial 4 finished with value: 0.223939 and parameters: {'x1': 0.42026494162838846, 'x2': 0.2175229115138555}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,238] Trial 2 finished with value: 1.099494 and parameters: {'x1': 0.8241694903158984, 'x2': 0.6482584131495605}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,249] Trial 5 finished with value: 0.489583 and parameters: {'x1': 0.6165836750706742, 'x2': 0.33076842412638574}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,259] Trial 8 finished with value: 1.624779 and parameters: {'x1': 0.9942998803639703, 'x2': 0.7975879798801359}. Best is trial 7 with value: 0.13314.
[I 2025-08-11 23:19:10,273] Trial 9 finished with value: 1.497936 and parameters: {'x1': 0.9652436480683448, 'x2': 0.7524894798074183}. Best is trial 7 with value: 0.13314.
```

You can also run the optimization by specifying a config file:
```bash
aiaccel-hpo optimize --config experiment/config.yaml
```

In this case, `experiment/` is used as a workspace.

You can also combine `aiaccel-hpo` with `aiaccel-job` as:
```bash
aiaccel-hpo optimize params.x1=[0,1] params.x2=[0,1] n_trials=100 n_max_jobs=10 -- \
    aiaccel-job local cpu {config.working_directory}/{job_name}.log -- \
        python objective.py --x1={x1} --x2={x2} {out_filename}
```