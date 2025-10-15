Job Executor
============

aiaccel-job makes it easy to submit jobs in local, PBS, and SGE environments.

Getting Started
---------------

Run the following command:

.. code-block:: bash

    aiaccel-job local cpu example.log -- echo "example"

Basic Usage
-----------

Choose the appropriate settings for [local, pbs, sge] and [cpu, gpu, train] based on the
execution environment.

.. code-block:: bash

    aiaccel-job [local, pbs, sge] --config job_config.yaml [cpu, gpu, train] LOG_FILENAME -- COMMAND

Configuration
~~~~~~~~~~~~~

Basic configuration example for PBS environment:

.. code-block:: yaml
    :caption: job_config.yaml

    walltime: "1:0:0"

    script_prologue: |
        echo Job ID: $PBS_JOBID
        echo Hostname: $(hostname)

        export CUDA_VISIBLE_DEVICES=all

    qsub: "qsub -P $JOB_GROUP -l walltime={args.walltime} -v USE_SSH=1"

    cpu:
        qsub_args: "-q rt_HF -l select=1"
        job: "{command}"

    cpu_array:
        n_tasks_per_proc: 128
        n_procs: 24
        qsub_args: "-q rt_HF -l select=1 -J 1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
        job: "{command}"

    gpu:
        qsub_args: "-q rt_HF -l select=1"
        job: "{command}"

    gpu_array:
        n_tasks_per_proc: 128
        n_procs: 8
        qsub_args: "-q rt_HF -l select=1 -J 1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
        job: "CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % 8 )) {command}"

    mpi:
        n_nodes: 1
        qsub_args: >-
            -q rt_HF
            -l select={args.n_nodes}:mpiprocs=$(( {args.n_procs} / {args.n_nodes} )):ompthreads=$(( {args.n_nodes} * 96 / {args.n_procs} ))
        job: |
            source /etc/profile.d/modules.sh
            module load hpcx

            mpirun -np {args.n_procs} -bind-to none -map-by slot \\
                -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
                {command}

    train:
        qsub_args: >-
            -q $( (({args.n_gpus}==1)) && printf rt_HG || printf rt_HF )
            -l select=$(( ({args.n_gpus} + 7) / 8 )):mpiprocs=$( (({args.n_gpus}==1)) && printf 1 || printf 8 ):ompthreads=$( (({args.n_gpus}==1)) && printf 8 || printf 12 )
        job: |
            source /etc/profile.d/modules.sh
            module load hpcx

            mpirun -np {args.n_gpus} -bind-to none -map-by slot \\
                -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
                -x MAIN_ADDR=$(hostname -i) \\
                -x MAIN_PORT=3000 \\
                -x COLUMNS=120 \\
                -x PYTHONUNBUFFERED=true \\
                {command}

Usage Examples
~~~~~~~~~~~~~~

Run using the CPU in a local environment:

.. code-block:: bash

    aiaccel-job local --config job_config.yaml cpu example.log -- echo "example"

Run using the GPU in a pbs environment:

.. code-block:: bash

    aiaccel-job pbs --config job_config.yaml gpu example.log -- echo "example"

Run using the GPU in a pbs environment:

.. code-block:: bash

    aiaccel-job sge --config job_config.yaml gpu example.log -- echo "example"
