########################
 Managing Job Workloads
########################

``aiaccel-job`` launches training, evaluation, and batch workloads through a single CLI.
The same command structure works on a local machine or through scheduler-specific
backends such as PBS, SGE, and Slurm. The job runner centers on three ideas:

- Keep scheduler logic in YAML templates so queue options, launch commands, and module
  setup can be reviewed and version-controlled.
- Use one CLI across single jobs, array-style fan-out, MPI jobs, and distributed
  training launches.
- Switch environments by changing only the backend name and config file, while keeping
  the payload command unchanged.

Its design is inspired by the job-management style used in `Kaldi
<https://kaldi-asr.org/>`_ and `ESPnet <https://espnet.github.io/espnet/>`_, while
adapting the workflow to aiaccel's configuration system. This guide explains the command
workflow, the YAML templates behind it, and the scheduler-specific tweaks that keep the
interface consistent across backends.

*************
 Basic Usage
*************

``aiaccel-job`` works like any other CLI: select a backend (``local``, ``pbs``, ``sge``,
or ``slurm``), pick a mode, point to a log file, and provide the command payload.
Configuration files supply the templates, but the invocation style stays nearly
identical across backends.

Running a job
=============

The basic invocation consists of the backend, the mode, ``LOG_FILENAME``, and the
command to run:

.. code-block:: bash

    aiaccel-job local --config job_config.yaml \
        cpu logs/example.log -- \
        python train.py --epochs 10

``aiaccel-job`` writes the rendered script next to the log file (for example,
``logs/example.sh``) and uses the selected backend to execute it. Omit ``--config`` to
fall back to the default backend config shipped with aiaccel, or set
``AIACCEL_JOB_CONFIG`` for a global default.

Command-line interface
======================

Every backend accepts ``--config`` (YAML path), ``--print_config`` (print the resolved
config before continuing), and ``--walltime`` (override scheduler defaults). The
positional ``LOG_FILENAME`` selects where logs and generated scripts are stored, and
``COMMAND`` captures the payload. Modes add the following options:

``aiaccel-job [backend] cpu [--n_tasks=N --n_tasks_per_proc=M --n_procs=K] LOG_FILENAME -- COMMAND``
    With no extra flags the command uses the ``cpu`` template. Specifying ``--n_tasks``
    switches to ``cpu-array``, exports ``TASK_INDEX`` / ``TASK_STEPSIZE`` for each
    worker, and fans the command out according to the backend template.

``aiaccel-job [backend] gpu [--n_tasks=N --n_tasks_per_proc=M --n_procs=K] LOG_FILENAME -- COMMAND``
    Mirrors the CPU flow with the ``gpu`` / ``gpu-array`` templates. Backend YAML
    typically sets queue arguments or ``CUDA_VISIBLE_DEVICES`` here.

``aiaccel-job [backend] mpi --n_procs=N [--n_nodes=K] LOG_FILENAME -- COMMAND``
    ``--n_procs`` is required. The backend template decides whether the payload is
    launched with ``mpirun``, ``srun``, or another MPI frontend. ``--n_nodes`` is
    optional and defaults to the value defined in the config.

``aiaccel-job [backend] train [--n_gpus=N] LOG_FILENAME -- COMMAND``
    Tailored for distributed training templates. ``--n_gpus`` controls how many ranks or
    GPUs the backend-specific training template requests.

When ``aiaccel-job local`` is used, the script still accepts ``--walltime``,
``--n_nodes``, and ``--n_tasks_per_proc`` for interface compatibility, but the local
runner warns that those options are ignored. In particular, local array execution splits
work as ``ceil(n_tasks / n_procs)`` tasks per worker.

Splitting array jobs
====================

Passing ``--n_tasks`` to the ``cpu`` / ``gpu`` modes automatically switches to the
corresponding ``*-array`` configuration. Each backend fans the command out by assigning
each worker a start index (``TASK_INDEX``) and a chunk size (``TASK_STEPSIZE``). For
example, :mod:`aiaccel.job.apps.local` renders a loop such as

.. code-block:: bash

    for LOCAL_PROC_INDEX in {1..8}; do
        TASK_INDEX=$(( 1 + 13 * (LOCAL_PROC_INDEX - 1) ))

        if [[ $TASK_INDEX -gt 100 ]]; then
            break
        fi

        TASK_INDEX=$TASK_INDEX \
        TASK_STEPSIZE=13 \
            python tools/preprocess.py arg1 arg2 ...
    done

where ``python tools/preprocess.py arg1 arg2 ...`` corresponds to the command provided
after ``--`` when invoking ``aiaccel-job``.

PBS, SGE, and Slurm follow the same pattern but seed ``TASK_INDEX`` from the scheduler's
array ID (``PBS_ARRAY_INDEX``, ``SGE_TASK_ID``, or ``SLURM_ARRAY_TASK_ID``). Inside the
job, call :func:`~aiaccel.job.utils.split_tasks` to process only the shard assigned to
the current worker:

.. code-block:: python
    :caption: Consuming array slices

    from aiaccel.job.utils import split_tasks

    src_fname_list = split_tasks(src_fname_list)

This keeps array jobs deterministic: the backend decides which shard is running, and
your script only needs to honor ``TASK_INDEX`` / ``TASK_STEPSIZE``.

***********************
 Cluster Configuration
***********************

Configuring ``local``, ``pbs``, ``sge``, and ``slurm`` follows the same structure, with
each backend adding its own submission command, array syntax, and MPI launcher.

Config essentials
=================

Every invocation loads a YAML file composed of the following building blocks:

.. list-table::
    :widths: 20 35 45
    :header-rows: 1

    - - Scope
      - Key fields
      - How they are used
    - - Global defaults
      - ``walltime``, ``script_prologue``, environment exports
      - Run before every job to log metadata, load modules, or set environment
        variables.
    - - Workload modes
      - ``cpu``, ``cpu-array``, ``gpu``, ``gpu-array``, ``mpi``, ``train``
      - Declare the ``job`` template plus optional scheduler arguments for each
        workload.
    - - Template helpers
      - ``{command}``, ``{args.*}``, ``_base_``, ``_inherit_``
      - Compose configs with Hydra-style inheritance and inject parsed CLI values into
        shell snippets.

The default files under `aiaccel/job/apps/config/
<https://github.com/aistairc/aiaccel/tree/main/aiaccel/job/apps/config>`_ can be used
as-is or copied and extended. They rely on :mod:`aiaccel.config`, so ``_base_`` /
``_inherit_`` work the same way described in :doc:`config`.

At minimum you need the shared metadata plus a ``job`` template for each mode you plan
to invoke:

.. code-block:: yaml
    :caption: job_config.yaml

    walltime: null

    script_prologue: |
        echo Hostname: $(hostname)
        export CUDA_VISIBLE_DEVICES=all

    cpu:
        job: "{command}"

    gpu:
        job: "CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % {args.n_procs} )) {command}"

    train:
        job: |
            mpirun -np {args.n_gpus} \
                -x MAIN_ADDR=$(hostname -i) \
                -x MAIN_PORT=3000 \
                {command}

Key fields:

- ``walltime`` propagates to schedulers that require a time limit.
- ``script_prologue`` runs before the job and is typically used for logging, ``module
  load`` commands, or environment variables.
- Each mode declares a ``job`` template. ``{command}`` is replaced with the shell-quoted
  payload and ``{args.*}`` accesses parsed CLI arguments such as ``n_gpus``,
  ``n_tasks``, or ``walltime``.

Local-specific settings
=======================

`aiaccel/job/apps/config/local.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/local.yaml>`_
keeps the settings minimal: ``walltime`` is ignored, ``script_prologue`` runs before
every job, and each mode simply renders the ``job`` template. Array jobs loop over the
requested number of local worker processes with ``TASK_INDEX`` / ``TASK_STEPSIZE``
exported into the environment, and logs are stored next to ``LOG_FILENAME``. Customize
this file to export additional environment variables, wrap commands in container
runtimes, or change the local launch command for MPI / training jobs.

PBS-specific settings
=====================

`aiaccel/job/apps/config/pbs.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/pbs.yaml>`_ can
be used as a template:

.. code-block:: yaml
    :caption: job_config.yaml (PBS excerpt)

    walltime: "1:0:0"

    script_prologue: |
        echo Job ID: $PBS_JOBID
        echo Hostname: $(hostname)
        export CUDA_VISIBLE_DEVICES=all

    qsub: "qsub -P $JOB_GROUP -l walltime={args.walltime} -v USE_SSH=1"

    cpu:
        qsub_args: "-q rt_HF -l select=1"
        job: "{command}"

    gpu-array:
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

            mpirun -np {args.n_procs} -bind-to none -map-by slot \
                -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \
                {command}

``qsub`` defines the base submission command and is formatted with ``{args}``, while
each mode contributes its own ``qsub_args``. Setting ``--n_tasks`` switches to the
``*-array`` configuration that iterates inside PBS array indices so you can launch
higher fan-out than the scheduler allows per array entry. ``use_scandir`` can be set to
``true`` when the shared filesystem requires explicit cache invalidation during status
polling. The bundled ``mpi`` and ``train`` templates also load ``hpcx`` and invoke
``mpirun`` with cluster-specific network settings, so they are usually the first pieces
to adapt to a new PBS environment.

SGE-specific settings
=====================

`aiaccel/job/apps/config/sge.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/sge.yaml>`_
mirrors the PBS template but adapts the syntax to SGE:

.. code-block:: yaml
    :caption: job_config.yaml (SGE excerpt)

    qsub: "qsub -g $JOB_GROUP -l h_rt={args.walltime}"

    cpu:
        qsub_args: "-l cpu_40=1"
        job: "{command}"

    gpu-array:
        n_tasks_per_proc: 128
        n_procs: 1
        qsub_args: "-l gpu_1=1 -t 1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
        job: "{command}"

    train:
        qsub_args: "-l $( (({args.n_gpus}==1)) && printf node_q || printf node_f )=$(( ({args.n_gpus} + 3) / 4 ))"
        job: |
            source /etc/profile.d/modules.sh
            module load openmpi

            n_gpus=$(nvidia-smi -L | wc -l)

            mpirun -np {args.n_gpus} -map-by ppr:$n_gpus:node:PE=48 \
                -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \
                -x MAIN_ADDR=$(hostname -i) \
                -x MAIN_PORT=3000 \
                -x COLUMNS=120 \
                -x PYTHONUNBUFFERED=true \
                {command}

The only CLI difference is choosing ``aiaccel-job sge ...``. Array jobs rely on
``SGE_TASK_ID`` and render a separate status file per chunk, so the same ``--n_tasks`` /
``--n_tasks_per_proc`` knobs apply. MPI and ``train`` sections typically load
site-specific modules, so copy the template and adjust queue names, slots, GPU labels,
or environment modules to match your cluster.

Slurm-specific settings
=======================

`aiaccel/job/apps/config/slurm.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/slurm.yaml>`_
uses ``sbatch`` / ``srun`` instead of ``qsub`` / ``mpirun``:

.. code-block:: yaml
    :caption: job_config.yaml (Slurm excerpt)

    sbatch: "sbatch --export=USE_SSH=1 --export=ALL"

    gpu:
        sbatch_args: "-p gpu1 -N 1"
        job: "{command}"

    gpu-array:
        n_tasks_per_proc: 64
        n_procs: 4
        sbatch_args: "-p gpu1 -N 1 --array=1-{args.n_tasks}:$(( {args.n_tasks_per_proc} * {args.n_procs} ))"
        job: "CUDA_VISIBLE_DEVICES=$(( LOCAL_PROC_INDEX % 8 )) {command}"

    train:
        sbatch_args: "-p gpu1 -N {args.n_gpus}"
        job: |
            export MAIN_ADDR=$(hostname -i)
            export MAIN_PORT=3000
            export COLUMNS=120
            export PYTHONUNBUFFERED=true
            srun -n {args.n_gpus} --cpu-bind=none --distribution=block:block {command}

Here ``sbatch`` provides the submission command, while each mode contributes its own
``sbatch_args``. The bundled ``mpi`` and ``train`` templates use ``srun`` directly, so
they are a good starting point if your Slurm cluster prefers native launches over
OpenMPI wrappers.

*****************
 Advanced Topics
*****************

Writing Custom Backends
=======================

If none of the bundled backends match your infrastructure, you can build a new backend
that mirrors ``aiaccel-job local/pbs/sge/slurm``. The helper
:func:`~aiaccel.job.apps.prepare_argument_parser` wires the shared CLI options and loads
your YAML template, so your script only needs to render the ``job`` snippet, emit a
shell script, and hand it off to the scheduler.

.. code-block:: python
    :caption: custom_backend.py

    import shlex

    from aiaccel.job.apps import prepare_argument_parser


    def main() -> None:
        config, parser, sub_parsers = prepare_argument_parser("custom.yaml")
        args = parser.parse_args()

        mode = args.mode + "-array" if getattr(args, "n_tasks", None) else args.mode
        job = config[mode].job.format(command=shlex.join(args.command), args=args)

        # TODO: render a shell script and submit it to your scheduler or execute it
        # locally.


    if __name__ == "__main__":
        main()

From there you can extend the skeleton just like the built-in backends: add scheduler
commands, implement array loops, or poll status files before returning. Because the CLI
flags and config semantics stay aligned with ``aiaccel-job``, users only need to switch
the backend name to run the same payload across different environments.

*****************
 Further reading
*****************

- `aiaccel/job/apps/config/
  <https://github.com/aistairc/aiaccel/tree/main/aiaccel/job/apps/config>`_ for the full
  reference templates used in this guide
- :doc:`config` for details on composing the YAML files that power ``aiaccel-job``
