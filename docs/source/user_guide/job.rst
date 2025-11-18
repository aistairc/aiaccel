Managing Job Workloads
======================

``aiaccel-job`` launches training, evaluation, or batch jobs on laptops, shared
servers, and PBS / SGE clusters through a single CLI. This guide explains the command
workflow, the YAML templates behind it, and the scheduler-specific tweaks that keep the
interface consistent across backends.

Core Concepts
-------------

The following ideas make it easier to understand how ``aiaccel-job`` ties the CLI and
configuration files together:

- Inspired by `Kaldi <https://kaldi-asr.org/>`_ / `ESPnet
  <https://espnet.github.io/espnet/>`_ job management, the CLI is template-driven and
  keeps the configuration file at the center of every CPU / GPU batch run.
- Provides six dedicated modes: ``cpu``, ``cpu-array``, ``gpu``, ``gpu-array``, ``mpi``,
  and ``train``. Individual jobs and array jobs share the same interface.
- Targets both ``local`` execution and HPC schedulers such as ``pbs`` / ``sge``, letting you
  reuse the exact same notation while switching backends.

Basic Usage
-----------

``aiaccel-job`` works like any other CLI: select a backend (``local``, ``pbs``, or
``sge``), pick the mode, point to a log file, and provide the command payload.
Configuration files supply the templates, but the invocation style stays identical on
every backend.

Running a job
~~~~~~~~~~~~~

The basic invocation consists of the backend (``local``), the mode, ``LOG_FILENAME``,
and the command to run:

.. code-block:: bash

    aiaccel-job local --config job_config.yaml \
        cpu logs/example.log -- \
        python train.py --epochs 10

``aiaccel-job`` writes the rendered script next to the log file (``logs/example.sh``),
streams stdout/stderr into ``logs/example.log``, and will spawn array-style workers
automatically when ``--n_tasks`` is supplied (``cpu-array`` / ``gpu-array`` is selected
in that case). Omit ``--config`` to fall back to the default ``local.yaml`` shipped with
aiaccel, or set ``AIACCEL_JOB_CONFIG`` for a global default.

Command-line interface
~~~~~~~~~~~~~~~~~~~~~~

Every backend accepts ``--config`` (YAML path), ``--print_config`` (dump and exit), and
``--walltime`` (override scheduler defaults). The positional ``LOG_FILENAME`` selects
where logs/scripts are stored and ``COMMAND`` captures the payload. Modes add the
following options:

``aiaccel-job [backend] cpu [--n_tasks=N --n_tasks_per_proc=M --n_procs=K] LOG_FILENAME -- COMMAND``
    With no extra flags the scheduler grants the entire node described in the ``cpu``
    template. Specifying ``--n_tasks`` switches to ``cpu-array`` so each worker gets a
    single CPU and receives ``TASK_INDEX`` / ``TASK_STEPSIZE``. ``--n_tasks_per_proc``
    controls how many tasks each worker handles before advancing, while ``--n_procs``
    defines how many workers run on the same node.

``aiaccel-job [backend] gpu [--n_tasks=N --n_tasks_per_proc=M --n_procs=K] LOG_FILENAME -- COMMAND``
    Mirrors the CPU behavior: without ``--n_tasks`` the job occupies the full GPU node
    described in the template, and providing ``--n_tasks`` activates ``gpu-array`` so
    each worker uses a single GPU. Fan-out is controlled by the same
    ``--n_tasks_per_proc`` / ``--n_procs`` knobs.

``aiaccel-job [backend] mpi --n_procs=N [--n_nodes=K] LOG_FILENAME -- COMMAND``
    ``--n_procs`` is required and forwarded directly to ``mpirun``. ``--n_nodes`` is
    optional and defaults to the config value if omitted.

``aiaccel-job [backend] train [--n_gpus=N] LOG_FILENAME -- COMMAND``
    Tailored for distributed training templates. ``--n_gpus`` specifies how many GPUs
    the job should consume; the scheduler arguments in the config translate that count
    into queue requests.

When ``aiaccel-job local`` is used, the script accepts ``--walltime``, ``--n_nodes`` and
``--n_tasks_per_proc`` for compatibility but emits warnings because they are ignored by
the local runner. PBS / SGE backends rely on the same interface to parameterize array
jobs and MPI launches.

Splitting array jobs
~~~~~~~~~~~~~~~~~~~~

Passing ``--n_tasks`` to the ``cpu`` / ``gpu`` modes automatically switches to the
``*-array`` configuration. Each backend handles the fan-out by iterating over the array
index and setting ``TASK_INDEX`` / ``TASK_STEPSIZE`` before running your command. For
example, :mod:`aiaccel.job.apps.local` renders a loop such as

.. code-block:: bash

    for LOCAL_PROC_INDEX in {1..${N_PROCS}}; do
        TASK_INDEX=$(( 1 + N_TASKS_PER_PROC * (LOCAL_PROC_INDEX - 1) ))
        TASK_INDEX=$TASK_INDEX \
        TASK_STEPSIZE=$N_TASKS_PER_PROC \
            python tools/preprocess.py arg1 arg2 ...
    done

where ``python tools/preprocess.py arg1 arg2 ...`` corresponds to the command provided
after ``--`` when invoking ``aiaccel-job``.

PBS / SGE versions follow the same pattern but seed ``TASK_INDEX`` with the scheduler's
array ID (``PBS_ARRAY_INDEX`` / ``SGE_TASK_ID``) so each slice writes
``${LOG_FILENAME}.${array}.${LOCAL_PROC_INDEX}.log``. Inside the job, call
:func:`aiaccel.job.slice_tasks` to process only the shard assigned to the current worker
(``aiaccel/torch/pipelines/base_pipeline.py`` already does this):

.. code-block:: python
    :caption: Consuming array slices

    from aiaccel.job import slice_tasks

    src_fname_list = slice_tasks(src_fname_list)

This approach keeps array jobs deterministic—the scheduler decides which chunk is
running, and your script only needs to honor ``TASK_INDEX`` / ``TASK_STEPSIZE`` to work
both locally and on HPC backends.

Cluster Configuration
---------------------

Configuring ``local``, ``pbs``, and ``sge`` targets follows the same structure but each
introduces scheduler-specific extensions for submission commands, array behavior, and
MPI launchers.

Config essentials
~~~~~~~~~~~~~~~~~

Every invocation loads a YAML file composed of the following building blocks:

.. list-table::
    :widths: 20 35 45
    :header-rows: 1

    * - Scope
      - Key fields
      - How they are used
    * - Global defaults
      - ``walltime``, ``script_prologue``, environment exports
      - Run before every job to log metadata, load modules, or set scheduler limits.
    * - Workload modes
      - ``cpu``, ``cpu-array``, ``gpu``, ``gpu-array``, ``mpi``, ``train``
      - Declare the ``job`` template plus optional queue arguments for each workload.
    * - Template helpers
      - ``{command}``, ``{args.*}``, ``_base_``, ``_inherit_``
      - Compose configs with Hydra-style inheritance and inject CLI overrides at render
        time.

The default files under `aiaccel/job/apps/config/
<https://github.com/aistairc/aiaccel/tree/main/aiaccel/job/apps/config>`_ can be used
as-is or copied and extended. They rely on :mod:`aiaccel.config`, so ``_base_`` /
``_inherit_`` work the same way described in :doc:`config`.

At minimum you only need the shared metadata and the job snippet for the mode you plan
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
- Each mode declares a ``job`` template. ``{command}`` is replaced with the CLI payload
  and ``{args.*}`` accesses runtime options (``n_gpus``, ``n_tasks`` etc.).

Local-specific settings
~~~~~~~~~~~~~~~~~~~~~~~

`aiaccel/job/apps/config/local.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/local.yaml>`_
keeps the settings minimal: ``walltime`` is ignored, ``script_prologue`` runs before
every job, and each mode simply renders the ``job`` template. Array jobs loop over the
requested number of processes with ``TASK_INDEX``/``TASK_STEPSIZE`` exported into the
environment, and logs are stored next to ``LOG_FILENAME``. Customize this file to export
additional environment variables, wrap commands in container runtimes, or change how
local array workers are fanned out.

PBS-specific settings
~~~~~~~~~~~~~~~~~~~~~

`aiaccel/job/apps/config/pbs.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/job/apps/config/pbs.yaml>`_ can
be used as a template:

.. code-block:: yaml
    :caption: job_config.yaml (PBS excerpt)

    walltime: "1:0:0"

    script_prologue: |
        echo Job ID: $PBS_JOBID
        echo Hostname: $(hostname)
        module load hpcx

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
            mpirun -np {args.n_procs} -bind-to none -map-by slot \
                -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \
                {command}

``qsub`` defines the base submission command and is formatted with ``{args}``, while
each mode contributes its own ``qsub_args``. Setting ``--n_tasks`` switches to the
``*-array`` configuration that iterates inside PBS array indices so you can launch
higher fan-out than the scheduler allows per array entry. ``use_scandir`` can be set to
``true`` when the shared filesystem requires explicit cache invalidation during status
polling. All other sections (``gpu``, ``train``) follow the same structure and can be
trimmed if unused.

SGE-specific settings
~~~~~~~~~~~~~~~~~~~~~

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
            module load openmpi
            mpirun -np {args.n_gpus} -map-by ppr:$(nvidia-smi -L | wc -l):node:PE=48 \
                -x MAIN_ADDR=$(hostname -i) \
                -x MAIN_PORT=3000 \
                {command}

The only CLI difference is choosing ``aiaccel-job sge ...``. Array jobs rely on
``SGE_TASK_ID`` and render a separate status file per chunk, so the same
``--n_tasks``/``--n_tasks_per_proc`` knobs apply. MPI and ``train`` sections typically
load site-specific modules, so copy the template and adjust queue names, slots, GPU
labels, or environment modules to match your cluster.

Advanced Topics
---------------

Writing a custom dispatcher
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If none of the bundled backends match your infrastructure, you can create a new
dispatcher while keeping the CLI compatible with existing workloads. The helper
:func:`aiaccel.job.apps.prepare_argument_parser` loads the job config, wires the shared
options (``--config``, ``--walltime``, etc.), and returns the parser plus sub-commands
for ``cpu`` / ``gpu`` / ``mpi`` / ``train``. A minimal entry point looks like:

.. code-block:: python
    :caption: custom_backend.py

    from aiaccel.job.apps import prepare_argument_parser

    config, parser, sub_parsers = prepare_argument_parser("custom.yaml")

    # Add backend-specific flags here if needed

    def main() -> None:
        args = parser.parse_args()
        config = config  # optionally resolve template paths
        # Submit or run jobs using args.mode, args.command, etc.

    if __name__ == "__main__":
        main()

By relying on ``prepare_argument_parser`` and mirroring the ``aiaccel-job`` option
names, existing scripts can switch to your dispatcher just by replacing the backend
keyword while keeping their commands and configs intact. This keeps programs portable:
as long as the CLI contract is preserved, researchers can run the same workload on any
cluster without touching the training script.

Further reading
---------------

- :doc:`config` for details on composing the YAML files that power ``aiaccel-job``
- `aiaccel/job/apps/config/
  <https://github.com/aistairc/aiaccel/tree/main/aiaccel/job/apps/config>`_ for the full
  reference templates used in this guide
