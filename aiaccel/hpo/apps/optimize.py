from typing import Any

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from importlib import resources
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

from aiaccel.config import load_config, pathlib2str_config, print_config, resolve_inherit


def main() -> None:
    # remove OmegaConf arguments from sys.argv
    oc_args = []
    if "--" in sys.argv:  # If there are additional arguments before '--', treat them as OmegaConf arguments
        sep_idx = sys.argv.index("--")
        sys.argv.pop(sep_idx)

        for ii in range(0, sep_idx)[::-1]:
            if "=" in sys.argv[ii] and not sys.argv[ii].startswith("-"):
                oc_args.append(sys.argv.pop(ii))

        oc_args = list(reversed(oc_args))

    # parse arguments
    parser = argparse.ArgumentParser(
        description="""\
A helper CLI to optimize hyperparameters using Optuna.
See complete usage: https://aistairc.github.io/aiaccel/user_guide/hpo.html .

Typical usages:
  aiaccel-hpo optimize params.x1=[0,1] params.x2=[0,1] -- ./objective.py --x1={x1} --x2={x2} {out_filename}
  aiaccel-hpo optimize --config=config.yaml ./objective.py --x1={x1} --x2={x2} {out_filename}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to the configuration file.")
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # load config
    base_config_path = resources.files(f"{__package__}.config")
    if args.config is None:
        args.config = base_config_path / "default.yaml"
        working_directory = Path.cwd().resolve() / f"aiaccel-hpo_{datetime.now():%Y-%m-%d-%H-%M-%S}"
    else:
        working_directory = args.config.parent.resolve()

    config = oc.merge(
        load_config(
            args.config,
            {
                "config_path": args.config,
                "working_directory": working_directory,
                "base_config_path": base_config_path,
            },
        ),
        oc.from_cli(oc_args),
    )
    if len(args.command) > 0:
        config.command = args.command

    print_config(config)

    # save config
    config.working_directory = Path(config.working_directory)
    config.working_directory.mkdir(parents=True, exist_ok=True)

    with open(config.working_directory / "merged_config.yaml", "w") as f:
        oc.save(pathlib2str_config(config), f)

    config = resolve_inherit(config)
    config.working_directory = Path(config.working_directory)  # maybe bug

    # build study and hparams manager
    study = instantiate(config.study)
    params = instantiate(config.params)

    # main loop
    futures: dict[Any, tuple[Trial, Path]] = {}
    submitted_job_count = 0
    finished_job_count = 0

    with ThreadPoolExecutor(config.n_max_jobs) as pool:
        while finished_job_count < config.n_trials:
            active_jobs = len(futures.keys())
            available_slots = max(0, config.n_max_jobs - active_jobs)

            # Submit job in ThreadPoolExecutor
            for _ in range(min(available_slots, config.n_trials - submitted_job_count)):
                trial = study.ask()

                out_filename = config.working_directory / f"trial_{trial.number:0>6}.json"

                future = pool.submit(
                    subprocess.run,
                    shlex.join(config.command).format(
                        config=config,
                        job_name=f"trial_{trial.number:0>6}",
                        out_filename=out_filename,
                        **params.suggest_hparams(trial),
                    ),
                    shell=True,
                    check=True,
                )

                futures[future] = trial, out_filename
                submitted_job_count += 1

            # Get result from out_filename and tell
            done_features, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done_features:
                trial, out_filename = futures.pop(future)

                with open(out_filename) as f:
                    y = json.load(f)

                out_filename.unlink()

                frozentrial = study.tell(trial, y)
                study._log_completed_trial(y if isinstance(y, list) else [y], frozentrial.number, frozentrial.params)
                finished_job_count += 1


if __name__ == "__main__":
    main()
