from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess

from hydra.utils import instantiate

import pytest

from aiaccel.config import load_config


@pytest.fixture()
def workspace_factory(tmp_path_factory):
    @contextmanager
    def _factory(data_name: str = "single_objective"):
        tmp_path = tmp_path_factory.mktemp("workspace")

        shutil.copytree(Path(__file__).parent / "data" / data_name, tmp_path, dirs_exist_ok=True)

        org_path = Path.cwd()

        try:
            os.chdir(tmp_path)
            yield tmp_path
        finally:
            os.chdir(org_path)

    return _factory


def test_from_config(workspace_factory) -> None:
    with workspace_factory() as workspace:
        subprocess.run("aiaccel-hpo optimize --config=config.yaml", shell=True, check=True)

        assert (workspace / "optuna.db").exists()

        config = load_config(workspace / "merged_config.yaml")
        study = instantiate(config.study)
        assert len(study.get_trials()) == 15


def test_from_cli(workspace_factory) -> None:
    with workspace_factory() as workspace:
        cmd = (
            "aiaccel-hpo optimize"
            " working_directory=./cli/"
            " n_trials=15"
            " n_max_jobs=1"
            " params.x1='[0,1]'"
            " params.x2='[0,1]'"
            " study.sampler._target_=optuna.samplers.TPESampler"
            " study.sampler.seed=0"
            " --"
            " python ./objective.py --x1={x1} --x2={x2} {out_filename}"
        )
        subprocess.run(cmd, shell=True, check=True)

        config = load_config(workspace / "cli" / "merged_config.yaml")
        study = instantiate(config.study)
        best_value = study.best_trial.value

    # check consistency with the config-style execution
    with workspace_factory() as workspace:
        subprocess.run("aiaccel-hpo optimize --config=config.yaml", shell=True, check=True)

        config = load_config(workspace / "merged_config.yaml")
        study = instantiate(config.study)

    assert best_value == study.best_trial.value


def test_resume(workspace_factory) -> None:
    with workspace_factory() as workspace:
        subprocess.run("aiaccel-hpo optimize --config=config.yaml", shell=True, check=True)
        subprocess.run("aiaccel-hpo optimize --config=config.yaml", shell=True, check=True)

        config = load_config(workspace / "merged_config.yaml")
        study = instantiate(config.study)
        assert len(study.get_trials()) == 30


def test_multi_objective(workspace_factory) -> None:
    with workspace_factory("multi_objective") as workspace:
        subprocess.run("aiaccel-hpo optimize --config=config.yaml", shell=True, check=True)

        config = load_config(workspace / "merged_config.yaml")
        study = instantiate(config.study)

        assert len(study.get_trials()) == 15

        assert all(len(trial.values) == 2 for trial in study.get_trials())
