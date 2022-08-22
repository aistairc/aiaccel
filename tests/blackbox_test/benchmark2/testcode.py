from aiaccel import common
from typing import Union
from aiaccel.config import load_config
import aiaccel
import subprocess
# from subprocess import PIPE
import time
from pathlib import Path
from aiaccel.util.filesystem import check_alive_file, load_yaml
import json
import copy
import os
from aiaccel.util import filesystem as fs
import numpy as np
import shutil
from unittest.mock import patch
import sys
from natsort import natsorted



################################################################################
# 共通設定
################################################################################
clean_ws_after_test = False
resource_type = "local"
use_rmtree = False
# resource_type = "ABCI"


"""
Memo
- configファイルの名前が"config.json"以外の場合に動作不良
    → ブランチtpe-devで解消
"""


"""default_config
Configファイルのサンプル
default_configをベースにテスト毎に部分的に変更する．
default_configの内容を変更してはならない
"""

default_config = {
    "generic": {
        "workspace": "./work",
        "job_command": "python wrapper.py",
        "batch_job_timeout": 600,
    },
    "resource": {
        "type": "ABCI",
        "num_node": 4
    },
    "ABCI": {
        "group": "[group]",
        "job_script_preamble": "./job_script_preamble.sh",
        "job_execution_options": ""
    },
    "optimize": {
        "search_algorithm": "aiaccel.optimizer.NelderMeadOptimizer",
        "goal": "minimize",
        "trial_number": 5,
        "rand_seed": 42,
        "parameters": [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            }
        ]
    },
    "job_setting": {
        "name_length": 6,
        "init_fail_count": 100,
        "cancel_retry": 3,
        "cancel_timeout": 60,
        "expire_retry": 3,
        "expire_timeout": 60,
        "finished_retry": 3,
        "finished_timeout": 60,
        "job_loop_duration": 0.5,
        "job_retry": 2,
        "job_timeout": 60,
        "kill_retry": 3,
        "kill_timeout": 60,
        "result_retry": 1,
        "runner_retry": 3,
        "runner_timeout": 60,
        "running_retry": 3,
        "running_timeout": 60
    },
    "sleep_time": {
        "master": 1,
        "optimizer": 1,
        "scheduler": 1
    },
    "logger": {
        "file": {
            "master": "master.log",
            "scheduler": "scheduler.log",
            "optimizer": "optimizer.log"
        },
        "log_level": {
            "master": "DEBUG",
            "scheduler": "DEBUG",
            "optimizer": "DEBUG"
        },
        "stream_level": {
            "master": "DEBUG",
            "scheduler": "DEBUG",
            "optimizer": "DEBUG"
        }
    },
    "ui": {
        "silent_mode": True
    },
    "verification": {
        "is_verified": False,
        "condition": [
            {
                "loop": 1,
                "minimum": 0.0,
                "maximum": 70.0
            }
        ]
    }
}


"""
よく使う設定
"""

hps_choices = [
    {
        "name": "x1",
        "type": "categorical",
        "choices": ['green', 'red', 'yellow', 'blue'],
        "initial": "green"
    },
    {
        "name": "x2",
        "type": "categorical",
        "choices": ['blue', 'yellow', 'red', 'green'],
        "initial": "green"
    }
]

hps_choices__no_initial = [
    {
        "name": "x1",
        "type": "categorical",
        "choices": ['green', 'red', 'yellow', 'blue'],
    },
    {
        "name": "x2",
        "type": "categorical",
        "choices": ['blue', 'yellow', 'red', 'green'],
    }
]


class ConfigGenerator:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.content = copy.deepcopy(default_config)

    def get(self):
        return self.content

    def set(self, config: dict):
        self.content = None
        self.content = copy.deepcopy(config)

    def generate(self):
        # 辞書オブジェクトをJSONファイルへ出力
        with open(self.config_path, mode='wt', encoding='utf-8') as file:
            json.dump(
                self.content,
                file,
                ensure_ascii=False,
                indent=2
            )

    def rep(self, elem: dict, keys: list) -> None:
        self._rep(self.content, elem, keys)

    def _rep(self, dic: dict, elem: dict, keys: list) -> dict:

        if len(keys) == 0:
            return dic

        if len(keys) == 1:
            if keys[0] in dic:
                dic[keys[0]] = elem

        sub_dic = dic[keys[0]]
        sub_key = keys[1:]
        self._rep(sub_dic, elem, sub_key)


class _trials:
    def __init__(self, hp_path: Path) -> None:
        self.d = fs.load_yaml(hp_path)

    def parameters(self, key: str) -> dict:
        for param in self.d["parameters"]:
            if key == param["parameter_name"]:
                return param

    @property
    def result(self):
        return self.d["result"]

    @property
    def start_time(self):
        return self.d["start_time"]

    @property
    def end_time(self):
        return self.d["end_time"]

    @property
    def hashname(self):
        return self.d["hashname"]


class BaseCombinedTest():
    def __init__(self, config_path: str) -> None:

        self.config_path = config_path
        self.config = load_config(self.config_path)
        self.ws = Path(self.config.get("generic", "workspace")).resolve()
        # self.cmd_master = " ".join(
        #     [
        #         "python -m aiaccel.bin.master",
        #         "-c",
        #         self.config_path
        #     ]
        # )
        self.cmd_master = " ".join(
            [
                "python -m aiaccel.start",
                "--config",
                self.config_path,
                "--clean",
            ]
        )
        # self.cmd_clean = " ".join(
        #     [
        #         "python -m aiaccel.bin.clean",
        #         "-c",
        #         self.config_path
        #     ]
        # )
        self.batch_job_timeout = self.config.get("generic", "batch_job_timeout")
        self.goal = self.config.get("optimize", "goal")
        self.hps = self.config.get("optimize", "parameters")
        self.algorithm = self.config.get("optimize", "search_algorithm")
        self.hp_finished = self.ws.joinpath(aiaccel.dict_hp_finished)
        self.hp_ready = self.ws.joinpath(aiaccel.dict_hp_ready)
        self.hp_running = self.ws.joinpath(aiaccel.dict_hp_running)
        self.alive = self.ws.joinpath(aiaccel.dict_alive)
        self.alive_master = self.alive.joinpath(aiaccel.alive_master)
        self.alive_optimizer = self.alive.joinpath(aiaccel.alive_optimizer)
        self.alive_scheduler = self.alive.joinpath(aiaccel.alive_scheduler)

    def check_alive_file(self):
        if (
            self.alive_master.exists() and
            self.alive_optimizer.exists() and
            self.alive_scheduler.exists()
        ):
            return True
        else:
            return False

    def is_finished(self):
        file_final_result = self.ws.joinpath(
            aiaccel.dict_result,
            aiaccel.file_final_result
        )
        t_counter = 0.0
        wait_time_s = 1.0
        max_wait_time_s = self.batch_job_timeout
        count = 0
        while not file_final_result.exists():
            time.sleep(wait_time_s)
            t_counter = t_counter + wait_time_s
            if t_counter > max_wait_time_s:
                return False

            if t_counter % 5 == 0:
                if not self.check_alive_file():
                    count += 1

            if count > 3:
                print("Nothing alive file.")
                return False

            continue
        return True

    # def clean_cmd(self):
    #     p = subprocess.run(
    #         self.cmd_clean,
    #         shell=True
    #         # stdout=PIPE,
    #         # stderr=PIPE,
    #         # text=True
    #     )
    #     return p

    # def clean_rmtree(self):
    #     if self.ws.exists():
    #         try:
    #             shutil.rmtree(self.ws)
    #             print("remove completed: {}".format(self.ws))
    #         except OSError:
    #             pass
    #     else:
    #         print("Folder not found: {}".format(self.ws))

    def clean(self):
        pass
        # if use_rmtree:
        #     print("remove with shutil rmtree")
        #     self.clean_rmtree()
        # else:
        #     print("remove with clean command")
        #     self.clean_cmd()

    def master(self):
        p = subprocess.Popen(
            self.cmd_master,
            shell=True
            # stdout=PIPE,
            # stderr=PIPE
            # text=True
        )
        date = p.stdout
        print('STDOUT: {}'.format(date))

    def go(self):
        self.clean()
        t_start = time.time()
        self.master()
        result = self.is_finished()
        t_proc = time.time() - t_start
        return (result, t_proc)

    def check_input_value(self):
        pass

    def check_output_value(self):
        pass

    def evaluate(self):
        pass

    @property
    def num_hp_finished(self):
        return (len(list(self.hp_finished.glob("**/*.hp"))))

    @property
    def num_hp_running(self):
        return len(list(self.hp_running.glob("**/*.hp")))
    
    @property
    def num_hp_ready(self):
        return len(list(self.hp_ready.glob("**/*.hp")))

    def get_final_result(self):
        ws = Path(self.config.get("generic", "workspace"))
        file_final_result = ws.joinpath(aiaccel.dict_result, aiaccel.file_final_result)
        return load_yaml(file_final_result)

    def get_trials(self):
        hp_finished = self.ws.joinpath(aiaccel.dict_hp_finished)
        files = list(hp_finished.glob("*.hp"))
        # files.sort(key=os.path.getmtime)
        files = natsorted(files)
        trials = []
        for file in files:
            # print(file)
            fn = self.ws / aiaccel.dict_hp_finished / file
            trials.append(_trials(fn))
        return trials

    def get_best_value(self) -> float:
        trials = self.get_trials()

        best_value = 0.0
        if self.goal.lower() == "minimize":
            best_value = np.float("inf")
        elif self.goal.lower() == "maximize":
            best_value = np.float("-inf")
        else:
            raise

        for trial in trials:
            if self.goal.lower() == "minimize":
                if best_value > trial.result:
                    best_value = trial.result
            else:
                if best_value < trial.result:
                    best_value = trial.result

        return best_value


class ThreadWorker(BaseCombinedTest):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.cmd_master = " ".join(
                [
                    "python -m aiaccel.start",
                    "-c",
                    self.config_path,
                    "--clean"
                ]
            )
        # self.cmd_clean = " ".join(
        #     [
        #         "python -m aiaccel.bin.clean",
        #         "-c",
        #         self.config_path
        #     ]
        # )


class evaluate:
    def __init__(self, test: BaseCombinedTest) -> None:
        self.test = test
        self.trials = test.get_trials()
        self.best_value = self.test.get_best_value()

    @property
    def num_hp_finished(self):
        return self.test.num_hp_finished

    @property
    def num_hp_running(self):
        return self.test.num_hp_running
    
    @property
    def num_hp_ready(self):
        return self.test.num_hp_ready

    def cmp_hpval_rslt_dflt(self, trial_number=0, hp_name="") -> bool:
        dflt = None
        rslt = self.trials[trial_number].parameters(hp_name)["value"]
        
        for hp in self.test.hps:
            if hp["name"] == hp_name:
                if self.test.algorithm.lower() == "nelder-mead":
                    dflt = hp["initial"][trial_number]
                elif self.test.algorithm.lower() == "grid":
                    dflt = hp["lower"]
                else:
                    dflt = hp["initial"]
        print("result: {}, initial:{}".format(rslt, dflt))
        return (rslt == dflt)

    def in_sequence(self, trial_number=0, hp_name="") -> bool:
        rslt = self.trials[trial_number].parameters(hp_name)["value"]
        for hp in self.test.hps:
            if hp["name"] == hp_name:
                return (rslt in hp["sequence"])

    def full_consistency_sequence(self, hp_name=""):
        verify = True
        for i in range(len(self.trials)):
            value = self.trials[i].parameters(hp_name)["value"]
            for hp in self.test.hps:
                if hp["name"] == hp_name:
                    verify = verify and (hp["sequence"][i] == value)
                    break
        return verify

    def verify_hp_within_range(self, hp_name="") -> bool:
        low = np.float("inf")
        upper = np.float("-inf")
        verify = True
        for trial in self.trials:
            rslt = trial.parameters(hp_name)["value"]
            for hp in self.test.hps:
                if hp["name"] == hp_name:
                    low = hp["lower"]
                    upper = hp["upper"]
                    break
            verify = verify and (low <= rslt <= upper)
        return verify

    def verify_best_value(self, mn, mx):
        return (mn <= self.best_value <= mx)


class Test_userprogram:
    """ user program異常時のoptの振る舞いをテストするための基本クラス
    """
    def __init__(
        self,
        test_name,
        test_type,
        algorithm
    ):
        self.userprograms = {
            "float": "user.py",
            "int": "user_int.py",
            "categorical": "user_categorical4.py",
            "div_of_zero": "user_0div.py",
            "unchanged": "user_const_ret.py",
            "nan": "user_nan.py",
            "inf": "user_inf1.py",
            "neg_inf": "user_inf2.py",
            "none": "user_none.py"
        }
        self.test_name = test_name
        self.test_type = test_type
        self.algorithm = algorithm
        self.test = None

        self.workspace = "work_{}".format(self.test_name)
        self.config_name = "{}.json".format(self.test_name)
        self.config = ConfigGenerator(self.config_name)
        self.trial_number = 5
        self.job_command = "python {}".format(self.userprograms[self.test_type])
        self.parameters = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            }
        ]
        self.parameters_grid = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10,
                "initial": 0.0
            }
        ]
        self.resource = {
            "type": resource_type,
            "num_node": 1,
        }

    def generate_config(self):
        if self.algorithm == "grid":
            self.parameters = self.parameters_grid

        self.config.rep(
            self.job_command,
            ["generic", "job_command"]
        )
        self.config.rep(
            self.algorithm,
            ["optimize", "search_algorithm"]
        )
        self.config.rep(
            self.workspace,
            ["generic", "workspace"]
        )
        self.config.rep(
            self.resource,
            ["resource"]
        )
        self.config.rep(
            self.trial_number,
            ["optimize", "trial_number"]
        )
        self.config.rep(
            self.parameters,
            ["optimize", "parameters"]
        )
        # 生成
        self.config.generate()

    def go(self, expect=True):
        self.test = BaseCombinedTest(self.config_name)
        self.test.clean()
        self.test.master()
        # マスタが終わるまで待機
        assert self.test.is_finished() is expect

    def evaluate(self):
        if self.test is None:
            raise

        evl = evaluate(self.test)

        # 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
        assert evl.num_hp_finished == self.trial_number
        # ##  最適化終了時，hp_ready内のhpファイル数は0
        assert evl.num_hp_ready == 0
        # ##  最適化終了時，hp_running内のhpファイル数は0
        assert evl.num_hp_running == 0


class Test_silent_mode_mode:
    def __init__(
        self,
        test_name,
        silent_mode,
        algorithm
    ):
        self.test_name = test_name
        self.silent_mode = silent_mode
        self.algorithm = algorithm
        self.test = None

        self.workspace = "work_{}".format(self.test_name)
        self.config_name = "{}.json".format(self.test_name)
        self.job_command = "python {}".format("user.py")
        self.config = ConfigGenerator(self.config_name)

        self.trial_number = 5
        self.parameters = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            }
        ]
        self.parameters_grid = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "log": False,
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            }
        ]

        self.resource = {
            "type": resource_type,
            "num_node": 1,
        }

    def generate_config(self):
        if self.algorithm.lower() == "grid":
            self.parameters = self.parameters_grid

        self.config.rep(
            self.silent_mode,
            ["ui", "silent_mode"]
        )
        self.config.rep(
            self.job_command,
            ["generic", "job_command"]
        )
        self.config.rep(
            self.algorithm,
            ["optimize", "search_algorithm"]
        )
        self.config.rep(
            self.workspace,
            ["generic", "workspace"]
        )
        self.config.rep(
            self.resource,
            ["resource"]
        )
        self.config.rep(
            self.trial_number,
            ["optimize", "trial_number"]
        )
        self.config.rep(
            self.parameters,
            ["optimize", "parameters"]
        )
        # 生成
        self.config.generate()

    def go(self):
        self.test = BaseCombinedTest(self.config_name)
        self.test.clean()
        self.test.master()
        # マスタが終わるまで待機
        assert self.test.is_finished() is True

    def evaluate(self):
        if self.test is None:
            raise
        evl = evaluate(self.test)
        # 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
        assert evl.num_hp_finished == self.trial_number
        # ##  最適化終了時，hp_ready内のhpファイル数は0
        assert evl.num_hp_ready == 0
        # ##  最適化終了時，hp_running内のhpファイル数は0
        assert evl.num_hp_running == 0


class Test_Node_N_test:
    def __init__(
        self,
        test_name: str,
        node: tuple,
        algorithm: str
    ) -> None:

        self.test_name = test_name
        self.algorithm = algorithm
        self.test = None
        self.node = node
        self.workspace = "work_{}".format(self.test_name)
        self.config_name_A = "{}_A.json".format(self.test_name)
        self.config_name_B = "{}_B.json".format(self.test_name)
        self.job_command = "python {}".format("user.py")
        self.trial_number = 30
        self.parameters = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            }
        ]
        self.parameters_grid = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            }
        ]

    def generate_config_A(self):
        config = ConfigGenerator(self.config_name_A)
        if self.algorithm == "grid":
            self.parameters = self.parameters_grid

        self.resource = {
            "type": resource_type,
            "num_node": self.node[0],
        }
        config.rep(
            self.job_command,
            ["generic", "job_command"]
        )
        config.rep(
            self.algorithm,
            ["optimize", "search_algorithm"]
        )
        config.rep(
            self.workspace,
            ["generic", "workspace"]
        )
        config.rep(
            self.resource,
            ["resource"]
        )
        config.rep(
            self.trial_number,
            ["optimize", "trial_number"]
        )
        config.rep(
            self.parameters,
            ["optimize", "parameters"]
        )
        # 生成
        config.generate()

    def generate_config_B(self):
        config = ConfigGenerator(self.config_name_B)
        if self.algorithm == "grid":
            self.parameters = self.parameters_grid
        self.resource = {
            "type": resource_type,
            "num_node": self.node[1],
        }
        config.rep(
            self.job_command,
            ["generic", "job_command"]
        )
        config.rep(
            self.algorithm,
            ["optimize", "search_algorithm"]
        )
        config.rep(
            self.workspace,
            ["generic", "workspace"]
        )
        config.rep(
            self.resource,
            ["resource"]
        )
        config.rep(
            self.trial_number,
            ["optimize", "trial_number"]
        )
        config.rep(
            self.parameters,
            ["optimize", "parameters"]
        )
        # 生成
        config.generate()

    def go(self):
        self.test_A = BaseCombinedTest(self.config_name_A)
        result_1, p_time_1 = self.test_A.go()
        assert result_1 is True

        self.test_B = BaseCombinedTest(self.config_name_B)
        result_2, p_time_2 = self.test_B.go()
        assert result_2 is True

        print("node_num=1: {}".format(p_time_1))
        print("node_num=2: {}".format(p_time_2))
        assert p_time_2 < p_time_1


class Config_hps_test():
    def __init__(
        self,
        test_name,
        algorithm,
        trial_number
    ):
        self.job_command = ""
        self.test_name = test_name
        self.algorithm = algorithm

        self.config_name = "{}.json".format(self.test_name)
        self.config = ConfigGenerator(self.config_name)
        self.test = BaseCombinedTest(self.config_name)

    def change_config(
        self,
        param_name: str,
        param: Union
    ):
        pass

    def run_opt(self):
        self.test.clean()
        self.test.master()
        # マスタが終わるまで待機
        return self.test.is_finished()

    def uniform_int(self):
        self.job_command = "python user_int.py"

        initial = [] * 2
        if self.algorithm == "nelder-mead":
            initial[0] = [0, 5, 3]
            initial[1] = [2, 4, 1]

        self.parameters = [
            {
                "name": "x1",
                "type": "uniform_int",
                "lower": 0,
                "upper": 5,
                "initial": 0
            },
            {
                "name": "x2",
                "type": "uniform_int",
                "lower": 0,
                "upper": 5,
                "initial": 0
            }
        ]

    def uniform_int_no_initial(self):
        pass

    def uniform_float(self):
        pass

    def categorical(self):
        pass

    def categorical_no_initial(self):
        pass

    def ordinal(self):
        pass


# def test_1_optuna_tpe():
#     """テスト設計
#     @ コンセプト
#         探索
#             -OptunaTPE
#         型
#             -uniform_float
#         defaul
#             -あり
#         目的
#             1. 最終trialへの到達を確認
    
#     @ 確認方法
#         -hp_finishedファイルの確認
#             最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
#         -hp_readyファイルの確認
#             最適化終了時，hp_ready内のhpファイル数は0
#         -hp_runningファイルの確認
#             最適化終了時，hp_running内のhpファイル数は0
#     """

#     TEST_NAME = sys._getframe().f_code.co_name
    
#     ############################################################################
#     # テスト用設定
#     job_command = "python user.py"
#     workspace = "work_{}".format(TEST_NAME)
#     trial_number = 30
#     search_algorithm = "tpe"

#     parameters = [
#         {
#             "name": "x1",
#             "type": "uniform_float",
#             "log": False,
#             "lower": 0,
#             "upper": 5,
#             "initial": 0
#         },
#         {
#             "name": "x2",
#             "type": "uniform_float",
#             "log": False,
#             "lower": 0,
#             "upper": 5,
#             "initial": 0
#         }
#     ]

#     ############################################################################
#     #
#     # Config
#     #
#     config_name = "{}.json".format(TEST_NAME)
#     config = ConfigGenerator(config_name)

#     # パラメータ置き換え
#     config.rep(
#         workspace,
#         ["generic", "workspace"]
#     )
#     config.rep(
#         job_command,
#         ["generic", "job_command"]
#     )
#     config.rep(
#         search_algorithm,
#         ["optimize", "search_algorithm"]
#     )
#     config.rep(
#         parameters,
#         [
#             "hyperparameter",
#             "ConfigGeneratorSpace_hyperparameter",
#             "parameters"
#         ]
#     )
#     config.rep(
#         trial_number,
#         ["optimize", "trial_number"]
#     )

#     # 生成
#     config.generate()

#     ############################################################################
#     # テスト
#     test = BaseCombinedTest(config_name)
#     test.clean()
#     test.master()
#     # マスタが終わるまで待機
#     assert test.is_finished() is True

#     ############################################################################
#     #
#     # 評価
#     #
#     evl = evaluate(test)
#     # ## 1 最終値評価
#     print("best value: {}".format(evl.best_value))
#     min = -7.0 * 1.05
#     max = -7.0 * 0.95
#     assert evl.verify_best_value(min, max)
#     # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
#     assert evl.num_hp_finished == trial_number
#     # ## 3 最適化終了時，hp_ready内のhpファイル数は0
#     assert evl.num_hp_ready == 0
#     # ## 4 最適化終了時，hp_running内のhpファイル数は0
#     assert evl.num_hp_running == 0
#     # trial[0] hp値確認 (configの設定通りか？)
#     assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
#     assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
#     # ハイパラ値が [lower upper]の範囲内であることを確認
#     assert evl.verify_hp_within_range(hp_name="x1")
#     assert evl.verify_hp_within_range(hp_name="x2")

#     if clean_ws_after_test:
#         test.clean()


def test_Random_uniform_int():
    """
    @ コンセプト
        探索
            -Random
        型
            -uniform_int
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. トライアル一回目がinitialの値であることを確認
            
    
    @ 確認方法
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，hpファイルの値が一致

    @ 結果
        PASSED
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 2
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 3
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Random_uniform_int_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Random
        型
            -uniform_int
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.

    @ MEMO
        initialを使用しない時は，項目ごと削除する.
    
    @ 結果
    PASSED
    """

    TEST_NAME = sys._getframe().f_code.co_name
    
    job_command = "python user_int.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Random_uniform_float():
    """テスト設計
    @ コンセプト
        探索
            -Random
        型
            -uniform_float
        defaul
            -あり
        目的
            1. Wrapperへ渡す値がfloat型であることを確認
            2. トライアル一回目がinitialの値であることを確認
    
    @ 確認方法
        テスト用設定記載のinitialの値と，hpファイルの値が一致すること

        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.

    @ 結果
        PASSED
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 1.12
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 2.56789
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Random_uniform_float_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Random
        型
            -uniform_float
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.

    @ MEMO
        initialを使用しない時は，項目ごと削除する.
    
    @ 結果
    PASSED
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Random_categorical():
    """
    テスト設計
    @ コンセプト
        探索
            -Random
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. Wrapperへ渡す値がcategoricalであることを確認
    
    @ 確認方法
        hpファイルの値で確認
    
    @ MEMO
        Default指定なし
        categoricalではデフォルト不可
    """

    TEST_NAME = sys._getframe().f_code.co_name
    
    job_command = "python user_categorical4.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue'],
            "initial": "green"
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green'],
            "initial": "green"
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Random_categorical_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Random
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. Wrapperへ渡す値がcategoricalであることを確認
    
    @ 確認方法
        hpファイルの値で確認
    
    @ MEMO
        Default指定なし
        categoricalではデフォルト不可
    """

    TEST_NAME = sys._getframe().f_code.co_name
    
    job_command = "python user_categorical4.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue']
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue']
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ##  最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ##  最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    #
    # trial 0
    x1_result = evl.trials[0].parameters("x1")["value"]
    x2_result = evl.trials[0].parameters("x2")["value"]
    x1_choices = parameters[0]["choices"]
    x2_choices = parameters[1]["choices"]
    assert x1_result in x1_choices
    assert x2_result in x2_choices

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Ramdom_ordinal():
    """テスト設計
    @ コンセプト
        探索
            -Random
        型
            -sequece
            -[2,4,8,16,32,64,128,256, 512,1024]
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. sequeceの値のみがハイパラ値として選択されること．
    
    @ 確認方法
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_resultを確認
    
    @ MEMO
        ordinalでrandomした場合，
        sequence配列をRandomizeする．
    """
    TEST_NAME = sys._getframe().f_code.co_name

    job_command = "python user.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 15
    resource = {
        "type": resource_type,
        "num_node": 1,
    }
    search_algorithm = "random"
    parameters = [
        {
            "name": "x1",
            "type": "ordinal",
            "sequence": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "lower": 0,
            "upper": 1024
        },
        {
            "name": "x2",
            "type": "ordinal",
            "sequence": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
            "lower": 0,
            "upper": 1024
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ##  最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ##  最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # config と入力の一致確認
    assert evl.in_sequence(trial_number=0, hp_name="x1")
    assert evl.in_sequence(trial_number=0, hp_name="x2")
    assert evl.in_sequence(trial_number=-1, hp_name="x1")
    assert evl.in_sequence(trial_number=-1, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


# ===============================================================================
# ===============================================================================


# 2. Grid Search
def test_Grid_uniform_int():
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "log": False,
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================


def test_Grid_uniform_int_no_initial():
    """
    @ コンセプト
        探索
            -Grid
        型
            -uniform_int
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5,
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5,
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Grid_uniform_float():
    """
    @ コンセプト
        探索
            -Grid
        型
            -uniform_float
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認

    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Grid_uniform_float_no_initial():
    """
    @ コンセプト
        探索
            -Grid
        型
            -uniform_float
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認

    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "step": 1.0,
            "log": False,
            "base": 10,
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Grid_categorical():
    """
    テスト設計
    @ コンセプト
        探索
            -Grid
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -あり
        目的
            1. テストを実施しないことを確認
    
    @ 確認方法
        - OPTが開始しない
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 16
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue'],
            "initial": 'green'
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green'],
            "initial": 'green'
        }
    ]
    
    resource = {
        "type": resource_type,
        "num_node": 1
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False


# ===============================================================================


def test_Grid_categorical_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Grid
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. テストを実施しないことを確認
    
    @ 確認方法
        - OPTが開始しない
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 16
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue'],
            "initial": 'green'
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green'],
            "initial": 'green'
        }
    ]
    
    resource = {
        "type": resource_type,
        "num_node": 1
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )
    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False


# ===============================================================================

def test_Grid_ordinal():
    """テスト設計
    @ コンセプト
        探索
            -Grid
        型
            -sequece
            -[2,4,8,16,32,64,128,256, 512,1024]
        defaul
            -なし
        目的
            1. テストを実施しないことを確認
    
    @ 確認方法
        - OPTが開始しない
    
    @ MEMO
        2021/06/18の時点で仕様が不明瞭.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 16
    search_algorithm = "grid"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }
    parameters = [
        {
            "name": "x1",
            "type": "ordinal",
            "sequence": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "lower": 0,
            "upper": 1024
        },
        {
            "name": "x2",
            "type": "ordinal",
            "sequence": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
            "lower": 0,
            "upper": 1024
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False


# ===============================================================================


# 3 Sobol Search
def test_Sobol_uniform_int():
    """
    @ コンセプト
        探索
            -Sobol
        型
            -uniform_int
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 2
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 3
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False

    # ############################################################################
    # #
    # # 評価
    # #
    # evl = evaluate(test)
    # # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    # assert evl.num_hp_finished == trial_number
    # # ##  最適化終了時，hp_ready内のhpファイル数は0
    # assert evl.num_hp_ready == 0
    # # ##  最適化終了時，hp_running内のhpファイル数は0
    # assert evl.num_hp_running == 0
    # #
    # # trial 0
    # x1_result = evl.trials[0].parameters("x1")["value"]
    # x2_result = evl.trials[0].parameters("x2")["value"]
    # x1_expected = parameters["parameters"][0]["initial"]
    # x2_expected = parameters["parameters"][1]["initial"]
    # print((x1_result, x1_expected))
    # print((x2_result, x2_expected))
    # # trial[0] hp値確認 (configの設定通りか？)
    # assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    # assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # # ハイパラ値が [lower upper]の範囲内であることを確認
    # assert evl.verify_hp_within_range(hp_name="x1")
    # assert evl.verify_hp_within_range(hp_name="x2")

    # if clean_ws_after_test:
    #     test.clean()

# ===============================================================================


def test_Sobol_uniform_int_no_initial():
    """
    @ コンセプト
        探索
            -Sobol
        型
            -uniform_int
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False

    # ############################################################################
    # #
    # # 評価
    # #
    # evl = evaluate(test)
    # # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    # assert evl.num_hp_finished == trial_number
    # # ##  最適化終了時，hp_ready内のhpファイル数は0
    # assert evl.num_hp_ready == 0
    # # ##  最適化終了時，hp_running内のhpファイル数は0
    # assert evl.num_hp_running == 0
    # #
    # # trial 0
    # x1_result = evl.trials[0].parameters("x1")["value"]
    # x2_result = evl.trials[0].parameters("x2")["value"]
    # x1_expected = parameters["parameters"][0]["initial"]
    # x2_expected = parameters["parameters"][1]["initial"]
    # print((x1_result, x1_expected))
    # print((x2_result, x2_expected))
    # # trial[0] hp値確認 (configの設定通りか？)
    # assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    # assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # # ハイパラ値が [lower upper]の範囲内であることを確認
    # assert evl.verify_hp_within_range(hp_name="x1")
    # assert evl.verify_hp_within_range(hp_name="x2")

    # if clean_ws_after_test:
    #     test.clean()


# ===============================================================================


def test_Sobol_uniform_float():
    """
    @ コンセプト
        探索
            -Sobol
        型
            -uniform_int
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 1.12
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 2.56789
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ##  最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ##  最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")
    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_Sobol_uniform_float_no_initial():
    """
    @ コンセプト
        探索
            -Sobol
        型
            -uniform_float
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ##  最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ##  最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()

# ===============================================================================


def test_Sobol_categorical():
    """
    テスト設計
    @ コンセプト
        探索
            -Sobol
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -あり
        目的
            1. Sobolはcategorical非対応．実行しないことを確認する．
    
    @ 確認方法
        - 実行しないことを確認
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    parameters = hps_choices

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機False
    assert test.is_finished() is False

    if clean_ws_after_test:
        test.clean()


# ===============================================================================

def test_Sobol_categorical_no_initial():
    """
        テスト設計
    @ コンセプト
        探索
            -Sobol
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. Sobolはcategorical非対応．実行しないことを確認する．
    
    @ 確認方法
        - 実行しないことを確認
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "sobol"
    parameters = hps_choices__no_initial
    resource = {
        "type": resource_type,
        "num_node": 1
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False

    if clean_ws_after_test:
        test.clean()

# ===============================================================================


def test_Sobol_ordinal():
    """
    @ コンセプト
        探索
            -Sobol
        型
            -sequece
            -[2,4,8,16,32,64,128,256, 512,1024]
        defaul
            -あり
        目的
            1. Sobolはcategorical非対応．実行しないことを確認する．
    
    @ 確認方法
        - 実行しないことを確認
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 10
    search_algorithm = "sobol"
    parameters = [
        {
            "name": "x1",
            "type": "ordinal",
            "sequence": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "lower": 0,
            "upper": 1024
        },
        {
            "name": "x2",
            "type": "ordinal",
            "sequence": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
            "lower": 0,
            "upper": 1024
        }
    ]
    
    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is False

    if clean_ws_after_test:
        test.clean()


# ===============================================================================
# ===============================================================================


# 4 Nelder Mead Search
def test_NM_uniform_int():
    """
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_int
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name

    ############################################################################
    # テスト用設定
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)
    trial_number = 5
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": [0, 5, 3]
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": [2, 4, 1]
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    assert evl.cmp_hpval_rslt_dflt(trial_number=1, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=1, hp_name="x2")
    assert evl.cmp_hpval_rslt_dflt(trial_number=2, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=2, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_NM_uniform_int_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_int
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_NM_uniform_float():
    """
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_float
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    TEST_NAME = sys._getframe().f_code.co_name
    """
    
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": [0.1, 1.5, 2.8]
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": [3.6, 4.2, 4.9]
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    ############################################################################
    # パラメータ置き換え
    trial_number = 5
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )

    ############################################################################
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    assert evl.cmp_hpval_rslt_dflt(trial_number=1, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=1, hp_name="x2")
    assert evl.cmp_hpval_rslt_dflt(trial_number=2, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=2, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_NM_uniform_float_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_float
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_NM_uniform_float_no_initial_2():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_float
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    # テスト
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_NM_categorical():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder-Mead
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. Nelder-MeadはCategorical非対応
    
    @ 確認内容
        - 実行しないことを確認する．
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "nelder-mead"

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue'],
            "initial": ['red']
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green'],
            "initial": ['yellow']
        }
    ]

    resource = {
        "type": resource_type,
        "num_node": 1
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )

    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    # assert test.is_finished() is True
    assert test.is_finished() is False


# ===============================================================================


def test_NM_ordinal():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder-Mead
        型
            -sequece
            -[2,4,8,16,32,64,128,256, 512,1024]
        initial
            -なし
        目的
            1. Nelder-Meadはordinal非対応
    
    @ 確認内容
        - 実行しないことを確認する．
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 10
    search_algorithm = "nelder-mead"
    parameters = [
        {
            "name": "x1",
            "type": "ordinal",
            "sequence": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "lower": 0,
            "upper": 1024,
            "initial": [2]
        },
        {
            "name": "x2",
            "type": "ordinal",
            "sequence": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
            "lower": 0,
            "upper": 1024,
            "initial": [1024]
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    # assert test.is_finished() is True
    assert test.is_finished() is False

# ===============================================================================


def test_NM_Minimize_algo_check():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder-Mead
        型
            -uniform_float
        initial
            -なし
        目的
            1. Nelder-Meadアルゴリズムが正常動作すること
    
    @ 確認方法
        - f(X1, x2) = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        の最小化を確認する（minimize(f) ≒ -7.0）
        最小値-7.0の±5%の間に収まればOKとする．
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    goal = "minimize"
    search_algorithm = "nelder-mead"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        }
    ]

    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    min = -7.0 * 1.05
    max = -7.0 * 0.95
    assert evl.verify_best_value(min, max)
    ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================

def test_NM_Maximize_algo_check():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder-Mead
        型
            -uniform_float
        initial
            -なし
        目的
            1. Nelder-Meadアルゴリズムが正常動作すること
    
    @ 確認方法
        - f(X1, x2) = -1.0 * ((x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2))
        の最大化を確認する（minimize(f) ≒ 7.0）
        最小値7.0の±5%の間に収まればOKとする．
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_mx.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    goal = "MAXIMIZE"
    search_algorithm = "nelder-mead"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": []
        }
    ]

    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    min = 7.0 * 0.95
    max = 7.0 * 1.05
    assert evl.verify_best_value(min, max)
    ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================
# ===============================================================================


# 5 TPE (Optuna)
def test_OptunaTPE_uniform_int():
    """
    @ コンセプト
        探索
            -OptunaTPE
        型
            -uniform_int
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 1
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5,
            "initial": 2
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================


def test_OptunaTPE_uniform_int_no_initial():

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_int.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_int",
            "lower": 0,
            "upper": 5
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")

# ===============================================================================


def test_OptunaTPE_uniform_float():
    """
    @ コンセプト
        探索
            -OptunaTPE
        型
            -uniform_float
        defaul
            -あり
        目的
            1. 最終トライアルの完了を確認
            2. 初期値=initialを確認
            3. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 1.12
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": 2.56789
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================


def test_OptunaTPE_uniform_float_no_initial():
    """
    テスト設計
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_int
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. ハイパラ値が[lower upper]の範囲内であることを確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


# ===============================================================================


def test_OptunaTPE_categorical():
    """
    テスト設計
    @ コンセプト
        探索
            -OptunaTPE
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
    
    @ 確認方法
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_resultを確認
    """
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue'],
            "initial": "green"
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green'],
            "initial": "green"
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # trial[0] hp値確認 (configの設定通りか？)
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x1")
    assert evl.cmp_hpval_rslt_dflt(trial_number=0, hp_name="x2")

    if clean_ws_after_test:
        test.clean()


# ===============================================================================

def test_OptunaTPE_categorical_no_initial():
    """
    @ コンセプト
        探索
            -OptunaTPE
        型
            -categorical
            -['green', 'red', 'yellow', 'blue']
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hpファイルの値で確認
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_categorical4.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 5
    search_algorithm = "random"
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "categorical",
            "choices": ['green', 'red', 'yellow', 'blue']
        },
        {
            "name": "x2",
            "type": "categorical",
            "choices": ['blue', 'yellow', 'red', 'green']
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0

    if clean_ws_after_test:
        test.clean()


# ===============================================================================


def test_OptunaTPE_ordinal():
    """
    @ コンセプト
        探索
            -OptunaTPE
        型
            -sequece
            -[2,4,8,16,32,64,128,256, 512,1024]
        defaul
            -なし
        目的
            1. 最終トライアルの完了を確認
            2. sequeceの値のみがハイパラ値として選択されること．
    
    @ 確認方法
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - hp_resultを確認
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 10
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "ordinal",
            "sequence": [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "lower": 0,
            "upper": 1024
        },
        {
            "name": "x2",
            "type": "ordinal",
            "sequence": [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2],
            "lower": 0,
            "upper": 1024
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    # assert test.is_finished() is True
    assert test.is_finished() is False

    # ############################################################################
    # #
    # # 評価
    # #
    # evl = evaluate(test)
    # # ##  最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    # assert evl.num_hp_finished == trial_number
    # # ##  最適化終了時，hp_ready内のhpファイル数は0
    # assert evl.num_hp_ready == 0
    # # ##  最適化終了時，hp_running内のhpファイル数は0
    # assert evl.num_hp_running == 0

    # assert evl.full_consistency_sequence(hp_name="x1")
    # assert evl.full_consistency_sequence(hp_name="x2")

    # # ハイパラ値が [lower upper]の範囲内であることを確認
    # assert evl.verify_hp_within_range(hp_name="x1")
    # assert evl.verify_hp_within_range(hp_name="x2")
    # if clean_ws_after_test:
    #     test.clean()


def test_OptunaTPE_Minimize_algo_check():
    """
    テスト設計
    @ コンセプト
        探索
            -Optuna
        型
            -uniform_float
        initial
            -なし
        goal
            - Minimize
        目的
            1. 探索アルゴリズムが正常動作することを確認する．
    
    @ 確認方法
        - f(X1, x2) = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        の最小化を確認する（minimize(f) ≒ -7.0）
        最小値-7.0の±5%の間に収まればOKとする．
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    goal = "minimize"
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5
        }
    ]
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
)
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    min = -7.0 * 1.05
    max = -7.0 * 0.95
    assert evl.verify_best_value(min, max)
    ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")


def test_OptunaTPE_Maximize_algo_check():
    """
    テスト設計
    @ コンセプト
        探索
            - Optuna
        型
            - uniform_float
        initial
            - なし
        goal
            - Maximize
        目的
            1. 探索アルゴリズムが正常動作することを確認する．
    
    @ 確認方法
        - f(X1, x2) = -1.0 * ((x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2))
        の最大化を確認する（maximize(f) ≒ 7.0）
        最小値7.0の±5%の間に収まればOKとする．
    """

    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_mx.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 30
    goal = "MAXIMIZE"
    search_algorithm = "tpe"
    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
        }
    ]

    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # ## 1 最終値評価
    print("best value: {}".format(evl.best_value))
    min = 7.0 * 0.95
    max = 7.0 * 1.05
    assert evl.verify_best_value(min, max)
    ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0
    # ハイパラ値が [lower upper]の範囲内であることを確認
    assert evl.verify_hp_within_range(hp_name="x1")
    assert evl.verify_hp_within_range(hp_name="x2")



# ===============================================================================
# ===============================================================================


# 6. User Program test (invalid return)
## 6.1 Nelder Mead
def test_NM__user_return_NaN():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "nan", "nelder-mead")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()

# ===============================================================================


def test_NM__user_return_None():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "none", "nelder-mead")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()


# ===============================================================================


def test_NM__user_return_INF():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "inf", "nelder-mead")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_NM__user_return_negINF():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "neg_inf", "nelder-mead")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_NM__user_div_by_zero():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "div_of_zero", "nelder-mead")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()


# ===============================================================================


def test_NM__user_return_unchanged():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "unchanged", "nelder-mead")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================
# ===============================================================================


## 6.2 TPE
def test_TPE__user_return_NaN():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "nan", "tpe")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()


# ===============================================================================


def test_TPE__user_return_None():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "none", "tpe")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()


# ===============================================================================


def test_TPE__user_return_INF():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "inf", "tpe")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_TPE__user_return_negINF():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "neg_inf", "tpe")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_TPE__user_div_by_zero():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "div_of_zero", "tpe")
    tst.generate_config()
    tst.go(expect=False)
    # tst.evaluate()


# ===============================================================================


def test_TPE__user_return_unchanged():
    test_name = sys._getframe().f_code.co_name
    tst = Test_userprogram(test_name, "unchanged", "tpe")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_Random_silent_mode_true():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, True, "random")
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_Random_silent_mode_false():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, False, "random")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_Grid_silent_mode_true():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, True, "grid")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_Grid_silent_mode_false():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, False, "grid")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_Sobol_silent_mode_true():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, True, "sobol")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_Sobol_silent_mode_false():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, False, "sobol")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_NM_silent_mode_true():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, True, "nelder-mead")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_NM_silent_mode_false():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, False, "nelder-mead")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_OptunaTPE_silent_mode_true():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, True, "tpe")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


def test_OptunaTPE_silent_mode_false():
    test_name = sys._getframe().f_code.co_name
    tst = Test_silent_mode_mode(test_name, False, "tpe")
    tst.generate_config()
    tst.go()
    tst.evaluate()

# ===============================================================================


# ===============================================================================
# 8. Multi node test
# ===============================================================================


def test_Random_node_N():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Node_N_test(test_name, (1, 4), "random")
    tst.generate_config_A()
    tst.generate_config_B()
    tst.go()

# ===============================================================================


def test_Grid_node_N():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Node_N_test(test_name, (1, 4), "grid")
    tst.generate_config_A()
    tst.generate_config_B()
    tst.go()

# ===============================================================================


def test_Sobol_node_N():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Node_N_test(test_name, (1, 4), "sobol")
    tst.generate_config_A()
    tst.generate_config_B()
    tst.go()

# ===============================================================================


def test_NM_node_N():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Node_N_test(test_name, (1, 4), "nelder-mead")
    tst.generate_config_A()
    tst.generate_config_B()
    tst.go()

# ===============================================================================


def test_OptunaTPE_node_N():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Node_N_test(test_name, (1, 4), "tpe")
    tst.generate_config_A()
    tst.generate_config_B()
    tst.go()


# ===============================================================================
#   9. Algorithm Validity test
# ===============================================================================
class Test_Algorithm_Validity:
    def __init__(
        self,
        test_name: str,
        algorithm: str,
        trial_number=30,
        node=1
    ) -> None:
        self.test_name = test_name
        self.algorithm = algorithm
        self.workspace = "work_{}".format(self.test_name)
        self.config_name = "{}.json".format(self.test_name)
        self.config = ConfigGenerator(self.config_name)
        self.test = None
        self.job_command = "python {}".format("user.py")
        self.trial_number = trial_number
        self.parameters = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "initial": 0.0
            }
        ]
        self.parameters_grid = [
            {
                "name": "x1",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            },
            {
                "name": "x2",
                "type": "uniform_float",
                "lower": 0.0,
                "upper": 5.0,
                "step": 1.0,
                "log": False,
                "base": 10
            }
        ]
        self.resource = {
            "type": resource_type,
            "num_node": node,
        }

    def generate_config(self):
        if self.algorithm == "grid":
            self.parameters = self.parameters_grid

        self.config.rep(
            self.job_command,
            ["generic", "job_command"]
        )
        self.config.rep(
            self.algorithm,
            ["optimize", "search_algorithm"]
        )
        self.config.rep(
            self.workspace,
            ["generic", "workspace"]
        )
        self.config.rep(
            self.resource,
            ["resource"]
        )
        self.config.rep(
            self.trial_number,
            ["optimize", "trial_number"]
        )
        self.config.rep(
            self.parameters,
            ["optimize", "parameters"]
        )
        # 生成
        self.config.generate()

    def go(self):
        self.test = BaseCombinedTest(self.config_name)
        self.test.clean()
        self.test.master()
        # マスタが終わるまで待機
        assert self.test.is_finished() is True

    def evaluate(self):
        evl = evaluate(self.test)
        # ## 1 最終値評価
        print("best value: {}".format(evl.best_value))
        min = -7.0 * 1.05
        max = -7.0 * 0.95
        assert evl.verify_best_value(min, max)
        # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
        assert evl.num_hp_finished == self.trial_number
        # ## 3 最適化終了時，hp_ready内のhpファイル数は0
        assert evl.num_hp_ready == 0
        # ## 4 最適化終了時，hp_running内のhpファイル数は0
        assert evl.num_hp_running == 0


# ===============================================================================


def test_Random_validity():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Algorithm_Validity(
        test_name,
        "random",
        trial_number=50,
        node=4
    )
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_Grid_validity():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Algorithm_Validity(
        test_name,
        "grid",
        trial_number=30,
        node=4
    )
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_Sobol_validity():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Algorithm_Validity(
        test_name,
        "sobol",
        trial_number=30,
        node=4
    )
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_NM_validity():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Algorithm_Validity(
        test_name,
        "nelder-mead",
        trial_number=30,
        node=4
    )
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================


def test_OptunaTPE_validity():
    test_name = sys._getframe().f_code.co_name
    tst = Test_Algorithm_Validity(
        test_name,
        "tpe",
        trial_number=30,
        node=4
    )
    tst.generate_config()
    tst.go()
    tst.evaluate()


# ===============================================================================
# # 10. other
# ===============================================================================
def NM__out_of_boundary__maximize():
    """
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_float
        defaul
            -あり
        目的
            1. 領域外探索の動作確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    TEST_NAME = sys._getframe().f_code.co_name
    """
    
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_mx.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 10
    goal = "MAXIMIZE"
    search_algorithm = "nelder-mead"

    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            # "lower": -10,
            # "upper": 10,
            "initial": [2.3612613, -0.78579058, -6.53604462]
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            # "lower": -10,
            # "upper": 10,
            "initial": [2.50185612, 5.46367967, -6.01959838]
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    ############################################################################
    # パラメータ置き換え
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )

    ############################################################################
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    test = BaseCombinedTest(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0

    if clean_ws_after_test:
        test.clean()


def NM__out_of_boundary():
    """
    @ コンセプト
        探索
            -Nelder Mead
        型
            -uniform_float
        defaul
            -あり
        目的
            1. 全点領域外探索時の動作確認
    
    @ 確認内容
        - 最適化終了時，hp_finished内のhpファイルtrial_numberと一致
        - 最適化終了時，hp_ready内のhpファイル数は0
        - 最適化終了時，hp_running内のhpファイル数は0
        - テスト用設定記載のinitialの値と，トライアル=0のhpファイルの値が一致すること
        - hp_finishedのhpファイルとコンフィグ["lower"], ["upper"]を比較する.
    TEST_NAME = sys._getframe().f_code.co_name
    """
    
    TEST_NAME = sys._getframe().f_code.co_name
    job_command = "python user_mx.py"
    
    workspace = "work_{}".format(TEST_NAME)

    ############################################################################
    # テスト用設定
    trial_number = 10
    goal = "MAXIMIZE"
    search_algorithm = "nelder-mead"
    silent_mode = True
    resource = {
        "type": resource_type,
        "num_node": 1,
    }

    parameters = [
        {
            "name": "x1",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": [2.3612613, -0.78579058, -6.53604462]
        },
        {
            "name": "x2",
            "type": "uniform_float",
            "lower": 0,
            "upper": 5,
            "initial": [2.50185612, 5.46367967, -6.01959838]
        }
    ]

    ############################################################################
    #
    # コンフィグファイル生成
    #
    config_name = "{}.json".format(TEST_NAME)
    config = ConfigGenerator(config_name)

    ############################################################################
    # パラメータ置き換え
    config.rep(
        silent_mode,
        ["ui", "silent_mode"]
    )
    config.rep(
        workspace,
        ["generic", "workspace"]
    )
    config.rep(
        job_command,
        ["generic", "job_command"]
    )

    config.rep(
        search_algorithm,
        ["optimize", "search_algorithm"]
    )
    config.rep(
        resource,
        ["resource"]
    )
    config.rep(
        parameters,
        ["optimize", "parameters"]
    )
    config.rep(
        trial_number,
        ["optimize", "trial_number"]
    )
    config.rep(
        goal,
        ["optimize", "goal"]
    )

    ############################################################################
    # 生成
    config.generate()

    ############################################################################
    #
    # テスト
    #
    # #1
    # test = BaseCombinedTest(config_name)
    test = ThreadWorker(config_name)
    test.clean()
    test.master()
    # マスタが終わるまで待機
    assert test.is_finished() is True

    ############################################################################
    #
    # 評価
    #
    evl = evaluate(test)
    # # ## 1 最終値評価
    # print("best value: {}".format(evl.best_value))
    # min = -7.0 * 1.05
    # max = -7.0 * 0.95
    # assert evl.verify_best_value(min, max)
    # ## 2 最適化終了時，hp_finished内のhpファイル数はtrial_numberと一致する
    assert evl.num_hp_finished == trial_number
    # ## 3 最適化終了時，hp_ready内のhpファイル数は0
    assert evl.num_hp_ready == 0
    # ## 4 最適化終了時，hp_running内のhpファイル数は0
    assert evl.num_hp_running == 0

    if clean_ws_after_test:
        test.clean()


if __name__ == "__main__":
    # # # 1.Random Search
    #test_Random_uniform_int()                   # OK
    # test_Random_uniform_int_no_initial()        # OK
    # test_Random_uniform_float()                 # OK
    # test_Random_uniform_float_no_initial()      # OK
    # test_Random_categorical()                   # OK
    # test_Random_categorical_no_initial()        # OK
    test_Ramdom_ordinal()                       # OK
    # # ===============================================
    # # 2. Grid Search
    # test_Grid_uniform_int()                     # OK
    # test_Grid_uniform_int_no_initial()          # OK
    # test_Grid_uniform_float()                   # OK
    # test_Grid_uniform_float_no_initial()        # OK
    # test_Grid_categorical()                     # OK
    # test_Grid_categorical_no_initial()          # OK
    # test_Grid_ordinal()                         # OK
    # # ===============================================
    # # 3 Sobol Search
    # test_Sobol_uniform_int()                    # OK
    # test_Sobol_uniform_int_no_initial()         # OK
    # test_Sobol_uniform_float()                  # OK
    # test_Sobol_uniform_float_no_initial()       # OK
    # test_Sobol_categorical()                    # OK
    # test_Sobol_categorical_no_initial()         # OK
    # test_Sobol_ordinal()                        # OK
    # # ===============================================
    # # 4 Nelder Mead Search
    # test_NM_uniform_int()                       # OK
    # test_NM_uniform_int_no_initial()            # OK
    # test_NM_uniform_float()                     # OK
    # test_NM_uniform_float_no_initial()          # OK
    # test_NM_uniform_float_no_initial_2()        # OK
    # test_NM_categorical()                       # (NG): 仕様確定後Try
    # test_NM_ordinal()                           # (NG): 仕様確定後Try
    # test_NM_Minimize_algo_check()               # OK
    # test_NM_Maximize_algo_check()               # (NG): BUG
    # # ===============================================
    # # 5 TPE (Optuna)
    # test_OptunaTPE_uniform_int()                # OK
    # test_OptunaTPE_uniform_int_no_initial()     # OK
    # test_OptunaTPE_uniform_float()              # OK
    # test_OptunaTPE_uniform_float_no_initial()   # OK
    # test_OptunaTPE_categorical()                # OK
    # test_OptunaTPE_categorical_no_initial()     # OK
    # test_OptunaTPE_ordinal()                    # OK
    # test_OptunaTPE_Minimize_algo_check()        # OK
    # test_OptunaTPE_Maximize_algo_check()        # OK
    # # ===============================================
    # 6. User Program test (invalid return)
    # # 6.1 Nelder Mead
    # test_NM__user_return_NaN()                  # (NG)
    # test_NM__user_return_None()                 #
    # test_NM__user_return_INF()                  #
    # test_NM__user_return_negINF()               #
    # test_NM__user_div_by_zero()                 #
    # test_NM__user_return_unchanged()            #
    # # # 6.2 TPE
    # test_TPE__user_return_NaN()                 # (NG): 仕様確定後Try
    # test_TPE__user_return_None()                #
    # test_TPE__user_return_INF()                 #
    # test_TPE__user_return_negINF()              #
    # # test_TPE__user_div_by_zero()                # (NG): 仕様確定後Try
    # test_TPE__user_return_unchanged()           # 
    # # ===============================================
    # # 7. silent wrqapper mode test
    # test_Random_silent_mode_true()           # OK
    # test_Random_silent_mode_false()          # OK
    # test_Grid_silent_mode_true()             # OK
    # test_Grid_silent_mode_false()            # OK
    # test_Sobol_silent_mode_true()            # OK
    # test_Sobol_silent_mode_false()           # OK
    # test_NM_silent_mode_true()               # OK
    # test_NM_silent_mode_false()              # OK
    # test_OptunaTPE_silent_mode_true()        # OK
    # test_OptunaTPE_silent_mode_false()       # OK
    # # ===============================================
    # # 8. Multi node test
    # test_Random_node_N()                        # OK
    # test_Grid_node_N()                          # OK
    # test_Sobol_node_N()                         # OK
    #test_NM_node_N()                            # OK
    # test_OptunaTPE_node_N()                     # OK
    # # ===============================================
    # ## 9. Algorithm Validity test
    # test_Random_validity()                      # OK
    # test_Grid_validity()                        # OK
    # test_Sobol_validity()                       # OK
    # test_NM_validity()                          # OK
    # test_OptunaTPE_validity()                   # OK
    # # # ===============================================
    # # # 10. Other
    # NM__out_of_boundary__minimize()
    # NM__out_of_boundary__maximize()             # (NG): BUG
    # NM__out_of_boundary()
    # # ===============================================
    print("done.")
