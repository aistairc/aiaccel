from __future__ import annotations

import os
from logging import StreamHandler, getLogger

from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter

from aiaccel.common import goal_maximize
from aiaccel.module import AbstractModule
from aiaccel.util.buffer import Buffer

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


class TensorBoard(AbstractModule):
    """A class for TensorBoard.
    Args:
        options (dict[str, str | int | bool]): A dictionary containing
    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        goal (str): A goal of optimization.
        writer (SummaryWriter): A SummaryWriter object.
        buff (Buffer): A Buffer object.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, "tensorboard")

        self.writer = SummaryWriter(str(self.workspace.tensorboard))

        self.buff = Buffer(["finished"])
        self.buff.d["finished"].set_max_len(2)

    def pre_process(self) -> None:
        # To avoid issues with graph rendering,
        # when providing an ID with 4-6 digits to the name parameter of add_hparams,
        # it is recommended to use a 7-digit ID by padding with leading zeros if necessary.

        self.name_length = self.config.job_setting.name_length
        if self.name_length < 7:
            logger.warning("Within Tensor Board, the name_length is treated as 7.")
            self.name_length = 7
        return None

    def inner_loop_main_process(self) -> bool:
        self.buff.d["finished"].Add(self.storage.get_finished())

        if self.buff.d["finished"].Len == 0:
            return True

        if self.buff.d["finished"].Len >= 2 and self.buff.d["finished"].has_difference() is False:
            return True

        if self.buff.d["finished"].Len == 1:
            trial_ids = self.buff.d["finished"].Now
        else:
            trial_ids = list(set(self.buff.d["finished"].Now) - set(self.buff.d["finished"].Pre))

        for trial_id in trial_ids:
            objective_ys, best_values = self.storage.result.get_any_trial_objective_and_best_value(
                trial_id, goals=self.goals
            )

            if objective_ys is None or best_values is None:
                continue

            tag_objectives = {}

            for i in range(len(self.goals)):
                goal = self.goals[i]
                objective_y = objective_ys[i]
                best_value = best_values[i]

                if len(self.goals) == 1:
                    tag_objective = "objective"
                    tag_min_or_max = "maximum" if goal == goal_maximize else "minimum"
                else:
                    tag_objective = f"objective_{i}_"
                    tag_min_or_max = f"maximum_{i}_" if goal == goal_maximize else f"minimum_{i}_"

                if objective_y is None:
                    self.writer.add_scalar(tag=tag_objective, scalar_value=float("nan"), global_step=trial_id)
                else:
                    self.writer.add_scalar(tag=tag_objective, scalar_value=objective_y, global_step=trial_id)
                self.writer.add_scalar(tag=tag_min_or_max, scalar_value=best_value, global_step=trial_id)

                # hyperparameters
                params = self.storage.hp.get_any_trial_params_dict(trial_id)
                _trial_id = self.zero_padding(trial_id)

                if objective_y is not None:
                    tag_objectives[f"hparam/{tag_objective}"] = objective_y

            self.writer.add_hparams(params, tag_objectives, name=_trial_id)

            self.writer.flush()

        return True

    def post_process(self) -> None:
        self.writer.close()
        return None

    def zero_padding(self, trial_id: int) -> str:
        return f"%0{self.name_length}d" % trial_id
