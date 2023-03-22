from __future__ import annotations

from tensorboardX import SummaryWriter

from aiaccel.common import goal_maximize, goal_minimize
from aiaccel.module import AbstractModule
from aiaccel.util.buffer import Buffer
from aiaccel.util.trialid import TrialId


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
    def __init__(self, options: dict[str, str | int | bool]) -> None:
        self.options = options
        super().__init__(self.options)

        self.goal = self.config.goal.get().lower()

        self.writer = SummaryWriter(str(self.ws / 'tensorboard'))

        self.buff = Buffer(['finished'])
        self.buff.d['finished'].set_max_len(2)

    def pre_process(self) -> None:
        return None

    def inner_loop_main_process(self) -> bool:
        self.buff.d['finished'].Add(self.storage.get_finished())

        if self.buff.d['finished'].Len == 0:
            return True

        if (
            self.buff.d['finished'].Len >= 2 and
            self.buff.d['finished'].has_difference() is False
        ):
            return True

        if self.buff.d['finished'].Len == 1:
            trial_ids = self.buff.d['finished'].Now
        else:
            trial_ids = list(set(self.buff.d['finished'].Now) - set(self.buff.d['finished'].Pre))

        for trial_id in trial_ids:

            # objective, best_value
            objective_y, best_value = self.storage.result.get_any_trial_objective_and_best_value(
                trial_id, self.goal
            )

            if objective_y is None or best_value is None:
                return True

            tag = 'minimum'
            if self.goal == goal_maximize:
                tag = 'maximum'

            self.writer.add_scalar(tag='objective', scalar_value=objective_y, global_step=trial_id)
            self.writer.add_scalar(tag=tag, scalar_value=best_value, global_step=trial_id)

            # hyperparameters
            params = self.storage.hp.get_any_trial_params_dict(trial_id)
            _trial_id = TrialId(self.config).zero_padding_any_trial_id(trial_id)

            self.writer.add_hparams(
                params, {'objective': objective_y}, name=_trial_id
            )

            self.writer.flush()

        self.writer.close()

        return True

    def post_process(self) -> None:
        return None
