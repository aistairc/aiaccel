from __future__ import annotations

from tensorboardX import SummaryWriter

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

        self.goal = self.config.goal.get()

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
            trial_ids.sort()

        for trial_id in trial_ids:

            # objective
            objective_y = self.storage.result.get_any_trial_objective(trial_id)
            self.writer.add_scalar('objective', objective_y, trial_id)

            # best
            bast_values = self.storage.result.get_bests(self.goal)
            if len(bast_values) > 0:
                print(bast_values, trial_id)
                self.writer.add_scalar(self.goal, bast_values[trial_id], trial_id)

            # hyperparameters
            params = self.storage.hp.get_any_trial_params_dict(trial_id)
            _trial_id = TrialId(self.config).zero_padding_any_trial_id(trial_id)

            self.writer.add_hparams(
                params, {'objective': objective_y}, name=_trial_id
            )

        self.writer.close()

        return True

    def post_process(self) -> None:
        return None
