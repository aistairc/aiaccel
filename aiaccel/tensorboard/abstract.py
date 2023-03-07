from __future__ import annotations

from tensorboardX import SummaryWriter

from aiaccel.module import AbstractModule
from aiaccel.util.buffer import Buffer


class AbstractTensorBoard(AbstractModule):
    def __init__(self, options: dict[str, str | int | bool]) -> None:
        self.options = options
        super().__init__(self.options)

        self.writer = SummaryWriter(str(self.ws / 'tensorboard'))

        self.buff = Buffer(['finished'])
        self.buff.d['finished'].set_max_len(2)

    def pre_process(self) -> None:
        return None

    def inner_loop_main_process(self) -> bool:
        self.buff.d['finished'].Add(self.storage.get_finished())
        if self.buff.d['finished'].Len < 2:
            return True

        if self.buff.d['finished'].has_difference():
            trial_ids = list(
                set(self.buff.d['finished'].Now) - set(self.buff.d['finished'].Pre)
            )
            for trial_id in trial_ids:
                objective_y = self.storage.result.get_any_trial_objective(trial_id)
                self.writer.add_scalar('objective', objective_y, trial_id)

            self.writer.close()

        return True

    def post_process(self) -> None:
        return None
