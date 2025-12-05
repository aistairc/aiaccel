import lightning as lt
from lightning.fabric.utilities.rank_zero import rank_zero_warn


class PrintUnusedParam(lt.Callback):
    def __init__(self):
        super().__init__()
        self._flag = False

    def on_after_backward(self, trainer, pl_module):
        if self._flag:
            return

        if trainer.is_global_zero:
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is None:
                    rank_zero_warn(f"[rank0] {name} is unused")
            self._flag = True
