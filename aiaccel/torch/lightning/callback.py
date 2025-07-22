import json

import lightning.pytorch as pl


class SaveParamCallback(pl.Callback):
    def __init__(self, param_name: str, output_path: str) -> None:
        super().__init__()
        self.param_name = param_name
        self.output_path = output_path

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        param_value = trainer.callback_metrics.get(self.param_name)
        if param_value is not None:
            param_value_item = param_value.item()
            with open(self.output_path, "w") as f:
                json.dump(param_value_item, f)
        else:
            print(f"Warning: '{self.param_name}' not found in callback_metrics.")
