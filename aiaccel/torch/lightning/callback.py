import json

import lightning


class SaveParamCallback(lightning.Callback):
    """
    Lightning Callback for save paramater in fit ends.

    Args:
        param_name (str): Parameter name to save
        output_path (str): File name to save
    """

    def __init__(self, param_name: str, output_path: str) -> None:
        super().__init__()
        self.param_name = param_name
        self.output_path = output_path

    def on_fit_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        param_value = trainer.callback_metrics.get(self.param_name)
        if param_value is not None:
            param_value_item = param_value.item()
            with open(self.output_path, "w") as f:
                json.dump(param_value_item, f)
        else:
            print(f"Warning: '{self.param_name}' not found in callback_metrics.")
