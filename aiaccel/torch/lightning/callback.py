import json

import lightning


class SaveMetricCallback(lightning.Callback):
    """
    Lightning Callback for save metric in fit ends.

    Args:
        metric_name (str): Metric name to save
        output_path (str): File name to save
    """

    def __init__(self, metric_name: str, output_path: str) -> None:
        super().__init__()
        self.metric_name = metric_name
        self.output_path = output_path

    def on_fit_end(self, trainer: lightning.Trainer, pl_module: lightning.LightningModule) -> None:
        metric_value = trainer.callback_metrics.get(self.metric_name)
        if metric_value is not None:
            metric_value_item = metric_value.item()
            with open(self.output_path, "w") as f:
                json.dump(metric_value_item, f)
        else:
            print(f"Warning: '{self.metric_name}' not found in callback_metrics.")
