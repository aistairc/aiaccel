from typing import Any

from torch import Tensor, nn
from torch.nn import functional as func
from torch.utils.data import DataLoader

import torchmetrics

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


class Resnet50Task(OptimizerLightningModule):
    def __init__(self, model: nn.Module, optimizer_config: OptimizerConfig, num_classes: int = 10):
        super().__init__(optimizer_config)
        self.model = model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def training_step(self, batch: DataLoader[Any], batch_idx: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = func.cross_entropy(logits, y)

        acc = self.train_accuracy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: DataLoader[Any], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = func.cross_entropy(logits, y)

        acc = self.val_accuracy(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
