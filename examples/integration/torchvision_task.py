import torch
from torch import nn
from torch.nn import functional as func

from torchmetrics.classification import MulticlassAccuracy

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


class ImageClassificationTask(OptimizerLightningModule):
    def __init__(self, model: nn.Module, optimizer_config: OptimizerConfig, num_classes: int = 10):
        super().__init__(optimizer_config)

        self.model = model
        if hasattr(self.model.fc, "in_features") and isinstance(self.model.fc.in_features, int):
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        logits = self(x)

        loss = func.cross_entropy(logits, y)

        acc = self.train_accuracy(logits, y)
        self.log_dict(
            {
                "training/loss": loss,
                "training/acc": acc,
            },
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch

        logits = self(x)

        loss = func.cross_entropy(logits, y)

        acc = self.val_accuracy(logits, y)
        self.log_dict(
            {
                "validation/loss": loss,
                "validation/acc": acc,
            },
            prog_bar=True,
        )
