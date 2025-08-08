import torch
from torch import nn
from torch.nn import functional as fn

from torchmetrics.classification import MulticlassAccuracy

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule


class ImageClassificationTask(OptimizerLightningModule):
    def __init__(self, model: nn.Module, optimizer_config: OptimizerConfig, num_classes: int = 10):
        super().__init__(optimizer_config)

        self.model = model

        self.training_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.validation_accuracy = MulticlassAccuracy(num_classes=num_classes)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        logits = self(x)

        loss = fn.cross_entropy(logits, y)

        self.log_dict(
            {
                "training/loss": loss,
                "training/accuracy": self.training_accuracy(logits, y),
            },
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch

        logits = self(x)

        loss = fn.cross_entropy(logits, y)

        self.log_dict(
            {
                "validation/loss": loss,
                "validation/accuracy": self.validation_accuracy(logits, y),
            },
            prog_bar=True,
        )
