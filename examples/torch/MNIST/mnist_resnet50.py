import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


# モデル定義
class MNISTResNet50(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # 入力層を1チャネル対応に置き換え
        self.model.conv1 = nn.Conv2d(
            in_channels=3,  # Grayscale → 3チャンネルに変換済み
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 出力層をMNISTの10クラス用に置き換え
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
