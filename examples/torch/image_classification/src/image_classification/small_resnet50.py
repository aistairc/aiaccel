from torch import nn

from torchvision import models


class SmallResNet50(nn.Sequential):
    def __init__(self, num_classes: int):
        super().__init__()

        self.base = models.resnet50(num_classes=num_classes)
        self.base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Linear(2048, 10)
