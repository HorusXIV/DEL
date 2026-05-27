from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

# Model for Baseline
class BaselineCNN(nn.Module):
    """
    A simple CNN architecture for SVHN classification.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 01
class DeepCNN(nn.Module):
    """
    A moderately deeper CNN for SVHN classification.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 02
class WideCNN(nn.Module):
    """
    A wider CNN for SVHN classification.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 03
class Kernel3CNN(nn.Module):
    """
    A baseline-width/depth CNN that uses 3x3 kernels instead of 5x5.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 04
class PoolLateCNN(nn.Module):
    """
    A CNN that delays pooling to preserve spatial detail longer.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 05
class DropoutCNN(nn.Module):
    """
    A baseline CNN with dropout in the classifier head.
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model for Hypothese 07
class BatchNormCNN(nn.Module):
    """
    A baseline CNN with Batch Normalization after each convolution.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# Funcions for for Hypothese 08
def _apply_xavier_initialization(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# Funcions for for Hypothese 08
def _apply_he_initialization(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

# Model A for Hypothese 08
class XavierInitCNN(BaselineCNN):
    """
    A baseline CNN with Xavier-initialized convolution and linear layers.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        _apply_xavier_initialization(self)

# Model B for Hypothese 08
class HeInitCNN(BaselineCNN):
    """
    A baseline CNN with He-initialized convolution and linear layers.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        _apply_he_initialization(self)

# Model for Hypothese 10
class TransferLearningCNN(nn.Module):
    """
    A transfer-learning model based on a pretrained ResNet18 backbone.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone = backbone

        for name, parameter in self.backbone.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False

        self._freeze_backbone_batchnorm()

    def _freeze_backbone_batchnorm(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in self.backbone.named_children():
            if name == "fc":
                module.train(mode)
            else:
                module.train(False)
        self._freeze_backbone_batchnorm()
        return self

    def forward(self, x):
        return self.backbone(x)
