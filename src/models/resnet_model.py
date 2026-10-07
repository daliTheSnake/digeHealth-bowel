"""ResNet-50 on log-Mel spectrograms, pretrained on ImageNet: a benchmark model."""

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.config import CONFIG
from src.models.frontend import MelFrontend
from src.windows import LABELS


class ResNet50Classifier(nn.Module):
    """ResNet-50 taking raw windows and returning one logit per label.

    The log-Mel spectrogram is treated as a one-channel image. Same contract
    as the other models, so the `Trainer` drives any of them.
    """

    def __init__(
        self,
        pretrained: bool = CONFIG.models.resnet50.pretrained,
        num_labels: int = len(LABELS),
    ) -> None:
        """
        Args:
            pretrained: start from the ImageNet weights. False trains from
                scratch, to see what the pretraining brings.
            num_labels: how many logits to return.
        """
        super().__init__()
        self.frontend = MelFrontend()
        # Standardize the log-Mel input, like the images ResNet was trained on.
        self.input_norm = nn.BatchNorm2d(1)

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.network = resnet50(weights=weights)
        # A spectrogram has one channel, not three: sum the RGB filters.
        rgb = self.network.conv1
        self.network.conv1 = nn.Conv2d(
            1, rgb.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.network.conv1.weight.data = rgb.weight.data.sum(dim=1, keepdim=True)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_labels)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape `(batch, samples)`, e.g. `(batch, 16000)`.

        Returns:
            The logits, shape `(batch, len(LABELS))`.
        """
        logits: torch.Tensor = self.network(self.input_norm(self.frontend(audio)))
        return logits
