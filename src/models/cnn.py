"""A small VGG-style CNN on log-Mel spectrograms, one logit per label.

The design follows what works for audio tagging, scaled down for ~3400 train
windows:
- Input: 64-band log-Mel spectrogram with a 10 ms hop over ~1 s, the setting
  of Hershey et al. (ICASSP 2017), "CNN Architectures for Large-Scale Audio
  Classification" (96 x 64 patches of 0.96 s).
- Blocks: two 3x3 convolutions, batch norm, ReLU and 2x2 average pooling, as
  in PANNs CNN10/CNN14 (Kong et al., 2020), themselves VGG-like.
- Head: mean over frequency, then max + mean over time, then a linear layer,
  also the PANNs recipe.
It is far smaller than PANNs CNN10 (~5 M parameters) to limit overfitting.
"""

import torch
from torch import nn

from src.config import CONFIG
from src.models.frontend import MelFrontend
from src.windows import LABELS

# Channels of the three blocks. PANNs CNN10 uses 64, 128, 256, 512.
CHANNELS = tuple(CONFIG.models.cnn.channels)


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each with batch norm and ReLU, then 2x2 pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


class BowelCNN(nn.Module):
    """Log-Mel front end, three conv blocks, global pooling, one logit per label.

    The output is a logit: no sigmoid, `BCEWithLogitsLoss` applies it.
    """

    def __init__(self, num_labels: int = len(LABELS)) -> None:
        super().__init__()
        self.frontend = MelFrontend()
        # Standardize the log-Mel input, whose scale depends on the recording.
        self.input_norm = nn.BatchNorm2d(1)

        in_channels = (1, *CHANNELS[:-1])
        self.blocks = nn.Sequential(
            *[
                ConvBlock(in_channels=i, out_channels=o)
                for i, o in zip(in_channels, CHANNELS)
            ]
        )
        self.dropout = nn.Dropout(CONFIG.models.cnn.dropout)
        self.classifier = nn.Linear(CHANNELS[-1], num_labels)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape `(batch, samples)`, e.g. `(batch, 16000)`.

        Returns:
            The logits, shape `(batch, len(LABELS))`.
        """
        x = self.input_norm(self.frontend(audio))
        x = self.blocks(x)
        # Mean over frequency, then max + mean over time: a fixed-size vector
        # whatever the duration, which sums peaks (short `sb`) and averages.
        x = x.mean(dim=2)
        x = x.max(dim=2).values + x.mean(dim=2)

        logits: torch.Tensor = self.classifier(self.dropout(x))
        return logits
