"""A frame-level detector: for each 10 ms, is a bowel sound going on?"""

import torch
from torch import nn

from src.config import CONFIG
from src.dataset import FRAMES
from src.models.frontend import MelFrontend

# Channels of the three blocks.
CHANNELS = tuple(CONFIG.models.detector.channels)
GRU_HIDDEN = CONFIG.models.detector.gru_hidden
# The Mel bands left after each block halved them.
BANDS = CONFIG.mel.n_mels // 2 ** len(CHANNELS)


class DetectorBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and ReLU, pooling frequency only.

    Time is never pooled, so the output keeps one step per Mel frame.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


class BowelDetector(nn.Module):
    """Log-Mel front end, convolutions over frequency, a GRU over time.

    The output is one logit per frame: no sigmoid, `BCEWithLogitsLoss`
    applies it.
    """

    def __init__(self, num_labels: int = 1) -> None:
        """
        Args:
            num_labels: unused, here so the detector builds like the other
                models. It always gives one logit per frame.
        """
        super().__init__()
        self.frontend = MelFrontend()
        self.input_norm = nn.BatchNorm2d(1)

        in_channels = (1, *CHANNELS[:-1])
        self.blocks = nn.Sequential(
            *[
                DetectorBlock(in_channels=i, out_channels=o)
                for i, o in zip(in_channels, CHANNELS)
            ]
        )
        # Each block pools the Mel bands by 2: BANDS are left per channel.
        self.gru = nn.GRU(
            input_size=CHANNELS[-1] * BANDS,
            hidden_size=GRU_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(2 * GRU_HIDDEN, 1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape `(batch, samples)`, e.g. `(batch, 16000)`.

        Returns:
            The logits, shape `(batch, FRAMES)`.
        """
        x = self.blocks(self.input_norm(self.frontend(audio)))
        # (batch, channels, bands, frames) -> (batch, frames, channels * bands)
        x = x.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x, _ = self.gru(x)

        logits: torch.Tensor = self.classifier(x).squeeze(-1)
        assert logits.shape[1] == FRAMES
        return logits
