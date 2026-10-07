"""HuBERT, pretrained on speech, fine-tuned on raw windows: a benchmark model."""

import torch
from torch import nn
from transformers import AutoModelForAudioClassification

from src.config import CONFIG
from src.windows import LABELS

PRETRAINED = CONFIG.models.hubert.pretrained


class HuBERTClassifier(nn.Module):
    """HuBERT taking raw windows and returning one logit per label.

    Same contract as the other models, so the `Trainer` drives any of them.
    """

    def __init__(
        self,
        freeze_encoder: bool = CONFIG.models.hubert.freeze_encoder,
        num_labels: int = len(LABELS),
    ) -> None:
        """
        Args:
            freeze_encoder: keep the pretrained convolutional feature encoder
                fixed, as is usual when fine-tuning, so only the transformer
                and the new classifier learn.
            num_labels: how many logits to return.
        """
        super().__init__()
        self.network = AutoModelForAudioClassification.from_pretrained(
            PRETRAINED, num_labels=num_labels
        )
        if freeze_encoder:
            self.network.freeze_feature_encoder()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape `(batch, samples)`, e.g. `(batch, 16000)` at 16 kHz.

        Returns:
            The logits, shape `(batch, len(LABELS))`.
        """
        logits: torch.Tensor = self.network(input_values=audio).logits
        return logits
