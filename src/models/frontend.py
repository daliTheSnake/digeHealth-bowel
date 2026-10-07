"""The log-Mel front end shared by the spectrogram models."""

import librosa
import torch
from torch import nn

from src.config import CONFIG
from src.dataset import SAMPLE_RATE

N_FFT = CONFIG.mel.n_fft
HOP_LENGTH = CONFIG.mel.hop_length
N_MELS = CONFIG.mel.n_mels
# Floor added before the log, so silence does not give -inf.
EPSILON = CONFIG.mel.epsilon


class MelFrontend(nn.Module):
    """Turn a batch of windows into log-Mel spectrograms, on the model's device."""

    def __init__(self) -> None:
        super().__init__()
        self.window: torch.Tensor
        self.filters: torch.Tensor
        self.register_buffer("window", torch.hann_window(N_FFT))
        # fmin 0 and fmax 8 kHz: the energy of `n`, `v` and the background is
        # below 100 Hz, so no low cut as in PANNs (fmin 50).
        filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
        self.register_buffer("filters", torch.from_numpy(filters))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: shape `(batch, samples)`.

        Returns:
            Shape `(batch, 1, N_MELS, frames)`: one channel, like an image.
        """
        stft = torch.stft(
            audio,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=self.window,
            return_complex=True,
        )
        mel = torch.matmul(self.filters, stft.abs() ** 2)

        return torch.log(mel + EPSILON).unsqueeze(1)
