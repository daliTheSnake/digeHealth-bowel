"""The windowed dataset: every recording cut into labelled windows."""

import logging
from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.utils import load_clean_events
from src.windows import WINDOW_SIZE, Window

logger = logging.getLogger(__name__)

# The recordings are 16 kHz and 48 kHz; every window is brought to this rate.
SAMPLE_RATE = 16000


@dataclass
class BowelDataset:
    """Windows grouped by recording, so a split never cuts a recording."""

    windows_by_recording: dict[str, list[Window]]

    @property
    def windows(self) -> list[Window]:
        """Every window, recording after recording."""
        return [
            window
            for windows in self.windows_by_recording.values()
            for window in windows
        ]

    @property
    def labels(self) -> np.ndarray:
        """The multi-hot labels of `windows`, shape `(n_windows, n_labels)`."""
        return np.array([window.labels for window in self.windows])

    @classmethod
    def from_dir(cls, data_dir: Path) -> "BowelDataset":
        """Build the dataset from a directory of WAV and TXT files.

        Args:
            data_dir: the directory, `{stem}.wav` paired with `{stem}.txt`.

        Returns:
            The windows of every recording that has an annotation file. The
            others are skipped with a warning. Events are cleaned by
            `load_clean_events`; no further outlier is removed.
        """
        windows_by_recording = {}

        for wav_path in sorted(data_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                logger.warning("%s: no matching annotation file", wav_path.name)
                continue
            duration = sf.info(wav_path).duration
            events = load_clean_events(path=txt_path, duration=duration)
            windows_by_recording[wav_path.name] = Window.make(
                events=events, duration=duration
            )

        return cls(windows_by_recording=windows_by_recording)

    def split(
        self, test_recordings: list[str]
    ) -> tuple["BowelDataset", "BowelDataset"]:
        """Split by recording, so no recording is in both sets.

        Args:
            test_recordings: the names of the recordings kept for the test.

        Returns:
            The train and the test datasets. Raises `ValueError` if a name
            is not a recording of this dataset.
        """
        unknown = set(test_recordings) - set(self.windows_by_recording)
        if unknown:
            raise ValueError(f"Unknown recordings: {sorted(unknown)}")

        train = {
            name: windows
            for name, windows in self.windows_by_recording.items()
            if name not in test_recordings
        }
        test = {name: self.windows_by_recording[name] for name in test_recordings}

        return (
            BowelDataset(windows_by_recording=train),
            BowelDataset(windows_by_recording=test),
        )


class WindowDataset(Dataset):
    """The windows of a `BowelDataset` as `(audio, labels)` pairs, read on demand.

    Meant for a `DataLoader`: shuffle the train set, never the test set.
    """

    def __init__(
        self, dataset: BowelDataset, data_dir: Path, sample_rate: int = SAMPLE_RATE
    ) -> None:
        """
        Args:
            dataset: the windows to serve, e.g. one side of `BowelDataset.split`.
            data_dir: the directory holding the WAV files.
            sample_rate: the rate every window is resampled to, in Hz.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.items = [
            (name, window)
            for name, windows in dataset.windows_by_recording.items()
            for window in windows
        ]
        self.native_rates = {
            name: sf.info(data_dir / name).samplerate
            for name in dataset.windows_by_recording
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one window.

        Args:
            index: the position in the dataset.

        Returns:
            The mono audio, `sample_rate * WINDOW_SIZE` float32 samples, and
            the multi-hot labels as float32, as `BCEWithLogitsLoss` expects.
        """
        name, window = self.items[index]
        native_rate = self.native_rates[name]

        audio, _ = sf.read(
            self.data_dir / name,
            start=round(window.start * native_rate),
            frames=round(WINDOW_SIZE * native_rate),
            dtype="float32",
            always_2d=True,
        )
        audio = audio.mean(axis=1)

        if native_rate != self.sample_rate:
            divisor = gcd(native_rate, self.sample_rate)
            audio = resample_poly(
                audio, self.sample_rate // divisor, native_rate // divisor
            ).astype(np.float32)

        return torch.from_numpy(audio), torch.tensor(window.labels, dtype=torch.float32)
