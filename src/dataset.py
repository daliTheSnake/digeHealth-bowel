"""The windowed dataset: every recording cut into labelled windows."""

import logging
from dataclasses import dataclass
from math import floor, gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from src.config import CONFIG
from src.utils import load_clean_events
from src.windows import EPSILON, WINDOW_SIZE, Window

logger = logging.getLogger(__name__)

# The recordings are 16 kHz and 48 kHz; every window is brought to this rate.
SAMPLE_RATE = CONFIG.audio.sample_rate

# Time split of a recording into blocks, whose role repeats with this cycle.
BLOCK_SIZE = CONFIG.split.block_size
# One cycle of blocks: 7 train, 2 validation, 1 test, i.e. 70 / 20 / 10 %.
BLOCK_CYCLE = CONFIG.split.cycle
SPLIT_MARGIN = CONFIG.split.margin

# A reject example has no `sb`, `mb` or `h` this close to its centre, in seconds.
REJECT_CLEARANCE = CONFIG.segments.reject_clearance
SEGMENT_SEED = CONFIG.segments.seed
# Classes of a segment: the three bowel sounds, then what the detector may
# wrongly propose (noise, voice, background).
SEGMENT_LABELS = ["h", "mb", "sb"]
REJECT = "reject"
CLASSES = [*SEGMENT_LABELS, REJECT]


def block_role(block: int, cycle: list[str]) -> str | None:
    """The role of a block of a recording, `None` before its start."""
    return cycle[block % len(cycle)] if block >= 0 else None


def assign_block(
    start: float, end: float, block_size: float, cycle: list[str], margin: float
) -> str | None:
    """Tell which set a span of a recording belongs to.

    Args:
        start: the start of the span, in seconds.
        end: the end of the span, in seconds.
        block_size: the length of a block, in seconds.
        cycle: the role of each block in a cycle: "train", "val" or "test".
        margin: the gap, in seconds, kept between a validation or test block
            and the train spans.

    Returns:
        "val" or "test" if the span lies in such a block, "train" if it is
        far enough from them, `None` if it crosses a block border or is too
        close to a validation or test block.
    """
    # An end exactly on a border belongs to the block before it.
    first = floor(start / block_size)
    last = floor((end - EPSILON) / block_size)
    if first == last and block_role(first, cycle) in ("val", "test"):
        return block_role(first, cycle)

    # The blocks within `margin` of the span: if any is a validation or test
    # block, the span is too close to train on.
    nearest = range(
        floor((start - margin) / block_size),
        floor((end + margin - EPSILON) / block_size) + 1,
    )
    if any(block_role(block, cycle) in ("val", "test") for block in nearest):
        return None

    return "train"


def read_audio(
    path: Path, native_rate: int, start: float, sample_rate: int
) -> np.ndarray:
    """Read `WINDOW_SIZE` seconds of a WAV: mono, resampled, not scaled.

    Args:
        path: the WAV file.
        native_rate: its sampling rate, in Hz.
        start: where to start reading, in seconds. May be negative, or run
            past the end: the missing samples are zeros.
        sample_rate: the rate to bring the audio to, in Hz.

    Returns:
        `sample_rate * WINDOW_SIZE` float32 samples.
    """
    frames = round(WINDOW_SIZE * native_rate)
    begin = round(start * native_rate)
    lead = max(-begin, 0)

    audio, _ = sf.read(
        path,
        start=begin + lead,
        frames=frames - lead,
        dtype="float32",
        always_2d=True,
    )
    audio = audio.mean(axis=1)
    audio = np.pad(audio, (lead, frames - lead - len(audio)))

    if native_rate != sample_rate:
        divisor = gcd(native_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, native_rate // divisor)

    return np.asarray(audio, dtype=np.float32)


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

    def split_blocks(
        self,
        block_size: float = BLOCK_SIZE,
        cycle: list[str] = BLOCK_CYCLE,
        margin: float = SPLIT_MARGIN,
    ) -> tuple["BowelDataset", "BowelDataset", "BowelDataset"]:
        """Split each recording in time into train, validation and test blocks.

        Blocks of `block_size` seconds follow one another, and their role
        repeats with `cycle`, which fixes the 70 / 20 / 10 split.

        Args:
            block_size: the length of a block, in seconds.
            cycle: the role of each block in a cycle: "train", "val" or "test".
            margin: the gap, in seconds, between a validation or test block and
                the train windows, which are dropped if closer than this.

        Returns:
            The train, validation and test datasets. A window across a block
            border, or too close to a validation or test block, is in none.
        """

        by_role: dict[str, dict[str, list[Window]]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        for name, windows in self.windows_by_recording.items():
            split: dict[str, list[Window]] = {"train": [], "val": [], "test": []}

            for window in windows:
                role = assign_block(window.start, window.end, block_size, cycle, margin)
                if role is not None:
                    split[role].append(window)

            for key, kept in split.items():
                by_role[key][name] = kept
            logger.info(
                "%s: %d train, %d validation, %d test, %d dropped",
                name,
                len(split["train"]),
                len(split["val"]),
                len(split["test"]),
                len(windows) - sum(len(kept) for kept in split.values()),
            )

        return (
            BowelDataset(windows_by_recording=by_role["train"]),
            BowelDataset(windows_by_recording=by_role["val"]),
            BowelDataset(windows_by_recording=by_role["test"]),
        )


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """The windows of a `BowelDataset` as `(audio, labels)` pairs, read on demand.

    Meant for a `DataLoader`: shuffle the train set, never the test set.
    """

    def __init__(
        self,
        dataset: BowelDataset,
        data_dir: Path,
        sample_rate: int = SAMPLE_RATE,
        normalize: bool = True,
        scales: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            dataset: the windows to serve, e.g. one side of
                `BowelDataset.split_blocks`.
            data_dir: the directory holding the WAV files.
            sample_rate: the rate every window is resampled to, in Hz.
            normalize: divide each recording by the median RMS of its
                windows, so recordings of different levels compare. The
                median ignores the long loud segments (`n`, `v`) that would
                inflate a global RMS. Reads every window once to get it.
            scales: the factor of each recording, used as is. Give the
                output of `scales_of` on the whole dataset to every split:
                computed from a split's own windows, the factors differ
                between train, validation and test.
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
        if scales is not None:
            self.scales = scales
        else:
            self.scales = {
                name: self.median_rms(name=name, windows=windows) if normalize else 1.0
                for name, windows in dataset.windows_by_recording.items()
            }

    @classmethod
    def scales_of(
        cls, dataset: BowelDataset, data_dir: Path, sample_rate: int = SAMPLE_RATE
    ) -> dict[str, float]:
        """The normalization factor of each recording of `dataset`.

        Args:
            dataset: the whole dataset, before any split.
            data_dir: the directory holding the WAV files.
            sample_rate: the rate every window is resampled to, in Hz.

        Returns:
            Recording -> median RMS of all its windows, to give as `scales` to
            every split so that they are normalized alike.
        """
        raw = cls(
            dataset=dataset,
            data_dir=data_dir,
            sample_rate=sample_rate,
            normalize=False,
        )

        return {
            name: raw.median_rms(name=name, windows=windows)
            for name, windows in dataset.windows_by_recording.items()
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one window.

        Args:
            index: the position in the dataset.

        Returns:
            The mono audio, `sample_rate * WINDOW_SIZE` float32 samples
            divided by its recording's median RMS, and the multi-hot labels
            as float32, as `BCEWithLogitsLoss` expects.
        """
        name, window = self.items[index]
        audio = self.read(name, window) / self.scales[name]

        return torch.from_numpy(audio), torch.tensor(window.labels, dtype=torch.float32)

    def median_rms(self, name: str, windows: list[Window]) -> float:
        """The median RMS of the windows of one recording, before scaling.

        Args:
            name: the recording.
            windows: its windows.

        Returns:
            The median RMS. Raises `ValueError` if it is zero, as a silent
            recording cannot be normalized.
        """
        rms = [np.sqrt((self.read(name, window) ** 2).mean()) for window in windows]
        median = float(np.median(rms))
        if median == 0:
            raise ValueError(f"{name} is silent, it cannot be normalized")

        return median

    def read(self, name: str, window: Window) -> np.ndarray:
        """Read the raw audio of one window: mono, resampled, not scaled.

        Args:
            name: the recording.
            window: the window to read.

        Returns:
            `sample_rate * WINDOW_SIZE` float32 samples.
        """
        return read_audio(
            path=self.data_dir / name,
            native_rate=self.native_rates[name],
            start=window.start,
            sample_rate=self.sample_rate,
        )


@dataclass
class Segment:
    """A one-second span centred on a sound, with its class."""

    center: float
    label: int

    @property
    def start(self) -> float:
        return self.center - WINDOW_SIZE / 2

    @property
    def end(self) -> float:
        return self.center + WINDOW_SIZE / 2


class SegmentDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Segments centred on `sb`, `mb`, `h` events or on reject spots, read on demand.

    Trains the second stage, which types a segment found by the detector.
    Each item is `(audio, class index)`, the index in `CLASSES`.
    """

    def __init__(
        self,
        segments_by_recording: dict[str, list[Segment]],
        data_dir: Path,
        scales: dict[str, float],
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """
        Args:
            segments_by_recording: the segments of each recording.
            data_dir: the directory holding the WAV files.
            scales: the normalization factor of each recording, from
                `WindowDataset.scales_of`, shared by every set.
            sample_rate: the rate every segment is resampled to, in Hz.
        """
        self.segments_by_recording = segments_by_recording
        self.data_dir = data_dir
        self.scales = scales
        self.sample_rate = sample_rate
        self.items = [
            (name, segment)
            for name, segments in segments_by_recording.items()
            for segment in segments
        ]
        self.native_rates = {
            name: sf.info(data_dir / name).samplerate for name in segments_by_recording
        }

    @classmethod
    def from_dir(
        cls,
        data_dir: Path,
        scales: dict[str, float],
        seed: int = SEGMENT_SEED,
        sample_rate: int = SAMPLE_RATE,
    ) -> "SegmentDataset":
        """Build the segments of every recording of a directory.

        One segment per `sb`, `mb` or `h` event, centred on its middle, and
        as many reject segments as `mb` events, centred on random spots with
        no such event within `REJECT_CLEARANCE`. The spots are drawn here,
        once, so the sets do not change between epochs.

        Args:
            data_dir: the directory, `{stem}.wav` paired with `{stem}.txt`.
            scales: the normalization factor of each recording.
            seed: the seed of the reject spots.
            sample_rate: the rate every segment is resampled to, in Hz.

        Returns:
            The segments of every recording that has an annotation file.
        """
        rng = np.random.default_rng(seed)
        segments_by_recording = {}

        for wav_path in sorted(data_dir.glob("*.wav")):
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                logger.warning("%s: no matching annotation file", wav_path.name)
                continue
            duration = sf.info(wav_path).duration
            events = [
                event
                for event in load_clean_events(path=txt_path, duration=duration)
                if event.label in SEGMENT_LABELS and event.end > event.start
            ]
            segments = [
                Segment(
                    center=(event.start + event.end) / 2,
                    label=CLASSES.index(event.label),
                )
                for event in events
            ]

            starts = np.array([event.start for event in events])
            ends = np.array([event.end for event in events])
            wanted = sum(event.label == "mb" for event in events)
            rejects = 0
            # Capped, in case the recording leaves no room for reject spots.
            for _ in range(100 * max(wanted, 1)):
                if rejects == wanted:
                    break
                center = rng.uniform(WINDOW_SIZE / 2, duration - WINDOW_SIZE / 2)
                near = (starts < center + REJECT_CLEARANCE) & (
                    ends > center - REJECT_CLEARANCE
                )
                if near.any():
                    continue
                segments.append(Segment(center=center, label=CLASSES.index(REJECT)))
                rejects += 1
            if rejects < wanted:
                logger.warning(
                    "%s: only %d reject spots of %d", wav_path.name, rejects, wanted
                )

            segments_by_recording[wav_path.name] = sorted(
                segments, key=lambda segment: segment.center
            )
            logger.info(
                "%s segments: %s",
                wav_path.name,
                ", ".join(
                    f"{label} {sum(s.label == i for s in segments)}"
                    for i, label in enumerate(CLASSES)
                ),
            )

        return cls(
            segments_by_recording=segments_by_recording,
            data_dir=data_dir,
            scales=scales,
            sample_rate=sample_rate,
        )

    def split_blocks(
        self,
        block_size: float = BLOCK_SIZE,
        cycle: list[str] = BLOCK_CYCLE,
        margin: float = SPLIT_MARGIN,
    ) -> tuple["SegmentDataset", "SegmentDataset", "SegmentDataset"]:
        """Split in time into train, validation and test, like `BowelDataset`.

        The blocks and the margin are those of `BowelDataset.split_blocks`,
        applied to the one-second span of each segment.

        Args:
            block_size: the length of a block, in seconds.
            cycle: the role of each block in a cycle: "train", "val" or "test".
            margin: the gap, in seconds, between a validation or test block and
                the train segments, which are dropped if closer than this.

        Returns:
            The train, validation and test datasets. A segment across a block
            border, or too close to a validation or test block, is in none.
        """
        by_role: dict[str, dict[str, list[Segment]]] = {
            "train": {},
            "val": {},
            "test": {},
        }

        for name, segments in self.segments_by_recording.items():
            split: dict[str, list[Segment]] = {"train": [], "val": [], "test": []}
            for segment in segments:
                role = assign_block(
                    segment.start, segment.end, block_size, cycle, margin
                )
                if role is not None:
                    split[role].append(segment)
            for key, kept in split.items():
                by_role[key][name] = kept
            logger.info(
                "%s: %d train, %d validation, %d test segments, %d dropped",
                name,
                len(split["train"]),
                len(split["val"]),
                len(split["test"]),
                len(segments) - sum(len(kept) for kept in split.values()),
            )

        def build(role: str) -> "SegmentDataset":
            return SegmentDataset(
                segments_by_recording=by_role[role],
                data_dir=self.data_dir,
                scales=self.scales,
                sample_rate=self.sample_rate,
            )

        return build("train"), build("val"), build("test")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one segment.

        Args:
            index: the position in the dataset.

        Returns:
            The mono audio, `sample_rate * WINDOW_SIZE` float32 samples
            divided by its recording's median RMS, and the class index as a
            long, as `CrossEntropyLoss` expects.
        """
        name, segment = self.items[index]
        audio = read_audio(
            path=self.data_dir / name,
            native_rate=self.native_rates[name],
            start=segment.start,
            sample_rate=self.sample_rate,
        )

        return (
            torch.from_numpy(audio / self.scales[name]),
            torch.tensor(segment.label, dtype=torch.long),
        )


# The detector looks at the log-Mel frames: one every `mel.hop_length` samples,
# and `FRAMES` of them per window (the centred STFT adds one at the end).
FRAME_HOP = CONFIG.mel.hop_length / SAMPLE_RATE
FRAMES = 1 + round(WINDOW_SIZE * SAMPLE_RATE) // CONFIG.mel.hop_length


class DetectionDataset(WindowDataset):
    """The windows of a `BowelDataset` as `(audio, frame targets)` pairs.

    Trains the first stage, which finds when a bowel sound (`sb`, `mb` or
    `h`, whatever its type) is going on. The target of a window has one 0/1
    per frame, shape `(FRAMES,)`: 1 where the frame centre lies in such an
    event. `n` and `v` are background.
    """

    def __init__(
        self,
        dataset: BowelDataset,
        data_dir: Path,
        sample_rate: int = SAMPLE_RATE,
        normalize: bool = True,
        scales: dict[str, float] | None = None,
    ) -> None:
        """Same arguments as `WindowDataset`; the events are read from `data_dir`."""
        super().__init__(
            dataset=dataset,
            data_dir=data_dir,
            sample_rate=sample_rate,
            normalize=normalize,
            scales=scales,
        )
        self.events = {}
        for name in dataset.windows_by_recording:
            events = load_clean_events(
                path=(data_dir / name).with_suffix(".txt"),
                duration=sf.info(data_dir / name).duration,
            )
            self.events[name] = [
                event
                for event in events
                if event.label in SEGMENT_LABELS and event.end > event.start
            ]

    def frame_targets(self, name: str, window: Window) -> np.ndarray:
        """The 0/1 target of each frame of a window.

        Args:
            name: the recording.
            window: the window.

        Returns:
            `FRAMES` float32 values, 1 where a frame centre is in an event.
        """
        centers = window.start + np.arange(FRAMES) * FRAME_HOP
        targets = np.zeros(FRAMES, dtype=np.float32)
        for event in self.events[name]:
            targets[(centers >= event.start) & (centers <= event.end)] = 1.0

        return targets

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one window.

        Args:
            index: the position in the dataset.

        Returns:
            The audio, as in `WindowDataset`, and its `FRAMES` float32 targets.
        """
        name, window = self.items[index]
        audio, _ = super().__getitem__(index)

        return audio, torch.from_numpy(self.frame_targets(name, window))
