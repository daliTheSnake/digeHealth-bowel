"""Cut a recording into fixed windows and label each one, multi-hot."""

from dataclasses import dataclass

import numpy as np

from src.event import KNOWN_LABELS, Event

# Column order of the multi-hot labels, sorted since a set has no stable order.
# A window with no active label is background: an all-zero row.
LABELS = sorted(KNOWN_LABELS)

WINDOW_SIZE = 1.0
HOP = 0.5
# An event labels a window when it overlaps it by at least the smaller of
# these two: a short event must be half inside, a long one only needs a touch.
MIN_OVERLAP_RATIO = 0.5
MIN_OVERLAP_SECONDS = 0.25
# Float tolerance, so an overlap of exactly the threshold still counts.
EPSILON = 1e-9


@dataclass
class Window:
    """One fixed-length slice of a recording, with its multi-hot labels."""

    start: float
    end: float
    labels: list[int]

    @property
    def active_labels(self) -> list[str]:
        """The names of the active labels, e.g. `["sb", "n"]`."""
        return [label for label, active in zip(LABELS, self.labels) if active]

    @property
    def is_background(self) -> bool:
        """Whether no label is active in this window."""
        return not any(self.labels)

    @staticmethod
    def starts(
        duration: float, window_size: float = WINDOW_SIZE, hop: float = HOP
    ) -> np.ndarray:
        """Start times of the windows covering a recording.

        Args:
            duration: the length of the recording, in seconds.
            window_size: the length of a window, in seconds.
            hop: the step between two window starts, in seconds.

        Returns:
            The starts, in seconds. A trailing partial window is dropped, so
            every window is full length.
        """
        count = int((duration - window_size) // hop) + 1
        return np.arange(max(count, 0)) * hop

    @staticmethod
    def label_matrix(
        events: list[Event],
        starts: np.ndarray,
        window_size: float = WINDOW_SIZE,
    ) -> np.ndarray:
        """Give each window its multi-hot labels from the events it overlaps.

        Args:
            events: the clean events of the recording.
            starts: the window starts, from `Window.starts`.
            window_size: the length of a window, in seconds.

        Returns:
            An array of shape `(len(starts), len(LABELS))`, 1 where the label
            is active. Events that do not end after they start are ignored.
        """
        labels = np.zeros((len(starts), len(LABELS)), dtype=np.int8)

        for event in events:
            length = event.end - event.start
            if length <= 0:
                continue
            overlap = np.minimum(event.end, starts + window_size) - np.maximum(
                event.start, starts
            )
            threshold = min(MIN_OVERLAP_RATIO * length, MIN_OVERLAP_SECONDS)
            labels[overlap >= threshold - EPSILON, LABELS.index(event.label)] = 1

        return labels

    @classmethod
    def make(
        cls,
        events: list[Event],
        duration: float,
        window_size: float = WINDOW_SIZE,
        hop: float = HOP,
    ) -> list["Window"]:
        """Cut a recording into labelled windows.

        Args:
            events: the clean events of the recording.
            duration: the length of the recording, in seconds.
            window_size: the length of a window, in seconds.
            hop: the step between two window starts, in seconds.

        Returns:
            The windows in time order, as laid out by `starts` and labelled
            by `label_matrix`.
        """
        starts = cls.starts(duration=duration, window_size=window_size, hop=hop)
        labels = cls.label_matrix(events=events, starts=starts, window_size=window_size)

        return [
            cls(start=start, end=start + window_size, labels=row)
            for start, row in zip(starts.tolist(), labels.tolist())
        ]
