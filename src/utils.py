"""Helpers shared by the scripts: reading annotation files, event durations."""

import logging
from pathlib import Path

import numpy as np

from src.event import KNOWN_LABELS, Event

logger = logging.getLogger(__name__)


def parse_events(path: Path, normalize: bool = True) -> tuple[list[Event], list[str]]:
    """Parse an annotation file.

    Args:
        path: the `.txt` file, one `start end label` per line.
        normalize: map variant labels to their canonical one, see
            `Event.normalize_label`.

    Returns:
        The parsed events and one message per malformed line, prefixed with
        the file name so they stay readable once gathered across recordings.
        Blank lines are skipped; malformed ones are reported, never raised.
    """
    events = []
    errors = []
    lines = path.read_text().splitlines()
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            errors.append(
                f"{path.name}:{line_number}: expected 3 fields, got {len(parts)}"
            )
            continue
        try:
            start = float(parts[0])
            end = float(parts[1])
        except ValueError:
            errors.append(f"{path.name}:{line_number}: invalid timestamps")
            continue
        event = Event(start=start, end=end, label=parts[2])
        if normalize:
            event.label = event.normalize_label()
        events.append(event)

    return events, errors


def load_clean_events(path: Path, duration: float) -> list[Event]:
    """Parse an annotation file and keep only the events that can be trusted.

    Args:
        path: the `.txt` file, one `start end label` per line.
        duration: the length of the matching WAV, in seconds.

    Returns:
        The events with normalized labels, minus those with a label still
        unknown after normalization or an end past `duration`. Malformed
        lines are dropped too. Logs a single message with the number of
        annotations dropped.
    """
    events, errors = parse_events(path=path)
    clean = [
        event
        for event in events
        if event.label in KNOWN_LABELS and max(event.start, event.end) <= duration
    ]
    dropped = len(errors) + len(events) - len(clean)
    logger.info("%s: dropped %d annotations due to anomalies", path.name, dropped)

    return clean


def durations_by_label(events: list[Event]) -> dict[str, list[float]]:
    """Group the event lengths by label.

    Args:
        events: the parsed events.

    Returns:
        Label -> lengths in seconds. Events that do not end after they start
        are left out, `check_events` reports them.
    """
    durations: dict[str, list[float]] = {}
    for event in events:
        if event.end > event.start:
            durations.setdefault(event.label, []).append(event.end - event.start)

    return durations


def all_durations(events: list[Event]) -> list[float]:
    """The valid event lengths, in seconds, whatever their label."""
    return [
        duration
        for durations in durations_by_label(events).values()
        for duration in durations
    ]


def duration_statistics(durations: list[float]) -> dict[str, float]:
    """Summarise a non-empty list of event lengths.

    Args:
        durations: the lengths, in seconds.

    Returns:
        `count`, `min`, `max`, `mean`, `median`, `std` and the `p25`, `p75`,
        `p90`, `p95` percentiles.
    """
    values = np.asarray(durations)
    return {
        "count": len(values),
        "min": values.min(),
        "max": values.max(),
        "mean": values.mean(),
        "median": np.median(values),
        "std": values.std(),
        "p25": np.percentile(values, 25),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "p95": np.percentile(values, 95),
    }
