"""Explore bowel sound recordings and check their annotation files.

Each `{stem}.wav` is paired with a `{stem}.txt` whose lines are
`start end label`. The recordings are never modified — the script logs what
it finds, then saves a text report and two figures to `--docs-dir`.
"""

import argparse
import csv
import logging
from collections import Counter
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.event import KNOWN_LABELS, LABEL_MAPPING, Event
from src.recording import Recording
from src.utils import (
    all_durations,
    duration_statistics,
    durations_by_label,
    parse_events,
)

logger = logging.getLogger(__name__)

# Issues listed per type before the rest is summarised as "... and N more".
MAX_SHOWN = 10

DOCS_DIR = Path("docs")
REPORT_NAME = "initial_dataset_report.txt"
DISTRIBUTION_FIGURE = "event_duration_distribution.png"
BY_LABEL_FIGURE = "event_duration_by_label.png"
TABLE_NAME = "event_duration_table.csv"

# Statistics written to the report, overall and per label.
OVERALL_STATISTICS = [
    "count",
    "min",
    "max",
    "mean",
    "median",
    "std",
    "p25",
    "p75",
    "p90",
    "p95",
]
LABEL_STATISTICS = ["count", "min", "max", "mean", "median", "p90"]
# Statistics of the per-label duration table, in column order.
TABLE_STATISTICS = ["count", "min", "median", "p90", "max"]


def check_events(events: list[Event], duration: float) -> dict[str, list[str]]:
    """Look for events that cannot be right.

    Args:
        events: the parsed events.
        duration: the length of the matching WAV, in seconds.

    Returns:
        Issue type -> descriptions, with an empty list for a clean type.
    """
    issues: dict[str, list[str]] = {
        "unknown_labels": [],
        "invalid_intervals": [],
        "negative_timestamps": [],
        "out_of_bounds": [],
    }

    for event in events:
        if event.label not in KNOWN_LABELS:
            issues["unknown_labels"].append(f"{event.duration}: {event.label}")
        if event.start >= event.end:
            issues["invalid_intervals"].append(event.duration)
        if event.start < 0 or event.end < 0:
            issues["negative_timestamps"].append(event.duration)
        if event.start > duration or event.end > duration:
            issues["out_of_bounds"].append(event.duration)

    return issues


def find_overlaps(events: list[Event]) -> list[str]:
    """Describe the events that start before the previous one has ended.

    Args:
        events: the parsed events.

    Returns:
        One description per overlapping pair. Overlaps are not an issue, so
        they are kept out of `check_events` and only reported for inspection.
    """
    overlaps = []

    # Sorted by start, so an overlap can only be with the next event.
    ordered = sorted(events, key=lambda event: event.start)
    for previous, current in pairwise(ordered):
        if current.start < previous.end:
            overlaps.append(
                f"{previous.duration} ({previous.label}) overlaps "
                f"{current.duration} ({current.label})"
            )

    return overlaps


def format_statistics(
    statistics: dict[str, float], keys: list[str], indent: str = ""
) -> list[str]:
    """Format statistics as aligned report lines.

    Args:
        statistics: the output of `duration_statistics`.
        keys: which statistics to write, in order.
        indent: prefix of every line.

    Returns:
        One line per key; `count` has no unit, the others are in seconds.
    """
    lines = []
    for key in keys:
        if key == "count":
            lines.append(f"{indent}Count  : {int(statistics[key])}")
        else:
            lines.append(f"{indent}{key.capitalize():<7}: {statistics[key]:.4f} s")

    return lines


def save_figure(path: Path) -> None:
    """Save the current figure to `path` and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved figure: %s", path)


def plot_duration_distribution(events: list[Event], docs_dir: Path) -> None:
    """Save a histogram of the event lengths.

    Args:
        events: the events of every recording.
        docs_dir: where the figure is saved.
    """
    durations = all_durations(events)
    if not durations:
        logger.warning("No valid event durations to plot.")
        return

    # Durations span orders of magnitude (median ~0.2 s, a few events of
    # minutes): log-spaced bins on a log axis keep the bulk readable.
    bins = np.logspace(np.log10(min(durations)), np.log10(max(durations)), 30).tolist()

    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=bins)
    plt.xscale("log")
    plt.xlabel("Event duration (s, log scale)")
    plt.ylabel("Number of events")
    plt.title("Distribution of event durations")
    save_figure(path=docs_dir / DISTRIBUTION_FIGURE)


def plot_duration_by_label(events: list[Event], docs_dir: Path) -> None:
    """Save a boxplot of the event lengths for each label.

    Args:
        events: the events of every recording.
        docs_dir: where the figure is saved.
    """
    durations = durations_by_label(events)
    if not durations:
        logger.warning("No valid event durations to plot by label.")
        return

    labels = sorted(durations)

    plt.figure(figsize=(8, 5))
    plt.boxplot([durations[label] for label in labels], tick_labels=labels)
    plt.xlabel("Label")
    plt.ylabel("Event duration (s)")
    plt.title("Event duration by label")
    save_figure(path=docs_dir / BY_LABEL_FIGURE)


def duration_table(events: list[Event]) -> list[list[str | float]]:
    """Build the per-label duration table.

    Args:
        events: the events of every recording.

    Returns:
        The header, one row per label, then an `all` row. Counts are
        integers, the other columns are in seconds.
    """
    rows: list[list[str | float]] = [["label", *TABLE_STATISTICS]]

    durations = durations_by_label(events)
    durations_by_row = {label: durations[label] for label in sorted(durations)}
    durations_by_row["all"] = all_durations(events)

    for label, label_durations in durations_by_row.items():
        statistics = duration_statistics(label_durations)
        rows.append([label, *(round(statistics[key], 4) for key in TABLE_STATISTICS)])

    return rows


def write_duration_table(events: list[Event], docs_dir: Path) -> None:
    """Save the per-label duration table as CSV.

    Args:
        events: the events of every recording.
        docs_dir: where the table is saved.
    """
    if not all_durations(events):
        logger.warning("No valid event durations for the table.")
        return

    table_path = docs_dir / TABLE_NAME
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as file:
        csv.writer(file).writerows(duration_table(events))
    logger.info("Saved table: %s", table_path)


def analyze_recording(wav_path: Path, txt_path: Path) -> Recording:
    """Read a recording and run every check on its events.

    Args:
        wav_path: the recording.
        txt_path: its annotation file.

    Returns:
        The recording, with its parsing errors, issues and overlaps.
    """
    info = sf.info(wav_path)
    events, parsing_errors = parse_events(path=txt_path)

    return Recording(
        name=wav_path.name,
        duration=info.duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        events=events,
        parsing_errors=parsing_errors,
        issues=check_events(events=events, duration=info.duration),
        overlaps=find_overlaps(events=events),
    )


def log_recording(recording: Recording) -> None:
    """Log the audio info, label counts, durations and issues of a recording.

    Args:
        recording: the analysed recording.
    """
    logger.info("=" * 60)
    logger.info("File: %s", recording.name)
    logger.info("Sample rate : %d Hz", recording.sample_rate)
    logger.info("Channels    : %d", recording.channels)
    logger.info("Duration    : %.3f s", recording.duration)
    logger.info("Events      : %d", len(recording.events))

    label_counts = Counter(event.label for event in recording.events)
    for label, count in sorted(label_counts.items()):
        logger.info("  %-4s %d", label, count)

    durations = all_durations(recording.events)
    if durations:
        statistics = duration_statistics(durations)
        logger.info("Event duration:")
        for line in format_statistics(statistics, ["min", "max", "mean", "median"]):
            logger.info("  %s", line)

    if recording.parsing_errors:
        logger.warning("Parsing errors: %d", len(recording.parsing_errors))
        for error in recording.parsing_errors:
            logger.warning("  %s", error)

    has_issues = bool(recording.parsing_errors)
    for issue_type, values in recording.issues.items():
        if not values:
            continue
        has_issues = True
        logger.warning("%s: %d", issue_type, len(values))
        for value in values[:MAX_SHOWN]:
            logger.warning("  %s", value)
        if len(values) > MAX_SHOWN:
            logger.warning("  ... and %d more", len(values) - MAX_SHOWN)

    if not has_issues:
        logger.info("No issues found")

    logger.info("Overlaps    : %d", len(recording.overlaps))
    for overlap in recording.overlaps[:MAX_SHOWN]:
        logger.info("  %s", overlap)
    if len(recording.overlaps) > MAX_SHOWN:
        logger.info("  ... and %d more", len(recording.overlaps) - MAX_SHOWN)


def section(title: str) -> list[str]:
    """The lines opening a report section: a blank line, the title, a rule."""
    return ["", title, "-" * 60]


def write_report(recordings: list[Recording], report_path: Path) -> None:
    """Write the dataset report, in full — nothing is truncated.

    Args:
        recordings: every analysed recording.
        report_path: the text file to write.
    """
    events = [event for recording in recordings for event in recording.events]
    total_duration = sum(recording.duration for recording in recordings)

    lines = ["INITIAL DATASET REPORT", "=" * 60]

    lines += section("Dataset overview")
    lines.append(f"Number of recordings : {len(recordings)}")
    lines.append(f"Number of events     : {len(events)}")
    lines.append(f"Total audio duration : {total_duration:.3f} s")

    lines += section("Recordings")
    for recording in recordings:
        lines.append(
            f"{recording.name}: {recording.duration:.3f} s, "
            f"{recording.sample_rate} Hz, {recording.channels} channel(s), "
            f"{len(recording.events)} events"
        )

    lines += section("Label distribution")
    label_counts = Counter(event.label for event in events)
    for label, count in sorted(label_counts.items()):
        lines.append(f"{label:<10} {count}")

    durations = all_durations(events)
    if durations:
        lines += section("Event duration statistics")
        statistics = duration_statistics(durations)
        lines += format_statistics(statistics, OVERALL_STATISTICS)

    lines += section("Event duration by label")
    for label, label_durations in sorted(durations_by_label(events).items()):
        statistics = duration_statistics(label_durations)
        lines.append(f"{label}:")
        lines += format_statistics(statistics, LABEL_STATISTICS, indent="  ")

    lines += section("Event anomalies")
    lines.append("Known label variations:")
    for variation, label in LABEL_MAPPING.items():
        lines.append(f"  {variation} -> {label}")
    for recording in recordings:
        if not any(recording.issues.values()):
            continue
        lines += ["", f"{recording.name}:"]
        for issue_type, values in recording.issues.items():
            if not values:
                continue
            lines.append(f"  {issue_type}: {len(values)}")
            lines += [f"    - {value}" for value in values]

    lines += section("Overlapping events")
    total_overlaps = sum(len(recording.overlaps) for recording in recordings)
    lines.append(f"Total overlapping events: {total_overlaps}")
    lines.append(
        "Overlaps are reported for inspection and are not automatically "
        "considered annotation errors."
    )
    for recording in recordings:
        if not recording.overlaps:
            continue
        lines += ["", f"{recording.name}:"]
        lines += [f"  - {overlap}" for overlap in recording.overlaps]

    parsing_errors = [
        error for recording in recordings for error in recording.parsing_errors
    ]
    if parsing_errors:
        lines += section("Parsing errors")
        lines += [f"- {error}" for error in parsing_errors]

    lines += section("Generated figures")
    lines += [DISTRIBUTION_FIGURE, BY_LABEL_FIGURE]

    lines += section("Generated tables")
    lines.append(TABLE_NAME)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved report: %s", report_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Explore bowel sound audio dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing WAV and TXT files.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="Directory where the report and figures are saved.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    docs_dir = args.docs_dir

    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    wav_files = sorted(data_dir.glob("*.wav"))
    txt_by_stem = {path.stem: path for path in data_dir.glob("*.txt")}

    logger.info("Data directory: %s", data_dir)
    logger.info("WAV files     : %d", len(wav_files))
    logger.info("TXT files     : %d", len(txt_by_stem))

    recordings = []
    for wav_path in wav_files:
        txt_path = txt_by_stem.get(wav_path.stem)
        if txt_path is None:
            logger.warning("%s: no matching annotation file", wav_path.name)
            continue
        recording = analyze_recording(wav_path=wav_path, txt_path=txt_path)
        log_recording(recording=recording)
        recordings.append(recording)

    events = [event for recording in recordings for event in recording.events]
    plot_duration_distribution(events=events, docs_dir=docs_dir)
    plot_duration_by_label(events=events, docs_dir=docs_dir)
    write_duration_table(events=events, docs_dir=docs_dir)
    write_report(recordings=recordings, report_path=docs_dir / REPORT_NAME)


if __name__ == "__main__":
    main()
