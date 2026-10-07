"""The two-stage pipeline: detect the sounds, then give each one its type."""

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import CONFIG
from src.dataset import (
    CLASSES,
    REJECT,
    SEGMENT_LABELS,
    BowelDataset,
    DetectionDataset,
    Segment,
    SegmentDataset,
    WindowDataset,
)
from src.event import Event
from src.localization import (
    average_frames,
    event_scores,
    frames_to_segments,
    match_events,
)
from src.training import DetectorTrainer, SegmentTrainer
from src.windows import Window

logger = logging.getLogger(__name__)

# The label of an event whose type is not known yet.
SOUND = "sound"
# What the post-processing tries, the best on the validation set is kept.
THRESHOLDS = CONFIG.pipeline.thresholds
MAX_GAPS = CONFIG.pipeline.max_gaps
MIN_DURATIONS = CONFIG.pipeline.min_durations
# A dip of this depth cuts a segment; 0 never cuts.
SPLIT_DIPS = CONFIG.pipeline.split_dips
# Tuning is done at this IoU; both are reported.
TUNING_IOU = CONFIG.pipeline.tuning_iou
REPORT_IOUS = CONFIG.pipeline.report_ious


def frame_scores(
    trainer: DetectorTrainer, dataset: WindowDataset
) -> dict[str, np.ndarray]:
    """The detector score of every frame, recording by recording.

    Args:
        trainer: the trained detector.
        dataset: the windows to run it on, annotated or not.

    Returns:
        Recording -> one score per 10 ms frame, windows that overlap being
        averaged, frames outside every window being 0.
    """
    _, probabilities = trainer.predict_proba(dataset)
    scores = {}
    for name in dict.fromkeys(name for name, _ in dataset.items):
        rows = [i for i, (item, _) in enumerate(dataset.items) if item == name]
        starts = np.array([dataset.items[i][1].start for i in rows])
        scores[name] = average_frames(probabilities[rows], starts)

    return scores


def reference_events(dataset: DetectionDataset) -> dict[str, list[Event]]:
    """The annotated `sb`, `mb` and `h` events that the windows of a set cover.

    A set holds separate blocks of a recording, so an event is kept only if
    its middle lies in the time covered by the windows: elsewhere the
    pipeline never looks.

    Args:
        dataset: the windows of one set.

    Returns:
        Recording -> its events, in time order.
    """
    events = {}
    for name, annotated in dataset.events.items():
        spans: list[list[float]] = []
        for item, window in dataset.items:
            if item != name:
                continue
            if spans and window.start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], window.end)
            else:
                spans.append([window.start, window.end])
        events[name] = sorted(
            (
                event
                for event in annotated
                if any(
                    start <= (event.start + event.end) / 2 <= end
                    for start, end in spans
                )
            ),
            key=lambda event: event.start,
        )

    return events


def detect(
    scores: dict[str, np.ndarray],
    threshold: float,
    max_gap: float,
    min_duration: float,
    split_dip: float = 0.0,
) -> dict[str, list[Event]]:
    """Find the sounds of every recording, with no type yet.

    Args:
        scores: the frame scores, from `frame_scores`.
        threshold: a frame is a sound from this score.
        max_gap: gaps up to this, in seconds, are filled.
        min_duration: shorter segments are dropped, in seconds.
        split_dip: depth of the score dips that cut a segment, 0 for none.

    Returns:
        Recording -> events labelled `SOUND`.
    """
    return {
        name: [
            Event(start=start, end=end, label=SOUND)
            for start, end in frames_to_segments(
                frames,
                threshold=threshold,
                max_gap=max_gap,
                min_duration=min_duration,
                split_dip=split_dip,
            )
        ]
        for name, frames in scores.items()
    }


def count_matches(
    reference: dict[str, list[Event]],
    predicted: dict[str, list[Event]],
    iou_threshold: float,
    by_label: bool = False,
) -> dict[str, tuple[int, int, int]]:
    """Count matched, annotated and predicted events.

    Args:
        reference: the annotated events by recording.
        predicted: the predicted events by recording.
        iou_threshold: the smallest IoU of a match.
        by_label: match each label on its own, so a pair needs the same label.

    Returns:
        Label -> `(matched, annotated, predicted)`; the single key `SOUND`
        when `by_label` is False, else one key per label of `SEGMENT_LABELS`.
    """
    groups = SEGMENT_LABELS if by_label else [None]
    counts = {}
    for group in groups:
        matched = annotated = found = 0
        for name, events in reference.items():
            ref = [e for e in events if group in (None, e.label)]
            pred = [e for e in predicted.get(name, []) if group in (None, e.label)]
            matched += len(match_events(ref, pred, iou_threshold))
            annotated += len(ref)
            found += len(pred)
        counts[group or SOUND] = (matched, annotated, found)

    return counts


def tune(
    scores: dict[str, np.ndarray], reference: dict[str, list[Event]]
) -> tuple[float, float, float, float]:
    """Choose the threshold, gap, minimum duration and split depth with the best F1.

    Args:
        scores: the frame scores of the validation set.
        reference: its annotated events.

    Returns:
        `(threshold, max_gap, min_duration, split_dip)`.
    """
    best: tuple[float, float, float, float] | None = None
    best_f1 = -1.0
    for params in itertools.product(THRESHOLDS, MAX_GAPS, MIN_DURATIONS, SPLIT_DIPS):
        predicted = detect(scores, *params)
        matched, annotated, found = count_matches(reference, predicted, TUNING_IOU)[
            SOUND
        ]
        f1 = event_scores(matched, annotated, found)["f1"]
        if f1 > best_f1:
            best, best_f1 = params, f1
    assert best is not None  # the F1 is never below 0, so the first grid point wins
    logger.info(
        "Post-processing chosen on validation: threshold %.2f, max gap %.2f s, "
        "min duration %.2f s, split dip %.2f (event F1 %.3f at IoU %.1f)",
        *best,
        best_f1,
        TUNING_IOU,
    )

    return best


def classify(
    trainer: SegmentTrainer,
    detected: dict[str, list[Event]],
    data_dir: Path,
    scales: dict[str, float],
) -> dict[str, list[Event]]:
    """Give each detected sound its type, dropping those judged `reject`.

    The classifier sees one second centred on the middle of the sound.

    Args:
        trainer: the trained segment classifier.
        detected: the sounds, from `detect`.
        data_dir: the directory holding the WAV files.
        scales: the normalization factor of each recording.

    Returns:
        Recording -> events labelled `h`, `mb` or `sb`.
    """
    if not any(detected.values()):
        return {name: [] for name in detected}

    dataset = SegmentDataset(
        segments_by_recording={
            name: [
                Segment(center=(event.start + event.end) / 2, label=0)
                for event in events
            ]
            for name, events in detected.items()
        },
        data_dir=data_dir,
        scales=scales,
    )
    _, probabilities = trainer.predict_proba(dataset)
    classes = probabilities.argmax(axis=1)

    typed: dict[str, list[Event]] = {name: [] for name in detected}
    for (name, _), event, label in zip(
        dataset.items,
        (event for events in detected.values() for event in events),
        classes,
    ):
        if CLASSES[label] != REJECT:
            typed[name].append(
                Event(start=event.start, end=event.end, label=CLASSES[label])
            )

    return typed


def report(
    name: str,
    reference: dict[str, list[Event]],
    detected: dict[str, list[Event]],
    typed: dict[str, list[Event]],
) -> None:
    """Log the event scores of a set: detection alone, then with the types.

    Args:
        name: how to call the set, e.g. "Test".
        reference: its annotated events.
        detected: the sounds found by the detector.
        typed: the same, typed by the classifier, rejects removed.
    """
    for iou in REPORT_IOUS:
        lines: list[tuple[str, int, int, int]] = []
        counts = count_matches(reference, detected, iou)[SOUND]
        lines.append(("detection (any type)", *counts))

        by_label = count_matches(reference, typed, iou, by_label=True)
        for label, label_counts in by_label.items():
            lines.append((f"  {label}", *label_counts))
        matched, annotated, found = (
            sum(counts[i] for counts in by_label.values()) for i in range(3)
        )
        lines.append(("typed (all)", matched, annotated, found))

        logger.info(
            "%s events, IoU >= %.1f:\n%s",
            name,
            iou,
            "\n".join(
                f"  {title:<22} P {s['precision']:.2f}  R {s['recall']:.2f}  "
                f"F1 {s['f1']:.2f}  ({m}/{a} annotated, {f} predicted)"
                for title, m, a, f in lines
                for s in [event_scores(m, a, f)]
            ),
        )


def run(
    detector: DetectorTrainer,
    classifier: SegmentTrainer,
    val_set: DetectionDataset,
    test_set: DetectionDataset,
    data_dir: Path,
    scales: dict[str, float],
    params_path: Path | None = None,
) -> None:
    """Tune the post-processing on validation, then report validation and test.

    Args:
        detector: the trained first stage.
        classifier: the trained second stage.
        val_set: the validation windows, to tune the post-processing.
        test_set: the test windows.
        data_dir: the directory holding the WAV files.
        scales: the normalization factor of each recording.
        params_path: where to save the chosen post-processing, for `predict_file`.
    """
    val_scores = frame_scores(detector, val_set)
    params = tune(val_scores, reference_events(val_set))
    if params_path is not None:
        save_params(params_path, params)

    for name, dataset, scores in [
        ("Validation", val_set, val_scores),
        ("Test", test_set, None),
    ]:
        scores = scores or frame_scores(detector, dataset)
        detected = detect(scores, *params)
        typed = classify(classifier, detected, data_dir, scales)
        report(name, reference_events(dataset), detected, typed)


PARAMS_FILE = "postprocessing.json"
PARAM_NAMES = ("threshold", "max_gap", "min_duration", "split_dip")


def save_params(path: Path, params: tuple[float, float, float, float]) -> None:
    """Save the post-processing chosen by `tune`.

    Args:
        path: the JSON file to write.
        params: `(threshold, max_gap, min_duration, split_dip)`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(zip(PARAM_NAMES, params)), indent=2))


def load_params(path: Path) -> tuple[float, float, float, float]:
    """Load what `save_params` wrote.

    Args:
        path: the JSON file.

    Returns:
        `(threshold, max_gap, min_duration, split_dip)`.
    """
    saved = json.loads(path.read_text())

    return tuple(saved[name] for name in PARAM_NAMES)


def predict_file(
    detector: DetectorTrainer,
    classifier: SegmentTrainer,
    path: Path,
    params: tuple[float, float, float, float],
) -> list[Event]:
    """Find the bowel sounds of one audio file: start, end and type.

    The file is normalized by its own median window RMS, as the training
    recordings were. A tail shorter than one window is not analysed.

    Args:
        detector: the trained first stage.
        classifier: the trained second stage.
        path: the audio file, any format `soundfile` reads.
        params: the post-processing, from `load_params`.

    Returns:
        The events labelled `h`, `mb` or `sb`, in time order. Raises
        `ValueError` if the file is shorter than one window.
    """
    data_dir, name = path.parent, path.name
    windows = Window.make(events=[], duration=sf.info(path).duration)
    if not windows:
        raise ValueError(f"{name} is shorter than one window, nothing to analyse")

    recording = BowelDataset(windows_by_recording={name: windows})
    scales = WindowDataset.scales_of(dataset=recording, data_dir=data_dir)
    dataset = WindowDataset(dataset=recording, data_dir=data_dir, scales=scales)

    detected = detect(frame_scores(detector, dataset), *params)
    typed = classify(classifier, detected, data_dir, scales)

    return typed[name]
