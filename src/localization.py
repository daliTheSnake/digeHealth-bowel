"""From window predictions to events: start time, end time and type."""

from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks

from src.dataset import FRAME_HOP, FRAMES
from src.event import Event
from src.training import THRESHOLD
from src.windows import HOP, LABELS, WINDOW_SIZE


def windows_to_events(
    probabilities: np.ndarray,
    starts: np.ndarray,
    threshold: float = THRESHOLD,
    window_size: float = WINDOW_SIZE,
    hop: float = HOP,
    max_gap: float = 0.0,
) -> list[Event]:
    """Merge the windows where a label is active into events.

    Each label is treated on its own, so events of different labels may
    overlap, as in the annotations. An event starts at the first window of a
    run of active windows and ends at the end of the last one, so its borders
    are only as precise as the windows: about `hop` seconds.

    Args:
        probabilities: the predicted probabilities, shape
            `(n_windows, len(LABELS))`, windows in time order.
        starts: the start of each window, in seconds.
        threshold: a label is active from this probability.
        window_size: the length of a window, in seconds.
        hop: the step between two consecutive windows, in seconds.
        max_gap: how many seconds of inactive windows may sit inside an event.
            Active windows whose starts are at most `hop + max_gap` apart are
            joined; 0 joins only consecutive windows. Windows from separate
            parts of a recording are never joined, as their starts are far.

    Returns:
        The events, sorted by start.
    """
    events = []

    for index, label in enumerate(LABELS):
        active = np.flatnonzero(probabilities[:, index] >= threshold)
        if len(active) == 0:
            continue

        first = last = float(starts[active[0]])
        for position in active[1:]:
            start = float(starts[position])
            if start - last <= hop + max_gap + 1e-9:
                last = start
                continue
            events.append(Event(start=first, end=last + window_size, label=label))
            first = last = start
        events.append(Event(start=first, end=last + window_size, label=label))

    return sorted(events, key=lambda event: event.start)


def average_frames(probabilities: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Lay the frame scores of overlapping windows on one time grid.

    Args:
        probabilities: the score of each frame of each window, shape
            `(n_windows, FRAMES)`, as given by the detector.
        starts: the start of each window, in seconds.

    Returns:
        One score per frame of the recording, from time 0. A frame covered
        by several windows gets their mean; one covered by none gets 0.
    """
    offsets = np.round(np.asarray(starts) / FRAME_HOP).astype(int)
    total = np.zeros(offsets.max() + FRAMES)
    count = np.zeros_like(total)
    for offset, scores in zip(offsets, probabilities):
        total[offset : offset + FRAMES] += scores
        count[offset : offset + FRAMES] += 1

    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def frames_to_segments(
    scores: np.ndarray,
    threshold: float = THRESHOLD,
    max_gap: float = 0.0,
    min_duration: float = 0.0,
    split_dip: float = 0.0,
) -> list[tuple[float, float]]:
    """Turn frame scores into `(start, end)` segments.

    Args:
        scores: one score per frame, frame `i` centred on `i * FRAME_HOP`.
        threshold: a frame is active from this score.
        max_gap: runs of active frames separated by at most this many seconds
            are joined into one segment.
        min_duration: shorter segments are dropped, in seconds.
        split_dip: a segment is cut at every dip of its score at least this
            deep, i.e. this far below the lower of the two peaks around it.
            0 never cuts. Cuts what the gap filling or the model's smoothing
            glued together.

    Returns:
        The segments, in seconds, in time order.
    """
    active = np.concatenate([[False], scores >= threshold, [False]])
    edges = np.flatnonzero(np.diff(active.astype(int)))
    runs = list(zip(edges[::2], edges[1::2] - 1))

    segments: list[list[int]] = []
    for first, last in runs:
        gap = (first - segments[-1][1] - 1) * FRAME_HOP if segments else None
        if gap is not None and gap <= max_gap + 1e-9:
            segments[-1][1] = last
        else:
            segments.append([first, last])

    if split_dip > 0:
        pieces = []
        for first, last in segments:
            # The frame of each dip is left out, which opens a gap there.
            dips, _ = find_peaks(-scores[first : last + 1], prominence=split_dip)
            bounds = [first - 1, *(first + dips), last + 1]
            pieces += [[start + 1, end - 1] for start, end in pairwise(bounds)]
        segments = pieces

    return [
        (max(first * FRAME_HOP - FRAME_HOP / 2, 0.0), last * FRAME_HOP + FRAME_HOP / 2)
        for first, last in segments
        if (last - first + 1) * FRAME_HOP >= min_duration - 1e-9
    ]


def match_events(
    reference: list[Event], predicted: list[Event], iou_threshold: float
) -> list[tuple[int, int]]:
    """Pair predicted events with reference events, one to one, by overlap.

    The labels are ignored: give lists of one label to match per label.
    Pairs are taken from the highest IoU down, as long as it reaches
    `iou_threshold`.

    Args:
        reference: the annotated events.
        predicted: the predicted events.
        iou_threshold: the smallest intersection over union of a pair.

    Returns:
        The `(reference index, predicted index)` pairs.
    """
    if not reference or not predicted:
        return []

    ref_start = np.array([event.start for event in reference])[:, None]
    ref_end = np.array([event.end for event in reference])[:, None]
    pred_start = np.array([event.start for event in predicted])[None, :]
    pred_end = np.array([event.end for event in predicted])[None, :]
    inter = np.clip(
        np.minimum(ref_end, pred_end) - np.maximum(ref_start, pred_start), 0, None
    )
    iou = inter / (ref_end - ref_start + pred_end - pred_start - inter)

    candidates = np.argwhere(iou >= iou_threshold)
    order = np.argsort(-iou[candidates[:, 0], candidates[:, 1]], kind="stable")
    used_ref, used_pred, pairs = set(), set(), []
    for ref_index, pred_index in candidates[order]:
        if ref_index in used_ref or pred_index in used_pred:
            continue
        used_ref.add(ref_index)
        used_pred.add(pred_index)
        pairs.append((int(ref_index), int(pred_index)))

    return pairs


def event_scores(matched: int, reference: int, predicted: int) -> dict[str, float]:
    """Precision, recall and F1 from event counts.

    Args:
        matched: how many events were paired.
        reference: how many annotated events there are.
        predicted: how many events were predicted.

    Returns:
        `precision`, `recall` and `f1`, each 0 when its denominator is 0.
    """
    precision = matched / predicted if predicted else 0.0
    recall = matched / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if matched else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
