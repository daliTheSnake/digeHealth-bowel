"""Preview the log-Mel images the CNN sees, with their labels.

Picks `--per-label` windows for each label, preferring windows where that
label is the only active one, and saves them as a grid to `--docs-dir`.
"""

import argparse
import logging
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import BowelDataset, WindowDataset
from src.models.frontend import N_MELS, MelFrontend
from src.windows import LABELS

logger = logging.getLogger(__name__)

DOCS_DIR = Path("docs")
FIGURE_NAME = "mel_preview.png"
COLUMNS = 5
# Mel bins labelled on the y axis, shown with their frequency in Hz.
TICKS = [0, 16, 32, 48, 63]


def pick_windows(dataset: WindowDataset, per_label: int, seed: int) -> list[int]:
    """Choose windows covering every label.

    Args:
        dataset: the windows to choose from.
        per_label: how many windows to pick for each label.
        seed: the random seed, so the preview is reproducible.

    Returns:
        Indices in `dataset`. For each label, windows where it is the only
        active label come first; others complete if there are too few.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([window.labels for _, window in dataset.items])
    picked = []

    for column in range(len(LABELS)):
        has_label = labels[:, column] == 1
        alone = has_label & (labels.sum(axis=1) == 1)
        candidates = np.flatnonzero(alone if alone.sum() >= per_label else has_label)
        picked += rng.choice(candidates, size=per_label, replace=False).tolist()

    return picked


def plot_mel_preview(dataset: WindowDataset, indices: list[int], path: Path) -> None:
    """Save the log-Mel image of each chosen window, titled with its labels.

    Args:
        dataset: the windows, normalized as the model will see them.
        indices: the windows to show.
        path: the figure to write.
    """
    frontend = MelFrontend()
    audio = torch.stack([dataset[index][0] for index in indices])
    with torch.no_grad():
        images = frontend(audio).squeeze(1).numpy()

    frequencies = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=8000)
    rows = -(-len(indices) // COLUMNS)
    figure, axes = plt.subplots(
        rows,
        COLUMNS,
        figsize=(3.2 * COLUMNS, 3.2 * rows),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    low, high = np.percentile(images, [1, 99])

    for axis, index, image in zip(axes.flat, indices, images):
        name, window = dataset.items[index]
        shown = axis.imshow(
            image,
            origin="lower",
            aspect="auto",
            vmin=low,
            vmax=high,
            extent=(0, image.shape[1] / 100, 0, N_MELS),
        )
        axis.set_title(
            f"{'+'.join(window.active_labels) or 'background'}\n"
            f"{name[:-4]} @ {window.start:.1f} s",
            fontsize=9,
        )
        axis.set_xlabel("Time (s)")
        axis.set_yticks(TICKS, [f"{frequencies[tick]:.0f}" for tick in TICKS])
    for axis in axes[:, 0]:
        axis.set_ylabel("Frequency (Hz, mel scale)")
    for axis in list(axes.flat)[len(indices) :]:
        axis.axis("off")
    figure.colorbar(shown, ax=axes, label="log power", shrink=0.8)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Preview log-Mel images.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--per-label", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.data_dir}")

    data = BowelDataset.from_dir(data_dir=args.data_dir)
    scales = WindowDataset.scales_of(dataset=data, data_dir=args.data_dir)
    dataset = WindowDataset(dataset=data, data_dir=args.data_dir, scales=scales)

    indices = pick_windows(dataset=dataset, per_label=args.per_label, seed=args.seed)
    plot_mel_preview(dataset=dataset, indices=indices, path=args.docs_dir / FIGURE_NAME)


if __name__ == "__main__":
    main()
