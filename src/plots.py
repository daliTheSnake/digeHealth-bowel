"""Figures of a model: its training curves, ROC curves and confusion matrices."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.windows import LABELS


def roc_aucs(true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """The AUC of each label, one against the rest.

    Args:
        true: the true multi-hot labels, shape `(n_windows, n_labels)`.
        probabilities: the predicted probabilities, same shape.

    Returns:
        Label -> AUC. A label without both a positive and a negative window
        has no ROC curve and is left out.
    """
    return {
        label: roc_auc_score(true[:, index], probabilities[:, index])
        for index, label in enumerate(LABELS)
        if true[:, index].min() != true[:, index].max()
    }


def plot_roc(true: np.ndarray, probabilities: np.ndarray, title: str = "") -> None:
    """Draw the ROC curve of each label, one against the rest, in a new figure.

    Args:
        true: the true multi-hot labels, shape `(n_windows, n_labels)`.
        probabilities: the predicted probabilities, same shape.
        title: the title of the figure.
    """
    aucs = roc_aucs(true=true, probabilities=probabilities)

    plt.figure(figsize=(6, 6))
    for index, label in enumerate(LABELS):
        if label not in aucs:
            continue
        false_positive_rate, true_positive_rate, _ = roc_curve(
            true[:, index], probabilities[:, index]
        )
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC {aucs[label]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title} ROC, macro AUC {np.mean(list(aucs.values())):.3f}")
    plt.legend(loc="lower right")


def confusion_matrix(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Count, over all labels at once, what the model gets right and wrong.

    A window has several labels, so there is no single "true class" per
    window. Each window is counted by what it holds, with a `none` class
    (last row and column) for "no label":
    - a label true and predicted: on the diagonal;
    - a label missed (true, not predicted) while another one is falsely
      predicted: row = the missed label, column = the false one;
    - a label missed with no false prediction: row = the label, column `none`;
    - a label falsely predicted with nothing missed: row `none`, column = it;
    - nothing true and nothing predicted: `none` / `none`.
    A window with several misses or false alarms is counted once per pair.

    Args:
        true: the true multi-hot labels, shape `(n_windows, n_labels)`.
        predicted: the predicted ones, same shape.

    Returns:
        The counts, shape `(n_labels + 1, n_labels + 1)`: true in rows,
        predicted in columns.
    """
    none = len(LABELS)
    matrix = np.zeros((none + 1, none + 1), dtype=int)

    for truth, prediction in zip(true.astype(bool), predicted.astype(bool)):
        for label in np.flatnonzero(truth & prediction):
            matrix[label, label] += 1
        missed = np.flatnonzero(truth & ~prediction)
        extra = np.flatnonzero(prediction & ~truth)
        if len(missed) and len(extra):
            for label in missed:
                for other in extra:
                    matrix[label, other] += 1
        elif len(missed):
            matrix[missed, none] += 1
        elif len(extra):
            matrix[none, extra] += 1
        elif not truth.any():
            matrix[none, none] += 1

    return matrix


def plot_confusion(true: np.ndarray, predicted: np.ndarray, title: str = "") -> None:
    """Draw one confusion matrix for all the labels, plus a `none` class.

    See `confusion_matrix` for how a multi-label window is counted. Cells show
    the count and the share of their row.

    Args:
        true: the true multi-hot labels, shape `(n_windows, n_labels)`.
        predicted: the predicted ones, same shape.
        title: the title of the figure.
    """
    matrix = confusion_matrix(true=true, predicted=predicted)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    names = [*LABELS, "none"]

    plt.figure(figsize=(7, 6))
    plt.imshow(shares, vmin=0, vmax=1, cmap="Blues")
    for row in range(len(names)):
        for column in range(len(names)):
            color = "white" if shares[row, column] > 0.5 else "black"
            plt.text(
                column,
                row,
                f"{matrix[row, column]}\n{shares[row, column]:.0%}",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )
    plt.xticks(range(len(names)), names)
    plt.yticks(range(len(names)), names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} confusion matrix".strip())


def plot_training(
    history: dict[str, list[float]], title: str = "", best_epoch: int | None = None
) -> None:
    """Draw the loss and the validation score of each epoch, in a new figure.

    Args:
        history: per epoch, in order, `train_loss`, `val_loss` and `val_f1`.
        title: the title of the figure.
        best_epoch: the epoch whose weights were kept, drawn as a dotted line.
            `None` when none was kept: a resumed run that beat nothing.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, (loss_axes, score_axes) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axes.plot(epochs, history["train_loss"], label="train")
    loss_axes.plot(epochs, history["val_loss"], label="validation")
    if best_epoch is not None:
        loss_axes.axvline(
            best_epoch, color="grey", linestyle=":", label=f"best epoch {best_epoch}"
        )
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Loss")
    loss_axes.set_title("Loss")
    loss_axes.legend()

    score_axes.plot(epochs, history["val_f1"], color="tab:green")
    if best_epoch is not None:
        score_axes.axvline(best_epoch, color="grey", linestyle=":")
    score_axes.set_xlabel("Epoch")
    score_axes.set_ylabel("F1")
    score_axes.set_title("Validation F1")

    figure.suptitle(title)
    figure.tight_layout()


def plot_frame_roc(
    true: np.ndarray, probabilities: np.ndarray, title: str = ""
) -> None:
    """Draw the ROC curve of the sound frames of the detector, in a new figure.

    Args:
        true: the true 0/1 target of every frame.
        probabilities: the predicted probabilities, same shape.
        title: the title of the figure.
    """
    true, probabilities = true.ravel(), probabilities.ravel()
    false_positive_rate, true_positive_rate, _ = roc_curve(true, probabilities)

    plt.figure(figsize=(6, 6))
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{title} ROC, AUC {roc_auc_score(true, probabilities):.3f}")
    plt.legend(loc="lower right")
