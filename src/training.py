"""Training and evaluation of a model, shared by every network."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.dataset import CLASSES, DetectionDataset, SegmentDataset, WindowDataset
from src.plots import (
    plot_confusion,
    plot_frame_roc,
    plot_roc,
    plot_training,
    roc_aucs,
)
from src.windows import LABELS

logger = logging.getLogger(__name__)

EPOCHS = CONFIG.training.epochs
BATCH_SIZE = CONFIG.training.batch_size
LEARNING_RATE = CONFIG.training.learning_rate
# A label is predicted when its probability reaches this threshold.
THRESHOLD = CONFIG.training.threshold
# Cap on the weight of a rare label's positives.
MAX_POS_WEIGHT = CONFIG.training.max_pos_weight


class Trainer[D: WindowDataset | SegmentDataset]:
    """Train, evaluate and reload any network mapping audio to logits.

    `D` is the dataset the trainer works on. The network only has to take a
    batch `(batch, samples)`, so every model shares this loop. A subclass sets
    the loss (`make_criterion`) and what is reported (`evaluate`).
    """

    def __init__(
        self,
        model: nn.Module,
        model_path: Path,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        threshold: float = THRESHOLD,
    ) -> None:
        """
        Args:
            model: the network.
            model_path: where the best weights are saved and loaded.
            epochs: how many passes over the train windows.
            batch_size: windows per step.
            learning_rate: the Adam learning rate.
            threshold: a label is predicted from this probability.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_path = model_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold = threshold

    def make_criterion(self, train_set: D) -> nn.Module:
        """The loss, which may depend on the train set, e.g. its class balance."""
        raise NotImplementedError

    def to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Turn logits into one probability per label."""
        return torch.sigmoid(logits)

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """Turn probabilities into the predicted multi-hot labels."""
        return (probabilities >= self.threshold).astype(int)

    def score(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """The number that picks the best epoch: the macro F1 over the labels."""
        return float(f1_score(true, predicted, average="macro", zero_division=0))

    def infer(self, dataset: D) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model on a set, without shuffling or learning.

        Args:
            dataset: the windows to predict.

        Returns:
            The logits and the true multi-hot labels, shape `(n_windows, n_labels)`.
        """
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []

        self.model.eval()
        with torch.no_grad():
            for audio, target in loader:
                logits.append(self.model(audio.to(self.device)))
                targets.append(target.to(self.device))

        return torch.cat(logits), torch.cat(targets)

    def predict_proba(self, dataset: D) -> tuple[np.ndarray, np.ndarray]:
        """Predict the probability of each label for every window.

        Args:
            dataset: the windows to predict.

        Returns:
            The true multi-hot labels and the probabilities, same shape.
        """
        logits, targets = self.infer(dataset)

        return (
            targets.cpu().numpy().astype(int),
            self.to_probabilities(logits).cpu().numpy(),
        )

    def predict(self, dataset: D) -> tuple[np.ndarray, np.ndarray]:
        """Predict the labels of every window of `dataset`.

        Args:
            dataset: the windows to predict.

        Returns:
            The true and the predicted multi-hot labels, same shape.
        """
        true, probabilities = self.predict_proba(dataset)

        return true, self.decide(probabilities)

    def evaluate(self, dataset: D, name: str = "", display: bool = False) -> float:
        """Log the scores of the model on a set, and draw the figures if asked.

        Args:
            dataset: the set to evaluate.
            name: how to call the set in the log and the figures, e.g. "Test".
            display: draw the figures. They are only built: the caller shows
                them with `plt.show()`.

        Returns:
            The score of the set, as the subclass defines it.
        """
        raise NotImplementedError

    def fit(
        self,
        train_set: D,
        val_set: D,
        name: str = "",
        display: bool = False,
        resume: bool = False,
    ) -> None:
        """Train, keeping the epoch with the best validation macro F1.

        The train windows are shuffled at each epoch. At the end the model
        holds the best weights, which are also saved to `model_path`.

        Args:
            train_set: the train windows.
            val_set: the validation windows, to choose the best epoch.
            name: how to call the model in the figure, e.g. "Training HuBERT".
            display: draw the loss and the validation F1 of each epoch. The
                figure is only built: the caller shows it with `plt.show()`.
            resume: start from the weights saved at `model_path` instead of
                from scratch, and train `epochs` more. The saved weights set
                the score to beat, so the file is only replaced by a better
                epoch. The optimizer starts afresh: only the weights are saved.
        """
        criterion = self.make_criterion(train_set)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
        }
        best_f1 = -1.0
        best_epoch = None
        if resume:
            self.load()
            _, best_f1 = self.validate(val_set, criterion)
            logger.info(
                "Resuming from %s: validation F1 %.3f", self.model_path, best_f1
            )

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for audio, target in loader:
                audio, target = audio.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(audio), target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(audio)

            val_loss, f1 = self.validate(val_set, criterion)
            history["train_loss"].append(train_loss / len(train_set))
            history["val_loss"].append(val_loss)
            history["val_f1"].append(f1)
            logger.info(
                "Epoch %d/%d: train loss %.4f, val loss %.4f, val F1 %.3f",
                epoch,
                self.epochs,
                history["train_loss"][-1],
                val_loss,
                f1,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), self.model_path)

        if best_epoch is None:
            logger.info("No epoch beat the saved weights, kept as they are")
        logger.info("Best validation F1: %.3f", best_f1)
        self.load()
        if display:
            plot_training(history, title=name, best_epoch=best_epoch)

    def validate(self, val_set: D, criterion: nn.Module) -> tuple[float, float]:
        """The loss and the score of the model on the validation set.

        Args:
            val_set: the validation windows.
            criterion: the loss.

        Returns:
            The validation loss and the score that picks the best epoch.
        """
        logits, targets = self.infer(val_set)
        predicted = self.decide(self.to_probabilities(logits).cpu().numpy())
        score = self.score(targets.cpu().numpy().astype(int), predicted)

        return criterion(logits, targets).item(), score

    def load(self) -> None:
        """Load the best weights saved by `fit`."""
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)


class WindowTrainer(Trainer[WindowDataset]):
    """Train a network that says which labels are in each window, several at once.

    One independent yes/no per label: a sigmoid and a binary cross-entropy.
    """

    def positive_weights(self, dataset: WindowDataset) -> torch.Tensor:
        """Weight each label's positives by negatives over positives.

        Offsets rare labels such as `h` in the loss.

        Args:
            dataset: the train windows.

        Returns:
            One weight per label, between 1 and `MAX_POS_WEIGHT`.
        """
        labels = np.array([window.labels for _, window in dataset.items])
        positives = labels.sum(axis=0)
        weights = np.clip(
            (len(labels) - positives) / np.maximum(positives, 1), 1.0, MAX_POS_WEIGHT
        )

        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def make_criterion(self, train_set: WindowDataset) -> nn.Module:
        """The loss: one independent yes/no per label."""
        return nn.BCEWithLogitsLoss(pos_weight=self.positive_weights(train_set))

    def evaluate(
        self, dataset: WindowDataset, name: str = "", display: bool = False
    ) -> float:
        """Log precision, recall, F1 and AUC of each label.

        Args:
            dataset: the windows to evaluate.
            name: how to call the set in the log and the figures, e.g. "Test".
            display: draw the ROC curves and the confusion matrix. The figures
                are only built: the caller shows them with `plt.show()`.

        Returns:
            The macro F1 over the labels.
        """
        true, probabilities = self.predict_proba(dataset)
        predicted = self.decide(probabilities)
        report = classification_report(
            true, predicted, target_names=LABELS, zero_division=0
        )
        logger.info("%s report:\n%s", name, report)
        aucs = roc_aucs(true=true, probabilities=probabilities)
        logger.info(
            "%s AUC: %s",
            name,
            ", ".join(f"{label} {auc:.3f}" for label, auc in aucs.items()),
        )

        if display:
            plot_roc(true=true, probabilities=probabilities, title=name)
            plot_confusion(true=true, predicted=predicted, title=name)

        return self.score(true, predicted)


class SegmentTrainer(Trainer[SegmentDataset]):
    """Train a network that gives each segment exactly one class of `CLASSES`.

    Same loop as `Trainer`, but with a softmax and `CrossEntropyLoss`: a
    segment is one of `h`, `mb`, `sb` or `reject`, never several.
    """

    def make_criterion(self, train_set: SegmentDataset) -> nn.Module:
        """The loss, each class weighted by the inverse of its frequency."""
        labels = np.array([segment.label for _, segment in train_set.items])
        counts = np.bincount(labels, minlength=len(CLASSES))
        weights = len(labels) / (len(CLASSES) * np.maximum(counts, 1))

        return nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )

    def to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Turn logits into class probabilities that sum to 1."""
        return torch.softmax(logits, dim=1)

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """Turn probabilities into the most probable class index."""
        return probabilities.argmax(axis=1)

    def evaluate(
        self, dataset: SegmentDataset, name: str = "", display: bool = False
    ) -> float:
        """Log precision, recall, F1 and the confusion matrix of each class.

        Args:
            dataset: the segments to evaluate.
            name: how to call the set in the log, e.g. "Test".
            display: also draw the confusion matrix. The figure is only
                built: the caller shows it with `plt.show()`.

        Returns:
            The macro F1 over the classes.
        """
        true, predicted = self.predict(dataset)
        logger.info(
            "%s report:\n%s",
            name,
            classification_report(
                true,
                predicted,
                labels=range(len(CLASSES)),
                target_names=CLASSES,
                zero_division=0,
            ),
        )
        matrix = confusion_matrix(true, predicted, labels=range(len(CLASSES)))
        logger.info(
            "%s confusion (rows true, columns predicted: %s):\n%s",
            name,
            ", ".join(CLASSES),
            matrix,
        )

        if display:
            ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot()
            plt.title(name)

        return float(f1_score(true, predicted, average="macro", zero_division=0))


class DetectorTrainer(Trainer[WindowDataset]):
    """Train a network that says, for each frame, whether a bowel sound is going on.

    Same loop as `Trainer`, but each window has `FRAMES` yes/no targets, and
    the score is the F1 of the "sound" frames.
    """

    def make_criterion(self, train_set: WindowDataset) -> nn.Module:
        """The loss, the sound frames weighted by background over sound.

        Raises `TypeError` unless `train_set` is a `DetectionDataset`: only it
        knows the frame targets. Predicting needs no annotation, so the other
        methods take any `WindowDataset`.
        """
        if not isinstance(train_set, DetectionDataset):
            raise TypeError("the detector trains on a DetectionDataset")
        sound = np.mean(
            [
                train_set.frame_targets(name, window).mean()
                for name, window in train_set.items
            ]
        )
        weight = np.clip((1 - sound) / max(sound, 1e-6), 1.0, MAX_POS_WEIGHT)

        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(weight, dtype=torch.float32, device=self.device)
        )

    def score(self, true: np.ndarray, predicted: np.ndarray) -> float:
        """The F1 of the sound frames, over all the frames of all the windows."""
        return float(f1_score(true.ravel(), predicted.ravel(), zero_division=0))

    def evaluate(
        self, dataset: WindowDataset, name: str = "", display: bool = False
    ) -> float:
        """Log precision, recall, F1 and AUC of the frames.

        Args:
            dataset: the windows to evaluate.
            name: how to call the set in the log, e.g. "Test".
            display: draw the ROC curve and the confusion matrix of the frames.
                The figures are only built: the caller shows them with
                `plt.show()`.

        Returns:
            The F1 of the sound frames.
        """
        true, probabilities = self.predict_proba(dataset)
        predicted = self.decide(probabilities)
        logger.info(
            "%s frame report:\n%s",
            name,
            classification_report(
                true.ravel(),
                predicted.ravel(),
                target_names=["background", "sound"],
                zero_division=0,
            ),
        )
        logger.info(
            "%s frame AUC: %.3f",
            name,
            roc_auc_score(true.ravel(), probabilities.ravel()),
        )

        if display:
            plot_frame_roc(true=true, probabilities=probabilities, title=name)
            ConfusionMatrixDisplay(
                confusion_matrix(true.ravel(), predicted.ravel()),
                display_labels=["background", "sound"],
            ).plot()
            plt.title(name)

        return self.score(true, predicted)
