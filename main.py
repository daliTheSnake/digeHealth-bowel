"""The bowel sound CLI: find the sounds of an audio file, or train and score models.

Predict:   `main.py predict recording.wav`
Train:     `main.py train detect --data-dir DATA --test`
Evaluate:  `main.py evaluate segments --model resnet50 --data-dir DATA`
Pipeline:  `main.py pipeline --model resnet50 --data-dir DATA`
"""

import argparse
import logging
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src import pipeline
from src.config import CONFIG
from src.dataset import (
    CLASSES,
    BowelDataset,
    DetectionDataset,
    SegmentDataset,
    WindowDataset,
)
from src.models.cnn import BowelCNN
from src.models.detector import BowelDetector
from src.models.hubert_model import HuBERTClassifier
from src.models.resnet_model import ResNet50Classifier
from src.training import (
    DetectorTrainer,
    SegmentTrainer,
    Trainer,
    WindowTrainer,
)

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(CONFIG.paths.weights_dir)
DEFAULT_CLASSIFIER = CONFIG.pipeline.classifier
LEARNING_RATES = CONFIG.models.learning_rates
# Model name -> (network class, learning rate, name shown in the figures).
# The CNN is on hold.
MODELS: dict[str, tuple[type[nn.Module], float, str]] = {
    "hubert": (HuBERTClassifier, LEARNING_RATES["hubert"], "HuBERT"),
    "resnet50": (ResNet50Classifier, LEARNING_RATES["resnet50"], "ResNet-50"),
    "cnn": (BowelCNN, LEARNING_RATES["cnn"], "BowelCNN"),
    "detector": (BowelDetector, LEARNING_RATES["detector"], "Detector"),
}
# The networks that type a window or a segment; the detector has its own task.
CLASSIFIERS = [name for name in MODELS if name != "detector"]
DETECTOR = "detector"
TASKS = ["windows", "segments", "detect"]


def set_seed(seed: int | None) -> None:
    """Seed the random generators, so a run can be repeated (`training.seed`).

    Args:
        seed: the seed, or `None` to leave the generators alone. With a seed,
            PyTorch is also asked for deterministic GPU kernels, which can be a
            little slower; an operation with none only warns.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Needed by deterministic matrix products on the GPU, set before it is used.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Seed: %d", seed)


def build_parser() -> argparse.ArgumentParser:
    """The command line: one subcommand per thing the program does."""
    parser = argparse.ArgumentParser(
        description="Find the start, end and type (sb, mb, h) of the bowel sounds "
        "of an audio file, or train and score the models behind it."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    weights = argparse.ArgumentParser(add_help=False)
    weights.add_argument(
        "--weights-dir",
        type=Path,
        default=WEIGHTS_DIR,
        help="Where the trained weights are (default: `paths.weights_dir`).",
    )
    data = argparse.ArgumentParser(add_help=False, parents=[weights])
    data.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="The directory of WAV and TXT files.",
    )

    classifier = argparse.ArgumentParser(add_help=False)
    classifier.add_argument(
        "--model",
        choices=CLASSIFIERS,
        help="The network, or the segment classifier of the pipeline. Not for "
        f"`detect`, which always trains the detector (default for `pipeline` and "
        f"`predict`: {DEFAULT_CLASSIFIER}).",
    )
    task = argparse.ArgumentParser(add_help=False)
    task.add_argument(
        "task",
        choices=TASKS,
        help="windows: detect the labels of every 1 s window; segments: type a "
        "1 s segment centred on a sound (second stage); detect: find the frames "
        "where a bowel sound goes on (first stage).",
    )

    predict = commands.add_parser(
        "predict",
        parents=[weights, classifier],
        help="Find the bowel sounds of a WAV file with the trained pipeline.",
    )
    predict.add_argument("audio", type=Path, help="The WAV file.")
    predict.add_argument(
        "--output",
        type=Path,
        help="Write the events here, one `start end label` per line, as in the "
        "annotation files. Default: print them.",
    )
    predict.set_defaults(run=run_predict)

    train = commands.add_parser(
        "train",
        parents=[task, data, classifier],
        help="Train a model, keep its best epoch on validation, score it.",
    )
    train.add_argument(
        "--epochs",
        type=int,
        help="How many epochs, instead of `training.epochs` of config.yaml. "
        "With --resume, that many more.",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the saved weights instead of starting afresh. They "
        "set the score to beat, so the file is only replaced by a better epoch.",
    )
    train.add_argument("--test", action="store_true", help="Also score the test set.")
    train.add_argument(
        "--display",
        action="store_true",
        help="Show the loss per epoch, the ROC curves and the confusion matrix.",
    )
    train.set_defaults(run=run_task)

    evaluate = commands.add_parser(
        "evaluate",
        parents=[task, data, classifier],
        help="Score the saved weights of a model on the validation and test sets.",
    )
    evaluate.add_argument("--val", action="store_true", help="Score validation only.")
    evaluate.add_argument("--test", action="store_true", help="Score test only.")
    evaluate.add_argument(
        "--display",
        action="store_true",
        help="Show the ROC curves and the confusion matrix.",
    )
    evaluate.set_defaults(run=run_task)

    commands.add_parser(
        "pipeline",
        parents=[data, classifier],
        help="Chain both stages, tune the post-processing on validation and "
        "score the events on validation and test.",
    ).set_defaults(run=run_pipeline)

    return parser


def parse_args() -> argparse.Namespace:
    """Read the command line.

    Returns:
        The parsed arguments. `model` is filled in: the default classifier for
        `predict` and `pipeline`, the detector for the `detect` task.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command in ("predict", "pipeline"):
        args.model = args.model or DEFAULT_CLASSIFIER
    elif args.task == "detect":
        if args.model is not None:
            parser.error("`detect` always trains the detector, give no --model")
        args.model = DETECTOR
    elif args.model is None:
        parser.error(f"--model is required for `{args.task}`")
    if args.command == "train" and args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")

    return args


def load_stages(args: argparse.Namespace) -> tuple[DetectorTrainer, SegmentTrainer]:
    """Load the trained detector and the trained segment classifier `--model`."""
    model_class, _, _ = MODELS[args.model]
    detector = DetectorTrainer(
        model=BowelDetector(), model_path=args.weights_dir / "detector.pt"
    )
    classifier = SegmentTrainer(
        model=model_class(num_labels=len(CLASSES)),
        model_path=args.weights_dir / f"{args.model}_segments.pt",
    )
    for trainer in (detector, classifier):
        if not trainer.model_path.exists():
            raise SystemExit(
                f"No trained model at {trainer.model_path}: train it first, see "
                "the README"
            )
        trainer.load()

    return detector, classifier


def run_predict(args: argparse.Namespace) -> None:
    """Find the bowel sounds of one audio file and print or save them."""
    if not args.audio.is_file():
        raise SystemExit(f"Not a file: {args.audio}")
    params_path = args.weights_dir / pipeline.PARAMS_FILE
    if not params_path.exists():
        raise SystemExit(
            f"No post-processing at {params_path}: run `pipeline` once, see the README"
        )
    detector, classifier = load_stages(args)

    try:
        events = pipeline.predict_file(
            detector=detector,
            classifier=classifier,
            path=args.audio,
            params=pipeline.load_params(params_path),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    lines = [f"{e.start:.3f} {e.end:.3f} {e.label}" for e in events]
    if args.output:
        args.output.write_text("\n".join(lines) + ("\n" if lines else ""))
        logger.info("%d events written to %s", len(events), args.output)
    else:
        print("\n".join(lines))


def run_pipeline(args: argparse.Namespace) -> None:
    """Score the detector followed by the segment classifier, event by event."""
    data = BowelDataset.from_dir(data_dir=args.data_dir)
    scales = WindowDataset.scales_of(dataset=data, data_dir=args.data_dir)
    _, val, test = data.split_blocks()
    detector, classifier = load_stages(args)

    pipeline.run(
        detector=detector,
        classifier=classifier,
        val_set=DetectionDataset(dataset=val, data_dir=args.data_dir, scales=scales),
        test_set=DetectionDataset(dataset=test, data_dir=args.data_dir, scales=scales),
        data_dir=args.data_dir,
        scales=scales,
        params_path=args.weights_dir / pipeline.PARAMS_FILE,
    )


def run_trainer[D: WindowDataset | SegmentDataset](
    trainer: Trainer[D],
    args: argparse.Namespace,
    display_name: str,
    train_set: Callable[[], D],
    val_set: Callable[[], D],
    test_set: Callable[[], D],
) -> None:
    """Train and/or score one model, whatever its task.

    Reading a set is slow, so each set is a function that builds it, and only
    the ones this run uses are called.

    Args:
        trainer: the model with its weights file.
        args: the command line, `train` or `evaluate`.
        display_name: how to call the model in the log and the figures.
        train_set: builds the train set.
        val_set: builds the validation set.
        test_set: builds the test set.
    """
    training = args.command == "train"
    # `train` always scores validation, which picks the best epoch; `evaluate`
    # scores both sets unless one is asked for.
    do_val = training or args.val or not args.test
    do_test = args.test or (not training and not args.val)

    if training:
        if args.resume and not trainer.model_path.exists():
            raise SystemExit(f"No weights to resume from at {trainer.model_path}")
        train = train_set()
        logger.info("Train: %d", len(train))
        validation = val_set()
        logger.info("Validation: %d", len(validation))
        trainer.fit(
            train_set=train,
            val_set=validation,
            name=f"Training {display_name}",
            display=args.display,
            resume=args.resume,
        )
    elif trainer.model_path.exists():
        trainer.load()
    else:
        raise SystemExit(f"No trained model at {trainer.model_path}: run `train` first")

    if do_val:
        validation = val_set()
        logger.info("Validation: %d", len(validation))
        trainer.evaluate(
            validation, name=f"Validation {display_name}", display=args.display
        )
    if do_test:
        test = test_set()
        logger.info("Test: %d", len(test))
        trainer.evaluate(test, name=f"Test {display_name}", display=args.display)
    if args.display:
        # One call for all the figures, so they open together.
        plt.show()


def run_task(args: argparse.Namespace) -> None:
    """Train or evaluate a model on the windows, the segments or the frames."""
    data = BowelDataset.from_dir(data_dir=args.data_dir)
    # One factor per recording, from the whole dataset, so that the three sets
    # are normalized alike.
    scales = WindowDataset.scales_of(dataset=data, data_dir=args.data_dir)

    model_class, learning_rate, display_name = MODELS[args.model]
    options: dict[str, Any] = {"learning_rate": learning_rate}
    if args.command == "train" and args.epochs:
        options["epochs"] = args.epochs

    if args.task == "segments":
        # The segment sets are cheap to build and read nothing.
        train, val, test = SegmentDataset.from_dir(
            data_dir=args.data_dir, scales=scales
        ).split_blocks()
        run_trainer(
            SegmentTrainer(
                model=model_class(num_labels=len(CLASSES)),
                model_path=args.weights_dir / f"{args.model}_segments.pt",
                **options,
            ),
            args,
            display_name,
            lambda: train,
            lambda: val,
            lambda: test,
        )
        return

    train_split, val_split, test_split = data.split_blocks()
    model_path = args.weights_dir / f"{args.model}.pt"
    if args.task == "detect":
        run_trainer(
            DetectorTrainer(model=model_class(), model_path=model_path, **options),
            args,
            display_name,
            lambda: DetectionDataset(train_split, args.data_dir, scales=scales),
            lambda: DetectionDataset(val_split, args.data_dir, scales=scales),
            lambda: DetectionDataset(test_split, args.data_dir, scales=scales),
        )
    else:
        run_trainer(
            WindowTrainer(model=model_class(), model_path=model_path, **options),
            args,
            display_name,
            lambda: WindowDataset(train_split, args.data_dir, scales=scales),
            lambda: WindowDataset(val_split, args.data_dir, scales=scales),
            lambda: WindowDataset(test_split, args.data_dir, scales=scales),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    set_seed(CONFIG.training.seed)
    if hasattr(args, "data_dir") and not args.data_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.data_dir}")
    args.run(args)


if __name__ == "__main__":
    main()
