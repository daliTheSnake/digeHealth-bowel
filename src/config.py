"""Global parameters, read from `config.yaml`.

Every key is optional: what the file leaves out keeps the default below, and a
key that does not exist is an error, so a typo never goes unnoticed. The file is
`config.yaml` at the root of the repository, or the one named by the
`BOWEL_CONFIG` environment variable.

The rest of the code reads `CONFIG` when it is imported, so edit the file
before running `main.py`. Weights trained with some values (windows, Mel
bands, model sizes) only work with the same values.
"""

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
ENV_VAR = "BOWEL_CONFIG"

SPLIT_ROLES = {"train", "val", "test"}


class ConfigError(ValueError):
    """The configuration file is not valid."""


@dataclass(frozen=True)
class AudioConfig:
    # Every window is brought to this rate, in Hz.
    sample_rate: int = 16000


@dataclass(frozen=True)
class WindowsConfig:
    size: float = 1.0
    hop: float = 0.5
    # An event labels a window when it overlaps it by at least the smaller of
    # `min_overlap_ratio` of the event and `min_overlap_seconds`.
    min_overlap_ratio: float = 0.5
    min_overlap_seconds: float = 0.25


@dataclass(frozen=True)
class SplitConfig:
    # Blocks of `block_size` seconds follow one another, their role repeating
    # with `cycle`, which fixes the split: 7 train, 2 val and 1 test is 70/20/10.
    block_size: float = 60.0
    cycle: list[str] = field(
        default_factory=lambda: [
            "train",
            "train",
            "val",
            "train",
            "test",
            "train",
            "train",
            "val",
            "train",
            "train",
        ]
    )
    # Train windows closer than this to a val or test block are dropped.
    margin: float = 1.0


@dataclass(frozen=True)
class SegmentsConfig:
    # A reject example has no sb, mb or h this close to its centre, in seconds.
    reject_clearance: float = 0.25
    # Seed of the random reject spots.
    seed: int = 0


@dataclass(frozen=True)
class MelConfig:
    n_fft: int = 512
    # 160 samples at 16 kHz: one frame every 10 ms.
    hop_length: int = 160
    n_mels: int = 64
    # Floor added before the log.
    epsilon: float = 1e-6


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 32
    # Used when a model has no entry of its own in `models.learning_rates`.
    learning_rate: float = 1e-3
    # A label or a frame is positive from this probability.
    threshold: float = 0.5
    # Cap on the weight of a rare label's positives in the loss.
    max_pos_weight: float = 10.0
    # Seed of the training, `null` for none: runs then differ.
    seed: int | None = None


@dataclass(frozen=True)
class HubertConfig:
    pretrained: str = "facebook/hubert-base-ls960"
    freeze_encoder: bool = True


@dataclass(frozen=True)
class ResNetConfig:
    # Start from the ImageNet weights.
    pretrained: bool = True


@dataclass(frozen=True)
class CnnConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    dropout: float = 0.5


@dataclass(frozen=True)
class DetectorConfig:
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    gru_hidden: int = 64


@dataclass(frozen=True)
class ModelsConfig:
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {
            "hubert": 2e-5,
            "resnet50": 1e-4,
            "cnn": 1e-3,
            "detector": 1e-3,
        }
    )
    hubert: HubertConfig = field(default_factory=HubertConfig)
    resnet50: ResNetConfig = field(default_factory=ResNetConfig)
    cnn: CnnConfig = field(default_factory=CnnConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass(frozen=True)
class PipelineConfig:
    # The segment classifier used when `--model` is not given to `predict` or `pipeline`.
    classifier: str = "resnet50"
    # The post-processing is chosen on the validation set among these values.
    thresholds: list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5, 0.6, 0.7])
    max_gaps: list[float] = field(default_factory=lambda: [0.0, 0.02, 0.04, 0.06, 0.1])
    min_durations: list[float] = field(default_factory=lambda: [0.02, 0.04, 0.06])
    split_dips: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    )
    # The IoU at which the choice is made, and the ones that are reported.
    tuning_iou: float = 0.5
    report_ious: list[float] = field(default_factory=lambda: [0.3, 0.5])


@dataclass(frozen=True)
class PathsConfig:
    weights_dir: str = "weights"


@dataclass(frozen=True)
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    segments: SegmentsConfig = field(default_factory=SegmentsConfig)
    mel: MelConfig = field(default_factory=MelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def build(cls: type, values: Any, where: str = "") -> Any:
    """Build a config section from the values of the file.

    Args:
        cls: the dataclass of the section.
        values: what the file holds for it, a mapping or `None` for nothing.
        where: the dotted path of the section, for the error messages.

    Returns:
        An instance of `cls`: the defaults, overridden by `values`. Raises
        `ConfigError` on an unknown key or a value of the wrong type.
    """
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ConfigError(f"{where or 'config'} must be a mapping, got {values!r}")

    defaults = cls()
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigError(f"unknown key in {where or 'config'}: {sorted(unknown)}")

    kwargs = {}
    for f in fields(cls):
        default = getattr(defaults, f.name)
        name = f"{where}.{f.name}" if where else f.name
        if f.name not in values:
            continue
        value = values[f.name]
        if is_dataclass(default):
            kwargs[f.name] = build(type(default), value, name)
        elif isinstance(default, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{name} must be a mapping, got {value!r}")
            extra = set(value) - set(default)
            if extra:
                raise ConfigError(f"unknown key in {name}: {sorted(extra)}")
            kwargs[f.name] = {**default, **value}
        elif isinstance(default, list):
            if not isinstance(value, list) or not value:
                raise ConfigError(f"{name} must be a non-empty list, got {value!r}")
            kwargs[f.name] = value
        elif default is None or f.name == "seed":
            if value is not None and not isinstance(value, int):
                raise ConfigError(f"{name} must be an integer or null, got {value!r}")
            kwargs[f.name] = value
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{name} must be true or false, got {value!r}")
            kwargs[f.name] = value
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{name} must be a number, got {value!r}")
            if isinstance(default, int) and value != int(value):
                raise ConfigError(f"{name} must be an integer, got {value!r}")
            kwargs[f.name] = type(default)(value)
        else:
            if not isinstance(value, type(default)):
                raise ConfigError(
                    f"{name} must be a {type(default).__name__}, got {value!r}"
                )
            kwargs[f.name] = value

    return cls(**kwargs)


def validate(config: Config) -> None:
    """Raise `ConfigError` on values that cannot work together."""
    if config.windows.size <= 0 or config.windows.hop <= 0:
        raise ConfigError("windows.size and windows.hop must be positive")
    if config.mel.hop_length <= 0 or config.audio.sample_rate <= 0:
        raise ConfigError("mel.hop_length and audio.sample_rate must be positive")
    if not set(config.split.cycle) <= SPLIT_ROLES:
        raise ConfigError(f"split.cycle only holds {sorted(SPLIT_ROLES)}")
    if not {"train", "val", "test"} <= set(config.split.cycle):
        raise ConfigError("split.cycle needs at least one train, val and test block")
    if config.models.cnn.dropout < 0 or config.models.cnn.dropout >= 1:
        raise ConfigError("models.cnn.dropout must be in [0, 1)")
    if config.pipeline.classifier == "detector":
        raise ConfigError(
            "pipeline.classifier is a segment classifier, not the detector"
        )


def load_config(path: Path | None = None) -> Config:
    """Read the configuration.

    Args:
        path: the YAML file. Default: the file named by the `BOWEL_CONFIG`
            environment variable, else `config.yaml` at the repository root.
            A missing default file means the defaults; a missing file that
            was asked for is an error.

    Returns:
        The validated configuration.
    """
    if path is None and ENV_VAR in os.environ:
        path = Path(os.environ[ENV_VAR])
        if not path.is_file():
            raise ConfigError(f"{ENV_VAR} points to a missing file: {path}")
    if path is None:
        path = CONFIG_PATH
        values = yaml.safe_load(path.read_text()) if path.is_file() else None
    else:
        values = yaml.safe_load(path.read_text())

    config: Config = build(Config, values)
    validate(config)

    return config


CONFIG = load_config()
