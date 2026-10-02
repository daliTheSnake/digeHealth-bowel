"""A bowel sound recording with its events and what was checked on them."""

from dataclasses import dataclass, field

from src.event import Event


@dataclass
class Recording:
    """One WAV file, its parsed annotation file and the checks run on it."""

    name: str
    duration: float
    sample_rate: int
    channels: int
    events: list[Event]
    parsing_errors: list[str] = field(default_factory=list)
    issues: dict[str, list[str]] = field(default_factory=dict)
    overlaps: list[str] = field(default_factory=list)
