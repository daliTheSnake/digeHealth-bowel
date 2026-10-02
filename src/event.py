"""A labelled event of a bowel sound recording."""

from dataclasses import dataclass

KNOWN_LABELS = {"sb", "mb", "h", "n", "v"}

LABEL_MAPPING = {
    "sbs": "sb",
    "b": "sb",
}


@dataclass
class Event:
    """One labelled event, in seconds from the start of the recording."""

    start: float
    end: float
    label: str

    @property
    def duration(self) -> str:
        """The interval formatted for logs, e.g. `1.250-1.480`."""
        return f"{self.start:.3f}-{self.end:.3f}"

    def normalize_label(self) -> str:
        """Normalize a raw annotation label to the canonical label."""
        return LABEL_MAPPING.get(self.label, self.label)
