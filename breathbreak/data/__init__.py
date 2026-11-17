"""Dataset loaders. Every loader returns a list of `Recording` objects."""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    subject: str            # person ID (used for leave-one-subject-out)
    session: str            # e.g. "1", "morning", "baseline"
    kind: str               # "ecg" or "ppg"
    signal: np.ndarray
    fs: float
    label: int | None       # 1 = fatigued, 0 = alert, None = unknown/control
    dataset: str
    is_baseline_session: bool = False   # True -> its first 2 min define the personal baseline
    extra: dict = field(default_factory=dict)
