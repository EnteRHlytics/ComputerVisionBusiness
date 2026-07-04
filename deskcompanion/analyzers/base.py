"""Analyzer protocol: anything with analyze(source, frame) -> list[Metric]."""
from typing import Protocol

import numpy as np

from ..store import Metric


class Analyzer(Protocol):
    def analyze(self, source: str, frame: np.ndarray) -> list[Metric]: ...
