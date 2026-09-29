from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    start_seconds: float
    end_seconds: float
    probability: float | None = None


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    words: tuple[RecognizedWord, ...] = ()
    language: str = "en"
    inference_seconds: float = 0.0


class Recognizer(Protocol):
    def warm(self) -> None: ...

    def transcribe(self, samples: np.ndarray, *, final: bool = False) -> RecognitionResult: ...

    def close(self) -> None: ...
