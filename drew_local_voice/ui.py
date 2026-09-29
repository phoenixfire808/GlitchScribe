from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DictationStatus:
    state: str
    message: str = ""
    microphone_level: float = 0.0
    provisional_text: str = ""
    profile_id: str = "general"


class StatusSink(Protocol):
    def update(self, status: DictationStatus) -> None: ...

    def show(self) -> None: ...

    def hide(self) -> None: ...
