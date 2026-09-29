from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower()).rstrip(".,!?;:")


@dataclass(frozen=True)
class CommandDefinition:
    phrase: str
    action: str
    value: Optional[str] = None
    source: str = "user"
    description: str = ""
    modes: tuple[str, ...] = ()
    requires_confirmation: bool = False

    @property
    def normalized_phrase(self) -> str:
        return normalize_phrase(self.phrase)


class CommandRegistry:
    """Mutable command/replacement catalog used by core, plugins, and users."""

    def __init__(
        self,
        commands: Iterable[CommandDefinition] = (),
        replacements: Optional[dict[str, str]] = None,
    ) -> None:
        self._commands: dict[str, CommandDefinition] = {}
        self._replacements: dict[str, str] = {}
        self._replacement_sources: dict[str, str] = {}
        for command in commands:
            self.register_command(command)
        for phrase, replacement in (replacements or {}).items():
            self.register_replacement(phrase, replacement)

    def register_command(self, command: CommandDefinition, *, replace: bool = True) -> None:
        key = command.normalized_phrase
        if not key:
            raise ValueError("command phrase cannot be empty")
        if not replace and key in self._commands:
            raise KeyError(f"command already registered: {key}")
        self._commands[key] = command

    def register_replacement(
        self,
        phrase: str,
        replacement: str,
        *,
        source: str = "user",
        replace: bool = True,
    ) -> None:
        key = normalize_phrase(phrase)
        if not key:
            raise ValueError("replacement phrase cannot be empty")
        if not replace and key in self._replacements:
            raise KeyError(f"replacement already registered: {key}")
        self._replacements[key] = str(replacement)
        self._replacement_sources[key] = source

    def command_for(self, text: str) -> Optional[CommandDefinition]:
        return self._commands.get(normalize_phrase(text))

    def commands(self) -> tuple[CommandDefinition, ...]:
        return tuple(self._commands.values())

    def replacements(self) -> dict[str, str]:
        return dict(self._replacements)

    def replacement_source(self, phrase: str) -> Optional[str]:
        return self._replacement_sources.get(normalize_phrase(phrase))
