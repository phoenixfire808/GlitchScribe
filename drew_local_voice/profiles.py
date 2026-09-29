from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DictationProfile:
    id: str
    label: str
    mode: str = "general"
    description: str = ""
    executable_patterns: tuple[str, ...] = ()
    title_patterns: tuple[str, ...] = ()
    control_types: tuple[int, ...] = ()
    overrides: dict[str, Any] = field(default_factory=dict)

    def matches(
        self,
        *,
        executable: str = "",
        window_title: str = "",
        control_type: Optional[int] = None,
    ) -> bool:
        executable_name = executable.lower()
        title = window_title.lower()
        executable_match = any(
            fnmatch.fnmatch(executable_name, pattern.lower())
            for pattern in self.executable_patterns
        )
        title_match = any(
            fnmatch.fnmatch(title, pattern.lower()) for pattern in self.title_patterns
        )
        control_match = control_type is not None and control_type in self.control_types
        return executable_match or title_match or control_match


class ProfileRegistry:
    def __init__(self, profiles: Iterable[DictationProfile] = ()) -> None:
        self._profiles: dict[str, DictationProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: DictationProfile, *, replace: bool = True) -> None:
        if not profile.id:
            raise ValueError("profile id cannot be empty")
        if not replace and profile.id in self._profiles:
            raise KeyError(f"profile already registered: {profile.id}")
        self._profiles[profile.id] = profile

    def get(self, profile_id: str) -> DictationProfile:
        return self._profiles[profile_id]

    def all(self) -> tuple[DictationProfile, ...]:
        return tuple(self._profiles.values())

    def resolve(
        self,
        *,
        explicit_id: Optional[str] = None,
        executable: str = "",
        window_title: str = "",
        control_type: Optional[int] = None,
    ) -> DictationProfile:
        if explicit_id and explicit_id != "automatic" and explicit_id in self._profiles:
            return self._profiles[explicit_id]
        for profile in reversed(self.all()):
            if profile.id != "general" and profile.matches(
                executable=executable,
                window_title=window_title,
                control_type=control_type,
            ):
                return profile
        return self._profiles["general"]


def built_in_profiles() -> tuple[DictationProfile, ...]:
    return (
        DictationProfile("general", "General", description="Balanced prose dictation."),
        DictationProfile("chat", "Chat", mode="chat", description="Conversational text with short continuation memory.", executable_patterns=("discord.exe", "slack.exe", "teams.exe"), overrides={"typing.continuation_window_seconds": 45.0}),
        DictationProfile("document", "Document", mode="document", description="Long-form writing and patient endpointing.", executable_patterns=("winword.exe", "notepad.exe", "notepad++.exe"), overrides={"endpoint.silence_seconds": 2.8}),
        DictationProfile("terminal", "Terminal", mode="terminal", description="Literal shell symbols without an inferred final period.", executable_patterns=("windowsterminal.exe", "wt.exe", "cmd.exe", "powershell.exe", "pwsh.exe", "mintty.exe"), overrides={"typing.automatic_punctuation": True, "typing.suppress_inferred_trailing_period": True, "typing.trailing_space": False}),
        DictationProfile("code", "Code", mode="code", description="Code symbols, identifiers, and literal casing.", executable_patterns=("code.exe", "pycharm*.exe", "idea*.exe", "devenv.exe"), overrides={"typing.smart_continuation": False, "typing.automatic_punctuation": True, "typing.suppress_inferred_trailing_period": True, "typing.trailing_space": False}),
        DictationProfile("search", "Search", mode="search", description="Short queries without terminal punctuation.", overrides={"typing.automatic_punctuation": False, "endpoint.silence_seconds": 1.2}),
    )
